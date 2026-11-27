import csv
import dataclasses
import json

import numpy as np
import pytest
import torch

from dac.controllers import make_controller, run_episode
from dac.inner import InnerConfig
from dac.outer_dqn import (
    LearnedController,
    OuterConfig,
    Transition,
    collapse_steps,
    epsilon_at,
    n_step_transitions_ready,
    train,
)
from dac.outer_env import OuterEnv

from conftest import TINY_CONFIG
from golden_cases import GOLDEN_DIR


def test_epsilon_schedule():
    outer_config = OuterConfig(eps_start=1.0, eps_end=0.1, eps_decay_fraction=0.5)
    assert epsilon_at(0, 100, outer_config) == 1.0
    assert epsilon_at(25, 100, outer_config) == pytest.approx(0.55)
    assert epsilon_at(50, 100, outer_config) == pytest.approx(0.1)
    assert epsilon_at(99, 100, outer_config) == pytest.approx(0.1)


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("outer")
    outer_config = dataclasses.replace(
        OuterConfig(), n_envs=2, rounds=2, learning_starts=2 * TINY_CONFIG.n_blocks, updates_per_step=2
    )
    train(outer_config, TINY_CONFIG, seed=0, out_dir=out_dir)
    return out_dir


def test_training_writes_model_and_log(trained_model, tiny_config):
    assert (trained_model / "model.pt").exists()
    with open(trained_model / "train_log.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * 2  # rounds * envs
    for row in rows:
        assert int(row["inner_seed"]) >= 1_000  # never a test seed
        action_counts = [int(row[f"a{action}"]) for action in range(9)]
        assert sum(action_counts) == tiny_config.n_blocks
        assert 0.0 <= float(row["auc"]) <= 1.0


def test_learned_controller_matches_outer_env(trained_model, tiny_config):
    """Driving OuterEnv with the network's greedy actions gives the same run as the controller."""
    model = str(trained_model / "model.pt")
    controller = LearnedController(model, tiny_config)
    history = run_episode(3, controller, tiny_config)

    outer_env = OuterEnv(tiny_config)
    observation, _ = outer_env.reset(options={"inner_seed": 3})
    scores = []
    terminated = False
    while not terminated:
        action = int(controller.q_values(observation).argmax())
        observation, _, terminated, _, info = outer_env.step(action)
        scores.append(info["result"].eval_score)
    outer_env.close()
    assert scores == [result.eval_score for result in history[1:]]


def test_ensemble_of_one_model_twice_acts_like_the_model(trained_model, tiny_config):
    model = str(trained_model / "model.pt")
    single = run_episode(3, LearnedController(model, tiny_config), tiny_config)
    doubled = run_episode(3, LearnedController((model, model), tiny_config), tiny_config)
    assert [result.eval_score for result in single] == [result.eval_score for result in doubled]


def test_ensemble_averages_q_values_over_models(trained_model, tiny_config):
    paths = [str(trained_model / "checkpoints" / "round_00.pt"), str(trained_model / "model.pt")]
    ensemble = LearnedController(paths, tiny_config)
    observation = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    separate = [LearnedController(path, tiny_config).q_values(observation) for path in paths]
    assert torch.allclose(ensemble.q_values(observation), (separate[0] + separate[1]) / 2)
    assert not torch.equal(separate[0], separate[1])


def test_make_controller_ddqn_checks_config(trained_model, tiny_config):
    model = str(trained_model / "model.pt")
    assert isinstance(make_controller("ddqn", tiny_config, 0, model=model), LearnedController)
    with pytest.raises(ValueError):
        make_controller("ddqn", dataclasses.replace(tiny_config, n_blocks=4), 0, model=model)


def fake_episode(length: int) -> list[Transition]:
    """Step t goes from observation [t] to [t+1] with reward 2**t; only the last step terminates."""
    return [
        Transition(np.array([float(step)]), step, 2.0**step, np.array([float(step + 1)]), step == length - 1)
        for step in range(length)
    ]


def emitted_after_each_step(episode: list[Transition], n_step: int, gamma: float) -> list[list[Transition]]:
    return [n_step_transitions_ready(episode[:taken], n_step, gamma) for taken in range(1, len(episode) + 1)]


def start_of(transition: Transition) -> int:
    return int(transition.observation[0])


def end_of(transition: Transition) -> int:
    return int(transition.next_observation[0])


def test_one_step_transitions_are_the_steps_themselves():
    episode = fake_episode(4)
    emitted = emitted_after_each_step(episode, n_step=1, gamma=0.9)
    assert [len(batch) for batch in emitted] == [1, 1, 1, 1]
    for step, (transition,) in zip(episode, emitted):
        assert transition.observation is step.observation and transition.next_observation is step.next_observation
        transition_fields = (transition.action, transition.reward, transition.terminated)
        assert transition_fields == (step.action, step.reward, step.terminated)


def test_three_step_transitions_are_emitted_progressively_with_a_short_tail():
    gamma = 0.5
    episode = fake_episode(5)
    emitted = emitted_after_each_step(episode, n_step=3, gamma=gamma)
    assert [[start_of(transition) for transition in batch] for batch in emitted] == [[], [], [0], [1], [2, 3, 4]]
    flat = [transition for batch in emitted for transition in batch]
    assert [end_of(transition) for transition in flat] == [3, 4, 5, 5, 5]
    assert [transition.terminated for transition in flat] == [False, False, True, True, True]
    assert [transition.action for transition in flat] == [0, 1, 2, 3, 4]
    # Rewards 2**t discounted by 0.5 per step: each later reward adds exactly 1 * 2**start.
    assert [transition.reward for transition in flat] == [3.0, 6.0, 12.0, 16.0, 16.0]


def test_n_step_longer_than_the_episode_waits_for_the_end():
    episode = fake_episode(4)
    emitted = emitted_after_each_step(episode, n_step=10, gamma=1.0)
    assert [len(batch) for batch in emitted] == [0, 0, 0, 4]
    tail = emitted[-1]
    assert [start_of(transition) for transition in tail] == [0, 1, 2, 3]
    assert all(end_of(transition) == 4 and transition.terminated for transition in tail)
    assert [transition.reward for transition in tail] == [15.0, 14.0, 12.0, 8.0]


def test_collapse_steps_discounts_each_reward_by_its_delay():
    steps = [Transition(np.zeros(1), 0, reward, np.zeros(1), False) for reward in (1.0, 2.0, 3.0)]
    assert collapse_steps(steps, gamma=0.9).reward == pytest.approx(1.0 + 0.9 * 2.0 + 0.81 * 3.0)


LEVELS_CONFIG = InnerConfig(block_steps=200, n_blocks=5, eval_episodes=2, learning_starts=50)
EPSILON_LEVELS = (0.03, 0.1, 0.3)
LR_LEVELS = (3e-5, 1e-4, 1e-3)


@pytest.fixture(scope="module")
def trained_with_options(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("outer_options")
    outer_config = dataclasses.replace(
        OuterConfig(), n_envs=2, rounds=2, learning_starts=2 * LEVELS_CONFIG.n_blocks, updates_per_step=2,
        n_step=3, epsilon_levels=EPSILON_LEVELS, lr_levels=LR_LEVELS,
    )
    train(outer_config, LEVELS_CONFIG, seed=0, out_dir=out_dir)
    return out_dir


def test_training_with_options_writes_checkpoints_and_buffer(trained_with_options):
    out_dir = trained_with_options
    final = torch.load(out_dir / "model.pt", weights_only=True)
    assert final["outer_config"]["n_step"] == 3
    assert tuple(final["outer_config"]["epsilon_levels"]) == EPSILON_LEVELS
    assert tuple(final["outer_config"]["lr_levels"]) == LR_LEVELS
    checkpoints = sorted(path.name for path in (out_dir / "checkpoints").iterdir())
    assert checkpoints == ["round_00.pt", "round_01.pt"]
    last_round = torch.load(out_dir / "checkpoints" / "round_01.pt", weights_only=True)
    for parameter_name, tensor in final["state_dict"].items():
        assert torch.equal(last_round["state_dict"][parameter_name], tensor)

    buffer = np.load(out_dir / "replay_buffer.npz")
    episodes = 2 * 2
    assert len(buffer["actions"]) == episodes * LEVELS_CONFIG.n_blocks  # one transition per step taken
    # Feature 0 is the progress through the run, block / n_blocks.
    progress_spanned = buffer["next_observations"][:, 0] - buffer["observations"][:, 0]
    blocks_spanned = np.round(progress_spanned * LEVELS_CONFIG.n_blocks)
    terminated = buffer["terminated"] == 1.0
    assert terminated.sum() == episodes * 3  # the last n_step transitions of each episode
    assert np.all(blocks_spanned[~terminated] == 3)
    assert np.all(buffer["next_observations"][terminated, 0] == 1.0)  # cut short at the end of the run

    meta = json.loads((out_dir / "train_meta.json").read_text())
    assert meta["transitions"] == episodes * LEVELS_CONFIG.n_blocks
    assert [lr_and_level[1] for lr_and_level in meta["actions"][:3]] == list(EPSILON_LEVELS)


def test_learned_controller_with_epsilon_levels_matches_outer_env(trained_with_options):
    controller = LearnedController(str(trained_with_options / "model.pt"), LEVELS_CONFIG)
    assert controller.action_set.epsilon_is_level and controller.action_set.lr_is_level
    history = run_episode(3, controller, LEVELS_CONFIG)
    for result in history[1:]:
        assert any(result.epsilon == pytest.approx(level, rel=1e-12) for level in EPSILON_LEVELS)
        assert any(result.lr == pytest.approx(level, rel=1e-12) for level in LR_LEVELS)

    outer_env = OuterEnv(LEVELS_CONFIG, epsilon_levels=EPSILON_LEVELS, lr_levels=LR_LEVELS)
    observation, _ = outer_env.reset(options={"inner_seed": 3})
    results = []
    terminated = False
    while not terminated:
        action = int(controller.q_values(observation).argmax())
        observation, _, terminated, _, info = outer_env.step(action)
        results.append(info["result"])
    outer_env.close()
    assert [(result.eval_score, result.epsilon, result.lr) for result in results] == [
        (result.eval_score, result.epsilon, result.lr) for result in history[1:]
    ]


def test_model_without_level_keys_loads_with_multiplier_actions():
    golden_inner_config = InnerConfig(block_steps=200, n_blocks=5, eval_episodes=2, learning_starts=50)
    checkpoint = torch.load(GOLDEN_DIR / "outer_model.pt", weights_only=True)
    assert "epsilon_levels" not in checkpoint["outer_config"]
    controller = LearnedController(str(GOLDEN_DIR / "outer_model.pt"), golden_inner_config)
    assert not controller.action_set.epsilon_is_level and not controller.action_set.lr_is_level


def test_training_refuses_levels_without_eps_init(tmp_path, tiny_config):
    with pytest.raises(ValueError, match="eps_init"):
        train(OuterConfig(epsilon_levels=(0.03, 0.3)), tiny_config, seed=0, out_dir=tmp_path)


def test_ensemble_refuses_models_with_different_actions(trained_with_options):
    # The golden model was trained with the same InnerConfig but multiplier actions.
    paths = (str(trained_with_options / "model.pt"), str(GOLDEN_DIR / "outer_model.pt"))
    with pytest.raises(ValueError, match="different actions"):
        LearnedController(paths, LEVELS_CONFIG)


@pytest.fixture(scope="module")
def trained_with_periods(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("outer_periods")
    outer_config = dataclasses.replace(
        OuterConfig(), n_envs=2, rounds=2, learning_starts=2 * LEVELS_CONFIG.n_blocks, updates_per_step=2,
        n_step=3, epsilon_levels=(0.1,), lr_levels=LR_LEVELS, period_levels=(100, 400),
    )
    train(outer_config, LEVELS_CONFIG, seed=0, out_dir=out_dir)
    return out_dir


def test_learned_controller_with_periods_matches_outer_env(trained_with_periods):
    model = torch.load(trained_with_periods / "model.pt", weights_only=True)
    assert tuple(model["outer_config"]["period_levels"]) == (100, 400)
    meta = json.loads((trained_with_periods / "train_meta.json").read_text())
    assert [action[2] for action in meta["actions"]] == [100, 400] * 3

    controller = LearnedController(str(trained_with_periods / "model.pt"), LEVELS_CONFIG)
    assert controller.action_set.periods == [100, 400] * 3
    history = run_episode(3, controller, LEVELS_CONFIG)

    outer_env = OuterEnv(LEVELS_CONFIG, epsilon_levels=(0.1,), lr_levels=LR_LEVELS, period_levels=(100, 400))
    observation, _ = outer_env.reset(options={"inner_seed": 3})
    results, periods = [], []
    terminated = False
    while not terminated:
        action = int(controller.q_values(observation).argmax())
        observation, _, terminated, _, info = outer_env.step(action)
        results.append(info["result"])
        periods.append(outer_env.run.agent.target_update_every)
    outer_env.close()
    assert [(result.eval_score, result.lr) for result in results] == [
        (result.eval_score, result.lr) for result in history[1:]
    ]
    assert set(periods) <= {100, 400}
