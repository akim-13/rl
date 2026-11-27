import dataclasses
import math

import gymnasium as gym
import numpy as np
import pytest

from dac.controllers import NoOp, run_episode
from dac.dqn import DuelingQNetwork
from dac.inner import InnerConfig, InnerRun
from helpers import histories_match


def run_all(seed: int, config: InnerConfig, multipliers: list[tuple[float, float]]) -> list:
    run = InnerRun(seed, config)
    for lr_multiplier, epsilon_multiplier in multipliers:
        run.run_block(lr_multiplier, epsilon_multiplier)
    run.close()
    return run.history


def test_block_zero_is_the_untrained_network(tiny_config):
    run = InnerRun(0, tiny_config)
    first = run.history[0]
    assert len(run.history) == 1 and run.block == 0 and not run.done
    assert first.block == 0 and first.env_steps == 0
    assert first.lr == tiny_config.lr_init and first.epsilon == tiny_config.eps_init
    assert first.train_episodes == 0 and first.train_seconds == 0.0
    assert math.isnan(first.train_return_mean) and math.isnan(first.loss_mean)
    assert 1 <= first.eval_score <= tiny_config.score_max
    run.close()


def test_blocks_count_steps_and_end_the_run(tiny_config):
    run = InnerRun(0, tiny_config)
    for block in range(1, tiny_config.n_blocks + 1):
        result = run.run_block()
        assert result.block == block == run.block
        assert result.env_steps == block * tiny_config.block_steps
        assert np.isfinite(result.loss_mean)  # learning_starts < block_steps, so updates happened
    assert run.done and len(run.history) == tiny_config.n_blocks + 1
    with pytest.raises(RuntimeError):
        run.run_block()
    run.close()


def test_same_seed_same_multipliers_is_deterministic(tiny_config):
    multipliers = [(0.5, 2.0), (2.0, 0.5), (1.0, 1.0)]
    first = run_all(7, tiny_config, multipliers)
    second = run_all(7, tiny_config, multipliers)
    assert histories_match(first, second)


def test_different_seeds_differ(tiny_config):
    seed_one = run_all(1, tiny_config, [(1.0, 1.0)] * 3)
    seed_two = run_all(2, tiny_config, [(1.0, 1.0)] * 3)
    assert not histories_match(seed_one, seed_two)


def test_multipliers_set_lr_and_epsilon(tiny_config):
    run = InnerRun(0, tiny_config)
    result = run.run_block(lr_multiplier=0.5, epsilon_multiplier=2.0)
    assert run.lr == tiny_config.lr_init * 0.5
    assert run.agent.lr == tiny_config.lr_init * 0.5
    assert result.lr == run.lr
    assert run.epsilon == tiny_config.eps_init * 2.0 == result.epsilon
    run.close()


def test_multipliers_are_clamped(tiny_config):
    run = InnerRun(0, tiny_config)
    run.run_block(lr_multiplier=1e6, epsilon_multiplier=1e-6)
    assert run.lr == tiny_config.lr_max and run.agent.lr == tiny_config.lr_max
    assert run.epsilon == tiny_config.eps_min
    run.run_block(lr_multiplier=1e-9, epsilon_multiplier=1e9)
    assert run.lr == tiny_config.lr_min and run.epsilon == tiny_config.eps_max
    run.close()


@pytest.mark.parametrize(
    "lr_multiplier, epsilon_multiplier", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, float("nan"))]
)
def test_non_positive_multiplier_raises(tiny_config, lr_multiplier, epsilon_multiplier):
    run = InnerRun(0, tiny_config)
    with pytest.raises(ValueError):
        run.run_block(lr_multiplier, epsilon_multiplier)
    assert run.block == 0  # nothing happened
    run.close()


def test_extra_evaluations_do_not_change_training(tiny_config):
    plain = run_all(3, tiny_config, [(1.0, 1.0)] * 3)

    run = InnerRun(3, tiny_config)
    for _ in range(3):
        run.evaluate()
        run.run_block()
        run.evaluate()
        run.evaluate()
    run.close()
    assert histories_match(plain, run.history)


def test_evaluation_is_repeatable(tiny_config):
    run = InnerRun(0, tiny_config)
    run.run_block()
    assert run.evaluate() == run.evaluate() == (run.history[-1].eval_score, run.history[-1].eval_std)
    run.close()


def test_time_limit_truncation_is_stored_as_not_terminated(tiny_config):
    # Constant pushing needs at least 8 steps to topple the pole, so with a
    # 5-step time limit every training episode ends by truncation, never by
    # termination. (Uses the train_env test seam of InnerRun.)
    short_environment = gym.make("CartPole-v1", max_episode_steps=5)
    run = InnerRun(0, tiny_config, train_env=short_environment)
    result = run.run_block()
    buffer = run.agent.buffer
    assert result.train_episodes == tiny_config.block_steps // 5
    assert result.train_return_mean == 5.0
    assert len(buffer) == tiny_config.block_steps
    assert np.all(buffer.terminated[: len(buffer)] == 0.0)
    run.close()


def test_real_terminations_are_stored_as_terminated(tiny_config):
    run = InnerRun(0, tiny_config)
    result = run.run_block()
    buffer = run.agent.buffer
    # Early CartPole episodes all end by the pole falling: one terminal flag each.
    assert result.train_episodes > 0
    assert int(buffer.terminated[: len(buffer)].sum()) == result.train_episodes
    run.close()


def test_config_rejects_initial_values_outside_their_range():
    with pytest.raises(ValueError):
        InnerConfig(lr_init=1.0)
    with pytest.raises(ValueError):
        InnerConfig(eps_init=0.001)
    assert dataclasses.replace(InnerConfig(), lr_init=1e-2).lr_init == 1e-2


def test_dueling_inner_agent_runs_and_differs_from_the_plain_one(tiny_config):
    dueling_config = dataclasses.replace(tiny_config, dueling=True)
    run = InnerRun(0, dueling_config)
    assert isinstance(run.agent.online_net, DuelingQNetwork)
    run.close()
    dueling_history = run_episode(0, NoOp(), dueling_config)
    plain_history = run_episode(0, NoOp(), tiny_config)
    assert len(dueling_history) == len(plain_history) == tiny_config.n_blocks + 1
    assert [result.loss_mean for result in dueling_history[1:]] != [result.loss_mean for result in plain_history[1:]]


def test_run_block_can_set_the_target_copy_period(tiny_config):
    run = InnerRun(0, tiny_config)
    try:
        run.run_block()
        assert run.agent.target_update_every == tiny_config.target_update_every
        run.run_block(1.0, 1.0, 400)
        assert run.agent.target_update_every == 400
        run.run_block(1.0, 1.0)  # None keeps the current period
        assert run.agent.target_update_every == 400
    finally:
        run.close()


def test_run_block_refuses_a_bad_target_copy_period(tiny_config):
    run = InnerRun(0, tiny_config)
    try:
        for bad in (0, -5, 2.5):
            with pytest.raises(ValueError, match="target_update_every"):
                run.run_block(1.0, 1.0, bad)
    finally:
        run.close()
