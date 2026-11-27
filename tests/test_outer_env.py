import math

import numpy as np
import pytest

from dac.controllers import NoOp, run_episode
from dac.inner import BlockResult
from dac.outer_env import OuterEnv, make_observation

from helpers import histories_match


def test_reset_returns_valid_observation_and_info(tiny_config):
    outer_env = OuterEnv(tiny_config)
    observation, info = outer_env.reset(seed=0)
    assert outer_env.observation_space.contains(observation)
    assert observation.dtype == np.float32 and observation.shape == (8,)
    assert isinstance(info, dict)
    assert isinstance(info["result"], BlockResult) and info["result"].block == 0
    assert isinstance(info["inner_seed"], int)
    outer_env.close()


def test_step_reward_and_episode_length(tiny_config):
    outer_env = OuterEnv(tiny_config)
    outer_env.reset(seed=0)
    for block in range(1, tiny_config.n_blocks + 1):
        step_outputs = outer_env.step(outer_env.noop_action)
        assert len(step_outputs) == 5
        observation, reward, terminated, truncated, info = step_outputs
        assert outer_env.observation_space.contains(observation)
        assert reward == info["result"].eval_score / tiny_config.score_max
        assert terminated == (block == tiny_config.n_blocks)
        assert truncated is False
        assert info["inner_seed"] == outer_env.inner_seed
    outer_env.close()


def test_noop_action_and_action_grid(tiny_config):
    outer_env = OuterEnv(tiny_config)
    assert outer_env.action_space.n == 9 == len(outer_env.actions)
    assert outer_env.actions[outer_env.noop_action] == (1.0, 1.0)
    with pytest.raises(ValueError):
        OuterEnv(tiny_config, lr_multipliers=(0.5, 2.0))
    outer_env.close()


def test_invalid_action_raises(tiny_config):
    outer_env = OuterEnv(tiny_config)
    outer_env.reset(seed=0)
    with pytest.raises(ValueError):
        outer_env.step(-1)
    with pytest.raises(ValueError):
        outer_env.step(outer_env.action_space.n)
    outer_env.close()


def test_same_inner_seed_and_actions_give_same_rewards(tiny_config):
    outer_env = OuterEnv(tiny_config)
    actions = [0, 8, 4]
    all_rewards = []
    for _ in range(2):
        outer_env.reset(options={"inner_seed": 5})
        assert outer_env.inner_seed == 5
        all_rewards.append([outer_env.step(action)[1] for action in actions])
    assert all_rewards[0] == all_rewards[1]
    outer_env.close()


def test_seed_pool_is_respected(tiny_config):
    pool = [11, 22, 33]
    outer_env = OuterEnv(tiny_config, inner_seed_pool=pool)
    seen = set()
    outer_env.reset(seed=0)
    seen.add(outer_env.inner_seed)
    for _ in range(9):
        _, info = outer_env.reset()
        seen.add(info["inner_seed"])
    assert seen <= set(pool) and len(seen) > 1
    # An explicit inner_seed still wins over the pool.
    outer_env.reset(options={"inner_seed": 99})
    assert outer_env.inner_seed == 99
    outer_env.close()


def test_reset_seed_makes_inner_seed_reproducible(tiny_config):
    outer_env = OuterEnv(tiny_config)
    outer_env.reset(seed=123)
    first = outer_env.inner_seed
    outer_env.reset(seed=123)
    assert outer_env.inner_seed == first
    outer_env.close()


def test_make_observation_features(tiny_config):
    config = tiny_config
    nan = math.nan
    previous = BlockResult(1, 200, config.lr_init, config.eps_init, 100.0, 0.0, nan, 0, nan, nan, nan, nan, 0.0, 0.0)
    result = BlockResult(2, 400, config.lr_max, config.eps_min, 150.0, 1.0, 50.0, 3, 9.0, 0.5, 20.0, 1.0, 0.0, 0.0)
    observation = make_observation(result, previous, config)
    expected = [
        2 / config.n_blocks,
        150.0 / config.score_max,
        50.0 / config.score_max,
        50.0 / config.score_max,
        1.0,  # lr at lr_max
        0.0,  # epsilon at eps_min
        1.0,  # log10(1 + 9)
        20.0 * (1 - config.gamma),
    ]
    assert observation == pytest.approx(np.array(expected, dtype=np.float32))
    # At block 0 the change feature is 0 and nan statistics become 0.
    first = make_observation(previous, None, config)
    assert first[2] == 0.0 and first[3] == 0.0 and first[6] == 0.0 and first[7] == 0.0
    assert not np.any(np.isnan(first))


EPSILON_LEVELS = (0.03, 0.1, 0.3)


def test_epsilon_levels_noop_run_equals_noop_controller(tiny_config):
    """With eps_init among the levels, always taking the noop action changes nothing, bit for bit."""
    outer_env = OuterEnv(tiny_config, epsilon_levels=EPSILON_LEVELS)
    assert outer_env.actions[outer_env.noop_action] == (1.0, tiny_config.eps_init)
    _, info = outer_env.reset(options={"inner_seed": 4})
    history = [info["result"]]
    terminated = False
    while not terminated:
        _, _, terminated, _, info = outer_env.step(outer_env.noop_action)
        history.append(info["result"])
    outer_env.close()
    assert histories_match(history, run_episode(4, NoOp(), tiny_config))


def test_epsilon_level_action_sets_epsilon_to_that_level(tiny_config):
    outer_env = OuterEnv(tiny_config, epsilon_levels=EPSILON_LEVELS)
    assert outer_env.action_space.n == 9
    outer_env.reset(options={"inner_seed": 0})
    chosen_pairs = [(1.0, 0.3), (2.0, 0.03), (0.5, 0.03)]
    expected_lr = tiny_config.lr_init
    for lr_multiplier, epsilon_level in chosen_pairs:
        action = outer_env.actions.index((lr_multiplier, epsilon_level))
        result = outer_env.step(action)[4]["result"]
        expected_lr *= lr_multiplier
        assert result.epsilon == pytest.approx(epsilon_level, rel=1e-12)
        assert result.lr == pytest.approx(expected_lr, rel=1e-12)
    outer_env.close()


def test_epsilon_levels_must_include_eps_init(tiny_config):
    with pytest.raises(ValueError, match="eps_init"):
        OuterEnv(tiny_config, epsilon_levels=(0.03, 0.3))


LR_LEVELS = (3e-5, 1e-4, 1e-3)


def test_lr_and_epsilon_levels_noop_run_equals_noop_controller(tiny_config):
    """With lr_init and eps_init among the levels, always taking the noop action changes nothing."""
    outer_env = OuterEnv(tiny_config, epsilon_levels=EPSILON_LEVELS, lr_levels=LR_LEVELS)
    assert outer_env.actions[outer_env.noop_action] == (tiny_config.lr_init, tiny_config.eps_init)
    _, info = outer_env.reset(options={"inner_seed": 4})
    history = [info["result"]]
    terminated = False
    while not terminated:
        _, _, terminated, _, info = outer_env.step(outer_env.noop_action)
        history.append(info["result"])
    outer_env.close()
    assert histories_match(history, run_episode(4, NoOp(), tiny_config))


def test_lr_level_action_sets_lr_to_that_level_in_one_block(tiny_config):
    outer_env = OuterEnv(tiny_config, epsilon_levels=EPSILON_LEVELS, lr_levels=LR_LEVELS)
    outer_env.reset(options={"inner_seed": 0})
    for lr_level, epsilon_level in [(1e-4, 0.1), (3e-5, 0.03), (1e-3, 0.3)]:
        action = outer_env.actions.index((lr_level, epsilon_level))
        result = outer_env.step(action)[4]["result"]
        assert result.lr == pytest.approx(lr_level, rel=1e-12)
        assert result.epsilon == pytest.approx(epsilon_level, rel=1e-12)
    outer_env.close()


def test_lr_levels_must_include_lr_init(tiny_config):
    with pytest.raises(ValueError, match="lr_init"):
        OuterEnv(tiny_config, lr_levels=(3e-5, 1e-4))


PERIOD_LEVELS = (100, 400)


def test_period_levels_noop_run_equals_noop_controller(tiny_config):
    """With target_update_every among the period levels, always taking the noop action changes nothing."""
    outer_env = OuterEnv(tiny_config, epsilon_levels=EPSILON_LEVELS, lr_levels=LR_LEVELS, period_levels=PERIOD_LEVELS)
    assert outer_env.action_space.n == 18
    noop = outer_env.noop_action
    assert outer_env.actions[noop] == (tiny_config.lr_init, tiny_config.eps_init)
    assert outer_env.action_set.periods[noop] == tiny_config.target_update_every
    _, info = outer_env.reset(options={"inner_seed": 4})
    history = [info["result"]]
    terminated = False
    while not terminated:
        _, _, terminated, _, info = outer_env.step(noop)
        history.append(info["result"])
    outer_env.close()
    assert histories_match(history, run_episode(4, NoOp(), tiny_config))


def test_period_action_sets_the_target_copy_period(tiny_config):
    outer_env = OuterEnv(tiny_config, epsilon_levels=(0.1,), lr_levels=LR_LEVELS, period_levels=PERIOD_LEVELS)
    outer_env.reset(options={"inner_seed": 0})
    for period in (400, 100, 400):
        action_index_and_period = enumerate(zip(outer_env.actions, outer_env.action_set.periods))
        action = next(
            index
            for index, (lr_and_epsilon, action_period) in action_index_and_period
            if lr_and_epsilon == (tiny_config.lr_init, 0.1) and action_period == period
        )
        outer_env.step(action)
        assert outer_env.run.agent.target_update_every == period
    outer_env.close()


def test_period_levels_must_include_target_update_every(tiny_config):
    with pytest.raises(ValueError, match="target_update_every"):
        OuterEnv(tiny_config, period_levels=(200, 400))
