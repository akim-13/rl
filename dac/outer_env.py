"""A's training run as a Gymnasium environment for the outer agent B.

One episode is one InnerRun, one step is one block: B picks an (lr_mult, eps_mult)
pair, A trains block_steps env steps with it, A's greedy policy is evaluated and
B gets that score scaled to [0, 1]. With epsilon_levels or lr_levels, that entry
is an absolute level instead of a multiplier (see ActionSet).
"""

import math
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dac.controllers import DEFAULT_EPS_MULTS, DEFAULT_LR_MULTS, ActionSet
from dac.inner import BlockResult, InnerConfig, InnerRun

N_FEATURES = 8

CHANGE_NOTHING = (1.0, 1.0)  # for multipliers; with epsilon levels it's (1.0, eps_init)
LARGEST_RANDOM_INNER_SEED = 2**31 - 1


def _log_position(value: float, lowest: float, highest: float) -> float:
    """Position of value on a log scale, from 0 at lowest to 1 at highest."""
    position = (math.log(value) - math.log(lowest)) / (math.log(highest) - math.log(lowest))
    return min(max(position, 0.0), 1.0)


def _zero_if_nan(value: float) -> float:
    return 0.0 if math.isnan(value) else value


def make_observation(
    result: BlockResult, previous: BlockResult | None, config: InnerConfig
) -> np.ndarray:
    """B's observation after a block (previous is None at block 0)."""
    score_change = 0.0 if previous is None else result.eval_score - previous.eval_score
    loss = _zero_if_nan(result.loss_mean)

    progress_through_run = result.block / config.n_blocks
    current_greedy_score = result.eval_score / config.score_max
    score_change_since_last_block = score_change / config.score_max
    training_episode_return = _zero_if_nan(result.train_return_mean) / config.score_max
    current_lr = _log_position(result.lr, config.lr_min, config.lr_max)
    current_epsilon = _log_position(result.epsilon, config.eps_min, config.eps_max)
    td_loss = math.log10(1.0 + loss)
    q_scale = _zero_if_nan(result.q_mean) * (1.0 - config.gamma)  # roughly the reward per step

    features = [
        progress_through_run,           # 0
        current_greedy_score,           # 1
        score_change_since_last_block,  # 2
        training_episode_return,        # 3
        current_lr,                     # 4
        current_epsilon,                # 5
        td_loss,                        # 6
        q_scale,                        # 7
    ]
    return np.array(features, dtype=np.float32)


class OuterEnv(gym.Env):
    """Each episode is one InnerRun; each step runs one block with the chosen multipliers."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: InnerConfig = InnerConfig(),
        lr_multipliers: Sequence[float] = DEFAULT_LR_MULTS,
        epsilon_multipliers: Sequence[float] = DEFAULT_EPS_MULTS,
        inner_seed_pool: Sequence[int] | None = None,
        epsilon_levels: Sequence[float] | None = None,
        lr_levels: Sequence[float] | None = None,
        period_levels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_set = ActionSet.build(lr_multipliers, epsilon_multipliers, epsilon_levels, lr_levels, period_levels)
        self.actions = self.action_set.pairs
        self.noop_action = noop_action_of(self.action_set, config)
        self.inner_seed_pool = _validated_seed_pool(inner_seed_pool)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_FEATURES,), dtype=np.float32)

        self.run: InnerRun | None = None
        self.inner_seed: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        inner_seed = self._choose_inner_seed(options)

        if self.run is not None:
            self.run.close()
        self.run = InnerRun(inner_seed, self.config)
        self.inner_seed = inner_seed

        untrained_result = self.run.history[0]
        observation = make_observation(untrained_result, None, self.config)
        return observation, {"result": untrained_result, "inner_seed": inner_seed}

    def _choose_inner_seed(self, options: dict | None) -> int:
        """options["inner_seed"] if given, else one from the pool, else random."""
        seed_was_given = options is not None and "inner_seed" in options
        if seed_was_given:
            return int(options["inner_seed"])
        if self.inner_seed_pool is not None:
            pool_position = int(self.np_random.integers(len(self.inner_seed_pool)))
            return self.inner_seed_pool[pool_position]
        return int(self.np_random.integers(LARGEST_RANDOM_INNER_SEED))

    def step(self, action):
        if self.run is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        choice = self.action_set.choice(int(action), self.run)

        previous = self.run.history[-1]
        result = self.run.run_block(*choice)

        observation = make_observation(result, previous, self.config)
        reward = result.eval_score / self.config.score_max
        terminated = self.run.done
        truncated = False
        info = {"result": result, "inner_seed": self.inner_seed}
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self.run is not None:
            self.run.close()
            self.run = None


def noop_action_of(action_set: ActionSet, config: InnerConfig) -> int:
    """Index of the action that keeps lr, epsilon and the target-copy period at their start values.

    Multiplier 1, or the start value itself for entries given as levels (period:
    config.target_update_every). ValueError if the grid has no such action.
    """
    keep_lr, keep_epsilon = CHANGE_NOTHING
    if action_set.lr_is_level:
        lr_levels = [lr_level for lr_level, _ in action_set.pairs]
        if config.lr_init not in lr_levels:
            raise ValueError(f"lr_init {config.lr_init} must be one of the lr levels")
        keep_lr = float(config.lr_init)
    if action_set.epsilon_is_level:
        epsilon_levels = [epsilon_level for _, epsilon_level in action_set.pairs]
        if config.eps_init not in epsilon_levels:
            raise ValueError(f"eps_init {config.eps_init} must be one of the epsilon levels")
        keep_epsilon = float(config.eps_init)
    change_nothing = (keep_lr, keep_epsilon)
    if change_nothing not in action_set.pairs:
        raise ValueError(f"the action grid must contain the 'change nothing' pair {change_nothing}")
    if action_set.periods is None:
        return action_set.pairs.index(change_nothing)
    keep_period = config.target_update_every
    if keep_period not in action_set.periods:
        raise ValueError(f"target_update_every {keep_period} must be one of the period levels")
    actions = zip(action_set.pairs, action_set.periods)
    return list(actions).index((change_nothing, keep_period))


def _validated_seed_pool(inner_seed_pool: Sequence[int] | None) -> list[int] | None:
    if inner_seed_pool is None:
        return None
    if len(inner_seed_pool) == 0:
        raise ValueError("inner_seed_pool must not be empty")
    return [int(seed) for seed in inner_seed_pool]
