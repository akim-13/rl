"""Fixed (non-learning) controllers, used as baselines for the outer agent B.

A controller is any callable `(run: InnerRun) -> (lr_mult, eps_mult)` giving the
multipliers for the next block. It may read run.history[-1], run.lr,
run.epsilon, run.block and run.config.

Controller keyword arguments (lr_mults, eps_start, threshold, ...) are also
their names on the command line (`--controller-arg key=value`) and in the saved
JSON files, which is why they're kept short.
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from dac.inner import BlockResult, InnerConfig, InnerRun

# Multiplier grid shared by RandomDiscrete and OuterEnv (B's action set)
DEFAULT_LR_MULTS = (0.5, 1.0, 2.0)
DEFAULT_EPS_MULTS = (0.5, 1.0, 2.0)

CONTROLLER_NAMES = ("noop", "random", "linear_eps", "lr_gate", "ddqn", "tabular")

# (lr_mult, eps_mult), optionally with a third entry setting A's target_update_every
# for the next block; InnerRun.run_block takes either.
Choice = tuple[float, float] | tuple[float, float, int]
Controller = Callable[[InnerRun], Choice]

LEAVE_UNCHANGED = 1.0


def action_grid(
    lr_multipliers: Sequence[float], epsilon_multipliers: Sequence[float]
) -> list[tuple[float, float]]:
    """Every (lr_mult, eps_mult) pair; the list index is B's action number. Epsilon varies fastest."""
    every_pair = itertools.product(lr_multipliers, epsilon_multipliers)
    return [(float(lr_multiplier), float(epsilon_multiplier)) for lr_multiplier, epsilon_multiplier in every_pair]


@dataclass(frozen=True)
class ActionSet:
    """A grid of discrete actions and how each one turns into run_block arguments.

    Each action is a pair (lr entry, epsilon entry). An entry is a multiplier
    unless the set was built with levels for it; then it's the absolute value
    for the next block, so repeated choices can't compound it to an extreme
    and any level is reachable in one block.
    With period levels, each pair is repeated once per period and action i
    also sets A's target-copy period to periods[i] for the next block.
    Shared by OuterEnv (B's actions), RandomDiscrete and the trained B.
    """

    pairs: list[tuple[float, float]]
    epsilon_is_level: bool
    lr_is_level: bool = False
    periods: list[int] | None = None  # per action; None means the period is never changed

    @classmethod
    def build(
        cls,
        lr_multipliers: Sequence[float],
        epsilon_multipliers: Sequence[float],
        epsilon_levels: Sequence[float] | None = None,
        lr_levels: Sequence[float] | None = None,
        period_levels: Sequence[int] | None = None,
    ) -> "ActionSet":
        """lr entries x epsilon entries (levels where given, else multipliers),
        times the periods if given. The period varies fastest."""
        epsilon_is_level = epsilon_levels is not None
        lr_is_level = lr_levels is not None
        epsilon_entries = epsilon_levels if epsilon_is_level else epsilon_multipliers
        lr_entries = lr_levels if lr_is_level else lr_multipliers
        grid = action_grid(lr_entries, epsilon_entries)
        if period_levels is None:
            return cls(grid, epsilon_is_level, lr_is_level)
        pairs = [pair for pair in grid for _ in period_levels]
        periods = [int(period) for _ in grid for period in period_levels]
        return cls(pairs, epsilon_is_level, lr_is_level, periods)

    def multipliers(self, action: int, run: InnerRun) -> tuple[float, float]:
        """(lr_mult, eps_mult) for this action; run.lr and run.epsilon are only read for levels."""
        lr_entry, epsilon_entry = self.pairs[action]
        lr_multiplier = lr_entry / run.lr if self.lr_is_level else lr_entry
        epsilon_multiplier = epsilon_entry / run.epsilon if self.epsilon_is_level else epsilon_entry
        return lr_multiplier, epsilon_multiplier

    def choice(self, action: int, run: InnerRun) -> Choice:
        """The multipliers for this action, plus its period if the set has periods."""
        multipliers = self.multipliers(action, run)
        if self.periods is None:
            return multipliers
        return (*multipliers, self.periods[action])


class NoOp:
    """Never changes anything, so lr and epsilon stay at their initial values (the manual baseline)."""

    def __call__(self, run: InnerRun) -> tuple[float, float]:
        return LEAVE_UNCHANGED, LEAVE_UNCHANGED


class RandomDiscrete:
    """A uniformly random action from the grid each block; stands in for an untrained B.

    With epsilon_levels (e.g. 0.03,0.1,0.3) or lr_levels the entries are
    absolute levels instead of multipliers, as in OuterEnv.
    """

    def __init__(
        self,
        lr_mults: Sequence[float] = DEFAULT_LR_MULTS,
        eps_mults: Sequence[float] = DEFAULT_EPS_MULTS,
        seed: int = 0,
        epsilon_levels: Sequence[float] | None = None,
        lr_levels: Sequence[float] | None = None,
        period_levels: Sequence[int] | None = None,
    ) -> None:
        self.action_set = ActionSet.build(lr_mults, eps_mults, epsilon_levels, lr_levels, period_levels)
        self.rng = np.random.default_rng(seed)

    def __call__(self, run: InnerRun) -> Choice:
        chosen_action = int(self.rng.integers(len(self.action_set.pairs)))  # exactly one draw per block
        return self.action_set.choice(chosen_action, run)


class LinearEpsDecay:
    """Epsilon falls linearly from eps_start (block 1) to eps_end (block decay_blocks), then stays.

    lr is left alone. This is the usual hand-made DQN exploration schedule.
    """

    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.05, decay_blocks: int = 10) -> None:
        if decay_blocks < 1:
            raise ValueError("decay_blocks must be at least 1")
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_blocks = decay_blocks

    def target_epsilon(self, block: int) -> float:
        """Epsilon wanted during `block` (1-based)."""
        if self.decay_blocks == 1:
            return self.eps_end
        fraction_of_decay_done = min((block - 1) / (self.decay_blocks - 1), 1.0)
        return self.eps_start + fraction_of_decay_done * (self.eps_end - self.eps_start)

    def __call__(self, run: InnerRun) -> tuple[float, float]:
        next_block = run.block + 1
        wanted_epsilon = self.target_epsilon(next_block)
        return LEAVE_UNCHANGED, wanted_epsilon / run.epsilon


class ScoreGatedLR:
    """Cut lr to lr_init * factor once the policy scores well, to protect it.

    The lr is lr_init * factor when the last evaluation scored >= threshold,
    otherwise lr_init. With sticky=True it stays reduced once reduced.
    Epsilon is left alone. With high_period, A's target-copy period is
    high_period while the lr is reduced and config.target_update_every
    otherwise, since a longer period steadies the targets once the policy is good.
    """

    def __init__(
        self, threshold: float = 400.0, factor: float = 0.1, sticky: bool = False, high_period: int | None = None
    ) -> None:
        self.threshold = threshold
        self.factor = factor
        self.sticky = sticky
        self.high_period = high_period
        self.lr_is_reduced = False

    def __call__(self, run: InnerRun) -> Choice:
        if run.block == 0:
            self.lr_is_reduced = False  # new run, forget the previous one
        score_is_high = run.history[-1].eval_score >= self.threshold
        if self.sticky:
            self.lr_is_reduced = self.lr_is_reduced or score_is_high
        else:
            self.lr_is_reduced = score_is_high
        lr_multiplier = self._wanted_lr(run.config) / run.lr
        if self.high_period is None:
            return lr_multiplier, LEAVE_UNCHANGED
        period = self.high_period if self.lr_is_reduced else run.config.target_update_every
        return lr_multiplier, LEAVE_UNCHANGED, period

    def _wanted_lr(self, config: InnerConfig) -> float:
        if self.lr_is_reduced:
            return config.lr_init * self.factor
        return config.lr_init


def make_controller(name: str, config: InnerConfig, seed: int, **controller_args) -> Controller:
    """Build a controller by name.

    `seed` is only used by "random"; `config` only by "ddqn" and "tabular", which
    check their model was trained with the same InnerConfig.
    """
    if name == "noop":
        return NoOp(**controller_args)
    if name == "random":
        return RandomDiscrete(seed=seed, **controller_args)
    if name == "linear_eps":
        return LinearEpsDecay(**controller_args)
    if name == "lr_gate":
        return ScoreGatedLR(**controller_args)
    if name == "ddqn":
        # Trained B, acting greedily; needs model=<path to model.pt>
        from dac.outer_dqn import LearnedController  # local: dac.outer_dqn imports this module

        return LearnedController(config=config, **controller_args)
    if name == "tabular":
        # Tabular baseline for B (dac.outer_tabular); needs model=<path to q_table.json>
        from dac.outer_tabular import TabularController  # local, as for "ddqn"

        return TabularController(config=config, **controller_args)
    raise ValueError(f"unknown controller {name!r}; choose from {CONTROLLER_NAMES}")


def run_episode(seed: int, controller: Controller, config: InnerConfig) -> list[BlockResult]:
    """One full training run of A, with `controller` choosing the multipliers each block."""
    run = InnerRun(seed, config)
    try:
        while not run.done:
            run.run_block(*controller(run))
    finally:
        run.close()
    return run.history


def random_policy_score(config: InnerConfig, episodes: int = 100, seed: int = 0) -> float:
    """Mean return of uniformly random actions, as a reference line for plots."""
    environment = gym.make(config.env_id)
    rng = np.random.default_rng(seed)
    returns = []
    for episode in range(episodes):
        # Only the first reset is seeded; later ones continue the env's own random numbers
        reset_seed = seed if episode == 0 else None
        returns.append(_random_episode_return(environment, rng, reset_seed))
    environment.close()
    return float(np.mean(returns))


def _random_episode_return(environment: gym.Env, rng: np.random.Generator, reset_seed: int | None) -> float:
    n_actions = int(environment.action_space.n)
    environment.reset(seed=reset_seed)
    episode_return = 0.0
    episode_ended = False
    while not episode_ended:
        random_action = int(rng.integers(n_actions))
        _, reward, terminated, truncated, _ = environment.step(random_action)
        episode_return += float(reward)
        episode_ended = terminated or truncated
    return episode_return
