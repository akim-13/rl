"""One training run of the inner agent A, advanced one block at a time.

A run is n_blocks blocks of block_steps env steps. Between blocks the learning
rate and epsilon can be changed (by a fixed controller or by the outer agent B),
and the greedy policy is evaluated after every block. A run is fully determined
by (seed, config, the sequence of multipliers).
"""

import math
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from dac.dqn import DQNAgent


@dataclass(frozen=True)
class InnerConfig:
    env_id: str = "CartPole-v1"
    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    buffer_size: int = 10_000
    batch_size: int = 64
    learning_starts: int = 500      # random-action env steps before the first gradient step
    # Gradient updates between hard target copies. In a sweep (10 seeds each,
    # lr 1e-3): 10 and 25 diverged, 50 diverged late,
    # 100 learned fastest and stayed stable, 200+ learned more slowly.
    target_update_every: int = 100
    double_dqn: bool = True
    dueling: bool = False           # dueling heads (Wang et al., 2016) instead of a plain MLP
    grad_clip: float = 10.0
    # Starting lr and epsilon. In the pilot (20 seeds each), fixed lr 3e-4, 5e-4 and
    # 2e-3 scored lower; epsilon 0.3 scored lower and 0.05 was not reliably better.
    lr_init: float = 1e-3
    eps_init: float = 0.1
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    eps_min: float = 0.01
    eps_max: float = 1.0
    # The training budget: part of the problem definition, not tuned.
    block_steps: int = 1000         # training env steps per block
    n_blocks: int = 30              # blocks per run = decisions per outer episode
    eval_episodes: int = 10
    eval_seed: int = 10_000         # same evaluation start states for every run and every block
    score_max: float = 500.0
    solved_threshold: float = 475.0

    def __post_init__(self) -> None:
        # Block 0 reports lr_init and eps_init unclamped, so they must already be in range
        if not 0 < self.lr_min <= self.lr_init <= self.lr_max:
            raise ValueError("need 0 < lr_min <= lr_init <= lr_max")
        if not 0 < self.eps_min <= self.eps_init <= self.eps_max <= 1:
            raise ValueError("need 0 < eps_min <= eps_init <= eps_max <= 1")
        if self.block_steps < 1 or self.n_blocks < 1 or self.eval_episodes < 1:
            raise ValueError("block_steps, n_blocks and eval_episodes must be at least 1")


@dataclass(frozen=True)
class BlockResult:
    block: int               # 0 = the untrained network, then 1..n_blocks
    env_steps: int           # training env steps so far
    lr: float                # lr used during this block (block 0: the initial value)
    epsilon: float           # epsilon used during this block (block 0: the initial value)
    eval_score: float        # mean greedy return over the eval episodes after this block
    eval_std: float
    train_return_mean: float # mean return of training episodes that ended in this block (nan if none)
    train_episodes: int      # training episodes that ended in this block
    loss_mean: float         # mean Huber loss over this block's gradient steps (nan if none)
    td_abs_mean: float
    q_mean: float
    grad_norm_mean: float
    train_seconds: float     # wall time spent training
    eval_seconds: float      # wall time spent evaluating


@dataclass
class _ExplorationDraws:
    """Random numbers for one block's epsilon-greedy choices, one entry per step."""

    coins: np.ndarray
    random_actions: np.ndarray


@dataclass
class _BlockTraining:
    finished_episode_returns: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    q_values: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def record_update(self, update_statistics: dict[str, float]) -> None:
        self.losses.append(update_statistics["loss"])
        self.td_errors.append(update_statistics["td_abs"])
        self.q_values.append(update_statistics["q_mean"])
        self.gradient_norms.append(update_statistics["grad_norm"])


@dataclass
class _Evaluation:
    score: float
    spread: float
    seconds: float


def _clamp(value: float, lowest: float, highest: float) -> float:
    return min(max(value, lowest), highest)


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(np.mean(values))


class InnerRun:
    """A Double DQN training run on CartPole, advanced block by block."""

    def __init__(
        self,
        seed: int,
        config: InnerConfig = InnerConfig(),
        *,
        train_env: gym.Env | None = None,
    ) -> None:
        """Set up the run and evaluate the untrained network as block 0.

        train_env is only for tests: it replaces gym.make(config.env_id) for the
        training env, e.g. to impose a very short time limit.
        """
        self.seed = seed
        self.config = config

        # Separate random streams, so using one more (e.g. more exploration)
        # doesn't shift the numbers the others see.
        weights_stream, environment_stream, exploration_stream, replay_stream = np.random.SeedSequence(seed).spawn(4)
        self.explore_rng = np.random.default_rng(exploration_stream)
        self.replay_rng = np.random.default_rng(replay_stream)

        self.train_env = train_env if train_env is not None else gym.make(config.env_id)
        self.eval_envs = [gym.make(config.env_id) for _ in range(config.eval_episodes)]
        self.n_actions = int(self.train_env.action_space.n)
        self.agent = self._make_agent(weights_seed=_first_number_of(weights_stream))

        self.block = 0      # blocks completed
        self.env_steps = 0
        self.lr = config.lr_init
        self.epsilon = config.eps_init

        # Only the first reset is seeded; later resets continue the env's own random numbers
        self._observation, _ = self.train_env.reset(seed=_first_number_of(environment_stream))
        self._episode_return = 0.0

        untrained_evaluation = self._timed_evaluation()
        self.history: list[BlockResult] = [self._block_result(_BlockTraining(), untrained_evaluation)]

    def _make_agent(self, weights_seed: int) -> DQNAgent:
        config = self.config
        return DQNAgent(
            self.train_env.observation_space.shape[0],
            self.n_actions,
            hidden=config.hidden,
            gamma=config.gamma,
            lr=config.lr_init,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_every=config.target_update_every,
            double=config.double_dqn,
            grad_clip=config.grad_clip,
            init_seed=weights_seed,
            dueling=config.dueling,
        )

    @property
    def done(self) -> bool:
        return self.block == self.config.n_blocks

    def run_block(
        self, lr_multiplier: float = 1.0, epsilon_multiplier: float = 1.0, target_update_every: int | None = None
    ) -> BlockResult:
        """Scale lr and epsilon, train for block_steps env steps, then evaluate.

        target_update_every, if given, becomes A's target-copy period (gradient
        updates between target copies) from this block on; None keeps the current one.
        """
        if self.done:
            raise RuntimeError("the run is finished: all blocks have been run")
        multipliers_are_positive = lr_multiplier > 0 and epsilon_multiplier > 0  # also False for nan
        if not multipliers_are_positive:
            raise ValueError(f"multipliers must be positive, got {lr_multiplier}, {epsilon_multiplier}")
        if target_update_every is not None:
            if int(target_update_every) != target_update_every or target_update_every < 1:
                raise ValueError(f"target_update_every must be a positive integer, got {target_update_every}")
            self.agent.target_update_every = int(target_update_every)

        self._apply_multipliers(lr_multiplier, epsilon_multiplier)
        training = self._train_for_one_block()
        evaluation = self._timed_evaluation()
        self.block += 1

        result = self._block_result(training, evaluation)
        self.history.append(result)
        return result

    def _apply_multipliers(self, lr_multiplier: float, epsilon_multiplier: float) -> None:
        config = self.config
        self.lr = _clamp(self.lr * lr_multiplier, config.lr_min, config.lr_max)
        self.epsilon = _clamp(self.epsilon * epsilon_multiplier, config.eps_min, config.eps_max)
        self.agent.set_lr(self.lr)

    def _train_for_one_block(self) -> _BlockTraining:
        started = time.perf_counter()
        exploration_draws = self._draw_exploration_randomness()
        training = _BlockTraining()
        for step_in_block in range(self.config.block_steps):
            action = self._choose_training_action(exploration_draws, step_in_block)
            self._take_training_step(action, training)
        training.seconds = time.perf_counter() - started
        return training

    def _draw_exploration_randomness(self) -> _ExplorationDraws:
        """Draw the whole block's exploration randomness up front.

        Entry t is used at step t whether it's needed or not, so two runs with
        the same seed see the same coins and random actions even if their
        epsilons differ. That makes same-seed controller comparisons fairer.
        """
        block_steps = self.config.block_steps
        coins = self.explore_rng.random(block_steps)
        random_actions = self.explore_rng.integers(0, self.n_actions, size=block_steps)
        return _ExplorationDraws(coins, random_actions)

    def _choose_training_action(self, exploration_draws: _ExplorationDraws, step_in_block: int) -> int:
        coin = exploration_draws.coins[step_in_block]
        random_action = exploration_draws.random_actions[step_in_block]
        still_filling_the_buffer = self.env_steps < self.config.learning_starts
        if still_filling_the_buffer:
            return int(random_action)
        return self.agent.act(self._observation, self.epsilon, coin, random_action)

    def _take_training_step(self, action: int, training: _BlockTraining) -> None:
        next_observation, reward, terminated, truncated, _ = self.train_env.step(action)
        # Only `terminated` is stored: a time-limit truncation isn't a real end, so the target still bootstraps
        self.agent.store(self._observation, action, reward, next_observation, terminated)
        self.env_steps += 1
        self._episode_return += float(reward)

        if self.env_steps >= self.config.learning_starts:
            training.record_update(self.agent.update(self.replay_rng))

        episode_ended = terminated or truncated
        if episode_ended:
            # Episodes run across block boundaries; the return counts for the block the episode ends in
            training.finished_episode_returns.append(self._episode_return)
            self._episode_return = 0.0
            self._observation, _ = self.train_env.reset()
        else:
            self._observation = next_observation

    def _timed_evaluation(self) -> _Evaluation:
        started = time.perf_counter()
        score, spread = self.evaluate()
        return _Evaluation(score, spread, seconds=time.perf_counter() - started)

    def _block_result(self, training: _BlockTraining, evaluation: _Evaluation) -> BlockResult:
        return BlockResult(
            block=self.block,
            env_steps=self.env_steps,
            lr=self.lr,
            epsilon=self.epsilon,
            eval_score=evaluation.score,
            eval_std=evaluation.spread,
            train_return_mean=_mean_or_nan(training.finished_episode_returns),
            train_episodes=len(training.finished_episode_returns),
            loss_mean=_mean_or_nan(training.losses),
            td_abs_mean=_mean_or_nan(training.td_errors),
            q_mean=_mean_or_nan(training.q_values),
            grad_norm_mean=_mean_or_nan(training.gradient_norms),
            train_seconds=training.seconds,
            eval_seconds=evaluation.seconds,
        )

    def evaluate(self) -> tuple[float, float]:
        """Mean and population std of greedy returns from the fixed evaluation start states.

        Uses its own envs and no random numbers, so it never changes the course of training.
        """
        observations = self._reset_evaluation_episodes()
        returns = np.zeros(self.config.eval_episodes)
        running_episodes = list(range(self.config.eval_episodes))
        while running_episodes:
            running_episodes = self._advance_running_episodes(running_episodes, observations, returns)
        return float(returns.mean()), float(returns.std())

    def _reset_evaluation_episodes(self) -> list[np.ndarray]:
        first_seed = self.config.eval_seed
        return [
            environment.reset(seed=first_seed + episode)[0]
            for episode, environment in enumerate(self.eval_envs)
        ]

    def _advance_running_episodes(
        self, running_episodes: list[int], observations: list[np.ndarray], returns: np.ndarray
    ) -> list[int]:
        """One greedy step in every running eval episode; returns the ones still running."""
        # One batched forward pass for all running episodes
        running_observations = np.stack([observations[episode] for episode in running_episodes])
        actions = self.agent.greedy_actions(running_observations)
        still_running = []
        for episode, action in zip(running_episodes, actions):
            observation, reward, terminated, truncated, _ = self.eval_envs[episode].step(int(action))
            returns[episode] += reward
            observations[episode] = observation
            if not (terminated or truncated):
                still_running.append(episode)
        return still_running

    def close(self) -> None:
        self.train_env.close()
        for environment in self.eval_envs:
            environment.close()


def _first_number_of(seed_sequence: np.random.SeedSequence) -> int:
    """An int seed from a SeedSequence, for libraries that only take ints."""
    return int(seed_sequence.generate_state(1)[0])
