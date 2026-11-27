"""Outer agent B: a Double DQN that sets A's learning rate and epsilon during training.

B uses the same DQNAgent as A (dac/dqn.py). Its environment is OuterEnv: one
episode is one training run of A, one step is one block, the observation is
make_observation(...) and the reward is A's greedy evaluation score after the
block, scaled to [0, 1]. So B's undiscounted episode return is 30 x the run's AUC.

Training steps several OuterEnvs in parallel worker processes, in lockstep.
Every episode lasts n_blocks, so the workers finish together and are reset
together. Each worker draws inner seeds from its own range, well away from the
test seeds 0..29 used for every comparison in results/.

Options (all off by default):
- n_step > 1: n-step targets (discounted sum of n rewards, then a bootstrap from
  the observation n steps later), so B leans less on its own, possibly drifting,
  estimate of the next state's value.
- epsilon_levels: the epsilon entry of B's actions is an absolute level (e.g.
  0.03, 0.1, 0.3) instead of a multiplier, so it can't compound to an extreme.
- lr_levels: same for the learning rate (e.g. 3e-5, 1e-4, 1e-3), so B can move
  it tenfold in one block, like lr_gate does.
- period_levels: each action also sets A's target-copy period (gradient updates
  between target-network copies) to one of these (e.g. 100, 400).
- dueling: dueling network heads for B.

Example:
    python -m dac.outer_dqn --name ddqn_s0 --seed 0 --envs 11 --rounds 36
then evaluate the greedy policy on the test seeds like any other controller:
    python -m dac.run --controller ddqn --controller-arg model=results/outer/ddqn_s0/model.pt \\
        --seeds 0:30 --workers 10 --name p2_ddqn_s0
"""

import argparse
import csv
import dataclasses
import functools
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import gymnasium as gym
import numpy as np
import torch

from dac.controllers import DEFAULT_EPS_MULTS, DEFAULT_LR_MULTS, ActionSet, Choice
from dac.dqn import DQNAgent, ReplayBuffer, make_q_network
from dac.inner import InnerConfig, InnerRun
from dac.outer_env import N_FEATURES, OuterEnv, make_observation, noop_action_of

OUTER_DIR = Path("results/outer")
TRAIN_SEED_START = 1_000          # never train on inner seeds below this (includes the test seeds 0..29)
TRAIN_SEEDS_PER_WORKER = 100_000  # worker i draws from [START + i*PER, START + (i+1)*PER)
FIRST_RESET_SEED_PER_B_SEED = 1_000  # first reset of the parallel envs uses seed * this
# Mixed into B's seed, so B's random streams differ from those InnerRun spawns from the same number
RANDOM_STREAMS_SALT = 7

TRAIN_LOG_FILE = "train_log.csv"
METADATA_FILE = "train_meta.json"
CHECKPOINT_DIR = "checkpoints"
REPLAY_BUFFER_ARRAYS = ("observations", "actions", "rewards", "next_observations", "terminated")


@dataclass(frozen=True)
class OuterConfig:
    n_envs: int = 11              # OuterEnvs (inner training runs) stepped in parallel
    rounds: int = 36              # one episode per env per round, so rounds * n_envs inner runs
    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.99           # episodes are only n_blocks long, so close to undiscounted
    lr: float = 5e-4
    buffer_size: int = 50_000     # more than all of B's training transitions, so nothing is ever dropped
    batch_size: int = 64
    learning_starts: int = 330    # transitions before the first gradient step (one round at 11 envs)
    updates_per_step: int = 11    # gradient steps per lockstep step: one per transition collected
    target_update_every: int = 250
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_fraction: float = 0.5  # fraction of all lockstep steps until epsilon hits eps_end
    n_step: int = 1               # rewards summed before bootstrapping; 1 is plain one-step targets
    epsilon_levels: tuple[float, ...] | None = None  # absolute epsilon levels instead of multipliers
    lr_levels: tuple[float, ...] | None = None       # absolute lr levels instead of multipliers
    period_levels: tuple[int, ...] | None = None     # each action also sets A's target-copy period to one of these
    dueling: bool = False         # dueling heads instead of a plain MLP

    def __post_init__(self) -> None:
        if self.n_step < 1:
            raise ValueError("n_step must be at least 1")


@dataclass(frozen=True)
class Transition:
    """One OuterEnv step as B saw it, or several consecutive steps collapsed into one."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool


def collapse_steps(steps: Sequence[Transition], gamma: float) -> Transition:
    """Collapse consecutive steps into one transition with reward sum_j gamma^j r_j,
    ending at the last step's next observation and terminated flag."""
    discounted_reward = 0.0
    for delay, step in enumerate(steps):
        discounted_reward += gamma**delay * step.reward
    first, last = steps[0], steps[-1]
    return Transition(first.observation, first.action, discounted_reward, last.next_observation, last.terminated)


def n_step_transitions_ready(episode_steps: Sequence[Transition], n_step: int, gamma: float) -> list[Transition]:
    """n-step transitions completed by the episode's latest step, oldest first.

    episode_steps is the episode's one-step transitions so far. The transition
    starting at step t covers t .. t+n_step-1 and is ready once its last step is
    taken. When the episode ends, all remaining ones are ready too: they are cut
    short at the end and keep terminated=True, so they never bootstrap. With
    n_step = 1 each step is returned as taken.
    """
    steps_taken = len(episode_steps)
    newest_full_start = steps_taken - n_step
    episode_has_ended = episode_steps[-1].terminated
    last_start = steps_taken - 1 if episode_has_ended else newest_full_start
    starts = range(max(newest_full_start, 0), last_start + 1)
    return [collapse_steps(episode_steps[start:start + n_step], gamma) for start in starts]


def epsilon_at(step: int, total_steps: int, config: OuterConfig) -> float:
    """B's epsilon at a lockstep step: linear from eps_start to eps_end, then flat."""
    decay_steps = max(1, int(config.eps_decay_fraction * total_steps))
    fraction_of_decay_done = min(step / decay_steps, 1.0)
    return config.eps_start + fraction_of_decay_done * (config.eps_end - config.eps_start)


def outer_action_set(config: OuterConfig, inner_config: InnerConfig) -> ActionSet:
    """B's actions as OuterEnv builds them; ValueError if there's no 'change nothing' action."""
    action_set = ActionSet.build(
        DEFAULT_LR_MULTS, DEFAULT_EPS_MULTS, config.epsilon_levels, config.lr_levels, config.period_levels
    )
    noop_action_of(action_set, inner_config)
    return action_set


def make_agent(config: OuterConfig, n_actions: int, seed: int) -> DQNAgent:
    return DQNAgent(
        N_FEATURES,
        n_actions,
        hidden=config.hidden,
        # n-step targets bootstrap n steps later, hence gamma^n
        gamma=config.gamma**config.n_step,
        lr=config.lr,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_update_every=config.target_update_every,
        double=True,
        grad_clip=config.grad_clip,
        init_seed=seed,
        dueling=config.dueling,
    )


def save_model(path: Path, agent: DQNAgent, config: OuterConfig, inner_config: InnerConfig, seed: int) -> None:
    """Save B's online network plus what LearnedController needs to rebuild the actions and check InnerConfig."""
    torch.save(
        {
            "state_dict": agent.online_net.state_dict(),
            "outer_config": dataclasses.asdict(config),
            "inner_config": dataclasses.asdict(inner_config),
            "lr_mults": list(DEFAULT_LR_MULTS),    # unused when outer_config has lr_levels
            "eps_mults": list(DEFAULT_EPS_MULTS),  # unused when outer_config has epsilon_levels
            "seed": seed,
        },
        path,
    )


def highest_q_value(agent: DQNAgent, observation: np.ndarray) -> float:
    with torch.no_grad():
        q_values = agent.online_net(torch.as_tensor(observation).unsqueeze(0))
    return float(q_values.max())


@dataclass
class RandomStreams:
    """Separate RNG streams for exploration and replay sampling."""

    explore: np.random.Generator
    replay: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> "RandomStreams":
        explore_stream, replay_stream = np.random.SeedSequence([seed, RANDOM_STREAMS_SALT]).spawn(2)
        return cls(np.random.default_rng(explore_stream), np.random.default_rng(replay_stream))


@dataclass
class Progress:
    """Counters kept across rounds."""

    total_steps: int     # lockstep steps in the whole training
    step: int = 0        # lockstep steps so far
    transitions: int = 0 # stored in B's replay buffer so far


@dataclass
class RoundRecord:
    """One round, i.e. one episode in every env."""

    inner_seeds: list[int]
    first_observations: np.ndarray  # (n_envs, N_FEATURES), block-0 observations
    epsilon_at_start: float
    episode_returns: np.ndarray     # (n_envs,), B's undiscounted return per env
    action_counts: np.ndarray       # (n_envs, n_actions)
    losses: list[float] = field(default_factory=list)


@dataclass
class _Training:
    """State shared by one training run of B."""

    config: OuterConfig
    inner_config: InnerConfig
    seed: int
    action_set: ActionSet
    agent: DQNAgent
    random_streams: RandomStreams
    progress: Progress
    envs: gym.vector.VectorEnv

    @property
    def n_actions(self) -> int:
        return len(self.action_set.pairs)


def train(config: OuterConfig, inner_config: InnerConfig, seed: int, out_dir: Path) -> None:
    """Train B. Writes model.pt (final network), checkpoints/round_<NN>.pt (network after each
    round, numbered as train_log.csv's round column), train_log.csv, replay_buffer.npz and
    train_meta.json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CHECKPOINT_DIR).mkdir(exist_ok=True)
    action_set = outer_action_set(config, inner_config)
    agent = make_agent(config, len(action_set.pairs), seed)
    envs = make_parallel_envs(config, inner_config)
    training = _Training(
        config=config,
        inner_config=inner_config,
        seed=seed,
        action_set=action_set,
        agent=agent,
        random_streams=RandomStreams.from_seed(seed),
        progress=Progress(total_steps=config.rounds * inner_config.n_blocks),
        envs=envs,
    )
    started = time.perf_counter()
    try:
        _run_all_rounds(training, out_dir, started)
    finally:
        envs.close()
    _save_replay_buffer(out_dir / "replay_buffer.npz", training.agent.buffer)
    _write_metadata(out_dir / METADATA_FILE, training, wall_seconds=time.perf_counter() - started)


def make_parallel_envs(config: OuterConfig, inner_config: InnerConfig) -> gym.vector.AsyncVectorEnv:
    """n_envs OuterEnvs in worker processes, each with its own inner-seed range."""
    make_env_functions = [
        functools.partial(
            OuterEnv,
            config=inner_config,
            inner_seed_pool=_training_seed_range(worker),
            epsilon_levels=config.epsilon_levels,
            lr_levels=config.lr_levels,
            period_levels=config.period_levels,
        )
        for worker in range(config.n_envs)
    ]
    return gym.vector.AsyncVectorEnv(make_env_functions, context="spawn")


def _training_seed_range(worker: int) -> range:
    first_seed = TRAIN_SEED_START + worker * TRAIN_SEEDS_PER_WORKER
    return range(first_seed, first_seed + TRAIN_SEEDS_PER_WORKER)


def _run_all_rounds(training: _Training, out_dir: Path, started: float) -> None:
    reference_observation = None
    with open(out_dir / TRAIN_LOG_FILE, "w", newline="") as log_file:
        write_log_header(log_file, training.n_actions)
        for round_index in range(training.config.rounds):
            record = _run_round(training, round_index)
            if reference_observation is None:
                # Sanity check always uses the same point: round 0's first block-0 observation
                reference_observation = record.first_observations[0]
            write_log_rows(log_file, round_index, record, training.inner_config)
            log_file.flush()
            _save_round_models(out_dir, round_index, training)
            _print_round_summary(round_index, record, training, reference_observation, started)


def _run_round(training: _Training, round_index: int) -> RoundRecord:
    """One lockstep episode in every env; stores B's transitions and trains B as it goes."""
    config, agent, progress = training.config, training.agent, training.progress
    observations, record = start_round(training.envs, round_index, training.seed, progress, config, training.n_actions)
    episode_steps: list[list[Transition]] = [[] for _ in range(config.n_envs)]

    for _ in range(training.inner_config.n_blocks):
        epsilon = epsilon_at(progress.step, progress.total_steps, config)
        greedy_actions = agent.greedy_actions(observations)
        actions = epsilon_greedy_actions(greedy_actions, epsilon, training.n_actions, training.random_streams.explore)
        next_observations, rewards, terminated, _, _ = training.envs.step(actions)
        for env in range(config.n_envs):
            step = transition_of_env(env, observations, actions, rewards, next_observations, terminated)
            episode_steps[env].append(step)
            progress.transitions += _store_ready_transitions(agent, episode_steps[env], config)
            record.action_counts[env, actions[env]] += 1
        if progress.transitions >= config.learning_starts:
            record.losses.extend(_gradient_steps(training))
        record.episode_returns += rewards
        observations = next_observations
        progress.step += 1

    check_every_env_finished(terminated)
    return record


def start_round(
    envs: gym.vector.VectorEnv,
    round_index: int,
    seed: int,
    progress: Progress,
    config: OuterConfig,
    n_actions: int,
) -> tuple[np.ndarray, RoundRecord]:
    """Reset every env for a new round; returns the block-0 observations and an empty record.

    Only the first round's reset is seeded; later resets continue each env's own random numbers.
    """
    is_first_round = round_index == 0
    reset_seed = seed * FIRST_RESET_SEED_PER_B_SEED if is_first_round else None
    observations, info = envs.reset(seed=reset_seed)
    record = RoundRecord(
        inner_seeds=np.asarray(info["inner_seed"]).tolist(),
        first_observations=observations,
        epsilon_at_start=epsilon_at(progress.step, progress.total_steps, config),
        episode_returns=np.zeros(config.n_envs),
        action_counts=np.zeros((config.n_envs, n_actions), dtype=int),
    )
    return observations, record


def epsilon_greedy_actions(
    greedy_actions: np.ndarray, epsilon: float, n_actions: int, explore_rng: np.random.Generator
) -> np.ndarray:
    """One action per env: random with probability epsilon, else the greedy one.

    Draws the coins and random actions for every env whether used or not, so the
    stream advances the same way whatever epsilon is.
    """
    n_envs = len(greedy_actions)
    coins = explore_rng.random(n_envs)
    random_actions = explore_rng.integers(0, n_actions, size=n_envs)
    return np.where(coins < epsilon, random_actions, greedy_actions)


def transition_of_env(
    env: int,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_observations: np.ndarray,
    terminated: np.ndarray,
) -> Transition:
    """One env's part of a lockstep step, as a one-step transition."""
    return Transition(
        observations[env], int(actions[env]), float(rewards[env]), next_observations[env], bool(terminated[env])
    )


def check_every_env_finished(terminated: np.ndarray) -> None:
    """Every episode lasts n_blocks, so after the last block every env must have terminated."""
    every_env_finished = np.all(terminated)
    if not every_env_finished:
        raise RuntimeError("lockstep broken: not every env finished its episode together")


def _store_ready_transitions(agent: DQNAgent, episode_steps: list[Transition], config: OuterConfig) -> int:
    """Store the n-step transitions the latest step completed; returns the count."""
    ready = n_step_transitions_ready(episode_steps, config.n_step, config.gamma)
    for transition in ready:
        agent.store(transition.observation, transition.action, transition.reward,
                    transition.next_observation, transition.terminated)
    return len(ready)


def _gradient_steps(training: _Training) -> list[float]:
    replay_rng = training.random_streams.replay
    return [training.agent.update(replay_rng)["loss"] for _ in range(training.config.updates_per_step)]


def actions_for_meta(action_set: ActionSet) -> list:
    """[lr entry, epsilon entry] per action, plus the target-copy period if there are periods."""
    if action_set.periods is None:
        return action_set.pairs
    return [[*pair, period] for pair, period in zip(action_set.pairs, action_set.periods)]


def write_log_header(log_file: TextIO, n_actions: int) -> None:
    columns = ["round", "env", "inner_seed", "epsilon_start", "auc"] + [f"a{action}" for action in range(n_actions)]
    csv.writer(log_file).writerow(columns)


def write_log_rows(log_file: TextIO, round_index: int, record: RoundRecord, inner_config: InnerConfig) -> None:
    """One train_log.csv row per env: its inner seed, the round's starting epsilon, AUC and action counts."""
    log = csv.writer(log_file)
    aucs = record.episode_returns / inner_config.n_blocks
    for env, inner_seed in enumerate(record.inner_seeds):
        log.writerow([round_index, env, inner_seed, f"{record.epsilon_at_start:.4f}", f"{aucs[env]:.6f}",
                      *record.action_counts[env].tolist()])


def _save_round_models(out_dir: Path, round_index: int, training: _Training) -> None:
    """model.pt is always the latest network; checkpoints/ keeps one per round."""
    model_path = out_dir / "model.pt"
    save_model(model_path, training.agent, training.config, training.inner_config, training.seed)
    shutil.copyfile(model_path, out_dir / CHECKPOINT_DIR / f"round_{round_index:02d}.pt")


def _print_round_summary(
    round_index: int, record: RoundRecord, training: _Training, reference_observation: np.ndarray, started: float
) -> None:
    config, progress = training.config, training.progress
    aucs = record.episode_returns / training.inner_config.n_blocks
    epsilon_now = epsilon_at(progress.step, progress.total_steps, config)
    loss_text = f"{np.mean(record.losses):.4f}" if record.losses else "-"
    # Rewards are in [0, 1], so no discounted return can exceed this; a bigger Q-value is an overestimate
    highest_possible_return = sum(config.gamma**block for block in range(training.inner_config.n_blocks))
    q_at_block_0 = highest_q_value(training.agent, reference_observation)
    print(
        f"round {round_index + 1}/{config.rounds}: eps {record.epsilon_at_start:.2f}->{epsilon_now:.2f} "
        f"AUC mean {aucs.mean():.3f} (min {aucs.min():.3f}, max {aucs.max():.3f}) "
        f"loss {loss_text} max Q at block 0 {q_at_block_0:.2f} (true value at most {highest_possible_return:.2f}) "
        f"| {time.perf_counter() - started:.0f} s",
        flush=True,
    )


def _save_replay_buffer(path: Path, buffer: ReplayBuffer) -> None:
    """Only the filled part of each buffer array, in storage order."""
    filled = len(buffer)
    np.savez(path, **{name: getattr(buffer, name)[:filled] for name in REPLAY_BUFFER_ARRAYS})


def config_metadata(seed: int, config: OuterConfig, inner_config: InnerConfig) -> dict:
    """The seed and both configs, the first entries of every training metadata file."""
    return {
        "seed": seed,
        "outer_config": dataclasses.asdict(config),
        "inner_config": dataclasses.asdict(inner_config),
    }


def _write_metadata(path: Path, training: _Training, wall_seconds: float) -> None:
    config = training.config
    metadata = {
        **config_metadata(training.seed, config, training.inner_config),
        "actions": actions_for_meta(training.action_set),
        "inner_runs": config.rounds * config.n_envs,
        "transitions": training.progress.transitions,
        "gradient_steps": training.agent.n_updates,
        "wall_seconds": wall_seconds,
        "versions": {"torch": torch.__version__, "gymnasium": gym.__version__, "numpy": np.__version__},
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


class LearnedController:
    """B's trained network acting greedily, with the same interface as the fixed controllers.

    `model` can also be several model files (different B seeds, or the last few round
    checkpoints). Then the action with the highest mean Q-value over the networks is taken,
    which averages out each network's estimation noise (as in Averaged-DQN, Anschel et al.,
    2017). Each network was trained on its own; only acting uses the average.
    """

    def __init__(self, model: str | Sequence[str], config: InnerConfig) -> None:
        model_paths = [model] if isinstance(model, str) else list(model)
        self.nets = []
        self.action_set = None
        for model_path in model_paths:
            net, action_set = _load_network(model_path, config)
            if self.action_set is not None and action_set != self.action_set:
                raise ValueError(f"{model_path} has different actions than {model_paths[0]}")
            self.nets.append(net)
            self.action_set = action_set
        self.actions = self.action_set.pairs

    def q_values(self, observation: np.ndarray) -> torch.Tensor:
        """Q-values for one observation, averaged over the networks; shape (1, n_actions)."""
        batch = torch.as_tensor(observation).unsqueeze(0)
        with torch.no_grad():
            return torch.stack([net(batch) for net in self.nets]).mean(dim=0)

    def __call__(self, run: InnerRun) -> Choice:
        best_action = int(self.q_values(current_observation(run)).argmax(dim=1)[0])
        return self.action_set.choice(best_action, run)


def current_observation(run: InnerRun) -> np.ndarray:
    """The observation OuterEnv would give B at this point of the run."""
    previous = run.history[-2] if len(run.history) >= 2 else None
    return make_observation(run.history[-1], previous, run.config)


def _load_network(model: str, config: InnerConfig) -> tuple[torch.nn.Module, ActionSet]:
    checkpoint = torch.load(model, weights_only=True)
    if inner_config_from_saved(checkpoint) != config:
        raise ValueError(f"{model} was trained with a different InnerConfig than the one given")
    outer_config = checkpoint["outer_config"]
    action_set = saved_action_set(outer_config, checkpoint["lr_mults"], checkpoint["eps_mults"])
    dueling = outer_config.get("dueling", False)  # a missing key means the option was off
    net = make_q_network(N_FEATURES, len(action_set.pairs), tuple(outer_config["hidden"]), dueling)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, action_set


def saved_action_set(
    outer_config: dict,
    lr_multipliers: Sequence[float] = DEFAULT_LR_MULTS,
    epsilon_multipliers: Sequence[float] = DEFAULT_EPS_MULTS,
) -> ActionSet:
    """B's actions rebuilt from a saved outer_config. A missing key means the option was off."""
    return ActionSet.build(
        lr_multipliers,
        epsilon_multipliers,
        outer_config.get("epsilon_levels"),
        outer_config.get("lr_levels"),
        outer_config.get("period_levels"),
    )


def inner_config_from_saved(saved: dict) -> InnerConfig:
    """The InnerConfig saved under saved["inner_config"], with tuples restored (JSON and torch.save give lists)."""
    return InnerConfig(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in saved["inner_config"].items()
    })


def parse_periods(text: str) -> tuple[int, ...]:
    """'100,400' -> (100, 400)."""
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from None


def parse_levels(text: str) -> tuple[float, ...]:
    """'0.03,0.1,0.3' -> (0.03, 0.1, 0.3)."""
    try:
        return tuple(float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dac.outer_dqn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="output folder under results/outer/")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for B (network, exploration, replay, inner-seed draws)")
    parser.add_argument("--envs", type=int, default=OuterConfig.n_envs)
    parser.add_argument("--rounds", type=int, default=OuterConfig.rounds)
    parser.add_argument("--n-step", type=int, default=OuterConfig.n_step,
                        help="rewards summed in B's targets before bootstrapping (default 1)")
    parser.add_argument("--epsilon-levels", type=parse_levels, default=None, metavar="E1,E2,...",
                        help="absolute epsilon levels for B's actions instead of multipliers, e.g. 0.03,0.1,0.3 "
                             "(must include A's eps_init)")
    parser.add_argument("--lr-levels", type=parse_levels, default=None, metavar="L1,L2,...",
                        help="absolute learning-rate levels for B's actions instead of multipliers, "
                             "e.g. 3e-5,1e-4,1e-3 (must include A's lr_init)")
    parser.add_argument("--period-levels", type=parse_periods, default=None, metavar="P1,P2,...",
                        help="each action also sets A's target-copy period to one of these, e.g. 100,400 "
                             "(must include A's target_update_every)")
    parser.add_argument("--dueling", action="store_true", help="dueling network heads for B (a DQN variant)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override an InnerConfig field (repeatable), as in dac.run")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _outer_config_from_arguments(args: argparse.Namespace, inner_config: InnerConfig) -> OuterConfig:
    """Defaults, but learning starts after one round and does one gradient step per transition."""
    n_envs = args.envs
    return dataclasses.replace(
        OuterConfig(),
        n_envs=n_envs,
        rounds=args.rounds,
        learning_starts=n_envs * inner_config.n_blocks,
        updates_per_step=n_envs,
        n_step=args.n_step,
        epsilon_levels=args.epsilon_levels,
        lr_levels=args.lr_levels,
        period_levels=args.period_levels,
        dueling=args.dueling,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Local import to break a cycle: dac.run imports the controllers, which import this module lazily
    from dac.run import config_with_overrides

    inner_config = config_with_overrides(args.overrides)
    config = _outer_config_from_arguments(args, inner_config)
    out_dir = OUTER_DIR / args.name
    if (out_dir / "model.pt").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a model; pass --overwrite to replace")
    train(config, inner_config, args.seed, out_dir)
    print(f"wrote {out_dir}/model.pt, {CHECKPOINT_DIR}/, train_log.csv, replay_buffer.npz, train_meta.json")


if __name__ == "__main__":
    main()
