"""Tabular baseline for B: Q-learning over a coarse grid of B's observation.

It tests whether B's problem needs function approximation. Everything except
the value function is the same as for B trained with dac.outer_dqn --n-step 10
--lr-levels 3e-5,1e-4,1e-3 --epsilon-levels 0.1: the environment, actions,
training budget, inner seeds, n-step targets and exploration schedule. Instead
of a network over all 8 observation features, the values are a table with one
row per grid cell, updated with the tabular Q-learning rule: one update per
n-step transition as soon as it is complete, with a constant step size.

Two grids (--state):
- score_block (the default): A's last greedy score in 5 bins (0-100, 100-200,
  200-300, 300-400, 400-500) x the block in 6 bins of 5 blocks, so 30 cells.
- all_features: all 8 features, each in 3 equal-width bins over its range in
  fixed-settings runs on training seeds (--bin-range-from, a dac.run result
  name). The lr feature gets one bin per lr level, and a feature that never
  varies in those runs gets a single bin.

Examples:
    python -m dac.outer_tabular --name tab_lronly_s0 --seed 0
    python -m dac.outer_tabular --name tab8_lronly_s0 --seed 0 \\
        --state all_features --bin-range-from trainseeds_noop
then evaluate the table like any other controller:
    python -m dac.run --controller tabular \\
        --controller-arg model=results/outer/tab_lronly_s0/q_table.json --seeds 30:60 --name val_tab_lronly_s0
"""

import argparse
import csv
import dataclasses
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from dac.controllers import ActionSet, Choice
from dac.inner import BlockResult, InnerConfig, InnerRun
from dac.outer_dqn import (
    METADATA_FILE,
    OUTER_DIR,
    TRAIN_LOG_FILE,
    OuterConfig,
    Progress,
    RandomStreams,
    RoundRecord,
    Transition,
    actions_for_meta,
    check_every_env_finished,
    config_metadata,
    current_observation,
    epsilon_at,
    epsilon_greedy_actions,
    inner_config_from_saved,
    make_parallel_envs,
    n_step_transitions_ready,
    outer_action_set,
    parse_levels,
    saved_action_set,
    start_round,
    transition_of_env,
    write_log_header,
    write_log_rows,
)
from dac.outer_env import N_FEATURES, make_observation, noop_action_of

# Positions in make_observation's feature vector
PROGRESS_FEATURE = 0
SCORE_FEATURE = 1
LR_FEATURE = 4

SCORE_BIN_EDGES = (0.2, 0.4, 0.6, 0.8)  # on score / 500, so 100, 200, 300, 400
BLOCKS_PER_BIN = 5
BINS_PER_FEATURE = 3
CONSTANT_FEATURE_SPREAD = 1e-9  # a feature whose range is narrower than this gets a single bin
STEP_SIZE = 0.1
TABLE_FILE = "q_table.json"

# The command's defaults are the settings of the dac.outer_dqn run the table is compared with
DEFAULT_N_STEP = 10
DEFAULT_LR_LEVELS = (3e-5, 1e-4, 1e-3)
DEFAULT_EPSILON_LEVELS = (0.1,)


@dataclass(frozen=True)
class StateGrid:
    """A grid over some of B's observation features: each feature is cut at its edges."""

    features: tuple[int, ...]
    edges: tuple[tuple[float, ...], ...]  # inner bin edges per feature, ascending

    @property
    def n_states(self) -> int:
        return math.prod(len(feature_edges) + 1 for feature_edges in self.edges)

    def index(self, observation: np.ndarray) -> int:
        """Cell number; the first feature varies slowest. Values beyond the edges fall in the end bins."""
        cell = 0
        for feature, feature_edges in zip(self.features, self.edges):
            bin_number = int(np.searchsorted(feature_edges, float(observation[feature]), side="right"))
            cell = cell * (len(feature_edges) + 1) + bin_number
        return cell

    def to_json(self) -> dict:
        return {"features": list(self.features), "edges": [list(feature_edges) for feature_edges in self.edges]}

    @classmethod
    def from_json(cls, saved: dict) -> "StateGrid":
        return cls(tuple(saved["features"]), tuple(tuple(feature_edges) for feature_edges in saved["edges"]))


def score_block_grid(n_blocks: int) -> StateGrid:
    """Block (block / n_blocks) in bins of BLOCKS_PER_BIN x greedy score in 100-point bins."""
    # Edges half a block before each bin start, so float rounding of block / n_blocks can't move a block
    block_edges = tuple((start - 0.5) / n_blocks for start in range(BLOCKS_PER_BIN, n_blocks, BLOCKS_PER_BIN))
    return StateGrid(features=(PROGRESS_FEATURE, SCORE_FEATURE), edges=(block_edges, SCORE_BIN_EDGES))


def state_index(observation: np.ndarray, n_blocks: int) -> int:
    """Cell of an observation in the score x block grid."""
    return score_block_grid(n_blocks).index(observation)


def all_feature_grid(
    range_runs: str, lr_levels: Sequence[float] | None, config: InnerConfig, bins: int = BINS_PER_FEATURE
) -> StateGrid:
    """Every feature in `bins` equal-width bins over its min..max in the dac.run result `range_runs`.

    The lr feature gets one bin per lr level instead, and a feature that never
    varies in those runs gets a single bin.
    """
    observations = _observations_of(Path("results") / f"{range_runs}.csv", config)
    edges = []
    for feature in range(N_FEATURES):
        if feature == LR_FEATURE and lr_levels:
            edges.append(_lr_level_edges(lr_levels, config))
        else:
            edges.append(_equal_width_edges(observations[:, feature], bins))
    return StateGrid(features=tuple(range(N_FEATURES)), edges=tuple(edges))


def _lr_level_edges(lr_levels: Sequence[float], config: InnerConfig) -> tuple[float, ...]:
    """Edges halfway between neighbouring lr levels' feature values, so each level has its own bin."""
    positions = sorted(_lr_feature_value(level, config) for level in lr_levels)
    return tuple((lower + upper) / 2 for lower, upper in zip(positions, positions[1:]))


def _equal_width_edges(values: np.ndarray, bins: int) -> tuple[float, ...]:
    lowest, highest = float(values.min()), float(values.max())
    feature_is_constant = highest - lowest < CONSTANT_FEATURE_SPREAD
    if feature_is_constant:
        return ()
    width = (highest - lowest) / bins
    return tuple(lowest + width * cut for cut in range(1, bins))


def _lr_feature_value(lr: float, config: InnerConfig) -> float:
    """The lr feature for this lr: its position on a log scale from lr_min to lr_max, as in make_observation."""
    return (math.log(lr) - math.log(config.lr_min)) / (math.log(config.lr_max) - math.log(config.lr_min))


def _observations_of(path: Path, config: InnerConfig) -> np.ndarray:
    """B's observation after every block of every run in a dac.run result file."""
    observations = []
    for history in _histories_by_seed(path).values():
        history.sort(key=lambda result: result.block)
        previous = None
        for result in history:
            observations.append(make_observation(result, previous, config))
            previous = result
    return np.array(observations)


def _histories_by_seed(path: Path) -> dict[int, list[BlockResult]]:
    histories: dict[int, list[BlockResult]] = {}
    with open(path) as result_file:
        for row in csv.DictReader(result_file):
            histories.setdefault(int(row["seed"]), []).append(_block_result_from_row(row))
    return histories


def _block_result_from_row(row: dict[str, str]) -> BlockResult:
    values = {}
    for block_result_field in dataclasses.fields(BlockResult):
        text = row[block_result_field.name]
        is_integer_field = block_result_field.type in (int, "int")
        values[block_result_field.name] = int(float(text)) if is_integer_field else float(text)
    return BlockResult(**values)


class QTable:
    """Q-values in a table, updated with the tabular Q-learning rule.

    In a cell that has never been updated, all actions tie at zero. Ties are
    broken towards the "change nothing" action, so an unvisited cell keeps
    A's settings, and otherwise towards the lowest action index.
    """

    def __init__(self, n_states: int, n_actions: int, bootstrap_discount: float, noop_action: int) -> None:
        self.q = np.zeros((n_states, n_actions))
        self.visits = np.zeros((n_states, n_actions), dtype=int)
        self.bootstrap_discount = bootstrap_discount  # gamma^n for n-step targets
        self.noop_action = noop_action

    def greedy(self, state: int) -> int:
        values = self.q[state]
        best_actions = np.flatnonzero(values == values.max())
        noop_is_among_the_best = self.noop_action in best_actions
        if noop_is_among_the_best:
            return self.noop_action
        return int(best_actions[0])

    def update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> float:
        """One Q-learning step towards reward + gamma^n * max Q(next state); returns the TD error."""
        bootstrap = 0.0 if terminated else self.bootstrap_discount * self.q[next_state].max()
        td_error = reward + bootstrap - self.q[state, action]
        self.q[state, action] += STEP_SIZE * td_error
        self.visits[state, action] += 1
        return float(td_error)

    @property
    def cells_visited(self) -> int:
        return int((self.visits.sum(axis=1) > 0).sum())


@dataclass
class _TabularTraining:
    """State shared by one training run of the table."""

    config: OuterConfig
    inner_config: InnerConfig
    seed: int
    grid: StateGrid
    action_set: ActionSet
    table: QTable
    explore_rng: np.random.Generator
    progress: Progress
    envs: gym.vector.VectorEnv

    @property
    def n_actions(self) -> int:
        return len(self.action_set.pairs)


def train(config: OuterConfig, inner_config: InnerConfig, seed: int, out_dir: Path, grid: StateGrid) -> None:
    """Train the table; writes q_table.json (after every round), train_log.csv and train_meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    action_set = outer_action_set(config, inner_config)
    noop_action = noop_action_of(action_set, inner_config)
    # n-step targets bootstrap n steps later, hence gamma^n
    table = QTable(grid.n_states, len(action_set.pairs), config.gamma**config.n_step, noop_action)
    started = time.perf_counter()
    envs = make_parallel_envs(config, inner_config)
    training = _TabularTraining(
        config=config,
        inner_config=inner_config,
        seed=seed,
        grid=grid,
        action_set=action_set,
        table=table,
        # The same exploration stream as dac.outer_dqn, so B seed k explores with the same coins
        explore_rng=RandomStreams.from_seed(seed).explore,
        progress=Progress(total_steps=config.rounds * inner_config.n_blocks),
        envs=envs,
    )
    try:
        _run_all_rounds(training, out_dir, started)
    finally:
        envs.close()
    _write_metadata(out_dir / METADATA_FILE, training, wall_seconds=time.perf_counter() - started)


def _run_all_rounds(training: _TabularTraining, out_dir: Path, started: float) -> None:
    with open(out_dir / TRAIN_LOG_FILE, "w", newline="") as log_file:
        write_log_header(log_file, training.n_actions)
        for round_index in range(training.config.rounds):
            record, td_errors = _run_round(training, round_index)
            write_log_rows(log_file, round_index, record, training.inner_config)
            log_file.flush()
            save_table(out_dir / TABLE_FILE, training.table, training.config, training.inner_config,
                       training.seed, training.grid)
            _print_round_summary(round_index, record, td_errors, training, started)


def _run_round(training: _TabularTraining, round_index: int) -> tuple[RoundRecord, list[float]]:
    """One lockstep episode in every env, updating the table as it goes; returns the record and TD errors."""
    config, progress = training.config, training.progress
    observations, record = start_round(training.envs, round_index, training.seed, progress, config, training.n_actions)
    episode_steps: list[list[Transition]] = [[] for _ in range(config.n_envs)]
    td_errors: list[float] = []

    for _ in range(training.inner_config.n_blocks):
        epsilon = epsilon_at(progress.step, progress.total_steps, config)
        actions = epsilon_greedy_actions(
            _greedy_actions(training, observations), epsilon, training.n_actions, training.explore_rng
        )
        next_observations, rewards, terminated, _, _ = training.envs.step(actions)
        for env in range(config.n_envs):
            step = transition_of_env(env, observations, actions, rewards, next_observations, terminated)
            episode_steps[env].append(step)
            td_errors.extend(_learn_from_ready_transitions(training, episode_steps[env]))
            record.action_counts[env, actions[env]] += 1
        record.episode_returns += rewards
        observations = next_observations
        progress.step += 1

    check_every_env_finished(terminated)
    return record, td_errors


def _greedy_actions(training: _TabularTraining, observations: np.ndarray) -> np.ndarray:
    return np.array([training.table.greedy(training.grid.index(observation)) for observation in observations])


def _learn_from_ready_transitions(training: _TabularTraining, episode_steps: list[Transition]) -> list[float]:
    """Update the table on the n-step transitions the latest step completed; returns their TD errors."""
    config, grid = training.config, training.grid
    ready = n_step_transitions_ready(episode_steps, config.n_step, config.gamma)
    return [
        training.table.update(
            grid.index(transition.observation),
            transition.action,
            transition.reward,
            grid.index(transition.next_observation),
            transition.terminated,
        )
        for transition in ready
    ]


def _print_round_summary(
    round_index: int, record: RoundRecord, td_errors: list[float], training: _TabularTraining, started: float
) -> None:
    aucs = record.episode_returns / training.inner_config.n_blocks
    table = training.table
    print(
        f"round {round_index + 1}/{training.config.rounds}: eps {record.epsilon_at_start:.2f} "
        f"AUC mean {aucs.mean():.3f} mean |TD error| {np.mean(np.abs(td_errors)):.3f} "
        f"cells visited {table.cells_visited}/{len(table.q)} "
        f"| {time.perf_counter() - started:.0f} s",
        flush=True,
    )


def _write_metadata(path: Path, training: _TabularTraining, wall_seconds: float) -> None:
    config, grid = training.config, training.grid
    metadata = {
        **config_metadata(training.seed, config, training.inner_config),
        "tabular": {"grid": grid.to_json(), "n_states": grid.n_states, "step_size": STEP_SIZE},
        "actions": actions_for_meta(training.action_set),
        "inner_runs": config.rounds * config.n_envs,
        "updates": int(training.table.visits.sum()),
        "wall_seconds": wall_seconds,
        "versions": {"gymnasium": gym.__version__, "numpy": np.__version__},
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def save_table(
    path: Path, table: QTable, config: OuterConfig, inner_config: InnerConfig, seed: int, grid: StateGrid
) -> None:
    path.write_text(json.dumps({
        "grid": grid.to_json(),
        "q": table.q.tolist(),
        "visits": table.visits.tolist(),
        "noop_action": table.noop_action,
        "outer_config": dataclasses.asdict(config),
        "inner_config": dataclasses.asdict(inner_config),
        "seed": seed,
    }) + "\n")


class TabularController:
    """The trained table acting greedily, with the same interface as the other controllers."""

    def __init__(self, model: str, config: InnerConfig) -> None:
        saved = json.loads(Path(model).read_text())
        if inner_config_from_saved(saved) != config:
            raise ValueError(f"{model} was trained with a different InnerConfig than the one given")
        self.action_set = saved_action_set(saved["outer_config"])
        self.table = QTable(len(saved["q"]), len(self.action_set.pairs), 0.0, saved["noop_action"])
        self.table.q = np.array(saved["q"])
        # A table saved without a grid uses the score x block grid
        self.grid = StateGrid.from_json(saved["grid"]) if "grid" in saved else score_block_grid(config.n_blocks)

    def __call__(self, run: InnerRun) -> Choice:
        action = self.table.greedy(self.grid.index(current_observation(run)))
        return self.action_set.choice(action, run)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dac.outer_tabular", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="output folder under results/outer/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=int, default=OuterConfig.n_envs)
    parser.add_argument("--rounds", type=int, default=OuterConfig.rounds)
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP)
    parser.add_argument("--lr-levels", type=parse_levels, default=DEFAULT_LR_LEVELS)
    parser.add_argument("--epsilon-levels", type=parse_levels, default=DEFAULT_EPSILON_LEVELS)
    parser.add_argument("--state", choices=("score_block", "all_features"), default="score_block")
    parser.add_argument("--bin-range-from", metavar="RESULT_NAME",
                        help="with --state all_features: dac.run result (fixed settings, training seeds) "
                             "whose observations set each feature's bin range")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _grid_from_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser, config: OuterConfig, inner_config: InnerConfig
) -> StateGrid:
    if args.state == "score_block":
        return score_block_grid(inner_config.n_blocks)
    if not args.bin_range_from:
        parser.error("--state all_features needs --bin-range-from")
    return all_feature_grid(args.bin_range_from, config.lr_levels, inner_config)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    inner_config = InnerConfig()
    config = dataclasses.replace(OuterConfig(), n_envs=args.envs, rounds=args.rounds, n_step=args.n_step,
                                 lr_levels=args.lr_levels, epsilon_levels=args.epsilon_levels)
    out_dir = OUTER_DIR / args.name
    if (out_dir / TABLE_FILE).exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a table; pass --overwrite to replace")
    grid = _grid_from_arguments(args, parser, config, inner_config)
    train(config, inner_config, args.seed, out_dir, grid)
    print(f"wrote {out_dir}/{TABLE_FILE}, {TRAIN_LOG_FILE}, {METADATA_FILE}")


if __name__ == "__main__":
    main()
