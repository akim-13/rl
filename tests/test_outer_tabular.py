import csv
import dataclasses
import json

import numpy as np
import pytest

from dac.inner import InnerConfig
from dac.outer_dqn import OuterConfig
from dac.outer_tabular import QTable, StateGrid, all_feature_grid, score_block_grid, state_index, train


def observation_at(block: int, score: float) -> np.ndarray:
    observation = np.zeros(8, dtype=np.float32)
    observation[0] = block / 30
    observation[1] = score / 500
    return observation


@pytest.mark.parametrize("block, score, expected", [
    (0, 0, 0), (0, 99, 0), (0, 100, 1), (0, 399, 3), (0, 400, 4), (0, 500, 4),
    (4, 450, 4), (5, 0, 5), (29, 450, 29), (30, 450, 29),
])
def test_state_index_bins_score_by_100_and_block_by_5(block, score, expected):
    assert state_index(observation_at(block, score), 30) == expected


def test_unvisited_state_prefers_the_change_nothing_action():
    table = QTable(n_states=2, n_actions=3, bootstrap_discount=0.9, noop_action=2)
    assert table.greedy(0) == 2


def test_update_moves_towards_the_n_step_target():
    table = QTable(n_states=2, n_actions=3, bootstrap_discount=0.5, noop_action=2)
    table.q[1] = [0.0, 4.0, 1.0]
    td_error = table.update(0, 0, reward=1.0, next_state=1, terminated=False)
    assert td_error == pytest.approx(3.0)       # 1 + 0.5 * 4 - 0
    assert table.q[0, 0] == pytest.approx(0.3)  # step size 0.1
    table.update(0, 1, reward=1.0, next_state=1, terminated=True)
    assert table.q[0, 1] == pytest.approx(0.1)  # no bootstrap at the end of a run
    assert table.greedy(0) == 0


def test_grid_index_is_mixed_radix_with_first_feature_slowest():
    grid = StateGrid(features=(2, 5), edges=((0.0,), (1.0, 2.0)))
    observation = np.zeros(8, dtype=np.float32)
    observation[2] = 0.5
    observation[5] = 1.5
    assert grid.n_states == 6
    assert grid.index(observation) == 1 * 3 + 1
    assert StateGrid.from_json(grid.to_json()) == grid


def test_all_feature_grid_bins_lr_by_level_and_drops_constant_features():
    grid = all_feature_grid("p2_noop", (3e-5, 1e-4, 1e-3), InnerConfig())
    assert len(grid.edges[4]) == 2      # one bin per lr level
    assert grid.edges[5] == ()          # epsilon never changes under fixed settings
    assert grid.n_states == 3**6 * 3


PINNED_INNER_CONFIG = InnerConfig(block_steps=200, n_blocks=5, eval_episodes=2, learning_starts=50)
PINNED_Q_VALUES = [
    [0.06552008001048759, 0.0, 0.018564353586718336],
    [0.0, 0.0, 0.00534618],
    [0.0, 0.0, 0.0],
    [0.037076748429687036, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]
PINNED_VISITS = [[16, 0, 2], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]]
PINNED_LOG_ROWS = [
    ["round", "env", "inner_seed", "epsilon_start", "auc", "a0", "a1", "a2"],
    ["0", "0", "86062", "1.0000", "0.185400", "4", "0", "1"],
    ["0", "1", "148318", "1.0000", "0.018400", "3", "0", "2"],
    ["1", "0", "64696", "0.0500", "0.018000", "5", "0", "0"],
    ["1", "1", "152182", "0.0500", "0.019000", "5", "0", "0"],
]


def test_training_reproduces_the_pinned_table_and_log(tmp_path):
    outer_config = dataclasses.replace(
        OuterConfig(), n_envs=2, rounds=2, n_step=3, lr_levels=(3e-5, 1e-4, 1e-3), epsilon_levels=(0.1,)
    )
    train(outer_config, PINNED_INNER_CONFIG, 0, tmp_path, score_block_grid(PINNED_INNER_CONFIG.n_blocks))

    saved = json.loads((tmp_path / "q_table.json").read_text())
    assert saved["q"] == PINNED_Q_VALUES
    assert saved["visits"] == PINNED_VISITS
    assert saved["noop_action"] == 2
    with open(tmp_path / "train_log.csv", newline="") as log_file:
        assert list(csv.reader(log_file)) == PINNED_LOG_ROWS
    metadata = json.loads((tmp_path / "train_meta.json").read_text())
    assert metadata["updates"] == 20
