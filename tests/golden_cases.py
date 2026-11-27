"""Reference ("golden") cases: recorded outputs of the program that it must reproduce exactly.

Every case runs the program the way a user does, through its command lines
(`python -m dac.run`, `python -m dac.outer_dqn`, `python -m dac.report`), in a
fresh folder, so the tests depend on what the code does and not on how it is
organised inside.

The reference files in tests/golden/ are written by tests/make_golden.py. Do not
regenerate them to make a failing test pass: a failure means behaviour changed.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# A small inner config: a whole run takes about a second, but A still learns a
# little, so the score-dependent controllers have something to react to.
SMALL_CONFIG_FLAGS = [
    "--set", "block_steps=200",
    "--set", "n_blocks=5",
    "--set", "eval_episodes=2",
    "--set", "learning_starts=50",
]
SMALL_SEEDS = "0:2"

# One batch per fixed controller, plus plain DQN to cover that branch of the TD target.
BATCH_CASES = {
    "noop": ["--controller", "noop"],
    "noop_plain_dqn": ["--controller", "noop", "--set", "double_dqn=false"],
    "random": ["--controller", "random"],
    "linear_eps": ["--controller", "linear_eps", "--controller-arg", "decay_blocks=3"],
    "lr_gate": ["--controller", "lr_gate", "--controller-arg", "threshold=20"],
    "lr_gate_sticky": ["--controller", "lr_gate", "--controller-arg", "threshold=20",
                       "--controller-arg", "sticky=true"],
}

# A tiny training run of the outer agent B, then B's greedy policy used as a controller.
OUTER_TRAINING_NAME = "golden"
OUTER_TRAINING_ARGS = ["--name", OUTER_TRAINING_NAME, "--seed", "0", "--envs", "2", "--rounds", "2"]
OUTER_MODEL_PATH = f"results/outer/{OUTER_TRAINING_NAME}/model.pt"
DDQN_BATCH_ARGS = ["--controller", "ddqn", "--controller-arg", f"model={OUTER_MODEL_PATH}"]

# Full-size runs with the default config, checked against the pilot's saved results.
FULL_SIZE_CASES = {"full_noop_seed0": "noop", "full_lr_gate_seed0": "lr_gate"}

# The report is made from fixed input files, so its whole printout is reproducible.
# The small runs never reach the solved score. The pilot's 30-seed results do,
# and with 30 seeds the bootstrap intervals depend on every detail of the resampling.
PILOT_REPORT_INPUTS = {"pilot_noop": "p2_noop", "pilot_lr_gate": "p2_lr_gate"}
REPORT_CASES = {
    "report_small": ["noop", "random", "linear_eps", "lr_gate"],
    "report_pilot": list(PILOT_REPORT_INPUTS),
}
REPORT_PLOT_NAME = "report_plot.png"

TIMING_COLUMNS = {"train_seconds", "eval_seconds"}
JSON_KEYS_THAT_MUST_MATCH = ("config", "controller", "controller_args", "seeds")


def run_module(module: str, args: list[str], cwd: Path) -> str:
    """Run `python -m module args` in cwd, as a user would; return what it printed."""
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    completed = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd, env=environment, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"python -m {module} {' '.join(args)} failed:\n{completed.stderr}")
    return completed.stdout


def run_small_batch(name: str, controller_args: list[str], cwd: Path) -> None:
    run_module(
        "dac.run",
        [*controller_args, *SMALL_CONFIG_FLAGS, "--seeds", SMALL_SEEDS, "--workers", "2", "--name", name],
        cwd,
    )


def train_small_outer_agent(cwd: Path) -> None:
    run_module("dac.outer_dqn", [*OUTER_TRAINING_ARGS, *SMALL_CONFIG_FLAGS], cwd)


def run_full_size_seed0(name: str, controller: str, cwd: Path) -> None:
    run_module("dac.run", ["--controller", controller, "--seeds", "0:1", "--name", name], cwd)


def make_report(csv_paths: list[Path], cwd: Path) -> str:
    return run_module("dac.report", [*map(str, csv_paths), "--out", REPORT_PLOT_NAME], cwd)


def read_csv_without_timings(path: Path) -> list[dict[str, str]]:
    """Rows as strings. Floats are written in full (repr), so equal strings mean bit-equal numbers."""
    with open(path, newline="") as file:
        return [
            {column: value for column, value in row.items() if column not in TIMING_COLUMNS}
            for row in csv.DictReader(file)
        ]
