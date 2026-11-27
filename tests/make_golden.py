"""Write the reference files in tests/golden/ from the code as it is now.

Run from the repository root:
    .venv/bin/python tests/make_golden.py
Do not re-run it to make tests/test_golden.py pass: a failure means behaviour changed (see tests/golden_cases.py).
"""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from golden_cases import (  # noqa: E402
    BATCH_CASES,
    DDQN_BATCH_ARGS,
    FULL_SIZE_CASES,
    GOLDEN_DIR,
    OUTER_TRAINING_NAME,
    PILOT_REPORT_INPUTS,
    REPO_ROOT,
    REPORT_CASES,
    make_report,
    run_small_batch,
    train_small_outer_agent,
)

PILOT_RESULTS = {"full_noop_seed0": "p2_noop", "full_lr_gate_seed0": "p2_lr_gate"}


def copy_result_pair(results_dir: Path, name: str) -> None:
    for suffix in (".csv", ".json"):
        shutil.copy(results_dir / f"{name}{suffix}", GOLDEN_DIR / f"{name}{suffix}")


def copy_seed0_rows_of_pilot_result(golden_name: str, pilot_name: str) -> None:
    """Copy the seed 0 rows of a saved full-size pilot result as a reference file."""
    source = REPO_ROOT / "results" / f"{pilot_name}.csv"
    with open(source, newline="") as file:
        reader = csv.DictReader(file)
        seed0_rows = [row for row in reader if row["seed"] == "0"]
        columns = reader.fieldnames
    with open(GOLDEN_DIR / f"{golden_name}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(seed0_rows)
    metadata = json.loads(source.with_suffix(".json").read_text())
    metadata["seeds"] = [0]  # only seed 0's rows were kept
    (GOLDEN_DIR / f"{golden_name}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        workdir = Path(temporary)
        results_dir = workdir / "results"

        for name, controller_args in BATCH_CASES.items():
            run_small_batch(name, controller_args, workdir)
            copy_result_pair(results_dir, name)

        train_small_outer_agent(workdir)
        outer_dir = results_dir / "outer" / OUTER_TRAINING_NAME
        for file_name in ("model.pt", "train_log.csv", "train_meta.json"):
            shutil.copy(outer_dir / file_name, GOLDEN_DIR / f"outer_{file_name}")
        run_small_batch("ddqn", DDQN_BATCH_ARGS, workdir)
        copy_result_pair(results_dir, "ddqn")

    for golden_name in FULL_SIZE_CASES:
        copy_seed0_rows_of_pilot_result(golden_name, PILOT_RESULTS[golden_name])

    for golden_name, pilot_name in PILOT_REPORT_INPUTS.items():
        for suffix in (".csv", ".json"):
            shutil.copy(REPO_ROOT / "results" / f"{pilot_name}{suffix}", GOLDEN_DIR / f"{golden_name}{suffix}")

    with tempfile.TemporaryDirectory() as temporary:
        for report_name, input_names in REPORT_CASES.items():
            report_inputs = [GOLDEN_DIR / f"{name}.csv" for name in input_names]
            (GOLDEN_DIR / f"{report_name}.txt").write_text(make_report(report_inputs, Path(temporary)))
    print(f"wrote reference files to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
