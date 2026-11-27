"""Run one controller over many seeds in parallel and save the results.

Example:
    python -m dac.run --controller noop --seeds 0:30 --workers 10 --name baseline
    python -m dac.run --controller linear_eps --seeds 0:30 --name lin \\
        --set block_steps=500 --controller-arg decay_blocks=15

Writes results/<name>.csv (one row per seed and block) and results/<name>.json
(config, controller, seeds, timings, library versions).
"""

import argparse
import csv
import dataclasses
import json
import multiprocessing
import platform
import sys
import time
import typing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import numpy as np
import torch

from dac.controllers import CONTROLLER_NAMES, make_controller, run_episode
from dac.inner import BlockResult, InnerConfig

RESULTS_DIR = Path("results")
BLOCK_FIELDS = [field.name for field in dataclasses.fields(BlockResult)]
CSV_COLUMNS = ["controller", "seed"] + BLOCK_FIELDS


def parse_seed_range(text: str) -> list[int]:
    """'A:B' -> [A, A+1, ..., B-1]."""
    try:
        start, stop = (int(part) for part in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected A:B, got {text!r}") from None
    if stop <= start:
        raise argparse.ArgumentTypeError(f"empty seed range {text!r}")
    return list(range(start, stop))


def parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ValueError(f"not a boolean: {text!r}")


def split_key_value(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise ValueError(f"expected key=value, got {text!r}")
    return key, value


def config_with_overrides(overrides: list[str]) -> InnerConfig:
    """Default InnerConfig with key=value overrides, parsed to each field's type."""
    field_types = {field.name: field.type for field in dataclasses.fields(InnerConfig)}
    changes = {}
    for override in overrides:
        key, text = split_key_value(override)
        if key not in field_types:
            raise ValueError(f"unknown InnerConfig field {key!r}; known: {sorted(field_types)}")
        changes[key] = _parse_config_value(key, text, field_types[key])
    return dataclasses.replace(InnerConfig(), **changes)


def _parse_config_value(key: str, text: str, field_type: type):
    if field_type is bool:
        return parse_bool(text)
    if field_type is int:
        return int(text)
    if field_type is float:
        return float(text)
    if field_type is str:
        return text
    if typing.get_origin(field_type) is tuple:  # hidden, written as 64,64
        return tuple(int(part) for part in text.split(","))
    raise ValueError(f"don't know how to parse field {key!r} of type {field_type}")


def parse_scalar(text: str):
    """Parse a controller argument as bool, int or float, falling back to the string."""
    try:
        return parse_bool(text)
    except ValueError:
        pass
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def parse_controller_args(items: list[str]) -> dict:
    """key=value items -> kwargs. A comma-separated value becomes a tuple (lr_mults=0.5,1,2);
    a trailing comma makes a one-element tuple, as in Python (epsilon_levels=0.1,)."""
    controller_args = {}
    for item in items:
        key, text = split_key_value(item)
        if "," in text:
            controller_args[key] = tuple(parse_scalar(part) for part in text.split(",") if part)
        else:
            controller_args[key] = parse_scalar(text)
    return controller_args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dac.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--controller", required=True, choices=CONTROLLER_NAMES)
    parser.add_argument("--seeds", required=True, type=parse_seed_range, help="A:B means seeds A..B-1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--name", required=True, help="output file stem under results/")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override an InnerConfig field (repeatable)")
    parser.add_argument("--controller-arg", dest="controller_args", action="append", default=[],
                        metavar="KEY=VALUE", help="keyword argument for the controller (repeatable)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing result files")
    return parser


@dataclass(frozen=True)
class _SeedTask:
    seed: int
    controller_name: str
    controller_args: dict
    config: InnerConfig


@dataclass(frozen=True)
class _SeedOutcome:
    seed: int
    history: list[BlockResult]
    seconds: float


def _run_one_seed(task: _SeedTask) -> _SeedOutcome:
    """One full training run in a worker. Top-level so the spawn pool can find it."""
    controller = make_controller(task.controller_name, task.config, task.seed, **task.controller_args)
    started = time.perf_counter()
    history = run_episode(task.seed, controller, task.config)
    return _SeedOutcome(task.seed, history, time.perf_counter() - started)


@dataclass
class _BatchResults:
    histories: dict[int, list[BlockResult]] = dataclasses.field(default_factory=dict)
    run_seconds: dict[int, float] = dataclasses.field(default_factory=dict)
    seconds: float = 0.0


def _run_all_seeds(tasks: list[_SeedTask], workers: int) -> _BatchResults:
    batch = _BatchResults()
    started = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for outcome in pool.imap_unordered(_run_one_seed, tasks):
            batch.histories[outcome.seed] = outcome.history
            batch.run_seconds[outcome.seed] = outcome.seconds
            _print_progress(outcome, finished=len(batch.histories), total=len(tasks))
    batch.seconds = time.perf_counter() - started
    return batch


def _print_progress(outcome: _SeedOutcome, finished: int, total: int) -> None:
    learning_scores = [result.eval_score for result in outcome.history[1:]]
    print(
        f"[{finished}/{total}] seed {outcome.seed}: final {learning_scores[-1]:.1f}, "
        f"best {max(learning_scores):.1f}, {outcome.seconds:.1f} s",
        flush=True,
    )


def _write_csv(csv_path: Path, controller_name: str, seeds: list[int], batch: _BatchResults) -> None:
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for seed in seeds:
            for result in batch.histories[seed]:
                writer.writerow({"controller": controller_name, "seed": seed, **dataclasses.asdict(result)})


def _write_metadata(
    json_path: Path,
    args: argparse.Namespace,
    config: InnerConfig,
    controller_args: dict,
    workers: int,
    batch: _BatchResults,
    argv: list[str] | None,
) -> None:
    seeds = args.seeds
    metadata = {
        "name": args.name,
        "config": dataclasses.asdict(config),
        "controller": args.controller,
        "controller_args": controller_args,
        "seeds": seeds,
        "run_wall_seconds": {str(seed): batch.run_seconds[seed] for seed in seeds},
        "batch_wall_seconds": batch.seconds,
        "workers": workers,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "numpy": np.__version__,
        },
        "arguments": argv if argv is not None else sys.argv[1:],
    }
    with open(json_path, "w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_with_overrides(args.overrides)
        controller_args = parse_controller_args(args.controller_args)
        # Build one controller up front so bad controller args fail before any training.
        make_controller(args.controller, config, 0, **controller_args)
    except (ValueError, TypeError) as error:
        parser.error(str(error))

    csv_path = RESULTS_DIR / f"{args.name}.csv"
    json_path = RESULTS_DIR / f"{args.name}.json"
    existing_files = [str(path) for path in (csv_path, json_path) if path.exists()]
    if existing_files and not args.overwrite:
        parser.error(f"{', '.join(existing_files)} already exist(s); pass --overwrite to replace")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    workers = max(1, min(args.workers, len(seeds)))
    tasks = [_SeedTask(seed, args.controller, controller_args, config) for seed in seeds]

    print(f"{args.name}: controller={args.controller} seeds={seeds[0]}..{seeds[-1]} workers={workers}")
    batch = _run_all_seeds(tasks, workers)
    _write_csv(csv_path, args.controller, seeds, batch)
    _write_metadata(json_path, args, config, controller_args, workers, batch, argv)
    print(f"wrote {csv_path} and {json_path} ({batch.seconds:.1f} s)")


if __name__ == "__main__":
    main()
