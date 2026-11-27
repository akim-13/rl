"""Summary table and learning-curve plot from result files written by dac.run.

Example:
    python -m dac.report results/baseline.csv results/random.csv --out results/plots/cmp.png

Each CSV needs its .json next to it (for score_max, block_steps and
solved_threshold). With several CSVs, each later one is also compared with the
first, seed by seed, on the seeds they share.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # only write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dac.controllers import random_policy_score  # noqa: E402
from dac.inner import InnerConfig  # noqa: E402
from dac.metrics import auc, bootstrap_ci, steps_to_threshold  # noqa: E402

# One colour per CSV, from a colourblind-safe palette. Plot at most eight CSVs,
# since a ninth would reuse the first colour.
LINE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE_INK = "#555555"
GRID_INK = "#e1e0d9"


@dataclass
class ResultSet:
    """One results/<name>.csv + .json pair."""

    name: str
    config: dict
    env_steps: np.ndarray          # (n_blocks + 1,), block 0 first
    scores: dict[int, np.ndarray]  # seed -> eval scores for blocks 0..n_blocks
    seconds: dict[int, float]      # seed -> training + evaluation wall seconds for the whole run

    def learning_scores(self, seed: int) -> np.ndarray:
        """Scores for blocks 1..n_blocks; block 0 is the untrained network."""
        return self.scores[seed][1:]

    def auc_by_seed(self) -> dict[int, float]:
        score_max = self.config["score_max"]
        return {seed: auc(self.learning_scores(seed), score_max) for seed in self.scores}


def load_results(csv_path: Path) -> ResultSet:
    metadata = _read_metadata(csv_path.with_suffix(".json"))
    rows_by_seed = _read_rows_by_seed(csv_path)

    scores, seconds = {}, {}
    env_steps = None
    for seed in sorted(rows_by_seed):
        rows = sorted(rows_by_seed[seed], key=_block_number)
        scores[seed] = np.array([float(row["eval_score"]) for row in rows])
        seconds[seed] = _wall_seconds_of_run(rows)
        env_steps_of_this_seed = np.array([int(row["env_steps"]) for row in rows])
        if env_steps is None:
            env_steps = env_steps_of_this_seed
        elif not np.array_equal(env_steps_of_this_seed, env_steps):
            raise SystemExit(f"{csv_path}: seed {seed} has different blocks from the other seeds")
    return ResultSet(csv_path.stem, metadata["config"], env_steps, scores, seconds)


def _read_metadata(json_path: Path) -> dict:
    if not json_path.exists():
        raise SystemExit(f"missing {json_path} (written by dac.run next to the CSV)")
    return json.loads(json_path.read_text())


def _read_rows_by_seed(csv_path: Path) -> dict[int, list[dict]]:
    rows_by_seed: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as file:
        for row in csv.DictReader(file):
            rows_by_seed[int(row["seed"])].append(row)
    if not rows_by_seed:
        raise SystemExit(f"{csv_path} has no rows")
    return rows_by_seed


def _block_number(row: dict) -> int:
    return int(row["block"])


def _wall_seconds_of_run(rows: list[dict]) -> float:
    return sum(float(row["train_seconds"]) + float(row["eval_seconds"]) for row in rows)


@dataclass
class Summary:
    """One line of the summary table."""

    name: str
    runs: int
    auc_mean: float
    auc_ci: tuple[float, float]
    auc_sd: float
    final_mean: float
    solved_fraction: float
    median_steps: float
    seconds_mean: float


def summarise(results: ResultSet) -> Summary:
    config = results.config
    aucs = np.array(list(results.auc_by_seed().values()))
    steps_to_solve = np.array([
        steps_to_threshold(results.learning_scores(seed), config["block_steps"], config["solved_threshold"])
        for seed in results.scores
    ])
    steps_of_solved_runs = steps_to_solve[np.isfinite(steps_to_solve)]
    some_run_solved = len(steps_of_solved_runs) > 0
    final_scores = [seed_scores[-1] for seed_scores in results.scores.values()]
    return Summary(
        name=results.name,
        runs=len(aucs),
        auc_mean=float(aucs.mean()),
        auc_ci=bootstrap_ci(aucs),
        # Sample standard deviation (ddof=1) of AUC across seeds.
        auc_sd=float(aucs.std(ddof=1)) if len(aucs) > 1 else math.nan,
        final_mean=float(np.mean(final_scores)),
        solved_fraction=len(steps_of_solved_runs) / len(steps_to_solve),
        median_steps=float(np.median(steps_of_solved_runs)) if some_run_solved else math.nan,
        seconds_mean=float(np.mean(list(results.seconds.values()))),
    )


def print_summary(summaries: list[Summary], threshold: float) -> None:
    name_width = max(len("name"), *(len(summary.name) for summary in summaries))
    header = (
        f"{'name':<{name_width}}  {'runs':>4}  {'AUC':>6}  {'95% CI':>16}  {'AUC sd':>6}  "
        f"{'final':>6}  {'reached':>7}  {'median steps':>12}  {'s/run':>6}"
    )
    print(f"Threshold for 'reached' and steps-to-threshold: eval score >= {threshold:g}")
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(_summary_line(summary, name_width))


def _summary_line(summary: Summary, name_width: int) -> str:
    low, high = summary.auc_ci
    interval = f"[{low:.3f}, {high:.3f}]"
    median_steps = "-" if math.isnan(summary.median_steps) else f"{summary.median_steps:.0f}"
    return (
        f"{summary.name:<{name_width}}  {summary.runs:>4}  {summary.auc_mean:>6.3f}  "
        f"{interval:>16}  {summary.auc_sd:>6.3f}  "
        f"{summary.final_mean:>6.1f}  {summary.solved_fraction:>7.2f}  {median_steps:>12}  "
        f"{summary.seconds_mean:>6.1f}"
    )


def print_paired_differences(all_results: list[ResultSet]) -> None:
    """Mean AUC difference from the first result set over shared seeds, with a CI."""
    first = all_results[0]
    first_aucs = first.auc_by_seed()
    print(f"\nPaired AUC difference against {first.name} (other minus {first.name}, same seeds):")
    for other in all_results[1:]:
        other_aucs = other.auc_by_seed()
        shared_seeds = sorted(set(first_aucs) & set(other_aucs))
        if not shared_seeds:
            print(f"  {other.name}: no shared seeds")
            continue
        differences = np.array([other_aucs[seed] - first_aucs[seed] for seed in shared_seeds])
        low, high = bootstrap_ci(differences)
        print(
            f"  {other.name}: mean {differences.mean():+.3f}, 95% CI [{low:+.3f}, {high:+.3f}], "
            f"{len(shared_seeds)} shared seeds"
        )


def plot_learning_curves(all_results: list[ResultSet], out_path: Path, title: str | None) -> None:
    """Mean eval score over seeds against env steps, with a 95% bootstrap CI band."""
    plt.rcParams.update({"font.size": 12})
    figure, axes = plt.subplots(figsize=(7, 4.5), layout="constrained")

    for position, results in enumerate(all_results):
        colour = LINE_COLOURS[position % len(LINE_COLOURS)]
        _plot_mean_with_band(axes, results, colour)
    config = all_results[0].config
    _draw_reference_lines(axes, config)
    _style_axes(axes, config, title)
    # Legend below the axes, since the curves can end up in any corner.
    figure.legend(loc="outside lower center", ncol=2, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"\nwrote {out_path}")


def _plot_mean_with_band(axes, results: ResultSet, colour: str) -> None:
    scores_by_seed_and_block = np.stack(list(results.scores.values()))
    seed_count, block_count = scores_by_seed_and_block.shape
    mean_scores = scores_by_seed_and_block.mean(axis=0)
    bands = np.array([bootstrap_ci(scores_by_seed_and_block[:, block]) for block in range(block_count)])
    lower_edges, upper_edges = bands[:, 0], bands[:, 1]
    label = f"{results.name} ({seed_count} seeds)"
    axes.plot(results.env_steps, mean_scores, color=colour, linewidth=2, label=label)
    axes.fill_between(results.env_steps, lower_edges, upper_edges, color=colour, alpha=0.2, linewidth=0)


def _draw_reference_lines(axes, config: dict) -> None:
    axes.axhline(config["solved_threshold"], color=REFERENCE_INK, linestyle="--", linewidth=1.5,
                 label=f"solved threshold ({config['solved_threshold']:g})")
    # JSON gives hidden back as a list, but InnerConfig needs a tuple.
    inner_config = InnerConfig(**{**config, "hidden": tuple(config["hidden"])})
    random_score = random_policy_score(inner_config)
    axes.axhline(random_score, color=REFERENCE_INK, linestyle=":", linewidth=1.5,
                 label=f"random actions ({random_score:.0f})")


def _style_axes(axes, config: dict, title: str | None) -> None:
    axes.set_xlabel("training environment steps")
    axes.set_ylabel("evaluation score (mean greedy return)")
    axes.set_ylim(0, config["score_max"] * 1.05)
    axes.set_xlim(left=0)
    axes.grid(True, color=GRID_INK, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    if title:
        axes.set_title(title)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dac.report", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csvs", nargs="+", type=Path, help="results/<name>.csv files written by dac.run")
    parser.add_argument("--out", type=Path, default=None, help="plot path (default results/plots/<first name>.png)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    all_results = [load_results(path) for path in args.csvs]
    solved_threshold = all_results[0].config["solved_threshold"]
    print_summary([summarise(results) for results in all_results], solved_threshold)
    if len(all_results) > 1:
        print_paired_differences(all_results)

    out_path = args.out or Path("results") / "plots" / f"{all_results[0].name}.png"
    plot_learning_curves(all_results, out_path, args.title)


if __name__ == "__main__":
    main()
