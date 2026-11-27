"""Figures for the evidence brief (report/brief.tex), written as PDFs to report/brief_figs/.

Run from the repository root:  .venv/bin/python scripts/brief/brief_figures.py
Every number comes from results/ (and, for the offline refits, from
scripts/brief/offline_refit_rerun_2026-11-23.out.txt). AUC and CIs as in dac.metrics.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dac.metrics import bootstrap_ci  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "report" / "brief_figs"

# One colour per entity, the same in every figure.
GREY, BLUE, ORANGE, AQUA, YELLOW = "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LR_SHADES = {3e-5: "#b9d4f4", 1e-4: "#6fa3e6", 1e-3: "#1d5aa8"}  # one hue, light = low lr

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
})
WIDTH = 5.5  # inches, the report's text width


def scores(name: str) -> dict[int, np.ndarray]:
    """seed -> evaluation scores for blocks 0..30. 'a+b' gives the per-seed mean of two files."""
    if "+" in name:
        parts = [scores(n) for n in name.split("+")]
        return {s: np.mean([p[s] for p in parts], axis=0) for s in parts[0]}
    by_seed = defaultdict(dict)
    with open(RESULTS / f"{name}.csv") as f:
        for row in csv.DictReader(f):
            by_seed[int(row["seed"])][int(row["block"])] = float(row["eval_score"])
    return {s: np.array([b[k] for k in sorted(b)]) for s, b in by_seed.items()}


def column(name: str, key: str) -> dict[int, np.ndarray]:
    by_seed = defaultdict(dict)
    with open(RESULTS / f"{name}.csv") as f:
        for row in csv.DictReader(f):
            by_seed[int(row["seed"])][int(row["block"])] = float(row[key])
    return {s: np.array([b[k] for k in sorted(b)]) for s, b in by_seed.items()}


def auc(name: str, seeds=None) -> dict[int, float]:
    s = scores(name)
    return {k: v[1:31].mean() / 500 for k, v in s.items() if seeds is None or k in seeds}


def paired_diff(base: str, other: str):
    a, b = auc(base), auc(other)
    d = np.array([b[s] - a[s] for s in sorted(a)])
    lo, hi = bootstrap_ci(d)
    return d.mean(), lo, hi


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_curves() -> None:
    """Main result: mean evaluation score per block on TEST, with 95% bootstrap bands."""
    curves = [
        ("Fixed settings", "p2_noop", GREY, "-"),
        ("Random on B's lr levels", "p2_random_lronly", YELLOW, "-"),
        ("Score-gated lr", "p2_lr_gate", BLUE, "-"),
        ("Score-gated lr + period 400 (selected on VAL)", "p2_lr_gate_hp400", AQUA, "--"),
        ("Learned controller B (mean of 2 B seeds)", "p2_e3_lronly_s0+p2_e4_lronly_s1", ORANGE, "-"),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    blocks = np.arange(31)
    for label, name, colour, style in curves:
        m = np.stack(list(scores(name).values()))
        mean = m.mean(axis=0)
        band = np.array([bootstrap_ci(m[:, k]) for k in blocks])
        ax.fill_between(blocks, band[:, 0], band[:, 1], color=colour, alpha=0.15, linewidth=0)
        ax.plot(blocks, mean, color=colour, linestyle=style, label=label)
    ax.axhline(475, color="#8a8984", linewidth=0.8, linestyle=":")
    ax.text(11.5, 480, "475 (solved)", fontsize=7, color="#52514e", va="bottom")
    ax.set_xlabel("Block (1,000 training steps of A each)")
    ax.set_ylabel("Greedy evaluation score of A")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 510)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.set_title("TEST seeds 0–29 (n = 30 runs per curve); bands are 95% bootstrap CIs of the mean", loc="left")
    save(fig, "curves_test")


def fig_lr_schedule() -> None:
    """Share of runs at each lr level per block: score-gated lr vs the two final B seeds (TEST)."""
    panels = [("Score-gated lr", "p2_lr_gate"), ("B, seed 0", "p2_e3_lronly_s0"), ("B, seed 1", "p2_e4_lronly_s1")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), sharey=True)
    blocks = np.arange(1, 31)
    for ax, (title, name) in zip(axes, panels):
        lr = np.stack(list(column(name, "lr").values()))[:, 1:31]
        bottom = np.zeros(30)
        for level, colour in LR_SHADES.items():
            share = np.isclose(lr, level, rtol=1e-6).mean(axis=0)
            ax.bar(blocks, share, bottom=bottom, width=1.0, color=colour, edgecolor="white", linewidth=0.6,
                   label=f"lr {level:.0e}".replace("e-0", "e-"))
            bottom += share
        ax.set_title(title, loc="left")
        ax.set_xlim(0.5, 30.5)
        ax.set_xlabel("Block")
        ax.grid(False)
    axes[0].set_ylabel("Share of the 30 runs")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    save(fig, "lr_schedule_test")


def fig_diagnosis() -> None:
    """(a) B's training loss per round, 1-step vs 10-step. (b) Offline refits: held-out block-0 max-Q."""
    def losses(stem):
        pat = re.compile(r"round (\d+)/\d+: .*?loss ([\d.]+)")
        out = {}
        for line in (RESULTS / f"{stem}.stdout.txt").read_text().splitlines():
            m = pat.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
        return out

    refits = defaultdict(list)
    text = (ROOT / "scripts/brief/offline_refit_rerun_2026-11-23.out.txt").read_text()
    for m in re.finditer(r"^(.+?)\s+s\d: .*?qmax@0/10/20/29\s+([\d.]+)/", text, re.M):
        refits[m.group(1).strip()].append(float(m.group(2)))
    names = {
        "1-step (B as is)": "1-step (as first B)", "1-step, target period 1000": "1-step, period 1000",
        "1-step, lr 1e-4": "1-step, lr 1e-4", "1-step, gamma 0.9": "1-step, γ 0.9*",
        "3-step": "3-step", "5-step": "5-step", "10-step": "10-step", "30-step (= MC return)": "30-step (Monte Carlo)",
    }

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.75})
    for stem, label, colour in [("outer_ddqn_s0", "first B (1-step)", ORANGE),
                                ("outer_ddqn_n10_lv_s0", "10-step B", BLUE)]:
        loss = losses(stem)
        a.plot(list(loss), list(loss.values()), color=colour, label=label)
    a.set_yscale("log")
    a.set_xlabel("Training round of B")
    a.set_ylabel("B's training loss")
    a.set_title("(a) B's loss during training", loc="left", pad=22)
    a.legend(frameon=False)

    keys = list(names)
    means = [np.mean(refits[k]) for k in keys]
    colours = [ORANGE if k.startswith("1-step") else BLUE for k in keys]
    y = np.arange(len(keys))[::-1]
    b.barh(y, means, color=colours, height=0.7)
    for i, k in zip(y, keys):
        b.scatter(refits[k], [i] * len(refits[k]), s=6, color="#0b0b0b", zorder=3)
    b.set_xscale("log")
    b.axvline(26.03, color="#0b0b0b", linewidth=0.8, linestyle="--")
    b.axvline(13.4, color="#0b0b0b", linewidth=0.8, linestyle=":")
    b.text(26.03, len(keys) - 0.35, "max\npossible\n26.03", ha="left", va="bottom", fontsize=6.3)
    b.text(13.4, len(keys) - 0.35, "actual\n13.4", ha="right", va="bottom", fontsize=6.3)
    b.set_yticks(y, [names[k] for k in keys])
    b.set_xlabel("Held-out max-Q at block 0 (log)")
    b.set_title("(b) Offline refits, 3 seeds each", loc="left", pad=22)
    b.grid(axis="y", visible=False)
    save(fig, "diagnosis")


def fig_period_sweep() -> None:
    """A's AUC against the target-copy period (seeds 0-9), buffer 10k and 50k."""
    seeds = set(range(10))
    small = {10: "diag_tgt10", 25: "diag_tgt25", 50: "diag_tgt50", 100: "diag_tgt100",
             200: "diag_tgt200", 500: "pilot_noop", 2000: "diag_tgt2000"}
    large = {100: "diag_buf50k_tgt100", 500: "diag_buf50k", 2000: "diag_buf50k_tgt2000"}
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.3))
    for runs, label, colour, marker, shift in [(small, "buffer 10,000 (used)", BLUE, "o", 1.0),
                                               (large, "buffer 50,000", GREY, "s", 1.08)]:
        xs, ys, errs = [], [], []
        for period, name in runs.items():
            v = np.array(list(auc(name, seeds).values()))
            lo, hi = bootstrap_ci(v)
            xs.append(period * shift); ys.append(v.mean()); errs.append((v.mean() - lo, hi - v.mean()))
        ax.errorbar(xs, ys, yerr=np.array(errs).T, color=colour, marker=marker, markersize=4,
                    linewidth=1.2, capsize=2, label=label)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 200, 500, 2000], ["10", "25", "50", "100", "200", "500", "2000"])
    ax.set_xlabel("Target-copy period of A (updates)")
    ax.set_ylabel("AUC (seeds 0–9)")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "period_sweep")


def fig_forest() -> None:
    """VAL: every B variant and the main hand rules, minus the score-gated lr rule."""
    rows = [
        ("B variants", None),
        ("lr multipliers × ε levels (2 seeds)", "val_ddqn_n10_lv_s0+val_ddqn_n10_lv_s1"),
        ("lr levels × ε levels", "val_e1_lrlv_s0"),
        ("\u2003– same, 72 rounds", "val_e2_lrlv_r72_s0"),
        ("\u2003– same, dueling", "val_e3_duel_s0"),
        ("lr levels only = final B (2 seeds)", "val_e3_lronly_s0+val_e4_lronly_s1"),
        ("\u2003– same, dueling", "val_e4_lronly_duel_s0"),
        ("\u2003– same, ensemble of 2 networks", "val_ens2_lronly"),
        ("\u2003– same, ensemble of 10 checkpoints", "val_snap10_lronly"),
        ("lr levels × period {100, 400} (2 seeds)", "val_e6_period_s0+val_e6_period_s1"),
        ("Hand rules", None),
        ("Fixed settings", "confirm_noop"),
        ("Threshold 300", "val_lr_gate_t300"),
        ("Threshold 450", "val_lr_gate_t450"),
        ("Cut factor 0.03", "val_lr_gate_f003"),
        ("High lr 2e-3 early", "val_hybrid2e3"),
        ("Period 50 until first 400", "val_period50_sticky"),
        ("Period 400 while score ≥ 400", "val_lr_gate_hp400"),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    y = len(rows)
    ticks, labels = [], []
    colour = ORANGE
    for label, name in rows:
        y -= 1
        ticks.append(y)
        labels.append(label)
        if name is None:
            colour = ORANGE if label == "B variants" else BLUE
            continue
        mean, lo, hi = paired_diff("val_lr_gate", name)
        ax.errorbar([mean], [y], xerr=[[mean - lo], [hi - mean]], fmt="o", color=colour, markersize=4,
                    capsize=2, linewidth=1.2)
    ax.axvline(0, color="#0b0b0b", linewidth=0.8)
    ax.set_yticks(ticks, labels)
    for tick, (label, name) in zip(ax.get_yticklabels(), rows):
        if name is None:
            tick.set_fontweight("bold")
    ax.set_xlabel("AUC difference to the score-gated lr rule, VAL seeds 30–59 (mean, 95% paired bootstrap CI)")
    ax.grid(axis="y", visible=False)
    save(fig, "forest_val")


def fig_b_training() -> None:
    """How quickly B learns: mean AUC of the 11 runs of A in each training round, final B's two seeds.

    Training rounds use fresh, random inner seeds and B's epsilon-greedy exploration, so they are not
    comparable with greedy TEST scores; the reference lines are fixed settings and the score-gated lr
    rule on training-only seeds 1000-1029."""
    seeds = {}
    eps = {}
    for name in ("e3_lronly_s0", "e4_lronly_s1"):
        by_round = defaultdict(list)
        with open(RESULTS / "outer" / name / "train_log.csv") as f:
            for row in csv.DictReader(f):
                by_round[int(row["round"])].append(float(row["auc"]))
                eps[int(row["round"])] = float(row["epsilon_start"])
        seeds[name] = by_round
    rounds = np.array(sorted(eps))
    x = rounds + 1
    pooled = [np.array(seeds["e3_lronly_s0"][r] + seeds["e4_lronly_s1"][r]) for r in rounds]
    mean = np.array([p.mean() for p in pooled])
    band = np.array([bootstrap_ci(p) for p in pooled])
    floor_round = next(r for r in rounds if eps[r] <= min(eps.values()) + 1e-9) + 1

    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.axvspan(0.5, floor_round - 0.5, color="#f3f2ee", linewidth=0)
    ax.text(1, 0.03, f"B's exploration ε decays from 1 to {min(eps.values()):.2f}", fontsize=7, color=GREY)
    ax.text(floor_round, 0.03, f"ε = {min(eps.values()):.2f}", fontsize=7, color=GREY)
    for ref, name, colour in [("Fixed settings", "trainseeds_noop", GREY),
                              ("Score-gated lr", "trainseeds_lr_gate", BLUE)]:
        value = np.mean(list(auc(name).values()))
        ax.axhline(value, color=colour, linewidth=1.0, linestyle="--")
        ax.text(36.6, value, f"{ref}\n{value:.3f}", fontsize=7, color=GREY, va="center")
    for name, style in [("e3_lronly_s0", (0, (1, 1.5))), ("e4_lronly_s1", (0, (3, 1.5)))]:
        ax.plot(x, [np.mean(seeds[name][r]) for r in rounds], color=ORANGE, linewidth=0.8, alpha=0.55,
                linestyle=style)
    ax.fill_between(x, band[:, 0], band[:, 1], color=ORANGE, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=ORANGE, label="B, mean of 2 B seeds (22 runs per round)")
    ax.plot([], [], color=ORANGE, linewidth=0.8, alpha=0.55, linestyle=(0, (1, 1.5)), label="B seed 0")
    ax.plot([], [], color=ORANGE, linewidth=0.8, alpha=0.55, linestyle=(0, (3, 1.5)), label="B seed 1")
    ax.set_xlim(0.5, 36.5)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("B's training round (11 training runs of A each)")
    ax.set_ylabel("AUC of A's runs in the round")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    ax.set_title("B during training; band = 95% bootstrap CI of the round mean", loc="left")
    save(fig, "b_training")


if __name__ == "__main__":
    fig_curves()
    fig_lr_schedule()
    fig_diagnosis()
    fig_period_sweep()
    fig_forest()
    fig_b_training()
