"""Headroom probe (not part of dac/): lr by lr_gate; A's target-copy period LONG once the score is high, else 100.
Usage: late_period.py LONG MODE NAME   MODE = gate (LONG while the last score >= 400) | sticky (LONG from the first score >= 400)
Writes results/NAME.csv (seed, block, lr, eval_score, q_mean) for validation seeds 30..59."""
import csv, sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dac.controllers import run_episode
from dac.inner import InnerConfig

LONG = int(sys.argv[1]); MODE = sys.argv[2]; NAME = sys.argv[3]; CONFIG = InnerConfig()


def rule(run):
    last = run.history[-1].eval_score
    best = max(result.eval_score for result in run.history)
    high = last >= 400 if MODE == "gate" else best >= 400
    run.agent.target_update_every = LONG if high else CONFIG.target_update_every
    target_lr = CONFIG.lr_init * 0.1 if last >= 400 else CONFIG.lr_init
    return target_lr / run.lr, 1.0


def one(seed):
    return seed, run_episode(seed, rule, CONFIG)


if __name__ == "__main__":
    assert MODE in ("gate", "sticky")
    with Pool(12) as pool, open(REPO / "results" / f"{NAME}.csv", "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["seed", "block", "lr", "eval_score", "q_mean"])
        for seed, history in pool.imap_unordered(one, range(30, 60)):
            for r in history: writer.writerow([seed, r.block, r.lr, r.eval_score, r.q_mean])
    print("wrote", NAME)
