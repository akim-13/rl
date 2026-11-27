"""Headroom probe (not part of dac/): lr by lr_gate; target-copy period SHORT early, 100 later.
Usage: period_rule.py SHORT MODE NAME   MODE = sticky (short until the first score >= 400) | gate (short while last < 400) | q<T (short while A's mean Q < T)
Writes results/NAME.csv (seed, block, lr, eval_score) for seeds 30..59."""
import csv, sys
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from dac.controllers import run_episode
from dac.inner import InnerConfig

SHORT = int(sys.argv[1]); MODE = sys.argv[2]; NAME = sys.argv[3]; CONFIG = InnerConfig()

def rule(run):
    last = run.history[-1].eval_score
    best = max(result.eval_score for result in run.history)
    if MODE.startswith("q<"):
        q_mean = run.history[-1].q_mean
        early = q_mean != q_mean or q_mean < float(MODE[2:])   # nan at block 0 counts as small
    else:
        early = best < 400 if MODE == "sticky" else last < 400
    run.agent.target_update_every = SHORT if early else CONFIG.target_update_every
    target_lr = CONFIG.lr_init * 0.1 if last >= 400 else CONFIG.lr_init
    return target_lr / run.lr, 1.0

def one(seed):
    return seed, run_episode(seed, rule, CONFIG)

if __name__ == "__main__":
    assert MODE in ("sticky", "gate") or MODE.startswith("q<")
    with Pool(12) as pool, open(__import__("pathlib").Path(__file__).resolve().parents[2] / "results" / f"{NAME}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "block", "lr", "eval_score", "q_mean"])
        for seed, history in pool.imap_unordered(one, range(30, 60)):
            for r in history: w.writerow([seed, r.block, r.lr, r.eval_score, r.q_mean])
    print("wrote", NAME)
