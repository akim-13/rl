"""Headroom probe (not part of dac/): lr HIGH from block 6 until the run first scores >= 400, else lr_gate.
Usage: hybrid.py HIGH NAME  -> results/NAME.csv (seed, block, lr, eval_score), seeds 30..59."""
import csv, sys
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from dac.controllers import run_episode
from dac.inner import InnerConfig

HIGH = float(sys.argv[1]); NAME = sys.argv[2]; CONFIG = InnerConfig()

def hybrid(run):
    next_block = len(run.history)            # the block about to be trained
    best_so_far = max(result.eval_score for result in run.history)
    last = run.history[-1].eval_score
    if next_block >= 6 and best_so_far < 400:
        target = HIGH
    else:
        target = CONFIG.lr_init * 0.1 if last >= 400 else CONFIG.lr_init
    return target / run.lr, 1.0

def one(seed):
    return seed, run_episode(seed, hybrid, CONFIG)

if __name__ == "__main__":
    with Pool(12) as pool, open(__import__("pathlib").Path(__file__).resolve().parents[2] / "results" / f"{NAME}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "block", "lr", "eval_score"])
        for seed, history in pool.imap_unordered(one, range(30, 60)):
            for r in history: w.writerow([seed, r.block, r.lr, r.eval_score])
    print("wrote", NAME)
