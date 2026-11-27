"""Summary numbers for one run.

`scores` is the 1-D array of eval scores for blocks 1..n_blocks (block 0,
the untrained network, is left out).
"""

import numpy as np


def outer_return(scores: np.ndarray, score_max: float) -> float:
    """B's return for a run: sum over blocks of eval_score / score_max."""
    return float(np.sum(scores) / score_max)


def auc(scores: np.ndarray, score_max: float) -> float:
    """Normalised area under the learning curve, in [0, 1]; rewards both speed and final level."""
    return float(np.mean(scores) / score_max)


def steps_to_threshold(scores: np.ndarray, block_steps: int, threshold: float) -> float:
    """Env steps at the end of the first block scoring >= threshold, inf if none."""
    positions_reaching_threshold = np.flatnonzero(np.asarray(scores) >= threshold)
    if len(positions_reaching_threshold) == 0:
        return float("inf")
    first_block = positions_reaching_threshold[0] + 1  # scores[0] belongs to block 1
    return float(first_block * block_steps)


def blocks_above(scores: np.ndarray, threshold: float) -> int:
    """Number of blocks scoring >= threshold."""
    return int(np.sum(np.asarray(scores) >= threshold))


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval (at `level`) for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    resample_positions = rng.integers(0, len(values), size=(n_resamples, len(values)))
    resampled_means = values[resample_positions].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(resampled_means, [tail, 1.0 - tail])
    return float(low), float(high)
