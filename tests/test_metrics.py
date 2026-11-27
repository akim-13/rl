import math

import numpy as np
import pytest

from dac.metrics import auc, blocks_above, bootstrap_ci, outer_return, steps_to_threshold

SCORES = np.array([100.0, 300.0, 480.0, 500.0])


def test_outer_return_and_auc():
    assert outer_return(SCORES, 500.0) == pytest.approx(1380.0 / 500.0)
    assert auc(SCORES, 500.0) == pytest.approx(345.0 / 500.0)
    assert auc(np.full(5, 500.0), 500.0) == 1.0
    assert auc(np.zeros(5), 500.0) == 0.0


def test_steps_to_threshold():
    assert steps_to_threshold(SCORES, block_steps=1000, threshold=475.0) == 3000.0
    assert steps_to_threshold(SCORES, block_steps=1000, threshold=100.0) == 1000.0
    assert steps_to_threshold(SCORES, block_steps=1000, threshold=500.0) == 4000.0
    assert steps_to_threshold(SCORES, block_steps=1000, threshold=501.0) == math.inf


def test_blocks_above():
    assert blocks_above(SCORES, 475.0) == 2
    assert blocks_above(SCORES, 0.0) == 4
    assert blocks_above(SCORES, 1000.0) == 0


def test_bootstrap_ci_constant_values():
    assert bootstrap_ci(np.full(10, 3.0)) == (3.0, 3.0)


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = np.random.default_rng(0).normal(5.0, 1.0, size=40)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high
    assert (low, high) == bootstrap_ci(values)
    # A wider level gives a wider interval.
    low99, high99 = bootstrap_ci(values, level=0.99)
    assert low99 < low and high < high99


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))
