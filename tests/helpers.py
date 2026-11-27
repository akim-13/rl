"""Shared helpers for the tests (imported as `helpers`; pytest puts tests/ on sys.path)."""

import dataclasses
import math

from dac.inner import BlockResult

TIMING_FIELDS = {"train_seconds", "eval_seconds"}


def _both_nan(first_value, second_value) -> bool:
    both_floats = isinstance(first_value, float) and isinstance(second_value, float)
    return both_floats and math.isnan(first_value) and math.isnan(second_value)


def results_match(first: BlockResult, second: BlockResult) -> bool:
    """Field-by-field equality, ignoring wall-clock times and treating nan == nan."""
    for field in dataclasses.fields(BlockResult):
        if field.name in TIMING_FIELDS:
            continue
        first_value, second_value = getattr(first, field.name), getattr(second, field.name)
        if _both_nan(first_value, second_value):
            continue
        if first_value != second_value:
            return False
    return True


def histories_match(first: list[BlockResult], second: list[BlockResult]) -> bool:
    same_length = len(first) == len(second)
    return same_length and all(
        results_match(first_result, second_result) for first_result, second_result in zip(first, second)
    )
