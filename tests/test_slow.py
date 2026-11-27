import pytest

from dac.inner import InnerConfig, InnerRun


@pytest.mark.slow
def test_default_config_learns_cartpole():
    """With the default config and no hyperparameter changes, seed 0 must clearly
    beat random play (about 20) at some block. Stops at the first such block."""
    run = InnerRun(0, InnerConfig())
    best = run.history[0].eval_score
    while not run.done and best <= 150:
        best = max(best, run.run_block().eval_score)
    run.close()
    assert best > 150, f"best eval score {best} after {run.block} blocks"
