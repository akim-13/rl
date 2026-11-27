import pytest

from dac.inner import InnerConfig

# Small enough that a whole run takes about a second.
TINY_CONFIG = InnerConfig(block_steps=200, n_blocks=3, eval_episodes=2, learning_starts=50)


@pytest.fixture
def tiny_config() -> InnerConfig:
    return TINY_CONFIG
