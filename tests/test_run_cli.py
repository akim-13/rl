"""The argument parsing helpers of dac.run (the batch itself is exercised by hand)."""

import argparse
import csv

import pytest

from dac.run import config_with_overrides, parse_controller_args, parse_seed_range

from golden_cases import SMALL_CONFIG_FLAGS, run_module


def test_parse_seed_range():
    assert parse_seed_range("0:4") == [0, 1, 2, 3]
    assert parse_seed_range("10:12") == [10, 11]
    for bad in ("4:4", "5", "a:b"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_seed_range(bad)


def test_config_overrides_use_the_field_types():
    config = config_with_overrides(
        ["block_steps=500", "lr_init=0.002", "double_dqn=false", "hidden=32,16", "env_id=CartPole-v0"]
    )
    assert config.block_steps == 500 and isinstance(config.block_steps, int)
    assert config.lr_init == 0.002
    assert config.double_dqn is False
    assert config.hidden == (32, 16)
    assert config.env_id == "CartPole-v0"


def test_config_overrides_reject_unknown_or_malformed():
    with pytest.raises(ValueError):
        config_with_overrides(["no_such_field=1"])
    with pytest.raises(ValueError):
        config_with_overrides(["block_steps"])
    with pytest.raises(ValueError):
        config_with_overrides(["block_steps=1.5"])


def test_parse_controller_args():
    kwargs = parse_controller_args(["decay_blocks=15", "eps_end=0.02", "sticky=true", "lr_mults=0.5,1,2"])
    assert kwargs == {"decay_blocks": 15, "eps_end": 0.02, "sticky": True, "lr_mults": (0.5, 1, 2)}
    assert isinstance(kwargs["decay_blocks"], int)


def test_random_controller_with_epsilon_levels_from_the_command_line(tmp_path):
    run_module(
        "dac.run",
        ["--controller", "random", "--controller-arg", "epsilon_levels=0.03,0.1,0.3",
         *SMALL_CONFIG_FLAGS, "--seeds", "0:1", "--name", "levels"],
        tmp_path,
    )
    with open(tmp_path / "results" / "levels.csv", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["block"] != "0"]
    assert rows
    for row in rows:
        assert any(float(row["epsilon"]) == pytest.approx(level, rel=1e-12) for level in (0.03, 0.1, 0.3))
