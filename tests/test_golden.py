"""The program must reproduce its recorded outputs exactly (see tests/golden_cases.py)."""

import dataclasses
import json

import pytest
import torch

from dac.inner import InnerConfig
from dac.outer_dqn import OuterConfig

from golden_cases import (
    BATCH_CASES,
    DDQN_BATCH_ARGS,
    FULL_SIZE_CASES,
    GOLDEN_DIR,
    JSON_KEYS_THAT_MUST_MATCH,
    OUTER_TRAINING_NAME,
    REPORT_CASES,
    REPORT_PLOT_NAME,
    make_report,
    read_csv_without_timings,
    run_full_size_seed0,
    run_small_batch,
    train_small_outer_agent,
)


def assert_result_pair_matches_golden(results_dir, name, golden_name=None):
    golden_name = golden_name or name
    produced_rows = read_csv_without_timings(results_dir / f"{name}.csv")
    expected_rows = read_csv_without_timings(GOLDEN_DIR / f"{golden_name}.csv")
    assert produced_rows == expected_rows

    produced_meta = json.loads((results_dir / f"{name}.json").read_text())
    expected_meta = json.loads((GOLDEN_DIR / f"{golden_name}.json").read_text())
    for key in JSON_KEYS_THAT_MUST_MATCH:
        if key == "config":
            continue
        assert produced_meta[key] == expected_meta[key], key
    assert_config_matches_golden(produced_meta["config"], expected_meta["config"], InnerConfig)


def assert_config_matches_golden(produced_config, expected_config, config_class):
    """Every recorded key keeps its recorded value; a key added since must hold its default.

    New options are added with defaults that keep the old behaviour, so a golden
    file written before an option existed still describes the same run.
    """
    for key, expected_value in expected_config.items():
        assert produced_config[key] == expected_value, f"{config_class.__name__}[{key!r}]"
    defaults = dataclasses.asdict(config_class())
    for key in produced_config.keys() - expected_config.keys():
        default = defaults[key]
        default = list(default) if isinstance(default, tuple) else default  # JSON and torch.save store lists
        assert produced_config[key] == default, f"new {config_class.__name__}[{key!r}] is not its default"


def assert_outer_config_matches_golden(produced_config, expected_config):
    assert_config_matches_golden(produced_config, expected_config, OuterConfig)


@pytest.mark.parametrize("name", sorted(BATCH_CASES))
def test_small_batch_matches_golden(name, tmp_path):
    run_small_batch(name, BATCH_CASES[name], tmp_path)
    assert_result_pair_matches_golden(tmp_path / "results", name)


def test_outer_training_and_its_policy_match_golden(tmp_path):
    train_small_outer_agent(tmp_path)
    outer_dir = tmp_path / "results" / "outer" / OUTER_TRAINING_NAME

    produced = torch.load(outer_dir / "model.pt", weights_only=True)
    expected = torch.load(GOLDEN_DIR / "outer_model.pt", weights_only=True)
    assert produced.keys() == expected.keys()
    assert produced["state_dict"].keys() == expected["state_dict"].keys()
    for parameter_name, expected_tensor in expected["state_dict"].items():
        assert torch.equal(produced["state_dict"][parameter_name], expected_tensor), parameter_name
    assert_outer_config_matches_golden(produced["outer_config"], expected["outer_config"])
    assert_config_matches_golden(produced["inner_config"], expected["inner_config"], InnerConfig)
    for key in expected.keys() - {"state_dict", "outer_config", "inner_config"}:
        assert produced[key] == expected[key], key

    assert read_csv_without_timings(outer_dir / "train_log.csv") == read_csv_without_timings(
        GOLDEN_DIR / "outer_train_log.csv"
    )
    produced_meta = json.loads((outer_dir / "train_meta.json").read_text())
    expected_meta = json.loads((GOLDEN_DIR / "outer_train_meta.json").read_text())
    assert_outer_config_matches_golden(produced_meta["outer_config"], expected_meta["outer_config"])
    assert_config_matches_golden(produced_meta["inner_config"], expected_meta["inner_config"], InnerConfig)
    for key in expected_meta.keys() - {"wall_seconds", "versions", "outer_config", "inner_config"}:
        assert produced_meta[key] == expected_meta[key], key

    run_small_batch("ddqn", DDQN_BATCH_ARGS, tmp_path)
    assert_result_pair_matches_golden(tmp_path / "results", "ddqn")


@pytest.mark.parametrize("report_name", sorted(REPORT_CASES))
def test_report_matches_golden(report_name, tmp_path):
    report_inputs = [GOLDEN_DIR / f"{name}.csv" for name in REPORT_CASES[report_name]]
    printed = make_report(report_inputs, tmp_path)
    assert printed == (GOLDEN_DIR / f"{report_name}.txt").read_text()
    assert (tmp_path / REPORT_PLOT_NAME).exists()


@pytest.mark.slow
@pytest.mark.parametrize("golden_name", sorted(FULL_SIZE_CASES))
def test_full_size_run_matches_pilot_results(golden_name, tmp_path):
    run_full_size_seed0(golden_name, FULL_SIZE_CASES[golden_name], tmp_path)
    assert_result_pair_matches_golden(tmp_path / "results", golden_name)
