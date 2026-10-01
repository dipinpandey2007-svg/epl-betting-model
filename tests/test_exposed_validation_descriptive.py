"""Logged descriptive access to the exposed validation season (access-log entry V5).

No test here reads any 2024-25 data: the guard is exercised on synthetic tables.
"""

import json

import pandas as pd
import pytest

from eplmodel.config import load_config
from eplmodel.evaluation import folds
from eplmodel.paths import TEST_SET_ACCESS_LOG
from eplmodel.splits import (
    DEVELOPMENT_SEASONS,
    LOGGED_EXPOSED_VALIDATION_ACCESSES,
    REGISTERED_EXPOSED_VALIDATION_SPECS,
    ExposedValidationError,
    HoldoutAccessError,
    require_logged_exposed_validation_access,
)
from experiments import exposed_validation_descriptive as evd

SCFG = load_config(evd.STAGE_CONFIG)
V5 = sorted(LOGGED_EXPOSED_VALIDATION_ACCESSES["V5"])


def _table(seasons):
    return pd.DataFrame({"Season": list(seasons), "Date": pd.Timestamp("2020-01-01"),
                         "HomeTeam": "A", "AwayTeam": "B", "match_id": [f"m{i}" for i in range(len(seasons))]})


def test_v5_authorises_exactly_the_frozen_arms_and_is_logged():
    assert set(SCFG["access"]["spec_ids"]) == LOGGED_EXPOSED_VALIDATION_ACCESSES["V5"]
    assert len(V5) == 9 and SCFG["protocol"]["access_entry"] == "V5"
    assert "V5" in folds._logged_entry_ids(TEST_SET_ACCESS_LOG)


def test_reproduction_registry_of_the_recorded_protocols_is_unchanged():
    assert len(REGISTERED_EXPOSED_VALIDATION_SPECS) == 8
    assert REGISTERED_EXPOSED_VALIDATION_SPECS.isdisjoint(LOGGED_EXPOSED_VALIDATION_ACCESSES["V5"])


@pytest.mark.parametrize("entry,specs", [("V6", V5), ("V5", V5[:-1]), ("V5", [*V5, "elo_k25_logreg_v1"])])
def test_unlogged_or_mismatched_accesses_are_refused(entry, specs):
    with pytest.raises(ExposedValidationError):
        require_logged_exposed_validation_access(entry, specs)
    with pytest.raises(ExposedValidationError):
        folds.build_exposed_validation_fold(_table(DEVELOPMENT_SEASONS), entry, specs)


def test_entry_must_be_recorded_in_the_access_log(tmp_path):
    log = tmp_path / "log.md"
    log.write_text("| V4 | x |\n", encoding="utf-8")
    with pytest.raises(ExposedValidationError, match="not recorded"):
        folds.build_exposed_validation_fold(_table(DEVELOPMENT_SEASONS), "V5", V5, log_path=log)


def test_logged_access_opens_only_the_exposed_season_with_earlier_history():
    fold = folds.build_exposed_validation_fold(_table(DEVELOPMENT_SEASONS), "V5", V5)
    assert fold.history == DEVELOPMENT_SEASONS[:-1] and fold.target == DEVELOPMENT_SEASONS[-1]
    assert set(fold.target_rows["Season"]) == {fold.target}


@pytest.mark.parametrize("season", ["2526", "2627"])
def test_holdout_rows_are_refused(season):
    with pytest.raises(HoldoutAccessError):
        folds.build_exposed_validation_fold(_table([*DEVELOPMENT_SEASONS, season]), "V5", V5)


def test_strict_selection_guard_still_refuses_the_exposed_season():
    with pytest.raises(ExposedValidationError):
        folds.build_fold(_table(DEVELOPMENT_SEASONS), DEVELOPMENT_SEASONS[:-1], DEVELOPMENT_SEASONS[-1])


def test_stage_config_names_no_season_and_applies_no_criterion():
    values = json.dumps(SCFG)
    for season in ("2425", "2024-25", "2526", "2627"):
        assert season not in values
    assert SCFG["protocol"]["criteria_applied"] is False and SCFG["evaluation"]["ranking_claims"] is False


def test_frozen_inputs_of_experiments_14_and_15_are_verified():
    import subprocess

    from eplmodel.paths import PROJECT_ROOT
    from experiments import full_coverage_poisson as fcp, market_benchmark as mb

    if subprocess.run(["git", "cat-file", "-e", f"{fcp.PREREG_COMMIT}^{{commit}}"], cwd=PROJECT_ROOT,
                      capture_output=True).returncode != 0:
        pytest.skip("pre-registration commit not in this checkout (shallow clone)")
    frozen = evd.check_frozen_inputs(SCFG, load_config(fcp.FULL_COVERAGE_CONFIG), load_config(mb.MARKET_CONFIG))
    assert frozen["experiment_14_metrics_sha256"] == evd.EXP14_METRICS_SHA256


def test_stage_is_not_in_run_all():
    from experiments import run_all
    assert evd not in run_all.EXPERIMENTS


def test_descriptive_results_are_locked_to_the_recorded_run():
    from eplmodel.data.checksums import content_sha256
    from eplmodel.paths import RESULTS_DIR

    lock = SCFG["locked"]
    metrics = RESULTS_DIR / SCFG["outputs"]["results_name"] / "metrics.json"
    assert lock["status"] == "locked" and content_sha256(metrics) == lock["metrics_sha256"]
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["provenance"]["git_commit"] == lock["run_commit"] and payload["provenance"]["git_dirty"] is False
    assert payload["results"]["predictions"]["sha256"] == lock["predictions_sha256"]
    assert payload["results"]["criteria_applied"] is False and payload["results"]["access_entry"] == "V5"


@pytest.mark.golden
def test_recorded_descriptive_validation_reproduces():
    from eplmodel.data.checksums import content_sha256
    from eplmodel.paths import PROCESSED_DEV_V2, RESULTS_DIR
    from eplmodel.reporting.results import _jsonable

    out = RESULTS_DIR / SCFG["outputs"]["results_name"]
    if not PROCESSED_DEV_V2.exists() or not (out / "predictions.csv").exists():
        pytest.skip("dev_v2 or the recorded predictions missing")
    assert content_sha256(out / "predictions.csv") == SCFG["locked"]["predictions_sha256"]
    recorded = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["results"]
    recorded.pop("predictions")
    again = json.loads(json.dumps(_jsonable(evd.run(write=False))))
    # The stage config hash changes only because [locked] was written after the run.
    for result in (recorded, again):
        result["frozen_inputs"].pop("stage_config_sha256")
    assert again == recorded
