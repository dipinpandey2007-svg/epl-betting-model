"""Hard write guard for locked results (eplmodel.reporting.locks), applied to the Experiment 16 historical stage.

No test runs any scoring: every attempted write is refused before data are loaded, which a sentinel proves.
"""

import json

import pytest

from eplmodel.config import load_config
from eplmodel.data.checksums import content_sha256
from eplmodel.paths import RESULTS_DIR
from eplmodel.reporting.locks import LockedResultsError, refuse_locked_overwrite
from eplmodel.reporting.results import write_results
from experiments import shots_information as exp

CFG = load_config(exp.SHOTS_CONFIG)
LOCKED_DIR = RESULTS_DIR / CFG["outputs"]["results_name"]


class Sentinel(Exception):
    """Raised by a patched step that would come after the guard."""


def _fingerprint():
    return {name: ((LOCKED_DIR / name).read_bytes(), (LOCKED_DIR / name).stat().st_mtime_ns)
            for name in ("metrics.json", "predictions.csv") if (LOCKED_DIR / name).exists()}


@pytest.fixture
def no_data_after_guard(monkeypatch):
    """Anything past the first guard raises Sentinel, so no data are loaded and nothing is scored or written."""
    def stop(*_args, **_kwargs):
        raise Sentinel()
    monkeypatch.setattr(exp, "check_frozen", stop)
    monkeypatch.setattr(exp, "load_data", stop)
    monkeypatch.setattr(exp, "write_predictions", stop)
    monkeypatch.setattr(exp, "write_results", stop)


def test_locked_historical_path_refuses_a_write_before_anything_runs(no_data_after_guard):
    assert CFG["locked"]["status"] == "locked"
    before = _fingerprint()
    with pytest.raises(LockedResultsError, match="locked"):
        exp.run(write=True)
    assert _fingerprint() == before


def test_locked_artifacts_are_byte_identical_to_the_lock():
    lock = CFG["locked"]
    assert content_sha256(LOCKED_DIR / "metrics.json") == lock["historical_metrics_sha256"]
    if (LOCKED_DIR / "predictions.csv").exists():                 # git-ignored; absent in CI
        assert content_sha256(LOCKED_DIR / "predictions.csv") == lock["historical_predictions_sha256"]


def test_read_only_rerun_is_not_blocked_by_the_guard(no_data_after_guard):
    """write=False (the golden reproduction path) passes the guard and reaches the next step."""
    with pytest.raises(Sentinel):
        exp.run(write=False)


def test_existing_result_files_are_protected_even_without_a_lock(tmp_path):
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LockedResultsError, match="already exist"):
        refuse_locked_overwrite(tmp_path, {"status": "not_run"})
    assert (tmp_path / "metrics.json").read_text(encoding="utf-8") == "{}"


def test_a_fresh_unlocked_output_path_remains_usable(tmp_path):
    fresh = tmp_path / "results"
    refuse_locked_overwrite(fresh / "new_protocol_stage", {"status": "not_run"})       # no exception
    path = write_results("new_protocol_stage", {"x": 1}, results_dir=fresh, data_path=exp.PROCESSED_DEV_V2) \
        if exp.PROCESSED_DEV_V2.exists() else None
    if path is not None:
        assert json.loads(path.read_text(encoding="utf-8"))["results"] == {"x": 1}


def test_guard_does_not_change_the_locked_interpretation():
    payload = json.loads((LOCKED_DIR / "metrics.json").read_text(encoding="utf-8"))
    r, lock = payload["results"], CFG["locked"]
    assert r["selection"]["b1"]["omega_selected"] == lock["omega_star_b1"] == 0.0
    assert r["selection"]["s1"]["omega_selected"] == lock["omega_star_s1"] == 0.0
    assert r["criterion_s"]["b1"]["reading"] == lock["reading_b1"] == "no_distinguishable_shot_information"
    assert r["criterion_s"]["s1"]["reading"] == lock["reading_s1"] == "no_distinguishable_shot_information"
    assert r["reading_primary"] == "no_distinguishable_shot_information"
