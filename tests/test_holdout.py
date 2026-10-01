"""The holdout access mechanism (eplmodel.holdout) and the separation of development code from it.

Everything here uses synthetic files in tmp_path: no test reads real holdout data.
"""

import json
import re

import pandas as pd
import pytest

from eplmodel import holdout
from eplmodel.data.checksums import content_sha256
from eplmodel.holdout import HoldoutAccess, load_holdout_config, load_holdout_matches, open_final_holdout
from eplmodel.paths import PROJECT_ROOT
from eplmodel.splits import FINAL_HOLDOUT_SEASONS, NEXT_HOLDOUT_SEASONS, HoldoutAccessError

SPECS = ["spec_a_v1", "spec_b_v1"]

CONFIG = """
[holdout]
protocol_id = "holdout_v1"
seasons = ["2526"]
next_holdout_seasons = ["2627"]
freeze_tag = "holdout-freeze-v1"
freeze_date = "2026-10-01"

[metrics]
primary = "log_loss"
secondary = ["brier", "calibration"]

[registration]
spec_ids = ["spec_a_v1", "spec_b_v1"]

[[access]]
entry_id = "H9"
spec_ids = ["spec_a_v1", "spec_b_v1"]
"""

LOG = """
| # | When | What |
|---|---|---|
| H9 | 2027-01-01 | final evaluation |
"""


def _synthetic_season(season: str = "2526") -> pd.DataFrame:
    teams = [f"T{i:02d}" for i in range(20)]
    fixtures = [(h, a) for h in teams for a in teams if h != a]  # 380 matches
    return pd.DataFrame({
        "Date": pd.date_range("2025-08-01", periods=len(fixtures)).strftime("%Y-%m-%d"),
        "HomeTeam": [h for h, _ in fixtures], "AwayTeam": [a for _, a in fixtures],
        "FTHG": 1.0, "FTAG": 0.0, "FTR": "H", "Season": season,
    })


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    """A synthetic project in which every access condition holds."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "holdout_v1.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "log.md").write_text(LOG, encoding="utf-8")
    rel = "data/holdout/processed/matches_2526.csv"
    (tmp_path / rel).parent.mkdir(parents=True)
    _synthetic_season().to_csv(tmp_path / rel, index=False)
    manifest = {"protocol_id": "holdout_v1", "seasons": ["2526"], "processed_file": rel,
                "files": {rel: content_sha256(tmp_path / rel)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(holdout, "working_tree_clean", lambda root: True)
    monkeypatch.setattr(holdout, "freeze_tag_is_ancestor", lambda tag, root: True)
    monkeypatch.setattr(holdout, "head_commit", lambda root: "0" * 40)
    return {"config_path": tmp_path / "configs" / "holdout_v1.toml", "log_path": tmp_path / "log.md",
            "manifest_path": tmp_path / "manifest.json", "root": tmp_path, "processed": tmp_path / rel}


def _open(sealed, entry_id="H9", spec_ids=SPECS):
    kwargs = {k: v for k, v in sealed.items() if k != "processed"}
    return open_final_holdout(entry_id, spec_ids, **kwargs)


def test_access_succeeds_when_every_condition_holds(sealed):
    access = _open(sealed)
    df = load_holdout_matches(access)
    assert len(df) == 380 and set(df["Season"]) == {"2526"}
    assert access.entry_id == "H9" and access.spec_ids == tuple(SPECS)


@pytest.mark.parametrize("entry_id,spec_ids,message", [
    ("H9", [], "No spec ids"),
    ("H9", ["spec_a_v1", "unregistered_v1"], "not pre-registered"),
    ("H8", SPECS, "not authorised"),
    ("H9", ["spec_a_v1"], "does not authorise exactly"),
])
def test_access_refuses_unregistered_or_unauthorised_requests(sealed, entry_id, spec_ids, message):
    with pytest.raises(HoldoutAccessError, match=message):
        _open(sealed, entry_id, spec_ids)


def test_access_refuses_entry_missing_from_the_log(sealed):
    sealed["log_path"].write_text("| H1 | sealed |\n", encoding="utf-8")
    with pytest.raises(HoldoutAccessError, match="not recorded"):
        _open(sealed)


def test_access_refuses_uncommitted_changes(sealed, monkeypatch):
    monkeypatch.setattr(holdout, "working_tree_clean", lambda root: False)
    with pytest.raises(HoldoutAccessError, match="uncommitted"):
        _open(sealed)


def test_access_refuses_commits_not_descending_from_the_freeze_tag(sealed, monkeypatch):
    monkeypatch.setattr(holdout, "freeze_tag_is_ancestor", lambda tag, root: False)
    with pytest.raises(HoldoutAccessError, match="freeze tag"):
        _open(sealed)


def test_access_refuses_unsealed_data(sealed):
    sealed["manifest_path"].unlink()
    with pytest.raises(HoldoutAccessError, match="not been sealed"):
        _open(sealed)


def test_access_refuses_modified_holdout_file(sealed):
    df = _synthetic_season()
    df.loc[0, ["FTHG", "FTR"]] = [0.0, "D"]
    df.loc[0, "FTAG"] = 0.0
    df.to_csv(sealed["processed"], index=False)
    with pytest.raises(HoldoutAccessError, match="checksum"):
        _open(sealed)


def test_holdout_file_changed_after_opening_is_refused(sealed):
    access = _open(sealed)
    sealed["processed"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(HoldoutAccessError, match="changed after access"):
        load_holdout_matches(access)


def test_access_objects_cannot_be_forged(sealed):
    with pytest.raises(HoldoutAccessError):
        HoldoutAccess("H9", tuple(SPECS), "0" * 40, sealed["processed"], "x")
    with pytest.raises(HoldoutAccessError):
        load_holdout_matches(object())


def test_frozen_config_agrees_with_season_roles_and_allows_no_access_yet():
    cfg = load_holdout_config()
    assert tuple(cfg["holdout"]["seasons"]) == FINAL_HOLDOUT_SEASONS
    assert tuple(cfg["holdout"]["next_holdout_seasons"]) == NEXT_HOLDOUT_SEASONS
    assert cfg["holdout"]["freeze_tag"] == "holdout-freeze-v1"
    assert cfg["metrics"]["primary"] == "log_loss"
    # No spec is registered yet, so the real holdout cannot be opened.
    assert cfg["registration"]["spec_ids"] == []
    assert cfg.get("access", []) == []
    with pytest.raises(HoldoutAccessError):
        open_final_holdout("H0", ["elo_k25_logreg_v1"])


def test_development_code_never_touches_the_holdout_module():
    pattern = re.compile(r"eplmodel\.holdout|from eplmodel import holdout|load_holdout_matches|open_final_holdout|_read_matches")
    offenders = [p.relative_to(PROJECT_ROOT).as_posix()
                 for p in sorted((PROJECT_ROOT / "experiments").rglob("*.py"))
                 if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
