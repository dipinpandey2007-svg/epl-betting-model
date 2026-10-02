"""Pre-registration of shots_information_v1: the frozen config parses and agrees with the locked choices.

No shot model exists yet. Only config and guard checks; the one data test reads raw-file HEADERS only.
"""

import json

import pytest

from eplmodel.config import load_config
from eplmodel.models import shots_spec as ss
from eplmodel.paths import PROJECT_ROOT, RAW_DIR
from eplmodel.splits import SELECTION_TARGET_SEASONS, TRAIN_SEASONS

CFG = load_config(ss.SHOTS_CONFIG)


def test_spec_loads_with_inherited_and_new_choices():
    spec = ss.load_shots_spec()
    assert spec.protocol_id == "shots_information_v1" and spec.status == "registered_not_implemented"
    assert spec.targets == SELECTION_TARGET_SEASONS and spec.max_season == "2122"
    assert spec.half_life_days == 730.0 and spec.max_goals == 10
    assert spec.primary_columns == ("HST", "AST") and spec.sensitivity_columns == ("HS", "AS")
    assert spec.omega_grid == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert spec.arms["B0"] == "poisson_tw_online_h730_v1_diag" and CFG["arms"]["primary"] == "B1"
    assert spec.practical_floor_log_loss == 0.002 and spec.b0_reproduction_tolerance == 1e-12


@pytest.mark.parametrize("section,key,value", [
    ("baseline", "half_life_days", 365.0),                                 # H may not be retuned
    ("protocol", "targets", ["1718", "1819", "1920", "2021", "2122", "2223"]),
    ("protocol", "inputs", ["match_results", "market_odds"]),
    ("protocol", "inputs", ["match_results", "xg"]),
    ("source", "read_columns", ["Date", "HomeTeam", "AwayTeam", "HST", "AST", "FTHG"]),   # no results in shots table
    ("blend", "omega_grid", [0.25, 0.5, 0.75, 1.0]),                       # B0 must stay nested
    ("blend", "level_and_home_from", "shot_model"),
    ("selection", "rule", "argmin"),
    ("arms", "primary", "S1"),
    ("baseline", "unseen_team_rule", "include"),
])
def test_disagreeing_configs_are_refused(section, key, value):
    cfg = json.loads(json.dumps(CFG))
    cfg[section][key] = value
    with pytest.raises(ss.ShotsSpecError):
        ss.validate(cfg)


def test_no_later_season_appears_in_any_config_value():
    values = json.dumps(CFG)
    for season in ("2223", "2324", "2425", "2526", "2627", "2024-25", "2025-26", "2026-27"):
        assert season not in values


def test_protocol_is_not_yet_run():
    assert CFG["locked"]["status"] == "not_run"
    assert not (PROJECT_ROOT / "results" / CFG["outputs"]["results_name"]).exists()


def test_source_record_is_referenced_and_present():
    record = PROJECT_ROOT / CFG["source"]["record"]
    text = record.read_text(encoding="utf-8")
    assert "HST = Home Team Shots on Target" in text and "HS = Home Team Shots" in text
    assert CFG["source"]["notes_sha256"] in text


def test_shot_columns_are_present_in_every_training_season_header():
    """Headers only: no value of any row is read."""
    import pandas as pd

    files = [RAW_DIR / f"E0_{s}.csv" for s in TRAIN_SEASONS]
    if not all(f.exists() for f in files):
        pytest.skip("raw files missing")
    for f in files:
        header = set(pd.read_csv(f, nrows=0).columns)
        assert set(ss.PRIMARY_COLUMNS) | set(ss.SENSITIVITY_COLUMNS) <= header
