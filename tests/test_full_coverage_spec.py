"""Pre-registration of full_coverage_poisson_v1: the config parses and agrees with the locked choices.

No model exists yet; these tests only check the frozen specification. The one real-data test reads the
fixture lists (team names and dates; no result is used) to confirm the registered group sizes.
"""

import json

import pandas as pd
import pytest

from eplmodel.analysis.promoted_teams import categorize_team, load_team_history
from eplmodel.config import load_config
from eplmodel.models import full_coverage_spec as fcs
from eplmodel.paths import PROCESSED_MATCHES, PROJECT_ROOT
from eplmodel.splits import SELECTION_TARGET_SEASONS, SEASON_ORDER, TRAIN_SEASONS

CFG = load_config(fcs.FULL_COVERAGE_CONFIG)


def test_spec_loads_with_the_locked_and_new_choices():
    spec = fcs.load_full_coverage_spec()
    assert spec.protocol_id == "full_coverage_poisson_v1" and spec.status == "registered_not_implemented"
    assert spec.targets == SELECTION_TARGET_SEASONS and spec.max_season == "2122"
    assert spec.half_life_days == 730.0 and spec.max_goals == 10
    assert spec.primary_arm == fcs.M2_HIERARCHICAL and set(spec.arms) == set(fcs.ARMS)
    assert spec.weight_reference == "refit_date" and spec.identification == "sum_to_zero_over_target_teams"
    assert spec.prior.variance_floor == 0.05 and spec.prior.returning_rule == "promoted_prior_plus_decayed_history"
    assert spec.anchor_tolerance == 1e-7


@pytest.mark.parametrize("section,key,value", [
    ("inherited", "half_life_days", 365.0),                                # H may not be retuned
    ("protocol", "targets", ["1718", "1819", "1920", "2021", "2122", "2223"]),
    ("protocol", "inputs", ["match_results", "market_odds"]),              # no market features
    ("selection", "tuned_values", ["variance_floor"]),                     # no tuning
    ("arms", "primary", "fc_promoted"),
    ("inherited", "warm_start", True),
    ("prior", "variance_floor", 0.0),
])
def test_disagreeing_configs_are_refused(section, key, value):
    cfg = json.loads(json.dumps(CFG))
    cfg[section][key] = value
    with pytest.raises(fcs.FullCoverageSpecError):
        fcs.validate(cfg)


def test_registered_group_sizes_are_internally_consistent():
    g = CFG["groups"]
    for t in SELECTION_TARGET_SEASONS:
        assert g["common"][t] + g["unseen"][t] == g["promoted"][t] + g["continuing_only"][t] == 380
    assert sum(g["full"].values()) == 1900 and sum(g["unseen"].values()) == 296
    assert sum(g["online_fits"].values()) == 586


def test_no_season_after_the_selection_folds_appears_in_any_config_value():
    values = json.dumps(CFG)                 # parsed values only; comments may mention seasons
    for season in ("2223", "2324", "2425", "2526", "2627", "2024-25", "2025-26", "2026-27"):
        assert season not in values


def test_protocol_is_not_yet_run():
    assert CFG["historical_locked"]["status"] == "not_run"
    assert not (PROJECT_ROOT / "results" / CFG["outputs"]["results_name"]).exists()


def test_registered_groups_hold_on_the_real_fixture_lists():
    """Fixtures only (teams, dates); the result columns are not read."""
    if not PROCESSED_MATCHES.exists():
        pytest.skip("data/processed/matches.csv missing")
    df = pd.read_csv(PROCESSED_MATCHES, usecols=["Date", "HomeTeam", "AwayTeam", "Season"], dtype={"Season": str})
    teams = {s: set(gr["HomeTeam"]) | set(gr["AwayTeam"]) for s, gr in df.groupby("Season")}
    th = load_team_history()
    g = CFG["groups"]
    for target in SELECTION_TARGET_SEASONS:
        i = TRAIN_SEASONS.index(target)
        history = TRAIN_SEASONS[:i]
        seen = set().union(*(teams[s] for s in history))
        tgt = df[df["Season"] == target]
        promoted = teams[target] - teams[TRAIN_SEASONS[i - 1]]
        unseen, returning = teams[target] - seen, promoted & seen
        involves = lambda ts: int((tgt["HomeTeam"].isin(ts) | tgt["AwayTeam"].isin(ts)).sum())
        assert sorted(promoted) == g["promoted_teams"][target]
        assert (involves(promoted), involves(unseen), involves(returning)) == (
            g["promoted"][target], g["unseen"][target], g["returning"][target])
        assert len(tgt) - involves(unseen) == g["common"][target]
        assert tgt["Date"].nunique() == g["online_fits"][target]
        yoyo = []
        for team in sorted(promoted):
            if team in seen:
                last = max(s for s in history if team in teams[s])
                out = SEASON_ORDER.index(target) - SEASON_ORDER.index(last) - 1
                if out <= g["yoyo_threshold"]:
                    yoyo.append(team)
            elif categorize_team(team, th, g["yoyo_threshold"]) == fcs.DiagnosticCategory.RECENT_YOYO:
                yoyo.append(team)
        assert yoyo == g["recent_yoyo_teams"][target]
