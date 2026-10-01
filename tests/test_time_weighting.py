"""Time-weighted static Poisson (protocol time_weighted_poisson_v1).

Every weighted fit in this file uses synthetic data. The only real-data tests read the protocol,
the checksum manifest, or fit the UNWEIGHTED static model (H = inf) on the five development folds
to check that it reproduces Experiment 10; no test fits a weighted model on real data, and no test
generates a 2024-25 prediction.
"""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from eplmodel.config import load_config
from eplmodel.evaluation import time_weighting as tw
from eplmodel.evaluation.validation import fold_data, goal_models_fold, prob_columns
from eplmodel.models.poisson import FORMULA, PoissonGoalModel, to_long_format
from eplmodel.models.time_weights import exponential_decay_weights, kish_effective_sample_size, match_age_days
from eplmodel.paths import PROCESSED_DEV_V2, PROJECT_ROOT
from eplmodel.splits import (
    DEVELOPMENT_SEASONS,
    REGISTERED_DEV_TEST_SPECS,
    SEASON_ORDER,
    SplitAccessError,
    selection_folds,
)
from experiments import time_weighted_poisson as twp
from experiments import update_policy_diagnostic, validation_2425
from test_validation import HISTORY, TARGET, TEAMS, _reverse_scores, _round_robin, _with_results, synthetic_league

TCFG = load_config(twp.TW_CONFIG)
VCFG = load_config(validation_2425.VALIDATION_CONFIG)
DCFG = load_config(update_policy_diagnostic.DIAGNOSTIC_CONFIG)
GRID = [float(h) for h in TCFG["candidate"]["half_life_grid_days"]]


@pytest.fixture(scope="module")
def league():
    return synthetic_league()


@pytest.fixture(scope="module")
def history_rows(league):
    return league[league["Season"].isin(HISTORY)].reset_index(drop=True)


def _params(model):
    return model.result_.params.sort_index()


# --- Weight function -----------------------------------------------------------------------------

def test_weights_halve_every_half_life():
    dates = pd.to_datetime(["2020-01-01", "2019-12-01", "2019-01-01", "2018-01-01"])
    w = exponential_decay_weights(dates, "2020-01-01", 365.0)
    assert w[0] == 1.0
    assert w[1] == pytest.approx(2 ** (-31 / 365))
    assert w[2] == pytest.approx(0.5)
    assert w[3] == pytest.approx(2 ** (-730 / 365))


def test_infinite_half_life_gives_unit_weights_and_same_day_matches_share_a_weight():
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2015-06-30"])
    assert exponential_decay_weights(dates, "2020-01-01", math.inf).tolist() == [1.0, 1.0, 1.0]
    w = exponential_decay_weights(dates, "2020-01-03", 183.0)
    assert w[0] == w[1] and 0 < w[2] < w[0] <= 1


@pytest.mark.parametrize("h", [0.0, -365.0, float("nan")])
def test_non_positive_half_life_is_refused(h):
    with pytest.raises(ValueError):
        exponential_decay_weights(pd.to_datetime(["2020-01-01"]), "2020-01-01", h)


def test_match_after_the_reference_date_is_refused():
    with pytest.raises(ValueError, match="after the reference"):
        match_age_days(pd.to_datetime(["2020-01-02"]), "2020-01-01")


def test_kish_effective_sample_size():
    assert kish_effective_sample_size(np.ones(10)) == pytest.approx(10)
    assert kish_effective_sample_size([1.0, 1e-9, 1e-9]) == pytest.approx(1.0, abs=1e-6)
    assert kish_effective_sample_size(3 * np.ones(4)) == pytest.approx(4)


# --- Weighted Poisson fit --------------------------------------------------------------------------

def test_unweighted_fit_is_unchanged(history_rows):
    """weights=None is exactly the established fit (same call as before this protocol)."""
    direct = smf.glm(formula=FORMULA, data=to_long_format(history_rows), family=sm.families.Poisson()).fit()
    model = PoissonGoalModel().fit(history_rows)
    np.testing.assert_array_equal(model.result_.params.to_numpy(), direct.params.to_numpy())


def test_unit_weights_reproduce_the_unweighted_fit(history_rows):
    a = PoissonGoalModel().fit(history_rows)
    b = PoissonGoalModel().fit(history_rows, weights=np.ones(len(history_rows)))
    np.testing.assert_allclose(_params(b), _params(a), rtol=0, atol=1e-10)


def test_fit_is_invariant_to_weight_scale_and_reference_date(history_rows):
    dates = history_rows["Date"]
    w1 = exponential_decay_weights(dates, dates.max(), 365.0)
    w2 = exponential_decay_weights(dates, dates.max() + pd.Timedelta(days=200), 365.0)
    np.testing.assert_allclose(w2 / w1, w2[0] / w1[0])         # a common factor
    fits = [PoissonGoalModel().fit(history_rows, weights=w) for w in (w1, w2, 7.0 * w1)]
    for other in fits[1:]:
        np.testing.assert_allclose(_params(other), _params(fits[0]), rtol=0, atol=1e-9)


def test_var_weights_and_freq_weights_give_the_same_point_estimates(history_rows):
    w = exponential_decay_weights(history_rows["Date"], history_rows["Date"].max(), 365.0)
    long, rows = to_long_format(history_rows), np.concatenate([w, w])
    freq = smf.glm(formula=FORMULA, data=long, family=sm.families.Poisson(), freq_weights=rows).fit()
    model = PoissonGoalModel().fit(history_rows, weights=w)
    np.testing.assert_allclose(model.result_.params.to_numpy(), freq.params.to_numpy(), rtol=0, atol=1e-9)


def test_integer_weights_equal_duplicated_matches(history_rows):
    counts = np.tile([1, 2, 3], len(history_rows))[: len(history_rows)]
    duplicated = history_rows.loc[history_rows.index.repeat(counts)].reset_index(drop=True)
    a = PoissonGoalModel().fit(history_rows, weights=counts.astype(float))
    b = PoissonGoalModel().fit(duplicated)
    np.testing.assert_allclose(_params(a), _params(b), rtol=0, atol=1e-9)


def test_both_goal_rows_of_a_match_get_the_match_weight(history_rows):
    w = exponential_decay_weights(history_rows["Date"], history_rows["Date"].max(), 274.0)
    model = PoissonGoalModel().fit(history_rows, weights=w)
    np.testing.assert_array_equal(model.result_.model.var_weights, np.concatenate([w, w]))


@pytest.mark.parametrize("bad", ["short", "zero", "negative", "nan"])
def test_invalid_weights_are_refused(history_rows, bad):
    w = np.ones(len(history_rows))
    if bad == "short":
        w = w[:-1]
    else:
        w[3] = {"zero": 0.0, "negative": -1.0, "nan": np.nan}[bad]
    with pytest.raises(ValueError):
        PoissonGoalModel().fit(history_rows, weights=w)


def test_recency_weighting_tracks_a_recent_change_in_strength():
    """Sensitivity: team D (not the reference team) is strong early and weak in the last season; weighting moves D's estimate."""
    rng = np.random.default_rng(5)
    teams = ["A", "B", "C", "D"]
    rows = []
    for season, a_strength in ((0, 0.6), (1, 0.6), (2, 0.6), (3, -0.6)):
        for r, (home, away) in enumerate(itertools.permutations(teams, 2)):
            for rep in range(6):
                s = {t: (a_strength if t == "D" else 0.0) for t in teams}
                rows.append({"Date": pd.Timestamp(2010 + season, 9, 1) + pd.Timedelta(days=r + 12 * rep),
                             "HomeTeam": home, "AwayTeam": away, "Season": str(season),
                             "FTHG": rng.poisson(np.exp(0.2 + s[home] - s[away])),
                             "FTAG": rng.poisson(np.exp(0.0 + s[away] - s[home]))})
    df = pd.DataFrame(rows)
    static = PoissonGoalModel().fit(df).team_effects()
    w = exponential_decay_weights(df["Date"], df["Date"].max(), 183.0)
    weighted = PoissonGoalModel().fit(df, weights=w).team_effects()
    assert weighted.loc["D", "attack"] < static.loc["D", "attack"]
    assert weighted.loc["D", "defence"] > static.loc["D", "defence"]


def test_team_effects_have_the_reference_team_at_zero(history_rows):
    effects = PoissonGoalModel().fit(history_rows).team_effects()
    first = sorted(set(history_rows["HomeTeam"]))[0]
    assert effects.loc[first].tolist() == [0.0, 0.0]


# --- Fold predictions ------------------------------------------------------------------------------

ARMS = {tw.grid_arm(h): h for h in (183.0, 730.0, math.inf)}


@pytest.fixture(scope="module")
def tw_fold(league):
    return tw.tw_fold_predictions(league, HISTORY, TARGET, ARMS, 10)


def test_arm_names_and_labels():
    assert tw.grid_arm(math.inf) == "poisson" and tw.grid_arm(365.0) == "poisson_tw_h365"
    assert [tw.half_life_label(h) for h in GRID] == ["183", "274", "365", "548", "730", "1095", "1460", "inf"]
    with pytest.raises(ValueError):
        tw.half_life_label(365.5)


def test_infinite_half_life_reproduces_the_established_static_poisson(league, tw_fold):
    preds, _ = tw_fold
    established, _ = goal_models_fold(fold_data(league, HISTORY, TARGET), HISTORY, TARGET,
                                      VCFG["poisson"], VCFG["dixon_coles"])
    assert list(preds.index) == list(established.index)
    np.testing.assert_array_equal(preds[prob_columns("poisson")].to_numpy(),
                                  established[prob_columns("poisson")].to_numpy())


def test_tw_predictions_ignore_every_target_outcome(league, tw_fold):
    preds, fitted = tw_fold
    preds2, fitted2 = tw.tw_fold_predictions(_reverse_scores(league, league["Season"] == TARGET),
                                             HISTORY, TARGET, ARMS, 10)
    pd.testing.assert_frame_equal(preds2, preds)
    assert fitted2 == fitted


def test_later_seasons_never_enter_the_tw_fit(league, tw_fold):
    preds2, _ = tw.tw_fold_predictions(_reverse_scores(league, league["Season"] == "1819"), HISTORY, TARGET, ARMS, 10)
    pd.testing.assert_frame_equal(preds2, tw_fold[0])


def test_history_outcomes_do_change_the_weighted_fit(league, tw_fold):
    preds2, _ = tw.tw_fold_predictions(_reverse_scores(league, league["Season"] == HISTORY[-1]),
                                       HISTORY, TARGET, ARMS, 10)
    assert not np.allclose(preds2[prob_columns("poisson_tw_h183")], tw_fold[0][prob_columns("poisson_tw_h183")])


def test_weighting_changes_predictions_but_not_eligibility(tw_fold):
    preds, fitted = tw_fold
    assert fitted["unseen_teams"] == ["NEW"] and fitted["n_common"] == len(preds) == 20
    assert not preds["HomeTeam"].eq("NEW").any() and not preds["AwayTeam"].eq("NEW").any()
    assert not np.allclose(preds[prob_columns("poisson_tw_h183")], preds[prob_columns("poisson")])


def test_tw_rows_sum_to_one_and_contain_no_results(tw_fold):
    preds, _ = tw_fold
    for arm in ARMS:
        np.testing.assert_allclose(preds[prob_columns(arm)].sum(axis=1), 1.0, atol=1e-12)
    assert not {"FTHG", "FTAG", "FTR"} & set(preds.columns)


def test_fit_diagnostics_are_recorded(tw_fold):
    _, fitted = tw_fold
    static, short = fitted["arms"]["poisson"], fitted["arms"]["poisson_tw_h183"]
    assert static["kish_ess"] == pytest.approx(static["n_history_matches"])
    assert static["spearman_net_strength_vs_static"] == pytest.approx(1.0)
    assert short["kish_ess"] < static["kish_ess"]
    assert short["last_season_weight_share"] > static["last_season_weight_share"]
    assert short["glm_converged"] and static["glm_converged"]
    assert set(short["team_kish_ess"]) == set(TEAMS)


def test_returning_teams_are_identified_from_fixtures():
    league = synthetic_league(target_teams=("A", "B", "C", "D", "E", "F"))
    hist = league[league["Season"].isin(("1415", "1516"))]
    hist = hist[~((hist["Season"] == "1516") & (hist["HomeTeam"].eq("F") | hist["AwayTeam"].eq("F")))]
    assert tw.returning_teams(hist, league[league["Season"] == "1718"]) == ["F"]


# --- Development-stage data guard ----------------------------------------------------------------

def test_restrict_to_development_drops_every_later_season():
    df = pd.DataFrame({"Season": list(DEVELOPMENT_SEASONS)})
    out = tw.restrict_to_development(df, "2122")
    assert list(out["Season"]) == list(DEVELOPMENT_SEASONS[:8])


@pytest.mark.parametrize("late", ["2223", "2324", "2425"])
def test_development_guard_refuses_later_seasons(late):
    with pytest.raises(SplitAccessError):
        tw.assert_development_only(pd.DataFrame({"Season": ["1718", late]}), "2122")


def _allow_short_seasons(monkeypatch):
    """Synthetic seasons have 60 matches; every other structural check (including FTR) stays active."""
    from eplmodel.data.validate import validate_matches
    monkeypatch.setattr(twp, "validate_matches", lambda df: validate_matches(df, matches_per_season=None))


def test_development_loader_drops_later_rows_before_validation(monkeypatch):
    """2022-25 rows are malformed here: they would fail validation if anything looked at them."""
    full = _synthetic_development_league()
    late = full["Season"].map(SEASON_ORDER.index) > SEASON_ORDER.index("2122")
    full.loc[late, "FTR"] = "X"
    monkeypatch.setattr(twp, "verify_file", lambda path: True)
    monkeypatch.setattr(twp, "load_matches", lambda path, validate: full.copy())
    _allow_short_seasons(monkeypatch)
    out = twp.load_development_matches("2122")
    assert sorted(out["Season"].unique(), key=SEASON_ORDER.index) == list(DEVELOPMENT_SEASONS[:8])
    # Sensitivity: without the cut, the malformed later rows are caught by validation.
    monkeypatch.setattr(twp.tw, "restrict_to_development", lambda df, max_season: df)
    from eplmodel.data.validate import DataValidationError
    with pytest.raises(DataValidationError, match="FTR"):
        twp.load_development_matches("2122")


# --- Selection of the half-life ----------------------------------------------------------------

def _losses(table):
    """{target: {H: per-match losses}} from {target: {H: list}}."""
    return {t: {h: np.asarray(v, dtype=float) for h, v in row.items()} for t, row in table.items()}


def _clusters(losses):
    return {t: np.array([f"{t}_{i}" for i in range(len(next(iter(v.values()))))]) for t, v in losses.items()}


def test_one_se_rule_picks_the_longest_half_life_within_one_se():
    # H=365 is best; 730 is 0.01 worse with large noise (within 1 SE); inf is clearly worse.
    noise = np.array([0.3, -0.3, 0.3, -0.3])
    losses = _losses({"1718": {365.0: [1.0] * 4, 730.0: list(1.01 + noise), math.inf: [1.2] * 4},
                      "1819": {365.0: [1.0] * 4, 730.0: list(1.01 - noise), math.inf: [1.2] * 4}})
    sel = tw.select_half_life(losses, _clusters(losses), 1.0)
    assert sel["h_min"] == "365" and sel["h_selected"] == "730"
    assert [r["within_one_se"] for r in sel["table"]] == [True, True, False]


def test_one_se_rule_keeps_h_min_when_others_are_clearly_worse():
    losses = _losses({"1718": {365.0: [1.0, 1.0, 1.0], math.inf: [1.1, 1.1, 1.1]}})
    assert tw.select_half_life(losses, _clusters(losses), 1.0)["h_selected"] == "365"


def test_exact_ties_go_to_the_longer_half_life_and_flat_curves_select_static():
    losses = _losses({"1718": {365.0: [1.0, 0.9], 730.0: [1.0, 0.9], math.inf: [1.0, 0.9]}})
    sel = tw.select_half_life(losses, _clusters(losses), 1.0)
    assert sel["h_min"] == "inf" and sel["h_selected"] == "inf"


def test_criterion_weights_targets_equally():
    # Pooled per match, 1718 (4 matches) would favour 365; equal target weights favour inf.
    losses = _losses({"1718": {365.0: [1.0] * 4, math.inf: [1.1] * 4},
                      "1819": {365.0: [1.5], math.inf: [1.3]}})
    sel = tw.select_half_life(losses, _clusters(losses), 0.0)
    assert sel["h_min"] == "inf"


def test_selection_refuses_mismatched_grids():
    losses = _losses({"1718": {365.0: [1.0], math.inf: [1.0]}, "1819": {730.0: [1.0], math.inf: [1.0]}})
    with pytest.raises(ValueError, match="same grid"):
        tw.select_half_life(losses, _clusters(losses), 1.0)


def _five_target_losses(seed=0):
    rng = np.random.default_rng(seed)
    return _losses({t: {h: list(1.0 + 0.01 * i + rng.normal(0, 0.05, 30)) for i, h in enumerate(GRID)}
                    for t in ("1718", "1819", "1920", "2021", "2122")})


def test_nested_selection_uses_only_earlier_targets():
    losses = _five_target_losses()
    nested = tw.nested_selection(losses, _clusters(losses), ["1819", "1920", "2021", "2122"], 1.0)
    assert {t: s["targets"] for t, s in nested.items()} == {
        "1819": ["1718"], "1920": ["1718", "1819"], "2021": ["1718", "1819", "1920"],
        "2122": ["1718", "1819", "1920", "2021"]}
    # Changing a target's outcomes cannot change the H selected for it or any earlier outer target.
    changed = {**losses, "1920": {h: v[::-1] + (0.5 if h == 183.0 else 0.0) for h, v in losses["1920"].items()}}
    nested2 = tw.nested_selection(changed, _clusters(changed), ["1819", "1920", "2021", "2122"], 1.0)
    assert nested2["1819"] == nested["1819"] and nested2["1920"] == nested["1920"]


def test_leave_one_target_out_covers_every_target():
    losses = _five_target_losses(1)
    out = tw.leave_one_target_out(losses, _clusters(losses), 1.0)
    assert set(out) == set(losses) and set(out.values()) <= {tw.half_life_label(h) for h in GRID}


# --- Evidence rules ------------------------------------------------------------------------------

def _outer(means, n=50, spread=0.001):
    alt = np.where(np.arange(n) % 2 == 0, spread, -spread)
    return {t: {"log_loss": m + alt, "brier": 0.5 * m + alt} for t, m in means.items()}


def _outer_clusters(outer):
    return {t: np.array([f"{t}_{i}" for i in range(len(v["log_loss"]))]) for t, v in outer.items()}


def test_development_criterion_met():
    outer = _outer({"1819": -0.004, "1920": -0.003, "2021": -0.005, "2122": 0.001})
    d = tw.development_criterion(outer, _outer_clusters(outer), 0.002, 2.0, 3)
    assert d["met"] and d["n_negative_folds"] == 3


@pytest.mark.parametrize("means,failed", [
    ({"1819": -0.001, "1920": -0.001, "2021": -0.001, "2122": -0.001}, "below_floor"),
    ({"1819": -0.010, "1920": -0.010, "2021": 0.001, "2122": 0.001}, "enough_negative_folds"),
    ({"1819": 0.0, "1920": 0.0, "2021": 0.0, "2122": 0.0}, "below_floor"),   # H = inf in every outer fold
])
def test_development_criterion_not_met(means, failed):
    outer = _outer(means)
    d = tw.development_criterion(outer, _outer_clusters(outer), 0.002, 2.0, 3)
    assert not d["met"] and not d["checks"][failed]


def test_development_criterion_requires_two_clustered_ses_and_brier_agreement():
    noisy = _outer({"1819": -0.003, "1920": -0.003, "2021": -0.003, "2122": -0.003}, spread=0.5)
    assert not tw.development_criterion(noisy, _outer_clusters(noisy), 0.002, 2.0, 3)["checks"]["beyond_se_multiple"]
    outer = _outer({"1819": -0.004, "1920": -0.004, "2021": -0.004, "2122": -0.004})
    for t in outer:
        outer[t]["brier"] = np.abs(outer[t]["brier"]) + 0.01
    d = tw.development_criterion(outer, _outer_clusters(outer), 0.002, 2.0, 3)
    assert not d["met"] and not d["checks"]["brier_same_sign"]


def test_validation_criterion():
    n = 100
    alt = np.where(np.arange(n) % 2 == 0, 0.001, -0.001)
    clusters = np.arange(n).astype(str)
    assert tw.validation_criterion(-0.004 + alt, -0.002 + alt, clusters, 0.002, 2.0)["met"]
    assert not tw.validation_criterion(-0.0015 + alt, -0.002 + alt, clusters, 0.002, 2.0)["met"]
    assert not tw.validation_criterion(-0.004 + alt, 0.002 + alt, clusters, 0.002, 2.0)["met"]


def test_registered_readings():
    assert tw.reading(True, True, False) == "out_of_sample_evidence_weighting_helps_static_poisson"
    assert tw.reading(True, False, False) == "historical_benefit_not_replicated_in_2425"
    assert tw.reading(False, True, False) == "2425_specific_observation_cannot_confirm"
    assert tw.reading(False, False, False) == "no_evidence_weighting_helps"
    assert tw.reading(True, None, True) == "no_weighting_selected"
    assert tw.reading(False, None, False) == "validation_not_yet_scored"


# --- Protocol ------------------------------------------------------------------------------------

def test_protocol_agrees_with_frozen_specs_and_code():
    twp.check_protocol(TCFG, VCFG, DCFG, load_config())


def test_registered_values_are_the_approved_ones():
    c, s, e = TCFG["candidate"], TCFG["selection"], TCFG["evidence"]
    assert GRID == [183.0, 274.0, 365.0, 548.0, 730.0, 1095.0, 1460.0, math.inf]
    assert (s["rule"], s["se_multiple"], s["se_type"]) == ("one_se_longest", 1.0, "clustered")
    assert e["practical_floor_log_loss"] == 0.002 and e["clustered_se_multiple"] == 2.0
    assert e["development_min_negative_outer_folds"] == 3
    assert TCFG["protocol"]["score_grid_on_validation"] is False
    assert (c["weighting"], c["age_unit"], c["truncation"]) == ("exponential_half_life_days", "days", "none")
    assert TCFG["nested"]["outer_targets"] == ["1819", "1920", "2021", "2122"]


def test_registered_populations_agree_with_the_update_policy_protocol():
    g, dg = TCFG["groups"], DCFG["groups"]
    assert g["expected_common"] == dg["expected_common"]
    assert g["expected_full"] == dg["expected_full"]
    assert g["expected_unseen_teams"] == dg["expected_unseen_teams"]


@pytest.mark.parametrize("section,key,value", [
    ("protocol", "development_targets", ["1718", "1819", "1920", "2021", "2223"]),
    ("protocol", "validation_target", "2526"),
    ("protocol", "score_grid_on_validation", True),
    ("protocol", "data_sha256", "0" * 64),
    ("candidate", "formula", "Goals ~ Team + Opponent"),
    ("candidate", "max_goals", 12),
    ("candidate", "in_season_updates", True),
    ("candidate", "unseen_team_rule", "start_at_average"),
    ("candidate", "truncation", "below_1e-3"),
    ("candidate", "half_life_grid_days", [183.0, 365.0]),
    ("candidate", "half_life_grid_days", [365.5, math.inf]),
    ("selection", "rule", "argmin"),
    ("nested", "outer_targets", ["1718", "1819", "1920", "2021", "2122"]),
    ("development_stage", "max_season", "2425"),
    ("reproduction", "experiment_10_sha256", "0" * 64),
    ("outputs", "in_run_all", True),
])
def test_protocol_mismatch_is_refused(section, key, value):
    tampered = json.loads(json.dumps(TCFG))
    tampered[section][key] = value
    with pytest.raises((twp.ProtocolMismatchError, ValueError)):
        twp.check_protocol(tampered, VCFG, DCFG, load_config())


def test_new_specs_are_not_registered_for_dev_test_or_holdout():
    assert TCFG["candidate"]["spec_id"] not in REGISTERED_DEV_TEST_SPECS
    tampered = json.loads(json.dumps(TCFG))
    tampered["candidate"]["spec_id"] = "poisson_static_v1"
    with pytest.raises(twp.ProtocolMismatchError):
        twp.check_protocol(tampered, VCFG, DCFG, load_config())


def test_group_check_refuses_unregistered_sizes():
    good = {"n_full": 380, "n_common": 342, "unseen_teams": ["Ipswich"]}
    twp.check_groups("2425", good, TCFG)
    with pytest.raises(twp.ProtocolMismatchError):
        twp.check_groups("2425", {**good, "n_common": 341}, TCFG)
    with pytest.raises(twp.ProtocolMismatchError):
        twp.check_groups("1718", good, TCFG)


def test_experiment_is_not_in_run_all():
    from experiments import run_all
    assert twp not in run_all.EXPERIMENTS
    assert "time_weighted" not in (PROJECT_ROOT / "experiments" / "run_all.py").read_text(encoding="utf-8")


# --- Validation-stage lock -------------------------------------------------------------------------

def test_unlocked_protocol_refuses_validation_before_loading_data(monkeypatch):
    unlocked = json.loads(json.dumps(TCFG))
    unlocked["locked"] = {"status": "unlocked", "half_life_days": "", "development_metrics_sha256": "",
                          "development_commit": ""}
    monkeypatch.setattr(twp, "load_all_configs", lambda: (unlocked, VCFG, DCFG, load_config()))

    def forbidden(*args, **kwargs):
        raise AssertionError("validation data must not be loaded while H* is unlocked")

    import eplmodel.data
    monkeypatch.setattr(eplmodel.data, "load_dev_matches", forbidden)
    monkeypatch.setattr(twp, "verify_file", forbidden)
    with pytest.raises(twp.LockError, match="not locked"):
        twp.run_validation(write=False)


def test_committed_lock_is_the_development_selection():
    """The committed [locked] H* is on the grid and, when the development metrics exist, equals their H*."""
    lock = TCFG["locked"]
    assert lock["status"] == "locked" and lock["half_life_days"] == 730.0 and 730.0 in GRID
    assert lock["development_commit"] == "ae547570c8d8368bfe30fd91f6a763237637d5f9"
    path = PROJECT_ROOT / "results" / TCFG["development_stage"]["results_name"] / "metrics.json"
    if path.exists():
        from eplmodel.data.checksums import content_sha256
        assert content_sha256(path) == lock["development_metrics_sha256"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["results"]["selection"]["h_star"] == tw.half_life_label(lock["half_life_days"])
        assert payload["provenance"]["git_commit"] == lock["development_commit"]
        assert payload["provenance"]["git_dirty"] is False


def _locked_setup(tmp_path, monkeypatch, h_star="365", lock_h=365.0, dirty=False, commit="abc", clean=True,
                  ancestor=True, sha=None):
    metrics_dir = tmp_path / TCFG["development_stage"]["results_name"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "metrics.json"
    path.write_text(json.dumps({"results": {"selection": {"h_star": h_star}, "criterion_d": {"met": False}},
                                "provenance": {"git_commit": "abc", "git_dirty": dirty}}), encoding="utf-8")
    from eplmodel.data.checksums import content_sha256
    monkeypatch.setattr(twp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(twp, "working_tree_clean", lambda: clean)
    monkeypatch.setattr(twp, "is_ancestor_of_head", lambda c: ancestor)
    cfg = json.loads(json.dumps(TCFG))
    cfg["locked"] = {"status": "locked", "half_life_days": lock_h,
                     "development_metrics_sha256": sha or content_sha256(path), "development_commit": commit}
    return cfg


def test_lock_accepts_a_committed_clean_lock(tmp_path, monkeypatch):
    assert twp.check_lock(_locked_setup(tmp_path, monkeypatch)) == 365.0


@pytest.mark.parametrize("kwargs,message", [
    ({"lock_h": 400.0}, "not on the registered grid"),
    ({"lock_h": 730.0}, "differs from the half-life selected"),
    ({"sha": "0" * 64}, "differ from"),
    ({"dirty": True}, "clean commit"),
    ({"commit": "def"}, "clean commit"),
    ({"ancestor": False}, "does not descend"),
    ({"clean": False}, "uncommitted changes"),
])
def test_lock_refusals(tmp_path, monkeypatch, kwargs, message):
    cfg = _locked_setup(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(twp.LockError, match=message):
        twp.check_lock(cfg)


def test_lock_refuses_a_non_numeric_half_life(tmp_path, monkeypatch):
    cfg = _locked_setup(tmp_path, monkeypatch)
    cfg["locked"]["half_life_days"] = ""
    with pytest.raises(twp.LockError, match="not a number"):
        twp.check_lock(cfg)


# --- End-to-end on synthetic data only ------------------------------------------------------------

def _synthetic_development_league(seed=2):
    """Every development season; strengths drift over time; a new team replaces F from 1718 on."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, season in enumerate(DEVELOPMENT_SEASONS):
        teams = TEAMS if SEASON_ORDER.index(season) < SEASON_ORDER.index("1718") else [*TEAMS[:5], "NEW"]
        strength = {t: 0.3 * np.sin(k / 2 + i) for i, t in enumerate([*TEAMS, "NEW"])}
        start = pd.Timestamp(2000 + int(season[:2]), 8, 15)
        for r, fixtures in enumerate(_round_robin(teams) * 2):
            for home, away in fixtures:
                rows.append({"Date": start + pd.Timedelta(weeks=r), "HomeTeam": home, "AwayTeam": away,
                             "FTHG": int(rng.poisson(np.exp(0.3 + strength[home] - strength[away]))),
                             "FTAG": int(rng.poisson(np.exp(0.1 + strength[away] - strength[home]))),
                             "Season": season})
    return _with_results(pd.DataFrame(rows))


def test_development_stage_runs_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    synthetic = _synthetic_development_league()
    late = synthetic["Season"].map(SEASON_ORDER.index) > SEASON_ORDER.index("2122")
    synthetic.loc[late, "FTR"] = "X"   # would break validation and scoring if any later row were used
    monkeypatch.setattr(twp, "verify_file", lambda path: True)
    monkeypatch.setattr(twp, "load_matches", lambda path, validate: synthetic.copy())
    _allow_short_seasons(monkeypatch)
    monkeypatch.setattr(twp, "check_groups", lambda target, fitted, tcfg: None)

    def recorded(path_key, sha_key, tcfg, targets):
        clean = synthetic[~late]
        frames = []
        for history, target in selection_folds(list(targets)):
            preds, _ = validation_2425.fold_predictions(clean, history, target, VCFG)
            frames.append(preds.assign(fold_target=target))
        return pd.concat(frames)

    monkeypatch.setattr(twp, "load_recorded", recorded)
    out = twp.run_development(write=False)
    assert out["seasons_loaded"] == list(DEVELOPMENT_SEASONS[:8])
    assert [f["target"] for f in out["folds"]] == ["1718", "1819", "1920", "2021", "2122"]
    assert all(f["fitted"]["reproduction_max_abs_diff"] == 0.0 for f in out["folds"])
    assert set(out["nested"]["selected"]) == {"1819", "1920", "2021", "2122"}
    assert out["selection"]["h_star"] in out["grid"]
    assert out["reading"] in ("validation_not_yet_scored", "no_weighting_selected")
    assert set(out["folds"][0]["scores"]) == set(out["grid"])


def test_validation_stage_scores_only_the_locked_half_life_on_synthetic_data(tmp_path, monkeypatch):
    synthetic = _synthetic_development_league(3)
    cfg = _locked_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(twp, "load_all_configs", lambda: (cfg, VCFG, DCFG, load_config()))
    monkeypatch.setattr(twp, "verify_file", lambda path: True)
    import eplmodel.data
    monkeypatch.setattr(eplmodel.data, "load_dev_matches", lambda: synthetic.copy())
    monkeypatch.setattr(twp, "check_groups", lambda target, fitted, tcfg: None)

    def recorded(path_key, sha_key, tcfg, targets):
        (history, target), = selection_folds(list(targets))
        from eplmodel.evaluation import update_policy as up
        preds, _ = up.diagnostic_fold_predictions(synthetic, history, target, VCFG, DCFG)
        return preds.assign(fold_target=target)

    monkeypatch.setattr(twp, "load_recorded", recorded)
    calls = []
    original = tw.tw_fold_predictions

    def spy(matches, history, target, arms, max_goals):
        calls.append((target, dict(arms)))
        return original(matches, history, target, arms, max_goals)

    monkeypatch.setattr(twp.tw, "tw_fold_predictions", spy)
    out = twp.run_validation(write=False)
    assert calls == [("2425", {"poisson_tw": 365.0})]           # the grid is never scored on the target
    assert out["h_star"] == "365" and out["n_common"] == 60
    assert max(out["decomposition"]["max_identity_residual"].values()) <= 1e-12
    assert out["reading"] in ("2425_specific_observation_cannot_confirm", "no_evidence_weighting_helps")
    split = out["decomposition"]["experiment_11_split"]
    total = split["poisson_minus_elo_f2"]["log_loss"]["mean"]
    parts = split["poisson_minus_poisson_tw"]["log_loss"]["mean"] + split["poisson_tw_minus_elo_f2"]["log_loss"]["mean"]
    assert total == pytest.approx(parts, abs=1e-12)


def test_validation_with_infinite_h_star_scores_nothing(tmp_path, monkeypatch):
    cfg = _locked_setup(tmp_path, monkeypatch, h_star="inf", lock_h=math.inf)
    monkeypatch.setattr(twp, "load_all_configs", lambda: (cfg, VCFG, DCFG, load_config()))

    def forbidden(*args, **kwargs):
        raise AssertionError("nothing may be loaded or fitted when H* = inf")

    import eplmodel.data
    monkeypatch.setattr(eplmodel.data, "load_dev_matches", forbidden)
    monkeypatch.setattr(twp.tw, "tw_fold_predictions", forbidden)
    out = twp.run_validation(write=False)
    assert out["h_star"] == "inf" and out["reading"] == "no_weighting_selected"


# --- Real data: unweighted static model only ------------------------------------------------------

@pytest.mark.golden
def test_static_arm_reproduces_experiment_10_on_the_development_folds():
    """H = inf only (the unweighted established fit) on folds 2017-18..2021-22; no weighted fit, no 2024-25 row."""
    if not PROCESSED_DEV_V2.exists():
        pytest.skip("data/processed/matches_dev_v2.csv missing")
    matches = twp.load_development_matches("2122")
    assert max(matches["Season"], key=SEASON_ORDER.index) == "2122"
    targets = TCFG["protocol"]["development_targets"]
    recorded = twp.load_recorded("experiment_10_predictions", "experiment_10_sha256", TCFG, targets)
    for history, target in selection_folds(targets):
        preds, fitted = tw.tw_fold_predictions(matches, history, target, {"poisson": math.inf}, 10)
        twp.check_groups(target, fitted, TCFG)
        assert twp.check_reproduction(target, preds, recorded, ["poisson"], 1e-12, common_only=True) <= 1e-12
