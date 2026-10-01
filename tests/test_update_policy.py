"""Update-policy diagnostic (protocol update_policy_diagnostic_v1).

All prediction and scoring tests use the synthetic league of test_validation. Tests on the real
data use the protocol, the checksum manifest and the fixture lists only: no test generates a
diagnostic prediction or scores anything on real data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from eplmodel.config import load_config
from eplmodel.constants import OUTCOMES
from eplmodel.data.checksums import content_sha256
from eplmodel.evaluation import update_policy as up
from eplmodel.evaluation.alignment import involves_teams
from eplmodel.evaluation.validation import fold_data, fold_predictions, online_elo_layer, prob_columns, unseen_teams
from eplmodel.models.elo import (
    EloOutcomeModel,
    final_ratings,
    home_shift_from_results,
    run_elo,
    season_start_elo,
)
from eplmodel.paths import PROCESSED_DEV_V2, PROCESSED_MATCHES, PROJECT_ROOT
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import (
    REGISTERED_DEV_TEST_SPECS,
    DevTestAccessError,
    HoldoutAccessError,
    selection_folds,
)
from experiments import update_policy_diagnostic as diag
from experiments import validation_2425
from test_validation import HISTORY, TARGET, TEAMS, _reverse_scores, _round_robin, _with_results, synthetic_league

VCFG = load_config(validation_2425.VALIDATION_CONFIG)
DCFG = load_config(diag.DIAGNOSTIC_CONFIG)
RATING_ARGS = {"k": 25, "home_adv": 0.0, "initial_rating": 1500.0}


@pytest.fixture(scope="module")
def league():
    return synthetic_league()


@pytest.fixture(scope="module")
def dfold(league):
    return up.diagnostic_fold_predictions(league, HISTORY, TARGET, VCFG, DCFG)


def _season_start_preds(league, elo_cfg=VCFG["elo"], f2_cfg=DCFG["elo_f2"], history=HISTORY, target=TARGET):
    data = fold_data(league, history, target)
    preds, fitted, _ = up.season_start_elo_fold(data, history, target, elo_cfg, f2_cfg)
    return preds, fitted


# --- Season-start ratings ------------------------------------------------------------------

def test_final_ratings_equal_the_online_state(league):
    hist = league[league["Season"].isin(HISTORY)]
    end = final_ratings(hist, **RATING_ARGS)
    online = run_elo(league, **RATING_ARGS)
    tgt = online[online["Season"] == TARGET]
    for team in set(tgt["HomeTeam"]) | set(tgt["AwayTeam"]):
        first = tgt[(tgt["HomeTeam"] == team) | (tgt["AwayTeam"] == team)].iloc[0]
        pre_match = first["EloHome"] if first["HomeTeam"] == team else first["EloAway"]
        assert pre_match == end.get(team, 1500.0)


def test_season_start_features_are_frozen_within_each_season(league):
    rated = season_start_elo(league, **RATING_ARGS)
    assert (rated.loc[rated["Season"] == HISTORY[0], ["EloHomeStart", "EloAwayStart"]] == 1500.0).all().all()
    for season in rated["Season"].unique():
        rows = rated[rated["Season"] == season]
        before = league[league["Season"] < season]
        expected = final_ratings(before, **RATING_ARGS)
        long = pd.concat([rows[["HomeTeam", "EloHomeStart"]].set_axis(["team", "r"], axis=1),
                          rows[["AwayTeam", "EloAwayStart"]].set_axis(["team", "r"], axis=1)])
        assert (long.groupby("team")["r"].nunique() == 1).all()
        for team, r in long.groupby("team")["r"].first().items():
            assert r == expected.get(team, 1500.0)


def test_season_start_elo_refuses_non_contiguous_seasons(league):
    shuffled = pd.concat([league[league["Season"] == "1516"], league[league["Season"] == "1415"],
                          league[league["Season"] == "1516"]])
    with pytest.raises(ValueError, match="contiguous"):
        season_start_elo(shuffled, **RATING_ARGS)


# --- No target-season leakage -----------------------------------------------------------------

def test_season_start_predictions_ignore_every_target_outcome(league):
    before, _ = _season_start_preds(league)
    after, _ = _season_start_preds(_reverse_scores(league, league["Season"] == TARGET))
    pd.testing.assert_frame_equal(before, after)


def test_frozen_ratings_are_constant_through_the_target_season(dfold):
    preds, _ = dfold
    long = pd.concat([preds[["HomeTeam", "elo_start_home"]].set_axis(["team", "r"], axis=1),
                      preds[["AwayTeam", "elo_start_away"]].set_axis(["team", "r"], axis=1)])
    assert (long.groupby("team")["r"].nunique() == 1).all()
    for arm in ("elo_f1", "elo_f2"):  # same fixture, same frozen ratings -> same probabilities all season
        by_fixture = preds.groupby(["HomeTeam", "AwayTeam"])[prob_columns(arm)].nunique()
        assert (by_fixture == 1).all().all()


def test_history_outcomes_do_change_the_season_start_arms(league):
    before, _ = _season_start_preds(league)
    after, _ = _season_start_preds(_reverse_scores(league, league["Season"] == HISTORY[-1]))
    assert not np.allclose(before[prob_columns("elo_f1")], after[prob_columns("elo_f1")])


def test_unseen_team_is_frozen_at_the_initial_rating(dfold):
    preds, _ = dfold
    assert (preds.loc[preds["HomeTeam"] == "NEW", "elo_start_home"] == 1500.0).all()
    assert (preds.loc[preds["AwayTeam"] == "NEW", "elo_start_away"] == 1500.0).all()


def test_returning_team_resumes_its_last_rating(league):
    """F misses 1718 and returns in 1819 (renamed from NEW): it resumes its end-of-1617 rating."""
    returned = league.copy()
    in_1819 = returned["Season"] == "1819"
    teams = ["HomeTeam", "AwayTeam"]
    returned.loc[in_1819, teams] = returned.loc[in_1819, teams].replace("NEW", "F")
    returned = _with_results(returned)
    history = (*HISTORY, "1718")
    preds, _ = _season_start_preds(returned, history=history, target="1819")
    preds = preds.join(returned.set_index("match_id")[["HomeTeam"]])
    expected = final_ratings(returned[returned["Season"].isin(HISTORY)], **RATING_ARGS)["F"]
    assert expected != 1500.0
    assert (preds.loc[preds["HomeTeam"] == "F", "elo_start_home"] == expected).all()


# --- F1 shares the online layer; F2 has its own ---------------------------------------------

def test_f1_uses_the_online_layer_object_and_shift(league):
    data = fold_data(league, HISTORY, TARGET)
    _, layer, shift = online_elo_layer(data, HISTORY, VCFG["elo"])
    preds, fitted = _season_start_preds(league)
    assert fitted["f1_home_shift"] == shift
    diff = (preds["elo_start_home"] + shift) - preds["elo_start_away"]
    np.testing.assert_array_equal(preds[prob_columns("elo_f1")].to_numpy(), layer.predict_proba(diff))


def test_f1_equals_online_elo_before_either_team_has_played(dfold):
    preds, fitted = dfold
    first = preds["first_match_both"].to_numpy()
    assert first.sum() == fitted["season_start_elo"]["n_first_match_both"] == 3   # round 1 of a 6-team league
    np.testing.assert_array_equal(preds.loc[first, prob_columns("elo_f1")].to_numpy(),
                                  preds.loc[first, prob_columns("elo")].to_numpy())
    assert not np.allclose(preds.loc[~first, prob_columns("elo_f1")], preds.loc[~first, prob_columns("elo")])


def test_f2_layer_is_fitted_on_season_start_features_of_history_seasons_two_to_n(league):
    preds, fitted = _season_start_preds(league)
    assert fitted["f2_fit_seasons"] == list(HISTORY[1:])
    rated = season_start_elo(league[league["Season"].isin(HISTORY)], **RATING_ARGS)
    rows = rated[rated["Season"].isin(HISTORY[1:])]
    assert fitted["f2_n_fit_matches"] == len(rows)
    shift = home_shift_from_results(rows["FTR"])
    assert fitted["f2_home_shift"] == shift
    model = EloOutcomeModel().fit((rows["EloHomeStart"] + shift) - rows["EloAwayStart"], rows["FTR"])
    diff = (preds["elo_start_home"] + shift) - preds["elo_start_away"]
    np.testing.assert_array_equal(preds[prob_columns("elo_f2")].to_numpy(), model.predict_proba(diff))


def test_f2_does_not_depend_on_the_online_layer(league):
    base, _ = _season_start_preds(league)
    other, _ = _season_start_preds(league, elo_cfg={**VCFG["elo"], "logistic_C": 0.01})
    pd.testing.assert_frame_equal(base[prob_columns("elo_f2")], other[prob_columns("elo_f2")])
    assert not np.allclose(base[prob_columns("elo_f1")], other[prob_columns("elo_f1")])


def test_diagnostic_arms_reuse_the_established_predictions_unchanged(league, dfold):
    preds, _ = dfold
    established, _ = fold_predictions(league, HISTORY, TARGET, VCFG)
    pd.testing.assert_frame_equal(preds[established.columns], established)


# --- Prediction table ---------------------------------------------------------------------------

def test_rows_sum_to_one_in_hda_order(dfold):
    preds, _ = dfold
    for arm in ("elo_f1", "elo_f2"):
        assert prob_columns(arm) == [f"{arm}_{o}" for o in OUTCOMES]
        np.testing.assert_allclose(preds[prob_columns(arm)].sum(axis=1), 1.0, atol=1e-12)
    order = np.argsort((preds["elo_start_home"] - preds["elo_start_away"]).to_numpy())
    for arm in ("elo_f1", "elo_f2"):   # P(H) rises with the home side's frozen rating advantage
        assert np.all(np.diff(preds[f"{arm}_H"].to_numpy()[order]) >= -1e-15)


def test_predictions_contain_no_results(dfold):
    preds, _ = dfold
    assert not {"FTHG", "FTAG", "FTR"} & set(preds.columns)


def test_folds_refuse_holdout_and_exposed_targets(league):
    with pytest.raises(HoldoutAccessError):
        up.diagnostic_fold_predictions(league, HISTORY, "2526", VCFG, DCFG)
    with pytest.raises(DevTestAccessError):
        up.diagnostic_fold_predictions(league, HISTORY, "2223", VCFG, DCFG)
    sealed = pd.concat([league, league[league["Season"] == TARGET].assign(Season="2526")])
    with pytest.raises(HoldoutAccessError):
        up.diagnostic_fold_predictions(sealed, HISTORY, TARGET, VCFG, DCFG)


# --- Segments ---------------------------------------------------------------------------------

def test_prior_games_use_fixtures_only_and_count_earlier_dates(league):
    tgt = league[league["Season"] == TARGET]
    prior = up.prior_games_played(tgt)
    rounds = ((tgt["Date"] - tgt["Date"].min()).dt.days // 7).to_numpy()
    np.testing.assert_array_equal(prior.to_numpy(), rounds)        # one match per team per round
    fixtures_only = up.prior_games_played(tgt[["match_id", "Date", "HomeTeam", "AwayTeam"]])
    pd.testing.assert_series_equal(prior, fixtures_only)


def test_prior_games_handle_uneven_schedules():
    fixtures = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-08", "2020-01-15"]),
        "HomeTeam": ["A", "C", "A", "B"], "AwayTeam": ["B", "D", "C", "A"]})
    fixtures["match_id"] = [f"m{i}" for i in range(4)]
    np.testing.assert_array_equal(up.prior_games_played(fixtures).to_numpy(), [0, 0, 1, 1.5])


@pytest.mark.parametrize("value,label", [(0, "0-9"), (9.5, "0-9"), (10, "10-18"), (18.5, "10-18"),
                                         (19, "19-28"), (28.5, "19-28"), (29, "29+"), (37, "29+")])
def test_segment_boundaries(value, label):
    seg = DCFG["segments"]
    assert up.assign_segments([value], seg["lower_edges"], seg["labels"])[0] == label


def test_segment_definition_errors():
    with pytest.raises(ValueError):
        up.assign_segments([-1], [0, 10], ["a", "b"])
    with pytest.raises(ValueError):
        up.assign_segments([1], [10, 0], ["a", "b"])
    with pytest.raises(ValueError):
        up.assign_segments([1], [0, 10], ["a"])


# --- Uncertainty, sign and decomposition ------------------------------------------------------

def test_clustered_se_equals_naive_with_one_match_per_cluster():
    rng = np.random.default_rng(3)
    left, right = rng.normal(size=50), rng.normal(size=50)
    out = up.paired_difference_clustered(left, right, np.arange(50))
    assert out["clustered_se"] == pytest.approx(out["naive_se"], rel=1e-12)
    assert out["n_clusters"] == 50


def test_clustered_se_hand_example():
    out = up.paired_difference_clustered([1, 2, 3, 6], [0, 0, 0, 0], ["a", "a", "b", "b"])
    # mean 3; deviations -2, -1, 0, 3; cluster sums -3, 3; SE^2 = 2/1 * 18 / 16 = 2.25
    assert out["mean"] == 3
    assert out["clustered_se"] == pytest.approx(1.5, rel=1e-12)
    assert np.isnan(up.paired_difference_clustered([1, 2], [0, 0], ["a", "a"])["clustered_se"])


def test_sign_convention_positive_means_right_hand_arm_better():
    preds = pd.DataFrame({**{f"elo_{o}": [p] for o, p in zip(OUTCOMES, (0.6, 0.2, 0.2))},
                          **{f"elo_f1_{o}": [p] for o, p in zip(OUTCOMES, (0.3, 0.3, 0.4))}})
    losses = up.per_match_losses(preds, ["H"], ("elo", "elo_f1"))
    diff = up._differences(losses, ("elo_f1_minus_elo",), ["d"])["elo_f1_minus_elo"]
    assert diff["log_loss"]["mean"] > 0             # online Elo (the right-hand arm) was better
    assert diff["label"] == "updating_effect"


def test_split_difference():
    assert up.split_difference("poisson_minus_elo_f2") == ("poisson", "elo_f2")
    assert up.split_difference("elo_f1_minus_elo") == ("elo_f1", "elo")
    with pytest.raises(ValueError):
        up.split_difference("poisson")


def test_decomposition_identity_holds_exactly_and_is_enforced():
    rng = np.random.default_rng(4)
    losses = {m: rng.exponential(size=200) for m in up.ARMS}
    assert up.check_decomposition(losses, up.COMPONENTS, up.TOTAL, 1e-12) < 1e-12
    assert up.check_decomposition(losses, up.FULL_GROUP_COMPONENTS, up.SUBTOTAL, 1e-12) < 1e-12
    with pytest.raises(up.DiagnosticCheckError):
        up.check_decomposition(losses, up.COMPONENTS[:2], up.TOTAL, 1e-12)


@pytest.fixture(scope="module")
def scored(league, dfold):
    preds, _ = dfold
    results = league.set_index("match_id").loc[preds.index, "FTR"]
    clusters = preds["Date"].dt.strftime("%Y-%m-%d").to_numpy()
    return up.score_fold(preds, results, clusters, DCFG)


def test_scored_components_add_up_to_the_total(scored):
    for metric in up.METRICS:
        dec = scored["common"]["decomposition"]
        parts = sum(dec[c][metric]["mean"] for c in up.COMPONENTS)
        assert parts == pytest.approx(dec[up.TOTAL][metric]["mean"], abs=1e-12)
        assert scored["common"]["max_identity_residual"][metric] <= 1e-12
        full = scored["full"]
        parts = sum(full["decomposition"][c][metric]["mean"] for c in up.FULL_GROUP_COMPONENTS)
        assert parts == pytest.approx(full["derived_subtotal"][up.SUBTOTAL][metric]["mean"], abs=1e-12)
    assert set(scored["common"]["decomposition"]) == {*up.COMPONENTS, up.TOTAL}
    assert set(scored["full"]["decomposition"]) == set(up.FULL_GROUP_COMPONENTS)
    assert "elo_f2_minus_elo" not in scored["common"]["decomposition"]   # never reported as an effect


def test_groups_and_segments_score_the_registered_models_and_matches(dfold, scored):
    preds, _ = dfold
    n_common = int(preds["in_common"].sum())
    assert scored["full"]["n_matches"] == len(preds) == 30
    assert scored["common"]["n_matches"] == n_common == 20
    assert scored["unseen"]["n_matches"] == 10
    for group in ("full", "common", "unseen"):
        assert tuple(scored[group]["scores"]) == up.GROUP_MODELS[group]
    for group in ("full", "common"):
        seg_n = sum(b["n_matches"] for b in scored[group]["segments"].values())
        assert seg_n == scored[group]["n_matches"]
        for block in scored[group]["segments"].values():
            if block["n_matches"]:
                assert all(r <= 1e-12 for r in block["max_identity_residual"].values())
    assert set(scored["common"]["side_comparisons"]) == {f"{a}_minus_{b}" for a, b in DCFG["side_comparisons"]["pairs"]}
    assert "poisson_minus_frequency_baseline" not in scored["full"]["side_comparisons"]


def test_calibration_summary_is_descriptive_and_complete(scored):
    cal = scored["common"]["calibration"]
    assert set(cal) == set(up.ARMS)
    for block in cal.values():
        assert sum(block["mean_predicted"].values()) == pytest.approx(1.0)
        assert 0 < block["mean_entropy_nats"] <= np.log(3) + 1e-12


def test_scoring_requires_alignment_by_match_id(league, dfold):
    preds, _ = dfold
    results = league.set_index("match_id").loc[preds.index, "FTR"]
    with pytest.raises(ValueError, match="aligned"):
        up.score_fold(preds, results.iloc[::-1], np.zeros(len(preds)), DCFG)


# --- Protocol --------------------------------------------------------------------------------

def test_protocol_agrees_with_frozen_specs_and_code():
    diag.check_protocol(DCFG, VCFG, load_config())   # uses the committed checksum manifest, not the data


@pytest.mark.parametrize("section,key,value", [
    ("season_start_elo", "k", 20),
    ("season_start_elo", "update_home_advantage", 43.0),
    ("season_start_elo", "between_season_regression", "towards_mean"),
    ("elo_f1", "layer", "fit_on_season_start_features"),
    ("elo_f2", "exclude_first_history_season", False),
    ("elo_f2", "logistic_C", 0.5),
    ("decomposition", "components", ["elo_f1_minus_elo", "elo_f2_minus_elo_f1", "poisson_minus_elo_f2"]),
    ("decomposition", "label_elo_f1_minus_elo", "season_start_effect"),
    ("protocol", "targets", ["1718", "2223"]),
    ("protocol", "data_sha256", "0" * 64),
    ("outputs", "in_run_all", True),
])
def test_protocol_mismatch_is_refused(section, key, value):
    tampered = json.loads(json.dumps(DCFG))
    tampered[section][key] = value
    with pytest.raises((diag.ProtocolMismatchError, validation_2425.ProtocolMismatchError)):
        diag.check_protocol(tampered, VCFG, load_config())


def test_diagnostic_arms_are_not_registered_for_dev_test_or_holdout():
    for arm in diag.DIAGNOSTIC_ARMS:
        assert DCFG["arms"][arm] not in REGISTERED_DEV_TEST_SPECS
    tampered = json.loads(json.dumps(DCFG))
    tampered["arms"]["elo_f1"] = "elo_k25_logreg_v1"
    with pytest.raises(diag.ProtocolMismatchError):
        diag.check_protocol(tampered, VCFG, load_config())


def test_registered_decomposition_matches_the_approved_identity():
    dec = DCFG["decomposition"]
    assert dec["identity"] == "poisson - elo = (poisson - elo_f2) + (elo_f2 - elo_f1) + (elo_f1 - elo)"
    assert dec["label_elo_f1_minus_elo"] == "updating_effect"
    assert dec["label_elo_f2_minus_elo_f1"] == "calibration_layer_effect"
    assert dec["derived_subtotal"] == "elo_f2_minus_elo"
    targets = [t for _, t in selection_folds(DCFG["protocol"]["targets"])]
    assert targets == ["1718", "1819", "1920", "2021", "2122", "2425"]


def test_fold_counts_must_match_the_registration():
    good = {"n_full": 380, "n_common": 342, "n_unseen": 38, "goal_models": {"unseen_teams": ["Ipswich"]}}
    diag.check_fold_counts("2425", None, good, DCFG)
    with pytest.raises(diag.ProtocolMismatchError):
        diag.check_fold_counts("2425", None, {**good, "n_common": 306, "n_unseen": 74}, DCFG)
    with pytest.raises(diag.ProtocolMismatchError):
        diag.check_fold_counts("1718", None, good, DCFG)


def test_reproduction_check_refuses_changed_established_predictions(league):
    preds, _ = fold_predictions(league, HISTORY, TARGET, VCFG)
    recorded = preds.assign(fold_target=TARGET)
    assert diag.check_reproduction(TARGET, preds, recorded, DCFG) == 0.0
    changed = recorded.copy()
    first_common = int(np.flatnonzero(changed["in_common"].to_numpy())[0])
    changed.iloc[first_common, changed.columns.get_loc("poisson_H")] += 1e-9
    with pytest.raises(diag.ProtocolMismatchError):
        diag.check_reproduction(TARGET, preds, changed, DCFG)


def test_experiment_is_not_in_run_all():
    from experiments import run_all
    assert diag not in run_all.EXPERIMENTS
    assert "update_policy" not in (PROJECT_ROOT / "experiments" / "run_all.py").read_text(encoding="utf-8")


# --- Real data: fixtures only -------------------------------------------------------------------

def test_registered_group_sizes_hold_on_the_real_fixture_lists():
    """Uses team names of the real fixtures only; no prediction or result is computed."""
    if not PROCESSED_DEV_V2.exists():
        pytest.skip("data/processed/matches_dev_v2.csv missing")
    from eplmodel.data import load_dev_matches
    fixtures = load_dev_matches()[["Season", "HomeTeam", "AwayTeam"]]
    g = DCFG["groups"]
    for history, target in selection_folds(DCFG["protocol"]["targets"]):
        hist = fixtures[fixtures["Season"].isin(history)]
        tgt = fixtures[fixtures["Season"] == target]
        unseen = unseen_teams(hist, tgt)
        assert unseen == g["expected_unseen_teams"][target]
        assert len(tgt) == g["expected_full"][target]
        assert int((~involves_teams(tgt, unseen)).sum()) == g["expected_common"][target]


# --- End-to-end on synthetic data only --------------------------------------------------------------

def test_experiment_runs_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    """Exercises run(), including the reproduction check, with a synthetic league; real data are never loaded."""
    from eplmodel.splits import DEVELOPMENT_SEASONS
    rng = np.random.default_rng(1)
    rows = []
    for season in DEVELOPMENT_SEASONS:
        teams = ["A", "B", "C", "D", "E", "NEW"] if season == "2425" else TEAMS
        start = pd.Timestamp(2000 + int(season[:2]), 8, 15)
        for r, fixtures in enumerate(_round_robin(teams)):
            for home, away in fixtures:
                rows.append({"Date": start + pd.Timedelta(weeks=r), "HomeTeam": home, "AwayTeam": away,
                             "FTHG": int(rng.poisson(1.5)), "FTAG": int(rng.poisson(1.1)), "Season": season})
    synthetic = _with_results(pd.DataFrame(rows))
    recorded = pd.concat([fold_predictions(synthetic, h, t, VCFG)[0].assign(fold_target=t)
                          for h, t in selection_folds(DCFG["protocol"]["targets"])])

    monkeypatch.setattr(diag, "load_dev_matches", lambda: synthetic)
    monkeypatch.setattr(diag, "verify_file", lambda path: True)
    monkeypatch.setattr(diag, "load_recorded_predictions", lambda dcfg: recorded)
    monkeypatch.setattr(diag, "check_fold_counts", lambda target, preds, fitted, dcfg: None)
    monkeypatch.setattr(diag, "write_predictions",
                        lambda name, preds: write_predictions(name, preds, results_dir=tmp_path))
    written = {}
    monkeypatch.setattr(diag, "write_results",
                        lambda name, result, data_path: written.update(data_path=data_path) or
                        write_results(name, result, results_dir=tmp_path, data_path=PROCESSED_MATCHES))

    out = diag.run(write=True)
    assert out["diagnostic_only"] is True
    assert [f["target"] for f in out["folds"]] == ["1718", "1819", "1920", "2021", "2122", "2425"]
    assert out["folds"][-1]["role"] == "primary_validation"
    assert all(f["fitted"]["reproduction_max_abs_diff"] == 0.0 for f in out["folds"])
    assert out["pooled_six_folds"]["unbiased_estimate"] is False
    common = out["primary_2425"]["groups"]["common"]
    assert common["n_matches"] == 20
    assert common["decomposition"]["elo_f1_minus_elo"]["label"] == "updating_effect"
    pooled = out["pooled_six_folds"]["groups"]["common"]
    parts = sum(pooled["decomposition"][c]["log_loss"]["mean"] for c in up.COMPONENTS)
    assert parts == pytest.approx(pooled["decomposition"][up.TOTAL]["log_loss"]["mean"], abs=1e-12)
    assert written["data_path"] == PROCESSED_DEV_V2
    path = tmp_path / "update_policy_diagnostic" / "predictions.csv"
    metrics = json.loads((tmp_path / "update_policy_diagnostic" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["results"]["predictions"]["sha256"] == content_sha256(path)
    assert metrics["results"]["predictions"]["n_rows"] == 6 * 30
