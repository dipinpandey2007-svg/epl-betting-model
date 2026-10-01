"""Fold validation harness (protocol validation_2425_v1).

Leakage, eligibility and ordering tests use a synthetic league. Tests on the real data use
only the five historical folds or the 2024-25 *fixture list*: no test generates a 2024-25
prediction.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss as sklearn_log_loss

from eplmodel.config import load_config
from eplmodel.constants import OUTCOMES
from eplmodel.data.checksums import content_sha256, load_manifest
from eplmodel.evaluation.alignment import involves_teams
from eplmodel.evaluation.metrics import (
    multiclass_brier,
    multiclass_log_loss,
    per_match_brier,
    per_match_log_loss,
)
from eplmodel.evaluation.validation import (
    GROUP_MODELS,
    MODELS,
    elo_fold,
    fold_data,
    fold_predictions,
    outcomes_for,
    paired_difference,
    prob_columns,
    score_groups,
    unseen_teams,
)
from eplmodel.models.scoreline import independent_poisson_grid
from eplmodel.paths import PROCESSED_DEV_V2, PROCESSED_MATCHES
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import DevTestAccessError, HoldoutAccessError, SplitAccessError, selection_folds
from experiments import validation_2425

VCFG = load_config(validation_2425.VALIDATION_CONFIG)
HISTORY = ("1415", "1516", "1617")
TARGET = "1718"
TEAMS = ["A", "B", "C", "D", "E", "F"]


def _round_robin(teams):
    """Double round robin (circle method): every team plays once per round."""
    teams = list(teams)
    rounds = []
    for _ in range(len(teams) - 1):
        rounds.append([(teams[i], teams[-1 - i]) for i in range(len(teams) // 2)])
        teams = [teams[0], teams[-1], *teams[1:-1]]
    return rounds + [[(a, h) for h, a in r] for r in rounds]


def synthetic_league(seed=0, target_teams=("A", "B", "C", "D", "E", "NEW"), extra_seasons=("1819",)):
    """Six-team seasons; the target season replaces F with an unseen team NEW."""
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip([*TEAMS, "NEW"], [0.4, 0.2, 0.0, 0.0, -0.2, -0.4, -0.3])}
    rows = []
    for season in (*HISTORY, TARGET, *extra_seasons):
        teams = list(target_teams) if season in (TARGET, *extra_seasons) else TEAMS
        start = pd.Timestamp(2000 + int(season[:2]), 8, 15)
        for r, fixtures in enumerate(_round_robin(teams)):
            for home, away in fixtures:
                hg = rng.poisson(np.exp(0.3 + strength[home] - strength[away]))
                ag = rng.poisson(np.exp(0.1 + strength[away] - strength[home]))
                rows.append({"Date": start + pd.Timedelta(weeks=r), "HomeTeam": home, "AwayTeam": away,
                             "FTHG": int(hg), "FTAG": int(ag), "Season": season})
    df = pd.DataFrame(rows).sort_values("Date", kind="stable").reset_index(drop=True)
    return _with_results(df)


def _with_results(df):
    df = df.copy()
    df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    df["match_id"] = df["Date"].dt.strftime("%Y-%m-%d") + "_" + df["HomeTeam"] + "_" + df["AwayTeam"]
    return df


def _reverse_scores(df, mask):
    """Swap home and away goals where mask is true (changes outcomes, keeps fixtures)."""
    df = df.copy()
    df.loc[mask, ["FTHG", "FTAG"]] = df.loc[mask, ["FTAG", "FTHG"]].to_numpy() + np.array([2, 0])
    return _with_results(df)


@pytest.fixture(scope="module")
def league():
    return synthetic_league()


@pytest.fixture(scope="module")
def fold(league):
    return fold_predictions(league, HISTORY, TARGET, VCFG)


# --- No target-season leakage into fitting ----------------------------------------

def test_fitting_ignores_target_season_outcomes(league, fold):
    preds, fitted = fold
    changed = _reverse_scores(league, league["Season"] == TARGET)
    preds2, fitted2 = fold_predictions(changed, HISTORY, TARGET, VCFG)
    assert fitted2["elo"] == fitted["elo"]
    assert fitted2["frequency_baseline"] == fitted["frequency_baseline"]
    for key in ("poisson_is_home_coef", "rho", "valid_rho_range", "unseen_teams"):
        assert fitted2["goal_models"][key] == fitted["goal_models"][key]
    assert fitted2["goal_models"]["rho_grid"].equals(fitted["goal_models"]["rho_grid"])
    static = [c for m in ("frequency_baseline", "poisson", "dixon_coles") for c in prob_columns(m)]
    pd.testing.assert_frame_equal(preds2[static], preds[static])


def test_later_seasons_never_enter_a_fold(league, fold):
    preds, fitted = fold
    changed = _reverse_scores(league, league["Season"] == "1819")
    preds2, fitted2 = fold_predictions(changed, HISTORY, TARGET, VCFG)
    pd.testing.assert_frame_equal(preds2, preds)
    assert set(fold_data(league, HISTORY, TARGET)["Season"]) == {*HISTORY, TARGET}


def test_history_outcomes_do_change_the_fit(league, fold):
    """Sensitivity check for the two tests above: the fit does respond to history outcomes."""
    changed = _reverse_scores(league, league["Season"] == HISTORY[-1])
    _, fitted2 = fold_predictions(changed, HISTORY, TARGET, VCFG)
    assert fitted2["elo"]["logistic_coef"] != fold[1]["elo"]["logistic_coef"]


@pytest.mark.parametrize("round_index", [0, 3, 7])
def test_prediction_unchanged_by_outcomes_on_or_after_its_date(league, fold, round_index):
    preds, _ = fold
    dates = sorted(league.loc[league["Season"] == TARGET, "Date"].unique())
    cutoff = dates[round_index]
    changed = _reverse_scores(league, (league["Season"] == TARGET) & (league["Date"] >= cutoff))
    preds2, _ = fold_predictions(changed, HISTORY, TARGET, VCFG)
    upto = preds["Date"] <= cutoff
    pd.testing.assert_frame_equal(preds2[upto], preds[upto])
    # Sensitivity: Elo is online, so predictions after the cutoff do respond to the changed results.
    after = preds["Date"] > cutoff
    assert not np.allclose(preds2.loc[after, prob_columns("elo")], preds.loc[after, prob_columns("elo")])


# --- Eligibility rules and groups ------------------------------------------------------

def test_unseen_team_is_excluded_from_the_common_group_only(league, fold):
    preds, fitted = fold
    data = fold_data(league, HISTORY, TARGET)
    target = data[data["Season"] == TARGET]
    assert unseen_teams(data[data["Season"].isin(HISTORY)], target) == ["NEW"] == fitted["goal_models"]["unseen_teams"]
    involves_new = involves_teams(target, ["NEW"]).to_numpy()
    assert (preds["in_common"].to_numpy() == ~involves_new).all()
    assert (fitted["n_full"], fitted["n_common"]) == (30, 20)
    goal_cols = prob_columns("poisson") + prob_columns("dixon_coles")
    assert preds.loc[~preds["in_common"], goal_cols].isna().all().all()
    assert preds.loc[preds["in_common"], goal_cols].notna().all().all()
    full_cols = prob_columns("elo") + prob_columns("frequency_baseline")
    assert preds[full_cols].notna().all().all()


def test_new_team_starts_elo_at_the_initial_rating(fold):
    preds, _ = fold
    first = preds[(preds["HomeTeam"] == "NEW") | (preds["AwayTeam"] == "NEW")].iloc[0]
    rating = first["elo_rating_home"] if first["HomeTeam"] == "NEW" else first["elo_rating_away"]
    assert rating == VCFG["elo"]["initial_rating"]


def test_groups_score_exactly_the_registered_models_and_matches(league, fold):
    preds, _ = fold
    out = score_groups(preds, outcomes_for(league, preds.index), ["elo", "frequency_baseline"])
    assert GROUP_MODELS == {"full": ("elo", "frequency_baseline"), "common": MODELS}
    assert tuple(VCFG["groups"]["full_models"]) == GROUP_MODELS["full"]
    assert tuple(VCFG["groups"]["common_models"]) == GROUP_MODELS["common"]
    assert set(out["full"]["scores"]) == {"elo", "frequency_baseline"}
    assert set(out["common"]["scores"]) == set(MODELS)
    assert {s["n_matches"] for s in out["full"]["scores"].values()} == {30}
    assert {s["n_matches"] for s in out["common"]["scores"].values()} == {20}
    assert set(out["full"]["paired_differences"]) == {"frequency_baseline_minus_elo", "elo_minus_frequency_baseline"}
    assert len(out["common"]["paired_differences"]) == 6


def test_scoring_requires_alignment_by_match_id(league, fold):
    preds, _ = fold
    with pytest.raises(ValueError, match="aligned"):
        score_groups(preds, outcomes_for(league, preds.index[::-1]), ["elo"])


# --- Probability order and row sums ----------------------------------------------------

def test_probability_rows_sum_to_one_in_hda_order(fold):
    preds, _ = fold
    for model in MODELS:
        cols = prob_columns(model)
        assert cols == [f"{model}_{o}" for o in ("H", "D", "A")]
        probs = preds[cols].dropna().to_numpy()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)
        assert ((probs > 0) & (probs < 1)).all()


def test_goal_model_home_column_is_home_win(fold):
    preds, _ = fold
    row = preds[preds["in_common"]].iloc[0]
    grid = independent_poisson_grid(row["expected_goals_home"], row["expected_goals_away"])
    p_home = np.tril(grid, k=-1).sum() / grid.sum()
    assert row["poisson_H"] == pytest.approx(p_home, abs=1e-12)


def test_scores_match_sklearn_after_explicit_reordering(league, fold):
    preds, _ = fold
    results = outcomes_for(league, preds.index)
    probs = preds[prob_columns("elo")].to_numpy()
    ours = multiclass_log_loss(results, probs)
    theirs = sklearn_log_loss(results, probs[:, [2, 1, 0]], labels=["A", "D", "H"])  # sklearn order is (A, D, H)
    assert ours == pytest.approx(theirs, abs=1e-12)


def test_per_match_metrics_average_to_the_established_metrics(league, fold):
    preds, _ = fold
    results = outcomes_for(league, preds.index)
    probs = preds[prob_columns("frequency_baseline")].to_numpy()
    assert per_match_log_loss(results, probs).mean() == pytest.approx(multiclass_log_loss(results, probs), abs=1e-15)
    assert per_match_brier(results, probs).mean() == pytest.approx(multiclass_brier(results, probs), abs=1e-15)


def test_paired_difference():
    out = paired_difference(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.0]))
    assert out["n_matches"] == 3 and out["mean"] == pytest.approx(1 / 6)
    assert out["sd"] == pytest.approx(np.std([-0.5, 0.0, 1.0], ddof=1))
    assert out["naive_se"] == pytest.approx(out["sd"] / np.sqrt(3))


def test_predictions_contain_no_results(fold):
    preds, _ = fold
    assert not {"FTR", "FTHG", "FTAG"} & set(preds.columns)


# --- Season roles and the holdout ------------------------------------------------------

@pytest.mark.parametrize("history,target,error", [
    (("2324", "2425"), "2526", HoldoutAccessError),
    (("2425", "2526"), "2627", HoldoutAccessError),
    (("2122",), "2223", DevTestAccessError),
    (("2122", "2223"), "2324", DevTestAccessError),
    (("1718", "1819"), "1819", SplitAccessError),
])
def test_folds_refuse_holdout_exposed_and_non_preceding_seasons(league, history, target, error):
    with pytest.raises(error):
        fold_data(league, history, target)


def test_fold_refuses_a_match_table_containing_holdout_rows(league):
    sealed = league.copy()
    sealed.loc[sealed.index[-1], "Season"] = "2526"
    with pytest.raises(HoldoutAccessError):
        fold_data(sealed, HISTORY, TARGET)


def test_registered_targets_are_the_selection_folds():
    targets = [t for _, t in selection_folds(VCFG["protocol"]["targets"])]
    assert targets == ["1718", "1819", "1920", "2021", "2122", "2425"]
    assert VCFG["protocol"]["primary_target"] == "2425"
    assert VCFG["protocol"]["previously_used_targets"] == targets[:5]
    assert VCFG["protocol"]["pooled_is_unbiased"] is False


# --- Pre-registration checks -------------------------------------------------------------

def test_protocol_agrees_with_frozen_specs():
    validation_2425.check_protocol(VCFG, load_config())


@pytest.mark.parametrize("section,key,value", [
    ("elo", "k", 20),
    ("elo", "update_home_advantage", 43.0),
    ("poisson", "max_goals", 8),
    ("dixon_coles", "rho_grid_step", 0.01),
    ("dixon_coles", "rho_estimation", "fixed"),
    ("protocol", "targets", ["1718", "2223"]),
    ("protocol", "data_sha256", "0" * 64),
])
def test_protocol_mismatch_is_refused(section, key, value):
    tampered = json.loads(json.dumps(VCFG))
    tampered[section][key] = value
    with pytest.raises(validation_2425.ProtocolMismatchError):
        validation_2425.check_protocol(tampered, load_config())


def test_primary_group_sizes_must_match_the_registration():
    good = {"n_full": 380, "n_common": 342, "goal_models": {"unseen_teams": ["Ipswich"]}}
    validation_2425.check_primary_counts(good, VCFG)
    with pytest.raises(validation_2425.ProtocolMismatchError):
        validation_2425.check_primary_counts({**good, "n_common": 306}, VCFG)


# --- Provenance and checksums ------------------------------------------------------------

def test_write_predictions_records_checksum(tmp_path, fold):
    preds, _ = fold
    info = write_predictions("demo", preds, results_dir=tmp_path)
    path = tmp_path / "demo" / "predictions.csv"
    assert info["sha256"] == content_sha256(path)
    assert info["n_rows"] == len(preds) == len(pd.read_csv(path))


def test_results_provenance_records_the_dataset_used(tmp_path):
    if not (PROCESSED_DEV_V2.exists() and PROCESSED_MATCHES.exists()):
        pytest.skip("processed datasets missing")
    manifest = load_manifest()
    for data_path, key in ((PROCESSED_DEV_V2, "data/processed/matches_dev_v2.csv"),
                           (None, "data/processed/matches.csv")):  # default stays dataset v1
        kwargs = {} if data_path is None else {"data_path": data_path}
        path = write_results("demo", {"x": 1}, results_dir=tmp_path, **kwargs)
        prov = json.loads(path.read_text(encoding="utf-8"))["provenance"]
        assert prov["data_file"] == key
        assert prov["data_sha256"] == manifest[key]


# --- Real data: five historical folds and the 2024-25 fixture list only ----------------------

@pytest.fixture(scope="module")
def dev_v2():
    if not PROCESSED_DEV_V2.exists():
        pytest.skip("data/processed/matches_dev_v2.csv missing")
    from eplmodel.data import load_dev_matches
    return load_dev_matches()


def test_ipswich_is_the_only_unseen_team_in_2425(dev_v2):
    """Uses the 2024-25 fixtures only (team names); no prediction or result is computed."""
    history, target = selection_folds()[-1]
    assert target == "2425"
    data = fold_data(dev_v2, history, target)
    tgt = data[data["Season"] == target][["HomeTeam", "AwayTeam"]]
    unseen = unseen_teams(data[data["Season"].isin(history)], tgt)
    assert unseen == VCFG["groups"]["expected_2425_unseen_teams"] == ["Ipswich"]
    assert len(tgt) == VCFG["groups"]["expected_2425_full"] == 380
    assert int((~involves_teams(tgt, unseen)).sum()) == VCFG["groups"]["expected_2425_common"] == 342


@pytest.mark.golden
def test_elo_reproduces_the_recorded_k25_fold_results(dev_v2, matches, golden):
    """Folds 2017-18..2021-22 reproduce Experiment 2a (K = 25) on dataset v1 and on dev_v2.

    The fold data stop at the target season, so no 2024-25 row is used. dev_v2 differs from v1
    only in the order of same-date rows, which moves the result by ~1e-13 (summation order).
    """
    g = golden["elo_k_selection"]
    recorded = dict(zip(g["validation_seasons"], g["fold_log_loss"]["25"]))
    for history, target in selection_folds()[:5]:
        for table in (matches, dev_v2):
            data = fold_data(table, history, target)
            assert "2425" not in set(data["Season"])
            preds, _ = elo_fold(data, history, target, VCFG["elo"])
            ll = multiclass_log_loss(outcomes_for(data, preds.index), preds[prob_columns("elo")].to_numpy())
            assert ll == pytest.approx(recorded[target], abs=1e-9, rel=0)


# --- End-to-end smoke test of the experiment on synthetic data only ---------------------------

def test_experiment_runs_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    """Exercises run() and its outputs with a synthetic league; the real 2024-25 data are never loaded."""
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

    monkeypatch.setattr(validation_2425, "load_dev_matches", lambda: synthetic)
    monkeypatch.setattr(validation_2425, "verify_file", lambda path: True)
    monkeypatch.setattr(validation_2425, "check_primary_counts", lambda fitted, vcfg: None)
    monkeypatch.setattr(validation_2425, "write_predictions",
                        lambda name, preds: write_predictions(name, preds, results_dir=tmp_path))
    # Provenance checksums its data file: use a placeholder under tmp_path, not the git-ignored dataset
    # (absent in CI). Provenance records that path relative to the project root, hence the second patch.
    placeholder = tmp_path / "placeholder_matches.csv"
    placeholder.write_text("Date,HomeTeam,AwayTeam\n", encoding="utf-8")
    from eplmodel.reporting import results as reporting
    monkeypatch.setattr(reporting, "PROJECT_ROOT", tmp_path)
    written = {}
    monkeypatch.setattr(validation_2425, "write_results",
                        lambda name, result, data_path: written.update(data_path=data_path) or
                        write_results(name, result, results_dir=tmp_path, data_path=placeholder))

    out = validation_2425.run(write=True)
    assert [f["target"] for f in out["folds"]] == ["1718", "1819", "1920", "2021", "2122", "2425"]
    assert [f["role"] for f in out["folds"]][-1] == "primary_validation"
    assert all(f["role"].startswith("historical_fold") for f in out["folds"][:5])
    assert out["pooled_six_folds"]["unbiased_estimate"] is False
    assert out["primary_2425"]["groups"]["common"]["scores"]["poisson"]["n_matches"] == 20
    assert written["data_path"] == PROCESSED_DEV_V2                    # provenance points at dev_v2
    metrics = json.loads((tmp_path / "validation_2425" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["provenance"]["data_sha256"] == content_sha256(placeholder)
    assert metrics["results"]["predictions"]["sha256"] == content_sha256(tmp_path / "validation_2425" / "predictions.csv")
    assert metrics["results"]["predictions"]["n_rows"] == 6 * 30
