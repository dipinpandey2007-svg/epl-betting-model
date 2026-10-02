"""Shots-information blend (protocol shots_information_v1, Experiment 16): implementation tests.

Every fit uses the synthetic league of test_validation with synthetic shot counts added (target 1718 with an
unseen team NEW, a later season 1819). No test reads real data or scores Experiment 16.
"""

import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from eplmodel.config import load_config
from eplmodel.data import shots as shots_data
from eplmodel.evaluation import online_poisson as op
from eplmodel.evaluation import shots_selection as sel
from eplmodel.evaluation.folds import build_fold
from eplmodel.evaluation.forecasts import check_forecast_frame, prob_columns
from eplmodel.models import shots as sm
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.paths import PROJECT_ROOT
from eplmodel.splits import HoldoutAccessError, SplitAccessError
from experiments import shots_information as exp
from test_validation import HISTORY, TARGET, _reverse_scores, synthetic_league

H = 730.0
OMEGAS = (0.0, 0.25, 0.5, 0.75, 1.0)
STRENGTH = {"A": 0.4, "B": 0.2, "C": 0.0, "D": 0.0, "E": -0.2, "F": -0.4, "NEW": -0.3}
FROZEN_CONFIG_CANONICAL_SHA256 = "a30033be8114de53bdcd9ca489e4d8419e1dd844991f3ddb02e67fe1da93d2fa"


def with_shots(df, seed=7):
    rng = np.random.default_rng(seed)
    df = df.copy()
    sh = np.array([STRENGTH[t] for t in df["HomeTeam"]]); sa = np.array([STRENGTH[t] for t in df["AwayTeam"]])
    df["HST"] = rng.poisson(np.exp(1.45 + sh - sa)).astype(float)
    df["AST"] = rng.poisson(np.exp(1.25 + sa - sh)).astype(float)
    df["HS"] = df["HST"] + rng.poisson(np.exp(1.9 + 0.5 * (sh - sa)))
    df["AS"] = df["AST"] + rng.poisson(np.exp(1.7 + 0.5 * (sa - sh)))
    return df


@pytest.fixture(scope="module")
def league():
    return with_shots(synthetic_league())


@pytest.fixture(scope="module")
def fold(league):
    return build_fold(league, HISTORY, TARGET)


@pytest.fixture(scope="module")
def run(fold):
    return sm.online_fold(fold.history_rows, fold.target_rows, OMEGAS, H, 10, 100, 1000)


def _cols(arm, w):
    return prob_columns(sm.arm_name(arm, w))


# --- Shots table ------------------------------------------------------------------------------------------

def _raw(tmp_path, season="1718", rows=None):
    rows = rows or [{"Date": "12/08/2017", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 0, "FTR": "H",
                     "HS": 12, "AS": 7, "HST": 5, "AST": 2},
                    {"Date": "13/08/2017", "HomeTeam": "C", "AwayTeam": "D", "FTHG": 1, "FTAG": 1, "FTR": "D",
                     "HS": "x", "AS": -1, "HST": 3.5, "AST": None}]
    pd.DataFrame(rows).to_csv(tmp_path / f"E0_{season}.csv", index=False)


def test_shots_reader_maps_columns_reads_no_result_and_marks_invalid(tmp_path):
    _raw(tmp_path)
    out = shots_data.read_season_shots("1718", ["1718"], raw_dir=tmp_path)
    assert list(out["match_id"]) == ["2017-08-12_A_B", "2017-08-13_C_D"]
    assert out.loc[0, ["HS", "AS", "HST", "AST"]].tolist() == [12.0, 7.0, 5.0, 2.0]
    assert out.loc[1, ["HS", "AS", "HST", "AST"]].isna().all()          # text, negative, fractional, missing
    assert not {"FTHG", "FTAG", "FTR"} & set(out.columns)


@pytest.mark.parametrize("season,error", [("2526", HoldoutAccessError), ("2627", HoldoutAccessError),
                                          ("2223", SplitAccessError)])
def test_shots_reader_refuses_holdout_and_unallowed_seasons(tmp_path, season, error):
    _raw(tmp_path, season)
    with pytest.raises(error):
        shots_data.read_season_shots(season, ["1718"], raw_dir=tmp_path)


def test_attach_shots_requires_every_match(league):
    shots = league[["match_id", "HS", "AS", "HST", "AST"]]
    joined = shots_data.attach_shots(league.drop(columns=["HS", "AS", "HST", "AST"]), shots)
    assert len(joined) == len(league)
    with pytest.raises(ValueError):
        shots_data.attach_shots(league.drop(columns=["HS", "AS", "HST", "AST"]), shots.iloc[1:])


# --- Centring and blend -------------------------------------------------------------------------------------

def test_centring_is_an_exact_reparameterisation(fold):
    model = PoissonGoalModel().fit(fold.history_rows)
    centring = list("ABCDE")
    c = sm.centred(model, centring)
    assert abs(c.attack.loc[centring].sum()) < 1e-12 and abs(c.defence.loc[centring].sum()) < 1e-12
    home, away = ["A", "C", "E"], ["B", "D", "A"]
    lam, mu = model.predict_rates(home, away)
    lam2, mu2 = sm.blended_rates(c, c, 0.3, home, away)            # blend of a model with itself = the model
    assert np.max(np.abs(lam - lam2)) < 1e-12 and np.max(np.abs(mu - mu2)) < 1e-12


def test_blend_keeps_goal_level_and_home_advantage():
    idx = pd.Index(["A", "B"])
    goal = sm.Centred(0.1, 0.25, pd.Series([0.2, -0.2], idx), pd.Series([-0.1, 0.1], idx))
    shot = sm.Centred(9.0, 9.0, pd.Series([0.4, -0.4], idx), pd.Series([-0.3, 0.3], idx))
    lam, mu = sm.blended_rates(goal, shot, 0.5, ["A"], ["B"])
    assert lam[0] == pytest.approx(np.exp(0.1 + 0.25 + 0.5 * 0.2 + 0.5 * 0.4 + 0.5 * 0.1 + 0.5 * 0.3))
    assert mu[0] == pytest.approx(np.exp(0.1 + 0.5 * -0.2 + 0.5 * -0.4 + 0.5 * -0.1 + 0.5 * -0.3))


def test_omega_labels():
    assert sm.omega_label(0.25) == "w025" and sm.arm_name("b1", 1.0) == "b1_w100"
    with pytest.raises(ValueError):
        sm.omega_label(0.333)


# --- Online fold: baseline, cutoff, missing data, coverage -------------------------------------------------

def test_omega_zero_reproduces_the_experiment_13_online_arm(league, run):
    preds, _ = run
    ref, _ = op.online_fold_predictions(league, HISTORY, TARGET, H, 10, 100, 1000)
    assert list(preds.index) == list(ref.index)
    for arm in sm.SHOT_ARMS:
        assert np.max(np.abs(preds[_cols(arm, 0.0)].to_numpy() - ref[prob_columns(op.ONLINE_ARM)].to_numpy())) < 1e-12


def test_positive_omega_uses_shot_information(run):
    preds, _ = run
    assert np.max(np.abs(preds[_cols("b1", 1.0)].to_numpy() - preds[_cols("b1", 0.0)].to_numpy())) > 1e-4
    assert np.max(np.abs(preds[_cols("b1", 0.5)].to_numpy() - preds[_cols("s1", 0.5)].to_numpy())) > 1e-6


def test_unseen_team_matches_are_neither_fitted_nor_scored(fold, run):
    preds, record = run
    assert not ((preds["HomeTeam"] == "NEW") | (preds["AwayTeam"] == "NEW")).any()
    assert record["unseen_teams"] == ["NEW"] and record["n_unseen_excluded"] == record["n_full"] - record["n_common"]
    assert record["centring_set"] == list("ABCDE")


@pytest.mark.parametrize("round_index", [3, 6])
def test_forecasts_use_only_strictly_earlier_goals_and_shots(league, run, round_index):
    preds, _ = run
    cut = sorted(preds["Date"].unique())[round_index]
    later = (league["Season"] == TARGET) & (league["Date"] >= cut)
    changed = _reverse_scores(league, later)
    changed.loc[later, ["HST", "AST", "HS", "AS"]] = changed.loc[later, ["AST", "HST", "AS", "HS"]].to_numpy() + 3
    f2 = build_fold(changed, HISTORY, TARGET)
    p2, _ = sm.online_fold(f2.history_rows, f2.target_rows, OMEGAS, H, 10, 100, 1000)
    cols = [c for a in sm.SHOT_ARMS for w in OMEGAS for c in _cols(a, w)]
    upto = (preds["Date"] <= cut).to_numpy()
    assert np.array_equal(p2.loc[upto, cols].to_numpy(), preds.loc[upto, cols].to_numpy())
    assert not np.allclose(p2.loc[~upto, cols].to_numpy(), preds.loc[~upto, cols].to_numpy())


def test_earlier_shots_change_later_shot_forecasts_but_not_the_baseline(league, run):
    preds, _ = run
    first = preds["Date"].min()
    hist = league["Season"].isin(HISTORY)
    changed = league.copy()
    changed.loc[hist, "HST"] = changed.loc[hist, "HST"] + 2
    f2 = build_fold(changed, HISTORY, TARGET)
    p2, _ = sm.online_fold(f2.history_rows, f2.target_rows, OMEGAS, H, 10, 100, 1000)
    on_first = (preds["Date"] == first).to_numpy()
    assert np.array_equal(p2.loc[on_first, _cols("b1", 0.0)].to_numpy(), preds.loc[on_first, _cols("b1", 0.0)].to_numpy())
    assert not np.allclose(p2.loc[on_first, _cols("b1", 1.0)].to_numpy(), preds.loc[on_first, _cols("b1", 1.0)].to_numpy())
    assert np.array_equal(p2[_cols("s1", 1.0)].to_numpy(), preds[_cols("s1", 1.0)].to_numpy())   # S1 ignores HST


def test_shot_fit_uses_the_goal_fit_rows_and_weights(fold):
    rows = op.fit_set(fold.history_rows, fold.target_rows[fold.target_rows["HomeTeam"] != "NEW"], fold.target_rows["Date"].max())
    _, w_goal, _ = op.checked_fit(rows, H, 100, 1000)
    _, w_shot, _ = op.checked_fit(sm.shot_frame(rows, ("HST", "AST")), H, 100, 1000)
    assert np.array_equal(w_goal, w_shot)
    assert np.allclose(w_goal, 2.0 ** (-(rows["Date"].max() - rows["Date"]).dt.days.to_numpy() / H))


def test_missing_shot_values_leave_only_that_arms_shot_fit(league, run):
    preds, _ = run
    changed = league.copy()
    drop = changed.index[changed["Season"] == "1516"][:5]
    changed.loc[drop, "HST"] = np.nan
    f2 = build_fold(changed, HISTORY, TARGET)
    p2, rec = sm.online_fold(f2.history_rows, f2.target_rows, OMEGAS, H, 10, 100, 1000)
    assert rec["n_rows_excluded_missing_shots"] == {"b1": 5, "s1": 0}
    assert np.array_equal(p2[_cols("s1", 1.0)].to_numpy(), preds[_cols("s1", 1.0)].to_numpy())
    assert np.array_equal(p2[_cols("b1", 0.0)].to_numpy(), preds[_cols("b1", 0.0)].to_numpy())   # goals unaffected
    assert not np.allclose(p2[_cols("b1", 1.0)].to_numpy(), preds[_cols("b1", 1.0)].to_numpy())


def test_shot_model_has_one_global_home_term(run):
    _, record = run
    for f in record["fits"]:
        assert np.isfinite(f["b1"]["shot_home_advantage"]) and np.isfinite(f["s1"]["shot_home_advantage"])
        assert f["b1"]["shot_home_advantage"] > 0                     # simulated home edge in shots on target


def test_forecasts_are_valid_complete_and_deterministic(fold, run):
    preds, record = run
    check_forecast_frame(preds, [sm.arm_name(a, w) for a in sm.SHOT_ARMS for w in OMEGAS], atol=1e-12)
    tgt = fold.target_rows
    common = tgt[(tgt["HomeTeam"] != "NEW") & (tgt["AwayTeam"] != "NEW")]
    assert record["n_fits"] == common["Date"].nunique()                  # one refit per common-group date
    assert record["all_converged"] and record["n_retried"] == 0
    again, _ = sm.online_fold(fold.history_rows, fold.target_rows, OMEGAS, H, 10, 100, 1000)
    pd.testing.assert_frame_equal(again, preds, check_exact=True)


def test_grid_must_start_at_the_baseline(fold):
    with pytest.raises(ValueError):
        sm.online_fold(fold.history_rows, fold.target_rows, (0.5, 1.0), H, 10, 100, 1000)


# --- Selection and evidence rules -----------------------------------------------------------------------------

def _losses(means_by_target, n=60, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    base = {t: rng.normal(1.0, noise, n) for t in means_by_target}
    return ({t: {w: base[t] + m for w, m in means.items()} for t, means in means_by_target.items()},
            {t: np.array([f"{t}_{i // 2}" for i in range(n)]) for t in means_by_target})


def test_one_se_rule_picks_the_smallest_omega_within_one_se():
    # Differences to omega_min = 0.5 alternate delta +/- 0.1 per match (n = 100, one match per cluster), so their
    # clustered SE is about 0.1 / sqrt(100) = 0.01: delta 0.005 is within one SE, delta 0.05 is not.
    n = 100
    base = np.linspace(0.8, 1.2, n)
    wiggle = 0.1 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    deltas = {0.0: 0.05, 0.25: 0.005, 0.5: 0.0, 0.75: 0.004, 1.0: 0.05}
    losses = {"1718": {w: base + (0.0 if w == 0.5 else d + wiggle) for w, d in deltas.items()}}
    out = sel.select_omega(losses, {"1718": np.arange(n)}, 1.0)
    table = {row["omega"]: row for row in out["table"]}
    assert out["omega_min"] == 0.5
    assert table[0.25]["within_one_se"] and table[0.75]["within_one_se"] and not table[0.0]["within_one_se"]
    assert out["omega_selected"] == 0.25                      # the smallest omega within one SE


def test_exact_ties_and_flat_curves_select_the_baseline():
    losses, cl = _losses({"1718": {w: 0.0 for w in OMEGAS}})
    assert sel.select_omega(losses, cl, 1.0)["omega_selected"] == 0.0


def test_nested_selection_uses_only_earlier_targets():
    losses, cl = _losses({"1718": {w: (0.0 if w == 1.0 else 0.05) for w in OMEGAS},
                          "1819": {w: 0.0 for w in OMEGAS},
                          "1920": {w: 0.0 for w in OMEGAS}})
    nested = sel.nested_selection(losses, cl, ["1819", "1920"], 1.0)
    assert nested["1819"]["targets"] == ["1718"] and nested["1920"]["targets"] == ["1718", "1819"]
    assert nested["1819"]["omega_selected"] == 1.0


def test_selection_refuses_mismatched_grids():
    losses, cl = _losses({"1718": {0.0: 0.0, 0.5: 0.0}, "1819": {0.0: 0.0}})
    with pytest.raises(ValueError):
        sel.select_omega(losses, cl, 1.0)


def test_criterion_s_readings():
    rng = np.random.default_rng(0)
    cl = {t: np.arange(300) for t in ("a", "b", "c", "d")}
    mk = lambda m: {t: {"log_loss": rng.normal(m, 0.01, 300), "brier": rng.normal(m, 0.01, 300)} for t in cl}
    assert sel.criterion_s(mk(-0.01), cl, 0.002, 2.0, 3)["reading"] == sel.HELPS
    assert sel.criterion_s(mk(0.01), cl, 0.002, 2.0, 3)["reading"] == sel.HURTS
    assert sel.criterion_s(mk(-0.001), cl, 0.002, 2.0, 3)["reading"] == sel.NEITHER


# --- Experiment orchestration on synthetic data (no real data, nothing written) --------------------------------

def test_evaluate_runs_end_to_end_on_two_synthetic_folds(league):
    staged = []
    for history, target in ((HISTORY, "1718"), ((*HISTORY, "1718"), "1819")):
        f = build_fold(league, history, target)
        preds, rec = sm.online_fold(f.history_rows, f.target_rows, OMEGAS, H, 10, 100, 1000)
        staged.append({"target": target, "history": history, "fold": f, "preds": preds, "record": rec})
    out = exp.evaluate(staged, league, OMEGAS, ["1819"], 1.0, 0.002, 2.0, 1)
    for arm in sm.SHOT_ARMS:
        assert out["selection"][arm]["omega_selected"] in OMEGAS
        assert out["criterion_s"][arm]["reading"] in (sel.HELPS, sel.HURTS, sel.NEITHER)
        assert set(out["grid"][arm]["1718"]) == {sm.omega_label(w) for w in OMEGAS}
    assert out["grid"]["b1"]["1718"]["w000"] == out["grid"]["s1"]["1718"]["w000"]   # both arms share B0
    assert {"all", "returning", "continuing"} <= set(out["descriptive_b1_star_minus_b0"])


# --- Protocol guards -----------------------------------------------------------------------------------------------

def test_config_equals_the_frozen_preregistration_without_git_history():
    cfg = dict(load_config(exp.SHOTS_CONFIG))
    cfg.pop("locked")
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == FROZEN_CONFIG_CANONICAL_SHA256


def test_config_and_document_equal_the_preregistration_commit():
    if subprocess.run(["git", "cat-file", "-e", f"{exp.PREREG_COMMIT}^{{commit}}"], cwd=PROJECT_ROOT,
                      capture_output=True).returncode != 0:
        pytest.skip("pre-registration commit not in this checkout (shallow clone)")
    assert exp.check_frozen(load_config(exp.SHOTS_CONFIG))["prereg_commit"] == exp.PREREG_COMMIT


def test_protocol_not_yet_run_and_not_in_run_all():
    from experiments import run_all
    assert load_config(exp.SHOTS_CONFIG)["locked"]["status"] == "not_run"
    assert not (PROJECT_ROOT / "results" / "shots_information_historical").exists()
    assert exp not in run_all.EXPERIMENTS
