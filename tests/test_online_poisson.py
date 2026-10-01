"""Online vs frozen time-weighted Poisson (protocol online_tw_poisson_diagnostic_v1).

Every fit in this file uses synthetic data. The only real-data test reads the fixture list (dates and
team names) to check the registered group sizes and online-fit counts; no test fits any model on real
data, and no test generates a 2024-25 prediction.
"""

import json

import numpy as np
import pandas as pd
import pytest

from eplmodel.config import load_config
from eplmodel.evaluation import online_poisson as op
from eplmodel.evaluation import time_weighting as tw
from eplmodel.evaluation import update_policy as up
from eplmodel.evaluation.alignment import involves_teams
from eplmodel.evaluation.validation import fold_data, prob_columns, unseen_teams
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.models.time_weights import exponential_decay_weights
from eplmodel.paths import PROCESSED_DEV_V2, PROJECT_ROOT
from eplmodel.splits import DEVELOPMENT_SEASONS, REGISTERED_DEV_TEST_SPECS, SEASON_ORDER, selection_folds
from experiments import online_tw_poisson_diagnostic as otp
from experiments import time_weighted_poisson as twp
from experiments import update_policy_diagnostic, validation_2425
from test_time_weighting import _synthetic_development_league
from test_validation import HISTORY, TARGET, _reverse_scores, synthetic_league

OCFG = load_config(otp.ONLINE_CONFIG)
TCFG = load_config(twp.TW_CONFIG)
VCFG = load_config(validation_2425.VALIDATION_CONFIG)
DCFG = load_config(update_policy_diagnostic.DIAGNOSTIC_CONFIG)
H = 730.0


def _online(league, maxiter=100, retry_maxiter=1000):
    return op.online_fold_predictions(league, HISTORY, TARGET, H, 10, maxiter, retry_maxiter)


@pytest.fixture(scope="module")
def league():
    return synthetic_league()


@pytest.fixture(scope="module")
def online(league):
    return _online(league)


@pytest.fixture(scope="module")
def frozen(league):
    return tw.tw_fold_predictions(league, HISTORY, TARGET, {op.FROZEN_ARM: H}, 10)


@pytest.fixture(scope="module")
def parts(league):
    """History rows and common-group target rows of the synthetic fold."""
    data = fold_data(league, HISTORY, TARGET)
    history_rows = data[data["Season"].isin(HISTORY)]
    tgt = data[data["Season"] == TARGET]
    common = tgt[~involves_teams(tgt, unseen_teams(history_rows, tgt))]
    return history_rows, common


def _target_dates(preds):
    return sorted(pd.unique(preds["Date"]))


# --- Fitting set and weights ---------------------------------------------------------------------

def test_fit_set_is_history_plus_target_matches_strictly_before_the_date(parts):
    history_rows, common = parts
    dates = sorted(pd.unique(common["Date"]))
    assert len(op.fit_set(history_rows, common, dates[0])) == len(history_rows)
    rows = op.fit_set(history_rows, common, dates[3])
    target_rows = rows[rows["Season"] == TARGET]
    assert len(target_rows) == int((common["Date"] < dates[3]).sum())
    assert (target_rows["Date"] < dates[3]).all()
    assert not (rows["Date"] == dates[3]).any()


def test_online_weights_use_the_experiment_12_formula_with_the_latest_fit_date(parts):
    history_rows, common = parts
    rows = op.fit_set(history_rows, common, sorted(pd.unique(common["Date"]))[5])
    age = (rows["Date"].max() - rows["Date"]).dt.days.to_numpy()
    np.testing.assert_allclose(op.online_weights(rows, H), 2.0 ** (-age / H), rtol=0, atol=1e-15)
    assert op.online_weights(rows, H).max() == 1.0


@pytest.mark.parametrize("bad", [np.inf, 0.0, -1.0, np.nan])
def test_online_weights_need_a_finite_positive_half_life(parts, bad):
    with pytest.raises(ValueError):
        op.online_weights(parts[0], bad)


def test_reference_date_does_not_change_an_online_fit(parts):
    history_rows, common = parts
    rows = op.fit_set(history_rows, common, sorted(pd.unique(common["Date"]))[6])
    later = rows["Date"].max() + pd.Timedelta(days=200)
    a = PoissonGoalModel().fit(rows, weights=op.online_weights(rows, H))
    b = PoissonGoalModel().fit(rows, weights=exponential_decay_weights(rows["Date"], later, H))
    teams = sorted(set(common["HomeTeam"]))
    np.testing.assert_allclose(a.predict_rates(teams, teams[::-1]), b.predict_rates(teams, teams[::-1]),
                               rtol=1e-8, atol=0)


# --- Prediction timing -----------------------------------------------------------------------------

def test_online_equals_frozen_exactly_on_the_first_target_date(online, frozen):
    preds, _ = online
    fpreds, _ = frozen
    first = preds["on_first_target_date"].to_numpy()
    assert first.sum() > 0
    np.testing.assert_allclose(preds.loc[first, prob_columns(op.ONLINE_ARM)].to_numpy(),
                               fpreds.loc[first, prob_columns(op.FROZEN_ARM)].to_numpy(), rtol=0, atol=1e-12)


def test_online_moves_away_from_frozen_after_target_matches(online, frozen):
    preds, _ = online
    later = ~preds["on_first_target_date"].to_numpy()
    assert not np.allclose(preds.loc[later, prob_columns(op.ONLINE_ARM)].to_numpy(),
                           frozen[0].loc[later, prob_columns(op.FROZEN_ARM)].to_numpy())


@pytest.mark.parametrize("round_index", [0, 3, 9])
def test_predictions_unchanged_by_outcomes_on_or_after_their_date(league, online, round_index):
    preds, _ = online
    d = _target_dates(preds)[round_index]
    changed = _reverse_scores(league, (league["Season"] == TARGET) & (league["Date"] >= d))
    preds2, _ = _online(changed)
    upto = preds["Date"] <= d
    pd.testing.assert_frame_equal(preds2[upto], preds[upto])


def test_no_outcome_on_a_date_affects_another_prediction_on_that_date(league, online):
    preds, _ = online
    d = _target_dates(preds)[4]
    one = preds.index[preds["Date"] == d][0]
    changed = _reverse_scores(league, league["match_id"] == one)
    preds2, _ = _online(changed)
    same_day = preds["Date"] == d
    pd.testing.assert_frame_equal(preds2[same_day], preds[same_day])


def test_earlier_target_outcomes_do_change_later_predictions(league, online):
    preds, _ = online
    d = _target_dates(preds)[3]
    changed = _reverse_scores(league, (league["Season"] == TARGET) & (league["Date"] < d))
    preds2, _ = _online(changed)
    later = preds["Date"] >= d
    assert not np.allclose(preds2.loc[later, prob_columns(op.ONLINE_ARM)],
                           preds.loc[later, prob_columns(op.ONLINE_ARM)])


def test_one_refit_per_date_equals_one_refit_per_match(online, parts):
    preds, _ = online
    history_rows, common = parts
    for match_id, row in common.set_index("match_id").iterrows():
        model, _, _ = op.checked_fit(op.fit_set(history_rows, common, row["Date"]), H, 100, 1000)
        lam, mu = model.predict_rates([row["HomeTeam"]], [row["AwayTeam"]])
        from eplmodel.models.scoreline import outcome_probabilities_from_rates
        np.testing.assert_allclose(outcome_probabilities_from_rates(lam, mu, 0.0, 10)[0],
                                   preds.loc[match_id, prob_columns(op.ONLINE_ARM)].to_numpy(dtype=float),
                                   rtol=0, atol=1e-12)


def test_history_outcomes_change_and_later_seasons_never_enter_the_online_fit(league, online):
    preds, _ = online
    hist_changed, _ = _online(_reverse_scores(league, league["Season"] == HISTORY[-1]))
    assert not np.allclose(hist_changed[prob_columns(op.ONLINE_ARM)], preds[prob_columns(op.ONLINE_ARM)])
    later, _ = _online(_reverse_scores(league, league["Season"] == "1819"))
    pd.testing.assert_frame_equal(later, preds)


def test_a_team_playing_twice_on_one_date_is_refused():
    rows = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
                         "HomeTeam": ["A", "C", "A"], "AwayTeam": ["B", "A", "C"]})
    with pytest.raises(op.OnlineCheckError):
        op.assert_no_team_twice_per_date(rows)
    op.assert_no_team_twice_per_date(rows.iloc[[0, 2]])


# --- Unseen teams and the fixed parameter space ------------------------------------------------------

def test_unseen_team_matches_are_neither_scored_nor_fitted(league, online, parts):
    preds, fitted = online
    _, common = parts
    assert fitted["unseen_teams"] == ["NEW"] and fitted["n_common"] == len(preds) == 20
    assert fitted["n_unseen_target_matches_excluded_from_fit"] == 10
    assert not (preds["HomeTeam"].eq("NEW") | preds["AwayTeam"].eq("NEW")).any()
    new_rows = (league["Season"] == TARGET) & (league["HomeTeam"].eq("NEW") | league["AwayTeam"].eq("NEW"))
    preds2, _ = _online(_reverse_scores(league, new_rows))
    pd.testing.assert_frame_equal(preds2, preds)
    n_hist = int(league["Season"].isin(HISTORY).sum())
    for fit in fitted["fits"]:
        assert fit["n_fit_matches"] == n_hist + int((common["Date"] < pd.Timestamp(fit["date"])).sum())


def test_a_changed_team_set_is_refused(league, monkeypatch):
    original = op.fit_set

    def with_unseen(history_rows, eligible_target_rows, date):
        rows = original(history_rows, eligible_target_rows, date)
        extra = league[(league["Season"] == TARGET) & league["HomeTeam"].eq("NEW") & (league["Date"] < date)]
        return pd.concat([rows, extra], ignore_index=True)

    monkeypatch.setattr(op, "fit_set", with_unseen)
    with pytest.raises(op.OnlineCheckError, match="team set"):
        _online(league)


# --- Fit records, convergence and the failure rule --------------------------------------------------

def test_fit_records_and_one_fit_per_target_date(online):
    preds, fitted = online
    assert fitted["n_fits"] == len(fitted["fits"]) == len(_target_dates(preds)) == 10
    assert fitted["all_converged"] and fitted["n_retried"] == 0
    shares = [f["target_season_weight_share"] for f in fitted["fits"]]
    assert shares[0] == 0.0 and all(b > a for a, b in zip(shares, shares[1:]))
    assert all(f["n_target_matches_in_fit"] == 2 * i for i, f in enumerate(fitted["fits"]))
    assert all(np.isfinite(f["max_abs_coef"]) and f["iterations"] >= 1 for f in fitted["fits"])


def test_rows_sum_to_one_and_contain_no_results(online):
    preds, _ = online
    np.testing.assert_allclose(preds[prob_columns(op.ONLINE_ARM)].sum(axis=1), 1.0, atol=1e-12)
    assert not {"FTHG", "FTAG", "FTR"} & set(preds.columns)


def test_an_invalid_fit_is_retried_once_with_the_same_irls_then_aborts(league, monkeypatch):
    calls = []
    original = PoissonGoalModel.fit

    def spy(self, matches, weights=None, maxiter=100):
        calls.append(maxiter)
        return original(self, matches, weights=weights, maxiter=maxiter)

    monkeypatch.setattr(PoissonGoalModel, "fit", spy)
    monkeypatch.setattr(op, "fit_problem", lambda model: "IRLS did not converge")
    with pytest.raises(op.FitFailure, match="after the registered retry"):
        _online(league)
    assert calls == [100, 1000]


def test_a_real_non_convergence_aborts_without_fallback(league):
    with pytest.raises(op.FitFailure):
        _online(league, maxiter=1, retry_maxiter=1)


def test_a_successful_retry_is_recorded_and_gives_the_same_fit(league, online):
    preds, fitted = _online(league, maxiter=1, retry_maxiter=1000)
    assert fitted["n_retried"] == fitted["n_fits"] and fitted["all_converged"]
    np.testing.assert_allclose(preds[prob_columns(op.ONLINE_ARM)], online[0][prob_columns(op.ONLINE_ARM)],
                               rtol=0, atol=1e-10)


def test_non_finite_rates_abort(league, monkeypatch):
    monkeypatch.setattr(PoissonGoalModel, "predict_rates",
                        lambda self, h, a: (np.full(len(list(h)), np.inf), np.ones(len(list(a)))))
    with pytest.raises(op.FitFailure, match="goal rate"):
        _online(league)


def test_first_date_gap(online, frozen):
    preds = online[0].join(frozen[0][prob_columns(op.FROZEN_ARM)])
    assert op.first_date_gap(preds) <= 1e-12
    first = preds.index[preds["on_first_target_date"]][0]
    preds.loc[first, f"{op.ONLINE_ARM}_H"] += 1e-6
    assert op.first_date_gap(preds) == pytest.approx(1e-6, abs=1e-12)


# --- Evidence rules ---------------------------------------------------------------------------------

def _per_fold(means, n=60, seed=0, brier_sign=None):
    rng = np.random.default_rng(seed)
    out, clusters = {}, {}
    for i, m in enumerate(means):
        t = f"t{i}"
        d = m + rng.normal(0, 0.002, n)
        b = (np.sign(m) if brier_sign is None else brier_sign) * np.abs(d) / 2
        out[t] = {"log_loss": d, "brier": b}
        clusters[t] = np.array([f"{t}_{j // 2}" for j in range(n)])
    return out, clusters


def test_criterion_u_met():
    out = op.criterion_u(*_per_fold([-0.01] * 5), 0.002, 2.0, 4)
    assert out["met"] and not out["mirror_met"] and out["reading"] == "historical_evidence_online_refitting_helps"
    assert out["n_negative_folds"] == 5 and out["n_folds"] == 5


def test_criterion_u_needs_four_of_five_folds():
    per_fold, clusters = _per_fold([-0.03, -0.03, -0.03, 0.001, 0.001])
    out = op.criterion_u(per_fold, clusters, 0.002, 2.0, 4)
    assert out["checks_helps"]["below_minus_floor"] and out["checks_helps"]["beyond_se_multiple"]
    assert out["n_negative_folds"] == 3 and not out["met"]
    assert out["reading"] == "no_distinguishable_updating_benefit"
    assert op.criterion_u(per_fold, clusters, 0.002, 2.0, 3)["met"]   # the rejected 3-of-5 rule would pass


def test_criterion_u_mirror_and_other_failures():
    assert op.criterion_u(*_per_fold([0.01] * 5), 0.002, 2.0, 4)["reading"] == \
        "historical_evidence_online_refitting_hurts"
    small = op.criterion_u(*_per_fold([-0.001] * 5), 0.002, 2.0, 4)
    assert not small["checks_helps"]["below_minus_floor"] and not small["met"]
    per_fold, clusters = _per_fold([-0.01] * 5, brier_sign=1.0)
    assert not op.criterion_u(per_fold, clusters, 0.002, 2.0, 4)["checks_helps"]["brier_negative"]


def _seg(mean, se):
    return {"mean": mean, "clustered_se": se}


def test_accumulation_pattern_is_descriptive():
    labels = OCFG["segments"]["labels"]
    good = {labels[0]: _seg(-0.001, 0.002), labels[1]: _seg(-0.004, 0.003), labels[2]: _seg(-0.008, 0.003),
            labels[3]: _seg(-0.01, 0.004)}
    out = op.accumulation_pattern(good, labels, 2.0)
    assert out["consistent_with_accumulation"] and out["descriptive_only"]
    early = {**good, labels[0]: _seg(-0.02, 0.003)}
    assert not op.accumulation_pattern(early, labels, 2.0)["consistent_with_accumulation"]


def test_validation_labels():
    assert op.validation_label(_seg(-0.02, 0.005), -0.004, 2.0) == "same_sign_as_historical_distinguishable"
    assert op.validation_label(_seg(0.02, 0.005), -0.004, 2.0) == "opposite_sign_distinguishable"
    assert op.validation_label(_seg(-0.005, 0.005), -0.004, 2.0) == "not_distinguishable"
    assert set(OCFG["validation"]["labels"]) == {"same_sign_as_historical_distinguishable", "not_distinguishable",
                                                 "opposite_sign_distinguishable"}


# --- Protocol ----------------------------------------------------------------------------------------

def test_protocol_agrees_with_the_locked_candidate_frozen_specs_and_code():
    otp.check_protocol(OCFG, TCFG, VCFG, DCFG, load_config())


def test_registered_values_are_the_approved_ones():
    o, u, f, g = OCFG["online"], OCFG["criterion_u"], OCFG["fitting"], OCFG["groups"]
    assert o["half_life_days"] == 730.0 == TCFG["locked"]["half_life_days"]
    assert (o["unseen_team_matches_in_online_fit"], o["team_parameter_space"]) == (False, "fixed_history_teams")
    assert (o["refit_cadence"], o["information_policy"], o["warm_start"]) == \
        ("once_per_target_date", "strictly_earlier_dates", False)
    assert (u["practical_floor_log_loss"], u["clustered_se_multiple"], u["min_folds_same_sign"]) == (0.002, 2.0, 4)
    assert (f["maxiter"], f["retry_maxiter"], f["on_failure"]) == (100, 1000, "abort_stage_no_fallback")
    assert g["expected_online_fits"] == {"1718": 98, "1819": 98, "1920": 110, "2021": 128, "2122": 119, "2425": 109}
    assert OCFG["protocol"]["validation_evidence_class"] == "diagnostic_descriptive_cannot_confirm"
    assert OCFG["protocol"]["pool_validation_with_historical"] is False


@pytest.mark.parametrize("section,key,value", [
    ("protocol", "historical_targets", ["1718", "1819", "1920", "2021", "2223"]),
    ("protocol", "validation_target", "2526"),
    ("protocol", "pool_validation_with_historical", True),
    ("protocol", "data_sha256", "0" * 64),
    ("online", "half_life_days", 365.0),
    ("online", "formula", "Goals ~ Team + Opponent"),
    ("online", "max_goals", 12),
    ("online", "weighting", "linear"),
    ("online", "reference_date", "prediction_date"),
    ("online", "refit_cadence", "once_per_match"),
    ("online", "fit_set", "history_plus_all_target_matches_strictly_before_date"),
    ("online", "warm_start", True),
    ("online", "unseen_team_matches_in_online_fit", True),
    ("fitting", "retry_maxiter", 5000),
    ("fitting", "on_failure", "fall_back_to_frozen"),
    ("criterion_u", "min_folds_same_sign", 3),
    ("criterion_u", "practical_floor_log_loss", 0.001),
    ("segments", "lower_edges", [0, 5, 19, 29]),
    ("decomposition", "components", ["poisson_minus_poisson_tw", "poisson_tw_minus_elo"]),
    ("reproduction", "experiment_11_sha256", "0" * 64),
    ("reproduction", "experiment_12_development_sha256", "0" * 64),
    ("reproduction", "experiment_12_development_column", "poisson_tw_h365"),
    ("historical_stage", "max_season", "2425"),
    ("outputs", "in_run_all", True),
])
def test_protocol_mismatch_is_refused(section, key, value):
    tampered = json.loads(json.dumps(OCFG))
    tampered[section][key] = value
    with pytest.raises(otp.ProtocolMismatchError):
        otp.check_protocol(tampered, TCFG, VCFG, DCFG, load_config())


def test_an_unlocked_candidate_is_refused():
    unlocked = json.loads(json.dumps(TCFG))
    unlocked["locked"]["status"] = "unlocked"
    with pytest.raises(otp.ProtocolMismatchError):
        otp.check_protocol(OCFG, unlocked, VCFG, DCFG, load_config())


def test_online_arm_is_not_registered_for_dev_test_or_holdout():
    assert OCFG["arms"]["poisson_tw_online"] not in REGISTERED_DEV_TEST_SPECS
    for reused in ("poisson_static_v1", "poisson_time_weighted_v1"):
        tampered = json.loads(json.dumps(OCFG))
        tampered["arms"]["poisson_tw_online"] = reused
        with pytest.raises(otp.ProtocolMismatchError):
            otp.check_protocol(tampered, TCFG, VCFG, DCFG, load_config())


def test_group_check_refuses_unregistered_sizes_and_fit_counts():
    good = {"n_full": 380, "n_common": 342, "unseen_teams": ["Ipswich"], "n_fits": 109}
    otp.check_groups("2425", good, OCFG)
    for key, value in [("n_common", 341), ("n_fits", 108), ("unseen_teams", [])]:
        with pytest.raises(otp.ProtocolMismatchError):
            otp.check_groups("2425", {**good, key: value}, OCFG)


def test_experiment_is_not_in_run_all():
    from experiments import run_all
    assert otp not in run_all.EXPERIMENTS
    assert "online_tw" not in (PROJECT_ROOT / "experiments" / "run_all.py").read_text(encoding="utf-8")


# --- Historical-stage data guard and the validation-stage lock ------------------------------------------

def _allow_short_seasons(monkeypatch):
    from eplmodel.data.validate import validate_matches
    monkeypatch.setattr(otp, "validate_matches", lambda df: validate_matches(df, matches_per_season=None))


def test_historical_loader_drops_later_rows_before_validation(monkeypatch):
    full = _synthetic_development_league()
    late = full["Season"].map(SEASON_ORDER.index) > SEASON_ORDER.index("2122")
    full.loc[late, "FTR"] = "X"
    monkeypatch.setattr(otp, "verify_file", lambda path: True)
    monkeypatch.setattr(otp, "load_matches", lambda path, validate: full.copy())
    _allow_short_seasons(monkeypatch)
    out = otp.load_historical_matches("2122")
    assert sorted(out["Season"].unique(), key=SEASON_ORDER.index) == list(DEVELOPMENT_SEASONS[:8])


def test_validation_refuses_to_load_data_before_the_historical_lock(monkeypatch):
    unlocked = json.loads(json.dumps(OCFG))
    unlocked["historical_locked"] = {"status": "unlocked", "historical_metrics_sha256": "", "historical_commit": ""}
    monkeypatch.setattr(otp, "load_all_configs", lambda: (unlocked, TCFG, VCFG, DCFG, load_config()))

    def forbidden(*args, **kwargs):
        raise AssertionError("2024-25 data must not be loaded before the historical lock")

    import eplmodel.data
    monkeypatch.setattr(eplmodel.data, "load_dev_matches", forbidden)
    monkeypatch.setattr(otp, "verify_file", forbidden)
    with pytest.raises(otp.LockError, match="not locked"):
        otp.run_validation(write=False)


def test_committed_lock_is_the_historical_stage_output():
    """The committed [historical_locked] values match the historical metrics file, when it exists."""
    lock = OCFG["historical_locked"]
    assert lock["status"] == "locked"
    assert lock["historical_commit"] == "7faadfc487c3d718122390a7dd23f332d09f2367"
    path = PROJECT_ROOT / "results" / OCFG["historical_stage"]["results_name"] / "metrics.json"
    if path.exists():
        from eplmodel.data.checksums import content_sha256
        assert content_sha256(path) == lock["historical_metrics_sha256"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["results"]["stage"] == "historical"
        assert payload["results"]["seasons_loaded"][-1] == "2122"
        assert payload["provenance"]["git_commit"] == lock["historical_commit"]
        assert payload["provenance"]["git_dirty"] is False


def _locked_setup(tmp_path, monkeypatch, stage="historical", dirty=False, commit="abc", clean=True, ancestor=True,
                  sha=None, mean=-0.004):
    metrics_dir = tmp_path / OCFG["historical_stage"]["results_name"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "metrics.json"
    path.write_text(json.dumps({"results": {"stage": stage, "reading": "no_distinguishable_updating_benefit",
                                            "criterion_u": {"pooled": {"log_loss": {"mean": mean}}}},
                                "provenance": {"git_commit": "abc", "git_dirty": dirty}}), encoding="utf-8")
    from eplmodel.data.checksums import content_sha256
    monkeypatch.setattr(otp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(otp, "working_tree_clean", lambda: clean)
    monkeypatch.setattr(otp, "is_ancestor_of_head", lambda c: ancestor)
    cfg = json.loads(json.dumps(OCFG))
    cfg["historical_locked"] = {"status": "locked", "historical_metrics_sha256": sha or content_sha256(path),
                                "historical_commit": commit}
    return cfg


def test_lock_accepts_a_committed_clean_lock(tmp_path, monkeypatch):
    assert otp.check_historical_lock(_locked_setup(tmp_path, monkeypatch))["stage"] == "historical"


@pytest.mark.parametrize("kwargs,message", [
    ({"sha": "0" * 64}, "differ from"),
    ({"stage": "validation"}, "not a historical-stage result"),
    ({"dirty": True}, "clean commit"),
    ({"commit": "def"}, "clean commit"),
    ({"ancestor": False}, "does not descend"),
    ({"clean": False}, "uncommitted changes"),
])
def test_lock_refusals(tmp_path, monkeypatch, kwargs, message):
    cfg = _locked_setup(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(otp.LockError, match=message):
        otp.check_historical_lock(cfg)


# --- End-to-end on synthetic data only ------------------------------------------------------------------

def _recorded_factory(synthetic):
    """Stand-in for the recorded Experiment 11 and 12 predictions, recomputed from the synthetic league."""
    def recorded(ocfg, path_key, sha_key, targets):
        frames = []
        for history, target in selection_folds(list(targets)):
            if path_key == "experiment_11_predictions":
                preds, _ = up.diagnostic_fold_predictions(synthetic, history, target, VCFG, DCFG)
            else:
                stage = "development" if "development" in path_key else "validation"
                column = ocfg["reproduction"][f"experiment_12_{stage}_column"]
                preds, _ = tw.tw_fold_predictions(synthetic, history, target, {column: H}, 10)
            frames.append(preds.assign(fold_target=target))
        return pd.concat(frames)
    return recorded


def test_historical_stage_runs_end_to_end_on_synthetic_data(monkeypatch):
    synthetic = _synthetic_development_league()
    late = synthetic["Season"].map(SEASON_ORDER.index) > SEASON_ORDER.index("2122")
    synthetic.loc[late, "FTR"] = "X"   # would break validation and scoring if any later row were used
    monkeypatch.setattr(otp, "verify_file", lambda path: True)
    monkeypatch.setattr(otp, "load_matches", lambda path, validate: synthetic.copy())
    _allow_short_seasons(monkeypatch)
    monkeypatch.setattr(otp, "check_groups", lambda target, fitted, ocfg: None)
    monkeypatch.setattr(otp, "load_recorded", _recorded_factory(synthetic[~late]))

    out = otp.run_historical(write=False)
    assert out["stage"] == "historical" and out["seasons_loaded"] == list(DEVELOPMENT_SEASONS[:8])
    assert [f["target"] for f in out["folds"]] == ["1718", "1819", "1920", "2021", "2122"]
    for fold in out["folds"]:
        fitted = fold["fitted"]
        assert fitted["reproduction_max_abs_diff"] == {"experiment_11": 0.0, "experiment_12": 0.0}
        assert fitted["first_date_max_abs_diff"] <= 1e-12
        assert fitted["online"]["n_fits"] == 20 and fitted["online"]["all_converged"]
        assert max(fold["max_identity_residual"].values()) <= 1e-12
    assert out["folds"][0]["fitted"]["unseen_teams"] == ["NEW"] and out["folds"][0]["n_matches"] == 40
    assert out["fits_total"] == 100 and out["retries_total"] == 0
    assert out["reading"] in ("historical_evidence_online_refitting_helps",
                              "historical_evidence_online_refitting_hurts", "no_distinguishable_updating_benefit")
    pooled = out["pooled"]
    parts = sum(pooled["decomposition"][c]["log_loss"]["mean"] for c in OCFG["decomposition"]["components"])
    assert parts == pytest.approx(pooled["decomposition"]["poisson_minus_elo"]["log_loss"]["mean"], abs=1e-12)
    assert pooled["primary"]["log_loss"]["n_matches"] == 40 + 4 * 60
    assert set(pooled["segments"]) == set(OCFG["segments"]["labels"])
    assert out["criterion_u"]["pooled"]["log_loss"]["mean"] == pytest.approx(pooled["primary"]["log_loss"]["mean"])


def test_validation_stage_scores_the_online_arm_once_on_synthetic_data(tmp_path, monkeypatch):
    synthetic = _synthetic_development_league(3)
    cfg = _locked_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(otp, "load_all_configs", lambda: (cfg, TCFG, VCFG, DCFG, load_config()))
    monkeypatch.setattr(otp, "verify_file", lambda path: True)
    import eplmodel.data
    monkeypatch.setattr(eplmodel.data, "load_dev_matches", lambda: synthetic.copy())
    monkeypatch.setattr(otp, "check_groups", lambda target, fitted, ocfg: None)
    monkeypatch.setattr(otp, "load_recorded", _recorded_factory(synthetic))
    calls = []
    original = op.online_fold_predictions

    def spy(matches, history, target, h, *args):
        calls.append((target, h))
        return original(matches, history, target, h, *args)

    monkeypatch.setattr(otp.op, "online_fold_predictions", spy)
    out = otp.run_validation(write=False)
    assert calls == [("2425", 730.0)]
    assert out["evidence_class"] == "diagnostic_descriptive_cannot_confirm"
    assert out["label"] in OCFG["validation"]["labels"]
    assert out["n_matches"] == 60 and max(out["max_identity_residual"].values()) <= 1e-12
    assert out["fitted"]["first_date_max_abs_diff"] <= 1e-12


# --- Real data: fixtures only ----------------------------------------------------------------------------

def test_registered_groups_and_online_fit_counts_hold_on_the_real_fixture_lists():
    """Dates and team names of the real fixtures only; no model is fitted and no result is read."""
    if not PROCESSED_DEV_V2.exists():
        pytest.skip("data/processed/matches_dev_v2.csv missing")
    from eplmodel.data import load_dev_matches
    fixtures = load_dev_matches()[["Season", "Date", "HomeTeam", "AwayTeam"]]
    g = OCFG["groups"]
    targets = [*OCFG["protocol"]["historical_targets"], OCFG["protocol"]["validation_target"]]
    for history, target in selection_folds(targets):
        hist = fixtures[fixtures["Season"].isin(history)]
        tgt = fixtures[fixtures["Season"] == target]
        op.assert_no_team_twice_per_date(tgt)
        unseen = unseen_teams(hist, tgt)
        common = tgt[~involves_teams(tgt, unseen)]
        assert unseen == g["expected_unseen_teams"][target]
        assert (len(tgt), len(common)) == (g["expected_full"][target], g["expected_common"][target])
        assert common["Date"].nunique() == g["expected_online_fits"][target]
