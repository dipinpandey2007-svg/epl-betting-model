"""Common evaluation harness: forecasts, folds, scoring, segments and reproduction.

Every test uses synthetic data, so all of them run in CI. The real-data reproduction gate for
Experiments 10-13 is tests/test_reproduction_recorded.py (golden).
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss as sklearn_log_loss

from eplmodel.constants import OUTCOMES
from eplmodel.evaluation import folds, forecasts, online_poisson, reproduction, scoring, segments
from eplmodel.evaluation import time_weighting as tw
from eplmodel.evaluation import update_policy as up
from eplmodel.evaluation import validation
from eplmodel.evaluation.metrics import multiclass_brier, multiclass_log_loss
from eplmodel.splits import (
    DevTestAccessError,
    ExposedValidationError,
    HoldoutAccessError,
    SplitAccessError,
)
from experiments import online_tw_poisson_diagnostic, time_weighted_poisson, update_policy_diagnostic
from recorded_snapshot import (
    FLOAT_ABS_TOL,
    FLOAT_REL_TOL,
    SNAPSHOT_FILE,
    assert_snapshot_close,
    compute_snapshot,
    snapshot_mismatches,
)
from test_validation import HISTORY, TARGET, _reverse_scores, synthetic_league


@pytest.fixture(scope="module")
def league():
    return synthetic_league()


@pytest.fixture(scope="module")
def fold(league):
    return folds.build_fold(league, HISTORY, TARGET)


def _frame(probs, ids=("m1", "m2"), arm="x"):
    df = pd.DataFrame(np.asarray(probs, dtype=float), columns=forecasts.prob_columns(arm),
                      index=pd.Index(list(ids), name="match_id"))
    df["Date"] = pd.Timestamp("2018-01-01")
    return df


# --- Reproduction of the recorded library code (CI gate) -----------------------------------------

def test_recorded_library_outputs_reproduce_the_pre_harness_snapshot():
    """Experiments 10-13's prediction and scoring functions give the outputs captured before the harness.

    Structure and every non-float value must match exactly; floats within the registered tolerance 1e-12,
    because BLAS code paths differ between CPUs and library versions in the last bits.
    """
    expected = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    assert_snapshot_close(compute_snapshot(), expected)


# --- The snapshot comparison itself ------------------------------------------------------------------

SAMPLE = {"scores": {"log_loss": 0.9475, "brier": 0.5597, "n": 380}, "labels": ["H", "D", "A"],
          "converged": True, "rho": -0.04, "unseen": ["Ipswich"], "z": None, "big": 1500.25}


def _copy(obj):
    return json.loads(json.dumps(obj))


def test_snapshot_tolerance_is_the_registered_one():
    assert FLOAT_REL_TOL == FLOAT_ABS_TOL == 1e-12 == reproduction.REGISTERED_TOLERANCE


def test_identical_structures_pass_deterministically():
    for _ in range(3):
        assert snapshot_mismatches(_copy(SAMPLE), _copy(SAMPLE)) == []
        assert_snapshot_close(_copy(SAMPLE), SAMPLE)


@pytest.mark.parametrize("delta", [1e-13, 5e-13, 9e-13])
def test_float_differences_below_1e_12_are_accepted(delta):
    changed = _copy(SAMPLE)
    changed["scores"]["log_loss"] += delta
    changed["rho"] -= delta
    assert snapshot_mismatches(changed, SAMPLE) == []


@pytest.mark.parametrize("delta", [2e-12, 1e-9, 1e-4])
def test_float_differences_above_1e_12_are_rejected(delta):
    changed = _copy(SAMPLE)
    changed["scores"]["log_loss"] += delta
    found = snapshot_mismatches(changed, SAMPLE)
    assert len(found) == 1 and found[0].startswith("/scores/log_loss:")


def test_relative_tolerance_scales_with_large_values_only_up_to_1e_12():
    changed = _copy(SAMPLE)
    changed["big"] = 1500.25 * (1 + 5e-13)               # inside rel_tol
    assert snapshot_mismatches(changed, SAMPLE) == []
    changed["big"] = 1500.25 * (1 + 1e-10)               # outside
    assert snapshot_mismatches(changed, SAMPLE)


@pytest.mark.parametrize("path,value", [
    (("scores", "n"), 381),            # integer count
    (("converged",), False),           # bool
    (("labels",), ["H", "A", "D"]),    # string order
    (("unseen",), ["Luton"]),          # team list
    (("z",), 0.0),                     # None vs float
    (("scores", "n"), 380.0),          # int vs float: no coercion
    (("converged",), 1),               # bool vs int: no coercion
])
def test_non_float_changes_are_rejected(path, value):
    changed = _copy(SAMPLE)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert snapshot_mismatches(changed, SAMPLE)


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("rho"),                               # missing key
    lambda s: s.update(extra=1),                          # unexpected key
    lambda s: s["labels"].append("X"),                    # list length
    lambda s: s.update(scores=[0.9475, 0.5597, 380]),     # dict replaced by list
])
def test_structural_changes_are_rejected(mutate):
    changed = _copy(SAMPLE)
    mutate(changed)
    assert snapshot_mismatches(changed, SAMPLE)


def test_failure_message_reports_only_the_first_few_mismatches():
    expected = {"values": [float(i) for i in range(50)]}
    actual = {"values": [float(i) + 1.0 for i in range(50)]}
    with pytest.raises(AssertionError) as err:
        assert_snapshot_close(actual, expected, max_reported=5)
    message = str(err.value)
    assert message.startswith("50 snapshot mismatch(es):")
    assert message.count("/values[") == 5 and "... and 45 more" in message


def test_reproduction_is_deterministic(league):
    a, _ = online_poisson.online_fold_predictions(league, HISTORY, TARGET, 730.0, 10, 100, 1000)
    b, _ = online_poisson.online_fold_predictions(league, HISTORY, TARGET, 730.0, 10, 100, 1000)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert reproduction.compare_predictions(a, b, [online_poisson.ONLINE_ARM]) == 0.0


def test_recorded_modules_use_the_harness_implementations():
    assert up.paired_difference_clustered is scoring.paired_difference_clustered
    assert up.per_match_losses is tw.per_match_losses is scoring.per_match_losses
    assert up.split_difference is online_poisson.split_difference is scoring.split_difference
    assert up.prior_games_played is segments.prior_games_played
    assert up.assign_segments is segments.assign_segments
    assert validation.paired_difference is scoring.paired_difference
    assert validation.prob_columns is forecasts.prob_columns
    assert validation.outcomes_for is forecasts.outcomes_for
    assert validation.unseen_teams is folds.unseen_teams
    assert tw.restrict_to_development is folds.restrict_to_max_season
    assert tw.assert_development_only is folds.assert_max_season


def test_frozen_experiment_helpers_agree_with_the_harness(league):
    """The recorded experiment scripts keep their own copies; they must equal the harness versions."""
    preds, _ = validation.fold_predictions(league, HISTORY, TARGET, validation_cfg())
    expected = scoring.date_clusters(preds, TARGET)
    for module in (update_policy_diagnostic, time_weighted_poisson, online_tw_poisson_diagnostic):
        assert np.array_equal(module.date_clusters(preds, TARGET), expected)


def validation_cfg():
    from eplmodel.config import load_config
    from experiments import validation_2425
    return load_config(validation_2425.VALIDATION_CONFIG)


# --- Forecast format and class ordering -----------------------------------------------------------

def test_probability_columns_follow_hda_order():
    assert OUTCOMES == ("H", "D", "A")
    assert forecasts.prob_columns("elo") == ["elo_H", "elo_D", "elo_A"]


def test_metrics_use_hda_order_and_differ_from_naive_sklearn():
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
    results = ["H", "A", "D"]
    expected = -np.mean(np.log([0.7, 0.5, 0.4]))
    assert multiclass_log_loss(results, probs) == pytest.approx(expected)
    # sklearn sorts labels to (A, D, H): it agrees only after the columns are reversed.
    assert sklearn_log_loss(results, probs[:, ::-1], labels=["A", "D", "H"]) == pytest.approx(expected)
    assert sklearn_log_loss(results, probs, labels=["A", "D", "H"]) != pytest.approx(expected)


def test_evaluation_set_scores_with_the_established_metrics():
    preds = _frame([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], ids=("2018-01-01_A_B", "2018-01-02_C_D"))
    matches = pd.DataFrame({"match_id": ["2018-01-02_C_D", "2018-01-01_A_B"], "FTR": ["A", "H"]})
    es = scoring.evaluation_set(preds, matches, "1718", ["x"])
    assert list(es.results) == ["H", "A"]                     # joined by match_id, not row position
    s = es.scores()["x"]
    assert s["log_loss"] == multiclass_log_loss(["H", "A"], forecasts.probabilities(preds, "x"))
    assert s["brier"] == multiclass_brier(["H", "A"], forecasts.probabilities(preds, "x"))
    assert list(es.clusters) == ["1718_2018-01-01", "1718_2018-01-01"]


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.rename_axis("id"), "indexed by match_id"),
    (lambda d: d.assign(FTR=["H", "A"]), "result columns"),
    (lambda d: d.drop(columns="x_D"), "lacks columns"),
    (lambda d: d.assign(x_H=[0.9, 0.2]), "do not sum to 1"),
    (lambda d: d.assign(x_H=[np.nan, 0.2]), "not finite"),
    (lambda d: pd.concat([d.iloc[:1], d.iloc[:1]]), "unique"),
])
def test_forecast_frames_are_checked(mutate, message):
    good = _frame([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    forecasts.check_forecast_frame(good, ["x"])
    with pytest.raises(forecasts.ForecastFormatError, match=message):
        forecasts.check_forecast_frame(mutate(good), ["x"])


# --- Folds, eligibility and information sets --------------------------------------------------------

def test_build_fold_isolates_history_and_target(league, fold):
    assert set(fold.history_rows["Season"]) == set(HISTORY)
    assert set(fold.target_rows["Season"]) == {TARGET}
    assert "1819" in set(league["Season"])                    # a later season exists but never enters
    assert fold.history_rows["Date"].max() < fold.target_rows["Date"].min()
    assert set(fold.fixtures().columns) == {"Date", "Season", "HomeTeam", "AwayTeam"}


def test_build_fold_uses_the_strict_selection_guard(league):
    for history, target, error in [(("2122",), "2425", ExposedValidationError),
                                   (("2122",), "2223", DevTestAccessError),
                                   (("2324",), "2526", HoldoutAccessError),
                                   (("1617", "1718"), "1718", SplitAccessError)]:
        with pytest.raises(error):
            folds.build_fold(league, history, target)
    holdout_rows = league.assign(Season=league["Season"].replace({"1819": "2526"}))
    with pytest.raises(HoldoutAccessError):
        folds.build_fold(holdout_rows, HISTORY, TARGET)


def test_common_group_excludes_unseen_teams_explicitly(fold):
    unseen, common = fold.common_group()
    assert unseen == ["NEW"]
    assert not (common["HomeTeam"].eq("NEW") | common["AwayTeam"].eq("NEW")).any()
    assert len(common) + int((fold.target_rows[["HomeTeam", "AwayTeam"]] == "NEW").any(axis=1).sum()) \
        == len(fold.target_rows)


def test_frozen_inputs_are_the_history_only(fold):
    assert fold.frozen_inputs() is fold.history_rows


def test_online_inputs_contain_only_strictly_earlier_target_matches(fold):
    dates = sorted(fold.target_rows["Date"].unique())
    first = fold.online_inputs(dates[0])
    assert len(first) == len(fold.history_rows)              # nothing from the target season yet
    d = dates[3]
    rows = fold.online_inputs(d)
    target_part = rows[rows["Season"] == TARGET]
    assert (target_part["Date"] < d).all()                   # same-day results never enter
    assert len(target_part) == int((fold.target_rows["Date"] < d).sum())
    with pytest.raises(SplitAccessError):
        fold.online_inputs(d, eligible_target_rows=fold.history_rows)


def test_online_predictions_do_not_depend_on_outcomes_on_or_after_their_date(league):
    preds, _ = online_poisson.online_fold_predictions(league, HISTORY, TARGET, 730.0, 10, 100, 1000)
    cut = sorted(preds["Date"].unique())[4]
    changed = _reverse_scores(league, (league["Season"] == TARGET) & (league["Date"] >= cut))
    again, _ = online_poisson.online_fold_predictions(changed, HISTORY, TARGET, 730.0, 10, 100, 1000)
    before = preds["Date"] <= cut
    assert reproduction.compare_predictions(again[before.to_numpy()], preds[before], [online_poisson.ONLINE_ARM],
                                            tolerance=0.0) == 0.0
    assert not np.allclose(again.loc[~before.to_numpy(), forecasts.prob_columns(online_poisson.ONLINE_ARM)],
                           preds.loc[~before, forecasts.prob_columns(online_poisson.ONLINE_ARM)])


def test_restrict_to_max_season_drops_later_seasons(league):
    out = folds.restrict_to_max_season(league, "1718")
    assert set(out["Season"]) == {*HISTORY, TARGET}
    with pytest.raises(SplitAccessError):
        folds.assert_max_season(league, "1718")


def test_summarize_fits():
    fits = [{"retried": False, "converged": True, "iterations": 5},
            {"retried": True, "converged": True, "iterations": 9}]
    assert folds.summarize_fits(fits) == {"n_fits": 2, "n_retried": 1, "all_converged": True, "max_iterations": 9}
    assert folds.summarize_fits([]) == {"n_fits": 0, "n_retried": 0, "all_converged": True, "max_iterations": 0}


# --- Paired differences, clustering, calibration ---------------------------------------------------

def test_paired_difference_sign_and_naive_se():
    d = scoring.paired_difference(np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 1.5]))
    assert d["mean"] == pytest.approx(0.5)                   # model minus reference
    assert d["naive_se"] == pytest.approx(np.std([-0.5, 0.5, 1.5], ddof=1) / np.sqrt(3))


def test_clustered_se_hand_example_and_naive_limit():
    left, right = np.array([1.0, 0.0, 2.0, 1.0]), np.zeros(4)
    out = scoring.paired_difference_clustered(left, right, ["a", "a", "b", "b"])
    # d - mean = [0, -1, 1, 0]; cluster sums [-1, 1]; SE^2 = 2/1 * 2 / 16
    assert out["clustered_se"] == pytest.approx(np.sqrt(2 * 2 / 16))
    single = scoring.paired_difference_clustered(left, right, ["a", "b", "c", "d"])
    assert single["clustered_se"] == pytest.approx(single["naive_se"])


def test_differences_and_identity_residual():
    losses = {"a": {"log_loss": np.array([1.0, 2.0]), "brier": np.array([0.5, 0.4])},
              "b": {"log_loss": np.array([0.5, 1.0]), "brier": np.array([0.3, 0.2])},
              "c": {"log_loss": np.array([0.2, 0.1]), "brier": np.array([0.1, 0.1])}}
    out = scoring.differences(losses, ["a_minus_b"], ["d1", "d2"])
    assert out["a_minus_b"]["log_loss"]["mean"] == pytest.approx(0.75)
    ll = {k: v["log_loss"] for k, v in losses.items()}
    assert scoring.identity_residual(ll, ["a_minus_b", "b_minus_c"], "a_minus_c") == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        scoring.split_difference("a_vs_b")


def test_calibration_in_the_large_and_gaps():
    probs = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    results = np.array(["H", "D"])
    cal = scoring.calibration_in_the_large(probs, results)
    assert cal["mean_predicted"] == pytest.approx({"H": 0.5, "D": 0.3, "A": 0.2})
    assert cal["observed"] == {"H": 0.5, "D": 0.5, "A": 0.0}
    assert cal["mean_entropy_nats"] == pytest.approx(-np.sum([0.5, 0.3, 0.2] * np.log([0.5, 0.3, 0.2])))
    gaps = scoring.calibration_gaps(probs, results, ["d1", "d2"])
    assert gaps["D"]["mean"] == pytest.approx(0.2) and gaps["A"]["mean"] == pytest.approx(-0.2)


def test_date_clusters_are_per_target_and_date():
    preds = _frame([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    assert list(scoring.date_clusters(preds, "1718")) == ["1718_2018-01-01", "1718_2018-01-01"]


# --- Segments ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,label", [(0, "0-9"), (9.5, "0-9"), (10, "10-18"), (18.5, "10-18"),
                                         (19, "19-28"), (29, "29+"), (37.5, "29+")])
def test_registered_segment_boundaries_are_unchanged(value, label):
    assert segments.assign_segments([value], [0, 10, 19, 29], ["0-9", "10-18", "19-28", "29+"])[0] == label


def test_prior_games_use_strictly_earlier_dates(fold):
    prior = segments.prior_games_played(fold.target_rows)
    first_date = fold.target_rows["Date"].min()
    on_first = fold.target_rows.loc[fold.target_rows["Date"] == first_date, "match_id"]
    assert (prior.loc[on_first] == 0).all()


# --- Reproduction rule -------------------------------------------------------------------------------

def test_compare_predictions_enforces_order_missing_values_and_tolerance():
    a = _frame([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    assert reproduction.compare_predictions(a, a.copy(), ["x"]) == 0.0
    with pytest.raises(reproduction.ReproductionError, match="matches differ"):
        reproduction.compare_predictions(a, a.iloc[::-1], ["x"])
    b = a.copy()
    b.loc["m1", "x_H"] += 1e-11
    with pytest.raises(reproduction.ReproductionError, match="differ from the record"):
        reproduction.compare_predictions(a, b, ["x"])
    c = a.copy()
    c.loc["m1", "x_H"] = np.nan
    with pytest.raises(reproduction.ReproductionError, match="missing"):
        reproduction.compare_predictions(a, c, ["x"])
    renamed = a.rename(columns=dict(zip(forecasts.prob_columns("x"), forecasts.prob_columns("y"))))
    assert reproduction.compare_predictions(a, renamed, ["x"], rename={"x": "y"}) == 0.0


def test_recorded_registry_matches_the_committed_metrics_and_configs():
    from eplmodel.config import load_config
    from eplmodel.paths import CONFIG_DIR
    o = load_config(CONFIG_DIR / "online_tw_poisson_diagnostic_v1.toml")["reproduction"]
    d = load_config(CONFIG_DIR / "update_policy_diagnostic_v1.toml")["reproduction"]
    assert reproduction.RECORDED["10"].recorded_sha256() == d["predictions_sha256"]
    assert reproduction.RECORDED["11"].recorded_sha256() == o["experiment_11_sha256"]
    assert reproduction.RECORDED["12-development"].recorded_sha256() == o["experiment_12_development_sha256"]
    assert reproduction.RECORDED["12-validation"].recorded_sha256() == o["experiment_12_validation_sha256"]
    assert set(reproduction.RECORDED_ARM_SPECS) >= {a for r in reproduction.RECORDED.values() for a in r.arms
                                                    if not a.startswith("poisson_tw_h")}
    assert reproduction.REGISTERED_TOLERANCE == o["tolerance"] == d["tolerance"] == 1e-12


def test_a_modified_recorded_file_is_refused(tmp_path, monkeypatch):
    rec = reproduction.RecordedPredictions("x", "fake", ("x",))
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "predictions.csv").write_text("match_id,x_H\n", encoding="utf-8")
    (tmp_path / "fake" / "metrics.json").write_text(json.dumps({"results": {"predictions": {"sha256": "0"}}}),
                                                    encoding="utf-8")
    monkeypatch.setattr(reproduction, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(reproduction, "PROJECT_ROOT", tmp_path)
    with pytest.raises(reproduction.ReproductionError, match="checksum"):
        reproduction.load_recorded_predictions(rec)
