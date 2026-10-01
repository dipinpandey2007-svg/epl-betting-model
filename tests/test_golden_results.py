"""Golden regression tests: the refactored code must reproduce every established result.

Golden values were captured at full precision from the original exploratory
script (see tests/golden/golden_values.json, "_source").

Tolerances
----------
- METRIC_TOL = 1e-9 (absolute) for log loss, Brier, coefficients, the home
  shift and probabilities. The refactor reproduces these bit-for-bit today;
  the tolerance only allows for floating-point summation-order effects.
- LOGLIK_TOL = 1e-6 (absolute) for log-likelihoods, which are sums over 3,040
  matches of magnitude ~8,700 (summation order alone moves them by ~1e-11).
- Counts, selected K, selected rho and team lists must match exactly.
- Rounded figures quoted in the documentation must agree to 4 decimal places.

A failure here means behaviour changed. That is only acceptable as a
deliberate, documented methodological change with updated golden values.
"""

import numpy as np
import pytest

from eplmodel.models.elo import EloOutcomeModel, elo_difference, home_shift_from_results, run_elo
from eplmodel.evaluation.metrics import score
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.models.scoreline import outcome_probabilities_from_rates
from eplmodel.splits import DEV_TEST_SEASONS, TRAIN_SEASONS, require_registered_dev_test_spec
from experiments import dixon_coles_rho_search, elo_dev_test, elo_k_selection, goal_models_dev_test, promoted_team_folds

pytestmark = pytest.mark.golden

METRIC_TOL = 1e-9
LOGLIK_TOL = 1e-6


def close(actual, expected, tol=METRIC_TOL):
    return actual == pytest.approx(expected, abs=tol, rel=0)


@pytest.fixture(scope="module")
def goal_results(matches):
    return goal_models_dev_test.run(write=False)


def test_elo_k_selection(matches, golden):
    g = golden["elo_k_selection"]
    out = elo_k_selection.run(write=False)
    table = out["k_table"]
    assert [f["validate"] for f in out["folds"]] == g["validation_seasons"]
    for _, row in table.iterrows():
        key = str(int(row["K"]))
        assert close(row["mean_log_loss"], g["mean_log_loss"][key])
        folds = [row[f"val_{s}"] for s in g["validation_seasons"]]
        assert close(np.array(folds), np.array(g["fold_log_loss"][key]))
    assert out["best_k"] == g["best_k"] == 25
    assert out["configured_k_matches_selection"]


def test_elo_dev_test(matches, golden):
    out = elo_dev_test.run(write=False)
    g = golden["elo_k25_logreg_v1"]
    assert close(out["home_shift"], g["home_shift"])
    assert out["elo"]["n_matches"] == g["n"] == 760
    assert close(out["elo"]["log_loss"], g["log_loss"])
    assert close(out["elo"]["brier"], g["brier"])
    b = golden["frequency_baseline_v1"]
    assert close(out["frequency_baseline"]["log_loss"], b["log_loss"])
    assert close(out["frequency_baseline"]["brier"], b["brier"])


@pytest.mark.parametrize("tol,atol", [(1e-10, 1e-7), (1e-4, 1e-5)])
def test_elo_home_shift_is_redundant_on_real_data(matches, tol, atol):
    """Documented claim: the ~43.08 shift added to the regression feature leaves the fitted model unchanged.

    Checked on training-season fitted probabilities only, so no development-test data are used.
    At sklearn's default tolerance the two fits differ by ~2e-6 (solver stopping); at a tight
    tolerance they coincide.
    """
    hist = run_elo(matches, k=25, home_adv=0.0)
    train = hist["Season"].isin(TRAIN_SEASONS).to_numpy()
    probs = []
    for shift in (0.0, home_shift_from_results(hist.loc[train, "FTR"])):
        diff = elo_difference(hist, shift)
        model = EloOutcomeModel(tol=tol).fit(diff[train], hist.loc[train, "FTR"])
        probs.append(model.predict_proba(diff[train]))
    assert np.allclose(probs[0], probs[1], atol=atol, rtol=0)


def test_elo_k20_exploratory_exposure_record(matches, golden):
    """Reproduces the first (pre-tuning) test-set exposure listed in the access log.

    That exploratory model used K=20 AND applied the home advantage inside the rating updates.
    """
    require_registered_dev_test_spec("elo_k20_exploratory")
    shift = home_shift_from_results(matches.loc[matches["Season"].isin(TRAIN_SEASONS), "FTR"])
    hist = run_elo(matches, k=20, home_adv=shift)
    train = hist["Season"].isin(TRAIN_SEASONS).to_numpy()
    test = hist["Season"].isin(DEV_TEST_SEASONS).to_numpy()
    diff = elo_difference(hist, shift)
    probs = EloOutcomeModel().fit(diff[train], hist.loc[train, "FTR"]).predict_proba(diff[test])
    s = score(hist.loc[test, "FTR"], probs)
    assert close(s["log_loss"], golden["elo_k20_exploratory"]["log_loss"])
    assert close(s["brier"], golden["elo_k20_exploratory"]["brier"])


def test_dixon_coles_rho_search(matches, golden):
    g = golden["dixon_coles_rho_search"]
    out = dixon_coles_rho_search.run(write=False)
    assert list(out["grid"]["rho"]) == g["rho"]
    assert close(out["grid"]["log_likelihood"].to_numpy(), np.array(g["log_likelihood"]), LOGLIK_TOL)
    assert close(out["log_likelihood_rho_0"], g["log_likelihood_rho_0"], LOGLIK_TOL)
    assert out["best_rho"] == g["best_rho"] == -0.04
    assert out["configured_rho_matches_selection"]
    assert close(out["poisson_is_home_coef"], golden["poisson_static_v1"]["is_home_coef"])


def test_common_subset(goal_results, golden):
    g = golden["common_subset"]
    assert goal_results["unseen_teams"] == g["unseen_teams"]
    assert goal_results["n_excluded_matches"] == g["n_excluded"] == 112
    assert goal_results["n_common_matches"] == g["n_common"] == 648


@pytest.mark.parametrize("key,golden_key", [
    ("poisson_static", "poisson_static_v1"),
    ("dixon_coles_staged", "dixon_coles_staged_v1"),
    ("elo_same_matches", "elo_k25_logreg_v1_common_subset"),
])
def test_common_subset_scores(goal_results, golden, key, golden_key):
    assert goal_results[key]["n_matches"] == 648
    assert close(goal_results[key]["log_loss"], golden[golden_key]["log_loss"])
    assert close(goal_results[key]["brier"], golden[golden_key]["brier"])


def test_single_fixture_probabilities(matches, golden):
    train = matches[matches["Season"].isin(TRAIN_SEASONS)]
    lam, mu = PoissonGoalModel().fit(train).predict_rates(["Man City"], ["Sheffield United"])
    poisson_probs = outcome_probabilities_from_rates(lam, mu, rho=0.0)[0]
    dc_probs = outcome_probabilities_from_rates(lam, mu, rho=-0.04)[0]
    assert close(poisson_probs, np.array(golden["poisson_static_v1"]["man_city_v_sheffield_united"]))
    assert close(dc_probs, np.array(golden["dixon_coles_staged_v1"]["man_city_v_sheffield_united"]))


def test_promoted_team_folds(matches, golden):
    out = promoted_team_folds.run(write=False)
    cols = ["val_season", "n_recent_yoyo", "n_long_absence_or_newcomer", "n_mixed", "n_total_affected"]
    assert out["folds"][cols].to_dict(orient="records") == golden["promoted_team_folds"]
    assert out["pooled"] == {"n_recent_yoyo": 34, "n_long_absence_or_newcomer": 404, "n_mixed": 4,
                             "n_total_affected": 442}


@pytest.mark.parametrize("value,documented", [
    ("elo_k25_logreg_v1.log_loss", 0.9527), ("elo_k25_logreg_v1.brier", 0.5642),
    ("elo_k25_logreg_v1.home_shift", 43.0774),
    ("poisson_static_v1.log_loss", 1.0058), ("poisson_static_v1.brier", 0.5989),
    ("elo_k25_logreg_v1_common_subset.log_loss", 0.9547), ("elo_k25_logreg_v1_common_subset.brier", 0.5661),
    ("dixon_coles_staged_v1.log_loss", 1.0072), ("dixon_coles_staged_v1.brier", 0.5994),
])
def test_documented_rounded_values(golden, value, documented):
    section, key = value.split(".")
    assert round(golden[section][key], 4) == documented
