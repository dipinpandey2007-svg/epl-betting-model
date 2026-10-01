"""Experiment 2b: score the frozen Elo baseline and the frequency baseline on the development test benchmark.

This re-scores registered specifications only (see docs/TEST_SET_ACCESS_LOG.md);
it reproduces recorded results and adds no new test-set exposure.
"""

import pandas as pd

from eplmodel.config import load_config
from eplmodel.constants import OUTCOMES
from eplmodel.data import load_matches
from eplmodel.evaluation.baselines import frequency_baseline
from eplmodel.evaluation.calibration import calibration_table
from eplmodel.evaluation.metrics import score
from eplmodel.models.elo import EloOutcomeModel, elo_difference, home_shift_from_results, run_elo
from eplmodel.paths import RESULTS_DIR
from eplmodel.reporting.plots import plot_calibration
from eplmodel.reporting.results import write_results
from eplmodel.splits import DEV_TEST_SEASONS, TRAIN_SEASONS, require_registered_dev_test_spec

NAME = "elo_dev_test"


def elo_predictions(matches: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, float]:
    """Frozen Elo baseline: (H, D, A) predictions for development-test matches, indexed by match_id."""
    require_registered_dev_test_spec(cfg["spec_id"])
    hist = run_elo(matches, k=cfg["selected_k"], home_adv=cfg["update_home_advantage"],
                   initial_rating=cfg["initial_rating"])
    train = hist["Season"].isin(TRAIN_SEASONS).to_numpy()
    test = hist["Season"].isin(DEV_TEST_SEASONS).to_numpy()
    shift = home_shift_from_results(hist.loc[train, "FTR"]) if cfg["apply_home_shift"] else 0.0
    diff = elo_difference(hist, shift)
    model = EloOutcomeModel(C=cfg["logistic_C"]).fit(diff[train], hist.loc[train, "FTR"])
    preds = pd.DataFrame(model.predict_proba(diff[test]), columns=list(OUTCOMES), index=hist.loc[test, "match_id"])
    return preds, shift


def run(write: bool = True) -> dict:
    config = load_config()
    cfg = config["elo"]
    matches = load_matches()
    preds, shift = elo_predictions(matches, cfg)
    test = matches[matches["Season"].isin(DEV_TEST_SEASONS)]
    train = matches[matches["Season"].isin(TRAIN_SEASONS)]

    require_registered_dev_test_spec(config["baseline"]["spec_id"])
    base = frequency_baseline(train["FTR"], len(test))

    calib = calibration_table(preds["H"].to_numpy(), (test["FTR"] == "H").to_numpy())
    result = {
        "spec_id": cfg["spec_id"],
        "home_shift": shift,
        "elo": score(test["FTR"], preds.loc[test["match_id"]].to_numpy()),
        "frequency_baseline": score(test["FTR"], base),
        "home_win_calibration": calib.reset_index().astype({"bin": str}),
    }
    if write:
        write_results(NAME, result)
        plot_calibration(calib, f"Elo K={cfg['selected_k']} (P(Home))",
                         "Calibration on development test benchmark (2022-24)",
                         RESULTS_DIR / NAME / "calibration_home.png")
    return result


if __name__ == "__main__":
    out = run()
    print("Elo:", out["elo"])
    print("Frequency baseline:", out["frequency_baseline"])
