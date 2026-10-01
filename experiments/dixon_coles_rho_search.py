"""Experiment 5: staged Dixon-Coles rho grid search (training seasons only).

The static Poisson model is fitted on the training seasons, its coefficients
are held fixed, and rho is chosen by maximising the training log-likelihood.
"""

import numpy as np

from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.models.dixon_coles import rho_bounds, rho_grid_search, staged_log_likelihood
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.paths import RESULTS_DIR
from eplmodel.reporting.plots import plot_rho_search
from eplmodel.reporting.results import write_results
from eplmodel.splits import TRAIN_SEASONS, assert_no_dev_test

NAME = "dixon_coles_rho_search"


def run(write: bool = True) -> dict:
    cfg = load_config()["dixon_coles"]
    matches = load_matches()
    assert_no_dev_test(TRAIN_SEASONS)
    train = matches[matches["Season"].isin(TRAIN_SEASONS)]

    goal_model = PoissonGoalModel().fit(train)
    lam, mu = goal_model.predict_rates(train["HomeTeam"], train["AwayTeam"])
    rhos = np.arange(cfg["rho_grid_start"], cfg["rho_grid_stop"], cfg["rho_grid_step"])
    grid, best_rho = rho_grid_search(train["FTHG"], train["FTAG"], lam, mu, rhos)

    result = {
        "poisson_is_home_coef": goal_model.is_home_coef,
        "n_train_matches": len(train),
        "valid_rho_range": rho_bounds(lam, mu),
        "log_likelihood_rho_0": staged_log_likelihood(0.0, train["FTHG"], train["FTAG"], lam, mu),
        "grid": grid,
        "best_rho": best_rho,
        "configured_rho": cfg["selected_rho"],
        "configured_rho_matches_selection": best_rho == cfg["selected_rho"],
    }
    if write:
        write_results(NAME, result)
        plot_rho_search(grid, best_rho, RESULTS_DIR / NAME / "rho_search.png")
    return result


if __name__ == "__main__":
    out = run()
    print(out["grid"].to_string(index=False))
    print(f"Best rho: {out['best_rho']} (configured: {out['configured_rho']})")
