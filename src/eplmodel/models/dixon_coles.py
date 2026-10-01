"""Dixon-Coles (1997) low-score dependence correction, fitted in a STAGED way.

P(X=x, Y=y) = tau(x, y; lam, mu, rho) * Poisson(x; lam) * Poisson(y; mu)

tau only changes the 0-0, 1-0, 0-1 and 1-1 scorelines, and the changes cancel
so the full (untruncated) distribution still sums to 1.

Staged fit (what this project currently does; NOT a full joint MLE):
1. fit the static Poisson GLM on training data;
2. hold its attack/defence/home coefficients fixed;
3. choose rho by maximising the training log-likelihood over a grid.
A joint fit would estimate all parameters together, and could move the team
coefficients as well as rho.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

RHO_GRID = np.arange(-0.30, 0.30, 0.02)  # -0.30 .. 0.28, as used for the recorded search


def tau_correction(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction for home goals x, away goals y; lam/mu are home/away expected goals."""
    if x == 0 and y == 0:
        return 1 - (lam * mu * rho)
    if x == 0 and y == 1:
        return 1 + (lam * rho)
    if x == 1 and y == 0:
        return 1 + (mu * rho)
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def tau_matrix(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    tau = np.ones((max_goals + 1, max_goals + 1))
    for x in (0, 1):
        for y in (0, 1):
            tau[x, y] = tau_correction(x, y, lam, mu, rho)
    return tau


def tau_vector(home_goals, away_goals, lam, mu, rho: float) -> np.ndarray:
    """Element-wise tau for arrays of observed scores and expected goals."""
    x, y = np.asarray(home_goals), np.asarray(away_goals)
    lam, mu = np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)
    tau = np.ones(len(x))
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def rho_bounds(lam, mu) -> tuple[float, float]:
    """Range of rho for which every tau is non-negative for all the given fixtures.

    Needs 1 - lam*mu*rho >= 0, 1 + lam*rho >= 0, 1 + mu*rho >= 0 and 1 - rho >= 0.
    """
    lam, mu = np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)
    lower = float(np.max(np.maximum(-1 / lam, -1 / mu)))
    upper = float(min(np.min(1 / (lam * mu)), 1.0))
    return lower, upper


def staged_log_likelihood(rho: float, home_goals, away_goals, lam, mu) -> float:
    """Log-likelihood of observed scores with the Poisson rates held fixed.

    Returns -inf if rho makes any probability non-positive, so grid searches
    can never select an invalid rho (the original loop would have produced NaN,
    which np.argmax treats as the maximum).
    """
    probs = tau_vector(home_goals, away_goals, lam, mu, rho) * poisson.pmf(home_goals, lam) * poisson.pmf(away_goals, mu)
    if np.any(probs <= 0):
        return -np.inf
    return float(np.sum(np.log(probs)))


def rho_grid_search(home_goals, away_goals, lam, mu, rhos=RHO_GRID) -> tuple[pd.DataFrame, float]:
    """Training log-likelihood for each candidate rho, and the best *valid* rho (rounded to 2 dp).

    A rho outside rho_bounds(lam, mu) makes some fixture's scoreline
    distribution invalid (a negative probability for 0-0, 0-1, 1-0 or 1-1).
    Its likelihood can still be finite when no such match ended with the
    affected score, so validity is reported separately and invalid values are
    never selected.
    """
    rhos = np.asarray(rhos, dtype=float)
    log_lik = np.array([staged_log_likelihood(r, home_goals, away_goals, lam, mu) for r in rhos])
    lower, upper = rho_bounds(lam, mu)
    valid = (rhos >= lower) & (rhos <= upper) & np.isfinite(log_lik)
    if not valid.any():
        raise ValueError("No candidate rho gives a valid likelihood.")
    best = round(float(rhos[int(np.argmax(np.where(valid, log_lik, -np.inf)))]), 2)
    table = pd.DataFrame({"rho": np.round(rhos, 2), "log_likelihood": log_lik, "valid_for_all_fixtures": valid})
    return table, best
