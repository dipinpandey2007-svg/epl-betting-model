"""Scoreline probability grids and their conversion to H/D/A probabilities.

grid[h, a] = P(home scores h, away scores a), for h, a = 0..max_goals.

Truncation: the grid stops at max_goals (default 10) per side. The H/D/A
probabilities are renormalised by the captured mass, i.e. the small
probability of more than 10 goals for a side is redistributed proportionally.
"""

import numpy as np
from scipy.stats import poisson

from eplmodel.models.dixon_coles import tau_matrix

MAX_GOALS = 10


def independent_poisson_grid(lam: float, mu: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))


def dixon_coles_grid(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Independent Poisson grid multiplied by the Dixon-Coles low-score correction. rho=0 gives the Poisson grid."""
    return tau_matrix(lam, mu, rho, max_goals) * independent_poisson_grid(lam, mu, max_goals)


def outcome_probabilities(grid: np.ndarray) -> np.ndarray:
    """(P(H), P(D), P(A)) from a scoreline grid, renormalised to sum to 1."""
    p_home = np.tril(grid, k=-1).sum()  # rows are home goals: h > a lies below the diagonal
    p_draw = np.trace(grid)
    p_away = np.triu(grid, k=1).sum()
    total = p_home + p_draw + p_away
    return np.array([p_home, p_draw, p_away]) / total


def outcome_probabilities_from_rates(lam, mu, rho: float = 0.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    """(n, 3) array of (H, D, A) probabilities for arrays of expected goals."""
    return np.array([outcome_probabilities(dixon_coles_grid(l, m, rho, max_goals)) for l, m in zip(lam, mu)])


def captured_mass(lam: float, mu: float, rho: float = 0.0, max_goals: int = MAX_GOALS) -> float:
    """Probability mass inside the truncated grid (before renormalisation)."""
    return float(dixon_coles_grid(lam, mu, rho, max_goals).sum())
