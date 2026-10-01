import numpy as np
import pytest
from scipy.stats import poisson

from eplmodel.models.dixon_coles import (
    rho_bounds,
    rho_grid_search,
    staged_log_likelihood,
    tau_correction,
    tau_matrix,
    tau_vector,
)
from eplmodel.models.scoreline import dixon_coles_grid


@pytest.mark.parametrize("x,y", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 3)])
def test_rho_zero_means_no_correction(x, y):
    assert tau_correction(x, y, lam=1.5, mu=1.2, rho=0) == 1.0


@pytest.mark.parametrize("x,y,expected", [
    (0, 0, 1.18), (1, 0, 0.88), (0, 1, 0.85), (1, 1, 1.10), (2, 2, 1.00),
])
def test_documented_tau_values(x, y, expected):
    """Hand-computed values recorded in RESULTS_LOG Experiment 4 (lam=1.5, mu=1.2, rho=-0.1)."""
    assert tau_correction(x, y, 1.5, 1.2, -0.1) == pytest.approx(expected)


def test_tau_vector_matches_scalar():
    rng = np.random.default_rng(1)
    x, y = rng.integers(0, 4, 50), rng.integers(0, 4, 50)
    lam, mu = rng.uniform(0.5, 3, 50), rng.uniform(0.5, 3, 50)
    expected = [tau_correction(a, b, l, m, -0.07) for a, b, l, m in zip(x, y, lam, mu)]
    assert np.allclose(tau_vector(x, y, lam, mu, -0.07), expected)


def test_tau_matrix_only_touches_low_scores():
    tau = tau_matrix(1.5, 1.2, -0.1, max_goals=5)
    assert tau[0, 0] == pytest.approx(1.18) and tau[1, 1] == pytest.approx(1.1)
    mask = np.ones_like(tau, dtype=bool)
    mask[:2, :2] = False
    assert (tau[mask] == 1.0).all()


@pytest.mark.parametrize("rho", [-0.15, -0.04, 0.0, 0.1])
def test_correction_preserves_total_probability(rho):
    assert dixon_coles_grid(1.6, 1.1, rho, max_goals=40).sum() == pytest.approx(1.0, abs=1e-12)


def test_log_likelihood_at_rho_zero_is_independent_poisson():
    x, y = np.array([0, 1, 2, 1]), np.array([0, 1, 0, 3])
    lam, mu = np.array([1.4, 1.1, 2.0, 0.9]), np.array([1.0, 1.3, 0.7, 1.5])
    expected = np.sum(poisson.logpmf(x, lam) + poisson.logpmf(y, mu))
    assert staged_log_likelihood(0.0, x, y, lam, mu) == pytest.approx(expected)


def test_rho_bounds():
    lam, mu = np.array([2.0, 1.0]), np.array([1.0, 0.5])
    lower, upper = rho_bounds(lam, mu)
    assert lower == pytest.approx(-0.5)  # -1/lam for lam=2 is the binding constraint
    assert upper == pytest.approx(0.5)   # 1/(lam*mu) = 0.5 for the first fixture


def test_invalid_rho_gives_minus_infinity_and_is_never_selected():
    # A 0-1 result with lam=4: tau(0,1) = 1 + 4*rho < 0 for rho < -0.25.
    x, y, lam, mu = np.array([0]), np.array([1]), np.array([4.0]), np.array([1.0])
    assert staged_log_likelihood(-0.3, x, y, lam, mu) == -np.inf
    table, best = rho_grid_search(x, y, lam, mu, rhos=np.array([-0.3, -0.2, 0.0]))
    assert best >= rho_bounds(lam, mu)[0]
    assert not table.loc[table["rho"] == -0.3, "valid_for_all_fixtures"].item()
