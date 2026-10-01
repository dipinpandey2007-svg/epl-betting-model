import numpy as np
import pytest

from eplmodel.models.scoreline import (
    MAX_GOALS,
    captured_mass,
    dixon_coles_grid,
    independent_poisson_grid,
    outcome_probabilities,
    outcome_probabilities_from_rates,
)


def test_default_truncation_is_ten_goals():
    assert MAX_GOALS == 10
    assert independent_poisson_grid(1.5, 1.2).shape == (11, 11)


def test_grid_orientation_rows_are_home_goals():
    grid = independent_poisson_grid(3.0, 0.3)
    assert grid[3, 0] > grid[0, 3]
    probs = outcome_probabilities(grid)
    assert probs[0] > probs[2]  # strong home side -> P(H) > P(A)


def test_equal_rates_give_symmetric_outcomes():
    p = outcome_probabilities(independent_poisson_grid(1.3, 1.3))
    assert p[0] == pytest.approx(p[2])


def test_probabilities_sum_to_one_after_truncation():
    p = outcome_probabilities(independent_poisson_grid(4.0, 3.5))
    assert p.sum() == pytest.approx(1.0, abs=1e-15)
    assert captured_mass(4.0, 3.5) < 1.0  # some mass lies beyond 10 goals


def test_renormalisation_close_to_untruncated():
    truncated = outcome_probabilities(independent_poisson_grid(2.5, 1.5, max_goals=10))
    full = outcome_probabilities(independent_poisson_grid(2.5, 1.5, max_goals=40))
    assert np.allclose(truncated, full, atol=1e-4)


def test_rho_zero_equals_independent_grid():
    assert np.array_equal(dixon_coles_grid(1.5, 1.2, 0.0), independent_poisson_grid(1.5, 1.2))


def test_negative_rho_raises_draw_probability():
    lam, mu = [1.3], [1.1]
    p_ind = outcome_probabilities_from_rates(lam, mu, rho=0.0)[0]
    p_dc = outcome_probabilities_from_rates(lam, mu, rho=-0.1)[0]
    assert p_dc[1] > p_ind[1]
