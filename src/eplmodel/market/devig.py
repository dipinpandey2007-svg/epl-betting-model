"""Implied probabilities from decimal odds and the pre-registered margin-removal methods.

For one match with decimal odds o_k (k = H, D, A), the raw implied probabilities are pi_k = 1 / o_k. Their
sum B (the booksum) exceeds 1 by the bookmaker's margin. Each method maps pi to probabilities p that sum to 1:

- proportional: p_k = pi_k / B. The margin is removed in proportion to each probability.
- power: p_k = pi_k ** c, with c >= 1 chosen so that sum_k p_k = 1. Raising to a power > 1 shrinks small
  probabilities relatively more than large ones, so more margin is removed from longshots.
- Shin: p_k = (sqrt(z**2 + 4 (1 - z) pi_k**2 / B) - z) / (2 (1 - z)), with z in [0, 1) chosen so that
  sum_k p_k = 1. z is the share of money the bookmaker attributes to insiders in Shin's model; the
  resulting correction is also larger for longshots.

If B = 1 exactly, all three return p = pi. B < 1 is not a valid single-bookmaker book (the caller excludes it).
The roots are found with brentq, then p is divided by its sum (a correction of order 1e-15).
"""

import math

import numpy as np
from scipy.optimize import brentq

XTOL = 1e-15
MAXITER = 500
POWER_C_UPPER = 100.0


class MarginRemovalError(ValueError):
    """The odds cannot be turned into probabilities by the requested method."""


def implied_probabilities(odds) -> np.ndarray:
    """1 / odds for one match's three decimal odds; every price must be finite and > 1."""
    o = np.asarray(odds, dtype=float)
    if o.shape != (3,) or not np.all(np.isfinite(o)) or np.any(o <= 1.0):
        raise MarginRemovalError(f"need three finite decimal odds > 1, got {odds!r}")
    return 1.0 / o


def _check_booksum(pi: np.ndarray) -> float:
    b = float(pi.sum())
    if b < 1.0:
        raise MarginRemovalError(f"booksum {b} < 1 is not a valid bookmaker book")
    return b


def _normalise(p: np.ndarray) -> np.ndarray:
    return p / p.sum()


def proportional(pi) -> tuple[np.ndarray, dict]:
    pi = np.asarray(pi, dtype=float)
    b = _check_booksum(pi)
    return pi / b, {"booksum": b}


def power(pi, xtol: float = XTOL, maxiter: int = MAXITER, c_upper: float = POWER_C_UPPER) -> tuple[np.ndarray, dict]:
    pi = np.asarray(pi, dtype=float)
    b = _check_booksum(pi)
    if b == 1.0:
        return pi.copy(), {"booksum": b, "power_c": 1.0}
    c = brentq(lambda k: float(np.sum(pi ** k)) - 1.0, 1.0, c_upper, xtol=xtol, maxiter=maxiter)
    return _normalise(pi ** c), {"booksum": b, "power_c": float(c)}


def shin_probabilities(pi: np.ndarray, z: float) -> np.ndarray:
    b = float(pi.sum())
    return (np.sqrt(z ** 2 + 4.0 * (1.0 - z) * pi ** 2 / b) - z) / (2.0 * (1.0 - z))


def shin(pi, xtol: float = XTOL, maxiter: int = MAXITER) -> tuple[np.ndarray, dict]:
    pi = np.asarray(pi, dtype=float)
    b = _check_booksum(pi)
    if b == 1.0:
        return pi.copy(), {"booksum": b, "shin_z": 0.0}
    # At z = 0 the sum is sqrt(B) > 1; as z -> 1 it tends to sum(pi^2) / B < 1, so a root lies in (0, 1).
    f = lambda z: float(shin_probabilities(pi, z).sum()) - 1.0
    z = brentq(f, 0.0, 1.0 - 1e-12, xtol=xtol, maxiter=maxiter)
    return _normalise(shin_probabilities(pi, z)), {"booksum": b, "shin_z": float(z)}


METHODS = {"shin": shin, "proportional": proportional, "power": power}


def remove_margin(odds, method: str) -> tuple[np.ndarray, dict]:
    """(H, D, A) probabilities and solver diagnostics for one match's decimal odds."""
    if method not in METHODS:
        raise MarginRemovalError(f"unknown method {method!r}; registered: {sorted(METHODS)}")
    p, info = METHODS[method](implied_probabilities(odds))
    if not (np.all(np.isfinite(p)) and np.all(p > 0) and math.isclose(p.sum(), 1.0, abs_tol=1e-12)):
        raise MarginRemovalError(f"{method} produced invalid probabilities {p}")
    return p, info
