"""Proper scoring rules for (H, D, A) probability forecasts.

Both functions take the column order explicitly (default OUTCOMES = H, D, A)
instead of relying on sklearn's alphabetical class ordering.

Brier score convention used throughout the project: the squared error summed
over the three outcomes, averaged over matches. It ranges from 0 (perfect) to
2; a uniform 1/3 forecast scores 2/3.
"""

import numpy as np

from eplmodel.constants import OUTCOMES

_EPS = np.finfo(float).eps  # same clipping as sklearn.metrics.log_loss


def _check(results, probs, outcomes) -> tuple[np.ndarray, np.ndarray]:
    results = np.asarray(results)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape != (len(results), len(outcomes)):
        raise ValueError(f"probs must have shape ({len(results)}, {len(outcomes)}), got {probs.shape}")
    unknown = set(results) - set(outcomes)
    if unknown:
        raise ValueError(f"results contain labels not in outcomes: {sorted(unknown)}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probs must sum to 1")
    return results, probs


def one_hot(results, outcomes=OUTCOMES) -> np.ndarray:
    results = np.asarray(results)
    return np.column_stack([(results == o).astype(float) for o in outcomes])


def multiclass_log_loss(results, probs, outcomes=OUTCOMES) -> float:
    """Mean negative log probability assigned to the observed outcome."""
    results, probs = _check(results, probs, outcomes)
    idx = np.array([outcomes.index(r) for r in results])
    p_observed = np.clip(probs[np.arange(len(results)), idx], _EPS, 1.0)
    return float(-np.mean(np.log(p_observed)))


def multiclass_brier(results, probs, outcomes=OUTCOMES) -> float:
    """Squared error summed over outcomes, averaged over matches."""
    results, probs = _check(results, probs, outcomes)
    return float(((probs - one_hot(results, outcomes)) ** 2).sum(axis=1).mean())


def per_match_log_loss(results, probs, outcomes=OUTCOMES) -> np.ndarray:
    """Negative log probability of the observed outcome, one value per match (mean = multiclass_log_loss)."""
    results, probs = _check(results, probs, outcomes)
    idx = np.array([outcomes.index(r) for r in results])
    return -np.log(np.clip(probs[np.arange(len(results)), idx], _EPS, 1.0))


def per_match_brier(results, probs, outcomes=OUTCOMES) -> np.ndarray:
    """Squared error summed over outcomes, one value per match (mean = multiclass_brier)."""
    results, probs = _check(results, probs, outcomes)
    return ((probs - one_hot(results, outcomes)) ** 2).sum(axis=1)


def score(results, probs, outcomes=OUTCOMES) -> dict:
    return {
        "n_matches": int(len(results)),
        "log_loss": multiclass_log_loss(results, probs, outcomes),
        "brier": multiclass_brier(results, probs, outcomes),
    }
