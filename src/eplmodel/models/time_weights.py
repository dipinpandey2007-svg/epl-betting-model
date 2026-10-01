"""Deterministic recency weights for historical matches (protocol time_weighted_poisson_v1).

    w_i = 2 ** (-age_i / H) = exp(-ln(2) * age_i / H)

age_i is the number of whole days between match i and a reference date, and
H is the half-life in days: a match H days older than another gets half its
weight. H = inf gives every match weight 1 (no weighting).

The reference date does not change a weighted maximum-likelihood fit: moving
it multiplies every weight by the same constant, and the weighted score
equations are unchanged by a common factor. The protocol uses the latest
history date, so weights lie in (0, 1].
"""

import math

import numpy as np
import pandas as pd


def match_age_days(dates, reference_date) -> np.ndarray:
    """Whole days from each match date to the reference date. Raises if any match is after it."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    ref = pd.Timestamp(reference_date).normalize()
    age = (ref - dates).dt.days.to_numpy(dtype=float)
    if np.any(age < 0):
        raise ValueError("a match is dated after the reference date")
    return age


def exponential_decay_weights(dates, reference_date, half_life_days: float) -> np.ndarray:
    """Weight 2 ** (-age / half_life_days) per match; all ones when half_life_days is inf."""
    half_life_days = float(half_life_days)
    if not half_life_days > 0:  # also rejects NaN
        raise ValueError("half_life_days must be positive (inf for no weighting)")
    age = match_age_days(dates, reference_date)
    if math.isinf(half_life_days):
        return np.ones(len(age))
    return np.exp2(-age / half_life_days)


def kish_effective_sample_size(weights) -> float:
    """(sum w)^2 / sum w^2: the number of equally weighted observations carrying the same information."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if w.size else 0.0
