"""Naive reference forecasts."""

import numpy as np
import pandas as pd

from eplmodel.constants import OUTCOMES


def frequency_baseline(train_results: pd.Series, n: int) -> np.ndarray:
    """Every match gets the training-set H/D/A frequencies, as an (n, 3) array in OUTCOMES order."""
    freqs = pd.Series(train_results).value_counts(normalize=True).reindex(list(OUTCOMES)).fillna(0.0).to_numpy()
    return np.tile(freqs, (n, 1))
