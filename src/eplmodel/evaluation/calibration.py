"""Reliability (calibration) tables."""

import numpy as np
import pandas as pd


def calibration_table(predicted, observed, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned mean prediction vs observed frequency for one binary event (e.g. home win).

    gap = observed - predicted; a positive gap means the event happened more
    often than the model said.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    bins = pd.qcut(predicted, q=n_bins, duplicates="drop")
    table = pd.DataFrame({"pred": predicted, "actual": observed, "bin": bins}).groupby("bin", observed=True).agg(
        mean_pred=("pred", "mean"), mean_actual=("actual", "mean"), n=("actual", "size")
    )
    table["gap"] = table["mean_actual"] - table["mean_pred"]
    return table
