"""Validity rules and missing-data accounting for one odds snapshot (protocol market_benchmark_v1 [validity]).

A match's price set is valid only if all three odds are present, numeric, finite and > 1.0, and the booksum
1/H + 1/D + 1/A lies in [min_booksum, max_booksum]. An invalid set gets the FIRST failing reason in this
order and no probability; nothing is imputed or carried over from another match, snapshot or bookmaker.
"""

import numpy as np
import pandas as pd

from eplmodel.market.odds import Snapshot, snapshot_columns

REASONS = ("missing", "non_numeric", "odds_not_above_1", "booksum_below_min", "booksum_above_max")
VALID = "valid"


def classify(odds_table: pd.DataFrame, snapshot: Snapshot, min_booksum: float, max_booksum: float) -> pd.Series:
    """Validity label per row: 'valid' or the first failing reason, indexed like odds_table."""
    raw = odds_table[snapshot_columns(snapshot)]
    missing = raw.isna().any(axis=1)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    non_numeric = ~missing & numeric.isna().any(axis=1)
    values = numeric.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    not_above_1 = ~missing & ~non_numeric & ~(finite & (values > 1.0).all(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        booksum = (1.0 / values).sum(axis=1)
    ok_so_far = ~(missing | non_numeric | not_above_1)
    below = ok_so_far & (booksum < min_booksum)
    above = ok_so_far & (booksum > max_booksum)
    label = np.select([missing, non_numeric, not_above_1, below, above], list(REASONS), default=VALID)
    return pd.Series(label, index=odds_table.index, name=f"{snapshot.name}_validity")


def coverage_table(odds_table: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per season: matches, valid count, valid share and the count of each invalid reason."""
    df = pd.DataFrame({"Season": odds_table["Season"].to_numpy(), "label": labels.to_numpy()})
    counts = df.groupby("Season")["label"].value_counts().unstack(fill_value=0)
    for col in (VALID, *REASONS):
        if col not in counts:
            counts[col] = 0
    out = counts[[VALID, *REASONS]].copy()
    out.insert(0, "n_matches", out.sum(axis=1))
    out["valid_share"] = out[VALID] / out["n_matches"]
    return out.reset_index()


def booksum_summary(odds_table: pd.DataFrame, snapshot: Snapshot, labels: pd.Series) -> pd.DataFrame:
    """Per season: min, median, mean and max booksum of the valid price sets (outcome-free)."""
    valid = labels.to_numpy() == VALID
    values = odds_table.loc[valid, snapshot_columns(snapshot)].apply(pd.to_numeric).to_numpy(dtype=float)
    b = pd.Series((1.0 / values).sum(axis=1), name="booksum")
    seasons = odds_table.loc[valid, "Season"].to_numpy()
    return b.groupby(seasons).agg(["min", "median", "mean", "max"]).rename_axis("Season").reset_index()
