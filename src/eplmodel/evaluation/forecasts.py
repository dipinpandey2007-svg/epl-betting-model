"""The standard forecast format of the evaluation harness, and how outcomes are joined to it.

A *forecast frame* is a DataFrame indexed by `match_id` (unique), holding for each arm the
columns `<arm>_H`, `<arm>_D`, `<arm>_A` in that order (eplmodel.constants.OUTCOMES), each row a
probability distribution. It never contains a result column: predicting and scoring are separate
steps, and outcomes are joined by match_id only when scoring (outcomes_for).
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from eplmodel.constants import OUTCOMES

# Columns that would reveal a match's outcome; a forecast frame must not contain them.
RESULT_COLUMNS = frozenset({"FTR", "FTHG", "FTAG"})


class ForecastFormatError(ValueError):
    """A forecast frame does not follow the standard format."""


def prob_columns(arm: str) -> list[str]:
    """The probability columns of one arm, in (H, D, A) order."""
    return [f"{arm}_{o}" for o in OUTCOMES]


def probabilities(preds: pd.DataFrame, arm: str) -> np.ndarray:
    """The (n, 3) probability array of one arm, columns in (H, D, A) order."""
    return preds[prob_columns(arm)].to_numpy(dtype=float)


def outcomes_for(matches: pd.DataFrame, match_ids: Iterable[str]) -> pd.Series:
    """Observed results (H/D/A) for the given match ids, in that order."""
    return matches.set_index("match_id").loc[list(match_ids), "FTR"]


def check_forecast_frame(preds: pd.DataFrame, arms: Sequence[str], atol: float = 1e-9) -> None:
    """Raise ForecastFormatError unless `preds` is a valid forecast frame for `arms`.

    Checks: index named match_id with unique values; no result column; every arm has its three
    (H, D, A) columns; probabilities are finite, in [0, 1] and each row sums to 1 within atol.
    """
    if preds.index.name != "match_id":
        raise ForecastFormatError("a forecast frame must be indexed by match_id")
    if not preds.index.is_unique:
        raise ForecastFormatError("match_id values must be unique")
    leaked = sorted(RESULT_COLUMNS & set(preds.columns))
    if leaked:
        raise ForecastFormatError(f"a forecast frame must not contain result columns: {leaked}")
    for arm in arms:
        missing = [c for c in prob_columns(arm) if c not in preds.columns]
        if missing:
            raise ForecastFormatError(f"arm {arm!r} lacks columns {missing}")
        p = probabilities(preds, arm)
        if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
            raise ForecastFormatError(f"arm {arm!r} has probabilities outside [0, 1] or not finite")
        if not np.allclose(p.sum(axis=1), 1.0, atol=atol, rtol=0.0):
            raise ForecastFormatError(f"arm {arm!r} has rows that do not sum to 1")
