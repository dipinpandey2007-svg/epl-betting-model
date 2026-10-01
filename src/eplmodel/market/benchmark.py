"""Market arms as forecast frames for the common evaluation harness.

An arm is one (snapshot, method) pair, e.g. market_close_shin = Pinnacle closing odds with Shin margin
removal. market_forecasts() returns, for the matches whose snapshot is valid, a frame indexed by match_id
with Date, Season, HomeTeam, AwayTeam and <arm>_H/D/A (eplmodel.evaluation.forecasts), plus the solver
diagnostics. It uses only the odds; no outcome is read.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eplmodel.evaluation.forecasts import check_forecast_frame, prob_columns
from eplmodel.market.coverage import VALID
from eplmodel.market.devig import remove_margin
from eplmodel.market.odds import SNAPSHOTS, Snapshot, snapshot_columns

SNAPSHOT_PREFIX = {"closing": "close", "pre_closing": "pre"}


@dataclass(frozen=True)
class MarketArm:
    snapshot: Snapshot
    method: str

    @property
    def name(self) -> str:
        return f"market_{SNAPSHOT_PREFIX[self.snapshot.name]}_{self.method}"


def arm_from_name(name: str) -> MarketArm:
    """'market_close_shin' -> MarketArm(PINNACLE_CLOSING, 'shin')."""
    prefix, snap, method = name.split("_", 2)
    by_prefix = {v: k for k, v in SNAPSHOT_PREFIX.items()}
    if prefix != "market" or snap not in by_prefix:
        raise ValueError(f"{name!r} is not a market arm name")
    return MarketArm(SNAPSHOTS[by_prefix[snap]], method)


def market_forecasts(odds_table: pd.DataFrame, labels: pd.Series, arm: MarketArm) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast frame of one arm on its valid matches, and per-match diagnostics (booksum, z or c)."""
    valid = (labels.to_numpy() == VALID)
    rows = odds_table[valid]
    odds = rows[snapshot_columns(arm.snapshot)].apply(pd.to_numeric).to_numpy(dtype=float)
    probs, info = [], []
    for o in odds:
        p, d = remove_margin(o, arm.method)
        probs.append(p)
        info.append(d)
    index = pd.Index(rows["match_id"], name="match_id")
    frame = rows.set_index("match_id")[["Date", "Season", "HomeTeam", "AwayTeam"]].copy()
    frame[prob_columns(arm.name)] = np.array(probs).reshape(-1, 3)
    check_forecast_frame(frame, [arm.name], atol=1e-12)
    return frame, pd.DataFrame(info, index=index)
