"""Historical shot counts from the raw football-data.co.uk files (protocol shots_information_v1).

Columns (data/reference/football_data_shot_columns.md): HS/AS = home/away team shots, HST/AST = home/away
team shots on target. They are POST-MATCH statistics of the match they describe, so they may only inform
forecasts of later matches; the fitting sets of eplmodel.models.shots guarantee that.

read_season_shots() reads only Date, HomeTeam, AwayTeam and the four shot columns. No result column is read.
Holdout seasons are refused and callers pass the seasons their protocol allows. Values are kept as read:
invalid entries become NaN so that the protocol's missing rule can be applied per arm.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from eplmodel.data.download import raw_season_path
from eplmodel.paths import RAW_DIR
from eplmodel.splits import SplitAccessError, assert_not_holdout

KEY_COLUMNS = ("Date", "HomeTeam", "AwayTeam")
SHOT_COLUMNS = ("HS", "AS", "HST", "AST")


def read_season_shots(season: str, allowed_seasons: Iterable[str], raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One season's shots: match_id, Season and HS, AS, HST, AST (float; NaN where missing or invalid)."""
    assert_not_holdout([season])
    if season not in set(allowed_seasons):
        raise SplitAccessError(f"Season {season!r} is not allowed by the shots protocol")
    path = raw_season_path(season, raw_dir)
    raw = pd.read_csv(path, usecols=[*KEY_COLUMNS, *SHOT_COLUMNS], dtype={c: object for c in SHOT_COLUMNS})
    raw = raw.dropna(subset=list(KEY_COLUMNS)).reset_index(drop=True)
    date = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")
    out = pd.DataFrame({"match_id": date.dt.strftime("%Y-%m-%d") + "_" + raw["HomeTeam"] + "_" + raw["AwayTeam"],
                        "Season": season})
    for col in SHOT_COLUMNS:
        values = pd.to_numeric(raw[col], errors="coerce").astype(float)
        valid = np.isfinite(values) & (values >= 0) & (values % 1 == 0)
        out[col] = values.where(valid)
    if not out["match_id"].is_unique:
        raise ValueError(f"{path.name}: duplicate match ids")
    return out


def read_shots(seasons: Sequence[str], allowed_seasons: Iterable[str], raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    allowed = list(allowed_seasons)
    return pd.concat([read_season_shots(s, allowed, raw_dir) for s in seasons], ignore_index=True)


def attach_shots(matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """The match table with HS, AS, HST, AST joined by match_id; every match must have a shots row."""
    missing = set(matches["match_id"]) - set(shots["match_id"])
    if missing:
        raise ValueError(f"{len(missing)} matches have no shots row")
    return matches.merge(shots[["match_id", *SHOT_COLUMNS]], on="match_id", how="left", validate="one_to_one")
