"""Odds ingestion: snapshot definitions and reading raw odds columns without any outcome.

Column meanings are taken from football-data.co.uk's notes (data/reference/football_data_odds_columns.md):
PSH/PSD/PSA are Pinnacle PRE-CLOSING odds, PSCH/PSCD/PSCA Pinnacle CLOSING odds. The two snapshots are
separate objects and are never combined.

read_season_odds() reads only the key columns (Date, HomeTeam, AwayTeam) and the requested odds columns,
so no result, score or post-match statistic can enter a market probability. Holdout seasons are refused,
and callers pass the seasons their protocol allows.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eplmodel.data.download import raw_season_path
from eplmodel.paths import RAW_DIR
from eplmodel.splits import SplitAccessError, assert_not_holdout

KEY_COLUMNS = ("Date", "HomeTeam", "AwayTeam")


@dataclass(frozen=True)
class Snapshot:
    """One bookmaker's 1X2 prices at one point in time; columns in (H, D, A) order."""

    name: str
    bookmaker: str
    columns: tuple[str, str, str]
    description: str


PINNACLE_CLOSING = Snapshot("closing", "Pinnacle", ("PSCH", "PSCD", "PSCA"),
                            "Pinnacle closing odds ('C' suffix); exact time not documented")
PINNACLE_PRE_CLOSING = Snapshot("pre_closing", "Pinnacle", ("PSH", "PSD", "PSA"),
                                "Pinnacle pre-closing odds; per-match collection time not documented")
SNAPSHOTS = {s.name: s for s in (PINNACLE_CLOSING, PINNACLE_PRE_CLOSING)}


def snapshot_columns(snapshot: Snapshot) -> list[str]:
    """Prefixed odds columns of a snapshot in an odds table, e.g. 'closing_odds_H'."""
    return [f"{snapshot.name}_odds_{o}" for o in ("H", "D", "A")]


def read_season_odds(season: str, snapshots: Sequence[Snapshot], allowed_seasons: Iterable[str],
                     raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One season's odds table: match_id, Date, Season, HomeTeam, AwayTeam and each snapshot's three odds.

    Values are kept exactly as read (missing stays missing, text stays text) so validity can be judged
    explicitly. Dates are parsed as in eplmodel.data.build, so match_id agrees with the processed data.
    """
    assert_not_holdout([season])
    if season not in set(allowed_seasons):
        raise SplitAccessError(f"Season {season!r} is not allowed by the market protocol")
    path = raw_season_path(season, raw_dir)
    wanted = [*KEY_COLUMNS, *(c for s in snapshots for c in s.columns)]
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in wanted if c not in header]
    if missing:
        raise KeyError(f"{path.name} lacks columns {missing}")
    raw = pd.read_csv(path, usecols=wanted, dtype={c: object for s in snapshots for c in s.columns})
    raw = raw.dropna(subset=list(KEY_COLUMNS)).reset_index(drop=True)
    date = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")
    out = pd.DataFrame({
        "match_id": date.dt.strftime("%Y-%m-%d") + "_" + raw["HomeTeam"] + "_" + raw["AwayTeam"],
        "Date": date, "Season": season, "HomeTeam": raw["HomeTeam"], "AwayTeam": raw["AwayTeam"],
    })
    for s in snapshots:
        for col, src in zip(snapshot_columns(s), s.columns):
            out[col] = raw[src]
    if not out["match_id"].is_unique:
        raise ValueError(f"{path.name}: duplicate match ids")
    return out


def read_odds(seasons: Sequence[str], snapshots: Sequence[Snapshot], allowed_seasons: Iterable[str],
              raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    allowed = list(allowed_seasons)
    return pd.concat([read_season_odds(s, snapshots, allowed, raw_dir) for s in seasons], ignore_index=True)
