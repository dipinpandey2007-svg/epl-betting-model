"""How often walk-forward validation folds contain teams unseen in their training seasons.

This sizes the promoted/unseen-team problem inside the training period, where
handling methods can later be compared without touching the development test
benchmark. Teams are categorised with a hand-compiled reference table
(data/reference/team_history.csv) of Premier League history *before* each
team's return, which is information available before kickoff.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from eplmodel.evaluation.alignment import teams_in
from eplmodel.paths import REFERENCE_DIR
from eplmodel.splits import assert_no_dev_test

TEAM_HISTORY_FILE = REFERENCE_DIR / "team_history.csv"
RECENT_YOYO = "recent_yoyo"
LONG_ABSENCE = "long_absence_or_newcomer"
MIXED = "mixed"


def load_team_history(path: Path = TEAM_HISTORY_FILE) -> dict[str, int | None]:
    """Map team -> PL seasons missed before returning (None = no earlier Premier League season)."""
    table = pd.read_csv(path, comment="#", dtype={"return_season": str, "last_pl_season_before_return": str})
    col = table["seasons_out_of_pl_before_return"]
    return {team: (None if pd.isna(v) else int(v)) for team, v in zip(table["team"], col)}


def categorize_team(team: str, history: dict[str, int | None], yoyo_threshold: int = 2) -> str:
    if team not in history:
        raise KeyError(f"{team!r} is missing from the team history table")
    seasons_out = history[team]
    if seasons_out is not None and seasons_out <= yoyo_threshold:
        return RECENT_YOYO
    return LONG_ABSENCE


def newly_appeared_teams(matches: pd.DataFrame, train_seasons: Sequence[str], val_season: str) -> set[str]:
    train = matches[matches["Season"].isin(train_seasons)]
    val = matches[matches["Season"] == val_season]
    return teams_in(val) - teams_in(train)


def fold_summary(
    matches: pd.DataFrame,
    folds: Sequence[tuple[Sequence[str], str]],
    history: dict[str, int | None],
    yoyo_threshold: int = 2,
) -> pd.DataFrame:
    """One row per fold: newly appeared teams and affected validation matches by category."""
    rows = []
    for train_seasons, val_season in folds:
        assert_no_dev_test([*train_seasons, val_season])
        new = newly_appeared_teams(matches, train_seasons, val_season)
        val = matches[matches["Season"] == val_season]
        counts = {RECENT_YOYO: 0, LONG_ABSENCE: 0, MIXED: 0}
        for home, away in zip(val["HomeTeam"], val["AwayTeam"]):
            cats = {categorize_team(t, history, yoyo_threshold) for t in (home, away) if t in new}
            if cats:
                counts[MIXED if len(cats) > 1 else cats.pop()] += 1
        rows.append({
            "train_seasons": f"{train_seasons[0]}-{train_seasons[-1]}",
            "val_season": val_season,
            "low_confidence_fold": len(train_seasons) == 1,
            "newly_appeared_teams": sorted(new),
            "n_recent_yoyo": counts[RECENT_YOYO],
            "n_long_absence_or_newcomer": counts[LONG_ABSENCE],
            "n_mixed": counts[MIXED],
            "n_total_affected": sum(counts.values()),
            "n_val_matches": len(val),
        })
    return pd.DataFrame(rows)
