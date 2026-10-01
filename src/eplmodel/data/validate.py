"""Schema and integrity checks for the processed match table."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"]


class DataValidationError(ValueError):
    pass


def validate_matches(df: pd.DataFrame, matches_per_season: int | None = 380) -> None:
    """Raise DataValidationError listing every failed check."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    problems = []
    nulls = df[REQUIRED_COLUMNS].isna().sum()
    if nulls.any():
        problems.append(f"null values: {nulls[nulls > 0].to_dict()}")
    if not set(df["FTR"]).issubset({"H", "D", "A"}):
        problems.append(f"unexpected FTR values: {sorted(set(df['FTR']) - {'H', 'D', 'A'})}")
    if (df["FTHG"] < 0).any() or (df["FTAG"] < 0).any():
        problems.append("negative goal counts")
    implied = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    n_bad = int((df["FTR"].to_numpy() != implied).sum())
    if n_bad:
        problems.append(f"{n_bad} rows where FTR disagrees with the score")
    if (df["HomeTeam"] == df["AwayTeam"]).any():
        problems.append("a team plays itself")
    n_dup = int(df.duplicated(["Date", "HomeTeam", "AwayTeam"]).sum())
    if n_dup:
        problems.append(f"{n_dup} duplicate (Date, HomeTeam, AwayTeam) rows")
    if not df["Date"].is_monotonic_increasing:
        problems.append("rows are not in chronological order")
    team_day = pd.concat([df[["Date", "HomeTeam"]].set_axis(["Date", "Team"], axis=1),
                          df[["Date", "AwayTeam"]].set_axis(["Date", "Team"], axis=1)])
    if team_day.duplicated().any():
        problems.append("a team plays more than once on the same date (sequential Elo would become order-dependent)")
    if matches_per_season is not None:
        counts = df.groupby("Season").size()
        wrong = counts[counts != matches_per_season]
        if len(wrong):
            problems.append(f"seasons without {matches_per_season} matches: {wrong.to_dict()}")

    if problems:
        raise DataValidationError("; ".join(problems))


def season_window(season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Dates a season's matches may fall on: 1 August of its first year to 31 July of its second (inclusive).

    Consecutive windows do not overlap, so every date belongs to exactly one
    season. The window ends in late July, not June, because 2019-20 finished on
    26 July 2020; every season in the data starts on or after 1 August (the
    earliest is 5 August 2022).
    """
    if len(season) != 4 or not season.isdigit() or int(season[2:]) != (int(season[:2]) + 1) % 100:
        raise DataValidationError(f"Malformed season code {season!r}")
    start_year = 2000 + int(season[:2])
    return pd.Timestamp(start_year, 8, 1), pd.Timestamp(start_year + 1, 7, 31)


def validate_season_dates(df: pd.DataFrame) -> None:
    """Raise DataValidationError if any match lies outside the date window of its Season label."""
    problems = []
    for season, dates in df.groupby("Season")["Date"]:
        start, end = season_window(season)
        n_out = int(((dates < start) | (dates > end)).sum())
        if n_out:
            problems.append(f"{n_out} {season} matches outside {start.date()}..{end.date()}")
    if problems:
        raise DataValidationError("; ".join(problems))
