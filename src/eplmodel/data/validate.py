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
