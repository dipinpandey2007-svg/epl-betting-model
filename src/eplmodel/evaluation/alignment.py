"""Helpers for comparing models on exactly the same matches."""

from collections.abc import Iterable

import pandas as pd


def involves_teams(matches: pd.DataFrame, teams: Iterable[str]) -> pd.Series:
    """Boolean mask: the match involves at least one of `teams`."""
    teams = set(teams)
    return matches["HomeTeam"].isin(teams) | matches["AwayTeam"].isin(teams)


def teams_in(matches: pd.DataFrame) -> set[str]:
    return set(matches["HomeTeam"]) | set(matches["AwayTeam"])


def select_by_match_id(predictions: pd.DataFrame, match_ids: Iterable[str]) -> pd.DataFrame:
    """Rows of a match_id-indexed prediction frame, in the order of `match_ids`. Fails on any missing id."""
    match_ids = list(match_ids)
    missing = set(match_ids) - set(predictions.index)
    if missing:
        raise KeyError(f"{len(missing)} match ids have no prediction")
    return predictions.loc[match_ids]
