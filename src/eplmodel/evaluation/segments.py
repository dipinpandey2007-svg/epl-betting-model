"""Within-season segments, defined from fixtures only (registered in update_policy_diagnostic_v1).

A match's position in the season is the mean, over its two teams, of the target-season matches
each has played on dates strictly before it. A match belongs to the segment whose lower edge is
the largest edge <= that value, so 9.5 falls in '0-9' with edges [0, 10, 19, 29].
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def prior_games_played(target_rows: pd.DataFrame) -> pd.Series:
    """Mean, over the two teams, of the target-season matches each played on dates strictly before this one.

    Uses fixtures (teams and dates) only, indexed by match_id.
    """
    dates = {team: np.sort(pd.concat([target_rows.loc[target_rows["HomeTeam"] == team, "Date"],
                                      target_rows.loc[target_rows["AwayTeam"] == team, "Date"]]).to_numpy())
             for team in set(target_rows["HomeTeam"]) | set(target_rows["AwayTeam"])}
    before = lambda team, date: int(np.searchsorted(dates[team], np.datetime64(date), side="left"))
    values = [(before(h, d) + before(a, d)) / 2
              for h, a, d in zip(target_rows["HomeTeam"], target_rows["AwayTeam"], target_rows["Date"])]
    return pd.Series(values, index=pd.Index(target_rows["match_id"], name="match_id"), name="prior_games")


def assign_segments(prior_games, lower_edges: Sequence[float], labels: Sequence[str]) -> np.ndarray:
    """Label of the segment whose lower edge is the largest edge <= the value (so 9.5 falls in '0-9')."""
    prior_games = np.asarray(prior_games, dtype=float)
    if len(lower_edges) != len(labels) or list(lower_edges) != sorted(lower_edges):
        raise ValueError("lower_edges must be sorted and match labels one to one")
    if np.any(prior_games < lower_edges[0]):
        raise ValueError("a value lies below the first segment edge")
    idx = np.searchsorted(np.asarray(lower_edges, dtype=float), prior_games, side="right") - 1
    return np.asarray(labels, dtype=object)[idx]
