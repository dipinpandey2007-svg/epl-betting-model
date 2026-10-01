"""Folds, eligibility and information sets for walk-forward evaluation.

A fold predicts one target season from its history (seasons strictly before it). This module
decides WHICH rows a model may see; how a model turns them into probabilities stays in the
model-specific code.

- build_fold(): the fold of a NEW selection target (2017-18 .. 2021-22). It uses the strict guard
  of amendment A1, so 2024-25, the exposed 2022-24 seasons and the holdout are refused. The
  recorded protocols of Experiments 10-13 keep their own legacy path
  (eplmodel.evaluation.validation.fold_data) and are not changed.
- Frozen vs online inputs: a frozen model is fitted once on the history rows; an online model,
  for a match on date d, on the history rows plus target rows dated STRICTLY before d. Every
  match on d therefore has the same information set, and no outcome dated d enters a prediction
  on d (dates carry no kickoff time).
- Eligibility: target matches involving a team absent from the history cannot be scored by a
  model with a fixed team parameter space. They form the 'unseen' group; the rest is 'common'.
  Models are only ever compared within one group.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from eplmodel.evaluation.alignment import involves_teams, teams_in
from eplmodel.splits import (
    SEASON_ORDER,
    SplitAccessError,
    assert_history_precedes,
    assert_not_holdout,
    assert_selection_target,
)


# --- Season restriction ----------------------------------------------------------------------

def assert_max_season(matches: pd.DataFrame, max_season: str) -> None:
    """Raise unless every row belongs to `max_season` or an earlier season."""
    limit = SEASON_ORDER.index(max_season)
    late = sorted(s for s in matches["Season"].unique() if SEASON_ORDER.index(s) > limit)
    if late:
        raise SplitAccessError(f"Development stage must not see seasons after {max_season}: found {late}")


def restrict_to_max_season(matches: pd.DataFrame, max_season: str) -> pd.DataFrame:
    """Drop every row from a season after `max_season`, then check that none is left."""
    limit = SEASON_ORDER.index(max_season)
    keep = matches["Season"].map(SEASON_ORDER.index) <= limit
    out = matches[keep.to_numpy()].reset_index(drop=True)
    assert_max_season(out, max_season)
    return out


# --- Eligibility (fixtures only) ----------------------------------------------------------------

def unseen_teams(history_rows: pd.DataFrame, target_rows: pd.DataFrame) -> list[str]:
    """Teams in the target fixtures that never appear in the history (needs fixtures only, not results)."""
    return sorted(teams_in(target_rows) - teams_in(history_rows))


def common_group(history_rows: pd.DataFrame, target_rows: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """The unseen teams and the target rows in which neither team is unseen (the 'common' group)."""
    unseen = unseen_teams(history_rows, target_rows)
    return unseen, target_rows[~involves_teams(target_rows, unseen)]


# --- Information sets ---------------------------------------------------------------------------

def rows_strictly_before(history_rows: pd.DataFrame, target_rows: pd.DataFrame, date) -> pd.DataFrame:
    """History rows, then the target rows dated strictly before `date` (an online model's fitting set)."""
    earlier = target_rows[target_rows["Date"] < pd.Timestamp(date)]
    return pd.concat([history_rows, earlier], ignore_index=True)


def summarize_fits(fits: Sequence[dict]) -> dict:
    """Accounting of one fold's online refits (each record has 'retried', 'converged', 'iterations')."""
    return {
        "n_fits": len(fits),
        "n_retried": int(sum(f["retried"] for f in fits)),
        "all_converged": all(f["converged"] for f in fits),
        "max_iterations": max((f["iterations"] for f in fits), default=0),
    }


# --- Folds for new selection work ---------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: rows of the history seasons and of the target season only."""

    history: tuple[str, ...]
    target: str
    history_rows: pd.DataFrame
    target_rows: pd.DataFrame

    def fixtures(self) -> pd.DataFrame:
        """Target fixtures without outcomes, indexed by match_id."""
        return self.target_rows.set_index("match_id")[["Date", "Season", "HomeTeam", "AwayTeam"]].copy()

    def common_group(self) -> tuple[list[str], pd.DataFrame]:
        return common_group(self.history_rows, self.target_rows)

    def frozen_inputs(self) -> pd.DataFrame:
        """What a frozen model may be fitted on: the history rows only."""
        return self.history_rows

    def online_inputs(self, date, eligible_target_rows: pd.DataFrame | None = None) -> pd.DataFrame:
        """What an online model may use for a match on `date`: history plus target rows dated strictly before it."""
        target = self.target_rows if eligible_target_rows is None else eligible_target_rows
        if not set(target["match_id"]) <= set(self.target_rows["match_id"]):
            raise SplitAccessError("eligible target rows must come from this fold's target season")
        return rows_strictly_before(self.history_rows, target, date)


def build_fold(matches: pd.DataFrame, history: Sequence[str], target: str) -> Fold:
    """The fold of a NEW selection target; later seasons never enter, holdout and retired targets are refused."""
    assert_not_holdout(matches["Season"].unique())
    assert_selection_target(target)
    history = tuple(history)
    assert_history_precedes(history, target)
    data = matches[matches["Season"].isin([*history, target])].reset_index(drop=True)
    missing = sorted((set(history) | {target}) - set(data["Season"]))
    if missing:
        raise ValueError(f"Fold seasons {missing} are missing from the match table")
    return Fold(history, target, data[data["Season"].isin(history)], data[data["Season"] == target])
