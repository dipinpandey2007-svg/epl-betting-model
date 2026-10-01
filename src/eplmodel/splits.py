"""Chronological season roles, splits and the guards that enforce them.

Season roles (protocol: docs/HOLDOUT_PROTOCOL.md)
-------------------------------------------------
- TRAIN (2014-15 .. 2021-22): fitting data and walk-forward selection folds.
- EXPOSED_DEV (2022-23, 2023-24): the *exposed development test benchmark*.
  These seasons were scored repeatedly during exploratory work (see
  docs/TEST_SET_ACCESS_LOG.md). They may be used as historical fitting
  information when predicting LATER seasons, but never as a validation or
  model-selection target. Only registered specs may be re-scored on them.
- VALIDATION (2024-25): a recent validation season, used alongside the
  training-season folds for model selection.
- HOLDOUT (2025-26): the sealed final holdout. Its outcomes must not be used
  for fitting, tuning, feature or specification selection, descriptive
  analysis or any other development decision. It is opened only through
  eplmodel.holdout.
- NEXT_HOLDOUT (2026-27): declared as the next holdout, to be used once
  2025-26 has been opened. Treated exactly like HOLDOUT by every guard here.

The guards prevent accidental misuse; they cannot stop deliberate circumvention.
"""

from collections.abc import Iterable
from enum import StrEnum


class SeasonRole(StrEnum):
    TRAIN = "train"
    EXPOSED_DEV = "exposed_dev"
    VALIDATION = "validation"
    HOLDOUT = "holdout"
    NEXT_HOLDOUT = "next_holdout"


SEASON_ORDER = (
    "1415", "1516", "1617", "1718", "1819",
    "1920", "2021", "2122", "2223", "2324",
    "2425", "2526", "2627",
)
TRAIN_SEASONS = SEASON_ORDER[:8]
EXPOSED_DEV_SEASONS = ("2223", "2324")
DEV_TEST_SEASONS = EXPOSED_DEV_SEASONS  # name used by the recorded experiments
VALIDATION_SEASONS = ("2425",)
FINAL_HOLDOUT_SEASONS = ("2526",)
NEXT_HOLDOUT_SEASONS = ("2627",)
RESERVED_HOLDOUT_SEASONS = FINAL_HOLDOUT_SEASONS + NEXT_HOLDOUT_SEASONS

# Seasons that development work may read: everything before the holdout.
DEVELOPMENT_SEASONS = TRAIN_SEASONS + EXPOSED_DEV_SEASONS + VALIDATION_SEASONS

# The dataset behind every recorded result (data/processed/matches.csv, 3,800 matches).
DATASET_V1_SEASONS = TRAIN_SEASONS + EXPOSED_DEV_SEASONS

SEASON_ROLES = {
    **{s: SeasonRole.TRAIN for s in TRAIN_SEASONS},
    **{s: SeasonRole.EXPOSED_DEV for s in EXPOSED_DEV_SEASONS},
    **{s: SeasonRole.VALIDATION for s in VALIDATION_SEASONS},
    **{s: SeasonRole.HOLDOUT for s in FINAL_HOLDOUT_SEASONS},
    **{s: SeasonRole.NEXT_HOLDOUT for s in NEXT_HOLDOUT_SEASONS},
}

# Validation targets for model selection: the training folds used so far
# (validating 2017-18 .. 2021-22, each with at least three earlier seasons)
# plus the recent validation season. 2024-25 is one season of 380 matches and
# must not be weighted as more precise than the multi-fold historical evidence.
SELECTION_VALIDATION_SEASONS = ("1718", "1819", "1920", "2021", "2122") + VALIDATION_SEASONS

# Model specifications that have already been scored on the development test
# benchmark. Re-scoring one of these reproduces a recorded result and adds no
# new exposure. Scoring anything else requires a deliberate decision and a new
# entry in docs/TEST_SET_ACCESS_LOG.md *before* this set is extended.
REGISTERED_DEV_TEST_SPECS = frozenset({
    "frequency_baseline_v1",
    "elo_k20_exploratory",
    "elo_k25_logreg_v1",
    "poisson_static_v1",
    "dixon_coles_staged_v1",
})


class SplitAccessError(RuntimeError):
    """A season would be used in a role the protocol forbids."""


class DevTestAccessError(SplitAccessError):
    """Raised when development-test data would be used for tuning or by an unregistered spec."""


class HoldoutAccessError(SplitAccessError):
    """Raised when sealed holdout data would be used outside the holdout access procedure."""


def season_role(season: str) -> SeasonRole:
    if season not in SEASON_ROLES:
        raise ValueError(f"Season {season!r} has no role in eplmodel.splits.SEASON_ORDER")
    return SEASON_ROLES[season]


def assert_not_holdout(seasons: Iterable[str]) -> None:
    """Raise if any season is the sealed final holdout or the declared next holdout.

    Call this wherever data are fitted, tuned, selected, described or loaded for development work.
    """
    overlap = sorted(set(seasons) & set(RESERVED_HOLDOUT_SEASONS))
    if overlap:
        raise HoldoutAccessError(
            f"Holdout seasons {overlap} are sealed and must not be used in development work "
            "(see docs/HOLDOUT_PROTOCOL.md)."
        )


def assert_no_dev_test(seasons: Iterable[str]) -> None:
    """Raise if any season is an exposed development-test season or a holdout season.

    This is the strict guard used by the recorded training-only experiments.
    To use 2022-24 as history for predicting a later season, use
    assert_history_precedes and assert_valid_selection_target instead.
    """
    seasons = list(seasons)
    assert_not_holdout(seasons)
    overlap = sorted(set(seasons) & set(EXPOSED_DEV_SEASONS))
    if overlap:
        raise DevTestAccessError(
            f"Development-test seasons {overlap} must not be used for fitting or model selection."
        )


def assert_valid_selection_target(season: str) -> None:
    """Raise unless `season` may be a validation or model-selection target.

    Exposed development-test seasons and holdout seasons never may.
    """
    assert_not_holdout([season])
    if season in EXPOSED_DEV_SEASONS:
        raise DevTestAccessError(
            f"Season {season!r} is part of the exposed development test benchmark and must not be "
            "used as a validation or model-selection target."
        )
    season_role(season)  # unknown seasons are refused too


def assert_history_precedes(history_seasons: Iterable[str], target_season: str) -> None:
    """Raise unless every history season comes strictly before the target season and none is a holdout."""
    history_seasons = list(history_seasons)
    assert_not_holdout(history_seasons)
    target_pos = SEASON_ORDER.index(target_season)
    late = sorted(s for s in history_seasons if SEASON_ORDER.index(s) >= target_pos)
    if late:
        raise SplitAccessError(f"History seasons {late} do not precede target season {target_season!r}.")


def require_registered_dev_test_spec(spec_id: str) -> None:
    """Allow scoring on the development test benchmark only for registered specifications."""
    if spec_id not in REGISTERED_DEV_TEST_SPECS:
        raise DevTestAccessError(
            f"Spec {spec_id!r} has not been registered for development-test scoring. "
            "Record the intended access in docs/TEST_SET_ACCESS_LOG.md and add it to "
            "REGISTERED_DEV_TEST_SPECS first."
        )


def expanding_window_folds(
    seasons: Iterable[str] = TRAIN_SEASONS, min_train_seasons: int = 1
) -> list[tuple[tuple[str, ...], str]]:
    """Walk-forward folds: each validation season is predicted from all seasons strictly before it.

    >>> expanding_window_folds(("a", "b", "c"), min_train_seasons=1)
    [(('a',), 'b'), (('a', 'b'), 'c')]
    """
    seasons = tuple(seasons)
    assert_not_holdout(seasons)
    if min_train_seasons < 1:
        raise ValueError("min_train_seasons must be at least 1")
    return [(seasons[:i], seasons[i]) for i in range(min_train_seasons, len(seasons))]


def selection_folds(
    validation_seasons: Iterable[str] = SELECTION_VALIDATION_SEASONS,
) -> list[tuple[tuple[str, ...], str]]:
    """Model-selection folds: each target is predicted from every development season before it.

    The exposed 2022-24 seasons appear only as history (in the 2024-25 fold),
    never as targets; holdout seasons appear in neither role.
    """
    folds = []
    for target in validation_seasons:
        assert_valid_selection_target(target)
        history = tuple(s for s in DEVELOPMENT_SEASONS if SEASON_ORDER.index(s) < SEASON_ORDER.index(target))
        assert_history_precedes(history, target)
        folds.append((history, target))
    return folds
