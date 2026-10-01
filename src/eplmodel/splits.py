"""Chronological season roles, splits and the guards that enforce them.

Season roles (protocol: docs/HOLDOUT_PROTOCOL.md)
-------------------------------------------------
- TRAIN (2014-15 .. 2021-22): fitting data and walk-forward selection folds.
- EXPOSED_DEV (2022-23, 2023-24): the *exposed development test benchmark*.
  These seasons were scored repeatedly during exploratory work (see
  docs/TEST_SET_ACCESS_LOG.md). They may be used as historical fitting
  information when predicting LATER seasons, but never as a validation or
  model-selection target. Only registered specs may be re-scored on them.
- EXPOSED_VALIDATION (2024-25): scored as validation in four pre-registered
  protocols (Experiments 10-13), then retired from selection by protocol
  amendment A1 / access-log entry P1 (2026-10-01). It may be used as
  historical fitting information when predicting LATER seasons, but never
  as a selection, tuning or confirmation target for new work. Only the specs
  in REGISTERED_EXPOSED_VALIDATION_SPECS may be re-scored on it, to
  reproduce the recorded protocols.
- HOLDOUT (2025-26): the sealed final holdout. Its outcomes must not be used
  for fitting, tuning, feature or specification selection, descriptive
  analysis or any other development decision. It is opened only through
  eplmodel.holdout.
- NEXT_HOLDOUT (2026-27): declared as the next holdout, to be used once
  2025-26 has been opened. Treated exactly like HOLDOUT by every guard here.

Selection targets (amendment A1)
--------------------------------
New selection work uses SELECTION_TARGET_SEASONS (2017-18 .. 2021-22) and the
strict guard assert_selection_target. SELECTION_VALIDATION_SEASONS and
assert_valid_selection_target are LEGACY: they keep the behaviour the
recorded protocols of Experiments 10-13 were registered with (including
2024-25) so that those experiments reproduce unchanged. New code must not use
them (tests/test_splits.py scans experiments/ for this).

The guards prevent accidental misuse; they cannot stop deliberate circumvention.
"""

from collections.abc import Iterable
from enum import StrEnum


class SeasonRole(StrEnum):
    TRAIN = "train"
    EXPOSED_DEV = "exposed_dev"
    EXPOSED_VALIDATION = "exposed_validation"
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
EXPOSED_VALIDATION_SEASONS = ("2425",)
VALIDATION_SEASONS = EXPOSED_VALIDATION_SEASONS  # legacy name used before amendment A1
FINAL_HOLDOUT_SEASONS = ("2526",)
NEXT_HOLDOUT_SEASONS = ("2627",)
RESERVED_HOLDOUT_SEASONS = FINAL_HOLDOUT_SEASONS + NEXT_HOLDOUT_SEASONS

# Seasons that development work may read: everything before the holdout.
DEVELOPMENT_SEASONS = TRAIN_SEASONS + EXPOSED_DEV_SEASONS + VALIDATION_SEASONS

# The dataset behind every recorded result (data/processed/matches.csv, 3,800 matches).
DATASET_V1_SEASONS = TRAIN_SEASONS + EXPOSED_DEV_SEASONS
# The development dataset with the validation season (data/processed/matches_dev_v2.csv, 4,180 matches).
DATASET_DEV_V2_SEASONS = DEVELOPMENT_SEASONS

SEASON_ROLES = {
    **{s: SeasonRole.TRAIN for s in TRAIN_SEASONS},
    **{s: SeasonRole.EXPOSED_DEV for s in EXPOSED_DEV_SEASONS},
    **{s: SeasonRole.EXPOSED_VALIDATION for s in EXPOSED_VALIDATION_SEASONS},
    **{s: SeasonRole.HOLDOUT for s in FINAL_HOLDOUT_SEASONS},
    **{s: SeasonRole.NEXT_HOLDOUT for s in NEXT_HOLDOUT_SEASONS},
}

# Selection targets for NEW work (amendment A1, 2026-10-01): the training-season
# folds validating 2017-18 .. 2021-22, each with at least three earlier seasons.
SELECTION_TARGET_SEASONS = ("1718", "1819", "1920", "2021", "2122")

# LEGACY, FROZEN: the fold targets registered by the recorded protocols
# validation_2425_v1, update_policy_diagnostic_v1, time_weighted_poisson_v1 and
# online_tw_poisson_diagnostic_v1 (Experiments 10-13), which check their configs
# against this tuple. It must never change and must not be used for new work.
SELECTION_VALIDATION_SEASONS = SELECTION_TARGET_SEASONS + EXPOSED_VALIDATION_SEASONS

# Specs scored on 2024-25 by each recorded protocol (new exposures only; the
# established arms that a later protocol re-scored are listed where first scored).
# Re-scoring one of these reproduces a recorded result and adds no new exposure.
# Scoring anything else on 2024-25 needs an entry in docs/TEST_SET_ACCESS_LOG.md
# first, is descriptive only, and can never select, tune or confirm anything.
RECORDED_EXPOSED_VALIDATION_PROTOCOLS = {
    "validation_2425_v1": frozenset({
        "elo_k25_logreg_v1", "frequency_baseline_v1", "poisson_static_v1", "dixon_coles_staged_v1"}),
    "update_policy_diagnostic_v1": frozenset({
        "elo_k25_frozen_ratings_online_layer_v1_diag", "elo_k25_season_start_v1_diag"}),
    "time_weighted_poisson_v1": frozenset({"poisson_time_weighted_v1"}),
    "online_tw_poisson_diagnostic_v1": frozenset({"poisson_tw_online_h730_v1_diag"}),
}
REGISTERED_EXPOSED_VALIDATION_SPECS = frozenset().union(*RECORDED_EXPOSED_VALIDATION_PROTOCOLS.values())

# Descriptive accesses to the exposed validation season logged after amendment A1 (rule 6): access-log entry id ->
# the exact specs that entry may score. Kept separate from the reproduction set above (whose protocols are
# frozen). Opened only through eplmodel.evaluation.folds.build_exposed_validation_fold; descriptive only.
LOGGED_EXPOSED_VALIDATION_ACCESSES = {
    "V5": frozenset({
        "poisson_fc_promoted_prior_h730_v1", "poisson_fc_hierarchical_h730_v1",
        "poisson_fc_hierarchical_break_h730_v1_sens",
        "market_pinnacle_close_shin_v1", "market_pinnacle_close_proportional_v1", "market_pinnacle_close_power_v1",
        "market_pinnacle_preclose_shin_v1", "market_pinnacle_preclose_proportional_v1",
        "market_pinnacle_preclose_power_v1",
    }),
}

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


class ExposedValidationError(SplitAccessError):
    """Raised when the retired 2024-25 season would be a new selection target or scored by an unregistered spec."""


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
    """LEGACY guard of the recorded protocols (Experiments 10-13): refuses exposed-dev and holdout seasons.

    It still accepts 2024-25 so that the recorded protocols reproduce
    unchanged. New selection work must use assert_selection_target.
    """
    assert_not_holdout([season])
    if season in EXPOSED_DEV_SEASONS:
        raise DevTestAccessError(
            f"Season {season!r} is part of the exposed development test benchmark and must not be "
            "used as a validation or model-selection target."
        )
    season_role(season)  # unknown seasons are refused too


def assert_selection_target(season: str) -> None:
    """Raise unless `season` may be a target of NEW selection, tuning or confirmation work (amendment A1).

    Only SELECTION_TARGET_SEASONS qualify: never the exposed 2022-24 seasons,
    the retired 2024-25 season or a holdout season.
    """
    assert_valid_selection_target(season)
    if season in EXPOSED_VALIDATION_SEASONS:
        raise ExposedValidationError(
            f"Season {season!r} was retired as a selection target by amendment A1 (docs/HOLDOUT_PROTOCOL.md); "
            "it may be history for later seasons only."
        )
    if season not in SELECTION_TARGET_SEASONS:
        raise SplitAccessError(f"Season {season!r} is not a registered selection target {SELECTION_TARGET_SEASONS}.")


def require_registered_exposed_validation_spec(spec_id: str) -> None:
    """Allow scoring on 2024-25 only for specs already scored there by a recorded protocol."""
    if spec_id not in REGISTERED_EXPOSED_VALIDATION_SPECS:
        raise ExposedValidationError(
            f"Spec {spec_id!r} has not been scored on 2024-25 by a recorded protocol. Any new 2024-25 scoring "
            "must be recorded in docs/TEST_SET_ACCESS_LOG.md first and is descriptive only."
        )


def require_logged_exposed_validation_access(entry_id: str, spec_ids) -> None:
    """Allow a descriptive access to the exposed validation season only for a logged entry and exactly its specs."""
    if entry_id not in LOGGED_EXPOSED_VALIDATION_ACCESSES:
        raise ExposedValidationError(f"{entry_id!r} is not a logged exposed-validation access.")
    if set(spec_ids) != LOGGED_EXPOSED_VALIDATION_ACCESSES[entry_id]:
        raise ExposedValidationError(f"Access {entry_id!r} authorises exactly "
                                     f"{sorted(LOGGED_EXPOSED_VALIDATION_ACCESSES[entry_id])}.")


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
    validation_seasons: Iterable[str] = SELECTION_TARGET_SEASONS,
) -> list[tuple[tuple[str, ...], str]]:
    """Model-selection folds: each target is predicted from every development season before it.

    The default targets are SELECTION_TARGET_SEASONS (amendment A1). Passing
    2024-25 explicitly is accepted only through the legacy guard so that the
    recorded protocols reproduce; new work must not do so. The exposed seasons
    appear only as history, never as targets; holdout seasons in neither role.
    """
    folds = []
    for target in validation_seasons:
        assert_valid_selection_target(target)
        history = tuple(s for s in DEVELOPMENT_SEASONS if SEASON_ORDER.index(s) < SEASON_ORDER.index(target))
        assert_history_precedes(history, target)
        folds.append((history, target))
    return folds
