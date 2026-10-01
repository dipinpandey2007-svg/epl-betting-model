"""Chronological season splits and guards around the exposed development test benchmark.

Terminology
-----------
- TRAIN_SEASONS: used to fit parameters and, via walk-forward folds, to select
  modelling choices.
- DEV_TEST_SEASONS (2022-23, 2023-24): the *exposed development test benchmark*.
  These seasons were scored repeatedly during exploratory work (see
  docs/TEST_SET_ACCESS_LOG.md), so they are no longer a pristine holdout. They
  must not be used to tune, select or compare any new methodological choice.
  A genuinely untouched final holdout will be created later from additional
  historical data.
"""

from collections.abc import Iterable

SEASON_ORDER = (
    "1415", "1516", "1617", "1718", "1819",
    "1920", "2021", "2122", "2223", "2324",
)
TRAIN_SEASONS = SEASON_ORDER[:8]
DEV_TEST_SEASONS = ("2223", "2324")

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


class DevTestAccessError(RuntimeError):
    """Raised when development-test data would be used for tuning or by an unregistered spec."""


def assert_no_dev_test(seasons: Iterable[str]) -> None:
    """Raise if any season used for fitting or model selection is a development-test season."""
    overlap = sorted(set(seasons) & set(DEV_TEST_SEASONS))
    if overlap:
        raise DevTestAccessError(
            f"Development-test seasons {overlap} must not be used for fitting or model selection."
        )


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
    if min_train_seasons < 1:
        raise ValueError("min_train_seasons must be at least 1")
    return [(seasons[:i], seasons[i]) for i in range(min_train_seasons, len(seasons))]
