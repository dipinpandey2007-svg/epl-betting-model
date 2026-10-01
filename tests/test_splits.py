import pytest

from eplmodel.splits import (
    DATASET_V1_SEASONS,
    DEV_TEST_SEASONS,
    DEVELOPMENT_SEASONS,
    EXPOSED_DEV_SEASONS,
    FINAL_HOLDOUT_SEASONS,
    NEXT_HOLDOUT_SEASONS,
    RESERVED_HOLDOUT_SEASONS,
    SEASON_ORDER,
    SEASON_ROLES,
    SELECTION_VALIDATION_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    DevTestAccessError,
    HoldoutAccessError,
    SeasonRole,
    SplitAccessError,
    assert_history_precedes,
    assert_no_dev_test,
    assert_not_holdout,
    assert_valid_selection_target,
    expanding_window_folds,
    require_registered_dev_test_spec,
    season_role,
    selection_folds,
)


def test_train_and_dev_test_partition_seasons():
    assert set(TRAIN_SEASONS).isdisjoint(DEV_TEST_SEASONS)
    assert TRAIN_SEASONS + DEV_TEST_SEASONS == DATASET_V1_SEASONS
    assert DEV_TEST_SEASONS == EXPOSED_DEV_SEASONS == ("2223", "2324")


def test_season_roles_follow_the_holdout_protocol():
    assert VALIDATION_SEASONS == ("2425",)
    assert FINAL_HOLDOUT_SEASONS == ("2526",)
    assert NEXT_HOLDOUT_SEASONS == ("2627",)
    assert list(SEASON_ROLES) == list(SEASON_ORDER)  # every season has exactly one role
    assert DEVELOPMENT_SEASONS + RESERVED_HOLDOUT_SEASONS == SEASON_ORDER
    assert set(DEVELOPMENT_SEASONS).isdisjoint(RESERVED_HOLDOUT_SEASONS)
    assert season_role("2223") is SeasonRole.EXPOSED_DEV
    assert season_role("2526") is SeasonRole.HOLDOUT
    with pytest.raises(ValueError):
        season_role("2728")


def test_season_order_is_chronological():
    starts = [int(s[:2]) for s in SEASON_ORDER]
    assert starts == list(range(14, 14 + len(SEASON_ORDER)))
    assert all(int(s[2:]) == int(s[:2]) + 1 for s in SEASON_ORDER)


@pytest.mark.parametrize("min_train,first_val,n_folds", [(3, "1718", 5), (1, "1516", 7)])
def test_folds_used_by_recorded_experiments(min_train, first_val, n_folds):
    folds = expanding_window_folds(TRAIN_SEASONS, min_train)
    assert len(folds) == n_folds
    assert folds[0][1] == first_val
    assert folds[-1] == (TRAIN_SEASONS[:-1], "2122")


def test_folds_are_strictly_chronological():
    for train, val in expanding_window_folds(TRAIN_SEASONS, 1):
        assert val not in train
        assert all(SEASON_ORDER.index(s) < SEASON_ORDER.index(val) for s in train)
        assert set(train).isdisjoint(DEV_TEST_SEASONS) and val not in DEV_TEST_SEASONS


def test_dev_test_seasons_cannot_be_used_for_tuning():
    assert_no_dev_test(TRAIN_SEASONS)
    with pytest.raises(DevTestAccessError):
        assert_no_dev_test(["2122", "2223"])


def test_unregistered_specs_cannot_score_dev_test():
    require_registered_dev_test_spec("elo_k25_logreg_v1")
    with pytest.raises(DevTestAccessError):
        require_registered_dev_test_spec("some_new_model")


# --- Holdout protocol (docs/HOLDOUT_PROTOCOL.md) ---------------------------------

@pytest.mark.parametrize("season", RESERVED_HOLDOUT_SEASONS)
def test_holdout_seasons_are_refused_by_every_guard(season):
    with pytest.raises(HoldoutAccessError):
        assert_not_holdout(["2122", season])
    with pytest.raises(HoldoutAccessError):
        assert_no_dev_test([season])
    with pytest.raises(HoldoutAccessError):
        assert_valid_selection_target(season)
    with pytest.raises(HoldoutAccessError):
        assert_history_precedes(["2324", season], "2627")
    with pytest.raises(HoldoutAccessError):
        expanding_window_folds(("2324", "2425", season))


def test_development_seasons_pass_the_holdout_guard():
    assert_not_holdout(DEVELOPMENT_SEASONS)


@pytest.mark.parametrize("season", EXPOSED_DEV_SEASONS)
def test_exposed_seasons_are_never_selection_targets(season):
    with pytest.raises(DevTestAccessError):
        assert_valid_selection_target(season)


def test_exposed_seasons_may_be_history_for_later_seasons():
    assert_history_precedes(TRAIN_SEASONS + EXPOSED_DEV_SEASONS, "2425")
    with pytest.raises(SplitAccessError):
        assert_history_precedes(["2122", "2425"], "2425")


def test_selection_folds_match_the_protocol():
    folds = selection_folds()
    assert [val for _, val in folds] == list(SELECTION_VALIDATION_SEASONS)
    assert SELECTION_VALIDATION_SEASONS == ("1718", "1819", "1920", "2021", "2122", "2425")
    # The training-season folds are exactly those of the recorded Elo K selection.
    assert folds[:5] == expanding_window_folds(TRAIN_SEASONS, 3)
    # 2024-25 is predicted from everything before it, including the exposed seasons as history.
    assert folds[5] == (TRAIN_SEASONS + EXPOSED_DEV_SEASONS, "2425")


def test_selection_folds_never_target_exposed_or_holdout_seasons():
    for history, target in selection_folds():
        assert target not in EXPOSED_DEV_SEASONS + RESERVED_HOLDOUT_SEASONS
        assert set(history).isdisjoint(RESERVED_HOLDOUT_SEASONS)
        assert all(SEASON_ORDER.index(s) < SEASON_ORDER.index(target) for s in history)
    with pytest.raises(DevTestAccessError):
        selection_folds(["2324"])
    with pytest.raises(HoldoutAccessError):
        selection_folds(["2526"])
