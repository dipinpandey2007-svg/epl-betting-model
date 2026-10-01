import pytest

from eplmodel.splits import (
    DEV_TEST_SEASONS,
    SEASON_ORDER,
    TRAIN_SEASONS,
    DevTestAccessError,
    assert_no_dev_test,
    expanding_window_folds,
    require_registered_dev_test_spec,
)


def test_train_and_dev_test_partition_seasons():
    assert set(TRAIN_SEASONS).isdisjoint(DEV_TEST_SEASONS)
    assert TRAIN_SEASONS + DEV_TEST_SEASONS == SEASON_ORDER
    assert DEV_TEST_SEASONS == ("2223", "2324")


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
