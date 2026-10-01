import warnings

import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from eplmodel.evaluation.baselines import frequency_baseline
from eplmodel.evaluation.calibration import calibration_table
from eplmodel.evaluation.metrics import multiclass_brier, multiclass_log_loss, one_hot


def test_log_loss_known_value():
    assert multiclass_log_loss(["H"], [[0.9, 0.05, 0.05]]) == pytest.approx(-np.log(0.9))


def test_sklearn_class_ordering_trap():
    """sklearn sorts labels to (A, D, H); an (H, D, A) array is silently misread even with labels=."""
    probs_hda = np.array([[0.9, 0.05, 0.05]])
    with warnings.catch_warnings():  # recent sklearn versions warn about it; older ones are silent
        warnings.simplefilter("ignore", UserWarning)
        assert sk_log_loss(["H"], probs_hda, labels=["H", "D", "A"]) == pytest.approx(-np.log(0.05))  # wrong
    assert sk_log_loss(["H"], probs_hda[:, [2, 1, 0]], labels=["A", "D", "H"]) == pytest.approx(-np.log(0.9))
    assert multiclass_log_loss(["H"], probs_hda) == pytest.approx(-np.log(0.9))


def test_log_loss_matches_sklearn_after_reordering():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet([2, 1, 1.5], size=200)
    results = rng.choice(["H", "D", "A"], size=200)
    expected = sk_log_loss(results, probs[:, [2, 1, 0]], labels=["A", "D", "H"])
    assert multiclass_log_loss(results, probs) == pytest.approx(expected, abs=1e-12)


def test_brier_definition():
    assert multiclass_brier(["H", "A"], [[1, 0, 0], [0, 0, 1]]) == 0.0
    assert multiclass_brier(["D"], [[1 / 3, 1 / 3, 1 / 3]]) == pytest.approx(2 / 3)
    assert multiclass_brier(["A"], [[1, 0, 0]]) == pytest.approx(2.0)
    # sum over outcomes, not per-class mean: (0.4-1)^2 + 0.3^2 + 0.3^2
    assert multiclass_brier(["H"], [[0.4, 0.3, 0.3]]) == pytest.approx(0.54)


def test_one_hot_uses_outcome_order():
    assert one_hot(["H", "D", "A"]).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.mark.parametrize("probs", [[[0.5, 0.5]], [[0.5, 0.3, 0.3]]])
def test_rejects_malformed_probabilities(probs):
    with pytest.raises(ValueError):
        multiclass_log_loss(["H"], probs)


def test_rejects_unknown_labels():
    with pytest.raises(ValueError):
        multiclass_brier(["X"], [[0.4, 0.3, 0.3]])


def test_frequency_baseline():
    probs = frequency_baseline(["H", "H", "D", "A"], n=3)
    assert probs.shape == (3, 3)
    assert probs[0].tolist() == [0.5, 0.25, 0.25]


def test_calibration_table_perfect_forecast():
    pred = np.repeat([0.1, 0.5, 0.9], 100)
    actual = np.concatenate([np.r_[np.ones(10), np.zeros(90)], np.r_[np.ones(50), np.zeros(50)],
                             np.r_[np.ones(90), np.zeros(10)]])
    table = calibration_table(pred, actual, n_bins=3)
    assert np.allclose(table["gap"], 0.0)
    assert table["n"].sum() == 300
