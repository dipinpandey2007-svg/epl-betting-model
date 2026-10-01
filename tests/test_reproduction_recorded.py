"""Reproduction gate: the current code regenerates the recorded predictions of Experiments 10-13.

For every recorded prediction file (checksum-verified against its committed metrics.json), each
fold is predicted again with the library code the recorded protocols use, and every arm is compared
row by row with the record at the registered tolerance 1e-12 (eplmodel.evaluation.reproduction).

Nothing is scored: no loss, metric or conclusion is recomputed, and nothing is written. Regenerating
the 2024-25 predictions reproduces registered protocols for registered specs only (amendment A1), so
it adds no exposure. Needs the git-ignored dataset dev_v2 and recorded predictions; golden marker.
"""

import math

import pytest

from eplmodel.config import load_config
from eplmodel.evaluation import online_poisson as op
from eplmodel.evaluation import time_weighting as tw
from eplmodel.evaluation import update_policy as up
from eplmodel.evaluation.folds import restrict_to_max_season
from eplmodel.evaluation.reproduction import (
    RECORDED,
    RECORDED_ARM_SPECS,
    REGISTERED_TOLERANCE,
    compare_predictions,
    load_recorded_predictions,
)
from eplmodel.evaluation.validation import fold_predictions, prob_columns
from eplmodel.paths import CONFIG_DIR, PROCESSED_DEV_V2
from eplmodel.splits import (
    EXPOSED_VALIDATION_SEASONS,
    SELECTION_VALIDATION_SEASONS,
    require_registered_exposed_validation_spec,
    selection_folds,
)

pytestmark = pytest.mark.golden

VCFG = load_config(CONFIG_DIR / "validation_2425_v1.toml")
DCFG = load_config(CONFIG_DIR / "update_policy_diagnostic_v1.toml")
TCFG = load_config(CONFIG_DIR / "time_weighted_poisson_v1.toml")
OCFG = load_config(CONFIG_DIR / "online_tw_poisson_diagnostic_v1.toml")
H_STAR = float(TCFG["locked"]["half_life_days"])
# The recorded fold targets (legacy definition, frozen by amendment A1).
FOLDS = selection_folds(SELECTION_VALIDATION_SEASONS)
HISTORICAL = [f for f in FOLDS if f[1] not in EXPOSED_VALIDATION_SEASONS]


@pytest.fixture(scope="module")
def matches():
    if not PROCESSED_DEV_V2.exists():
        pytest.skip("data/processed/matches_dev_v2.csv missing")
    if any(not r.path.exists() for r in RECORDED.values()):
        pytest.skip("recorded predictions missing (results/*/predictions.csv are git-ignored)")
    from eplmodel.data import load_dev_matches
    return load_dev_matches()


@pytest.fixture(scope="module")
def recorded(matches):
    return {name: load_recorded_predictions(r) for name, r in RECORDED.items()}


@pytest.fixture(scope="module")
def diagnostic(matches):
    """Experiment 11's predictions per fold (which contain Experiment 10's arms)."""
    return {t: up.diagnostic_fold_predictions(matches, h, t, VCFG, DCFG)[0] for h, t in FOLDS}


@pytest.fixture(scope="module")
def frozen_and_online(matches):
    """Frozen (H*) and online time-weighted predictions per fold, on the common group."""
    f = OCFG["fitting"]
    out = {}
    for history, target in FOLDS:
        data = matches if target in EXPOSED_VALIDATION_SEASONS else restrict_to_max_season(matches, "2122")
        frozen, _ = tw.tw_fold_predictions(data, history, target, {op.FROZEN_ARM: H_STAR}, 10)
        online, fitted = op.online_fold_predictions(data, history, target, H_STAR, 10, f["maxiter"],
                                                    f["retry_maxiter"])
        assert fitted["n_fits"] == OCFG["groups"]["expected_online_fits"][target]
        out[target] = (frozen, online)
    return out


def _fold(recorded_frame, target):
    return recorded_frame[recorded_frame["fold_target"] == target]


def _common(diag, frozen=None, online=None):
    common = diag[diag["in_common"].astype(bool)]
    if frozen is not None:
        common = common.join(frozen[prob_columns(op.FROZEN_ARM)])
    if online is not None:
        common = common.join(online[prob_columns(op.ONLINE_ARM)])
    return common


def test_2425_arms_are_registered_specs():
    for name in ("10", "11", "12-validation", "13-validation"):
        for arm in RECORDED[name].arms:
            require_registered_exposed_validation_spec(RECORDED_ARM_SPECS[arm])


@pytest.mark.parametrize("history,target", FOLDS)
def test_experiment_10_predictions_reproduce(matches, recorded, history, target):
    preds, _ = fold_predictions(matches, history, target, VCFG)
    worst = compare_predictions(preds, _fold(recorded["10"], target), RECORDED["10"].arms)
    assert worst <= REGISTERED_TOLERANCE


@pytest.mark.parametrize("history,target", FOLDS)
def test_experiment_11_predictions_reproduce(recorded, diagnostic, history, target):
    worst = compare_predictions(diagnostic[target], _fold(recorded["11"], target), RECORDED["11"].arms)
    assert worst <= REGISTERED_TOLERANCE


@pytest.mark.parametrize("history,target", HISTORICAL)
def test_experiment_12_development_predictions_reproduce(matches, recorded, history, target):
    grid = {tw.grid_arm(float(h)): float(h) for h in TCFG["candidate"]["half_life_grid_days"]}
    assert math.inf in grid.values()
    preds, _ = tw.tw_fold_predictions(restrict_to_max_season(matches, "2122"), history, target, grid, 10)
    worst = compare_predictions(preds, _fold(recorded["12-development"], target), RECORDED["12-development"].arms)
    assert worst <= REGISTERED_TOLERANCE


def test_experiment_12_validation_predictions_reproduce(recorded, diagnostic, frozen_and_online):
    (target,) = EXPOSED_VALIDATION_SEASONS
    common = _common(diagnostic[target], frozen=frozen_and_online[target][0])
    worst = compare_predictions(common, recorded["12-validation"], RECORDED["12-validation"].arms)
    assert worst <= REGISTERED_TOLERANCE


@pytest.mark.parametrize("history,target", FOLDS)
def test_experiment_13_predictions_reproduce(recorded, diagnostic, frozen_and_online, history, target):
    name = "13-validation" if target in EXPOSED_VALIDATION_SEASONS else "13-historical"
    common = _common(diagnostic[target], *frozen_and_online[target])
    worst = compare_predictions(common, _fold(recorded[name], target), RECORDED[name].arms)
    assert worst <= REGISTERED_TOLERANCE
