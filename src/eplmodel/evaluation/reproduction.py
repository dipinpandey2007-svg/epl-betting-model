"""Frozen prediction records of the recorded experiments, and row-by-row reproduction checks.

Each recorded experiment wrote its predictions to results/<name>/predictions.csv (git-ignored,
derived from third-party data) and recorded the file's content SHA-256 in the committed
results/<name>/metrics.json. A recorded file is used only after its checksum is verified.

compare_predictions() is the reproduction rule of the recorded protocols: the same matches in
the same order, the same missing values, and every probability within the registered tolerance
(1e-12).
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eplmodel.data.checksums import content_sha256
from eplmodel.evaluation.forecasts import prob_columns
from eplmodel.paths import PROJECT_ROOT, RESULTS_DIR

REGISTERED_TOLERANCE = 1e-12

# Spec behind each prediction-column prefix of the recorded runs (2024-25 arms must be in
# eplmodel.splits.REGISTERED_EXPOSED_VALIDATION_SPECS).
RECORDED_ARM_SPECS = {
    "elo": "elo_k25_logreg_v1",
    "frequency_baseline": "frequency_baseline_v1",
    "poisson": "poisson_static_v1",
    "dixon_coles": "dixon_coles_staged_v1",
    "elo_f1": "elo_k25_frozen_ratings_online_layer_v1_diag",
    "elo_f2": "elo_k25_season_start_v1_diag",
    "poisson_tw": "poisson_time_weighted_v1",
    "poisson_tw_online": "poisson_tw_online_h730_v1_diag",
}


class ReproductionError(RuntimeError):
    """Recorded predictions are missing, altered, or not reproduced."""


@dataclass(frozen=True)
class RecordedPredictions:
    experiment: str          # RESULTS_LOG experiment, e.g. "10" or "12-development"
    results_name: str        # results/<results_name>/
    arms: tuple[str, ...]    # prediction-column prefixes produced by that run

    @property
    def path(self):
        return RESULTS_DIR / self.results_name / "predictions.csv"

    def recorded_sha256(self) -> str:
        """The checksum recorded in the committed metrics.json of the run."""
        metrics = json.loads((RESULTS_DIR / self.results_name / "metrics.json").read_text(encoding="utf-8"))
        return metrics["results"]["predictions"]["sha256"]


RECORDED = {r.experiment: r for r in (
    RecordedPredictions("10", "validation_2425", ("elo", "frequency_baseline", "poisson", "dixon_coles")),
    RecordedPredictions("11", "update_policy_diagnostic",
                        ("elo", "elo_f1", "elo_f2", "poisson", "dixon_coles", "frequency_baseline")),
    RecordedPredictions("12-development", "time_weighted_poisson_development",
                        ("poisson", "poisson_tw_h183", "poisson_tw_h274", "poisson_tw_h365", "poisson_tw_h548",
                         "poisson_tw_h730", "poisson_tw_h1095", "poisson_tw_h1460")),
    RecordedPredictions("12-validation", "time_weighted_poisson_validation",
                        ("elo", "elo_f1", "elo_f2", "poisson", "poisson_tw", "dixon_coles", "frequency_baseline")),
    RecordedPredictions("13-historical", "online_tw_poisson_historical",
                        ("poisson_tw_online", "poisson_tw", "poisson", "elo", "elo_f1", "elo_f2", "dixon_coles",
                         "frequency_baseline")),
    RecordedPredictions("13-validation", "online_tw_poisson_validation",
                        ("poisson_tw_online", "poisson_tw", "poisson", "elo", "elo_f1", "elo_f2", "dixon_coles",
                         "frequency_baseline")),
)}


def load_recorded_predictions(record: RecordedPredictions, targets: Iterable[str] | None = None) -> pd.DataFrame:
    """The recorded predictions after checksum verification, optionally restricted to fold targets."""
    if not record.path.exists():
        raise ReproductionError(f"{record.path.relative_to(PROJECT_ROOT).as_posix()} is missing")
    if content_sha256(record.path) != record.recorded_sha256():
        raise ReproductionError(f"{record.path.relative_to(PROJECT_ROOT).as_posix()} differs from its recorded checksum")
    recorded = pd.read_csv(record.path, index_col="match_id", dtype={"Season": str, "fold_target": str})
    return recorded if targets is None else recorded[recorded["fold_target"].isin(list(targets))]


def compare_predictions(new: pd.DataFrame, recorded: pd.DataFrame, arms: Sequence[str],
                        tolerance: float = REGISTERED_TOLERANCE, rename: Mapping[str, str] | None = None) -> float:
    """Largest |new - recorded| probability over `arms`, or ReproductionError.

    Both frames are indexed by match_id and must list the same matches in the same order.
    `rename` maps an arm in `new` to its column prefix in `recorded` (default: the same name).
    """
    if list(new.index) != list(recorded.index):
        raise ReproductionError("matches differ from the recorded predictions")
    rename = dict(rename or {})
    a = new[[c for arm in arms for c in prob_columns(arm)]].to_numpy(dtype=float)
    b = recorded[[c for arm in arms for c in prob_columns(rename.get(arm, arm))]].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ReproductionError("missing predictions differ from the recorded ones")
    worst = float(np.nanmax(np.abs(a - b))) if a.size and not np.all(np.isnan(a)) else 0.0
    if worst > tolerance:
        raise ReproductionError(f"predictions differ from the record by {worst:.3e} > {tolerance:.0e}")
    return worst
