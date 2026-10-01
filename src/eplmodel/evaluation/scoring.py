"""Scoring of forecast frames: per-match losses, paired differences, clustered SEs, calibration.

All scoring goes through eplmodel.evaluation.metrics, which takes the (H, D, A) column order
explicitly; sklearn.metrics.log_loss (classes sorted to A, D, H) is never used.

Paired differences are always (left loss - right loss) per match, so a negative mean means the
left arm has the lower loss. Two standard errors are reported:

- naive SE = sd(d, ddof=1) / sqrt(n), treating matches as independent;
- date-clustered SE: SE^2 = G / (G - 1) * sum_g (sum_{i in g} (d_i - mean d))^2 / n^2 with G
  clusters, so matches on the same date (cluster label "<target>_<date>") may be correlated.
  With one match per cluster it equals the naive SE.

EvaluationSet bundles a forecast frame with its outcomes and clusters, aligned by match_id.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eplmodel.constants import OUTCOMES
from eplmodel.evaluation.forecasts import check_forecast_frame, outcomes_for, probabilities, prob_columns
from eplmodel.evaluation.metrics import per_match_brier, per_match_log_loss, score

METRICS = ("log_loss", "brier")


def per_match_losses(preds: pd.DataFrame, results, arms: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    """{arm: {'log_loss': per-match array, 'brier': per-match array}} for the rows of `preds`."""
    results = np.asarray(results)
    out = {}
    for arm in arms:
        p = preds[prob_columns(arm)].to_numpy(dtype=float)
        out[arm] = {"log_loss": per_match_log_loss(results, p), "brier": per_match_brier(results, p)}
    return out


def paired_difference(model_losses: np.ndarray, reference_losses: np.ndarray) -> dict:
    """Mean per-match difference (model minus reference; negative = model better) with a naive standard error.

    The standard error treats matches as independent, ignoring correlation between matches on the same date.
    """
    d = np.asarray(model_losses, dtype=float) - np.asarray(reference_losses, dtype=float)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    return {"n_matches": n, "mean": float(d.mean()), "sd": sd, "naive_se": sd / np.sqrt(n) if n > 1 else float("nan")}


def paired_difference_clustered(left_losses, right_losses, clusters) -> dict:
    """Mean of (left - right) per match, with a naive SE and a cluster-robust SE.

    Clustered SE^2 = G / (G - 1) * sum_g (sum_{i in g} (d_i - mean d))^2 / n^2, with G clusters.
    With one match per cluster it equals the naive SE^2 = var(d, ddof=1) / n.
    """
    d = np.asarray(left_losses, dtype=float) - np.asarray(right_losses, dtype=float)
    clusters = np.asarray(clusters)
    if len(clusters) != len(d):
        raise ValueError("clusters must have one label per match")
    n = len(d)
    out = {"n_matches": n, "mean": float(d.mean()) if n else float("nan"),
           "sd": float("nan"), "naive_se": float("nan"), "clustered_se": float("nan"), "n_clusters": 0}
    if n == 0:
        return out
    sums = pd.Series(d - d.mean()).groupby(clusters).sum().to_numpy()
    g = len(sums)
    out["n_clusters"] = int(g)
    if n > 1:
        out["sd"] = float(d.std(ddof=1))
        out["naive_se"] = out["sd"] / float(np.sqrt(n))
    if g > 1:
        out["clustered_se"] = float(np.sqrt(g / (g - 1) * np.sum(sums ** 2)) / n)
    return out


def date_clusters(preds: pd.DataFrame, target: str) -> np.ndarray:
    """Cluster label per match: '<target>_<YYYY-MM-DD>', so pooled folds never share a cluster."""
    return (target + "_" + pd.to_datetime(preds["Date"]).dt.strftime("%Y-%m-%d")).to_numpy()


def split_difference(name: str) -> tuple[str, str]:
    """'poisson_minus_elo_f2' -> ('poisson', 'elo_f2')."""
    left, sep, right = name.partition("_minus_")
    if not sep or not left or not right:
        raise ValueError(f"{name!r} is not of the form '<model>_minus_<model>'")
    return left, right


def differences(losses: Mapping[str, Mapping[str, np.ndarray]], names: Sequence[str], clusters, mask=None,
                metrics: Sequence[str] = METRICS) -> dict:
    """Paired (left - right) per-match differences with naive and clustered SEs, for each name and metric."""
    clusters = np.asarray(clusters)
    mask = np.ones(len(clusters), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    out = {}
    for name in names:
        left, right = split_difference(name)
        out[name] = {m: paired_difference_clustered(losses[left][m][mask], losses[right][m][mask], clusters[mask])
                     for m in metrics}
    return out


def identity_residual(losses: Mapping[str, np.ndarray], components: Sequence[str], total: str) -> float:
    """Largest per-match |total - sum(components)| for one metric's losses; ~0 when the identity holds."""
    parts = []
    for name in components:
        left, right = split_difference(name)
        parts.append(losses[left] - losses[right])
    left, right = split_difference(total)
    residual = (losses[left] - losses[right]) - np.sum(parts, axis=0)
    return float(np.max(np.abs(residual))) if residual.size else 0.0


def calibration_in_the_large(probs: np.ndarray, results) -> dict:
    """Descriptive: mean predicted vs observed H/D/A frequencies, and mean entropy (nats) as sharpness."""
    p = np.asarray(probs, dtype=float)
    results = np.asarray(results)
    entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=1)
    return {"mean_predicted": dict(zip(OUTCOMES, p.mean(axis=0).tolist())),
            "observed": {o: float(np.mean(results == o)) for o in OUTCOMES},
            "mean_entropy_nats": float(entropy.mean())}


def calibration_gaps(probs: np.ndarray, results, clusters) -> dict:
    """Per outcome, mean(observed - predicted) with naive and clustered SEs (calibration in the large)."""
    p = np.asarray(probs, dtype=float)
    results = np.asarray(results)
    return {o: paired_difference_clustered((results == o).astype(float), p[:, k], clusters)
            for k, o in enumerate(OUTCOMES)}


@dataclass(frozen=True)
class EvaluationSet:
    """A forecast frame with its outcomes and cluster labels, aligned by match_id.

    Build it with evaluation_set(); it is the standard input of every harness evaluation.
    """

    preds: pd.DataFrame
    results: pd.Series
    clusters: np.ndarray
    arms: tuple[str, ...]

    def losses(self) -> dict[str, dict[str, np.ndarray]]:
        return per_match_losses(self.preds, self.results, self.arms)

    def scores(self) -> dict:
        return {a: score(self.results.to_numpy(), probabilities(self.preds, a)) for a in self.arms}

    def differences(self, names: Sequence[str], mask=None) -> dict:
        return differences(self.losses(), names, self.clusters, mask)

    def calibration(self, arm: str) -> dict:
        p = probabilities(self.preds, arm)
        return {**calibration_in_the_large(p, self.results),
                "gaps": calibration_gaps(p, self.results, self.clusters)}


def evaluation_set(preds: pd.DataFrame, matches: pd.DataFrame, target: str, arms: Sequence[str]) -> EvaluationSet:
    """Check the forecast frame, then join outcomes by match_id and attach date clusters."""
    check_forecast_frame(preds, arms)
    results = outcomes_for(matches, preds.index)
    if list(results.index) != list(preds.index):
        raise ValueError("outcomes must be aligned with predictions by match_id")
    return EvaluationSet(preds, results, date_clusters(preds, target), tuple(arms))
