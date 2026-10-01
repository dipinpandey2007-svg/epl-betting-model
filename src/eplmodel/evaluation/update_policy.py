"""Update-policy diagnostic (protocol update_policy_diagnostic_v1).

Separates Elo's in-season updating from its calibration layer and from the
static goal models. Each fold adds two season-start Elo arms to the
established predictions of eplmodel.evaluation.validation:

- F1 (elo_f1): ratings frozen at the end of the fold history, fed into the
  SAME fitted calibration layer (logistic regression and home shift) as
  online Elo. Only the ratings differ from online Elo.
- F2 (elo_f2): the same frozen ratings with its own layer, fitted on history
  rows whose feature is the rating difference at the start of that row's
  season (the first history season, where every such feature is zero, is
  excluded). Only the layer differs from F1.

On matches that every arm can score, the per-match losses decompose exactly:

    poisson - elo = (poisson - elo_f2) + (elo_f2 - elo_f1) + (elo_f1 - elo)

- elo_f1 - elo: the updating effect (primary);
- elo_f2 - elo_f1: the calibration-layer effect;
- poisson - elo_f2: history weighting and model family, not separable here.

Every difference is (left loss - right loss), so a positive value means the
right-hand arm has the lower loss. elo_f2 - elo is only a derived subtotal,
never an effect.

Predictions are made from fixtures and history only. Outcomes are joined
afterwards by match_id, exactly as in the validation harness.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from eplmodel.constants import OUTCOMES
from eplmodel.evaluation.calibration import calibration_table
from eplmodel.evaluation.metrics import per_match_brier, per_match_log_loss, score
from eplmodel.evaluation.validation import fold_data, fold_predictions, online_elo_layer, prob_columns
from eplmodel.models.elo import (
    EloOutcomeModel,
    elo_difference,
    final_ratings,
    home_shift_from_results,
    season_start_elo,
)
from eplmodel.splits import SEASON_ORDER

ARMS = ("elo", "elo_f1", "elo_f2", "poisson", "dixon_coles", "frequency_baseline")
ELO_ARMS = ("elo", "elo_f1", "elo_f2", "frequency_baseline")
GROUP_MODELS = {"full": ELO_ARMS, "common": ARMS, "unseen": ELO_ARMS}

COMPONENTS = ("poisson_minus_elo_f2", "elo_f2_minus_elo_f1", "elo_f1_minus_elo")
TOTAL = "poisson_minus_elo"
FULL_GROUP_COMPONENTS = ("elo_f2_minus_elo_f1", "elo_f1_minus_elo")
SUBTOTAL = "elo_f2_minus_elo"
LABELS = {
    "elo_f1_minus_elo": "updating_effect",
    "elo_f2_minus_elo_f1": "calibration_layer_effect",
    "poisson_minus_elo_f2": "history_weighting_or_model_family",
    TOTAL: "total_gap",
    SUBTOTAL: "season_start_vs_online_each_with_own_layer_not_an_effect",
}
METRICS = ("log_loss", "brier")


class DiagnosticCheckError(RuntimeError):
    """A registered invariant of the diagnostic does not hold."""


def split_difference(name: str) -> tuple[str, str]:
    """'poisson_minus_elo_f2' -> ('poisson', 'elo_f2')."""
    left, sep, right = name.partition("_minus_")
    if not sep or not left or not right:
        raise ValueError(f"{name!r} is not of the form '<model>_minus_<model>'")
    return left, right


# --- Predictions ------------------------------------------------------------------------------

def season_start_elo_fold(
    data: pd.DataFrame, history: Sequence[str], target: str, elo_cfg: dict, f2_cfg: dict
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """F1 and F2 predictions for every target match, indexed by match_id. No target outcome is used.

    Also returns online Elo's target probabilities computed from the same
    fitted layer object as F1, so the caller can check that the layer F1 uses
    is the one behind the established online predictions.
    """
    history = sorted(history, key=SEASON_ORDER.index)
    hist, layer, shift = online_elo_layer(data, history, elo_cfg)
    history_rows = data[data["Season"].isin(history)]
    tgt = data[data["Season"] == target]
    rating_args = {"k": elo_cfg["k"], "home_adv": elo_cfg["update_home_advantage"],
                   "initial_rating": elo_cfg["initial_rating"]}

    # Ratings at the start of the target season, from the history rows only.
    start = final_ratings(history_rows, **rating_args)
    start_home = tgt["HomeTeam"].map(lambda t: start.get(t, elo_cfg["initial_rating"])).to_numpy(dtype=float)
    start_away = tgt["AwayTeam"].map(lambda t: start.get(t, elo_cfg["initial_rating"])).to_numpy(dtype=float)

    # F1: frozen ratings into the online layer (same object, same home shift).
    f1 = layer.predict_proba((start_home + shift) - start_away)
    online_from_layer = layer.predict_proba(elo_difference(hist, shift)[(hist["Season"] == target).to_numpy()])

    # F2: its own layer, fitted on season-start features of history seasons 2..n.
    rated = season_start_elo(history_rows, **rating_args)
    fit_seasons = history[1:] if f2_cfg["exclude_first_history_season"] else history
    fit_rows = rated[rated["Season"].isin(fit_seasons)]
    f2_shift = home_shift_from_results(fit_rows["FTR"]) if f2_cfg["apply_home_shift"] else 0.0
    f2_feature = ((fit_rows["EloHomeStart"] + f2_shift) - fit_rows["EloAwayStart"]).to_numpy()
    f2_layer = EloOutcomeModel(C=f2_cfg["logistic_C"], tol=f2_cfg["logistic_tol"]).fit(f2_feature, fit_rows["FTR"])
    f2 = f2_layer.predict_proba((start_home + f2_shift) - start_away)

    index = pd.Index(tgt["match_id"], name="match_id")
    out = pd.concat([pd.DataFrame(f1, columns=prob_columns("elo_f1"), index=index),
                     pd.DataFrame(f2, columns=prob_columns("elo_f2"), index=index)], axis=1)
    out["elo_start_home"] = start_home
    out["elo_start_away"] = start_away
    fitted = {
        "f1_layer": "online_fold_layer",
        "f1_home_shift": shift,
        "f2_home_shift": f2_shift,
        "f2_fit_seasons": list(fit_seasons),
        "f2_n_fit_matches": int(len(fit_rows)),
        "f2_logistic_classes": [str(c) for c in f2_layer.lr_.classes_],
        "f2_logistic_intercept": f2_layer.lr_.intercept_.tolist(),
        "f2_logistic_coef": f2_layer.lr_.coef_.ravel().tolist(),
    }
    return out, fitted, online_from_layer


def prior_games_played(target_rows: pd.DataFrame) -> pd.Series:
    """Mean, over the two teams, of the target-season matches each played on dates strictly before this one.

    Uses fixtures (teams and dates) only, indexed by match_id.
    """
    dates = {team: np.sort(pd.concat([target_rows.loc[target_rows["HomeTeam"] == team, "Date"],
                                      target_rows.loc[target_rows["AwayTeam"] == team, "Date"]]).to_numpy())
             for team in set(target_rows["HomeTeam"]) | set(target_rows["AwayTeam"])}
    before = lambda team, date: int(np.searchsorted(dates[team], np.datetime64(date), side="left"))
    values = [(before(h, d) + before(a, d)) / 2
              for h, a, d in zip(target_rows["HomeTeam"], target_rows["AwayTeam"], target_rows["Date"])]
    return pd.Series(values, index=pd.Index(target_rows["match_id"], name="match_id"), name="prior_games")


def assign_segments(prior_games, lower_edges: Sequence[float], labels: Sequence[str]) -> np.ndarray:
    """Label of the segment whose lower edge is the largest edge <= the value (so 9.5 falls in '0-9')."""
    prior_games = np.asarray(prior_games, dtype=float)
    if len(lower_edges) != len(labels) or list(lower_edges) != sorted(lower_edges):
        raise ValueError("lower_edges must be sorted and match labels one to one")
    if np.any(prior_games < lower_edges[0]):
        raise ValueError("a value lies below the first segment edge")
    idx = np.searchsorted(np.asarray(lower_edges, dtype=float), prior_games, side="right") - 1
    return np.asarray(labels, dtype=object)[idx]


def diagnostic_fold_predictions(
    matches: pd.DataFrame, history: Sequence[str], target: str, vcfg: dict, dcfg: dict
) -> tuple[pd.DataFrame, dict]:
    """All six arms' (H, D, A) predictions for one fold, plus segment labels. No results included.

    Runs two registered invariants before returning:
    - the online probabilities from F1's layer object equal the established online Elo predictions;
    - on matches where neither team has played a target-season match yet, F1 equals online Elo.
    """
    preds, fitted = fold_predictions(matches, history, target, vcfg)
    data = fold_data(matches, history, target)
    season_start, ss_fitted, online_from_layer = season_start_elo_fold(
        data, history, target, vcfg["elo"], dcfg["elo_f2"])
    if not np.array_equal(online_from_layer, preds[prob_columns("elo")].to_numpy()):
        raise DiagnosticCheckError(f"{target}: F1's layer does not reproduce the online Elo predictions")

    tgt = data[data["Season"] == target]
    prior = prior_games_played(tgt)
    preds = preds.join(season_start)
    preds["prior_games"] = prior.reindex(preds.index).to_numpy()
    seg = dcfg["segments"]
    preds["segment"] = assign_segments(preds["prior_games"], seg["lower_edges"], seg["labels"])
    preds["first_match_both"] = preds["prior_games"].to_numpy() == 0

    first = preds["first_match_both"].to_numpy()
    gap = np.abs(preds.loc[first, prob_columns("elo_f1")].to_numpy() - preds.loc[first, prob_columns("elo")].to_numpy())
    if gap.size and gap.max() > seg["first_match_invariant_tolerance"]:
        raise DiagnosticCheckError(f"{target}: F1 differs from online Elo before any target-season match")

    fitted["season_start_elo"] = {**ss_fitted, "n_first_match_both": int(first.sum())}
    fitted["n_unseen"] = int((~preds["in_common"]).sum())
    return preds, fitted


# --- Scoring ----------------------------------------------------------------------------------

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


def per_match_losses(preds: pd.DataFrame, results, models: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    results = np.asarray(results)
    out = {}
    for m in models:
        probs = preds[prob_columns(m)].to_numpy(dtype=float)
        out[m] = {"log_loss": per_match_log_loss(results, probs), "brier": per_match_brier(results, probs)}
    return out


def check_decomposition(losses: Mapping[str, np.ndarray], components: Sequence[str], total: str,
                        tolerance: float) -> float:
    """Raise unless the components add up to the total for every match; return the largest residual."""
    parts = []
    for name in components:
        left, right = split_difference(name)
        parts.append(losses[left] - losses[right])
    left, right = split_difference(total)
    residual = (losses[left] - losses[right]) - np.sum(parts, axis=0)
    worst = float(np.max(np.abs(residual))) if residual.size else 0.0
    if worst > tolerance:
        raise DiagnosticCheckError(f"decomposition of {total} fails: max residual {worst:.3e} > {tolerance:.0e}")
    return worst


def _differences(losses, names, clusters) -> dict:
    out = {}
    for name in names:
        left, right = split_difference(name)
        out[name] = {"label": LABELS.get(name),
                     **{m: paired_difference_clustered(losses[left][m], losses[right][m], clusters) for m in METRICS}}
    return out


def score_block(preds: pd.DataFrame, results, clusters, models: Sequence[str], decomposition: str | None,
                side_pairs: Sequence[Sequence[str]], tolerance: float) -> dict:
    """Scores, decomposition and side comparisons for one set of matches.

    decomposition: 'common' (three components and the total), 'elo' (the two Elo components and the subtotal)
    or None.
    """
    results = np.asarray(results)
    block: dict = {"n_matches": int(len(results))}
    if len(results) == 0:
        return block
    losses = per_match_losses(preds, results, models)
    block["scores"] = {m: score(results, preds[prob_columns(m)].to_numpy(dtype=float)) for m in models}
    if decomposition == "common":
        block["max_identity_residual"] = {
            m: check_decomposition({a: losses[a][m] for a in models}, COMPONENTS, TOTAL, tolerance) for m in METRICS}
        block["decomposition"] = _differences(losses, (*COMPONENTS, TOTAL), clusters)
        block["derived_subtotal"] = _differences(losses, (SUBTOTAL,), clusters)
    elif decomposition == "elo":
        block["max_identity_residual"] = {
            m: check_decomposition({a: losses[a][m] for a in models}, FULL_GROUP_COMPONENTS, SUBTOTAL, tolerance)
            for m in METRICS}
        block["decomposition"] = _differences(losses, FULL_GROUP_COMPONENTS, clusters)
        block["derived_subtotal"] = _differences(losses, (SUBTOTAL,), clusters)
    pairs = [f"{a}_minus_{b}" for a, b in side_pairs if a in models and b in models]
    block["side_comparisons"] = _differences(losses, pairs, clusters)
    return block


def calibration_summary(preds: pd.DataFrame, results, models: Sequence[str], n_bins: int) -> dict:
    """Descriptive: mean prediction vs observed rate per outcome, mean entropy, home-win reliability table."""
    results = np.asarray(results)
    observed = {o: float(np.mean(results == o)) for o in OUTCOMES}
    out = {}
    for m in models:
        probs = preds[prob_columns(m)].to_numpy(dtype=float)
        entropy = -np.sum(np.where(probs > 0, probs * np.log(np.where(probs > 0, probs, 1.0)), 0.0), axis=1)
        out[m] = {
            "mean_predicted": dict(zip(OUTCOMES, probs.mean(axis=0).tolist())),
            "observed": observed,
            "mean_entropy_nats": float(entropy.mean()),
            "home_win_table": calibration_table(probs[:, 0], results == "H", n_bins).reset_index().astype({"bin": str}),
        }
    return out


def score_fold(preds: pd.DataFrame, results: pd.Series, clusters, dcfg: dict) -> dict:
    """All groups, the segments of the full and common groups, and descriptive calibration."""
    if list(results.index) != list(preds.index):
        raise ValueError("results must be aligned with predictions by match_id")
    tol = dcfg["decomposition"]["identity_tolerance"]
    pairs = dcfg["side_comparisons"]["pairs"]
    clusters = np.asarray(clusters)
    common = preds["in_common"].to_numpy(dtype=bool)
    masks = {"full": np.ones(len(preds), dtype=bool), "common": common, "unseen": ~common}
    kind = {"full": "elo", "common": "common", "unseen": "elo"}
    segments = preds["segment"].to_numpy()

    out = {}
    for group, mask in masks.items():
        models = GROUP_MODELS[group]
        res = results.to_numpy()
        block = score_block(preds[mask], res[mask], clusters[mask], models, kind[group], pairs, tol)
        if group in ("full", "common"):
            block["segments"] = {
                label: score_block(preds[mask & (segments == label)], res[mask & (segments == label)],
                                   clusters[mask & (segments == label)], models, kind[group], pairs, tol)
                for label in dcfg["segments"]["labels"]}
            if mask.any():
                block["calibration"] = calibration_summary(preds[mask], res[mask], models,
                                                           dcfg["calibration"]["home_win_bins"])
        out[group] = block
    return out
