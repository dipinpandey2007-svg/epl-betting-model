"""Time-weighted static Poisson (protocol time_weighted_poisson_v1): predictions, selection, evidence rules.

The candidate is poisson_static_v1 fitted on a recency-weighted history,
w = 2 ** (-age / H) with age in days from the latest history date
(eplmodel.models.time_weights). It is fitted once per fold and never updated
during the target season. H = inf is the established static model and uses
the unweighted fit, so it reproduces poisson_static_v1 exactly.

Three separate steps, never mixed:
- fitting: Poisson coefficients from the fold history only, for a given H;
- selecting H (a hyperparameter): from development-target outcomes only, by
  the one-SE rule (select_half_life); the nested estimate re-selects H for
  each outer target from earlier targets only (nested_selection);
- validating: the locked H* scored once on 2024-25 (experiments module).

Predictions never contain results; outcomes are joined by match_id when
scoring. All probability columns are in (H, D, A) order.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eplmodel.constants import OUTCOMES
from eplmodel.evaluation.alignment import involves_teams, teams_in
from eplmodel.evaluation.metrics import per_match_brier, per_match_log_loss
from eplmodel.evaluation.update_policy import paired_difference_clustered
from eplmodel.evaluation.validation import fold_data, prob_columns, unseen_teams
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.models.scoreline import outcome_probabilities_from_rates
from eplmodel.models.time_weights import exponential_decay_weights, kish_effective_sample_size
from eplmodel.splits import SEASON_ORDER, SplitAccessError

STATIC_ARM = "poisson"
METRICS = ("log_loss", "brier")


# --- Names ------------------------------------------------------------------------------------

def half_life_label(h: float) -> str:
    """'inf' or the whole number of days, e.g. 365.0 -> '365'."""
    h = float(h)
    if math.isinf(h):
        return "inf"
    if h != int(h) or h <= 0:
        raise ValueError(f"grid half-lives must be positive whole days, got {h}")
    return str(int(h))


def grid_arm(h: float) -> str:
    """Prediction-column prefix for a grid value; H = inf is the static arm itself."""
    return STATIC_ARM if math.isinf(float(h)) else f"poisson_tw_h{half_life_label(h)}"


# --- Development-stage data guard ------------------------------------------------------------

def restrict_to_development(matches: pd.DataFrame, max_season: str) -> pd.DataFrame:
    """Drop every row from a season after `max_season`, then check that none is left."""
    limit = SEASON_ORDER.index(max_season)
    keep = matches["Season"].map(SEASON_ORDER.index) <= limit
    out = matches[keep.to_numpy()].reset_index(drop=True)
    assert_development_only(out, max_season)
    return out


def assert_development_only(matches: pd.DataFrame, max_season: str) -> None:
    """Raise unless every row belongs to `max_season` or an earlier season."""
    limit = SEASON_ORDER.index(max_season)
    late = sorted(s for s in matches["Season"].unique() if SEASON_ORDER.index(s) > limit)
    if late:
        raise SplitAccessError(f"Development stage must not see seasons after {max_season}: found {late}")


# --- Fitting and predicting -------------------------------------------------------------------

def fit_half_life(history_rows: pd.DataFrame, h: float) -> tuple[PoissonGoalModel, np.ndarray]:
    """Fit the goal model on the history for one half-life. H = inf uses the unweighted (established) fit."""
    weights = exponential_decay_weights(history_rows["Date"], history_rows["Date"].max(), h)
    if math.isinf(float(h)):
        return PoissonGoalModel().fit(history_rows), weights
    return PoissonGoalModel().fit(history_rows, weights=weights), weights


def fit_diagnostics(model: PoissonGoalModel, static: PoissonGoalModel, history_rows: pd.DataFrame,
                    weights: np.ndarray) -> dict:
    """Descriptive: coefficient stability against the static fit, training weight and effective sample size."""
    effects, static_effects = model.team_effects(), static.team_effects()
    net = effects["attack"] - effects["defence"]
    static_net = (static_effects["attack"] - static_effects["defence"]).reindex(net.index)
    last = history_rows["Season"].to_numpy() == max(history_rows["Season"], key=SEASON_ORDER.index)
    per_team = {}
    for team in sorted(teams_in(history_rows)):
        mask = involves_teams(history_rows, [team]).to_numpy()
        per_team[team] = kish_effective_sample_size(weights[mask])
    team_ess = np.array(list(per_team.values()))
    return {
        "glm_converged": bool(model.result_.converged),
        "is_home_coef": model.is_home_coef,
        "attack_sd": float(effects["attack"].std(ddof=0)),
        "defence_sd": float(effects["defence"].std(ddof=0)),
        "spearman_net_strength_vs_static": float(spearmanr(net.to_numpy(), static_net.to_numpy())[0]),
        "n_history_matches": int(len(weights)),
        "sum_of_weights": float(weights.sum()),
        "kish_ess": kish_effective_sample_size(weights),
        "last_season_weight_share": float(weights[last].sum() / weights.sum()),
        "team_kish_ess_min": float(team_ess.min()),
        "team_kish_ess_median": float(np.median(team_ess)),
        "team_kish_ess": per_team,
    }


def returning_teams(history_rows: pd.DataFrame, target_rows: pd.DataFrame) -> list[str]:
    """Target teams present somewhere in the history but not in its last season (fixtures only)."""
    last_season = max(history_rows["Season"], key=SEASON_ORDER.index)
    last = teams_in(history_rows[history_rows["Season"] == last_season])
    return sorted((teams_in(target_rows) & teams_in(history_rows)) - last)


def tw_fold_predictions(matches: pd.DataFrame, history: Sequence[str], target: str,
                        arms: Mapping[str, float], max_goals: int) -> tuple[pd.DataFrame, dict]:
    """(H, D, A) predictions of each arm {name: half-life} for the target's common-group matches.

    Each arm is fitted once on the history rows only; no target-season outcome is used.
    Indexed by match_id, with Date and the fixture; no results.
    """
    data = fold_data(matches, history, target)
    history_rows = data[data["Season"].isin(history)]
    tgt = data[data["Season"] == target]
    unseen = unseen_teams(history_rows, tgt)
    common = tgt[~involves_teams(tgt, unseen)]
    returning = returning_teams(history_rows, tgt)

    preds = common.set_index("match_id")[["Date", "Season", "HomeTeam", "AwayTeam"]].copy()
    preds["involves_returning_team"] = involves_teams(common, returning).to_numpy()
    static, _ = fit_half_life(history_rows, math.inf)
    fitted = {"target": target, "history": list(history), "unseen_teams": unseen, "returning_teams": returning,
              "n_full": int(len(tgt)), "n_common": int(len(common)), "arms": {}}
    for arm, h in arms.items():
        model, weights = fit_half_life(history_rows, h)
        if model.unseen_teams(teams_in(tgt)) != set(unseen):
            raise AssertionError("eligibility rule and fitted model disagree on unseen teams")
        lam, mu = model.predict_rates(common["HomeTeam"], common["AwayTeam"])
        probs = outcome_probabilities_from_rates(lam, mu, 0.0, max_goals)
        preds[prob_columns(arm)] = probs
        fitted["arms"][arm] = {"half_life_days": half_life_label(h),
                               **fit_diagnostics(model, static, history_rows, weights)}
    return preds, fitted


# --- Scoring ----------------------------------------------------------------------------------

def per_match_losses(preds: pd.DataFrame, results, arms: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    results = np.asarray(results)
    out = {}
    for arm in arms:
        p = preds[prob_columns(arm)].to_numpy(dtype=float)
        out[arm] = {"log_loss": per_match_log_loss(results, p), "brier": per_match_brier(results, p)}
    return out


def sharpness_and_calibration(preds: pd.DataFrame, results, arm: str) -> dict:
    """Descriptive: mean predicted vs observed H/D/A frequencies and mean entropy (nats)."""
    p = preds[prob_columns(arm)].to_numpy(dtype=float)
    results = np.asarray(results)
    entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=1)
    return {"mean_predicted": dict(zip(OUTCOMES, p.mean(axis=0).tolist())),
            "observed": {o: float(np.mean(results == o)) for o in OUTCOMES},
            "mean_entropy_nats": float(entropy.mean())}


# --- Selection of the half-life (hyperparameter) --------------------------------------------

def select_half_life(losses: Mapping[str, Mapping[float, np.ndarray]], clusters: Mapping[str, np.ndarray],
                     se_multiple: float) -> dict:
    """One-SE rule over the given targets.

    losses[target][H] is the per-match log loss of half-life H on that target's common group;
    clusters[target] the matching cluster labels (e.g. target + date).

    1. criterion(H) = mean over targets of the target's mean log loss (equal target weights);
    2. H_min = argmin, exact ties broken towards the longer half-life;
    3. H_sel = the longest H whose pooled per-match difference (H - H_min) is
       <= se_multiple x its clustered SE (H_min always qualifies).
    """
    targets = list(losses)
    if not targets:
        raise ValueError("at least one target is needed to select a half-life")
    grid = sorted(losses[targets[0]], key=float)
    if any(sorted(losses[t], key=float) != grid for t in targets):
        raise ValueError("every target must be scored on the same grid")
    criterion = {h: float(np.mean([losses[t][h].mean() for t in targets])) for h in grid}
    best = min(criterion.values())
    h_min = max(h for h in grid if criterion[h] == best)
    pooled_clusters = np.concatenate([np.asarray(clusters[t]) for t in targets])
    rows = []
    for h in grid:
        d = paired_difference_clustered(np.concatenate([losses[t][h] for t in targets]),
                                        np.concatenate([losses[t][h_min] for t in targets]), pooled_clusters)
        eligible = h == h_min or bool(d["mean"] <= se_multiple * d["clustered_se"])
        rows.append({"half_life_days": half_life_label(h), "criterion_mean_log_loss": criterion[h],
                     "diff_vs_h_min": d["mean"], "diff_vs_h_min_clustered_se": d["clustered_se"],
                     "within_one_se": eligible})
    h_sel = max(h for h, row in zip(grid, rows) if row["within_one_se"])
    return {"targets": targets, "h_min": half_life_label(h_min), "h_selected": half_life_label(h_sel),
            "h_selected_days": float(h_sel), "table": rows}


def nested_selection(losses: Mapping[str, Mapping[float, np.ndarray]], clusters: Mapping[str, np.ndarray],
                     outer_targets: Sequence[str], se_multiple: float) -> dict[str, dict]:
    """For each outer target, H selected from the targets strictly before it (chronological order)."""
    order = sorted(losses, key=SEASON_ORDER.index)
    out = {}
    for outer in outer_targets:
        inner = [t for t in order if SEASON_ORDER.index(t) < SEASON_ORDER.index(outer)]
        out[outer] = select_half_life({t: losses[t] for t in inner}, {t: clusters[t] for t in inner}, se_multiple)
    return out


def leave_one_target_out(losses, clusters, se_multiple: float) -> dict[str, str]:
    """Descriptive: the half-life selected with each target left out in turn."""
    return {left_out: select_half_life({t: v for t, v in losses.items() if t != left_out},
                                       {t: v for t, v in clusters.items() if t != left_out},
                                       se_multiple)["h_selected"]
            for left_out in losses}


# --- Pre-registered evidence rules ----------------------------------------------------------

def development_criterion(outer: Mapping[str, Mapping[str, np.ndarray]], clusters: Mapping[str, np.ndarray],
                          floor: float, se_multiple: float, min_negative_folds: int) -> dict:
    """Criterion D on the nested outer folds.

    outer[target] = {'log_loss': d, 'brier': d}: per-match (TW at the nested H - static) differences.
    D holds if the pooled log-loss mean < -floor, |mean| > se_multiple x clustered SE, the fold mean is
    negative in at least min_negative_folds folds, and the pooled Brier mean is negative.
    """
    targets = list(outer)
    cl = np.concatenate([np.asarray(clusters[t]) for t in targets])
    zeros = np.zeros(len(cl))
    pooled = {m: paired_difference_clustered(np.concatenate([outer[t][m] for t in targets]), zeros, cl)
              for m in METRICS}
    negative = sum(bool(np.mean(outer[t]["log_loss"]) < 0) for t in targets)
    ll = pooled["log_loss"]
    checks = {
        "below_floor": bool(ll["mean"] < -floor),
        "beyond_se_multiple": bool(abs(ll["mean"]) > se_multiple * ll["clustered_se"]),
        "enough_negative_folds": negative >= min_negative_folds,
        "brier_same_sign": bool(pooled["brier"]["mean"] < 0),
    }
    return {"pooled": pooled, "n_negative_folds": negative, "n_folds": len(targets), "checks": checks,
            "met": all(checks.values())}


def validation_criterion(d_log_loss, d_brier, clusters, floor: float, se_multiple: float) -> dict:
    """Criterion V on 2024-25: per-match (TW at H* - static) differences."""
    zeros = np.zeros(len(np.asarray(d_log_loss)))
    ll = paired_difference_clustered(d_log_loss, zeros, clusters)
    br = paired_difference_clustered(d_brier, zeros, clusters)
    checks = {
        "below_floor": bool(ll["mean"] < -floor),
        "beyond_se_multiple": bool(abs(ll["mean"]) > se_multiple * ll["clustered_se"]),
        "brier_same_sign": bool(br["mean"] < 0),
    }
    return {"log_loss": ll, "brier": br, "checks": checks, "met": all(checks.values())}


def reading(development_met: bool, validation_met: bool | None, h_star_is_inf: bool) -> str:
    """The pre-registered reading of criteria D and V."""
    if h_star_is_inf:
        return "no_weighting_selected"
    if validation_met is None:
        return "validation_not_yet_scored"
    if development_met and validation_met:
        return "out_of_sample_evidence_weighting_helps_static_poisson"
    if development_met:
        return "historical_benefit_not_replicated_in_2425"
    if validation_met:
        return "2425_specific_observation_cannot_confirm"
    return "no_evidence_weighting_helps"
