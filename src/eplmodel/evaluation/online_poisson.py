"""Online time-weighted Poisson (protocol online_tw_poisson_diagnostic_v1): refits, checks, evidence rules.

The frozen arm is poisson_time_weighted_v1: the Poisson goal model fitted once
on the fold history with weights w = 2 ** (-age / H), H = 730 days
(eplmodel.evaluation.time_weighting). The online arm maximises the SAME
weighted likelihood, refitted before every target date d on

    S(d) = fold history  +  common-group target matches dated strictly before d,

with age measured in days to the latest date in S(d). Weight ratios between
two matches do not depend on that reference date, so the only difference from
the frozen arm is which matches are in the likelihood. On the first target
date S(d) is the history, so the online fit equals the frozen fit.

Information policy: a prediction for a match on date d uses only matches dated
strictly before d. Every match on d therefore has the same information set, so
one refit per date is exactly a refit per match, and no outcome dated d enters
any prediction on d.

Unseen teams (absent from the fold history, from fixtures) are never scored
and their target matches never enter an online fit, so the online model has
the same fixed team parameter space as the frozen model all season.

A fit is valid only if IRLS converged, every parameter is finite, and every
predicted goal rate is finite and positive. An invalid fit is retried once
with the same IRLS and a higher iteration limit; if it is still invalid,
FitFailure is raised. There is no fallback to any other model.

Predictions never contain results; outcomes are joined by match_id when
scoring. All probability columns are in (H, D, A) order.
"""

import math
import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from eplmodel.evaluation.alignment import involves_teams, teams_in
from eplmodel.evaluation.update_policy import (
    check_decomposition,
    paired_difference_clustered,
    per_match_losses,
    split_difference,
)
from eplmodel.evaluation.validation import fold_data, prob_columns, unseen_teams
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.models.scoreline import outcome_probabilities_from_rates
from eplmodel.models.time_weights import exponential_decay_weights, kish_effective_sample_size

FROZEN_ARM = "poisson_tw"
ONLINE_ARM = "poisson_tw_online"
PRIMARY = f"{ONLINE_ARM}_minus_{FROZEN_ARM}"   # negative = online better
ARMS = (ONLINE_ARM, FROZEN_ARM, "poisson", "elo", "elo_f1", "elo_f2", "dixon_coles", "frequency_baseline")
METRICS = ("log_loss", "brier")


class FitFailure(RuntimeError):
    """An online fit is invalid after the registered retry. The stage must abort; there is no fallback."""


class OnlineCheckError(RuntimeError):
    """A registered invariant of the online arm does not hold."""


# --- Fitting sets and weights -----------------------------------------------------------------

def assert_no_team_twice_per_date(rows: pd.DataFrame) -> None:
    """Raise if any team has two matches on one date (fixtures only)."""
    team_day = pd.concat([rows[["Date", "HomeTeam"]].set_axis(["Date", "Team"], axis=1),
                          rows[["Date", "AwayTeam"]].set_axis(["Date", "Team"], axis=1)])
    if team_day.duplicated().any():
        raise OnlineCheckError("a team plays more than once on the same date")


def fit_set(history_rows: pd.DataFrame, eligible_target_rows: pd.DataFrame, date) -> pd.DataFrame:
    """History rows, then the eligible target rows dated strictly before `date` (chronological order)."""
    earlier = eligible_target_rows[eligible_target_rows["Date"] < pd.Timestamp(date)]
    return pd.concat([history_rows, earlier], ignore_index=True)


def online_weights(rows: pd.DataFrame, half_life_days: float) -> np.ndarray:
    """2 ** (-age / H), age in days to the latest date in `rows`. H must be finite (no unweighted shortcut)."""
    if not (math.isfinite(float(half_life_days)) and float(half_life_days) > 0):
        raise ValueError("the online arm needs a finite, positive half-life")
    return exponential_decay_weights(rows["Date"], rows["Date"].max(), half_life_days)


# --- Checked fits ------------------------------------------------------------------------------

def fit_problem(model: PoissonGoalModel) -> str | None:
    """Why a fit is invalid, or None: IRLS must have converged and every parameter must be finite."""
    if not bool(model.result_.converged):
        return "IRLS did not converge"
    if not np.all(np.isfinite(np.asarray(model.result_.params, dtype=float))):
        return "non-finite parameters"
    return None


def checked_fit(rows: pd.DataFrame, half_life_days: float, maxiter: int,
                retry_maxiter: int) -> tuple[PoissonGoalModel, np.ndarray, dict]:
    """Weighted fit on `rows`; one retry of the same IRLS with `retry_maxiter`; FitFailure if still invalid."""
    weights = online_weights(rows, half_life_days)
    model = PoissonGoalModel().fit(rows, weights=weights, maxiter=maxiter)
    retried = False
    if fit_problem(model) is not None:
        retried = True
        model = PoissonGoalModel().fit(rows, weights=weights, maxiter=retry_maxiter)
        problem = fit_problem(model)
        if problem is not None:
            raise FitFailure(f"online fit on {len(rows)} matches up to {rows['Date'].max():%Y-%m-%d} "
                             f"is invalid after the registered retry: {problem}")
    info = {"converged": bool(model.result_.converged),
            "iterations": int(model.result_.fit_history["iteration"]),
            "retried": retried}
    return model, weights, info


def checked_rates(model: PoissonGoalModel, home_teams, away_teams) -> tuple[np.ndarray, np.ndarray]:
    lam, mu = model.predict_rates(home_teams, away_teams)
    if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(mu)) and np.all(lam > 0) and np.all(mu > 0)):
        raise FitFailure("an online fit predicts a non-finite or non-positive goal rate")
    return lam, mu


# --- Online predictions --------------------------------------------------------------------------

def online_fold_predictions(matches: pd.DataFrame, history: Sequence[str], target: str, half_life_days: float,
                            max_goals: int, maxiter: int, retry_maxiter: int) -> tuple[pd.DataFrame, dict]:
    """(H, D, A) online predictions for the target's common-group matches, refitted once per target date.

    Indexed by match_id, with Date, the fixture and 'on_first_target_date'; no results.
    """
    started = time.perf_counter()
    data = fold_data(matches, history, target)
    history_rows = data[data["Season"].isin(history)]
    tgt = data[data["Season"] == target]
    assert_no_team_twice_per_date(tgt)
    unseen = unseen_teams(history_rows, tgt)
    common = tgt[~involves_teams(tgt, unseen)]
    history_teams = frozenset(teams_in(history_rows))
    dates = sorted(pd.unique(common["Date"]))

    preds = common.set_index("match_id")[["Date", "Season", "HomeTeam", "AwayTeam"]].copy()
    probs = np.full((len(common), 3), np.nan)
    common_dates = common["Date"].to_numpy()
    fits = []
    for date in dates:
        rows = fit_set(history_rows, common, date)
        model, weights, info = checked_fit(rows, half_life_days, maxiter, retry_maxiter)
        if model.teams_ != history_teams:
            raise OnlineCheckError(f"{target} {pd.Timestamp(date):%Y-%m-%d}: the online team set differs from "
                                   "the history's (fixed parameter space)")
        on = common_dates == date
        lam, mu = checked_rates(model, common["HomeTeam"][on], common["AwayTeam"][on])
        probs[on] = outcome_probabilities_from_rates(lam, mu, 0.0, max_goals)
        in_target = rows["Season"].to_numpy() == target
        fits.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "n_fit_matches": int(len(rows)),
            "n_target_matches_in_fit": int(in_target.sum()),
            **info,
            "max_abs_coef": float(np.max(np.abs(np.asarray(model.result_.params, dtype=float)))),
            "is_home_coef": model.is_home_coef,
            "target_season_weight_share": float(weights[in_target].sum() / weights.sum()),
            "kish_ess": kish_effective_sample_size(weights),
        })
    if np.isnan(probs).any():
        raise OnlineCheckError(f"{target}: a common-group match has no online prediction")

    preds[prob_columns(ONLINE_ARM)] = probs
    preds["on_first_target_date"] = (common_dates == dates[0]) if dates else np.zeros(0, dtype=bool)
    fitted = {
        "target": target,
        "history": list(history),
        "half_life_days": float(half_life_days),
        "unseen_teams": unseen,
        "n_full": int(len(tgt)),
        "n_common": int(len(common)),
        "n_unseen_target_matches_excluded_from_fit": int(len(tgt) - len(common)),
        "n_fits": len(fits),
        "n_retried": int(sum(f["retried"] for f in fits)),
        "all_converged": all(f["converged"] for f in fits),
        "max_iterations": max((f["iterations"] for f in fits), default=0),
        "runtime_seconds": time.perf_counter() - started,
        "fits": fits,
    }
    return preds, fitted


def first_date_gap(preds: pd.DataFrame) -> float:
    """Largest |online - frozen| probability on the first target date (registered to be ~0)."""
    first = preds["on_first_target_date"].to_numpy(dtype=bool)
    gap = np.abs(preds.loc[first, prob_columns(ONLINE_ARM)].to_numpy(dtype=float)
                 - preds.loc[first, prob_columns(FROZEN_ARM)].to_numpy(dtype=float))
    return float(gap.max()) if gap.size else 0.0


# --- Scoring ------------------------------------------------------------------------------------

def differences(losses: Mapping[str, Mapping[str, np.ndarray]], names: Sequence[str], clusters,
                mask=None) -> dict:
    """Paired (left - right) per-match differences with naive and clustered SEs, for each metric."""
    clusters = np.asarray(clusters)
    mask = np.ones(len(clusters), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    out = {}
    for name in names:
        left, right = split_difference(name)
        out[name] = {m: paired_difference_clustered(losses[left][m][mask], losses[right][m][mask], clusters[mask])
                     for m in METRICS}
    return out


def mean_abs_prob_change(preds: pd.DataFrame, mask=None) -> float:
    """Outcome-free: mean over matches of the summed |online - frozen| H/D/A probability."""
    mask = np.ones(len(preds), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    diff = (preds[prob_columns(ONLINE_ARM)].to_numpy(dtype=float)
            - preds[prob_columns(FROZEN_ARM)].to_numpy(dtype=float))[mask]
    return float(np.abs(diff).sum(axis=1).mean())


def score_block(clusters, losses: Mapping, cfg: dict) -> dict:
    """Scores, the primary difference, the decomposition, the updating parallel and context comparisons."""
    dec, ctx = cfg["decomposition"], cfg["context"]
    block = {
        "n_matches": int(len(np.asarray(clusters))),
        "scores": {a: {m: float(losses[a][m].mean()) for m in METRICS} for a in ARMS},
        "primary": differences(losses, [PRIMARY], clusters)[PRIMARY],
        "max_identity_residual": {m: check_decomposition({a: losses[a][m] for a in ARMS}, dec["components"],
                                                         dec["total"], dec["identity_tolerance"])
                                  for m in METRICS},
        "decomposition": differences(losses, [*dec["components"], dec["total"]], clusters),
        "updating_parallel": differences(losses, dec["updating_parallel"], clusters),
        "context": differences(losses, [f"{a}_minus_{b}" for a, b in ctx["pairs"]], clusters),
    }
    return block


def segment_blocks(preds: pd.DataFrame, clusters, losses: Mapping, cfg: dict) -> dict:
    """Primary difference, Elo's updating component and outcome-free probability change, per segment."""
    segments = preds["segment"].to_numpy()
    out = {}
    for label in cfg["segments"]["labels"]:
        mask = segments == label
        out[label] = {
            "n_matches": int(mask.sum()),
            **differences(losses, [PRIMARY, "elo_f1_minus_elo"], clusters, mask),
            "mean_abs_prob_change_vs_frozen": mean_abs_prob_change(preds, mask),
        }
    return out


def concat_losses(per_fold: Sequence[Mapping[str, Mapping[str, np.ndarray]]]) -> dict:
    """Pool per-match losses of several folds (in the given order)."""
    return {a: {m: np.concatenate([f[a][m] for f in per_fold]) for m in METRICS} for a in per_fold[0]}


def losses_for(preds: pd.DataFrame, results) -> dict:
    return per_match_losses(preds, results, ARMS)


# --- Pre-registered evidence rules ---------------------------------------------------------------

def criterion_u(per_fold: Mapping[str, Mapping[str, np.ndarray]], clusters: Mapping[str, np.ndarray],
                floor: float, se_multiple: float, min_folds: int) -> dict:
    """Criterion U on the historical folds, and its mirror for harm.

    per_fold[target] = {'log_loss': d, 'brier': d}: per-match (online - frozen) differences.
    U: pooled log-loss mean < -floor, |mean| > se_multiple x clustered SE, fold mean < 0 in at least
    min_folds folds, pooled Brier mean < 0. The mirror reverses every sign.
    """
    targets = list(per_fold)
    cl = np.concatenate([np.asarray(clusters[t]) for t in targets])
    zeros = np.zeros(len(cl))
    pooled = {m: paired_difference_clustered(np.concatenate([per_fold[t][m] for t in targets]), zeros, cl)
              for m in METRICS}
    fold_means = {t: {m: float(np.mean(per_fold[t][m])) for m in METRICS} for t in targets}
    n_negative = sum(fold_means[t]["log_loss"] < 0 for t in targets)
    n_positive = sum(fold_means[t]["log_loss"] > 0 for t in targets)
    ll, br = pooled["log_loss"], pooled["brier"]
    beyond = bool(abs(ll["mean"]) > se_multiple * ll["clustered_se"])
    helps = {"below_minus_floor": bool(ll["mean"] < -floor), "beyond_se_multiple": beyond,
             "enough_negative_folds": n_negative >= min_folds, "brier_negative": bool(br["mean"] < 0)}
    hurts = {"above_floor": bool(ll["mean"] > floor), "beyond_se_multiple": beyond,
             "enough_positive_folds": n_positive >= min_folds, "brier_positive": bool(br["mean"] > 0)}
    met, mirror = all(helps.values()), all(hurts.values())
    reading = ("historical_evidence_online_refitting_helps" if met else
               "historical_evidence_online_refitting_hurts" if mirror else
               "no_distinguishable_updating_benefit")
    return {"pooled": pooled, "fold_means": fold_means,
            "equal_fold_weight_mean_log_loss": float(np.mean([fold_means[t]["log_loss"] for t in targets])),
            "n_negative_folds": int(n_negative), "n_positive_folds": int(n_positive), "n_folds": len(targets),
            "checks_helps": helps, "checks_hurts": hurts, "met": met, "mirror_met": mirror, "reading": reading}


def accumulation_pattern(segment_log_loss: Mapping[str, Mapping], labels: Sequence[str], se_multiple: float) -> dict:
    """Descriptive check of the pre-registered accumulation hypothesis.

    segment_log_loss[label] is the paired (online - frozen) log-loss summary of a segment.
    Consistent if |Delta(first)| <= se_multiple x clustered SE and the last two segments are below the first.
    """
    first, third, last = (segment_log_loss[labels[0]], segment_log_loss[labels[2]], segment_log_loss[labels[3]])
    checks = {"first_segment_near_zero": bool(abs(first["mean"]) <= se_multiple * first["clustered_se"]),
              "third_below_first": bool(third["mean"] < first["mean"]),
              "last_below_first": bool(last["mean"] < first["mean"])}
    return {"checks": checks, "consistent_with_accumulation": all(checks.values()), "descriptive_only": True}


def validation_label(summary: Mapping, historical_mean: float, se_multiple: float) -> str:
    """2024-25 label (diagnostic observation only) from its paired (online - frozen) log-loss summary."""
    if not abs(summary["mean"]) > se_multiple * summary["clustered_se"]:
        return "not_distinguishable"
    if np.sign(summary["mean"]) == np.sign(historical_mean):
        return "same_sign_as_historical_distinguishable"
    return "opposite_sign_distinguishable"
