"""Fold-based validation of the established specifications (protocol validation_2425_v1).

A *fold* predicts one target season from its history: every development
season strictly before it (eplmodel.splits.selection_folds). Information policy
for every fold:

- The fold's data are restricted to its history and target seasons, so later
  seasons never enter, and nothing is fitted on target-season outcomes.
- Elo ratings run online through the target season. Each prediction uses the
  pre-match ratings, which depend only on matches dated strictly earlier (no
  team plays twice on one date). The home shift and the logistic regression
  are fitted once on the history and held fixed for the whole target season.
- The Poisson goal model is fitted once on the history and is not updated
  during the target season. Dixon-Coles rho is re-estimated on the history
  with the frozen staged grid (never on the target season).
- A target match involving a team absent from the history cannot be scored by
  the goal models. Such matches are in the 'full' group but not in the
  'common' group, and models are only ever compared within one group.

Predicting and scoring are separate steps: fold_predictions() returns
probabilities without results, and score_groups() joins the results by
match_id afterwards. All probability columns are in (H, D, A) order.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from eplmodel.constants import OUTCOMES
from eplmodel.evaluation.alignment import involves_teams, teams_in
from eplmodel.evaluation.baselines import frequency_baseline
from eplmodel.evaluation.calibration import calibration_table
from eplmodel.evaluation.metrics import per_match_brier, per_match_log_loss, score
from eplmodel.models.dixon_coles import rho_bounds, rho_grid_search
from eplmodel.models.elo import EloOutcomeModel, elo_difference, home_shift_from_results, run_elo
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.models.scoreline import captured_mass, outcome_probabilities_from_rates
from eplmodel.splits import assert_history_precedes, assert_not_holdout, assert_valid_selection_target

MODELS = ("elo", "frequency_baseline", "poisson", "dixon_coles")
GROUP_MODELS = {"full": ("elo", "frequency_baseline"), "common": MODELS}


def prob_columns(model: str) -> list[str]:
    return [f"{model}_{o}" for o in OUTCOMES]


def fold_data(matches: pd.DataFrame, history: Sequence[str], target: str) -> pd.DataFrame:
    """The rows of the history and target seasons only, in their original (chronological) order."""
    assert_not_holdout(matches["Season"].unique())
    assert_valid_selection_target(target)
    assert_history_precedes(history, target)
    data = matches[matches["Season"].isin([*history, target])].reset_index(drop=True)
    missing = sorted((set(history) | {target}) - set(data["Season"]))
    if missing:
        raise ValueError(f"Fold seasons {missing} are missing from the match table")
    return data


def unseen_teams(history_rows: pd.DataFrame, target_rows: pd.DataFrame) -> list[str]:
    """Teams in the target fixtures that never appear in the history (needs fixtures only, not results)."""
    return sorted(teams_in(target_rows) - teams_in(history_rows))


def elo_fold(data: pd.DataFrame, history: Sequence[str], target: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """elo_k25_logreg_v1 predictions for the target season: online ratings, calibration layer fitted on history."""
    hist = run_elo(data, k=cfg["k"], home_adv=cfg["update_home_advantage"], initial_rating=cfg["initial_rating"])
    train = hist["Season"].isin(history).to_numpy()
    tgt = (hist["Season"] == target).to_numpy()
    shift = home_shift_from_results(hist.loc[train, "FTR"]) if cfg["apply_home_shift"] else 0.0
    diff = elo_difference(hist, shift)
    model = EloOutcomeModel(C=cfg["logistic_C"], tol=cfg["logistic_tol"]).fit(diff[train], hist.loc[train, "FTR"])
    out = pd.DataFrame(model.predict_proba(diff[tgt]), columns=prob_columns("elo"),
                       index=pd.Index(hist.loc[tgt, "match_id"], name="match_id"))
    out["elo_rating_home"] = hist.loc[tgt, "EloHome"].to_numpy()
    out["elo_rating_away"] = hist.loc[tgt, "EloAway"].to_numpy()
    fitted = {
        "home_shift": shift,
        "logistic_classes": [str(c) for c in model.lr_.classes_],
        "logistic_intercept": model.lr_.intercept_.tolist(),
        "logistic_coef": model.lr_.coef_.ravel().tolist(),
        "n_history_matches": int(train.sum()),
    }
    return out, fitted


def goal_models_fold(
    data: pd.DataFrame, history: Sequence[str], target: str, poisson_cfg: dict, dc_cfg: dict
) -> tuple[pd.DataFrame, dict]:
    """poisson_static_v1 and dixon_coles_staged_v1 predictions for the common target matches.

    Both are fitted once on the history. rho is chosen by the staged grid on the
    history likelihood (invalid values never selected).
    """
    train = data[data["Season"].isin(history)]
    tgt = data[data["Season"] == target]
    unseen = unseen_teams(train, tgt)
    common = tgt[~involves_teams(tgt, unseen)]

    goal_model = PoissonGoalModel().fit(train)
    if goal_model.unseen_teams(teams_in(tgt)) != set(unseen):
        raise AssertionError("eligibility rule and fitted model disagree on unseen teams")
    lam_train, mu_train = goal_model.predict_rates(train["HomeTeam"], train["AwayTeam"])
    rhos = np.arange(dc_cfg["rho_grid_start"], dc_cfg["rho_grid_stop"], dc_cfg["rho_grid_step"])
    grid, rho = rho_grid_search(train["FTHG"], train["FTAG"], lam_train, mu_train, rhos)

    lam, mu = goal_model.predict_rates(common["HomeTeam"], common["AwayTeam"])
    max_goals = poisson_cfg["max_goals"]
    index = pd.Index(common["match_id"], name="match_id")
    out = pd.concat([
        pd.DataFrame(outcome_probabilities_from_rates(lam, mu, 0.0, max_goals), columns=prob_columns("poisson"),
                     index=index),
        pd.DataFrame(outcome_probabilities_from_rates(lam, mu, rho, max_goals), columns=prob_columns("dixon_coles"),
                     index=index),
    ], axis=1)
    out["expected_goals_home"] = lam
    out["expected_goals_away"] = mu
    fitted = {
        "unseen_teams": unseen,
        "poisson_is_home_coef": goal_model.is_home_coef,
        "rho": rho,
        "rho_grid": grid,
        "valid_rho_range": rho_bounds(lam_train, mu_train),
        "min_captured_mass_poisson": float(min((captured_mass(l, m, 0.0, max_goals) for l, m in zip(lam, mu)),
                                               default=float("nan"))),
    }
    return out, fitted


def fold_predictions(matches: pd.DataFrame, history: Sequence[str], target: str,
                     config: dict) -> tuple[pd.DataFrame, dict]:
    """All four models' (H, D, A) predictions for every target match, indexed by match_id. No results included.

    Goal-model columns are NaN for matches outside the common group.
    """
    data = fold_data(matches, history, target)
    tgt = data[data["Season"] == target]
    preds = tgt.set_index("match_id")[["Date", "Season", "HomeTeam", "AwayTeam"]].copy()

    elo, elo_fitted = elo_fold(data, history, target, config["elo"])
    goals, goal_fitted = goal_models_fold(data, history, target, config["poisson"], config["dixon_coles"])
    train = data[data["Season"].isin(history)]
    freq = pd.DataFrame(frequency_baseline(train["FTR"], len(tgt)), columns=prob_columns("frequency_baseline"),
                        index=preds.index)

    preds["in_common"] = preds.index.isin(goals.index)
    preds = preds.join(elo).join(freq).join(goals)
    fitted = {
        "target": target,
        "history": list(history),
        "elo": elo_fitted,
        "goal_models": goal_fitted,
        "frequency_baseline": dict(zip(OUTCOMES, freq.iloc[0].tolist())) if len(freq) else {},
        "n_full": int(len(preds)),
        "n_common": int(preds["in_common"].sum()),
    }
    return preds, fitted


def outcomes_for(matches: pd.DataFrame, match_ids) -> pd.Series:
    """Observed results (H/D/A) for the given match ids, in that order."""
    return matches.set_index("match_id").loc[list(match_ids), "FTR"]


def paired_difference(model_losses: np.ndarray, reference_losses: np.ndarray) -> dict:
    """Mean per-match difference (model minus reference; negative = model better) with a naive standard error.

    The standard error treats matches as independent, ignoring correlation between matches on the same date.
    """
    d = np.asarray(model_losses, dtype=float) - np.asarray(reference_losses, dtype=float)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    return {"n_matches": n, "mean": float(d.mean()), "sd": sd, "naive_se": sd / np.sqrt(n) if n > 1 else float("nan")}


def score_groups(preds: pd.DataFrame, results: pd.Series, reference_models: Sequence[str]) -> dict:
    """Scores per group and model, and paired per-match differences against each reference model in the group."""
    if list(results.index) != list(preds.index):
        raise ValueError("results must be aligned with predictions by match_id")
    out = {}
    for group, models in GROUP_MODELS.items():
        mask = np.ones(len(preds), dtype=bool) if group == "full" else preds["in_common"].to_numpy(dtype=bool)
        res = results.to_numpy()[mask]
        probs = {m: preds.loc[mask, prob_columns(m)].to_numpy(dtype=float) for m in models}
        losses = {m: {"log_loss": per_match_log_loss(res, p), "brier": per_match_brier(res, p)}
                  for m, p in probs.items()}
        paired = {}
        for ref in reference_models:
            if ref not in models:
                continue
            for m in models:
                if m != ref:
                    paired[f"{m}_minus_{ref}"] = {metric: paired_difference(losses[m][metric], losses[ref][metric])
                                                  for metric in ("log_loss", "brier")}
        out[group] = {"scores": {m: score(res, p) for m, p in probs.items()}, "paired_differences": paired}
    return out


def home_win_calibration(preds: pd.DataFrame, results: pd.Series, n_bins: int) -> dict:
    """Descriptive reliability tables for P(home win), per group and model."""
    out = {}
    for group, models in GROUP_MODELS.items():
        mask = np.ones(len(preds), dtype=bool) if group == "full" else preds["in_common"].to_numpy(dtype=bool)
        home_won = (results.to_numpy()[mask] == "H")
        out[group] = {m: calibration_table(preds.loc[mask, f"{m}_H"].to_numpy(dtype=float), home_won, n_bins)
                      .reset_index().astype({"bin": str}) for m in models}
    return out
