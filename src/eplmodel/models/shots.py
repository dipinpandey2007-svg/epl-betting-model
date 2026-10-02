"""Shots-information blend on the Experiment 13 online model (protocol shots_information_v1, Experiment 16).

Frozen specification: configs/shots_information_v1.toml, docs/preregistration/shots_information_v1.md.

At each target date d of a fold, on Experiment 13's fitting set F(d) (history + common-group target matches
dated strictly before d) and its weights (w = 2 ** (-age/730)):

- goal model: Experiment 13's Poisson GLM, fitted by the same function (eplmodel.evaluation.online_poisson);
- shot model: the same GLM with shot counts in place of goals (shots on target HST/AST for B1, total shots
  HS/AS for S1); a match with a missing or invalid value in that arm's two columns is left out of that
  arm's shot fit only.

Both models' team effects are centred to sum to zero over C = the target teams present in the fold history,
with each intercept re-expressed (an exact re-parameterisation). The forecast keeps the goal model's level and
home advantage and blends the relative strengths:

    log lambda_home = mu + h + (1-w) att_i + w sa_i + (1-w) def_j + w sd_j
    log lambda_away = mu     + (1-w) att_j + w sa_j + (1-w) def_i + w sd_i

w = 0 is the Experiment 13 online forecast (B0). Matches of teams absent from the fold history are neither
fitted nor scored, as in Experiment 13. Forecasts never contain results.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eplmodel.evaluation import online_poisson as op
from eplmodel.evaluation.alignment import teams_in
from eplmodel.evaluation.folds import common_group
from eplmodel.evaluation.forecasts import prob_columns
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.models.scoreline import outcome_probabilities_from_rates

SHOT_ARMS = {"b1": ("HST", "AST"), "s1": ("HS", "AS")}   # primary: shots on target; sensitivity: total shots


def omega_label(omega: float) -> str:
    """0.25 -> 'w025' (whole percent, three digits)."""
    pct = round(100 * float(omega))
    if not 0 <= pct <= 100 or abs(100 * float(omega) - pct) > 1e-9:
        raise ValueError(f"omega must be a whole percentage in [0, 1], got {omega}")
    return f"w{pct:03d}"


def arm_name(arm: str, omega: float) -> str:
    return f"{arm}_{omega_label(omega)}"


@dataclass(frozen=True)
class Centred:
    """A fitted GLM's level, home term and team effects, centred to sum to zero over a team set."""

    mu: float
    h: float
    attack: pd.Series
    defence: pd.Series


def centred(model: PoissonGoalModel, centring: Sequence[str]) -> Centred:
    """Exact re-parameterisation: att' = att - mean_C(att), def' = def - mean_C(def), mu' = mu + both means."""
    eff = model.team_effects()
    ma = float(eff.loc[list(centring), "attack"].mean())
    md = float(eff.loc[list(centring), "defence"].mean())
    return Centred(float(model.result_.params["Intercept"]) + ma + md, model.is_home_coef,
                   eff["attack"] - ma, eff["defence"] - md)


def blended_rates(goal: Centred, shot: Centred, omega: float, home_teams, away_teams) -> tuple[np.ndarray, np.ndarray]:
    """Expected goals of the blend for each fixture (goal level and home advantage, blended relative strengths)."""
    att = (1.0 - omega) * goal.attack + omega * shot.attack
    dfn = (1.0 - omega) * goal.defence + omega * shot.defence
    home, away = list(home_teams), list(away_teams)
    lam = np.exp(goal.mu + goal.h + att.loc[home].to_numpy() + dfn.loc[away].to_numpy())
    mu = np.exp(goal.mu + att.loc[away].to_numpy() + dfn.loc[home].to_numpy())
    return lam, mu


def valid_shot_rows(rows: pd.DataFrame, columns: tuple[str, str]) -> np.ndarray:
    """Rows usable by a shot fit: both of the arm's columns present (the reader has already set invalid values to NaN)."""
    return rows[list(columns)].notna().all(axis=1).to_numpy()


def shot_frame(rows: pd.DataFrame, columns: tuple[str, str]) -> pd.DataFrame:
    """The rows with the arm's shot counts in the goal columns, so the GLM code is reused unchanged."""
    out = rows.copy()
    out["FTHG"], out["FTAG"] = rows[columns[0]].to_numpy(), rows[columns[1]].to_numpy()
    return out


def online_fold(history_rows: pd.DataFrame, target_rows: pd.DataFrame, omegas: Sequence[float], half_life_days: float,
                max_goals: int, maxiter: int, retry_maxiter: int) -> tuple[pd.DataFrame, dict]:
    """Forecasts of every (arm, omega) for the common-group target matches, refitted once per target date.

    history_rows and target_rows must carry HS, AS, HST, AST (eplmodel.data.shots.attach_shots). Returns a frame
    indexed by match_id with Date, Season, HomeTeam, AwayTeam and <arm>_w<pct>_H/D/A, and per-fit records.
    """
    if not omegas or float(omegas[0]) != 0.0:
        raise ValueError("the omega grid must start with 0 (the baseline)")
    tgt = target_rows
    op.assert_no_team_twice_per_date(tgt)
    unseen, common = common_group(history_rows, tgt)
    history_teams = frozenset(teams_in(history_rows))
    centring = sorted(teams_in(common))                    # target teams present in the fold history
    dates = sorted(pd.unique(common["Date"]))
    cdates = common["Date"].to_numpy()

    names = [arm_name(a, w) for a in SHOT_ARMS for w in omegas]
    probs = {n: np.full((len(common), 3), np.nan) for n in names}
    fits = []
    for d in dates:
        rows = op.fit_set(history_rows, common, d)          # strictly earlier target matches only
        goal_model, weights, goal_info = op.checked_fit(rows, half_life_days, maxiter, retry_maxiter)
        if goal_model.teams_ != history_teams:
            raise op.OnlineCheckError("the goal model's team set differs from the history's")
        goal = centred(goal_model, centring)
        on = cdates == d
        record = {"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "n_fit_matches": int(len(rows)),
                  "goal": goal_info, "goal_home_advantage": goal.h}
        for arm, cols in SHOT_ARMS.items():
            ok = valid_shot_rows(rows, cols)
            shot_rows = shot_frame(rows[ok], cols)
            shot_model, _, shot_info = op.checked_fit(shot_rows, half_life_days, maxiter, retry_maxiter)
            if shot_model.teams_ != history_teams:
                raise op.OnlineCheckError(f"{arm}: the shot model's team set differs from the history's")
            shot = centred(shot_model, centring)
            for w in omegas:
                lam, mu = blended_rates(goal, shot, float(w), common["HomeTeam"][on], common["AwayTeam"][on])
                if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(mu)) and np.all(lam > 0) and np.all(mu > 0)):
                    raise op.FitFailure("non-finite or non-positive goal rate")
                probs[arm_name(arm, w)][on] = outcome_probabilities_from_rates(lam, mu, 0.0, max_goals)
            record[arm] = {**shot_info, "n_rows_excluded_missing_shots": int((~ok).sum()),
                           "shot_home_advantage": shot.h}
        fits.append(record)

    preds = common.set_index("match_id")[["Date", "Season", "HomeTeam", "AwayTeam"]].copy()
    for n in names:
        if np.isnan(probs[n]).any():
            raise op.OnlineCheckError(f"{n}: a common-group match has no forecast")
        preds[prob_columns(n)] = probs[n]
    record = {"unseen_teams": unseen, "n_full": int(len(tgt)), "n_common": int(len(common)),
              "n_unseen_excluded": int(len(tgt) - len(common)), "centring_set": centring, "n_fits": len(fits),
              "all_converged": all(f["goal"]["converged"] and all(f[a]["converged"] for a in SHOT_ARMS) for f in fits),
              "n_retried": int(sum(f["goal"]["retried"] + sum(f[a]["retried"] for a in SHOT_ARMS) for f in fits)),
              "n_rows_excluded_missing_shots": {a: max((f[a]["n_rows_excluded_missing_shots"] for f in fits), default=0)
                                                for a in SHOT_ARMS},
              "fits": fits}
    return preds, record
