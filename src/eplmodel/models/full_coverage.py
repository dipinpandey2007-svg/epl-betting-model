"""Full-coverage dynamic Poisson goal model (protocol full_coverage_poisson_v1, Experiment 15).

Implements the frozen specification (configs/full_coverage_poisson_v1.toml, PA1 included;
docs/preregistration/full_coverage_poisson_v1.md). Nothing here is tuned.

Model, for a match m with home team i and away team j:

    log E[home goals] = mu + h + att_i + def_j,     log E[away goals] = mu + att_j + def_i

Penalised (MAP) objective at refit date d of target season S, maximised by damped Newton:

    sum_m w_m(d) [x_m eta_home - exp(eta_home) + y_m eta_away - exp(eta_away)]      (log x! terms dropped)
      - 1/2 sum_{penalised target teams t} [(att_t - a_t)^2 / tau_att^2 + (def_t - b_t)^2 / tau_def^2]

- w_m(d) = 2 ** (-age / 730), age in days from the match to the refit date d.
- Parameterisation (PA1): the effects of the constrained teams (the target teams) sum to zero, imposed by
  eliminating the alphabetically last of them; history-only teams are free and unpenalised; mu and h
  have no prior.
- Priors (empirical Bayes, fitted once per target from seasons 2015-16 .. S-1, frozen for the season):
  promoted teams N(a_P, tau^2), continuing teams N(a_C, tau^2) with a_C = -n_P a_P / (n - n_P).
- Arms: M1 (prior on promoted teams only), M2 (prior on every target team; primary), S1 (M2 with the
  pre-absence matches of returning teams moved to a separate history-only identity). The 'anchor' mode
  (no prior, unseen-team matches excluded) is the M0 reproduction check against Experiment 13.

Every fit starts cold and deterministically. A failed existence check raises ExistenceCheckError and a
failed solve FitFailure; there is no fallback.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eplmodel.evaluation.alignment import involves_teams, teams_in
from eplmodel.evaluation.forecasts import prob_columns
from eplmodel.models.full_coverage_spec import M1_PROMOTED, M2_HIERARCHICAL, S1_IDENTITY_BREAK
from eplmodel.models.scoreline import outcome_probabilities_from_rates
from eplmodel.models.time_weights import exponential_decay_weights
from eplmodel.splits import SEASON_ORDER, TRAIN_SEASONS, SplitAccessError, assert_not_holdout

FIRST_DATA_SEASON = TRAIN_SEASONS[0]          # 2014-15: promoted clubs not identifiable, excluded from EB
PRE_ABSENCE_SUFFIX = " [pre-absence]"         # S1 identity of a returning team's matches before its absence
ANCHOR = "anchor"                             # M0 reproduction mode (not an arm)
ARMS = (M1_PROMOTED, M2_HIERARCHICAL, S1_IDENTITY_BREAK)


class FitFailure(RuntimeError):
    """The solver did not reach the registered convergence criterion. The stage must abort; no fallback."""


class ExistenceCheckError(RuntimeError):
    """A registered pre-fit existence check failed. The stage must abort; no fallback."""


@dataclass(frozen=True)
class SolverSettings:
    gradient_tolerance: float = 1e-9
    max_iterations: int = 200
    step_halvings: int = 40


# --- Parameterisation -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Parameterisation:
    """theta = (mu, h, att of free teams, def of free teams).

    Free teams = constrained teams except the alphabetically last one (eliminated), then the
    unconstrained (history-only) teams. The eliminated team's effect is minus the sum of the other
    constrained teams' effects, so effects sum to zero over the constrained teams.
    """

    constrained: tuple[str, ...]
    unconstrained: tuple[str, ...]
    eliminated: str = field(init=False)
    free: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        if len(self.constrained) < 2 or list(self.constrained) != sorted(self.constrained):
            raise ValueError("constrained teams must be sorted and at least two")
        if set(self.constrained) & set(self.unconstrained):
            raise ValueError("a team cannot be both constrained and unconstrained")
        object.__setattr__(self, "eliminated", self.constrained[-1])
        object.__setattr__(self, "free", tuple(self.constrained[:-1]) + tuple(sorted(self.unconstrained)))

    @property
    def n_free(self) -> int:
        return len(self.free)

    @property
    def n_params(self) -> int:
        return 2 + 2 * self.n_free

    def effect_row(self, team: str, kind: str) -> np.ndarray:
        """Coefficients giving the team's attack ('att') or defence ('def') effect as a linear function of theta."""
        offset = 2 if kind == "att" else 2 + self.n_free
        row = np.zeros(self.n_params)
        if team == self.eliminated:
            row[offset:offset + len(self.constrained) - 1] = -1.0
        else:
            row[offset + self.free.index(team)] = 1.0
        return row

    def effects(self, theta: np.ndarray) -> pd.DataFrame:
        teams = [*self.constrained, *sorted(self.unconstrained)]
        return pd.DataFrame({"attack": [self.effect_row(t, "att") @ theta for t in teams],
                             "defence": [self.effect_row(t, "def") @ theta for t in teams]},
                            index=pd.Index(teams, name="team"))


def design(rows: pd.DataFrame, par: Parameterisation) -> np.ndarray:
    """Design matrix with two rows per match, interleaved: 2k = home goals of match k, 2k + 1 = away goals."""
    n = len(rows)
    X = np.zeros((2 * n, par.n_params))
    X[:, 0] = 1.0
    X[0::2, 1] = 1.0
    cache = {(t, kind): par.effect_row(t, kind) for t in set(rows["HomeTeam"]) | set(rows["AwayTeam"])
             for kind in ("att", "def")}
    for k, (home, away) in enumerate(zip(rows["HomeTeam"], rows["AwayTeam"])):
        X[2 * k] += cache[(home, "att")] + cache[(away, "def")]
        X[2 * k + 1] += cache[(away, "att")] + cache[(home, "def")]
    return X


def goals(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack([rows["FTHG"].to_numpy(float), rows["FTAG"].to_numpy(float)]).ravel()


@dataclass(frozen=True)
class Prior:
    """Gaussian prior on the effects of the listed teams: means {team: (a, b)} and standard deviations."""

    means: Mapping[str, tuple[float, float]]
    tau_att: float
    tau_def: float

    def matrices(self, par: Parameterisation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(L, m, precision): penalty 1/2 sum precision * (L theta - m)^2."""
        rows, means, prec = [], [], []
        for team in sorted(self.means):
            if team not in par.constrained:
                raise ValueError(f"prior on {team!r}, which is not a constrained (target) team")
            a, b = self.means[team]
            rows += [par.effect_row(team, "att"), par.effect_row(team, "def")]
            means += [a, b]
            prec += [1.0 / self.tau_att ** 2, 1.0 / self.tau_def ** 2]
        if not rows:
            return np.zeros((0, par.n_params)), np.zeros(0), np.zeros(0)
        return np.array(rows), np.array(means), np.array(prec)


NO_PRIOR = Prior({}, math.inf, math.inf)


# --- Objective and solver --------------------------------------------------------------------------

def objective(theta, X, y, w2, L, m, prec) -> tuple[float, np.ndarray, np.ndarray]:
    """Penalised weighted log-likelihood (without log x! terms), its gradient and Hessian."""
    eta = X @ theta
    lam = np.exp(eta)
    resid = L @ theta - m
    f = float(np.sum(w2 * (y * eta - lam)) - 0.5 * np.sum(prec * resid ** 2))
    g = X.T @ (w2 * (y - lam)) - L.T @ (prec * resid)
    H = -(X.T * (w2 * lam)) @ X - (L.T * prec) @ L
    return f, g, H


def solve(X, y, w2, prior_mats, start: np.ndarray, settings: SolverSettings) -> tuple[np.ndarray, dict]:
    """Damped Newton from `start`; FitFailure unless max |gradient| <= tolerance with finite parameters."""
    L, m, prec = prior_mats
    theta = start.astype(float).copy()
    f, g, H = objective(theta, X, y, w2, L, m, prec)
    for iteration in range(settings.max_iterations + 1):
        if not (np.all(np.isfinite(theta)) and np.isfinite(f) and np.all(np.isfinite(g))):
            raise FitFailure("non-finite parameters or objective")
        if np.max(np.abs(g)) <= settings.gradient_tolerance:
            return theta, {"iterations": iteration, "max_abs_gradient": float(np.max(np.abs(g))), "objective": f}
        if iteration == settings.max_iterations:
            break
        step = np.linalg.solve(-H, g)
        t = 1.0
        # Accept a step that does not decrease the objective beyond its floating-point round-off (near the
        # optimum the change is below the round-off of f, and a strict comparison would stall the solver).
        slack = 64.0 * np.finfo(float).eps * (1.0 + abs(f))
        for _ in range(settings.step_halvings + 1):
            candidate = theta + t * step
            f_new, g_new, H_new = objective(candidate, X, y, w2, L, m, prec)
            if np.isfinite(f_new) and f_new >= f - slack:
                break
            t /= 2.0
        else:
            raise FitFailure(f"no non-decreasing step after {settings.step_halvings} halvings "
                             f"(max |gradient| {np.max(np.abs(g)):.3e})")
        theta, f, g, H = candidate, f_new, g_new, H_new
    raise FitFailure(f"not converged after {settings.max_iterations} iterations (max |gradient| "
                     f"{np.max(np.abs(g)):.3e})")


def cold_start(par: Parameterisation, prior: Prior, y: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """mu = log weighted mean goals of the fit set, h = 0, penalised teams' free effects at their prior means, others 0."""
    theta = np.zeros(par.n_params)
    theta[0] = math.log(float(np.sum(w2 * y)) / float(np.sum(w2)))
    for team, (a, b) in prior.means.items():
        if team != par.eliminated:
            k = par.free.index(team)
            theta[2 + k], theta[2 + par.n_free + k] = a, b
    return theta


def check_existence(rows: pd.DataFrame, w: np.ndarray, par: Parameterisation, prior: Prior) -> None:
    """The registered pre-fit checks: positive total goals; every unpenalised team scored and conceded;
    every constrained team without a fitted match carries a prior."""
    hg, ag = rows["FTHG"].to_numpy(float), rows["FTAG"].to_numpy(float)
    if not np.sum(w * (hg + ag)) > 0:
        raise ExistenceCheckError("the fitting set has no weighted goals")
    home, away = rows["HomeTeam"].to_numpy(), rows["AwayTeam"].to_numpy()
    played = set(home) | set(away)
    for team in [*par.constrained, *par.unconstrained]:
        if team in prior.means:
            continue
        if team not in played:
            raise ExistenceCheckError(f"{team!r} has no fitted match and no prior")
        scored = np.sum(w[home == team] * hg[home == team]) + np.sum(w[away == team] * ag[away == team])
        conceded = np.sum(w[home == team] * ag[home == team]) + np.sum(w[away == team] * hg[away == team])
        if not (scored > 0 and conceded > 0):
            raise ExistenceCheckError(f"unpenalised team {team!r} has no weighted goals scored or conceded")


def fit(rows: pd.DataFrame, w: np.ndarray, par: Parameterisation, prior: Prior,
        settings: SolverSettings = SolverSettings()) -> tuple[np.ndarray, dict]:
    """Existence checks, then the penalised fit from a cold start."""
    check_existence(rows, w, par, prior)
    X, y, w2 = design(rows, par), goals(rows), np.repeat(np.asarray(w, dtype=float), 2)
    return solve(X, y, w2, prior.matrices(par), cold_start(par, prior, y, w2), settings)


def rates(theta: np.ndarray, rows: pd.DataFrame, par: Parameterisation) -> tuple[np.ndarray, np.ndarray]:
    eta = design(rows, par) @ theta
    return np.exp(eta[0::2]), np.exp(eta[1::2])


# --- Empirical-Bayes season-start prior ------------------------------------------------------------

@dataclass(frozen=True)
class EmpiricalBayes:
    seasons: tuple[str, ...]
    a_promoted: float
    b_promoted: float
    tau_att: float
    tau_def: float
    n_promoted: int
    n_continuing: int
    pooled_var_att: float
    pooled_var_def: float
    mean_se2_att: float
    mean_se2_def: float
    floor_binding_att: bool
    floor_binding_def: bool
    corr_att_def: float
    team_seasons: pd.DataFrame


def previous_season(season: str) -> str:
    i = SEASON_ORDER.index(season)
    if i == 0:
        raise ValueError(f"{season!r} has no previous season")
    return SEASON_ORDER[i - 1]


def single_season_effects(season_rows: pd.DataFrame, settings: SolverSettings = SolverSettings()) -> pd.DataFrame:
    """Unweighted single-season MLE with effects summing to zero over the season's teams, and their SEs.

    SEs from the inverse Fisher information of the constrained fit, mapped to every team (incl. the
    eliminated one) by the delta method.
    """
    par = Parameterisation(tuple(sorted(teams_in(season_rows))), ())
    theta, _ = fit(season_rows, np.ones(len(season_rows)), par, NO_PRIOR, settings)
    X, y = design(season_rows, par), goals(season_rows)
    _, _, H = objective(theta, X, y, np.ones(len(y)), *NO_PRIOR.matrices(par))
    cov = np.linalg.inv(-H)
    out = []
    for team in par.constrained:
        ra, rd = par.effect_row(team, "att"), par.effect_row(team, "def")
        out.append({"team": team, "attack": float(ra @ theta), "defence": float(rd @ theta),
                    "se_attack": float(np.sqrt(ra @ cov @ ra)), "se_defence": float(np.sqrt(rd @ cov @ rd))})
    return pd.DataFrame(out)


def empirical_bayes(history_rows: pd.DataFrame, target: str, variance_floor: float,
                    settings: SolverSettings = SolverSettings()) -> EmpiricalBayes:
    """Prior hyperparameters for target season `target` from seasons strictly between 2014-15 and the target."""
    seasons = sorted(set(history_rows["Season"]), key=SEASON_ORDER.index)
    late = [s for s in seasons if SEASON_ORDER.index(s) >= SEASON_ORDER.index(target)]
    if late:
        raise SplitAccessError(f"empirical Bayes for {target} received rows of seasons {late}")
    assert_not_holdout(seasons)
    eb_seasons = [s for s in seasons if SEASON_ORDER.index(s) > SEASON_ORDER.index(FIRST_DATA_SEASON)]
    frames = []
    for s in eb_seasons:
        prev = previous_season(s)
        if prev not in seasons:
            raise ValueError(f"EB season {s} needs {prev} in the history to identify promoted teams")
        rows = history_rows[history_rows["Season"] == s]
        eff = single_season_effects(rows, settings)
        eff["season"] = s
        eff["promoted"] = ~eff["team"].isin(teams_in(history_rows[history_rows["Season"] == prev]))
        frames.append(eff)
    if not frames:
        raise ValueError(f"no empirical-Bayes season before {target}")
    ts = pd.concat(frames, ignore_index=True)
    groups = ts.groupby("promoted")
    n, n_groups = len(ts), groups.ngroups

    def pooled(col):
        ss = sum(float(((g[col] - g[col].mean()) ** 2).sum()) for _, g in groups)
        return ss / (n - n_groups)

    var_att, var_def = pooled("attack"), pooled("defence")
    se2_att, se2_def = float((ts["se_attack"] ** 2).mean()), float((ts["se_defence"] ** 2).mean())
    floor2 = variance_floor ** 2
    tau2_att, tau2_def = max(var_att - se2_att, floor2), max(var_def - se2_def, floor2)
    promoted = ts[ts["promoted"]]
    return EmpiricalBayes(
        seasons=tuple(eb_seasons), a_promoted=float(promoted["attack"].mean()),
        b_promoted=float(promoted["defence"].mean()), tau_att=math.sqrt(tau2_att), tau_def=math.sqrt(tau2_def),
        n_promoted=int(ts["promoted"].sum()), n_continuing=int((~ts["promoted"]).sum()),
        pooled_var_att=var_att, pooled_var_def=var_def, mean_se2_att=se2_att, mean_se2_def=se2_def,
        floor_binding_att=var_att - se2_att < floor2, floor_binding_def=var_def - se2_def < floor2,
        corr_att_def=float(np.corrcoef(ts["attack"], ts["defence"])[0, 1]), team_seasons=ts)


def continuing_means(eb: EmpiricalBayes, n_target: int, n_promoted_target: int) -> tuple[float, float]:
    """a_C = -n_P a_P / (n - n_P): prior means then sum to zero over the target teams."""
    k = n_promoted_target / (n_target - n_promoted_target)
    return -k * eb.a_promoted, -k * eb.b_promoted


# --- Target-season structure (fixtures only) --------------------------------------------------------

@dataclass(frozen=True)
class TargetStructure:
    target: str
    teams: tuple[str, ...]
    promoted: tuple[str, ...]
    continuing: tuple[str, ...]
    returning: tuple[str, ...]
    unseen: tuple[str, ...]


def target_structure(history_rows: pd.DataFrame, target_rows: pd.DataFrame) -> TargetStructure:
    """Groups from fixtures: promoted = not in the previous season; returning = promoted but in the history."""
    seasons = set(target_rows["Season"])
    if len(seasons) != 1:
        raise ValueError("target rows must hold one season")
    (target,) = seasons
    prev = previous_season(target)
    prev_rows = history_rows[history_rows["Season"] == prev]
    if prev_rows.empty:
        raise ValueError(f"the history must contain {prev}")
    teams, prev_teams, seen = teams_in(target_rows), teams_in(prev_rows), teams_in(history_rows)
    promoted = teams - prev_teams
    return TargetStructure(target, tuple(sorted(teams)), tuple(sorted(promoted)), tuple(sorted(teams & prev_teams)),
                           tuple(sorted(promoted & seen)), tuple(sorted(teams - seen)))


def arm_prior(arm: str, structure: TargetStructure, eb: EmpiricalBayes) -> Prior:
    a_c, b_c = continuing_means(eb, len(structure.teams), len(structure.promoted))
    promoted = {t: (eb.a_promoted, eb.b_promoted) for t in structure.promoted}
    if arm == M1_PROMOTED:
        return Prior(promoted, eb.tau_att, eb.tau_def)
    if arm in (M2_HIERARCHICAL, S1_IDENTITY_BREAK):
        return Prior({**promoted, **{t: (a_c, b_c) for t in structure.continuing}}, eb.tau_att, eb.tau_def)
    raise ValueError(f"unknown arm {arm!r}")


def break_identities(history_rows: pd.DataFrame, returning: Sequence[str]) -> pd.DataFrame:
    """S1: a returning team's history matches (all before its absence) move to a separate history-only identity."""
    rows = history_rows.copy()
    rename = {t: t + PRE_ABSENCE_SUFFIX for t in returning}
    rows["HomeTeam"] = rows["HomeTeam"].replace(rename)
    rows["AwayTeam"] = rows["AwayTeam"].replace(rename)
    return rows


# --- Online predictions for one fold ---------------------------------------------------------------

def online_fold(history_rows: pd.DataFrame, target_rows: pd.DataFrame, arm: str, eb: EmpiricalBayes | None,
                half_life_days: float, max_goals: int,
                settings: SolverSettings = SolverSettings()) -> tuple[pd.DataFrame, dict]:
    """(H, D, A) forecasts for the target season, refitted once per target date on strictly earlier matches.

    arm is M1, M2, S1, or ANCHOR (no prior; matches involving a team unseen in the history are neither
    fitted nor predicted, as in Experiment 13). Returns a forecast frame indexed by match_id (no results)
    and per-fit records.
    """
    assert_not_holdout(set(history_rows["Season"]) | set(target_rows["Season"]))
    target_seasons = set(target_rows["Season"])
    if len(target_seasons) != 1:
        raise ValueError("target rows must hold one season")
    (target,) = target_seasons
    if any(SEASON_ORDER.index(s) >= SEASON_ORDER.index(target) for s in set(history_rows["Season"])):
        raise SplitAccessError("history rows must precede the target season")
    structure = target_structure(history_rows, target_rows)
    tgt = target_rows.sort_values("Date", kind="stable")
    hist = history_rows
    if arm == ANCHOR:
        tgt = tgt[~involves_teams(tgt, structure.unseen).to_numpy()]
        constrained = tuple(sorted(set(structure.teams) - set(structure.unseen)))
        prior = NO_PRIOR
    else:
        if eb is None:
            raise ValueError("arms M1, M2 and S1 need the empirical-Bayes prior")
        if arm == S1_IDENTITY_BREAK:
            hist = break_identities(history_rows, structure.returning)
        constrained = structure.teams
        prior = arm_prior(arm, structure, eb)
    par = Parameterisation(constrained, tuple(sorted(teams_in(hist) - set(constrained))))
    all_rows = pd.concat([hist, tgt], ignore_index=True)
    is_target = np.r_[np.zeros(len(hist), dtype=bool), np.ones(len(tgt), dtype=bool)]
    dates = all_rows["Date"].to_numpy()

    probs = np.full((len(tgt), 3), np.nan)
    tgt_dates = tgt["Date"].to_numpy()
    fits = []
    for d in sorted(pd.unique(tgt_dates)):
        use = ~is_target | (dates < d)                       # strictly earlier target matches only
        rows = all_rows[use]
        w = exponential_decay_weights(rows["Date"], pd.Timestamp(d), half_life_days)
        theta, info = fit(rows, w, par, prior, settings)
        on = tgt_dates == d
        lam, mu = rates(theta, tgt[on], par)
        if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(mu)) and np.all(lam > 0) and np.all(mu > 0)):
            raise FitFailure("non-finite or non-positive goal rate")
        probs[on] = outcome_probabilities_from_rates(lam, mu, 0.0, max_goals)
        fits.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "n_fit_matches": int(len(rows)),
                     "n_target_matches_in_fit": int((is_target & use).sum()), **info,
                     "home_advantage": float(theta[1]), "intercept": float(theta[0])})
    if np.isnan(probs).any():
        raise FitFailure("a target match has no forecast")
    name = arm if arm != ANCHOR else "anchor"
    preds = tgt.set_index("match_id")[["Date", "Season", "HomeTeam", "AwayTeam"]].copy()
    preds[prob_columns(name)] = probs
    record = {"arm": arm, "target": structure.target, "n_constrained": len(constrained),
              "n_unconstrained": len(par.unconstrained), "eliminated_team": par.eliminated,
              "n_penalised": len(prior.means), "n_fits": len(fits),
              "max_iterations": max((f["iterations"] for f in fits), default=0),
              "max_abs_gradient": max((f["max_abs_gradient"] for f in fits), default=0.0), "fits": fits}
    return preds, record


# --- Diagnostic taxonomy (Experiment 7 thresholds; reporting only) -------------------------------------

def recent_yoyo_teams(history_rows: pd.DataFrame, structure: TargetStructure, team_history: Mapping[str, int | None],
                      yoyo_threshold: int) -> list[str]:
    """Promoted teams at most `yoyo_threshold` seasons out of the PL: counted from the fold history when the
    team is in it, otherwise from team_history.csv (eplmodel.analysis.promoted_teams.categorize_team)."""
    from eplmodel.analysis.promoted_teams import RECENT_YOYO, categorize_team

    seasons = sorted(set(history_rows["Season"]), key=SEASON_ORDER.index)
    out = []
    for team in structure.promoted:
        if team in structure.returning:
            last = max(s for s in seasons if team in teams_in(history_rows[history_rows["Season"] == s]))
            if SEASON_ORDER.index(structure.target) - SEASON_ORDER.index(last) - 1 <= yoyo_threshold:
                out.append(team)
        elif categorize_team(team, team_history, yoyo_threshold) == RECENT_YOYO:
            out.append(team)
    return out
