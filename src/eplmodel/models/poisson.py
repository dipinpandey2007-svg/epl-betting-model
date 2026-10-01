"""Static independent Poisson goal model (Maher-style), fitted as a GLM.

    log E[Goals] = Intercept + Team(attack) + Opponent(defence) + IsHome

Each match contributes two rows: the home side's goals (IsHome=1) and the away
side's goals (IsHome=0). Team strengths are constant over the whole fitting
window, so this is a *static* model: unlike Elo it is not updated as results
arrive. It cannot predict for a team absent from the fitting data; this raises
UnseenTeamError instead of failing inside patsy.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FORMULA = "Goals ~ Team + Opponent + IsHome"


class UnseenTeamError(ValueError):
    pass


def to_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (home rows first, then away rows): Team, Opponent, Goals, Season, IsHome."""
    home_rows = matches[["HomeTeam", "AwayTeam", "FTHG", "Season"]].copy()
    home_rows.columns = ["Team", "Opponent", "Goals", "Season"]
    home_rows["IsHome"] = 1
    away_rows = matches[["AwayTeam", "HomeTeam", "FTAG", "Season"]].copy()
    away_rows.columns = ["Team", "Opponent", "Goals", "Season"]
    away_rows["IsHome"] = 0
    return pd.concat([home_rows, away_rows], ignore_index=True)


class PoissonGoalModel:
    def fit(self, matches: pd.DataFrame, weights=None) -> "PoissonGoalModel":
        """Fit the GLM; `weights` (one per match, optional) weight both of a match's goal rows equally.

        weights=None is the unweighted poisson_static_v1 fit, unchanged. With
        weights, the weighted log-likelihood sum_i w_i * loglik_i is maximised
        (statsmodels var_weights; freq_weights give the same point estimates).
        Only point estimates are meaningful: the GLM's standard errors are not
        used anywhere.
        """
        long = to_long_format(matches)
        if weights is None:
            self.result_ = smf.glm(formula=FORMULA, data=long, family=sm.families.Poisson()).fit()
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (len(matches),):
                raise ValueError(f"need one weight per match ({len(matches)}), got shape {w.shape}")
            if not np.all(np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("weights must be finite and strictly positive")
            row_weights = np.concatenate([w, w])  # home rows, then away rows, as in to_long_format
            self.result_ = smf.glm(formula=FORMULA, data=long, family=sm.families.Poisson(),
                                   var_weights=row_weights).fit()
        self.teams_ = frozenset(long["Team"]) | frozenset(long["Opponent"])
        return self

    def team_effects(self) -> pd.DataFrame:
        """Attack ('Team') and defence ('Opponent') effects per team; the reference team's are 0.

        A higher defence effect means the team concedes more.
        """
        params = self.result_.params
        teams = sorted(self.teams_)
        attack = [float(params.get(f"Team[T.{t}]", 0.0)) for t in teams]
        defence = [float(params.get(f"Opponent[T.{t}]", 0.0)) for t in teams]
        return pd.DataFrame({"attack": attack, "defence": defence}, index=pd.Index(teams, name="team"))

    @property
    def is_home_coef(self) -> float:
        return float(self.result_.params["IsHome"])

    def unseen_teams(self, teams: Iterable[str]) -> set[str]:
        return set(teams) - self.teams_

    def predict_rates(self, home_teams, away_teams) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals (lambda for the home side, mu for the away side) for each fixture."""
        home_teams, away_teams = list(home_teams), list(away_teams)
        unseen = self.unseen_teams(home_teams + away_teams)
        if unseen:
            raise UnseenTeamError(f"No fitted coefficients for teams: {sorted(unseen)}")
        lam = self.result_.predict(pd.DataFrame({"Team": home_teams, "Opponent": away_teams, "IsHome": 1}))
        mu = self.result_.predict(pd.DataFrame({"Team": away_teams, "Opponent": home_teams, "IsHome": 0}))
        return np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)
