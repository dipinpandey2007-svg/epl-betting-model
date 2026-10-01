"""Elo team ratings and the Elo -> H/D/A outcome model.

The validated Elo baseline (elo_k25_logreg_v1) is a two-stage model:

1. Ratings. Every team starts at 1500 and ratings are updated sequentially
   through all matches with K = 25 and **no home-advantage term in the update**
   (home_adv=0). Only pre-match ratings are used as features, so each
   prediction depends only on results before kickoff.
2. Outcome probabilities. A multinomial logistic regression (sklearn default
   L2 penalty, C=1.0) maps the pre-match rating difference to P(H), P(D), P(A).
   It is fitted on training-season matches only.

The feature is (EloHome + home_shift) - EloAway with home_shift ~= 43.08,
derived from training-season results. This shift is REDUNDANT: the regression
has an unpenalised intercept that absorbs any constant added to its single
feature, so the optimum is the same with home_shift=0. In practice the lbfgs
solver stops at sklearn's default tolerance (1e-4) at slightly different
points, so fitted probabilities differ by ~2e-6; with a tight tolerance they
agree to ~1e-8 (tests/test_elo.py, tests/test_golden_results.py). The shift is
kept only so the recorded numbers are reproduced exactly. Applying home
advantage *inside the rating updates* is a different model and would need its
own validation experiment.
"""

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from eplmodel.constants import OUTCOMES, RESULT_TO_SCORE

INITIAL_RATING = 1500.0


def expected_score(rating_a, rating_b):
    """Expected score of A against B (works element-wise on arrays)."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def actual_score(home_goals, away_goals) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def update_ratings(rating_home, rating_away, home_goals, away_goals, k, home_adv=0.0):
    """Return post-match (home, away) ratings. The update is zero-sum."""
    exp_home = expected_score(rating_home + home_adv, rating_away)
    exp_away = expected_score(rating_away, rating_home + home_adv)
    act_home = actual_score(home_goals, away_goals)
    act_away = 1 - act_home
    return rating_home + k * (act_home - exp_home), rating_away + k * (act_away - exp_away)


def run_elo(
    matches: pd.DataFrame,
    k: float,
    home_adv: float = 0.0,
    initial_rating: float = INITIAL_RATING,
) -> pd.DataFrame:
    """Run Elo through `matches` in row order and return a copy with pre-match EloHome / EloAway.

    Every team appearing anywhere in `matches` starts at `initial_rating`, so a
    newly promoted team enters at the league-average rating (a known weakness,
    see docs/PROJECT_STATE.md), and a team returning after relegation resumes
    the rating it had when it left.
    """
    teams = set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    ratings = {team: initial_rating for team in teams}
    elo_home = np.empty(len(matches))
    elo_away = np.empty(len(matches))

    for i, (home, away, hg, ag) in enumerate(
        zip(matches["HomeTeam"], matches["AwayTeam"], matches["FTHG"], matches["FTAG"])
    ):
        elo_home[i], elo_away[i] = ratings[home], ratings[away]
        ratings[home], ratings[away] = update_ratings(ratings[home], ratings[away], hg, ag, k, home_adv)

    out = matches.copy()
    out["EloHome"] = elo_home
    out["EloAway"] = elo_away
    return out


def home_shift_from_results(results: pd.Series) -> float:
    """Rating gap whose Elo expected score equals the mean home 'actual score' (draw = 0.5)."""
    mean_home = results.map(RESULT_TO_SCORE).mean()
    return 400 * math.log10(mean_home / (1 - mean_home))


def elo_difference(history: pd.DataFrame, home_shift: float = 0.0) -> np.ndarray:
    return ((history["EloHome"] + home_shift) - history["EloAway"]).to_numpy()


class EloOutcomeModel:
    """Multinomial logistic regression from Elo difference to (H, D, A) probabilities."""

    def __init__(self, C: float = 1.0, tol: float = 1e-4):
        # C=1.0 and tol=1e-4 are the sklearn defaults used for every recorded result.
        self.C = C
        self.tol = tol

    def fit(self, elo_diff, results) -> "EloOutcomeModel":
        self.lr_ = LogisticRegression(C=self.C, tol=self.tol, max_iter=100 if self.tol >= 1e-4 else 10_000)
        self.lr_.fit(np.asarray(elo_diff, dtype=float).reshape(-1, 1), np.asarray(results))
        return self

    def predict_proba(self, elo_diff) -> np.ndarray:
        """Probabilities with columns in OUTCOMES order (H, D, A), regardless of sklearn's class order."""
        probs = self.lr_.predict_proba(np.asarray(elo_diff, dtype=float).reshape(-1, 1))
        order = [list(self.lr_.classes_).index(o) for o in OUTCOMES]
        return probs[:, order]
