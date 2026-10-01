import itertools

import numpy as np
import pandas as pd
import pytest

from eplmodel.models.poisson import PoissonGoalModel, UnseenTeamError, to_long_format


@pytest.fixture
def league():
    rng = np.random.default_rng(3)
    strength = {"A": 0.4, "B": 0.1, "C": -0.1, "D": -0.4}
    rows = []
    for _ in range(20):
        for home, away in itertools.permutations(strength, 2):
            rows.append({"HomeTeam": home, "AwayTeam": away, "Season": "x",
                         "FTHG": rng.poisson(np.exp(0.3 + strength[home] - strength[away])),
                         "FTAG": rng.poisson(np.exp(0.05 + strength[away] - strength[home]))})
    return pd.DataFrame(rows)


def test_long_format_has_two_rows_per_match(league):
    long = to_long_format(league)
    n = len(league)
    assert len(long) == 2 * n
    assert long["IsHome"].iloc[:n].eq(1).all() and long["IsHome"].iloc[n:].eq(0).all()
    assert long.loc[0, ["Team", "Opponent", "Goals"]].tolist() == league.loc[0, ["HomeTeam", "AwayTeam", "FTHG"]].tolist()
    assert long.loc[n, ["Team", "Opponent", "Goals"]].tolist() == league.loc[0, ["AwayTeam", "HomeTeam", "FTAG"]].tolist()


def test_fit_recovers_home_advantage_and_ranks_teams(league):
    model = PoissonGoalModel().fit(league)
    assert model.is_home_coef > 0
    lam, mu = model.predict_rates(["A"], ["D"])
    assert lam[0] > mu[0]


def test_unseen_team_raises(league):
    model = PoissonGoalModel().fit(league)
    assert model.unseen_teams(["A", "Z"]) == {"Z"}
    with pytest.raises(UnseenTeamError):
        model.predict_rates(["A"], ["Z"])
