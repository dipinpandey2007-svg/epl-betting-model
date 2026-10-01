import numpy as np
import pandas as pd
import pytest

from eplmodel.constants import OUTCOMES
from eplmodel.models.elo import (
    EloOutcomeModel,
    actual_score,
    expected_score,
    home_shift_from_results,
    run_elo,
    update_ratings,
)


def toy_matches(results):
    """Round of fixtures between teams A-D with the given (home_goals, away_goals) in order."""
    fixtures = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("D", "A"), ("C", "B")]
    rows = []
    for i, ((home, away), (hg, ag)) in enumerate(zip(fixtures, results)):
        ftr = "H" if hg > ag else "A" if hg < ag else "D"
        rows.append({"Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "Season": "x",
                     "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag, "FTR": ftr})
    return pd.DataFrame(rows)


def test_expected_score_known_values():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) == pytest.approx(1 / (1 + 10 ** (-0.5)))
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == pytest.approx(1.0)


def test_actual_score():
    assert actual_score(3, 1) == 1.0
    assert actual_score(0, 2) == 0.0
    assert actual_score(1, 1) == 0.5


@pytest.mark.parametrize("home_adv", [0.0, 43.0])
def test_update_is_zero_sum(home_adv):
    new_home, new_away = update_ratings(1520, 1480, 2, 0, k=25, home_adv=home_adv)
    assert (new_home - 1520) + (new_away - 1480) == pytest.approx(0.0, abs=1e-12)


def test_update_known_value():
    assert update_ratings(1500, 1500, 2, 0, k=20) == (1510.0, 1490.0)


def test_run_elo_stores_pre_match_ratings():
    hist = run_elo(toy_matches([(2, 0), (1, 1), (0, 1), (3, 3), (2, 2), (0, 0)]), k=20)
    assert hist.loc[0, ["EloHome", "EloAway"]].tolist() == [1500.0, 1500.0]
    # A won match 0, so A's pre-match rating for match 2 is its post-match-0 rating.
    assert hist.loc[2, "EloHome"] == 1510.0


def test_run_elo_has_no_lookahead():
    """Changing a later result must not change any earlier pre-match rating."""
    base = [(2, 0), (1, 1), (0, 1), (3, 3), (2, 2), (0, 0)]
    altered = base[:3] + [(0, 5)] + base[4:]
    h1 = run_elo(toy_matches(base), k=25)
    h2 = run_elo(toy_matches(altered), k=25)
    cols = ["EloHome", "EloAway"]
    assert h1.loc[:3, cols].equals(h2.loc[:3, cols])  # up to and including the altered match
    assert not h1.loc[4:, cols].equals(h2.loc[4:, cols])


def test_home_shift_inverts_expected_score():
    results = pd.Series(["H"] * 46 + ["D"] * 25 + ["A"] * 29)
    shift = home_shift_from_results(results)
    assert expected_score(shift, 0) == pytest.approx(results.map({"H": 1, "D": 0.5, "A": 0}).mean())
    assert home_shift_from_results(pd.Series(["H", "A", "D", "D"])) == pytest.approx(0.0)


@pytest.fixture
def synthetic_elo_data():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 150, 3000)
    p_home = 1 / (1 + np.exp(-(0.2 + diff / 200)))
    u = rng.random(3000)
    results = np.where(u < p_home * 0.8, "H", np.where(u < p_home * 0.8 + 0.25, "D", "A"))
    return diff, results


def test_outcome_model_columns_follow_outcomes_order(synthetic_elo_data):
    diff, results = synthetic_elo_data
    probs = EloOutcomeModel().fit(diff, results).predict_proba(np.array([400.0, -400.0]))
    assert OUTCOMES == ("H", "D", "A")
    assert probs[0].argmax() == 0  # strong home favourite -> column 0 (H) largest
    assert probs[1].argmax() == 2  # strong away favourite -> column 2 (A) largest
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_constant_feature_shift_is_absorbed_by_intercept(synthetic_elo_data):
    """Same optimum with or without the shift; default-tolerance fits differ only by solver stopping noise."""
    diff, results = synthetic_elo_data
    for tol, atol in [(1e-10, 1e-7), (1e-4, 1e-5)]:
        p0 = EloOutcomeModel(tol=tol).fit(diff, results).predict_proba(diff)
        p1 = EloOutcomeModel(tol=tol).fit(diff + 43.08, results).predict_proba(diff + 43.08)
        assert np.allclose(p0, p1, atol=atol, rtol=0)
