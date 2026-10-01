"""Experiments 3 and 6: static Poisson and staged Dixon-Coles on the development test benchmark.

The static goal models cannot predict for teams absent from the training
seasons (Nott'm Forest, Luton), so they are scored on the remaining common
subset, and the frozen Elo baseline is scored on exactly the same matches.

Interpretation caveat: Elo keeps updating its ratings through the test seasons
from earlier test results (legitimate: known before kickoff), while the static
goal models are frozen at the end of 2021-22 and assume constant team strength
across 2014-2022. A lower Elo loss here therefore mixes model family with
update dynamics; it does not show that Elo is intrinsically better than a
goal model.

Re-scores registered specifications only; adds no new test-set exposure.
"""

import numpy as np

from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.evaluation.alignment import involves_teams, select_by_match_id, teams_in
from eplmodel.evaluation.metrics import score
from eplmodel.models.poisson import PoissonGoalModel
from eplmodel.models.scoreline import captured_mass, outcome_probabilities_from_rates
from eplmodel.reporting.results import write_results
from eplmodel.splits import DEV_TEST_SEASONS, TRAIN_SEASONS, require_registered_dev_test_spec
from experiments.elo_dev_test import elo_predictions

NAME = "goal_models_dev_test"


def run(write: bool = True) -> dict:
    config = load_config()
    matches = load_matches()
    train = matches[matches["Season"].isin(TRAIN_SEASONS)]
    test = matches[matches["Season"].isin(DEV_TEST_SEASONS)]

    goal_model = PoissonGoalModel().fit(train)
    unseen = sorted(teams_in(test) - goal_model.teams_)
    excluded = involves_teams(test, unseen)
    common = test[~excluded]

    lam, mu = goal_model.predict_rates(common["HomeTeam"], common["AwayTeam"])
    max_goals = config["poisson"]["max_goals"]
    rho = config["dixon_coles"]["selected_rho"]

    require_registered_dev_test_spec(config["poisson"]["spec_id"])
    poisson_probs = outcome_probabilities_from_rates(lam, mu, rho=0.0, max_goals=max_goals)
    require_registered_dev_test_spec(config["dixon_coles"]["spec_id"])
    dc_probs = outcome_probabilities_from_rates(lam, mu, rho=rho, max_goals=max_goals)
    elo_all, _ = elo_predictions(matches, config["elo"])
    elo_common = select_by_match_id(elo_all, common["match_id"]).to_numpy()

    masses = np.array([captured_mass(l, m, 0.0, max_goals) for l, m in zip(lam, mu)])
    result = {
        "unseen_teams": unseen,
        "n_dev_test_matches": len(test),
        "n_excluded_matches": int(excluded.sum()),
        "n_common_matches": len(common),
        "poisson_is_home_coef": goal_model.is_home_coef,
        "dixon_coles_rho": rho,
        "min_captured_mass_poisson": float(masses.min()),
        "poisson_static": score(common["FTR"], poisson_probs),
        "dixon_coles_staged": score(common["FTR"], dc_probs),
        "elo_same_matches": score(common["FTR"], elo_common),
    }
    if write:
        write_results(NAME, result)
    return result


if __name__ == "__main__":
    out = run()
    for key in ("poisson_static", "dixon_coles_staged", "elo_same_matches"):
        print(f"{key}: {out[key]}")
