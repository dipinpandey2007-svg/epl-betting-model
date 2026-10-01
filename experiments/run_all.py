"""Reproduce every recorded experiment: python -m experiments.run_all"""

from experiments import (
    dixon_coles_rho_search,
    elo_dev_test,
    elo_k_selection,
    goal_models_dev_test,
    promoted_team_folds,
)

EXPERIMENTS = [elo_k_selection, elo_dev_test, dixon_coles_rho_search, goal_models_dev_test, promoted_team_folds]


def main() -> None:
    for module in EXPERIMENTS:
        print(f"Running {module.NAME} ...")
        module.run(write=True)
    print("Done. Results written to results/<experiment>/metrics.json")


if __name__ == "__main__":
    main()
