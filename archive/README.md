# Archive

Reference copies of the original exploratory code. **Not maintained, not imported, and not part of the package.**

`exploratory/` holds the scripts and figures exactly as they were at the end of the private, pre-refactor development
history (the public repository starts from a fresh export). They were moved here only after the refactored package
reproduced every established result (`tests/test_golden_results.py`). They are covered by the repository's MIT licence;
the data they read are not.

| File | Replaced by |
|---|---|
| `elo.py` (~900-line research script: Elo, Poisson, Dixon-Coles, promoted-team folds) | `src/eplmodel/models/`, `src/eplmodel/evaluation/`, `src/eplmodel/analysis/`, `experiments/` |
| `download_data.py` | `python -m eplmodel.data.download` |
| `build_dataset.py` | `python -m eplmodel.data.build` |
| `explore_data.py` | checks now in `src/eplmodel/data/validate.py` |
| `calibration_home.png` | exploratory K=20 model on the 2022-24 period (see docs/TEST_SET_ACCESS_LOG.md, entry 1) |
| `calibration_home_k25.png`, `rho_search.png` | regenerated in `results/elo_dev_test/` and `results/dixon_coles_rho_search/` |

Known issues in the archived `elo.py`, kept for the record:

- it prints a wrong Poisson / Dixon-Coles log loss (~1.36) before the corrected value, because sklearn's
  `log_loss` sorts class labels to (A, D, H);
- it fits a Poisson model and computes a home advantage on all ten seasons, including 2022-24, for illustration;
- `TEAM_HISTORY` records Watford as 15 seasons out of the Premier League; the correct value is 8;
- running it end to end blocks on `plt.show()` and writes figures into the working directory.
