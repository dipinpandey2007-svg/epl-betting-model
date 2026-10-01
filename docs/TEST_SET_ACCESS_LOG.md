# Test-set access log

## Status of 2022-23 and 2023-24

From 2026-10-01 these two seasons (760 matches) are an **exposed development test benchmark**, not a final
holdout. They were scored several times during exploratory work (listed below), so any further model choice
made while looking at them would be tuned on them, at least implicitly.

Rules from now on:

1. The 2022-24 seasons must not be used to tune, select or compare any new methodological choice. Choices are
   made on walk-forward folds inside the training seasons.
2. Re-scoring a specification already listed below reproduces a recorded result and adds no new exposure.
   The code enforces this: `eplmodel.splits.require_registered_dev_test_spec` refuses any spec not in
   `REGISTERED_DEV_TEST_SPECS`.
3. Scoring any other specification on 2022-24 needs a deliberate decision, recorded here *before* it happens,
   and then a new entry in `REGISTERED_DEV_TEST_SPECS`.
4. A genuinely untouched final holdout will be created later from additional historical data. It will be
   defined (seasons and freeze date) before anyone looks at the results for it.

## Exposure history

Dates for the exploratory phase come from file timestamps (September 2026); the exact order of events
is taken from the order of code in `archive/exploratory/elo.py`.

| # | When | What was scored on 2022-24 | Result | Spec id | Notes |
|---|---|---|---|---|---|
| 1 | Sep 2026, exploratory | Elo **K=20** with home advantage (43.08) **inside the rating updates**, then logistic regression | LL 0.9540, Brier 0.5649 | `elo_k20_exploratory` | Scored **before** K was tuned. Also produced `calibration_home.png`. |
| 1b | same | Frequency baseline (training H/D/A rates) | LL 1.0525, Brier 0.6355 | `frequency_baseline_v1` | |
| 2 | Sep 2026, after walk-forward K selection | Elo **K=25**, ratings updated **without** home advantage, logistic regression | LL 0.9527, Brier 0.5642 | `elo_k25_logreg_v1` | Frozen Elo baseline. Also produced `calibration_home_k25.png`. |
| 3 | Sep 2026 | Static Poisson GLM on the 648-match common subset | LL 1.0058, Brier 0.5989 | `poisson_static_v1` | The first printed log loss (1.357) was wrong because of sklearn's class ordering; the corrected value is the one recorded. |
| 4 | Sep 2026 | Elo K=25 restricted to the same 648 matches | LL 0.9547, Brier 0.5661 | `elo_k25_logreg_v1` | |
| 5 | Sep 2026 | Staged Dixon-Coles, rho = -0.04 (chosen on training data), 648 matches | LL 1.0072, Brier 0.5994 | `dixon_coles_staged_v1` | |
| 6 | 2026-10-01, repository audit | (a) Elo K=25 with and without the 43.08 feature shift; (b) an **unregistered variant** with home advantage applied inside the Elo updates | (a) LL 0.952718 both; (b) LL 0.952583 | (a) `elo_k25_logreg_v1`; (b) none | (b) was run only to diagnose what the code does. It is an exposure of a candidate future change: the planned validation experiment on applying home advantage inside Elo updates must be decided on training folds, and this number must not influence it. |
| 7 | 2026-10-01, refactor | Re-scored entries 1–5 to verify the refactored code | identical to the values above | registered specs | Reproduction only; no new exposure. |

### Indirect uses of 2022-24 data (not scoring, but recorded for completeness)

- An exploratory Poisson GLM was fitted on all ten seasons to illustrate the model (IsHome 0.2175 vs 0.2108
  on training seasons). It was never used for evaluation.
- An exploratory Elo home advantage was computed from all ten seasons before the training-only value replaced it.
- Goal-count means, variances and Poisson fit were inspected on all ten seasons; this informed the remark
  about mild overdispersion.
- `explore_data.py` (now `archive/exploratory/explore_data.py`) inspected the 2023-24 raw file's columns and date range.
