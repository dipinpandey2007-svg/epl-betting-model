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
4. From 2026-10-01, 2022-24 may be used as **historical fitting information** when predicting later seasons, but
   never as a validation or model-selection target (`eplmodel.splits.assert_valid_selection_target`).
5. The final holdout is defined in [HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md): 2025-26 is sealed and 2026-27 is
   the declared next holdout. Holdout accesses are recorded below as entries H0, H1, … *before* they happen.

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

## Final holdout (2025-26) — protocol `holdout_v1`

Every action involving the holdout is recorded here **before** it happens. An entry id in the first column is what
`eplmodel.holdout.open_final_holdout` looks for; an authorised evaluation entry must also appear as an `[[access]]`
table in `configs/holdout_v1.toml`.

| # | When | Action | Outcomes seen? | Specs scored | Notes |
|---|---|---|---|---|---|
| H0 | 2026-10-01 | Protocol `holdout_v1` frozen (tag `holdout-freeze-v1`): 2025-26 sealed, 2026-27 declared next holdout, metrics fixed (log loss primary; Brier, calibration secondary) | no | none | No 2024-25 or 2025-26 data downloaded or inspected by this project at the time of the freeze (the pre-freeze audit requested HTTP headers only, to confirm the file URLs exist). The season is public knowledge; the seal applies to this project's development process (protocol §2). |

## Validation season (2024-25) usage

2024-25 is a selection target and will be scored repeatedly. Each comparison scored on it is recorded in
`RESULTS_LOG.md`, so the number of choices made with it can be audited.

| # | When | Action | Scored? |
|---|---|---|---|
| V0 | 2026-10-01 | Downloaded and validated into dataset `dev_v2` (RESULTS_LOG Experiment 9). Structural checks only: coverage, schema, score/result consistency and date windows. No outcome statistics. | no |
| V1 | 2026-10-01 (registered before any 2024-25 prediction) | Protocol `validation_2425_v1` (`configs/validation_2425_v1.toml`). The run will score the established specs on the six selection folds (targets 2017-18 … 2021-22 and 2024-25; history = every development season before the target). Specs: `elo_k25_logreg_v1` (K = 25, online ratings, calibration layer fitted once on the history); `frequency_baseline_v1`; `poisson_static_v1` (fitted once on the history); `dixon_coles_staged_v1` (rho re-estimated on the history by the frozen staged grid). Groups: full (Elo, baseline; 380 in 2024-25) and common, with no team absent from the history (all four models; 342 in 2024-25, Ipswich excluded). Metrics: log loss, Brier, paired per-match differences against Elo and the baseline (naive SE), and a descriptive home-win calibration table. Results are reported per fold, 2024-25 separately as the primary result, and pooled over six folds as descriptive context. Folds 1–5 were used for development and K selection, so the pooled figure is not an unbiased estimate. No spec, hyperparameter or treatment may be changed on the basis of these results. | **scored**: executed once, 2026-10-01 at 11:19 UTC, with `python -m experiments.validation_2425` at commit `3d35c550971dc4eaa93d78179d92eb423fc8610b` (the pre-registration commit, pushed at 11:18 UTC; `git_dirty: false`). Data `matches_dev_v2.csv` SHA-256 `c726bd5cb30315bb18baf5805733059f4075b922cef1243d8533dbf7b39fd807`. Predictions `results/validation_2425/predictions.csv` (2,280 rows, git-ignored) content SHA-256 `4e1db61efe28c9961cc5dc46c4ffe5cee6e5ac67c5672a6d8fec63507029c80f`. Metrics and full provenance in `results/validation_2425/metrics.json`; results in RESULTS_LOG Experiment 10. Nothing was changed before or after the run. |
