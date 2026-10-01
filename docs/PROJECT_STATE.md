# EPL Betting Model — Current State

_Last updated: 2026-10-01 (after the repository refactor)._

## Current phase

The refactor is complete: the exploratory script is now a tested package that reproduces every recorded result.
No new methodological experiment has been started since.

Completed baseline stages: data pipeline, Elo, static Poisson, staged Dixon-Coles (see `RESULTS_LOG.md`).

## Data

- 10 EPL seasons, 2014-15 to 2023-24, 3,800 matches (football-data.co.uk).
- Raw and processed files are not committed. Recreate them with the download and build commands in the README;
  `data/checksums.json` identifies the exact dataset behind all results.
- Hand-compiled reference table: `data/reference/team_history.csv` (every row sourced and checked on 2026-10-01).

## Splits

| Role | Seasons | Matches |
|---|---|---|
| Training (fitting + walk-forward selection) | 2014-15 … 2021-22 | 3,040 |
| **Exposed development test benchmark** | 2022-23, 2023-24 | 760 |
| Validation (model selection, with the training folds) | 2024-25 | not yet downloaded |
| **Sealed final holdout** ([protocol](HOLDOUT_PROTOCOL.md)) | 2025-26 (next: 2026-27) | not yet downloaded |

The development test has been scored several times (`TEST_SET_ACCESS_LOG.md`). It may be re-scored only for the
registered, frozen specifications and must not be used to tune, select or compare new choices.

## Frozen specifications (`configs/baselines_v1.toml`)

| Spec | Summary | Dev-test result |
|---|---|---|
| `elo_k25_logreg_v1` | Elo K=25, ratings updated **without** home advantage; multinomial logistic regression on the pre-match rating difference (the 43.08 feature shift is redundant) | LL 0.9527, Brier 0.5642 (760) |
| `frequency_baseline_v1` | Training H/D/A frequencies | LL 1.0525, Brier 0.6355 (760) |
| `poisson_static_v1` | `Goals ~ Team + Opponent + IsHome`, training seasons, static | LL 1.0058, Brier 0.5989 (648) |
| `dixon_coles_staged_v1` | Poisson coefficients fixed; rho = -0.04 from a training-likelihood grid | LL 1.0072, Brier 0.5994 (648) |
| Elo on the same 648 | — | LL 0.9547, Brier 0.5661 |

## Methodological limitations still open

1. **Unseen/promoted teams.**
   - The static goal models cannot score 112 of the 760 development-test matches (Nott'm Forest, Luton).
   - Inside training, about 17% of validation-fold matches involve an unseen team (Experiment 7).
   - Elo hides the same problem by starting new teams at the league average.
   - Excluding these matches is not an acceptable final policy.
2. **Unfair Elo vs goal-model comparison.** Elo updates dynamically through the test period; the Poisson model is
   static and assumes constant team strength for eight seasons. The current gap mixes model family with update
   dynamics.
3. **Dixon-Coles is staged, not joint MLE.** It showed no development-test benefit in this static setup.
4. **Elo design choices not yet validated.**
   - Promoted teams start at 1500.
   - There is no regression towards the mean between seasons.
   - Home advantage is applied only through the regression intercept, not inside the updates (any change needs a
     training-fold experiment).
   - The 2014-15 burn-in season is used when fitting the regression.
5. **No uncertainty estimates.** Differences between models (for example Poisson vs Dixon-Coles, 0.0014 log loss)
   have no standard errors or paired tests yet.
6. **Calibration** is assessed only for P(home win), with decile bins, on the development test.
7. **Exposed test set.** 2022-24 is exposed. The final holdout (2025-26) is sealed by protocol (2026-10-01) but its
   data have not yet been acquired.
8. **Limited data.** Only results are used. The raw files also contain shots and bookmaker odds (including
   closing odds), and these are not used yet. There are no xG, lineup or injury data, and no data from lower
   divisions.

## Recommended next research milestone

**A common walk-forward evaluation harness, with a sealed final holdout defined first.**

1. ~~Define and seal a final holdout before looking at it~~: done 2026-10-01 ([HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md),
   tag `holdout-freeze-v1`). Next: add 2024-25 to the development data and perform the sealed acquisition of 2025-26.
2. Build one evaluation harness that scores every model on the same training-season walk-forward folds:
   - the same information and update policy for every model (for example, goal models refitted or time-weighted
     as each season progresses);
   - paired comparisons with uncertainty estimates (for example, bootstrap over matches).
3. Inside that harness, run the deferred experiments as separate, pre-registered comparisons:
   - promoted-team handling, for both the goal models and Elo;
   - home advantage inside the Elo updates;
   - a dynamic or time-decayed Poisson model;
   - joint Dixon-Coles.

Only after this should the project move to stage 4+ features (xG, market odds) in `ROADMAP.md`.
