# EPL Betting Model — Current State

_Last updated: 2026-10-01 (after Experiment 11, the update-policy diagnostic)._

## Current phase

The refactor is complete: the exploratory script is now a tested package that reproduces every recorded result.
Since then:

- 2024-25 has been acquired (Experiment 9).
- The established specs have been scored on the selection folds (Experiment 10).
- The pre-registered update-policy diagnostic has been run and recorded (Experiment 11).

No model or specification has been changed.

Completed baseline stages: data pipeline, Elo, static Poisson, staged Dixon-Coles (see `RESULTS_LOG.md`).

## Data

- Dataset v1: 10 EPL seasons, 2014-15 to 2023-24, 3,800 matches (football-data.co.uk). All recorded results use it.
- Dataset dev_v2 (2026-10-01): v1 plus the 2024-25 validation season, 4,180 matches (`RESULTS_LOG.md` Experiment 9).
  Used by the 2024-25 validation run (Experiment 10).
- Raw and processed files are not committed. Recreate them with the download and build commands in the README;
  `data/checksums.json` identifies the exact dataset behind all results.
- Hand-compiled reference table: `data/reference/team_history.csv` (every row sourced and checked on 2026-10-01).

## Splits

| Role | Seasons | Matches |
|---|---|---|
| Training (fitting + walk-forward selection) | 2014-15 … 2021-22 | 3,040 |
| **Exposed development test benchmark** | 2022-23, 2023-24 | 760 |
| Validation (model selection, with the training folds) | 2024-25 | 380, **scored in two pre-registered runs** (2026-10-01: Experiments 10 and 11) |
| **Sealed final holdout** ([protocol](HOLDOUT_PROTOCOL.md)) | 2025-26 (next: 2026-27) | not yet downloaded |

The development test has been scored several times (`TEST_SET_ACCESS_LOG.md`). It may be re-scored only for the
registered, frozen specifications and must not be used to tune, select or compare new choices.

2024-25 has now been scored once as validation (protocol `validation_2425_v1`, RESULTS_LOG Experiment 10). Its results are observed validation evidence and must not be used to tune the established specifications. The 2024-25 results (log loss / Brier, from `results/validation_2425/metrics.json`) were:

| Group | Elo K=25 | Frequency baseline | Static Poisson | Staged Dixon-Coles (rho = −0.02, refitted on 2014-24) |
|---|---|---|---|---|
| full (380) | 0.9848 / 0.5887 | 1.0812 / 0.6558 | — | — |
| common (342; Ipswich excluded) | 0.9836 / 0.5879 | 1.0760 / 0.6520 | 1.0854 / 0.6573 | 1.0857 / 0.6572 |

Elo is online through the season, while the goal models are static, so the gap mixes model family with update policy.

**Experiment 11** (protocol `update_policy_diagnostic_v1`, diagnostic only) decomposed this 2024-25 gap on the 342
common matches. It added two frozen season-start Elo arms:

- **F1:** ratings frozen at the start of the season, with online Elo's own layer;
- **F2:** the same frozen ratings, with its own season-start layer.

> Poisson − Online (+0.1019) = (Poisson − F2) + (F2 − F1) + (F1 − Online)

| Component | 2024-25 log loss (clustered SE) | Pre-registered rule |
|---|---|---|
| F1 − Online: updating | +0.0327 (0.0106) | distinguishable; positive in all 5 historical folds |
| F2 − F1: calibration layer | −0.0020 (0.0018) | not distinguishable |
| Poisson − F2: remaining static-model / history difference | +0.0712 (0.0132) | 2024-25-specific observation (opposite sign in 4 of 5 historical folds); does not establish its cause |

- **In-season updating explains about a third of the 2024-25 gap.** Most of the gap is already present at the start of
  the season.
- **Poisson − F2 mixes history weighting with model family.** This design does not separate them.
- **The pooled six-fold figures are not an unbiased estimate.**

The 2025-26 holdout remains sealed and unacquired.

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
   static and assumes constant team strength over its whole history. Experiment 11 isolated the updating component
   (about a third of the 2024-25 gap). The rest is a difference at the start of the season, in which history weighting
   and model family are still confounded. No goal model yet updates or time-weights its history.
3. **Dixon-Coles is staged, not joint MLE.** It showed no development-test benefit in this static setup.
4. **Elo design choices not yet validated.**
   - Promoted teams start at 1500.
   - There is no regression towards the mean between seasons.
   - Home advantage is applied only through the regression intercept, not inside the updates (any change needs a
     training-fold experiment).
   - The 2014-15 burn-in season is used when fitting the regression.
5. **Limited uncertainty estimates.**
   - Experiments 10 and 11 report paired per-match differences with naive SEs; Experiment 11 adds date-clustered SEs.
   - The development-test results (Experiments 2-6) still have none.
   - With six folds, fold-to-fold variation is the main uncertainty.
6. **Calibration** is assessed only for P(home win), with decile bins, on the development test.
7. **Exposed test set.** 2022-24 is exposed. The final holdout (2025-26) is sealed by protocol (2026-10-01) but its
   data have not yet been acquired.
8. **Limited data.** Only results are used. The raw files also contain shots and bookmaker odds (including
   closing odds), and these are not used yet. There are no xG, lineup or injury data, and no data from lower
   divisions.

## Recommended next research milestone

**A common walk-forward evaluation harness, with a sealed final holdout defined first.**

1. ~~Define and seal a final holdout before looking at it~~: done 2026-10-01 ([HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md),
   tag `holdout-freeze-v1`). 2024-25 was added as dataset dev_v2 and scored as validation (Experiment 10), then used
   by the update-policy diagnostic (Experiment 11), both on 2026-10-01. The sealed acquisition of 2025-26 has not been
   done.
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
