# EPL Betting Model — Results Log

This file records important experiments and decisions so that results do not exist only inside individual chat
conversations. Every model result in Experiments 2–7 is reproduced by `python -m experiments.run_all` (written to
`results/<experiment>/metrics.json`) and locked by `tests/test_golden_results.py`. Experiment 10 is reproduced by
`python -m experiments.validation_2425`, Experiment 11 by `python -m experiments.update_policy_diagnostic`, and
Experiment 12 by `python -m experiments.time_weighted_poisson --stage development` then `--stage validation`,
Experiment 13 by `python -m experiments.online_tw_poisson_diagnostic --stage historical` then `--stage validation`,
Experiment 14 by `python -m experiments.market_benchmark`, Experiment 15 by
`python -m experiments.full_coverage_poisson`, and the descriptive 2024-25 validation V5 by
`python -m experiments.exposed_validation_descriptive`. None of them is in `run_all`. The recorded predictions
of Experiments 10-13 are regenerated row by row by `tests/test_reproduction_recorded.py`, and Experiment 14's
metrics by `tests/test_market.py` (both golden). Data-acquisition records are not
produced by `run_all`. Their checksums and coverage are locked by `tests/test_data.py` against
`data/checksums.json`: dataset v1 also through the golden data fixture, and dataset dev_v2 (Experiment 9) by the
`dev_v2` tests.

**Terminology.** "Development test" = seasons 2022-23 and 2023-24. These were scored repeatedly during
exploratory work, so they are an *exposed development test benchmark*, not a final holdout. See
`TEST_SET_ACCESS_LOG.md`.

## Experiment 1 — Historical dataset construction

### Data

10 EPL seasons, 2014-15 through 2023-24 (football-data.co.uk, `E0` files). 3,800 matches, exactly 380 per season.

### Notes

- Raw season CSVs: `data/raw/` (not committed; recreated with `python -m eplmodel.data.download`).
- Processed dataset: `data/processed/matches.csv` with Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Season.
- Mixed date formats were handled by parsing day-first with mixed formats.
- One blank row (end of `E0_1415.csv`) was dropped.
- Team names were checked for consistency; FTR agrees with the score in every row; no team plays twice on one date.
- The content checksum of the processed file is recorded in `data/checksums.json`. A fresh download on
  2026-10-01 rebuilt a byte-identical processed file.

## Experiment 2 — Elo baseline

### Objective

Create a simple, interpretable team-strength baseline.

### Specification as actually implemented (`elo_k25_logreg_v1`)

Corrected on 2026-10-01: earlier versions of this log described a "training-derived home advantage of 43.08 Elo
points" as part of the Elo model. That is not how the code works.

1. **Ratings.** Every team starts at 1500. Ratings are updated match by match, in date order, with K = 25 and
   **no home-advantage term in the update**. A draw counts as 0.5. Only pre-match ratings are used as features.
2. **Outcome probabilities.** A multinomial logistic regression (sklearn defaults: L2 penalty, C = 1.0, lbfgs,
   tol = 1e-4) maps the pre-match rating difference to P(H), P(D), P(A). It is fitted on the training seasons.
3. **The 43.08 shift.** The regression input is `(EloHome + 43.08) - EloAway`, where 43.08 is the rating gap whose
   Elo expected score equals the training-season mean home score. This shift is **redundant**: the regression's
   unpenalised intercept absorbs any constant added to its only feature, so the fitted model is the same without it.
   In practice the solver stops at slightly different points, so fitted probabilities differ by about 2e-6; with a
   tight solver tolerance they coincide. The shift is kept only so recorded numbers reproduce exactly.

Home advantage therefore enters the model only through the regression intercept. Applying it *inside the rating
updates* would be a different model (ratings would evolve differently); that needs its own validation experiment.

### Validation (Experiment 2a, `results/elo_k_selection/`)

K was selected by expanding-window walk-forward validation inside the training seasons: 5 folds, each validating one
season (2017-18 to 2021-22) using all earlier seasons; the home shift and regression are refitted per fold.

| K | 10 | 15 | 20 | **25** | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| Mean fold log loss | 0.9785 | 0.9739 | 0.9721 | **0.9717** | 0.9720 | 0.9737 | 0.9761 |

The curve is flat between K = 20 and 30 (differences ≤ 0.0004), so the exact choice of K matters little.

### Development test (Experiment 2b, `results/elo_dev_test/`)

Train 2014-15 to 2021-22 (3,040 matches); scored on 2022-24 (760 matches).

| Model | Log loss | Brier |
|---|---|---|
| Elo K=25 + logistic regression | 0.9527 | 0.5642 |
| Frequency baseline (training H/D/A rates) | 1.0525 | 0.6355 |

### Interpretation

Elo is the primary simple benchmark for later models. This result does not imply the model beats bookmakers.

### Limitations

- Newly promoted teams start at 1500 (league average) even though promoted sides are usually weaker; teams returning
  after relegation resume a rating that may be years old. This is an implicit and untested way of handling promoted teams.
- 2014-15 is a burn-in season (everyone starts at 1500) but its matches are included when fitting the regression.
- No regression of ratings towards the mean between seasons.

## Experiment 3 — Independent (static) Poisson

### Objective

Model home and away goal counts and derive full scoreline probabilities.

### Model (`poisson_static_v1`)

`Goals ~ Team + Opponent + IsHome`, a Poisson GLM on two rows per match, fitted on the training seasons only
(IsHome = 0.2108). Team strengths are **constant across 2014-2022**, and the model is not updated during the
development-test seasons.

Scoreline grid: 0-10 goals per side; H/D/A probabilities are renormalised by the captured mass (at least 0.9994
on the scored matches).

Poisson assumptions were checked against observed goal distributions; the raw goal data show mild overdispersion
relative to a simple Poisson assumption. (That check used all ten seasons; see the access log.)

### Test limitation

Nott'm Forest and Luton do not appear in the training seasons, so the static model has no coefficients for them.
112 of the 760 development-test matches involve at least one of them and were excluded, leaving 648 matches.

### Results on the 648-match common subset

| Model | Log loss | Brier |
|---|---|---|
| Static Poisson | 1.0058 | 0.5989 |
| Elo K=25 (same 648 matches) | 0.9547 | 0.5661 |

### Interpretation

On this subset Elo has lower loss than the static Poisson model **under different update dynamics**:

- Elo keeps updating through 2022-24 using earlier 2022-24 results (legitimate, since they are known before kickoff),
  and its ratings respond to changes in team strength over time;
- the static Poisson model is frozen at the end of 2021-22 and assumes each team's strength was constant for eight seasons.

The comparison therefore mixes **model family** (rating vs goal model) with **update policy** (dynamic vs static).
It does **not** show that Elo is intrinsically better than a goal model. A fair comparison needs both models under
the same information and update policy (for example, a Poisson model refitted or time-weighted as the season
progresses), evaluated on training-season folds.

## Experiment 4 — Dixon-Coles tau correction

### Objective

Adjust the independent Poisson scoreline probabilities for the low-score dependence captured by the Dixon-Coles correction.

### Unit tests (now in `tests/test_dixon_coles.py`)

For rho = 0, tau = 1. For lambda = 1.5, mu = 1.2, rho = -0.1: tau(0,0) = 1.18, tau(1,0) = 0.88, tau(0,1) = 0.85,
tau(1,1) = 1.10, tau(2,2) = 1.00. All pass. The correction also preserves total probability (tested on an untruncated grid).

## Experiment 5 — Dixon-Coles rho grid search (staged)

### Objective

Estimate the low-score correction parameter from training data only.

### Method

**Staged, not joint, maximum likelihood**: the training-season Poisson coefficients are held fixed and only rho is
chosen, by maximising the training log-likelihood over rho = -0.30, -0.28, …, 0.28 (`results/dixon_coles_rho_search/`).

### Result

- Best grid value: **rho = -0.04**, training log-likelihood ≈ -8743.491.
- rho = 0 (independent Poisson): ≈ -8744.497. The gain from rho is about 1.0 log-likelihood unit over 3,040 matches.
- Nearby: -0.08 → -8745.208; -0.06 → -8744.006; -0.02 → -8743.655; 0.02 → -8746.018.

Added 2026-10-01: for the training fixtures, rho must lie in about (-0.275, 0.366) for every scoreline probability to
be non-negative. The grid values -0.30 and -0.28 fall outside this range; their likelihoods are finite only because
no affected match ended 0-1. This does not affect the selected value, and the grid search now never selects an
invalid rho.

### Interpretation

The training data fit slightly better with a small negative rho. This alone is not evidence of better out-of-sample prediction.

## Experiment 6 — Staged Dixon-Coles on the development test

(Previously run but not recorded here; added 2026-10-01. `results/goal_models_dev_test/`.)

### Data

Same 648-match common subset as Experiment 3; rho = -0.04 fixed from Experiment 5.

### Results

| Model | Log loss | Brier |
|---|---|---|
| Static Poisson | 1.0058 | 0.5989 |
| Staged Dixon-Coles (rho = -0.04) | 1.0072 | 0.5994 |
| Elo K=25 | 0.9547 | 0.5661 |

### Interpretation

The staged Dixon-Coles correction did **not** improve development-test predictions; it was slightly worse than the
independent Poisson model (+0.0014 log loss). The difference is small and its uncertainty has not been estimated.
Together with the tiny training-likelihood gain, there is no evidence that the correction helps in this static setup.

### Decision

Keep the staged Dixon-Coles result as recorded. Do not tune rho or anything else on the development test. Joint
estimation and dynamic goal models are future work, to be validated on training folds.

## Experiment 7 — Unseen/promoted teams inside the training folds

(Previously run but not recorded here; added 2026-10-01. `results/promoted_team_folds/`.)

### Objective

Measure how often a walk-forward validation fold contains teams absent from its training seasons, so promoted-team
handling can later be compared on training data rather than on the development test.

### Method

Seven expanding folds validating 2015-16 to 2021-22. Each newly appearing team is categorised with
`data/reference/team_history.csv` (Premier League seasons missed before returning, known before kickoff):
"recent yo-yo" if absent ≤ 2 seasons, otherwise "long absence or newcomer". A match is "mixed" when it involves one
of each.

### Results

| Validate | Newly appeared teams | Yo-yo | Long absence / new | Mixed | Affected / 380 |
|---|---|---|---|---|---|
| 2015-16* | Bournemouth, Norwich, Watford | 34 | 70 | 4 | 108 |
| 2016-17 | Middlesbrough | 0 | 38 | 0 | 38 |
| 2017-18 | Brighton, Huddersfield | 0 | 74 | 0 | 74 |
| 2018-19 | Cardiff, Fulham, Wolves | 0 | 108 | 0 | 108 |
| 2019-20 | Sheffield United | 0 | 38 | 0 | 38 |
| 2020-21 | Leeds | 0 | 38 | 0 | 38 |
| 2021-22 | Brentford | 0 | 38 | 0 | 38 |
| **Pooled** | | **34** | **404** | **4** | **442 / 2,660** |

\* Low-confidence fold: only one training season (2014-15).

### Interpretation

About 17% of validation matches involve a team the static model has never seen, and almost all of them are
long-absence or first-time teams. Simply excluding these matches is therefore not a viable long-term policy. The
yo-yo category is too small (one team) to evaluate on its own.

### Data correction

The exploratory table listed Watford as 15 seasons out of the Premier League before 2015-16; the correct value is 8.
Their category (long absence) and all counts are unchanged.

## Experiment 8 — Refactor reproduction check (2026-10-01)

The exploratory script was refactored into the `eplmodel` package and `experiments/`. Every recorded value above
was reproduced from the same data (processed-file checksum in `data/checksums.json`):

- metrics, coefficients, home shift and probabilities match **exactly** (difference 0.0);
- log-likelihoods differ by at most 1.5e-11 (summation order);
- K = 25, rho = -0.04, the 112/648 split and all fold counts match exactly.

The golden regression tests use tolerances of 1e-9 for metrics and 1e-6 for log-likelihoods. The original script is
archived at `archive/exploratory/elo.py`.

## Experiment 9 — Validation-season data acquisition: 2024-25, dataset `dev_v2` (2026-10-01)

### Objective

Add the 2024-25 validation season ([HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md) §1) to the development data, without
changing dataset v1 and without touching the sealed 2025-26 holdout. Nothing was scored: this is data acquisition
only.

### Acquisition record

| Item | Value |
|---|---|
| Source URL | `https://www.football-data.co.uk/mmz4281/2425/E0.csv` (302 redirect to `https://football-data.co.uk/...`) |
| Accessed | 2026-10-01, 10:49:51–10:49:57 UTC, via `python -m eplmodel.data.download --seasons 2425` |
| Raw file | `data/raw/E0_2425.csv`: 380 rows, 120 columns, `Div` = `E0` only; SHA-256 `4b05602b…0399a9` |
| Processed file | `data/processed/matches_dev_v2.csv` (`python -m eplmodel.data.build --dataset dev_v2`): 4,180 rows, 2014-15 … 2024-25, 380 per season; SHA-256 `c726bd5c…39fd807` |
| 2024-25 coverage | 380 matches, 2024-08-16 to 2025-05-25, 20 teams (Ipswich is the only team absent from dataset v1) |

Full hashes are in `data/checksums.json`. Neither file is committed (Football-Data terms).

### Validation checks (all passed)

- Raw file: only `E0` rows; no missing Date, HomeTeam, AwayTeam, FTHG, FTAG or FTR; 20 teams, each with 19 home and
  19 away matches; no duplicate fixture.
- Processed file (`validate_matches`): schema; no nulls; FTR consistent with the score; no team plays itself or twice
  on one date; no duplicate rows; chronological order; 380 matches per season.
- `load_dev_matches`: exactly the 11 `dev_v2` seasons and no holdout season (2025-26, 2026-27).
- Season date windows (`validate_season_dates`, run by the build and by `load_dev_matches`): every match's date lies
  between 1 August of its season's first year and 31 July of the next. The windows do not overlap, so a match
  *dated* in 2025-26's window cannot pass under a 2024-25 label. This checks the `Date` column only; it cannot detect
  a match whose recorded date is itself wrong.
- The latest date in the file is 2025-05-25. `tests/test_data.py` also requires every `dev_v2` date to be before
  2025-07-01.
- Build safety: the build refuses to write any season set other than a dataset's own to that dataset's canonical file,
  so dev_v2 cannot replace `matches.csv`. A missing checksum record for v1 makes the build fail.
- Dataset v1 is unchanged. `data/processed/matches.csv` was rebuilt byte-identical to its recorded checksum. The
  2014-24 rows of `dev_v2` equal v1 match for match.
- `dev_v2` rebuilds deterministically to the recorded checksum.

These checks are locked by `tests/test_data.py`, not by the golden tests: the `dev_v2` checksum, coverage, v1
equality and rebuild tests (these skip when the data are absent, as in CI), and the synthetic-data guard tests.

### Note: row order within a date

The build sorts by `Date` with pandas' default sort, which is not stable. Within a date, the 2014-24 rows of `dev_v2`
are therefore in a different order from v1: 2,045 rows on 452 dates. The dates are in the same sequence, and every
date has the same matches. No team plays twice on one date, so sequential Elo ratings are unaffected (checked: maximum
difference 0.0). However, fits that depend on row order only through floating-point summation can differ in the last
digits. The recorded results remain defined on dataset v1. The sort was not changed, because that would change v1's
bytes.

### Not done

- No model was fitted, tuned, selected or scored on 2024-25, and no outcome statistics were computed for it. 2024-25
  has not yet been used for model selection. For
  the row-order check above, Elo was run through `dev_v2`, but only ratings for 2014-24 matches were compared.
- `data/reference/team_history.csv` has no row for Ipswich (2024-25 entrant). It must be sourced before the
  promoted-team analysis is extended to the 2024-25 fold.

## Experiment 10 — Established specifications on the selection folds; 2024-25 validation (2026-10-01)

`experiments/validation_2425.py`, protocol `validation_2425_v1` (`configs/validation_2425_v1.toml`), results in
`results/validation_2425/metrics.json`.

### Pre-registration

The protocol and access-log entry V1 were committed and pushed (commit `3d35c55`, 11:18 UTC) **before** any
2024-25 prediction was generated. The experiment was then run once, from that clean commit (11:19 UTC). Nothing was
changed before or after the run.

### Objective

Score the established specifications on the six selection folds under one information policy, so that future
candidate models have a baseline row in every fold. No decision about the established specifications is taken from
these results.

### Data and folds

- Dataset `dev_v2` (`matches_dev_v2.csv`, SHA-256 `c726bd5c…39fd807`).
- Targets: 2017-18, 2018-19, 2019-20, 2020-21, 2021-22 and **2024-25**. Each target is predicted from every development
  season before it (`eplmodel.splits.selection_folds`). 2022-24 is history only, in the 2024-25 fold.
- **2024-25 is the primary new validation result.** Folds 2017-18 to 2021-22 were used during development, including
  the selection of K = 25 (Experiment 2a), so they are historical context.

### Specifications and information policy (all frozen before the run)

| Spec | Frozen | Re-estimated once on the fold history | During the target season |
|---|---|---|---|
| `elo_k25_logreg_v1` | K = 25, start 1500, no home advantage in updates, logistic C = 1.0, tol = 1e-4 | home shift (redundant) and logistic regression | ratings updated **online**; each prediction uses pre-match ratings from earlier-dated matches only |
| `poisson_static_v1` | formula, 10-goal grid with H/D/A renormalisation | GLM coefficients | **static** (not updated) |
| `dixon_coles_staged_v1` | staged method, rho grid −0.30 … 0.28 (step 0.02), invalid values never selected | Poisson coefficients, then rho by history likelihood | **static** |
| `frequency_baseline_v1` | — | H/D/A rates | static |

Groups (fixed in advance; models are compared only within a group, aligned by `match_id`):

- **full**: all target matches; Elo and the baseline.
- **common**: target matches in which neither team is absent from the history; all four models.

### 2024-25: primary validation result

**Eligibility.** 380 matches. Ipswich is the only team absent from 2014-15 to 2023-24, so 38 matches are excluded from
the common group, leaving **342**. All three numbers equal the registered values and were checked before scoring.

| Group | Model | n | Log loss | Brier |
|---|---|---|---|---|
| full | Elo K=25 | 380 | 0.9848 | 0.5887 |
| full | Frequency baseline | 380 | 1.0812 | 0.6558 |
| common | Elo K=25 | 342 | 0.9836 | 0.5879 |
| common | Frequency baseline | 342 | 1.0760 | 0.6520 |
| common | Static Poisson | 342 | 1.0854 | 0.6573 |
| common | Staged Dixon-Coles | 342 | 1.0857 | 0.6572 |

Paired per-match differences (model minus reference; negative = model better). The standard errors are naive: they
ignore correlation between matches on the same date.

| Group | Comparison | Log loss (SE) | Brier (SE) |
|---|---|---|---|
| full | baseline − Elo | +0.0964 (0.0216) | +0.0671 (0.0150) |
| common | baseline − Elo | +0.0924 (0.0227) | +0.0641 (0.0157) |
| common | Poisson − Elo | +0.1019 (0.0188) | +0.0695 (0.0133) |
| common | Dixon-Coles − Elo | +0.1022 (0.0190) | +0.0694 (0.0133) |
| common | Poisson − baseline | +0.0094 (0.0207) | +0.0053 (0.0145) |
| common | Dixon-Coles − baseline | +0.0098 (0.0209) | +0.0052 (0.0145) |

**Values fitted on 2014-15 to 2023-24 (3,800 matches):**

- Elo home shift: 46.1679 (redundant).
- Elo logistic regression, classes in sklearn's (A, D, H) order; predictions are reordered to (H, D, A):
  - intercepts 0.080876, −0.237062, 0.156186;
  - coefficients on the rating difference −0.0038644, 0.0000365, 0.0038279.
- Poisson IsHome coefficient: 0.217549. The smallest captured grid mass was 0.99940.
- Dixon-Coles **rho = −0.02**, re-estimated on this history as registered (Experiment 5's −0.04 came from 2014-15 to
  2021-22).
  - Valid range for these fixtures: (−0.2425, 0.2908). Grid values −0.30, −0.28 and −0.26 were invalid and never
    eligible.
  - History log-likelihood: −11094.242 at rho = −0.02 against −11094.738 at rho = 0.
- Frequency baseline: H 0.44947, D 0.23316, A 0.31737.

### Historical folds 2017-18 to 2021-22: development context

These folds were used during development and for selecting K = 25.

| Target | Common n (excluded teams) | Elo full (380) | Baseline full | Elo common | Baseline common | Poisson common | DC common | rho |
|---|---|---|---|---|---|---|---|---|
| 2017-18 | 306 (Brighton, Huddersfield) | 0.9666 / 0.5729 | 1.0668 / 0.6444 | 0.9665 / 0.5720 | 1.0670 / 0.6449 | 0.9836 / 0.5847 | 0.9833 / 0.5845 | −0.02 |
| 2018-19 | 272 (Cardiff, Fulham, Wolves) | 0.9110 / 0.5324 | 1.0459 / 0.6313 | 0.9020 / 0.5265 | 1.0518 / 0.6353 | 0.9033 / 0.5303 | 0.9047 / 0.5313 | −0.04 |
| 2019-20 | 342 (Sheffield United) | 0.9786 / 0.5812 | 1.0645 / 0.6434 | 0.9737 / 0.5766 | 1.0611 / 0.6412 | 0.9708 / 0.5768 | 0.9711 / 0.5768 | −0.04 |
| 2020-21 | 342 (Leeds) | 1.0480 / 0.6199 | 1.0890 / 0.6629 | 1.0489 / 0.6207 | 1.0944 / 0.6665 | 1.0482 / 0.6263 | 1.0499 / 0.6268 | −0.04 |
| 2021-22 | 342 (Brentford) | 0.9544 / 0.5660 | 1.0697 / 0.6479 | 0.9460 / 0.5613 | 1.0709 / 0.6485 | 0.9539 / 0.5651 | 0.9537 / 0.5650 | −0.04 |

Each cell is log loss / Brier. The Elo full-group log losses reproduce Experiment 2a's K = 25 fold values. This is
checked to 1e-9 by `tests/test_validation.py`.

### Pooled six folds: descriptive context, not an unbiased estimate

**K = 25 was selected on folds 2017-18 to 2021-22, so this aggregate is not an unbiased estimate of out-of-sample
performance.** The common group is the union of each fold's common matches.

| Group | Model | n | Log loss | Brier |
|---|---|---|---|---|
| full | Elo K=25 | 2,280 | 0.9739 | 0.5769 |
| full | Frequency baseline | 2,280 | 1.0695 | 0.6476 |
| common | Elo K=25 | 1,946 | 0.9726 | 0.5759 |
| common | Frequency baseline | 1,946 | 1.0709 | 0.6486 |
| common | Static Poisson | 1,946 | 0.9942 | 0.5923 |
| common | Staged Dixon-Coles | 1,946 | 0.9947 | 0.5925 |

Pooled paired log-loss differences (common group, naive SE in brackets):

- Poisson − Elo: +0.0215 (0.0062)
- Dixon-Coles − Elo: +0.0220 (0.0063)
- Poisson − baseline: −0.0768 (0.0090)
- baseline − Elo: +0.0983 (0.0109)

### Interpretation

- **Elo beats the frequency baseline in every fold**, by 0.04–0.13 log loss. In 2024-25 the gap is 0.096, about 4.5
  naive SEs.
- **The goal models score at the level of the baseline in 2024-25.** Poisson − baseline was +0.009 (SE 0.021),
  indistinguishable from it. In folds 1–5 they were close to Elo. That a static goal model fell this far in one season
  is an observation, not an explanation.
- **Information-policy caveat** (stated before the run):
  - Elo updates its ratings online through the target season, while Poisson and Dixon-Coles are fitted once at the
    start of the season.
  - The goal models also assume each team's strength was constant across the whole history, which is ten seasons in
    this fold.
  - The Elo vs goal-model gap therefore mixes **model family** with **update policy**. It does not show that rating
    models are intrinsically better than goal models.
- **Dixon-Coles and Poisson are indistinguishable** in every fold (differences ≤ 0.002), as in Experiment 6.
- **The SEs understate the uncertainty**, because they ignore correlation between matches on the same date. This is a
  single validation season of 380 matches.

### Decision

- The results are recorded as observed validation evidence. **No specification, hyperparameter, eligibility rule or
  promoted-team treatment is changed because of them**, and they must not be used to tune the established
  specifications.
- New candidate models (for example a dynamic goal model, or the deferred frozen-start Elo diagnostic) need their own
  pre-registered experiment on the same folds.

### Provenance

| Item | Value |
|---|---|
| Code | commit `3d35c550971dc4eaa93d78179d92eb423fc8610b`, `git_dirty: false` |
| Data | `data/processed/matches_dev_v2.csv`, SHA-256 `c726bd5cb30315bb18baf5805733059f4075b922cef1243d8533dbf7b39fd807` |
| Predictions | `results/validation_2425/predictions.csv`, 2,280 rows, git-ignored. Content SHA-256 (CRLF normalised to LF) `4e1db61efe28c9961cc5dc46c4ffe5cee6e5ac67c5672a6d8fec63507029c80f` |
| Environment | Python 3.13.15; numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.0 |

Not yet locked by golden tests. The run is not in `experiments.run_all`.

## Experiment 11 — Update-policy diagnostic: online vs season-start Elo vs static goal models (2026-10-01)

`experiments/update_policy_diagnostic.py`, protocol `update_policy_diagnostic_v1`
(`configs/update_policy_diagnostic_v1.toml`), results in `results/update_policy_diagnostic/metrics.json`.
**A diagnostic only: no model is selected, tuned or changed because of these results.**

### Pre-registration

- The protocol and access-log entry V2 were committed and pushed (commit `33b9c62`, 11:42 UTC) **before** any diagnostic
  code ran on real data.
- The implementation was committed and pushed next (commit `b1bf69d`, 11:50 UTC). Its tests use synthetic data, plus
  one check that reads team names from the real fixture lists.
- The experiment was then run once, from that clean commit (11:52 UTC). Nothing was changed before or after the run.

### Objective

Experiment 10 found a large gap between the static goal models and online Elo in 2024-25 (Poisson − Elo +0.102 log
loss). This experiment separates how much of that gap comes from Elo's in-season updating, how much from its
calibration layer, and how much from the remaining difference between a static goal model and a frozen Elo model.

### Data and folds

- Dataset `dev_v2` (`matches_dev_v2.csv`, SHA-256 `c726bd5c…39fd807`).
- The six selection folds of Experiment 10: targets 2017-18 to 2021-22 and **2024-25**. Each target is predicted from
  every development season before it. 2022-24 is history only. The exposed development test and 2025-26 are not used.
- 2024-25 is the primary result. Folds 1-5 are historical context: they were used during development, including the
  selection of K = 25.

### Arms

| Arm | Spec id | Ratings | Calibration layer | Changes during the target season |
|---|---|---|---|---|
| Online Elo | `elo_k25_logreg_v1` | updated after every match | fitted once on the history (online pre-match features) | ratings |
| **F1** | `elo_k25_frozen_ratings_online_layer_v1_diag` | frozen at the end of the history | **the same fitted layer and home shift as online Elo** | nothing |
| **F2** | `elo_k25_season_start_v1_diag` | frozen at the end of the history | its own layer, fitted on season-start features from history seasons 2..n | nothing |
| Static Poisson | `poisson_static_v1` | — | — | nothing |
| Staged Dixon-Coles | `dixon_coles_staged_v1` | — | — | nothing |
| Frequency baseline | `frequency_baseline_v1` | — | — | nothing |

Further rules for the frozen arms:

- They use K = 25, start at 1500, apply no home advantage in the updates and no regression to the mean between seasons.
- A team absent from the history (Ipswich in 2024-25) stays at 1500 for the whole season. A returning team resumes its
  last history rating.
- Every frozen prediction is made before the first target match, from history results only.

The two diagnostic spec ids are not registered for development-test or holdout scoring.

### Decomposition (pre-registered)

On the common group (matches the goal models can score), the per-match losses decompose exactly:

> Poisson − Online = (Poisson − F2) + (F2 − F1) + (F1 − Online)

- **F1 − Online is the updating component.** F1 and online Elo share the layer; only the ratings differ.
- **F2 − F1 is the calibration-layer component.** Same frozen ratings; only the layer differs.
- **Poisson − F2 is the remaining static-model / history difference.** Both arms use only information from before the
  season starts. It mixes history weighting (Poisson weights every history season equally; Elo ratings favour recent
  results) with model family. **This design does not separate the two.**
- Online − F2 is only a derived subtotal, never an effect.
- Sign: each difference is left loss minus right loss, so a positive value means the right-hand arm is better.
- **"Distinguishable" rule:** |mean| > 2 × the date-clustered SE, **and** the 2024-25 sign agrees with at least 3 of
  the 5 historical folds.

### Checks before scoring (all passed, all six folds)

- protocol against the frozen specs, the code and the data checksum;
- registered group sizes and unseen teams;
- online Elo, Poisson, Dixon-Coles and the baseline reproduced the recorded Experiment 10 predictions (largest
  difference about 1e-16);
- F1's layer object reproduced the online Elo predictions exactly;
- F1 equalled online Elo on every match where neither team had yet played in the target season (10 matches per fold;
  8 in 2020-21).
- The identity held to a largest per-match residual of 1.1e-16 (tolerance 1e-12), in every fold, segment, group and the
  pooled data, for both metrics.

### 2024-25: primary result

Scores (log loss / Brier):

| Model | Full (380) | Common (342) | Unseen-team (38, descriptive) |
|---|---|---|---|
| Online Elo | 0.9848 / 0.5887 | 0.9836 / 0.5879 | 0.9957 / 0.5967 |
| F1 | 1.0162 / 0.6086 | 1.0163 / 0.6082 | 1.0155 / 0.6120 |
| F2 | 1.0144 / 0.6074 | 1.0143 / 0.6070 | 1.0152 / 0.6110 |
| Static Poisson | — | 1.0854 / 0.6573 | — |
| Staged Dixon-Coles | — | 1.0857 / 0.6572 | — |
| Frequency baseline | 1.0812 / 0.6558 | 1.0760 / 0.6520 | 1.1281 / 0.6902 |

Decomposition on the common group (mean per-match difference; naive SE / date-clustered SE):

| Component | Log loss | Brier | Rule |
|---|---|---|---|
| **F1 − Online** (updating) | **+0.0327** (0.0116 / 0.0106) | +0.0203 (0.0079 / 0.0072) | **distinguishable**: 3.1 clustered SEs, positive in 5 of 5 historical folds |
| **F2 − F1** (calibration layer) | −0.0020 (0.0017 / 0.0018) | −0.0011 (0.0009 / 0.0009) | not distinguishable (1.1 clustered SEs) |
| **Poisson − F2** (static model / history) | **+0.0712** (0.0130 / 0.0132) | +0.0503 (0.0092 / 0.0092) | **2024-25-specific observation**: 5.4 clustered SEs, but positive in only 1 of 5 historical folds |
| Sum of the three components | +0.101864 | +0.069460 | |
| **Poisson − Online** (total, computed directly) | **+0.1019** (0.0188 / 0.0194) | +0.0695 (0.0133 / 0.0132) | distinguishable: 5.3 clustered SEs, positive in 3 of 5 historical folds |
| *Derived subtotal* F2 − Online | +0.0307 (0.0111 / 0.0105) | +0.0192 (0.0077 / 0.0071) | not an effect |

Descriptive shares of the 2024-25 total log-loss gap: updating 32%, calibration layer −2%, Poisson − F2 70%.

Full group (380): F1 − Online +0.0314 (0.0111 / 0.0106); F2 − F1 −0.0018 (0.0016 / 0.0017); subtotal F2 − Online
+0.0296 (0.0107 / 0.0105).

Unseen-team group (38, descriptive): F1 − Online +0.0198 (clustered SE 0.0386); F2 − F1 −0.0003 (0.0038).

Side comparisons (common group, log loss, clustered SE):

- Dixon-Coles − Poisson: +0.0003 (0.0006).
- Each arm minus the frequency baseline:

  | Arm | Difference |
  |---|---|
  | Online Elo | −0.0924 (0.0224) |
  | F1 | −0.0597 (0.0227) |
  | F2 | −0.0617 (0.0215) |
  | Poisson | +0.0094 (0.0196) |
  | Dixon-Coles | +0.0098 (0.0198) |

Fitted on 2014-15 to 2023-24 for the 2024-25 fold:

- F1 uses online Elo's home shift (46.1679) and layer.
- F2's layer was fitted on 3,420 matches from 2015-16 to 2023-24, with home shift 45.4650.
- F2 coefficients on the rating difference, in classes (A, D, H): −0.0037289, 0.0002050, 0.0035239. Online Elo's are
  −0.0038644, 0.0000365, 0.0038279. F2's layer is flatter, as expected for stale ratings.

### All six folds: components on the common group

Log loss, mean (date-clustered SE):

| Target | n | F1 − Online | F2 − F1 | Poisson − F2 | Total |
|---|---|---|---|---|---|
| 2017-18 | 306 | +0.0138 (0.0117) | −0.0015 (0.0050) | +0.0048 (0.0074) | +0.0171 (0.0172) |
| 2018-19 | 272 | +0.0102 (0.0085) | +0.0012 (0.0024) | −0.0101 (0.0117) | +0.0013 (0.0152) |
| 2019-20 | 342 | +0.0002 (0.0092) | −0.0001 (0.0017) | −0.0030 (0.0119) | −0.0029 (0.0146) |
| 2020-21 | 342 | +0.0188 (0.0144) | −0.0027 (0.0016) | −0.0169 (0.0127) | −0.0008 (0.0152) |
| 2021-22 | 342 | +0.0071 (0.0078) | +0.0013 (0.0017) | −0.0005 (0.0091) | +0.0079 (0.0124) |
| **2024-25** | 342 | **+0.0327** (0.0106) | −0.0020 (0.0018) | **+0.0712** (0.0132) | **+0.1019** (0.0194) |

Naive SEs, in the same order:

| Target | F1 − Online | F2 − F1 | Poisson − F2 | Total |
|---|---|---|---|---|
| 2017-18 | 0.0116 | 0.0054 | 0.0078 | 0.0157 |
| 2018-19 | 0.0095 | 0.0025 | 0.0104 | 0.0139 |
| 2019-20 | 0.0095 | 0.0018 | 0.0122 | 0.0147 |
| 2020-21 | 0.0138 | 0.0017 | 0.0130 | 0.0155 |
| 2021-22 | 0.0081 | 0.0017 | 0.0084 | 0.0105 |
| 2024-25 | 0.0116 | 0.0017 | 0.0130 | 0.0188 |

Brier, mean (date-clustered SE):

| Target | F1 − Online | F2 − F1 | Poisson − F2 | Total |
|---|---|---|---|---|
| 2017-18 | +0.0100 (0.0079) | +0.0004 (0.0027) | +0.0024 (0.0049) | +0.0127 (0.0110) |
| 2018-19 | +0.0069 (0.0059) | +0.0016 (0.0013) | −0.0048 (0.0072) | +0.0038 (0.0087) |
| 2019-20 | +0.0007 (0.0057) | +0.0007 (0.0010) | −0.0012 (0.0073) | +0.0002 (0.0089) |
| 2020-21 | +0.0120 (0.0086) | −0.0016 (0.0008) | −0.0048 (0.0073) | +0.0056 (0.0092) |
| 2021-22 | +0.0052 (0.0053) | +0.0004 (0.0010) | −0.0017 (0.0063) | +0.0038 (0.0083) |
| 2024-25 | +0.0203 (0.0072) | −0.0011 (0.0009) | +0.0503 (0.0092) | +0.0695 (0.0132) |

Scores on the common group (log loss / Brier):

| Target | Online Elo | F1 | F2 | Poisson | Dixon-Coles | Baseline |
|---|---|---|---|---|---|---|
| 2017-18 | 0.9665 / 0.5720 | 0.9803 / 0.5820 | 0.9788 / 0.5823 | 0.9836 / 0.5847 | 0.9833 / 0.5845 | 1.0670 / 0.6449 |
| 2018-19 | 0.9020 / 0.5265 | 0.9122 / 0.5335 | 0.9134 / 0.5351 | 0.9033 / 0.5303 | 0.9047 / 0.5313 | 1.0518 / 0.6353 |
| 2019-20 | 0.9737 / 0.5766 | 0.9739 / 0.5773 | 0.9738 / 0.5780 | 0.9708 / 0.5768 | 0.9711 / 0.5768 | 1.0611 / 0.6412 |
| 2020-21 | 1.0489 / 0.6207 | 1.0677 / 0.6327 | 1.0650 / 0.6311 | 1.0482 / 0.6263 | 1.0499 / 0.6268 | 1.0944 / 0.6665 |
| 2021-22 | 0.9460 / 0.5613 | 0.9531 / 0.5665 | 0.9545 / 0.5669 | 0.9539 / 0.5651 | 0.9537 / 0.5650 | 1.0709 / 0.6485 |
| 2024-25 | 0.9836 / 0.5879 | 1.0163 / 0.6082 | 1.0143 / 0.6070 | 1.0854 / 0.6573 | 1.0857 / 0.6572 | 1.0760 / 0.6520 |

Full-group Elo components (log loss, clustered SE):

| Target | F1 − Online | F2 − F1 |
|---|---|---|
| 2017-18 | +0.0058 (0.0107) | +0.0007 (0.0041) |
| 2018-19 | +0.0141 (0.0079) | +0.0006 (0.0018) |
| 2019-20 | +0.0018 (0.0084) | −0.0002 (0.0016) |
| 2020-21 | +0.0218 (0.0131) | −0.0020 (0.0015) |
| 2021-22 | +0.0043 (0.0066) | +0.0011 (0.0016) |
| 2024-25 | +0.0314 (0.0106) | −0.0018 (0.0017) |

### Pooled six folds: descriptive context, not an unbiased estimate

**K = 25 was selected on folds 2017-18 to 2021-22, so this aggregate is not an unbiased estimate of out-of-sample
performance.** Clusters are (fold target, date).

| Group | Component | Log loss (naive / clustered SE) | Brier (naive / clustered SE) |
|---|---|---|---|
| common (1,946) | F1 − Online | +0.0139 (0.0045 / 0.0044) | +0.0092 (0.0029 / 0.0028) |
| common | F2 − F1 | −0.0007 (0.0011 / 0.0010) | −0.0000 (0.0006 / 0.0006) |
| common | Poisson − F2 | +0.0083 (0.0046 / 0.0048) | +0.0072 (0.0030 / 0.0031) |
| common | Total Poisson − Online | +0.0215 (0.0062 / 0.0066) | +0.0164 (0.0041 / 0.0042) |
| full (2,280) | F1 − Online | +0.0132 (0.0041 / 0.0040) | +0.0091 (0.0027 / 0.0026) |
| full | F2 − F1 | −0.0003 (0.0010 / 0.0009) | +0.0002 (0.0006 / 0.0005) |

Pooled common-group log loss:

| Model | Log loss |
|---|---|
| Online Elo | 0.9726 |
| F1 | 0.9866 |
| F2 | 0.9859 |
| Poisson | 0.9942 |
| Dixon-Coles | 0.9947 |
| Frequency baseline | 1.0709 |

The pooled Poisson − F2 value is dominated by 2024-25.

### Segments (within-season position)

Each match's segment is the mean, over its two teams, of the target-season matches already played (fixtures only).
Common group, log loss, mean (date-clustered SE):

| Segment | 2024-25 n | F1 − Online | F2 − F1 | Poisson − F2 | Total |
|---|---|---|---|---|---|
| 0–9 | 90 | +0.0043 (0.0058) | +0.0007 (0.0034) | +0.0570 (0.0264) | +0.0620 (0.0295) |
| 10–18 | 82 | +0.0423 (0.0169) | −0.0069 (0.0038) | +0.0575 (0.0268) | +0.0928 (0.0351) |
| 19–28 | 90 | +0.0481 (0.0250) | −0.0008 (0.0037) | +0.0944 (0.0291) | +0.1417 (0.0443) |
| 29+ | 80 | +0.0375 (0.0292) | −0.0014 (0.0029) | +0.0750 (0.0221) | +0.1111 (0.0434) |

| Segment | Pooled n | F1 − Online | F2 − F1 | Poisson − F2 | Total |
|---|---|---|---|---|---|
| 0–9 | 512 | +0.0042 (0.0037) | +0.0003 (0.0021) | +0.0089 (0.0085) | +0.0134 (0.0099) |
| 10–18 | 467 | +0.0229 (0.0072) | +0.0009 (0.0020) | +0.0037 (0.0102) | +0.0275 (0.0131) |
| 19–28 | 508 | +0.0130 (0.0100) | −0.0008 (0.0021) | +0.0123 (0.0108) | +0.0246 (0.0139) |
| 29+ | 459 | +0.0167 (0.0123) | −0.0032 (0.0021) | +0.0077 (0.0088) | +0.0212 (0.0160) |

- **The updating component behaves as registered.** It is about 0 in the first segment, where the frozen and online
  ratings have barely diverged, and positive later, both in 2024-25 and pooled.
- **The 2024-25 Poisson − F2 gap is already present in the first segment** (+0.057), before updating could matter. It
  is a difference at the season-start information level.
- As registered:
  - F2 − F1 is not interpreted by segment.
  - Poisson − F2 by segment reflects strength drift during the season, not updating.
  - All segment results are descriptive.

Full-group segments are in `metrics.json`. 2024-25 F1 − Online by segment: +0.0034, +0.0367, +0.0503, +0.0363.

### Calibration diagnostics (descriptive)

| Common group | 2024-25 mean P(H/D/A) | 2024-25 mean entropy | Pooled mean P(H/D/A) | Pooled mean entropy |
|---|---|---|---|---|
| Online Elo | 0.450 / 0.229 / 0.321 | 0.9568 | 0.453 / 0.230 / 0.317 | 0.9399 |
| F1 | 0.452 / 0.229 / 0.319 | 0.9581 | 0.455 / 0.232 / 0.313 | 0.9490 |
| F2 | 0.449 / 0.227 / 0.324 | 0.9670 | 0.452 / 0.231 / 0.317 | 0.9598 |
| Static Poisson | 0.447 / 0.231 / 0.322 | 0.9936 | 0.451 / 0.234 / 0.315 | 0.9798 |
| Staged Dixon-Coles | 0.445 / 0.236 / 0.319 | 0.9953 | 0.447 / 0.242 / 0.311 | 0.9822 |
| Frequency baseline | 0.449 / 0.233 / 0.317 | 1.0632 | 0.453 / 0.241 / 0.306 | 1.0639 |
| *Observed* | *0.421 / 0.243 / 0.336* | | *0.431 / 0.233 / 0.336* | |

- Every model over-predicted home wins in 2024-25.
- F2 is less sharp than F1 (higher entropy), consistent with its flatter layer.
- The 10-bin home-win reliability tables (about 34 matches per bin in 2024-25) are in `metrics.json`. They are too
  small to interpret.

### Interpretation

- **F1 − Online is the updating component.**
  - It is **distinguishable** under the pre-registered rule in 2024-25: +0.0327 log loss, 3.1 clustered SEs.
  - It is **positive in every historical fold** (+0.0002 to +0.0188), though within 2 SEs in each.
  - Its segment pattern matches the mechanism: near zero before ratings diverge, positive afterwards.
  - In-season updating therefore explains part of Elo's advantage: about a third of the 2024-25 gap.
- **F2 − F1 is the calibration-layer component.** It is **not distinguishable** (−0.0020, 1.1 clustered SEs) and is
  small in every fold. Recalibrating the layer for stale ratings changes little. The updating result does not depend
  on the choice between F1 and F2.
- **Poisson − F2 is the remaining static-model / history difference.**
  - In 2024-25 it is +0.0712 (5.4 clustered SEs), about 70% of the gap.
  - In folds 1-5 it was small and negative in four of five; static Poisson was slightly better than frozen Elo.
  - It therefore fails the sign-consistency part of the rule. **The 2024-25 Poisson − F2 result is a 2024-25-specific
    observation and does not establish its cause.**
  - It fits the hypothesis stated before the run: equal weighting of a 10-season history hurts static Poisson. But
    history weighting and model family are not separated by this design, history length grows with the fold, and a
    hypothesis prompted by 2024-25 cannot be confirmed on 2024-25.
- **The answer to Experiment 10's question.** The 2024-25 gap between Elo and the goal models is not mainly a matter of
  online updating. Most of it is already present at the start of the season.
- **Dixon-Coles and Poisson** remain indistinguishable (+0.0003).
- **Date-clustered SEs are close to the naive ones**, so same-date correlation matters little. The main uncertainty is
  the small number of folds (six seasons, one of them primary).

### Limitations

- **K = 25 was selected for online Elo on folds 1-5 and was not re-tuned for the frozen arms.** F1 and F2 are the
  frozen versions of the established spec, not the best possible season-start Elo.
- **History weighting and model family are confounded** in Poisson − F2. History length is also confounded with the
  fold.
- **The pooled six-fold result is not an unbiased estimate.** 2024-25 is a single season of 380 matches.
- **The unseen-team group is too small to interpret** (38 matches in 2024-25). The common group still includes indirect
  effects of new teams on the online ratings of other teams.
- **2024-25 has now been used for two scored comparisons** (Experiments 10 and 11).

### Decision

- **Recorded as diagnostic evidence.** No specification, hyperparameter, eligibility rule, interpretation rule or
  promoted-team treatment is changed because of these results.
- **The two diagnostic arms are not deployment candidates.**
- **A natural follow-up, not started:** a time-weighted or dynamic goal model, compared with online Elo under the same
  online information policy.
  - It would need its own pre-registered experiment on the selection folds.
  - Its confirmatory evidence could not rest on 2024-25.

### Provenance

| Item | Value |
|---|---|
| Pre-registration | commit `33b9c62` (config and access-log entry V2) |
| Code | commit `b1bf69d05aacc6913ecfc29d4364f3c9d65c9ef1`, `git_dirty: false` |
| Data | `data/processed/matches_dev_v2.csv`, SHA-256 `c726bd5cb30315bb18baf5805733059f4075b922cef1243d8533dbf7b39fd807` |
| Predictions | `results/update_policy_diagnostic/predictions.csv`, 2,280 rows, git-ignored. Content SHA-256 (CRLF normalised to LF) `73251375050967fd00c81f66c1c0d5c640e975aa43a2d1042fa234304eacf5e8` |
| Metrics | `results/update_policy_diagnostic/metrics.json`, content SHA-256 `f97f4c698d6fa8044d3af24db95eb3a83bdd642f14378ac6fd952e4463747091` |
| Reproduction reference | Experiment 10 predictions, content SHA-256 `4e1db61efe28c9961cc5dc46c4ffe5cee6e5ac67c5672a6d8fec63507029c80f` |
| Environment | Python 3.13.15; numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.0 |

Not yet locked by golden tests. The run is not in `experiments.run_all`.

## Experiment 12 — Time-weighted static Poisson: does recency weighting of the history help? (2026-10-01)

`experiments/time_weighted_poisson.py`, protocol `time_weighted_poisson_v1` (`configs/time_weighted_poisson_v1.toml`),
results in `results/time_weighted_poisson_development/metrics.json` and
`results/time_weighted_poisson_validation/metrics.json`.

### Pre-registration and run sequence

| Step | Commit | What |
|---|---|---|
| Pre-registration | `b94bba1` | Protocol and access-log entry V3, committed and pushed **before** any time-weighted model was fitted on real data |
| Implementation | `ae54757` | Code and tests; no weighted fit on real data |
| Development stage | run at `ae54757` (`git_dirty: false`) | Every grid half-life on the five development folds; seasons up to 2021-22 only loaded |
| Lock | `e8bc661` | H\* = 730 days written to `[locked]` with the development metrics checksum |
| Validation stage | run at `e8bc661` (`git_dirty: false`) | 2024-25 scored **once, for H = 730 only** |

Nothing was changed before or after either run. The grid, selection rule, 0.002 floor, nested procedure and evidence
criteria are exactly as registered.

### Objective

Experiment 11 found a large 2024-25 gap between static Poisson and frozen season-start Elo (Poisson − F2 = +0.0712)
that was absent in the historical folds. One explanation is that static Poisson weights a 10-season history equally.
This experiment isolates **history weighting within the same model family**: the candidate is static during the
target season, so it is not combined with online updating.

### Candidate (`poisson_time_weighted_v1`)

- `poisson_static_v1` unchanged (`Goals ~ Team + Opponent + IsHome`, 10-goal grid, H/D/A renormalised, unseen-team
  matches excluded), except that each history match is weighted:

  w = 2^(−age / H), with age = days from the match to the latest history date.

- Both goal rows of a match get the same weight (statsmodels `var_weights`). There is no truncation, so the set of seen
  teams and the common groups are unchanged.
- **H is a hyperparameter.** It defines the objective; it cannot be chosen by training likelihood. The Poisson
  coefficients are fitted parameters, estimated by weighted maximum likelihood on the history for a given H.
- H = ∞ is `poisson_static_v1` exactly. It reproduced the Experiment 10 predictions in every fold (largest difference
  about 1e-16).
- Grid (days): 183, 274, 365, 548, 730, 1095, 1460, ∞.

### Data and evidence roles

- Dataset `dev_v2` (SHA-256 `c726bd5c…39fd807`).
- **Development:** targets 2017-18 … 2021-22. The development stage cut the data to seasons ≤ 2021-22 straight after
  reading the file.
- **Validation:** 2024-25, H\* only. The grid was never scored on 2024-25.
- 2022-23 and 2023-24 were history only (in the 2024-25 fit). 2025-26 was not accessed.

### Development stage: selection of H

The criterion is the mean over the five targets of each target's mean log loss on the common group. Differences are
pooled per match, with clusters (target, date).

| H (days) | Criterion | Diff vs H_min | Clustered SE | Within 1 SE |
|---|---|---|---|---|
| 183 | 0.96547 | +0.00130 | 0.00122 | no |
| **274 (H_min)** | **0.96437** | 0 | — | yes |
| 365 | 0.96440 | −0.00007 | 0.00073 | yes |
| 548 | 0.96524 | +0.00067 | 0.00157 | yes |
| **730 (H\*)** | 0.96615 | +0.00154 | 0.00204 | yes |
| 1095 | 0.96753 | +0.00289 | 0.00254 | no |
| 1460 | 0.96842 | +0.00378 | 0.00281 | no |
| ∞ (static) | 0.97197 | +0.00734 | 0.00363 | no |

**H\* = 730 days** (about two seasons): the longest half-life within one clustered SE of H_min. Leave-one-target-out
selections: 730, 730, 548, 730, 365 (leaving out 2017-18 … 2021-22 in turn).

In-sample per fold, H = 730 vs static (log loss / Brier). **Optimistically biased: H\* was selected on these folds.**

| Target | History seasons | Static | H = 730 | Log-loss diff (clustered SE) |
|---|---|---|---|---|
| 2017-18 | 3 | 0.9836 / 0.5847 | 0.9780 / 0.5812 | −0.0057 (0.0024) |
| 2018-19 | 4 | 0.9033 / 0.5303 | 0.8972 / 0.5255 | −0.0061 (0.0033) |
| 2019-20 | 5 | 0.9708 / 0.5768 | 0.9659 / 0.5735 | −0.0049 (0.0036) |
| 2020-21 | 6 | 1.0482 / 0.6263 | 1.0418 / 0.6218 | −0.0063 (0.0044) |
| 2021-22 | 7 | 0.9539 / 0.5651 | 0.9479 / 0.5616 | −0.0061 (0.0049) |

Every finite H from 548 to 1460 is better than static in all five folds, in log loss and Brier. H = 183 is worse than
static in 2019-20 (+0.0004) and 2021-22 (+0.0055).

### Development stage: nested estimate and criterion D

For each outer target, H was selected by the same rule from earlier targets only (2018-19 selects from 2017-18 alone).
Differences are time-weighted − static (negative = weighting better).

| Outer target | Nested H | Log loss (clustered SE) | Brier (clustered SE) |
|---|---|---|---|
| 2018-19 | 274 | −0.0115 (0.0075) | −0.0087 (0.0051) |
| 2019-20 | 183 | +0.0004 (0.0107) | −0.0004 (0.0069) |
| 2020-21 | 365 | −0.0100 (0.0071) | −0.0074 (0.0043) |
| 2021-22 | 365 | −0.0031 (0.0082) | −0.0011 (0.0056) |
| **Pooled (1,298)** | | **−0.0058** (naive 0.0043 / clustered 0.0043) | −0.0042 (0.0028 / 0.0028) |

| Criterion D check | Result |
|---|---|
| Pooled log loss < −0.002 | yes (−0.0058) |
| \|mean\| > 2 clustered SEs | **no** (1.34 SEs) |
| Negative in ≥ 3 of 4 outer folds | yes (3) |
| Pooled Brier negative | yes |

**Criterion D: not met.** The historical gain is consistently signed in-sample but the nested estimate of the
selection procedure is not distinguishable from zero. H\* was not changed because of this.

### Development diagnostics (descriptive)

H = 730 vs static, by target (2017-18 … 2021-22):

- Kish effective sample size: 1050/1140, 1323/1520, 1543/1900, 1697/2280, 1827/2660 matches. At H = 183 about 510–590
  in every fold.
- Weight share of the last history season: 0.45, 0.39, 0.36, 0.34, 0.32 (static 0.33 … 0.14; H = 183 about 0.75).
- Per-team effective sample size: minimum 37.6–37.8, median 105–137. Never below 31.6 at any H.
- All GLM fits converged.
- Rank correlation of net team strength with the static fit: 0.95–0.99 at H = 730 (0.90–0.98 at H = 183). The spread
  of attack and defence effects grows slightly as H shortens; mean entropy falls (sharper forecasts).
- **Home advantage (registered COVID caveat).** In the 2021-22 fold the IsHome coefficient falls from 0.220 (static) to
  0.173 at H = 730 and 0.075 at H = 183, because 2020-21, played largely without crowds, gets the most weight. In the
  other folds it moves by at most 0.03. Short half-lives are therefore penalised in that fold for a reason unrelated
  to team-strength staleness.
- Returning vs continuously present teams, H = 730 − static log loss: 2017-18 Newcastle (34) −0.0122 vs −0.0048;
  2019-20 Aston Villa, Norwich (70) +0.0066 vs −0.0079; 2020-21 Fulham, West Brom (70) −0.0012 vs −0.0076;
  2021-22 Norwich, Watford (70) −0.0205 vs −0.0023; 2018-19 no returning team (−0.0061).

### 2024-25 validation: primary result (H = 730)

Common group: 342 matches (380 minus the 38 involving Ipswich), as registered.

| Model | Log loss | Brier |
|---|---|---|
| Static Poisson | 1.0854 | 0.6573 |
| **Time-weighted Poisson, H = 730** | **1.0417** | **0.6273** |

Paired difference **time-weighted − static** (109 match dates as clusters):

| Metric | Mean | Naive SE | Date-clustered SE |
|---|---|---|---|
| Log loss | **−0.0438** | 0.0076 | 0.0077 |
| Brier | −0.0300 | 0.0053 | 0.0053 |

| Criterion V check | Result |
|---|---|
| Log loss < −0.002 | yes (−0.0438) |
| \|mean\| > 2 clustered SEs | yes (5.7 SEs) |
| Brier negative | yes |

**Criterion V: met.**

### Context comparisons (2024-25 common group; not effects of weighting)

These mix model family and/or in-season updating. Log loss, date-clustered SE:

| Comparison | Log loss | Brier |
|---|---|---|
| Time-weighted − online Elo | +0.0581 (0.0140) | +0.0395 (0.0096) |
| Time-weighted − F2 | +0.0274 (0.0076) | +0.0203 (0.0053) |
| Time-weighted − Dixon-Coles | −0.0441 (0.0078) | −0.0299 (0.0054) |
| Time-weighted − frequency baseline | −0.0343 (0.0187) | −0.0247 (0.0134) |

Scores of the other arms reproduce Experiments 10 and 11: online Elo 0.9836 / 0.5879, F1 1.0163 / 0.6082,
F2 1.0143 / 0.6070, Dixon-Coles 1.0857 / 0.6572, baseline 1.0760 / 0.6520.

### Connection to Experiment 11

Per match on the 342 matches, checked before reporting (largest residual 1.1e-16, tolerance 1e-12):

> Poisson − Online = (Poisson − TW) + (TW − F2) + (F2 − F1) + (F1 − Online)

| Component | Log loss (naive / clustered SE) | Brier (naive / clustered SE) |
|---|---|---|
| Poisson − TW: history weighting at H\* | +0.0438 (0.0076 / 0.0077) | +0.0300 (0.0053 / 0.0053) |
| TW − F2: remainder | +0.0274 (0.0080 / 0.0076) | +0.0203 (0.0056 / 0.0053) |
| F2 − F1: calibration layer | −0.0020 (0.0017 / 0.0018) | −0.0011 (0.0009 / 0.0009) |
| F1 − Online: updating | +0.0327 (0.0116 / 0.0106) | +0.0203 (0.0079 / 0.0072) |
| **Poisson − Online (total)** | **+0.1019** (0.0188 / 0.0194) | +0.0695 (0.0133 / 0.0132) |

- Weighting at H\* closed **0.0438 of the 0.0712 Poisson − F2 gap (about 62%)**.
- **The remaining TW − F2 gap (+0.0274) is not a pure model-family effect.** It also contains goals vs results as the
  input, Elo's own implicit recency (K = 25 smooths over matches, not days), the weighting's functional form, and any
  difference between H\* and a 2024-25-optimal H (which was deliberately never estimated).
- That weighting closes part of the gap shows it is *sufficient* to close that part on these matches. It does not show
  that staleness was the reason for F2's advantage.

### 2024-25 diagnostics (descriptive)

Time-weighted − static by within-season segment (Experiment 11 segments; log loss / Brier, clustered SE):

| Segment | n | Log loss | Brier |
|---|---|---|---|
| 0–9 | 90 | −0.0448 (0.0157) | −0.0332 (0.0112) |
| 10–18 | 82 | −0.0331 (0.0151) | −0.0214 (0.0095) |
| 19–28 | 90 | −0.0566 (0.0156) | −0.0392 (0.0113) |
| 29+ | 80 | −0.0392 (0.0150) | −0.0248 (0.0103) |

The gain is already present in the first segment, as expected for a difference in season-start information.

- **Returning teams** (Leicester, Southampton; 70 matches): −0.0911 (0.0154). Continuously present teams (272):
  −0.0316 (0.0088). The gain is largest for returning teams but not confined to them.
- **Calibration and sharpness.** Mean P(H/D/A): static 0.447 / 0.231 / 0.322 (entropy 0.9936), time-weighted
  0.445 / 0.227 / 0.328 (0.9903); observed 0.421 / 0.243 / 0.336. Both over-predict home wins.
- **Fit on 2014-15 … 2023-24 (3,800 matches):** Kish effective sample size 2,069; last-season (2023-24) weight share
  0.30; per-team effective sample size minimum 37.8, median 140.8; IsHome 0.2003 (static 0.2175); rank correlation of
  net strength with static 0.966; converged.

### Interpretation (pre-registered reading)

- **D not met, V met → "2024-25-specific observation that cannot confirm".** This is the registered reading and it is
  not upgraded.
- Historically, recency weighting gave a small, consistently signed in-sample gain (about 0.006 log loss at H\*), but
  the honest nested estimate (−0.0058, 1.34 clustered SEs) is not distinguishable from zero. In 2024-25 the gain is
  about seven times larger.
- The hypothesis that equal weighting of a long history hurts static Poisson was prompted by 2024-25, so 2024-25
  cannot confirm it, however clear the 2024-25 result is.
- **History length is confounded with the fold.** The development folds had 3–7 history seasons; 2024-25 had 10.
  A mechanism in which equal weighting hurts more as history lengthens would fit the pattern, but this design cannot
  separate it from 2024-25 being unusual. 2022-23 and 2023-24 can never be targets, so no historical fold with a
  similar history length exists.
- The 2021-22 development fold is affected by the COVID-era home-advantage shift, which weighting cannot separate from
  team-strength recency because IsHome is shared.

### Limitations

- Only four nested outer folds; 2018-19 selects H from a single earlier fold.
- One weighting form (exponential in days) and one grid; no other forms were tested.
- 2024-25 has now been used for three scored comparisons (Experiments 10, 11 and 12).
- The pooled development figures are not an unbiased estimate of the locked candidate; only the nested figures and
  2024-25 are honest out-of-sample estimates for their own seasons.
- Unseen/promoted teams are still excluded from the goal-model comparisons.

### Decision

- **Recorded as is.** H\* = 730 days stays locked. No model, hyperparameter, protocol, eligibility rule or evidence
  criterion is changed because of these results.
- **The 2024-25 result must not be used to retune the candidate** (for example to choose a different half-life).
- `poisson_time_weighted_v1` is not registered for 2022-24 or holdout scoring. Confirming the mechanism would need a
  future pre-registered evaluation on a season not used to generate the hypothesis.

### Provenance

| Item | Value |
|---|---|
| Pre-registration | commit `b94bba1` (config and access-log entry V3) |
| Development run | commit `ae547570c8d8368bfe30fd91f6a763237637d5f9`, `git_dirty: false` |
| Development metrics | `results/time_weighted_poisson_development/metrics.json`, content SHA-256 `e21495a1a7695179787d319ad8954014274669582ad53634c9831de181d5a215` |
| Development predictions | `results/time_weighted_poisson_development/predictions.csv`, 1,604 rows, git-ignored, content SHA-256 `96831b967d1841a71bbd95f141147446691891e55e8a939477036ea521c1c3ba` |
| Lock | commit `e8bc661e4bf5abcef8d50f95f59a17d6b25ee0d3` |
| Validation run | commit `e8bc661e4bf5abcef8d50f95f59a17d6b25ee0d3`, `git_dirty: false` |
| Validation metrics | `results/time_weighted_poisson_validation/metrics.json`, content SHA-256 `fda8da0931609eb3b3df98c0cc7aaccd684dd5f4526fbb02d676afcbb7952086` |
| Validation predictions | `results/time_weighted_poisson_validation/predictions.csv`, 342 rows, git-ignored, content SHA-256 `3dcc77393ae57a9752f172920a31bb107fe9af16d55d6789990b5d8713c4e37b` |
| Data | `data/processed/matches_dev_v2.csv`, SHA-256 `c726bd5cb30315bb18baf5805733059f4075b922cef1243d8533dbf7b39fd807` |
| Reproduction references | Experiment 10 predictions `4e1db61e…c80f`; Experiment 11 predictions `73251375…e5f8` |
| Environment | Python 3.13.15; numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.0 |

Not yet locked by golden tests. The run is not in `experiments.run_all`.

## Experiment 13 — Online vs frozen time-weighted Poisson: does in-season updating help within the Poisson family? (2026-10-01)

`experiments/online_tw_poisson_diagnostic.py`, protocol `online_tw_poisson_diagnostic_v1`
(`configs/online_tw_poisson_diagnostic_v1.toml`), results in `results/online_tw_poisson_historical/metrics.json` and
`results/online_tw_poisson_validation/metrics.json`. **Diagnostic only: no specification is adopted.**

### Pre-registration and run sequence

| Step | Commit | What |
|---|---|---|
| Pre-registration | `b91217e` | Protocol and access-log entry V4, committed and pushed **before** any online weighted Poisson model was fitted on real data |
| Implementation | `4ad9782` | Code and tests; no online fit on real data |
| CI fix (test-only) | `7faadfc` | Two pre-existing end-to-end tests read the git-ignored dataset (CI red since `3d35c55`); fixed without touching library, model, experiment or config code |
| Historical stage | run at `7faadfc` (`git_dirty: false`) | Online and frozen arms on targets 2017-18 … 2021-22; seasons up to 2021-22 only loaded |
| Lock | `d5c515c` | `[historical_locked]` records the historical metrics checksum and the run commit |
| 2024-25 stage | run at `d5c515c` (`git_dirty: false`) | Online arm at H = 730 only, once; diagnostic observation (fourth scored use of 2024-25) |

Nothing was changed before or after the run. H, the refit cadence, the unseen-team treatment, the evidence criteria and
the interpretation rules are exactly as registered.

### Objective

Experiment 11 showed that in-season updating explains part of online Elo's advantage (F1 − Online). Experiment 12
showed that recency weighting (H\* = 730 days) improves the *static* Poisson model in 2024-25. This experiment asks
whether the Poisson model also benefits from in-season updating once its recency weighting is fixed. The primary
comparison is **within the Poisson family**: online minus frozen time-weighted Poisson on identical matches.

### Arms

- **Frozen** `poisson_time_weighted_v1`: H = 730 days (locked in Experiment 12), fitted once on the fold history,
  never updated in-season. Reproduced from the Experiment 12 predictions.
- **Online** `poisson_tw_online_h730_v1_diag` (the only new arm): the same weighted likelihood, w = 2^(−age/730),
  refitted **once per target date** on

  S(d) = fold history + common-group target matches dated **strictly before** d,

  with age in days to the latest date in S(d).
  - The reference date is a convention only. Weight ratios between matches do not depend on it, so the online arm
    differs from the frozen arm only in which matches are in the likelihood.
  - Every match on a date has the same information set, so one refit per date equals one refit per match.
  - No outcome dated d enters any prediction on d.
  - All parameters are refitted (intercept, attack, defence, IsHome). There are no warm starts.
- **Unseen teams:** their target matches are neither scored nor used in any online fit. Both arms therefore keep the
  fixed team parameter space of the history (decision approved before registration).
- **Fit validity:** converged, finite parameters, finite positive rates. An invalid fit gets one retry of the same IRLS
  with maxiter 1000; otherwise the stage aborts, with no fallback.
- Established arms (online Elo, F1, F2, static Poisson, Dixon-Coles, baseline) are reproduced from the Experiment 11
  predictions.

### Data and evidence roles

- Dataset `dev_v2` (SHA-256 `c726bd5c…39fd807`). The historical stage cut the data to seasons ≤ 2021-22 straight after
  reading the file.
- Targets 2017-18 … 2021-22 (`selection_folds`), common groups 306 / 272 / 342 / 342 / 342 (1,604 matches).
- No hyperparameter is selected.
  - **H = 730 was selected on these same folds for the frozen arm** (Experiment 12). The online arm inherits it, so any
    selection bias favours the frozen arm.
  - Elo's K = 25 was also selected on these folds.
- The historical stage did not load or score 2024-25; the 2024-25 stage ran only after the historical lock (see below).
  2022-23 and 2023-24 are never targets. 2025-26 was not accessed.

### Pre-scoring checks (all passed)

- Protocol, data and recorded-prediction checksums; registered group sizes and unseen teams.
- Registered online-fit counts: 98 / 98 / 110 / 128 / 119 (553).
- Reproduction of the Experiment 11 and 12 predictions: largest difference 1.1e-16.
- Online = frozen on each fold's first target date: largest gap 2.8e-17.
- Decomposition identity: largest per-match residual 1.1e-16.

### Primary result: online − frozen (negative = online better)

| Target | n | Online LL / Brier | Frozen LL / Brier | ΔLL (naive / clustered SE) | ΔBrier (naive / clustered SE) |
|---|---|---|---|---|---|
| 2017-18 | 306 | 0.9629 / 0.5710 | 0.9780 / 0.5812 | −0.0151 (0.0053 / 0.0059) | −0.0102 (0.0037 / 0.0042) |
| 2018-19 | 272 | 0.8874 / 0.5188 | 0.8972 / 0.5255 | −0.0097 (0.0043 / 0.0040) | −0.0067 (0.0030 / 0.0028) |
| 2019-20 | 342 | 0.9620 / 0.5703 | 0.9659 / 0.5735 | −0.0039 (0.0049 / 0.0041) | −0.0032 (0.0034 / 0.0027) |
| 2020-21 | 342 | 1.0202 / 0.6085 | 1.0418 / 0.6218 | −0.0217 (0.0081 / 0.0086) | −0.0133 (0.0051 / 0.0055) |
| 2021-22 | 342 | 0.9397 / 0.5558 | 0.9479 / 0.5616 | −0.0082 (0.0050 / 0.0053) | −0.0059 (0.0035 / 0.0036) |
| **Pooled** | **1,604** | **0.9572 / 0.5667** | **0.9689 / 0.5746** | **−0.0117** (0.0026 / 0.0027) | **−0.0079** (0.0017 / 0.0018) |

Clusters are (target, date): 553. The equal-fold-weight mean ΔLL is −0.0117. The frozen scores equal Experiment 12's
H = 730 values.

| Criterion U check | Result |
|---|---|
| Pooled ΔLL < −0.002 | yes (−0.0117) |
| \|mean\| > 2 clustered SEs | yes (4.4 SEs) |
| ΔLL < 0 in ≥ 4 of 5 folds | yes (**5 of 5**) |
| Pooled ΔBrier < 0 | yes (−0.0079) |

**Criterion U: met. Mirror (harm) criterion: not met** (no fold positive). **Registered reading:
`historical_evidence_online_refitting_helps`.**

### Segments (pre-registered accumulation hypothesis; descriptive)

Experiment 11 segments (mean prior target-season games of the two teams). Pooled historical, clustered SEs:

| Segment | n | ΔLL | ΔBrier | Mean summed \|ΔP\| vs frozen | Elo F1 − Online (LL) |
|---|---|---|---|---|---|
| 0–9 | 422 | −0.0051 (0.0022) | −0.0035 (0.0014) | 0.030 | +0.0042 (0.0043) |
| 10–18 | 385 | −0.0133 (0.0043) | −0.0091 (0.0029) | 0.064 | +0.0188 (0.0078) |
| 19–28 | 418 | −0.0106 (0.0058) | −0.0069 (0.0038) | 0.077 | +0.0055 (0.0109) |
| 29+ | 379 | −0.0187 (0.0078) | −0.0125 (0.0052) | 0.091 | +0.0123 (0.0137) |

Per fold, ΔLL by segment:

| Target | 0–9 | 10–18 | 19–28 | 29+ |
|---|---|---|---|---|
| 2017-18 | −0.0117 | −0.0139 | −0.0140 | −0.0209 |
| 2018-19 | −0.0055 | −0.0091 | +0.0020 | −0.0278 |
| 2019-20 | +0.0026 | −0.0105 | +0.0074 | −0.0168 |
| 2020-21 | −0.0069 | −0.0318 | −0.0283 | −0.0207 |
| 2021-22 | −0.0047 | −0.0002 | −0.0182 | −0.0091 |

- **The registered accumulation pattern is not met.** The later segments are below 0–9, as hypothesised, but Δ(0–9)
  is 2.3 clustered SEs from zero, outside the registered "near zero" band (≤ 2 SEs).
- **The gain is already present early.** The 0–9 segment includes matches in which teams have played up to about nine
  games, and shared parameters (intercept, IsHome, the connected team network) move after any target result.
- **The size of the online adjustment grows steadily** through the season (mean |ΔP| 0.030 → 0.091).
- Segment results are descriptive and do not affect the reading.

### Comparison with Elo's updating gain (descriptive)

Each family's updating gain within its own family (left − right; positive = the updating arm is better):

| Target | Poisson: frozen TW − online TW | Elo: F1 − Online |
|---|---|---|
| 2017-18 | +0.0151 | +0.0138 |
| 2018-19 | +0.0097 | +0.0102 |
| 2019-20 | +0.0039 | +0.0002 |
| 2020-21 | +0.0217 | +0.0188 |
| 2021-22 | +0.0082 | +0.0071 |
| Pooled | **+0.0117** (clustered SE 0.0027) | **+0.0099** (0.0048) |

- The two updating gains are of similar size and move together across folds.
- **The pre-registered expectation was not borne out.** It said the Poisson gain would be *smaller* than Elo's, because
  the current season carries only part of the weight at H = 730. It was an expectation, not a criterion.
- **The difference between the two gains is not a model-family effect.** The families differ in input (goals vs
  results), update mechanism (global refit vs per-team sequential updates), recency form (days vs per match) and
  calibration layer.

### Decomposition and context comparisons (pooled historical; descriptive)

Per match on the common group, checked to 1e-12 (left − right; positive = right-hand arm better):

| Component | ΔLL (clustered SE) | ΔBrier |
|---|---|---|
| Static Poisson − frozen TW (weighting) | +0.0058 (0.0018) | +0.0039 |
| Frozen TW − online TW (Poisson updating) | +0.0117 (0.0027) | +0.0079 |
| Online TW − online Elo | −0.0131 (0.0046) | −0.0066 |
| **Static Poisson − online Elo (total)** | **+0.0044** (0.0067) | +0.0051 |

Context comparisons (not effects), log loss with clustered SE:

| Comparison | ΔLL (clustered SE) |
|---|---|
| Online TW − online Elo | −0.0131 (0.0046) |
| Frozen TW − F2 | −0.0110 (0.0039) |
| Online TW − static Poisson | −0.0175 (0.0034) |
| Online TW − frequency baseline | −0.1127 (0.0095) |
| Frozen TW − frequency baseline | −0.1009 (0.0095) |

Pooled historical scores (log loss / Brier):

| Model | Log loss / Brier |
|---|---|
| Online TW | 0.9572 / 0.5667 |
| Frozen TW | 0.9689 / 0.5746 |
| Static Poisson | 0.9747 / 0.5785 |
| Online Elo | 0.9703 / 0.5734 |
| F1 | 0.9802 / 0.5803 |
| F2 | 0.9799 / 0.5805 |
| Dixon-Coles | 0.9752 / 0.5787 |
| Frequency baseline | 1.0698 / 0.6479 |

On these folds, online TW Poisson scores below online Elo. This mixes model family with update mechanism, recency form
and hyperparameters chosen on these same folds (H = 730, K = 25). It is not evidence that one family is better.

### Home advantage: the COVID caveat

IsHome coefficient through each season (first fit, ¼, ½, ¾, last fit):

| Target | Frozen | Online trajectory |
|---|---|---|
| 2017-18 | 0.268 | 0.268, 0.262, 0.255, 0.273, 0.272 |
| 2018-19 | 0.278 | 0.278, 0.269, 0.262, 0.276, 0.276 |
| 2019-20 | 0.259 | 0.259, 0.252, 0.234, 0.251, 0.248 |
| 2020-21 | 0.251 | **0.251, 0.218, 0.203, 0.194, 0.172** |
| 2021-22 | 0.173 | 0.173, 0.163, 0.178, 0.164, 0.161 |

- **2020-21 was played largely without crowds** (public knowledge, stated in the protocol before the run).
  - The online model re-learns the lower home advantage during the season; the frozen model cannot.
  - Mean P(home win) in 2020-21: online 0.437, frozen 0.448, observed 0.371.
- **2020-21 has the largest gain (−0.0217), and part of it is this home-advantage re-learning.** That is a component of
  "updating" specific to that season. The other four folds are also negative (−0.0039 to −0.0151).

### Other diagnostics (descriptive)

- **Current-season weight share at the last fit:** 0.33, 0.28, 0.31, 0.30, 0.28.
- **Kish effective sample size** grows by about 80–200 matches over a season (2017-18: 1,050 → 1,255; 2021-22: 1,827 →
  1,909).
- **Returning vs continuously present teams** (ΔLL): pooled returning −0.0173 (clustered SE 0.0088, 244 matches),
  continuous −0.0107 (0.0027, 1,360). Per fold:

  | Target | Returning teams (matches) | Returning ΔLL |
  |---|---|---|
  | 2017-18 | Newcastle (34) | −0.0270 |
  | 2018-19 | none | — |
  | 2019-20 | Aston Villa, Norwich (70) | −0.0129 |
  | 2020-21 | Fulham, West Brom (70) | −0.0379 |
  | 2021-22 | Norwich, Watford (70) | +0.0034 |

- **Calibration and sharpness.** The online arm is sharper (lower entropy) in three folds and slightly less sharp in
  2019-20 and 2020-21.

  | Target | Online mean P(H/D/A), entropy | Frozen mean P(H/D/A), entropy | Observed H/D/A |
  |---|---|---|---|
  | 2017-18 | 0.455 / 0.234 / 0.311, 0.9792 | 0.457 / 0.235 / 0.307, 0.9859 | 0.451 / 0.252 / 0.297 |
  | 2018-19 | 0.459 / 0.228 / 0.313, 0.9516 | 0.459 / 0.233 / 0.309, 0.9645 | 0.467 / 0.199 / 0.335 |
  | 2019-20 | 0.453 / 0.227 / 0.319, 0.9624 | 0.456 / 0.228 / 0.316, 0.9579 | 0.456 / 0.234 / 0.310 |
  | 2020-21 | 0.437 / 0.238 / 0.325, 0.9767 | 0.448 / 0.236 / 0.315, 0.9648 | 0.371 / 0.228 / 0.401 |
  | 2021-22 | 0.433 / 0.229 / 0.338, 0.9638 | 0.432 / 0.231 / 0.337, 0.9750 | 0.430 / 0.237 / 0.333 |

### Fitting statistics

| Target | Online fits | Retries | Converged | IRLS iterations | max \|coef\| | Online runtime (s) |
|---|---|---|---|---|---|---|
| 2017-18 | 98 | 0 | all | 5 | 0.996 | 5.0 |
| 2018-19 | 98 | 0 | all | 5 | 1.138 | 5.9 |
| 2019-20 | 110 | 0 | all | 5 | 1.070 | 8.2 |
| 2020-21 | 128 | 0 | all | 5 | 1.005 | 10.7 |
| 2021-22 | 119 | 0 | all | 5 | 1.096 | 11.4 |

553 fits in total, no retry, no failure. The whole stage took 53 s.

### 2024-25 diagnostic stage (diagnostic observation only)

Run once, after the historical lock (`d5c515c`), with `--stage validation` from a clean tree. H = 730 only; no other
half-life was fitted or scored. **2024-25 is diagnostic evidence only: it cannot be used to tune, select or confirm
anything**, and it is not pooled with the historical folds. This was the **fourth scored use of 2024-25** (after
Experiments 10, 11 and 12); the online arm was the only new prediction set.

Common group: 342 matches (380 minus the 38 involving Ipswich, which were neither scored nor fitted). Pre-scoring
checks passed:

- reproduction of the Experiment 11 and 12 predictions: largest difference 9.7e-17;
- online = frozen on the first target date: gap 0.0;
- registered group sizes and 109 online fits.

**Primary result** (online − frozen; negative = online better; 109 match dates as clusters):

| Arm | Log loss | Brier |
|---|---|---|
| Online TW Poisson | **1.0187** | **0.6118** |
| Frozen TW Poisson | 1.0417 | 0.6273 |

| Online − frozen | Mean | Naive SE | Date-clustered SE |
|---|---|---|---|
| Log loss | **−0.0230** | 0.0050 | 0.0046 |
| Brier | −0.0156 | 0.0035 | 0.0032 |

**Registered label: `same_sign_as_historical_distinguishable`.** The difference is more than 2 clustered SEs from
zero, with the same sign as the historical pooled −0.0117. As registered, this is a diagnostic observation. It does not
alter the historical reading and selects nothing.

**Segments** (online − frozen, clustered SE; Elo's updating gain for comparison):

| Segment | n | ΔLL (naive / clustered SE) | ΔBrier (clustered SE) | Mean summed \|ΔP\| vs frozen | Elo F1 − Online |
|---|---|---|---|---|---|
| 0–9 | 90 | −0.0042 (0.0028 / 0.0023) | −0.0028 (0.0016) | 0.018 | +0.0043 (0.0058) |
| 10–18 | 82 | −0.0221 (0.0064 / 0.0055) | −0.0137 (0.0039) | 0.046 | +0.0423 (0.0169) |
| 19–28 | 90 | −0.0380 (0.0127 / 0.0106) | −0.0265 (0.0070) | 0.084 | +0.0481 (0.0250) |
| 29+ | 80 | −0.0281 (0.0143 / 0.0134) | −0.0195 (0.0093) | 0.094 | +0.0375 (0.0292) |

In 2024-25 the gain is small in the first segment and larger once current-season information has accumulated. The
registered accumulation check applies to the pooled historical result only (where it was not met), so it is not
applied here. Descriptive.

**Home advantage.** IsHome is 0.200 frozen. The online trajectory (first fit, ¼, ½, ¾, last fit) is 0.200, 0.198,
0.175, 0.171, 0.175, with a range of 0.167–0.201. This is a mild in-season decline, nothing like 2020-21.

**Other diagnostics (descriptive):**

- **Current-season weight share:** 0, 0.089, 0.162, 0.222, 0.272 at the same points. Kish effective sample size rises
  from 2,069 to 2,086.
- **Returning teams** (Leicester, Southampton; 70 matches): −0.0696 (clustered SE 0.0144). Continuously present teams
  (272): −0.0110 (0.0046). Most of the gain comes from the returning teams.
- **Calibration and sharpness:**

  | | Mean P(H/D/A) | Entropy |
  |---|---|---|
  | Online | 0.441 / 0.228 / 0.332 | 0.9939 |
  | Frozen | 0.445 / 0.227 / 0.328 | 0.9903 |
  | Observed | 0.421 / 0.243 / 0.336 | — |

  The online forecasts are slightly closer to the observed home and away rates and slightly less sharp. The mean
  summed |ΔP| versus frozen is 0.060.
- **Fitting:** 109 refits (as registered), 0 retries, all converged in 5 IRLS iterations, max |coef| 1.141. Runtime:
  22.9 s for the online arm, 26.8 s for the stage.

**Context comparisons** (not effects; left − right, positive = right-hand arm better):

| Comparison | ΔLL (naive / clustered SE) | ΔBrier (clustered SE) |
|---|---|---|
| Online TW − online Elo | +0.0351 (0.0107 / 0.0110) | +0.0239 (0.0074) |
| Frozen TW − F2 | +0.0274 (0.0080 / 0.0076) | +0.0203 (0.0053) |
| Online TW − static Poisson | −0.0668 (0.0099 / 0.0098) | −0.0455 (0.0068) |
| Online TW − frequency baseline | −0.0573 (0.0192 / 0.0178) | −0.0402 (0.0130) |
| Frozen TW − frequency baseline | −0.0343 (0.0203 / 0.0187) | −0.0247 (0.0134) |

Other 2024-25 scores reproduce Experiments 10–12 (log loss / Brier):

| Model | Log loss / Brier |
|---|---|
| Online Elo | 0.9836 / 0.5879 |
| F1 | 1.0163 / 0.6082 |
| F2 | 1.0143 / 0.6070 |
| Static Poisson | 1.0854 / 0.6573 |
| Dixon-Coles | 1.0857 / 0.6572 |
| Frequency baseline | 1.0760 / 0.6520 |

- **Online Elo still has the lowest 2024-25 log loss.** This reverses the historical folds, where online TW was ahead
  of online Elo (−0.0131).
- **Updating gain within each family in 2024-25:** Poisson (frozen TW − online TW) +0.0230 (clustered SE 0.0046) vs
  Elo (F1 − Online) +0.0327 (0.0106). Historically the two were +0.0117 and +0.0099. Their difference is not a family
  effect.

**Decomposition connecting Experiments 11, 12 and 13** (2024-25 common group; per match):

| Component | ΔLL (naive / clustered SE) | ΔBrier (clustered SE) |
|---|---|---|
| Static Poisson − frozen TW (weighting, Experiment 12) | +0.0438 (0.0076 / 0.0077) | +0.0300 (0.0053) |
| Frozen TW − online TW (Poisson updating, Experiment 13) | +0.0230 (0.0050 / 0.0046) | +0.0156 (0.0032) |
| Online TW − online Elo (remainder) | +0.0351 (0.0107 / 0.0110) | +0.0239 (0.0074) |
| **Static Poisson − online Elo (total)** | **+0.1019** (0.0188 / 0.0194) | +0.0695 (0.0132) |

- **The identity holds per match.** The experiment's check found a largest residual of 0 (log loss) and 2.8e-17
  (Brier). An independent recomputation from the predictions file gives +0.0438 + 0.0230 + 0.0351 = +0.1019 = the
  total, with a largest per-match residual of 0.
- The total equals the Experiment 11 and 12 value.
- **How the 2024-25 gap divides:** weighting closes about 43% of the static Poisson vs online Elo gap, Poisson updating
  about 23%, and about 34% remains.
- **The cross-family remainder (+0.0351) is not attributed to any single cause.** It mixes model family (goals vs
  results) with the update mechanism (global refit vs per-team sequential updates), the recency form (days vs per
  match), the calibration layer, Elo's use of Ipswich results, and hyperparameters chosen for different arms
  (H = 730 for the frozen Poisson, K = 25 for online Elo).

### Interpretation (pre-registered reading)

- **Criterion U met → historical evidence that in-season refitting improves the 730-day time-weighted Poisson.** The
  gain is negative in every fold, in log loss and Brier, and 4.4 clustered SEs from zero pooled.
- **It is about the size of Elo's own updating gain** on the same folds.
- **Part of the gain is season-specific:** 2020-21 includes home-advantage re-learning during the crowd-free season.
- **The pre-registered accumulation pattern was not met,** because the gain is already distinguishable in the first
  segment. This is descriptive and does not change the reading.
- **This is diagnostic evidence about update policy within the Poisson family.** It does not establish that the online
  model is a better *specification* for deployment, and it says nothing about markets or profitability.
- **2024-25: `same_sign_as_historical_distinguishable`** (online − frozen −0.0230, clustered SE 0.0046).
  - It is a diagnostic observation consistent in sign with the historical result.
  - It is not a confirmation, and it cannot be used to tune, select or confirm anything.
  - Online Elo still scores best in 2024-25. The remaining online TW − online Elo gap is not attributed to a
    single cause.

### Limitations

- H = 730 and K = 25 were selected on these folds. The online arm uses an H chosen for the frozen arm.
- Date-clustered SEs do not capture serial correlation within a season (the online fits share one trajectory). The
  5-of-5 sign count is the more robust evidence.
- There are only five historical folds, and history length (3–7 seasons) is confounded with the fold.
- Unseen/promoted teams are still excluded from scoring and from the online fits.
- 2024-25 is a single season, now scored in four pre-registered experiments (10–13). Its result is a diagnostic
  observation only.

### Decision

- **Recorded as is.** The historical result is locked in `[historical_locked]`, and the 2024-25 result is recorded
  unchanged. No model, hyperparameter, protocol, eligibility rule or evidence criterion is changed because of
  either stage.
- **No specification is adopted.** `poisson_tw_online_h730_v1_diag` remains a diagnostic arm, not registered for
  2022-24 or holdout scoring.
- **2024-25 must not be used to tune, select or confirm anything**, including H (which stays 730 days).
- **Any adoption of an online Poisson specification** would need its own pre-registered evaluation on a season not
  used to generate the hypothesis.

### Provenance

| Item | Value |
|---|---|
| Pre-registration | commit `b91217e` (config and access-log entry V4) |
| Implementation | commit `4ad9782`; test-only CI fix `7faadfc` |
| Historical run | commit `7faadfc487c3d718122390a7dd23f332d09f2367`, `git_dirty: false` |
| Metrics | `results/online_tw_poisson_historical/metrics.json`, content SHA-256 `c0f2e7b1558a4b5e0e232c21f19bb6fd10b31f28bf772634eb0ab2e263fa4379` |
| Predictions | `results/online_tw_poisson_historical/predictions.csv`, 1,604 rows, git-ignored, content SHA-256 `c5cd256d1757d180ac7113bef391a87a990132a1c023b2b718ba14f515c587f9` |
| Lock | commit `d5c515c21fb228a1807a126fecc63aebf3344b00` |
| 2024-25 run | commit `d5c515c21fb228a1807a126fecc63aebf3344b00`, `git_dirty: false` |
| 2024-25 metrics | `results/online_tw_poisson_validation/metrics.json`, content SHA-256 `e2f325d8ac23a73cdbcfbf4c3ff7750072bd0f0ff35aea42d344ed7d0ce85b9a` |
| 2024-25 predictions | `results/online_tw_poisson_validation/predictions.csv`, 342 rows, git-ignored, content SHA-256 `4b9ad0ffc0a635872af21810e680139c066082e3645f4521ebdd4d350de29fc8` |
| Data | `data/processed/matches_dev_v2.csv`, SHA-256 `c726bd5cb30315bb18baf5805733059f4075b922cef1243d8533dbf7b39fd807` |
| Reproduction references | Experiment 11 predictions `73251375…e5f8`; Experiment 12 development predictions `96831b96…c3ba`; Experiment 12 validation predictions `3dcc7739…e37b` |
| Environment | Python 3.13.15; numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.0 |

Not yet locked by golden tests. The run is not in `experiments.run_all`.

## Experiment 14 — Market-implied benchmark from Pinnacle 1X2 odds (2026-10-02)

Protocol `market_benchmark_v1` (`configs/market_benchmark_v1.toml`). Script `experiments/market_benchmark.py`,
library `eplmodel.market`. **A benchmark, not a model:** nothing is fitted, tuned or selected, and no arm is ranked.

### Pre-registration and run sequence

1. Column meanings were checked against football-data.co.uk's own notes, not assumed. The source is recorded in
   `data/reference/football_data_odds_columns.md` (fetched 2026-10-02, SHA-256 `6ecd41a9…452d`). In the notes'
   words, the odds keys "are for pre-closing odds. For the closing odds, as below but with an additional "C"
   character…", and "PSH and PH = Pinnacle home win odds".
2. Outcome-free checks of the raw files followed: headers in all development seasons, plus values for 2014-15 …
   2021-22 (presence, validity, booksums, a 1:1 join to the processed matches, and the raw checksums).
3. The protocol was committed in `ac60693` **before any market probability was computed**.
4. The implementation was committed in `e2ecf00`.
5. The recorded run used commit `e2ecf00` (`git_dirty: false`).
6. One earlier run of the same code from the uncommitted tree was discarded. Its output was deleted unrecorded, it
   was not used for any decision, and it was identical to the recorded run.

### Objective

Provide a fixed market-implied reference against which future football models can be compared on the same matches.

### Data and evidence roles

| Seasons | Use |
|---|---|
| 2014-15 … 2021-22 | Odds values read for coverage and validity (outcome-free) |
| 2017-18 … 2021-22 | Scored: the historical selection targets (`SELECTION_TARGET_SEASONS`; `assert_selection_target`). Outcomes from dev_v2, cut to seasons ≤ 2021-22 straight after reading |
| 2014-15 | Burn-in for the comparison models; coverage only |
| 2022-23, 2023-24 (exposed), 2024-25 (retired, amendment A1) | **Not read for odds values and not scored.** The specification was fixed without any 2024-25 information |
| 2025-26, 2026-27 | Sealed; refused by the odds reader |

### Specification (fixed before scoring)

| Item | Value |
|---|---|
| Primary snapshot | Pinnacle **closing** `PSCH`, `PSCD`, `PSCA` |
| Secondary snapshot | Pinnacle **pre-closing** `PSH`, `PSD`, `PSA` (not "opening") |
| Primary margin removal | **Shin**: p_k = (√(z² + 4(1−z)π_k²/B) − z) / (2(1−z)), z ∈ [0, 1) solving Σp = 1 |
| Sensitivity methods | proportional p_k = π_k / B; power p_k = π_k^c with Σπ_k^c = 1 |
| Definitions | π_k = 1/odds_k, B = Σπ_k; brentq with xtol 1e-15, then division by the sum |
| Validity | All three prices present, numeric, finite and > 1.0; booksum in [1.0, 1.10]. Invalid price sets are counted by reason and excluded; no imputation, no fallback to the other snapshot or another bookmaker |
| Arms | 2 snapshots × 3 methods = 6, each reported separately; spec id of the primary arm `market_pinnacle_close_shin_v1` |
| Scoring | The common harness: forecast frames checked; log loss and Brier score in (H, D, A) order; calibration in the large with date-clustered SEs; paired (left − right) differences with naive and date-clustered SEs |

The probabilities use the odds only. The reader takes Date, HomeTeam, AwayTeam and the six odds columns, so no
result column is read.

### Coverage (both snapshots)

Every season from 2014-15 to 2021-22 has 380 matches with valid closing and pre-closing prices. There are no
missing, non-numeric, ≤ 1.0 or out-of-range price sets, so coverage is 100%.

| Season | Closing booksum (min / median / max) | Pre-closing booksum (min / median / max) |
|---|---|---|
| 2014-15 | 1.0141 / 1.0203 / 1.0266 | 1.0125 / 1.0202 / 1.0251 |
| 2015-16 | 1.0170 / 1.0203 / 1.0263 | 1.0172 / 1.0203 / 1.0238 |
| 2016-17 | 1.0168 / 1.0205 / 1.0299 | 1.0151 / 1.0203 / 1.0324 |
| 2017-18 | 1.0123 / 1.0206 / 1.0340 | 1.0166 / 1.0203 / 1.0454 |
| 2018-19 | 1.0176 / 1.0223 / 1.0380 | 1.0173 / 1.0223 / 1.0379 |
| 2019-20 | 1.0209 / 1.0289 / 1.0413 | 1.0210 / 1.0295 / 1.0453 |
| 2020-21 | 1.0172 / 1.0233 / 1.0388 | 1.0183 / 1.0236 / 1.0591 |
| 2021-22 | 1.0005 / 1.0238 / 1.0442 | 1.0205 / 1.0240 / 1.0470 |

Median Shin z per season is 0.010–0.015 for both snapshots.

### Results by season (log loss / Brier; 380 matches each, 1,900 pooled)

| Arm | 2017-18 | 2018-19 | 2019-20 | 2020-21 | 2021-22 | Pooled |
|---|---|---|---|---|---|---|
| **close, Shin (primary)** | 0.9406 / 0.5570 | 0.8897 / 0.5202 | 0.9731 / 0.5746 | 0.9979 / 0.5924 | 0.9363 / 0.5541 | **0.9475 / 0.5597** |
| close, proportional | 0.9405 / 0.5569 | 0.8907 / 0.5208 | 0.9722 / 0.5744 | 0.9971 / 0.5921 | 0.9367 / 0.5543 | 0.9474 / 0.5597 |
| close, power | 0.9409 / 0.5571 | 0.8890 / 0.5200 | 0.9733 / 0.5747 | 0.9983 / 0.5926 | 0.9360 / 0.5540 | 0.9475 / 0.5597 |
| pre-closing, Shin | 0.9447 / 0.5595 | 0.8913 / 0.5211 | 0.9732 / 0.5749 | 1.0071 / 0.5981 | 0.9357 / 0.5536 | 0.9504 / 0.5614 |
| pre-closing, proportional | 0.9446 / 0.5594 | 0.8925 / 0.5218 | 0.9722 / 0.5747 | 1.0059 / 0.5978 | 0.9363 / 0.5538 | 0.9503 / 0.5615 |
| pre-closing, power | 0.9450 / 0.5596 | 0.8906 / 0.5208 | 0.9733 / 0.5749 | 1.0075 / 0.5983 | 0.9356 / 0.5535 | 0.9504 / 0.5614 |

Calibration in the large of the primary arm, pooled (observed − mean predicted, date-clustered SE): H +0.0007
(0.0103), D −0.0127 (0.0097), A +0.0120 (0.0097). Mean entropy is 0.9457 nats.

### Paired differences (pooled over 1,900 matches; left − right; descriptive)

| Difference | Log loss (clustered SE) | Brier |
|---|---|---|
| close proportional − close Shin | −0.00009 (0.00033) | +0.00003 |
| close power − close Shin | −0.00001 (0.00017) | −0.00002 |
| pre proportional − pre Shin | −0.00013 (0.00033) | +0.00002 |
| pre power − pre Shin | −0.00002 (0.00017) | −0.00001 |
| pre Shin − close Shin | +0.00289 (0.00167) | +0.00175 |

The three margin-removal methods differ by at most about 0.001 log loss in any season, with signs that change from
season to season. The pre-closing snapshot scores worse than the closing one in 3 of 5 seasons, ties in 2019-20 and
scores better in 2021-22. The pooled difference is 1.7 clustered SEs.

### Context: primary benchmark vs recorded model predictions (same matches; descriptive, not selection evidence)

Recorded, checksum-verified predictions of Experiments 10 (full group) and 13 (historical stage, common group). No
model is refitted.

| Comparison | Matches | Log loss (clustered SE) | Brier | Per season (log loss) |
|---|---|---|---|---|
| market − online Elo (full) | 1,900 | −0.0242 (0.0051) | −0.0148 | −0.0260, −0.0213, −0.0054, −0.0501, −0.0181 |
| market − frequency baseline (full) | 1,900 | −0.1197 (0.0101) | −0.0863 | all negative |
| market − static Poisson (common) | 1,604 | −0.0296 (0.0069) | −0.0205 | all negative |
| market − time-weighted Poisson (common) | 1,604 | −0.0238 (0.0066) | −0.0166 | all negative |
| market − online time-weighted Poisson (common) | 1,604 | −0.0121 (0.0051) | −0.0088 | 4 of 5 negative (2019-20 +0.0016) |
| market − online Elo (common) | 1,604 | −0.0252 (0.0055) | −0.0154 | all negative |

### Interpretation

- **The benchmark is established.** Pinnacle closing odds with Shin margin removal score 0.9475 log loss and 0.5597
  Brier on the 1,900 historical-fold matches, with full coverage.
- **The margin-removal choice barely matters for these scores.** Pinnacle's booksum is only about 2–3%, so the methods
  differ by about 1e-4 pooled. This does not make the choice irrelevant for edge or CLV calculations on longshots,
  where the methods differ most.
- **Closing vs pre-closing.** The closing snapshot is lower in pooled loss, which is consistent with later prices
  carrying more information. The difference is not distinguishable at 2 SEs and is not uniform across seasons.
- **On the same matches, every established arm scores worse than the closing market** (METHODOLOGY §7, claim 4).
  For example, the market is 0.0242 log loss below online Elo and 0.0121 below the online time-weighted Poisson
  diagnostic arm.
- **These are context, not evidence for any model choice.** The models use only past results, while the closing
  price reflects all information up to kickoff. The model arms were also tuned or selected on these same folds.

### Limitations

- The closing time is not documented. The per-match collection time of the pre-closing prices (and so the
  information they contain) is not established; no information cutoff is inferred.
- One bookmaker. A single sharp book is not the whole market, and other books are deliberately not used.
- Five seasons; the date-clustered SEs ignore any serial correlation within a season.
- The source notes are football-data's current version; the historical files may have been compiled under earlier
  wording.

### Decision

- **Fixed as the reference benchmark** for future comparisons on the historical folds: primary arm
  `market_pinnacle_close_shin_v1`. The other five arms remain registered sensitivity analyses.
- **Nothing is selected or changed** because of these results.
- **Not registered** for 2022-24, 2024-25 or holdout scoring. Any such use needs its own logged registration.
- **CLV is not implemented.** It needs a separately defined pre-match snapshot and an explicit per-prediction
  information cutoff, which the current data do not establish.

### Provenance

| Item | Value |
|---|---|
| Pre-registration | commit `ac6069394aa56ad0309dca19336c03c1caed9537` |
| Implementation and run | commit `e2ecf00704dca4f1e8435aadbb0b3872ef07a53a`, `git_dirty: false` |
| Metrics | `results/market_benchmark/metrics.json`, content SHA-256 `55a77b4f57a0f55a8a6daf68aabba12bc552972bda98e837ea05ccac6eebcf56` |
| Predictions | `results/market_benchmark/predictions.csv`, 1,900 rows (all six arms), git-ignored, content SHA-256 `79fb6b34dc5251f114e8ac9d7a01145f3c65135b1fc8570417fa7bcc39efc9c8` |
| Odds | raw `E0_1415.csv` … `E0_2122.csv`, each matching `data/checksums.json` |
| Outcomes | `data/processed/matches_dev_v2.csv`, SHA-256 `c726bd5cb30315bb18baf5805733059f4075b922cef1243d8533dbf7b39fd807`, cut to ≤ 2021-22 |
| Environment | Python 3.13.15; numpy 2.5.2, pandas 3.0.5, scipy 1.18.1 |

Reproduced by `tests/test_market.py::test_recorded_market_benchmark_reproduces` (golden). Not in `experiments.run_all`.

## Experiment 15 — Full-coverage dynamic Poisson with empirical-Bayes season-start priors (2026-10-02)

Protocol `full_coverage_poisson_v1`. Pre-registration: `docs/preregistration/full_coverage_poisson_v1.md` and
`configs/full_coverage_poisson_v1.toml`. Code: `eplmodel.models.full_coverage`, `experiments/full_coverage_poisson.py`.
Results: `results/full_coverage_poisson_historical/metrics.json`. Historical stage only; locked in `[historical_locked]`.

### Pre-registration and run sequence

| Step | Commit | What |
|---|---|---|
| Pre-registration | `5034f56` | Design, config, typed spec; nothing implemented |
| Amendment PA1 | `10e99d1` | Identifiability parameterisation and rationale of the inherited 0.002 margin; no frozen value changed |
| Implementation | `f555637` | Model, experiment, the pre-registered test plan; no real-data prediction |
| First run | at `f555637` | Stopped while loading the Experiment 14 comparator (its file has no `fold_target` column). This was after stage 1 (all checks and forecasts) and **before any outcome was joined**; nothing was scored or written |
| Fix | `d609b95` | The comparator is loaded by season; nothing else changed |
| **Recorded run** | at `d609b95`, `git_dirty: false` | Stage 1 checks, then scoring; metrics reproduce exactly on a rerun |

Two notes from the implementation, recorded in `f555637`:

- **A slip in the test plan.** The pre-registered test plan said "doubling every weight equals halving tau²". The
  registered objective implies the opposite: doubling the weights is equivalent to **doubling** tau² (halving the prior
  precision). The test checks that identity. The model is unaffected.
- **Numerical conventions, not model choices.** Damped-Newton steps are accepted within a 64·ε·(1 + |f|) round-off
  allowance, and M1's unpenalised continuing teams start at 0. The optimum is unique, and the registered gradient
  tolerance (1e-9) certifies every fit.

### Objective

Forecast every match of each target season, including promoted, returning and never-seen teams, with the online
time-weighted Poisson model (H = 730, Experiments 12-13). Empirical-Bayes season-start priors make this possible. The
question is whether this degrades the matches the Experiment 13 model could already score.

### Data and evidence roles

- **Targets.** 2017-18 … 2021-22 only (amendment A1), via `selection_folds()` and `build_fold`. dev_v2 is cut to
  seasons ≤ 2021-22 straight after reading.
- **No other season.** Nothing from 2022-23 onward is read; no 2024-25, 2025-26 or 2026-27 information is used.
- **No other inputs.** No market, shots or squad input.
- **Nothing selected.** No value was tuned, and M2 was fixed as primary in advance.

### Checks before scoring (all passed, all five folds)

- The config equals the pre-registration, apart from the later `[historical_locked]`, and the document is unchanged.
- The registered group sizes and team lists hold: promoted teams, recent yo-yo teams, full / common / unseen /
  promoted / returning / continuing-only.
- The EB team-season counts hold: 6/34 … 18/102.
- **M0 anchor.** The implementation without priors and with unseen-team matches excluded reproduces the recorded
  Experiment 13 online predictions to at most **7.0e-14** (registered tolerance 1e-7).
- **Coverage.** M1, M2 and S1 each give 380 valid forecasts per target, with exactly the registered 105 / 108 / 115 /
  135 / 123 refits.
- **Convergence.** Every fit converged: at most 6 Newton iterations, max |gradient| ≤ 9.9e-10 (M1) and ≤ 1.9e-13 (M2,
  S1). **No fit failed or aborted.**

### Empirical-Bayes priors (per target, from seasons 2015-16 … S − 1 only)

| Target | Promoted / continuing team-seasons | a_P (attack) | b_P (defence) | tau_att | tau_def | Floor binds | corr(att, def) |
|---|---|---|---|---|---|---|---|
| 2017-18 | 6 / 34 | −0.280 | +0.178 | 0.237 | 0.211 | no | −0.62 |
| 2018-19 | 9 / 51 | −0.303 | +0.136 | 0.260 | 0.212 | no | −0.61 |
| 2019-20 | 12 / 68 | −0.298 | +0.154 | 0.262 | 0.231 | no | −0.65 |
| 2020-21 | 15 / 85 | −0.306 | +0.148 | 0.260 | 0.219 | no | −0.65 |
| 2021-22 | 18 / 102 | −0.289 | +0.152 | 0.269 | 0.206 | no | −0.63 |

Promoted teams start with about 25% fewer goals scored (e^−0.29) and 16% more conceded than an average team.

### Coverage and scores (log loss / Brier, pooled over the five targets)

| Group (matches) | M1 `fc_promoted` | **M2 `fc_hier`** | S1 `fc_hier_break` | M0 (Exp 13) | online Elo | frequency baseline | market (Exp 14) |
|---|---|---|---|---|---|---|---|
| full (1,900) | 0.9592 / 0.5680 | **0.9622 / 0.5701** | 0.9631 / 0.5705 | — | 0.9717 / 0.5745 | 1.0672 / 0.6460 | 0.9475 / 0.5597 |
| common (1,604) | 0.9573 / 0.5667 | 0.9609 / 0.5692 | 0.9618 / 0.5695 | 0.9572 / 0.5667 | 0.9703 / 0.5734 | 1.0698 / 0.6479 | 0.9451 / 0.5580 |
| unseen (296) | 0.9692 | 0.9695 | 0.9699 | — | 0.9793 | 1.0527 | 0.9608 |
| promoted (540) | 0.9419 | 0.9420 | 0.9450 | — | 0.9550 | 1.0627 | 0.9366 |
| returning (260) | 0.9070 | 0.9066 | 0.9130 | — | 0.9207 | 1.0707 | 0.9071 |
| continuing-only (1,360) | 0.9660 | 0.9703 | 0.9703 | — | 0.9783 | 1.0690 | 0.9519 |
| recent yo-yo (186) | 0.9215 | 0.9214 | 0.9276 | — | 0.9368 | 1.0860 | 0.9269 |
| long absence or newcomer (366) | 0.9507 | 0.9507 | 0.9521 | — | 0.9619 | 1.0487 | 0.9416 |

Coverage: M2 gives valid forecasts for **1,900 / 1,900** matches (380 per target). M2's full-group log loss by target:
0.9658, 0.9052, 0.9718, 1.0178, 0.9505.

### Primary criterion: common-group non-inferiority, M2 − M0

| Pooled (1,604 matches) | Value |
|---|---|
| Mean log-loss difference | **+0.0037** (date-clustered SE 0.0014); Brier +0.0024 |
| Upper bound (mean + 2 SE) | +0.0065, **not** below the 0.002 margin |
| Lower bound (mean − 2 SE) | +0.0009, **not** above the margin |
| Per target | +0.0051, +0.0150, +0.0008, −0.0033, +0.0035 |
| **Registered reading** | `inconclusive_on_common_group` → **`full_coverage_inconclusive`** |

### Key secondary and decomposition (pooled; left − right; descriptive except the U readings)

| Comparison (group, matches) | Log loss (clustered SE) | Brier | Per target | Reading |
|---|---|---|---|---|
| M2 − Elo (unseen, 296) | −0.0097 (0.0107) | −0.0049 | −0.0104, −0.0211, +0.0309, −0.0139, −0.0126 | U: **not distinguishable** (4 of 5 folds negative) |
| M2 − baseline (unseen, 296) | −0.0832 (0.0226) | −0.0604 | all negative | U: **helps** |
| M1 − M0 (common, 1,604) | +0.0001 (0.0007) | +0.0000 | mixed, all within ±0.0014 | — |
| M2 − M1 (full, 1,900) | +0.0031 (0.0011) | +0.0021 | +0.0049, +0.0097, −0.0000, −0.0020, +0.0028 | — |
| M2 − M1 (common, 1,604) | +0.0036 (0.0013) | +0.0024 | | — |
| M2 − M1 (promoted, 540) | +0.0001 (0.0017) | +0.0002 | | — |
| M2 − M1 (continuing-only, 1,360) | +0.0043 (0.0014) | +0.0028 | | — |
| S1 − M2 (returning, 260) | +0.0064 (0.0054) | +0.0028 | +0.0019, —, +0.0069, +0.0055, +0.0091 | — |
| M2 − Elo (full, 1,900; context) | −0.0095 (0.0045) | −0.0043 | all negative | — |
| M2 − market (full, 1,900; context) | +0.0147 (0.0047) | +0.0104 | 4 of 5 positive | — |

Segments of the primary difference (M2 − M0, common): +0.0049 (0-9, n 422), +0.0048 (10-18), +0.0003 (19-28),
+0.0051 (29+). Each has a clustered SE of about 0.003.

Calibration in the large of M2 (full group, observed − predicted): H −0.0073, D −0.0088, A +0.0162. Each has a
clustered SE of about 0.01.

### Interpretation (pre-registered reading)

- **Full coverage works.** Every match of every target season gets a valid forecast, including the 296 unseen-team
  matches that Experiment 13 could not score. No fit failed.
- **The primary criterion is inconclusive, and the point estimate is against M2.** On the matches M0 can score, M2 is
  +0.0037 worse. The 2-SE interval [+0.0009, +0.0065] excludes zero but straddles the 0.002 margin, so neither
  non-inferiority nor inferiority is established.
- **The decomposition locates the cost.** Admitting the promoted teams' matches and priors leaves the common group
  unchanged (M1 − M0 = +0.0001). The cost comes entirely from the continuing-team prior that M2 adds (M2 − M1 on
  continuing-only matches +0.0043, 3.1 SEs), while M2 and M1 are equal on promoted-team matches.
- **On the unseen-team matches**, M2 beats the frequency baseline clearly. It is ahead of online Elo by 0.0097, which
  is not distinguishable (one SE).
- **The identity break (S1) is not better** for returning teams: +0.0064, not distinguishable. This matches the primary
  rule of keeping their decayed history.
- **Context only.** Every arm remains well behind the closing market (+0.0147).

### Limitations

- **One promoted-prior form.** The EB reference sets are small early on (6 promoted team-seasons for 2017-18).
- **The continuing-team prior's effect is a historical observation.** It is measured on the same five folds that
  produced it. It does not license replacing the primary arm.
- **Clustered SEs.** These ignore serial correlation within a season.

### Decision

- **Recorded as is and locked** (`[historical_locked]`: reading `full_coverage_inconclusive`).
- **M2 stays the registered primary arm of this protocol.** M1's descriptively better common-group score must not be
  used to swap the primary arm after the fact. Dropping or changing the continuing-team prior would be a new
  specification, needing its own pre-registration and evidence from a season not used here.
- **No 2024-25 stage was run.** A descriptive 2024-25 stage would need its own access-log entry first. 2025-26 remains
  sealed.

### Provenance

| Item | Value |
|---|---|
| Run | commit `d609b952ff6dc4dc3a63717f54ab7bd4fa9a7bfe`, `git_dirty: false` |
| Metrics | `results/full_coverage_poisson_historical/metrics.json`, content SHA-256 `c50267ce7ab911f36703e0291ab772a9265fb9ba89197ae10883deeedc82a3ca` |
| Predictions | `results/full_coverage_poisson_historical/predictions.csv`, 1,900 rows, git-ignored, content SHA-256 `69963f996c08f5c88353247ce715c9a2a2e829c12bb6c937be1e04b9449f5600` |
| Data | `data/processed/matches_dev_v2.csv`, SHA-256 `c726bd5cb30315bb18baf5805733059f4075b922cef1243d8533dbf7b39fd807`, cut to ≤ 2021-22 |
| Comparators | recorded predictions of Experiments 13 (historical), 10 and 14, checksum-verified |
| Environment | Python 3.13.15; numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.0 |

Reproduced by `tests/test_full_coverage.py::test_recorded_experiment_15_reproduces` (golden). Not in `experiments.run_all`.

## Descriptive validation V5 — frozen Experiments 14 and 15 on 2024-25 (2026-10-02)

Protocol `exposed_validation_descriptive_v1` (`configs/exposed_validation_descriptive_v1.toml`). Access-log entry
**V5**, registered before any 2024-25 data was read for it. Script `experiments/exposed_validation_descriptive.py`.
Results `results/exposed_validation_descriptive/metrics.json`, locked in `[locked]`.

**Evidence class: exposed and descriptive only.** 2024-25 has now been scored five times (V1-V5). These results cannot
select, tune or confirm anything, and no criterion is applied. In particular, the Experiment 15 non-inferiority rule is
not applied to this season. No parameter, prior, hyperparameter, arm, market specification, scoring rule or evidence
criterion was changed, before or after.

### Run sequence

| Step | Commit | What |
|---|---|---|
| Registration | `4ad86fe` | Entry V5. The logged-access registry (`LOGGED_EXPOSED_VALIDATION_ACCESSES`), separate from the unchanged V1-V4 reproduction set. `build_exposed_validation_fold`, which opens the season only for a logged entry and exactly its specs. The stage code |
| First run | at `4ad86fe` | **Aborted at a pre-scoring check, before any outcome was joined.** The registered EB team-season count [24, 136] was an arithmetic slip (8 seasons instead of 9); nothing was written |
| Correction | `9aa1ad8` | [27, 153], verified from the fixture lists only and recorded in the V5 entry |
| **Recorded run** | at `9aa1ad8`, `git_dirty: false` | Every pre-scoring check passed; metrics reproduce exactly on a rerun |

### What was run (all frozen, unchanged)

- **Experiment 15:**
  - M1 `fc_promoted`, M2 `fc_hier` (primary) and S1 `fc_hier_break`, using the same code path as the locked historical
    stage. History is 2014-15 … 2023-24, with 2022-24 as history only.
  - EB priors from 2015-16 … 2023-24 (27 promoted and 153 continuing team-seasons): a_P −0.293, b_P +0.198,
    τ_att 0.261, τ_def 0.206. The floor does not bind.
  - 109 refits per arm, all converged (≤ 6 iterations, max |gradient| ≤ 6.1e-13).
  - The M0 anchor reproduces the recorded Experiment 13 2024-25 predictions to **2.3e-15**.
- **M0** is the recorded Experiment 13 online arm. It **cannot score Ipswich's 38 matches** under its registered
  definition, so it covers 342 of 380.
- **Experiment 14:** all six market arms, with closing odds + Shin as primary. The validity rules are unchanged. Odds
  are valid for 380 of 380 matches in both snapshots. Book sums: closing 1.0063 / 1.0294 / 1.0423, pre-closing
  1.0272 / 1.0357 / 1.0534 (min / median / max).
- **Promoted teams:**
  - Ipswich is unseen in the history. Its Experiment 7 category is **unclassified**, because `team_history.csv` has no
    Ipswich row and none was invented.
  - Leicester and Southampton are returning teams, one season out, so they are recent yo-yo teams.

### Coverage and scores (log loss / Brier)

| Arm | full (380) | common (342) | unseen: Ipswich (38) | returning / recent yo-yo (74) | continuing-only (272) |
|---|---|---|---|---|---|
| M1 `fc_promoted` | 1.0088 / 0.6048 | 1.0169 | 0.9359 | 0.8220 | 1.0714 |
| **M2 `fc_hier`** | **1.0037 / 0.6010** | 1.0114 | 0.9342 | 0.8172 | 1.0657 |
| S1 `fc_hier_break` | 0.9888 / 0.5912 | 0.9938 | 0.9434 | 0.7406 | 1.0662 |
| M0 (Exp 13, recorded) | — (342 only) | 1.0187 / 0.6118 | not scorable | 0.8105 (70) | 1.0722 |
| market close, Shin (primary) | 0.9666 / 0.5753 | 0.9651 | 0.9794 | 0.7338 | 1.0318 |
| market close, proportional / power | 0.9664 / 0.9667 | | | | |
| market pre-closing, Shin / proportional / power | 0.9705 / 0.9703 / 0.9709 | | | | |
| online Elo (Exp 10, recorded) | 0.9848 / 0.5887 | 0.9836 | 0.9957 | 0.7262 | 1.0594 |
| frequency baseline | 1.0812 / 0.6558 | 1.0760 | 1.1281 | 1.0514 | 1.0847 |

### Paired differences (left − right; date-clustered SE; descriptive, no reading)

| Comparison (group, matches) | 2024-25 | Historical Experiment 15 (2017-22, locked) |
|---|---|---|
| M2 − M0 (common, 342) | **−0.0073** (0.0021) | +0.0037 (0.0014) |
| M1 − M0 (common) | −0.0017 (0.0009) | +0.0001 (0.0007) |
| M2 − M1 (full, 380) | −0.0052 (0.0018) | +0.0031 (0.0011) |
| M2 − M1 (continuing-only, 272) | −0.0057 (0.0024) | +0.0043 (0.0014) |
| M2 − M1 (promoted, 108) | −0.0039 (0.0024) | +0.0001 (0.0017) |
| S1 − M2 (returning, 74) | **−0.0766** (0.0207) | +0.0064 (0.0054) |
| M2 − Elo (unseen, 38) | −0.0615 (0.0279) | −0.0097 (0.0107) |
| M2 − baseline (unseen) | −0.1939 (0.0679) | −0.0832 (0.0226) |
| M2 − Elo (full, 380; context) | +0.0189 (0.0111) | −0.0095 (0.0045) |
| M2 − market (full; context) | +0.0371 (0.0121) | +0.0147 (0.0047) |

Market (Experiment 14 registered comparisons, full 380):

- proportional − Shin, closing: −0.0001 (0.0007); power − Shin, closing: +0.0002 (0.0003). The pre-closing
  equivalents are similar.
- pre-closing Shin − closing Shin: +0.0040 (0.0039).
- Context: market − Elo −0.0182 (0.0082); market − online time-weighted Poisson (common) −0.0535 (0.0128).

Segments of M2 − M0 (common): −0.0035, −0.0152, −0.0083, −0.0023 (0-9, 10-18, 19-28, 29+).

### Comparison with the historical findings (descriptive)

- **The market benchmark behaves as it did historically.** The margin-removal methods differ by about 1e-4, closing is
  at least as good as pre-closing, and the market leads every model arm.
- **The Experiment 15 within-family differences reverse sign in 2024-25.**
  - M2 is better than M0 and M1 here, whereas historically it was worse on the common group.
  - The identity break (S1) is much better on Leicester's and Southampton's matches (−0.0766), whereas historically it
    was slightly worse and not distinguishable.
  - Experiment 13 had already recorded that 2024-25's online gain was concentrated in these two returning teams.
- **What the reversals mean.** They are features of one exposed season, which had already been used four times, and
  they are not confirmation of anything. Under the pre-registration they **cannot** be used to choose M1 or M2, to
  change the continuing-team prior, or to change the returning-team rule.
- **Online Elo stays ahead of M2 on the full group in 2024-25** (+0.0189). The historical folds showed the opposite
  sign. This too is descriptive only.

### Decision

- **Recorded and locked as is.** Nothing was selected, tuned or changed.
- **The historical Experiment 14 and 15 results are unchanged and remain the evidence of record.**
- **Confirmation** of any of these specifications is possible only on the sealed 2025-26 holdout, if they are
  pre-registered there, or on a later season.

### Provenance

| Item | Value |
|---|---|
| Run | commit `9aa1ad8781d8ddbc21c6bd5b755dbe48b300cca1`, `git_dirty: false` |
| Metrics | `results/exposed_validation_descriptive/metrics.json`, content SHA-256 `b036f5bf70f1db7a975426131eb3ee831ffae6058629922207e8d951b2c2a267` |
| Predictions | `results/exposed_validation_descriptive/predictions.csv`, 380 rows, git-ignored, content SHA-256 `2a025f3e696bf7c1a6a0b0bd38bbd2238d70742e01d0289db8e6edf379569b31` |
| Data | `data/processed/matches_dev_v2.csv`, SHA-256 `c726bd5cb30315bb18baf5805733059f4075b922cef1243d8533dbf7b39fd807`; raw `E0_2425.csv` matching `data/checksums.json` |
| Frozen inputs | Experiment 15 config at `10e99d1` + lock (metrics `c50267ce…a3ca`); Experiment 14 metrics `55a77b4f…cf56` |

Reproduced by `tests/test_exposed_validation_descriptive.py::test_recorded_descriptive_validation_reproduces` (golden).

## Future experiment template

```
## Experiment N — <name>

### Objective
### Data            (training / validation folds; state explicitly that the development test is not used)
### Model           (specification and spec id)
### Parameters
### Results         (link results/<experiment>/metrics.json)
### Comparison      (baselines scored on the same matches under the same information)
### Interpretation  (which of the claims in METHODOLOGY.md §7 does this support?)
### Limitations
### Decision
```
