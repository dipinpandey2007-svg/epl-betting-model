# EPL Betting Model — Results Log

This file records important experiments and decisions so that results do not exist only inside individual chat
conversations. Every model result in Experiments 2–7 is reproduced by `python -m experiments.run_all` (written to
`results/<experiment>/metrics.json`) and locked by `tests/test_golden_results.py`. Experiment 10 is reproduced by
`python -m experiments.validation_2425`, and Experiment 11 by `python -m experiments.update_policy_diagnostic`.
Neither is in `run_all` or locked by golden tests yet. Data-acquisition records are not
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
