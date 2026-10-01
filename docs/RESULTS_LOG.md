# EPL Betting Model — Results Log

This file records important experiments and decisions so that results do not exist only inside individual chat
conversations. Every model result below is reproduced by `python -m experiments.run_all` (written to
`results/<experiment>/metrics.json`) and locked by `tests/test_golden_results.py`. Data-acquisition records are not
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
