# EPL Betting Model — Methodology Rules

## 1. Leakage rule

For every prediction, use only information that would have been available before kickoff.

Never use future information, including:

- future match results
- future team ratings
- future xG or performance
- post-match statistics
- future transfers
- future injuries or lineup information
- future market movements when constructing an earlier prediction
- final-test outcomes for tuning model parameters

If a feature would not have been known at the prediction timestamp, it cannot be used.

## 2. Chronological data discipline

Football data are time-dependent.

Training, validation and test periods should respect chronology.

Never randomly shuffle historical matches when doing something that could allow information from the future to influence the past.

## 3. Training vs validation vs test

Always distinguish:

- Training data: used to fit model parameters.
- Validation data: used to select/tune modeling choices.
- Final test data: held out until the model specification is frozen.

Do not tune the final test set.

A model that fits training data better is not automatically a better predictive model.

### Current status of the test periods (2026-10-01)

- 2022-23 and 2023-24 are an **exposed development test benchmark**. They were scored repeatedly during
  exploratory work (`TEST_SET_ACCESS_LOG.md`) and must not be used to tune, select or compare new methodological
  choices. They may be re-scored only for the registered, frozen specifications; the code enforces this
  (`eplmodel.splits.require_registered_dev_test_spec`).
- 2022-24 may be used as historical fitting information for later seasons, but never as a validation or
  model-selection target (`eplmodel.splits.assert_valid_selection_target`).
- The rule for new selection work is `eplmodel.splits.selection_folds()`: the training-season folds validating
  2017-18 to 2021-22 only (`SELECTION_TARGET_SEASONS`), with the strict guard `eplmodel.splits.assert_selection_target`.
- **2024-25 is an exposed validation season** (amendment A1, [HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md) §9, access-log
  entry P1, 2026-10-01). It was scored in four pre-registered protocols (Experiments 10-13). It may be used as
  historical fitting information for later seasons, but never to select, tune or confirm anything. It may be
  re-scored only to reproduce those protocols (`eplmodel.splits.require_registered_exposed_validation_spec`). A new,
  descriptive-only access needs an access-log entry first, a matching registry entry
  (`eplmodel.splits.LOGGED_EXPOSED_VALIDATION_ACCESSES`), and opening through
  `eplmodel.evaluation.folds.build_exposed_validation_fold`. The first such access was V5 (Experiments 14-15). Before
  A1 it was a selection target alongside the training folds; the recorded protocols keep that legacy definition
  (`SELECTION_VALIDATION_SEASONS`) so they reproduce unchanged.
- The final holdout is 2025-26, sealed by [HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md) before any of its data were
  acquired; 2026-27 is the declared next holdout. Every data path refuses holdout seasons
  (`eplmodel.splits.assert_not_holdout`); only `eplmodel.holdout` can open it.
- Every access to a test period is recorded in `TEST_SET_ACCESS_LOG.md`.

## 4. Walk-forward evaluation

Use chronological walk-forward validation when a parameter or model choice affects sequential prediction or evolving team strength.

For example, Elo K was selected using chronological validation.

For static distributional parameters, direct training-only maximum likelihood can be appropriate, but any such choice must be justified rather than assumed.

## 5. Probabilistic evaluation

The project is primarily about probability quality, not just picking winners.

Important metrics:

- Log loss
- Brier score
- Calibration

Later, when market data are incorporated:

- Market-implied probability comparison
- Closing-line value (CLV)
- Long-run simulated betting performance / ROI

Accuracy should never be treated as the sole success criterion.

### Market benchmark (protocol `market_benchmark_v1`, Experiment 14)

- **Source and meaning.** Pinnacle 1X2 odds from the football-data.co.uk files. Per the site's notes
  (`data/reference/football_data_odds_columns.md`), `PSH/PSD/PSA` are pre-closing and `PSCH/PSCD/PSCA` closing odds.
- **Primary benchmark:** closing odds, Shin margin removal (`market_pinnacle_close_shin_v1`). Secondary snapshot:
  pre-closing. Sensitivity methods: proportional and power. All six arms are fixed in advance and reported separately;
  none is chosen by score, the two snapshots are never mixed, and invalid or missing prices are excluded, never
  imputed.
- **Not a model and not a selection candidate.** Comparisons with it are context (claim 4 in §7), never evidence for
  a model choice. Scored only on the historical folds 2017-18 … 2021-22.
- **Odds are not features.** Closing odds are known only at kickoff, and the pre-closing collection time is not
  documented per match. Using either as a model input, or computing CLV, needs a separately defined snapshot and an
  explicit information cutoff for every prediction; neither exists yet.

## 6. Baselines

Every more-complex model should be compared against appropriate simpler baselines.

Examples:

- Naive historical outcome-rate baseline
- Elo
- Independent Poisson
- Dixon-Coles
- Later ML models
- Market-implied probabilities

A complex model should earn its complexity by demonstrating meaningful out-of-sample improvement.

## 7. Interpretation discipline

Separate these claims:

1. The model fits the training data better.
2. The model predicts held-out matches better.
3. The model is better calibrated.
4. The model is better than the market.
5. The model produces profitable betting performance.

These are different claims and must not be conflated.

## 8. Research skepticism

For every apparent improvement, ask:

- Is it out of sample?
- Was the test set touched?
- Could the improvement come from tuning noise?
- Is the effect economically meaningful?
- Is the improvement stable across time?
- Does it survive comparison with strong baselines?
- Is there a simpler explanation?

Do not chase small improvements merely because a metric moved in the desired direction.

## 9. Reproducibility

Record:

- data period
- features used
- model specification
- hyperparameters
- train/validation/test definitions
- metrics
- important random seeds where relevant
- known limitations

Important numerical results should be written to the project results/state files rather than existing only in chat messages.

How this is implemented:

- Frozen model specifications live in `configs/` with a spec id. Changing a value there is a methodological change.
- Each experiment is a script in `experiments/` that writes `results/<experiment>/metrics.json`. That file also
  records the git commit, a dirty-tree flag, the content checksum of the data, and the Python and package versions.
- `data/checksums.json` identifies the exact dataset behind the recorded results.
- `tests/test_golden_results.py` reproduces every established result within stated tolerances. A deliberate
  methodological change must update the golden values *and* the results log together, with the reason.

### Common evaluation harness (2026-10-02)

Every new experiment is evaluated through one layer in `eplmodel.evaluation`. Model-specific prediction code stays
outside it; the harness receives standard forecast frames and evaluates them the same way every time.

| Module | Role |
|---|---|
| `folds` | `build_fold()` for new selection targets (strict guard of amendment A1; later seasons never enter); frozen inputs (history only) vs online inputs (history plus target matches dated *strictly* before the match date); the common/unseen split from fixtures; online-refit accounting |
| `forecasts` | the forecast frame: indexed by unique `match_id`, columns `<arm>_H`, `<arm>_D`, `<arm>_A`, rows summing to 1, no result columns; outcomes are joined by `match_id` only when scoring |
| `scoring` | per-match log loss and Brier (via `metrics`, never sklearn's A/D/H ordering), paired (left − right) differences with naive and date-clustered SEs, decomposition residuals, calibration in the large per outcome with SEs; `EvaluationSet` bundles a frame with its outcomes and clusters |
| `segments` | the within-season segments registered in Experiment 11 (unchanged) |
| `reproduction` | the recorded prediction files of Experiments 10-13 (checksums from their committed metrics.json) and the row-by-row comparison at the registered tolerance 1e-12 |

**Relationship to the recorded experiments.** Experiments 10-13 are frozen records. Their scripts in `experiments/`,
configs and results are unchanged. The library modules they call (`validation`, `update_policy`, `time_weighting`,
`online_poisson`) now import the shared implementations under their old names, with identical code. This is checked
in two ways:

- `tests/test_evaluation_harness.py` (runs in CI) requires every prediction and scoring output of those modules on the
  synthetic league to equal, exactly, a snapshot captured from the code *before* the harness was introduced
  (`tests/golden/recorded_synthetic_snapshot.json`).
- `tests/test_reproduction_recorded.py` (golden, needs the data) regenerates every row of the six recorded prediction
  files and compares them at 1e-12. It scores nothing and changes no conclusion.

The recorded scripts keep their own private helpers (for example `date_clusters` and the lock checks); a test checks
that these agree with the harness versions.

### Full-coverage dynamic Poisson (`full_coverage_poisson_v1`, Experiment 15)

Registered 2026-10-02 (amendment PA1), implemented in `eplmodel.models.full_coverage`, run once on the historical
folds and locked (RESULTS_LOG Experiment 15). See `docs/preregistration/full_coverage_poisson_v1.md`
and `configs/full_coverage_poisson_v1.toml`. It links the earlier stages as follows:

- **Experiments 12-13:** it inherits H = 730 days, exponential weights and the once-per-date online refits. It adds
  empirical-Bayes Gaussian priors on team attack and defence:
  - promoted teams (absent from the previous season) are centred on the mean first-season strength of earlier promoted
    team-seasons;
  - continuing teams use a centred mean;
  - the variance is pooled and deconvolved.

  This lets every match of a 20-team season be forecast. All prior hyperparameters are estimated inside each fold from
  seasons before the target only, and no value is tuned.
- **Experiment 7:** its promoted-team taxonomy is used for diagnostics only.
- **Experiment 14:** the market benchmark is context only; odds are never a model input.
- **Evidence.** Scored only on the amendment-A1 selection targets. The primary reading is coverage plus common-group
  non-inferiority against the Experiment 13 online arm.

## 9a. Scoring conventions

- Probability arrays always have columns in the order (H, D, A) (`eplmodel.constants.OUTCOMES`).
  sklearn sorts class labels to (A, D, H), so the project computes log loss and Brier itself, with the column order
  passed explicitly (`eplmodel.evaluation.metrics`).
- Brier score is the squared error summed over the three outcomes and averaged over matches (range 0 to 2; a uniform
  forecast scores 2/3).
- Scoreline grids are truncated at 10 goals per side and the H/D/A probabilities renormalised by the captured mass.
- Models compared on a subset (for example the 648 matches without unseen teams) are aligned by `match_id`,
  never by row position.

## 10. Development style

I am learning Python and statistics through this project.

Therefore:

- Explain important mathematical ideas before code.
- Explain what each important block of code is doing.
- Prefer small, testable implementation steps.
- Do not dump a huge black-box implementation unless explicitly requested.
- When there is a methodological weakness, say so directly.
- Do not silently change assumptions.
