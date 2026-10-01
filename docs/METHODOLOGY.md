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
- The rule for future and new selection work is `eplmodel.splits.selection_folds()`: the training-season folds
  validating 2017-18 to 2021-22, plus 2024-25, reported pooled and separately. The recorded experiments predate this
  rule and used training folds only. 2024-25 has been acquired and validated (dataset dev_v2, 2026-10-01) but has
  **not yet** been used for model selection or scoring.
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
