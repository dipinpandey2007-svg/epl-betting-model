# EPL Betting Model — Claude Code Instructions

## 1. Project objective

Build a serious, research-grade English Premier League football betting
probability model.

The long-term objective is to investigate whether genuine betting-market
inefficiencies can be identified, not merely to maximize match prediction
accuracy.

The project should eventually progress through:

1. Historical data pipeline
2. Elo baseline
3. Poisson / Dixon-Coles
4. xG and team-performance features
5. Squad / injury / lineup information
6. Tactical / contextual features
7. Market odds benchmark
8. Machine-learning models
9. Probability calibration
10. Walk-forward / backtesting
11. Closing-line value
12. ROI and proper scoring metrics
13. Structured football knowledge

Do not pretend future stages are complete when they are not. Status per stage:
`docs/ROADMAP.md`.

---

## 2. Research principles

- Prevent look-ahead bias and data leakage at all costs.
- Every prediction must use only information available before kickoff.
- Preserve chronological train / validation / test separation.
- Never tune on a test period.
- Use walk-forward evaluation where appropriate.
- Focus on probabilistic quality rather than accuracy alone.
- Important metrics include log loss, Brier score and calibration.
- Later stages will incorporate market-implied probabilities, CLV and ROI.
- Be skeptical of small improvements.
- Distinguish training fit from genuine out-of-sample improvement.
- Never fabricate data, results or sources.
- Do not recommend real-money betting during model development.

---

## 3. Development philosophy

The exploratory script has been refactored (2026-10-01) into a tested package.
The original lives in `archive/exploratory/` for reference only.

- Preserve validated methodology and results unless a deliberate change is
  justified, documented and tested.
- Do not silently change model definitions.
- Keep data, models, evaluation, experiments and reporting separate.
- Add tests for important mathematical and statistical functions.
- Make experiments reproducible.
- Keep future/incomplete stages clearly marked. Do not add fake or placeholder
  implementations merely to make the project look complete.

I am learning Python and statistics through this project. Explain important
mathematical and methodological decisions, but do not force every tiny coding
step to be manually copied by me. When working on the repository, prefer
making coherent changes directly to the codebase.

---

## 4. Commands

**Interpreter.** Always use this repository's own virtual environment,
`.venv` in the repository root (`.venv\Scripts\python.exe` on Windows,
`.venv/bin/python` elsewhere), never one belonging to another project.
A shell may inherit another project's activated environment; then `python`
silently runs *that* project's `eplmodel`, without this repository's guards.
Before downloading data or running experiments, check that both of these
point inside this repository:

```bash
python -c "import sys; print(sys.executable)"
python -c "import eplmodel; print(eplmodel.__file__)"
```

In a non-interactive shell, call `.venv/Scripts/python.exe` explicitly (or
activate `.venv` in the same command), since activation does not persist.
`.vscode/settings.json` selects this interpreter in VS Code.

```bash
pip install -e ".[dev]"                 # inside this repository's .venv (Python >= 3.11)
python -m eplmodel.data.download        # raw CSVs -> data/raw/ (git-ignored)
python -m eplmodel.data.build           # -> data/processed/matches.csv, validated + checksum-checked
python -m eplmodel.data.download --seasons 2425 && python -m eplmodel.data.build --dataset dev_v2
                                        # -> data/processed/matches_dev_v2.csv (adds 2024-25 validation)
python -m experiments.run_all           # reproduce all results -> results/<experiment>/metrics.json
python -m experiments.elo_k_selection   # or any single experiment
python -m pytest                        # all tests (~2 min with data: the Exp 10-13 reproduction gate refits 662 online models)
python -m pytest -m "not golden"        # skip golden regression tests
python -m pytest tests/test_elo.py::test_run_elo_has_no_lookahead
```

Use the virtual environment's interpreter for `python`. CI
(`.github/workflows/tests.yml`) runs the non-golden tests on Linux and Windows,
Python 3.11-3.13, plus a minimum-dependency job; golden tests run manually.

---

## 5. Architecture

- `src/eplmodel/`: library code. `data/` (download, build, load, validate,
  checksums), `models/` (elo, poisson, scoreline, dixon_coles), `evaluation/`
  (metrics, calibration, baselines, alignment, walk_forward, the common
  harness `forecasts`/`folds`/`scoring`/`segments`/`reproduction`, and the
  recorded-protocol modules validation, update_policy, time_weighting,
  online_poisson), `analysis/`
  (promoted-team folds), `market/` (Pinnacle odds snapshots, margin removal,
  validity, market forecast frames), `reporting/` (metrics.json with provenance, figures),
  `splits.py` (season roles, selection folds, dev-test and holdout guards),
  `holdout.py` (the only way to open the sealed holdout), `constants.py`.
- `experiments/`: one module per recorded experiment, each with
  `run(write=True)`. The golden tests call these same functions.
- `configs/baselines_v1.toml`: the frozen specifications (spec ids, K, rho,
  grids). Changing a value there is a methodological change.
- `tests/golden/golden_values.json`: full-precision values captured from the
  original script; `tests/test_golden_results.py` must keep passing.
- Common evaluation harness (2026-10-02). New experiments use
  `evaluation.folds.build_fold` (strict A1 guard), forecast frames
  (`evaluation.forecasts`: match_id index, `<arm>_H/D/A`, no result columns)
  and `evaluation.scoring` (losses, paired and date-clustered differences,
  calibration). Model-specific prediction code stays outside it. Experiments
  10-13 are frozen records: their scripts are unchanged and their library
  modules re-export the harness implementations. Two gates protect them:
  `tests/test_evaluation_harness.py` (CI; exact match to the pre-harness
  synthetic snapshot `tests/golden/recorded_synthetic_snapshot.json`) and
  `tests/test_reproduction_recorded.py` (golden; regenerates every recorded
  prediction row, tolerance 1e-12).
- `docs/`: METHODOLOGY, RESULTS_LOG, PROJECT_STATE, TEST_SET_ACCESS_LOG,
  ROADMAP, PROJECT_OVERVIEW.

---

## 6. Current validated state

Data: dataset v1 = EPL 2014-15 to 2023-24, 3,800 matches (all recorded results).
Dataset dev_v2 = v1 + 2024-25, 4,180 matches (`load_dev_matches()`), acquired
2026-10-01 and used by the 2024-25 validation run. Train = 2014-15 to 2021-22.

**2022-23 and 2023-24 are an exposed development test benchmark**, not a
final holdout. They may be used as historical fitting information for later
seasons, but never to tune, select or compare any new choice (never a
validation or selection target). Only specs in `REGISTERED_DEV_TEST_SPECS`
(`splits.py`) may be scored on them, and any new access must be logged in
`docs/TEST_SET_ACCESS_LOG.md` *first*.

**Final holdout: 2025-26 is sealed** (`docs/HOLDOUT_PROTOCOL.md`, tag
`holdout-freeze-v1`). Its outcomes must not be used for fitting, tuning,
feature or spec selection, descriptive analysis or any development decision.
Never download, inspect or load it outside `eplmodel.holdout`. 2026-27 is the
declared next holdout, under the same rules; 2025-26 remains sealed and
unacquired. 2024-25 was the validation season. It has been scored once as
validation (protocol `validation_2425_v1`, RESULTS_LOG Experiment 10): Elo
0.9848 log loss on all 380 matches; on the 342 without Ipswich, Elo 0.9836,
frequency baseline 1.0760, static Poisson 1.0854, staged Dixon-Coles 1.0857.
These are observed validation evidence and must NOT be used to tune the
established specifications. 2024-25 was then used once more by the
pre-registered update-policy diagnostic (protocol `update_policy_diagnostic_v1`,
RESULTS_LOG Experiment 11; diagnostic only, nothing selected or changed).
**Amendment A1 (access-log entry P1, 2026-10-01) retired 2024-25 as a
selection target.** It is now an exposed validation season: history for later
seasons only, never a selection, tuning or confirmation target, and re-scored
only for `REGISTERED_EXPOSED_VALIDATION_SPECS` to reproduce Experiments 10-13.
New selection work uses `splits.selection_folds()` (default targets 2017-18 to
2021-22, `SELECTION_TARGET_SEASONS`) and the strict guard
`splits.assert_selection_target`. `SELECTION_VALIDATION_SEASONS` and
`assert_valid_selection_target` are frozen legacy definitions kept only so the
recorded protocols reproduce; a static test in `tests/test_splits.py` refuses
their use (or the code `2425`) in any other experiment module or config.

| Spec | Dev-test log loss / Brier |
|---|---|
| `elo_k25_logreg_v1` (760 matches) | 0.9527 / 0.5642 |
| `frequency_baseline_v1` (760) | 1.0525 / 0.6355 |
| `poisson_static_v1` (648) | 1.0058 / 0.5989 |
| `dixon_coles_staged_v1`, rho = -0.04 (648) | 1.0072 / 0.5994 |
| Elo on the same 648 | 0.9547 / 0.5661 |

Elo as actually implemented: ratings updated **without** a home-advantage
term (home_adv=0), then a multinomial logistic regression on the pre-match
rating difference. The 43.08 shift added to that feature is redundant (the
intercept absorbs it; fits differ by only ~2e-6 from solver tolerance).
Applying home advantage inside the updates would be a new model needing its own
training-fold experiment.

Do not describe the Elo vs Poisson gap as Elo being intrinsically better: Elo
updates through the test seasons while the Poisson model is static, so the
comparison mixes model family with update dynamics.

Experiment 11 decomposed the 2024-25 gap on the 342 common matches:
Poisson − Online (+0.1019) = (Poisson − F2) + (F2 − F1) + (F1 − Online).
F1 = frozen season-start ratings with online Elo's layer; F2 = frozen ratings
with its own season-start layer.

- F1 − Online is the updating component: +0.0327. It is distinguishable under
  the pre-registered rule and positive in all five historical folds.
- F2 − F1 is the calibration-layer component: −0.0020, not distinguishable.
- Poisson − F2 is the remaining static-model / history difference: +0.0712.
  This is a 2024-25-specific observation (opposite sign in 4 of 5 historical
  folds). It does not establish its cause, and history weighting and model
  family are not separated.

So updating explains about a third of the 2024-25 gap. The pooled six-fold
figures are not an unbiased estimate. Do not use these results to tune
anything. A hypothesis prompted by them cannot be confirmed on 2024-25.

Experiment 12 (protocol `time_weighted_poisson_v1`, RESULTS_LOG Experiment 12)
has been run and recorded. It tested static Poisson fitted on a
recency-weighted history (w = 2^(−age/H), age in days; not updated in-season).

- H* = 730 days, selected by the one-SE rule on the development folds
  (2017-18 to 2021-22), is locked in the config.
- Development criterion D was **not met**: nested estimate −0.0058 log loss
  (clustered SE 0.0043).
- Validation criterion V was met on 2024-25: TW − static −0.0438 (clustered
  SE 0.0077). The pre-registered reading is **"2024-25-specific observation;
  cannot confirm"**.
- Weighting closed 0.0438 of the 0.0712 Poisson − F2 gap. The remaining
  TW − F2 gap is not a pure model-family effect.

2024-25 has now been exposed in three pre-registered experiments (10, 11 and
12). It must not be used to retune the Experiment 12 candidate or to confirm
any future candidate or hypothesis.

Experiment 13 (protocol `online_tw_poisson_diagnostic_v1`, diagnostic only)
compares the frozen time-weighted Poisson (H = 730) with the same weighted
likelihood refitted before every target date. The refit uses the history plus
target matches dated strictly before that date; unseen-team matches are neither
scored nor fitted.

- Historical stage (2017-18 to 2021-22, 1,604 matches): online − frozen
  −0.0117 log loss (clustered SE 0.0027), negative in 5 of 5 folds.
  Criterion U met; the harm mirror not met. The reading is
  `historical_evidence_online_refitting_helps`. No spec is adopted.
- The gain is about the size of Elo's updating gain on the same folds
  (+0.0099).
- Caveats: 2020-21 includes home-advantage re-learning; the registered
  accumulation pattern was not met; H = 730 was selected on these folds for
  the frozen arm.
- The historical result is locked in `[historical_locked]`.
- 2024-25 stage (H = 730 only, 342 matches): online − frozen −0.0230
  (clustered SE 0.0046). The registered label is
  `same_sign_as_historical_distinguishable`: a diagnostic observation only
  that cannot select, tune or confirm.
- 2024-25 decomposition: static Poisson − online Elo (+0.1019) = weighting
  (+0.0438) + Poisson updating (+0.0230) + online TW − online Elo (+0.0351).
  The remainder is not attributed to a single cause.

2024-25 has now been scored in four pre-registered experiments (10-13).

Experiment 14 (protocol `market_benchmark_v1`, pre-registered in `ac60693`)
is a market benchmark, not a model or selection candidate.

- Column meanings were verified from football-data's notes
  (`data/reference/football_data_odds_columns.md`): `PSH/PSD/PSA` are
  Pinnacle pre-closing odds and `PSCH/PSCD/PSCA` closing odds.
- The primary arm is closing odds with Shin margin removal. Pre-closing odds
  and proportional/power methods are registered sensitivity arms, all reported
  separately; none was chosen by score.
- It is scored on 2017-18 to 2021-22 only, with 100% coverage. The primary
  arm scores 0.9475 log loss / 0.5597 Brier on 1,900 matches.
- On the same matches, every established arm scores worse. For example,
  market − online Elo is −0.0242 (clustered SE 0.0051). This is context
  only, never selection evidence.
- 2022-25 odds were neither read nor scored. Odds are not model features.
  CLV is not implemented: it needs a defined pre-match snapshot and an
  explicit information cutoff.

Experiment 15 (protocol `full_coverage_poisson_v1`, pre-registered with amendment
PA1, implemented in `eplmodel.models.full_coverage`, run once and locked
2026-10-02): `docs/preregistration/full_coverage_poisson_v1.md`,
`configs/full_coverage_poisson_v1.toml`.

- Model: online time-weighted Poisson (H = 730, inherited) as a MAP fit with
  empirical-Bayes Gaussian priors on attack and defence. Promoted teams
  (absent from the previous season) get the mean first-season strength of
  earlier promoted team-seasons; continuing teams get a centred mean. Returning
  teams use the promoted prior plus their decayed history.
- Home advantage is global. Effects sum to zero over the 20 target teams.
- Priors are re-applied at every refit; no value is tuned.
- Primary arm `fc_hier`; arms M1 and S1 are ablation/sensitivity.
- Scored only on 2017-18 to 2021-22 (1,900 matches, 100% coverage, no fit failed).
- Primary arm M2 scores 0.9622 log loss / 0.5701 Brier on the full group.
- Primary non-inferiority vs the Experiment 13 online arm (common group) is
  INCONCLUSIVE: M2 − M0 +0.0037 (clustered SE 0.0014). The cost comes from the
  continuing-team prior (M2 − M1 on continuing-only +0.0043).
- On unseen-team matches, M2 − Elo is −0.0097 (SE 0.0107), not distinguishable.
- M2 stays primary; nothing is adopted or swapped. No 2024-25 stage has been
  run: it needs its own access-log entry first.

---

## 7. Known methodological issues and conventions

- **Promoted/unseen teams.** The static goal models cannot predict Nott'm
  Forest or Luton (112 of 760 matches excluded → 648 common subset). Inside
  training folds about 17% of validation matches are affected. Elo hides the
  same problem by starting new teams at 1500. Handling methods must be compared
  on training folds, never on 2022-24.
- **Staged Dixon-Coles.** The Poisson coefficients are fixed and rho is chosen
  on training likelihood. This is not joint MLE. Grid values outside
  `rho_bounds` are flagged and never selected.
- **Class ordering.** All probability arrays use (H, D, A) columns
  (`constants.OUTCOMES`). sklearn sorts labels to (A, D, H); use
  `eplmodel.evaluation.metrics`, never `sklearn.metrics.log_loss` directly.
- **Brier** = squared error summed over the 3 outcomes, averaged over matches.
- **Scoreline grid** truncated at 10 goals per side, then H/D/A renormalised.
- **Alignment.** Compare models on subsets by `match_id`, not row position.

---

## 8. Data handling

Raw and processed match CSVs are git-ignored: football-data.co.uk
redistribution terms are unverified. `data/checksums.json` (LF-normalised
SHA-256) identifies the dataset behind all results. Never commit match CSVs.
The public repository is a fresh export of the private development history,
which contained the CSVs and is never pushed. The project's MIT licence does
not cover third-party football data.

---

## 9. Reproducibility requirements

Every meaningful experiment should record: data period, train / validation /
test definition, features, model specification, hyperparameters, evaluation
metrics, important assumptions, limitations, and decision / conclusion. Add it
as a script in `experiments/`, an entry in `docs/RESULTS_LOG.md` (template at
the bottom) and, once frozen, a spec in `configs/` and golden values in the tests.
