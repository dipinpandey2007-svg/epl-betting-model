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
python -m pytest                        # all tests (~5 s)
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
  (metrics, calibration, baselines, alignment, walk_forward,
  validation, update_policy), `analysis/`
  (promoted-team folds), `reporting/` (metrics.json with provenance, figures),
  `splits.py` (season roles, selection folds, dev-test and holdout guards),
  `holdout.py` (the only way to open the sealed holdout), `constants.py`.
- `experiments/`: one module per recorded experiment, each with
  `run(write=True)`. The golden tests call these same functions.
- `configs/baselines_v1.toml`: the frozen specifications (spec ids, K, rho,
  grids). Changing a value there is a methodological change.
- `tests/golden/golden_values.json`: full-precision values captured from the
  original script; `tests/test_golden_results.py` must keep passing.
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
unacquired. 2024-25 is the validation season. It has been scored once as
validation (protocol `validation_2425_v1`, RESULTS_LOG Experiment 10): Elo
0.9848 log loss on all 380 matches; on the 342 without Ipswich, Elo 0.9836,
frequency baseline 1.0760, static Poisson 1.0854, staged Dixon-Coles 1.0857.
These are observed validation evidence and must NOT be used to tune the
established specifications. 2024-25 was then used once more by the
pre-registered update-policy diagnostic (protocol `update_policy_diagnostic_v1`,
RESULTS_LOG Experiment 11; diagnostic only, nothing selected or changed).
The rule for future selection work is `splits.selection_folds()`
(validating 2017-18 to 2021-22 and 2024-25), reporting 2024-25 both pooled
and separately. The recorded experiments
predate this rule and use training folds only.

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
