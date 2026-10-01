# Final-holdout protocol (holdout_v1)

| | |
|---|---|
| Protocol id | `holdout_v1` |
| Frozen | 2026-10-01, git tag `holdout-freeze-v1` |
| Final holdout | **2025-26** (season code `2526`) |
| Next holdout | **2026-27** (`2627`), used once 2025-26 has been opened |
| Machine-readable form | `configs/holdout_v1.toml`, season roles in `src/eplmodel/splits.py` |
| Access record | `docs/TEST_SET_ACCESS_LOG.md`, entries H0, H1, … |

This protocol was frozen **before** any 2024-25 or 2025-26 data were downloaded, inspected or scored by this
project.

## 1. Season roles

| Role | Seasons | Use |
|---|---|---|
| Training | 2014-15 … 2021-22 | Fitting; walk-forward selection folds |
| Exposed development benchmark | 2022-23, 2023-24 | **Historical fitting information only** when predicting later seasons. Never a validation or model-selection target. Registered specs may be re-scored to reproduce recorded results. |
| Validation | 2024-25 | Recent validation season, part of model selection |
| **Final holdout (sealed)** | **2025-26** | Opened once, by the procedure in §5 |
| Next holdout (declared) | 2026-27 | Treated exactly like the final holdout by every guard |

## 2. What "sealed" means

2025-26 is **operationally sealed**. After the seal, its outcomes must not be used for:

- model fitting;
- tuning;
- feature selection;
- specification selection;
- descriptive analysis (including simple rates such as home-win or draw frequency, goal averages or league tables);
- any other development decision.

**2025-26 is not "unknown to the world".** The season is complete and its results are public. The researcher may
know them from following football, and AI assistants used on this project may have seen them in their training
data. The seal does not claim otherwise. It is a seal on this project's *development process*: every modelling
choice must be justified with evidence from the training folds or from 2024-25, never by appeal to what happened in
2025-26. Such knowledge is a residual risk that the code cannot remove, and it is stated here so that the final
result can be read with it in mind.

Information about 2025-26 that was known **before the season started** is not sealed. For example, the identity of
the promoted clubs is not sealed, because a pre-kickoff forecaster would have it. It must still be sourced and
recorded like any other reference data.

## 3. Model selection before the holdout

Selection uses `eplmodel.splits.selection_folds()`:

| Fold target | History (fitting information) |
|---|---|
| 2017-18, 2018-19, 2019-20, 2020-21, 2021-22 | all training seasons before the target (the folds of the recorded Elo K selection) |
| 2024-25 | 2014-15 … 2023-24, including the exposed seasons as history |

- Report the **pooled** selection result and **2024-25 separately**.
- 2024-25 is a single season of 380 matches. It must not be treated as more precise than the multi-fold historical
  evidence, and a choice should not rest on 2024-25 alone.
- Every comparison scored on 2024-25 is recorded in `RESULTS_LOG.md`, so its own exposure can be audited later.
- The exposed 2022-24 seasons are never fold targets (`assert_valid_selection_target`).

## 4. Information policy for every prediction

A prediction for a match may use only information available before that match. The processed data have a date
but no kickoff time, so the operational rule is: **a match on date d may use results with a date strictly before
d**, never same-day results.

Dynamic models (for example Elo, or a goal model refitted as a season progresses) may therefore use results from
earlier in the same season, including earlier holdout matches during the final evaluation. They must never use
anything from after the prediction date. The common walk-forward harness, to be built next, will enforce this
rule by giving each model only the history before the cutoff.

## 5. Accessing the holdout

Holdout outcomes are read only through `eplmodel.holdout.open_final_holdout` and `load_holdout_matches`. Access
requires **all** of the following:

1. **Pre-registration.** Every spec to be scored is listed in `[registration] spec_ids` of
   `configs/holdout_v1.toml`, committed before access. The primary metric (log loss) and the secondary metrics
   (Brier score, calibration) are fixed in the same file now.
2. **An authorised entry.** An `[[access]]` table in that config names the entry id and exactly the specs to be
   scored, and the same entry id is written in `TEST_SET_ACCESS_LOG.md` **before** the run.
3. **A committed state.** The working tree is clean outside `results/`, and HEAD descends from `holdout-freeze-v1`.
4. **Sealed data.** `data/holdout_manifest.json` exists and the processed holdout file matches its recorded checksum.

The holdout data never live in `data/raw/` or `data/processed/`:

- they go in the git-ignored `data/holdout/`;
- `eplmodel.data.download`, `eplmodel.data.build` and `eplmodel.data.load_matches` all refuse holdout seasons;
- no module in `experiments/` may import the holdout module (checked by `tests/test_holdout.py`).

The final evaluation will live outside `experiments/` and will never be called by `experiments.run_all`.

These checks make it impossible to use the holdout **by mistake**. They do not and cannot prevent deliberate
circumvention; that is a matter of research discipline, and this document is the commitment.

## 6. Sealed acquisition (next step, not yet done)

The 2025-26 file will be downloaded once by a dedicated command that:

- writes only to `data/holdout/`;
- records the raw and processed content checksums in `data/holdout_manifest.json` (committed), which pins the
  upstream version;
- runs structural validation (columns, 380 matches, score/result consistency) and reports **pass/fail only**,
  printing nothing derived from match outcomes.

This is recorded as an access-log entry ("sealed, not scored"). Until then there are no holdout data on disk.

## 7. Single use, then the next holdout

- The holdout is scored **once**, for the list of pre-registered specs. Re-running the same authorised entry
  reproduces the same numbers and adds no exposure; scoring any further spec is a new, logged exposure.
- Once scored, 2025-26 becomes an exposed benchmark. The guards are then updated in a logged protocol change.
- **2026-27 is declared the next holdout.** It is under the same seal from now: it is not downloaded, inspected or
  used in development until it becomes the active holdout and a protocol version for it is frozen.

## 8. Changing this protocol

Any change to this document, `configs/holdout_v1.toml` or the season roles in `splits.py` is a protocol change. It
must be committed and recorded in `TEST_SET_ACCESS_LOG.md` before any holdout access that relies on it.
