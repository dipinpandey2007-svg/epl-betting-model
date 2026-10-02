# Pre-registration: shots information (`shots_information_v1`, planned Experiment 16)

| | |
|---|---|
| Registered | 2026-10-02, before any implementation, fit or score |
| Config (frozen choices) | `configs/shots_information_v1.toml` |
| Typed view | `eplmodel.models.shots_spec` |
| Source record | `data/reference/football_data_shot_columns.md` |
| Status | design only; no prediction or result exists |
| Builds on | Experiment 13 (online time-weighted Poisson, H = 730: the baseline), Experiment 12 (H lock, selection procedure), amendment A1 (selection targets) |

## 1. Question

Does pre-match **historical shot information** add predictive information to the established goals-only online model?
This is an incremental-information test. The baseline is unchanged, and the shot model may only re-weight how team
strengths are estimated.

**Why shots might help.** Goals are rare: about 2.7 per match. A team's goal record is therefore a noisy measure of
its strength. Shots, and especially shots on target (about 8–9 per match), are measured with much more data. If
relative attacking and defensive strength shows up in shots as it does in goals, shot-based strengths estimate it with
less noise.

## 2. Data actually available (verified, section B)

| Column | Source definition (football-data notes) | Coverage 2014-15 … 2021-22 |
|---|---|---|
| `HST` / `AST` | "Home/Away Team Shots on Target" | 3,040 / 3,040 matches, all valid integers |
| `HS` / `AS` | "Home/Away Team Shots" | 3,040 / 3,040, all valid integers |

- **Other columns.** No woodwork, offside or xG columns exist in the files. Corners, fouls and cards exist but are
  not shot fields and are not used.
- **What the notes leave undefined.** They do not say whether blocked shots count, or whether goals count as shots on
  target. In the data, goals ≤ shots always, goals > shots on target in 18 matches (0.6%), and shots on target >
  shots in 1 match. See the source record.
- **What was read.** Values of 2022-23 onward were not read. Shots are **post-match** statistics of the match they
  describe, so they are information only for **later** matches.

## 3. Information set and cutoff (section A)

For a target match on date d of target season S, the forecast may use:

    F(d) = { all matches of the fold history }  ∪  { common-group target matches with Date < d }

with their goals **and** shot counts, exactly B0's fitting set (Experiment 13).

- No shot value of any match dated ≥ d enters any fit used on d.
- The target match's own shots are never used.
- Every match on d shares the same information set, so there is one refit per date.
- No season after 2021-22 is loaded.

## 4. Model (sections B, C, D, E)

### B0, the baseline (unchanged)

Experiment 13's online arm. At each target date d, a Poisson GLM is fitted on F(d) with weights
w = 2^(−age/730), age in days:

    log E[home goals] = mu + h + att_i + def_j,     log E[away goals] = mu + att_j + def_i

### Shot model (new; same structure, same rows, same weights)

    log E[home shots] = nu + h_s + sa_i + sd_j,      log E[away shots] = nu + sa_j + sd_i

- **Primary:** shots = shots on target (`HST`, `AST`).
- **Sensitivity:** shots = total shots (`HS`, `AS`).
- **Venue (section C).** One global home term h_s; team effects are pooled over venue. This mirrors the goal model
  and avoids halving each team's data. There is no team-specific home split.
- **Weighting (section D).** The locked H = 730 is inherited; no new temporal parameter.

### Centring

For each fit, both models' team effects are re-expressed to sum to zero over C(d): the target teams present in the
fold history (B0's eligible target teams). Each model's intercept is adjusted to match. This is an exact
re-parameterisation of each maximum-likelihood fit, so B0's forecasts are unchanged.

### B1 forecast (section E): blend the relative strengths, keep the goal level

    log lambda_home = mu + h + (1−ω)·att_i + ω·sa_i + (1−ω)·def_j + ω·sd_j
    log lambda_away = mu     + (1−ω)·att_j + ω·sa_j + (1−ω)·def_i + ω·sd_i

- ω ∈ {0, 0.25, 0.5, 0.75, 1}.
- **ω = 0 is B0 exactly.** ω = 1 uses only shot-based relative strengths. The goal level mu and the home advantage h
  always come from the goal model.
- Probabilities come from the inherited 10-goal independent-Poisson grid.

### Why this form

| Alternative | Why it was not chosen |
|---|---|
| Shot rates as extra covariates in the goal GLM | Collinear with the team effects in the same fit. Valid only with lagged, row-by-row pre-match features for every history match. Its coefficient would mostly measure short-term form, not the information content of shots |
| Stacking goal and shot predictions with fitted weights | Needs a separately held-out fitting layer for the weights. Weights fitted on the same rows as the effects are biased towards goals |
| A joint latent-strength model | More parameters and a non-standard likelihood for a first test |
| **Blend of centred relative strengths (chosen)** | One interpretable weight with B0 nested at ω = 0. The goal scale and home advantage are untouched. Each model is a standard GLM that the project already fits and tests |

**Stated assumption and limitation.** Relative strengths are taken to be comparable on the log scale of goals and of
shots. If stronger teams also convert chances better, shot-based strengths are compressed relative to goal-based
ones. The selected ω then absorbs that trade-off; no conversion parameter is added.

## 5. Promoted, returning and missing data (section F)

- **Never-seen teams.** Under B0's registered rule, matches of a team absent from the fold history (newly promoted,
  never seen) are **neither fitted nor scored, in every arm**. These are 296 of 1,900 target matches
  (74/108/38/38/38). They are reported as excluded, not silently deleted. The shot and goal models share B0's fixed
  team parameter space.
- **Returning teams** (in the fold history but absent from the previous season) keep their decayed history in both
  models, as in B0.
- **The Experiment 15 priors are not reused.** Combining shots with full coverage would change two things at once;
  that is a possible later stage built on frozen results of both.
- **Missing shot values.** A match with a missing, non-integer or negative value in the arm's two shot columns is
  left out of that arm's shot fit only; its goals still enter the goal fit, and the count is reported. There are
  none in 2014-15 to 2021-22, and nothing is imputed.

## 6. Arms and ablations (sections H, I)

| Arm | Content | Role |
|---|---|---|
| B0 | Experiment 13 online arm (ω = 0) | baseline, recorded predictions; ω = 0 must reproduce them to 1e-12 |
| **B1** | shots-on-target blend, ω by the selection rule | **primary**: does shot-on-target information add to B0? |
| S1 | total-shots blend, same procedure | sensitivity: shots vs shots on target |

**One further descriptive ablation, the full ω grid (including ω = 1, shots only for relative strength) for each
arm.** It shows whether any gain comes from combining the two sources or from replacing goals altogether. It is
reported in-sample and never used to choose anything beyond the registered rule. No other arm is part of this
protocol.

## 7. Hyperparameter control (section G)

| Choice | Status |
|---|---|
| H = 730, weights, refit cadence, strictly-earlier information, fitting set, unseen-team rule, IRLS settings and failure rule, independent Poisson, 10-goal grid | **inherited, locked** (Experiments 12-13) |
| Shot model form, global home term, centring set, blend form with goal level, ω grid {0, 0.25, 0.5, 0.75, 1}, shots on target as primary, total shots as sensitivity, missing rule | **new, fixed now** |
| ω | **the only selected value**, by the rule below, separately for B1 and S1 |

### Selection rule (as in Experiment 12, but towards the baseline)

1. On each target's common group, score every ω.
2. The criterion is the mean over targets of each target's mean log loss.
3. ω_min is the argmin.
4. **One-SE rule:** choose the **smallest** ω (closest to B0) whose pooled per-match difference to ω_min is at most 1
   date-clustered SE. Exact ties go to the smaller ω.

Two estimates follow:

- **Nested estimate:** for outer targets 2018-19 … 2021-22, ω is selected from earlier targets only, and the outer
  target is scored at that ω.
- **Locked value:** ω* comes from all five targets and is written to `[locked]` in a separate commit before any later
  stage.

The 2024-25 season plays no part in any of this.

## 8. Evaluation (sections J, K)

- **Data and harness.** dev_v2 cut to seasons ≤ 2021-22, the targets 2017-18 … 2021-22 (`selection_folds()`), and
  the common evaluation harness.
- **Scoring population.** The Experiment 13 common group: 306/272/342/342/342 = **1,604 matches**. These are
  identical matches for every arm, aligned by match_id. Refits per fold: 98/98/110/128/119.
- **Metrics.** Log loss (primary) and Brier; paired (left − right) differences with naive and date-clustered SEs,
  clusters (target, date); per-target and pooled; Experiment 11 segments; calibration in the large; returning vs
  continuing matches.

**Criterion S (B1, nested outer folds).** All four conditions must hold:

- pooled B1 − B0 log loss < −0.002;
- |mean| > 2 clustered SE;
- negative in at least 3 of 4 outer folds;
- pooled Brier < 0.

If they hold, the reading is `historical_evidence_shots_on_target_add_information`. The mirror reading is "hurt";
otherwise the reading is `no_distinguishable_shot_information`. A locked ω* = 0 is additionally reported as "no shot
weight selected". S1 is read with the same rule and reported separately.

**Context only, never selection.** After the shot specification is evaluated, B1 may be compared descriptively with
online Elo (Experiment 10) and the Experiment 14 market benchmark on the same matches. No odds enter any arm.

## 9. Validation seasons (sections L, M)

- **2024-25** is an exposed validation season. It is not used for any choice here: shot variable, formulation,
  weighting, ω, or whether the shot model is kept. A descriptive 2024-25 stage may follow only after `[locked]` is
  committed and a separate access-log entry with a `LOGGED_EXPOSED_VALIDATION_ACCESSES` registration exists.
- **2025-26** (sealed holdout) and **2026-27** (declared next holdout) are not read, downloaded or used.

## 10. Implementation and test plan (sections N, O; not written yet)

Planned:

- a shots table read from raw files (key and shot columns only, raw checksums verified);
- `eplmodel.models.shots` (shot GLM, centring, blend);
- `experiments/shots_information.py` (development stage: grid, selection, nested estimate, criterion S).

| Test | What it proves |
|---|---|
| No post-match leakage | A forecast on d is unchanged when any shot or goal value of a match dated ≥ d changes; earlier values do change it |
| Strict cutoff | Every row of every shot fit is dated < d; the target match's shots are never read |
| Aggregation and centring | Centred effects sum to zero over C(d); the re-parameterisation leaves each model's fitted rates unchanged (1e-12) |
| Weighting | Shot-fit weights equal 2^(−age/730) and are identical to the goal fit's |
| Missing data | A row with a missing or invalid shot value leaves that arm's shot fit only; the counts are reported |
| Venue | One global h_s; the home-team row uses h_s and the away row does not |
| Promoted and returning | Unseen-team matches are neither fitted nor scored in any arm; a returning team's history enters both models with decayed weights |
| ω = 0 | Reproduces the recorded Experiment 13 online predictions to 1e-12 |
| Probability validity | Finite rows summing to 1 in (H, D, A) order |
| Determinism | Two runs give bit-identical forecasts |
| Coverage | Registered common-group sizes and refit counts hold before scoring |
| Selection rule | One-SE towards the smaller ω; tie-break; nested selection uses earlier targets only |
| Guards | Targets outside the selection set refused; holdout seasons refused; no result column in the shots table; no market, xG or ML input |
