# Pre-registration: full-coverage dynamic Poisson goal model (`full_coverage_poisson_v1`, planned Experiment 15)

| | |
|---|---|
| Registered | 2026-10-02, before any implementation, fit or score |
| Config (frozen choices) | `configs/full_coverage_poisson_v1.toml` |
| Typed view of the config | `eplmodel.models.full_coverage_spec` |
| Status | design only; no prediction or result exists |
| Amendments | PA1 (2026-10-02): clarifies the identifiability parameterisation and records the rationale of the inherited 0.002 criterion (section 11) |
| Builds on | Experiment 12 (H = 730 locked), Experiment 13 (online refitting), Experiment 7 (promoted-team taxonomy), Experiment 14 (market benchmark, context only), amendment A1 (selection targets) |

## 1. Research question and motivation

The online time-weighted Poisson model of Experiment 13 cannot score a match involving a team absent from the fold
history: 296 of the 1,900 historical target matches (16%), every season. Elo hides the problem by starting new teams
at the average rating.

**Question.** Can an online time-weighted Poisson model, given explicit empirical-Bayes season-start priors, produce a
valid forecast for every match of a 20-team season (including newly promoted, returning and never-before-PL teams),
without degrading the matches the existing online model can already score?

## 2. What is known before the first match of target season S

| Information | Known before S? | Used |
|---|---|---|
| Results of every season before S that is in the fold history | yes | likelihood; empirical-Bayes (EB) estimation |
| Fixture list of S (the 20 teams, home/away, dates) | yes (published before the season) | team set, groups, refit dates |
| Which target teams played the PL in S − 1 | yes (from S − 1 results) | prior group |
| `data/reference/team_history.csv` (seasons out of the PL before a return) | yes (pre-season public facts, sourced) | diagnostic taxonomy only |
| Results of S | only those dated strictly before the match being predicted | online refits |
| Championship results, transfers, odds, xG, lineups | not in the project data | **not used** |

**Is the project data enough for a promoted-team prior?** Yes, for a simple one:

- The data show which team-seasons were newly promoted for every season from 2015-16 on (absent from the previous
  data season). The reference set has 6 promoted team-seasons for target 2017-18, rising to 18 for 2021-22
  (`[empirical_bayes] expected_team_seasons`).
- The 2014-15 promoted clubs cannot be identified from the data. `team_history.csv` covers only teams that appear
  later in the window, so 2014-15 is left out of the EB set rather than adding a new external source.
- Championship data would describe promoted teams before their first PL match and could sharpen the prior a great
  deal. That is a separate future stage with its own data acquisition and protocol; it is not used here.

## 3. Model (sections A, F, G)

### Likelihood

For a match m with home team i and away team j, the goals x (home) and y (away) are independent Poisson:

    log E[x_m] = mu + h + att_i + def_j
    log E[y_m] = mu       + att_j + def_i

`def` is the team's conceding effect (higher = concedes more), as in `poisson_static_v1`.

- **Independent Poisson.** It is the base of Experiments 12-13, so the anchor to Experiment 13 is exact. A Dixon-Coles
  adjustment is **not** included: it would change two things at once. On the five historical folds of Experiment 10,
  staged Dixon-Coles − Poisson was −0.0003, +0.0014, +0.0003, +0.0017 and −0.0002 log loss, so no benefit was seen.
  It remains a possible separate arm.
- **Home advantage** h is one global league parameter with no prior. It is re-estimated at every refit from the fitting
  set only, so it can move during a season (as in Experiment 13, for example in 2020-21). It is never computed from any
  match outside the fitting set, which avoids the earlier full-data home-advantage leakage of the exploratory Elo
  work. There is no team-specific home advantage.
- **Probabilities.** The scoreline grid is truncated at 10 goals per side and H/D/A renormalised, as before.

### Weights (inherited, Experiment 12)

w_m(d) = 2^(−age_m(d) / 730), where age_m(d) is the number of days from match m to the refit date d (the date being
predicted). Without priors the reference date does not change the fit. With priors the absolute weight scale matters,
so it is fixed here: a match played just before d has weight close to 1 and counts as roughly one observation against
the prior.

### Objective at refit date d of target season S (MAP)

    maximise  sum_{m in F_S(d)} w_m(d) [log Pois(x_m; lambda_m) + log Pois(y_m; mu_m)]
              - sum_{t in T_S} [ (att_t - a_t)^2 / (2 tau_att^2) + (def_t - b_t)^2 / (2 tau_def^2) ]
    subject to sum_{t in T_S} att_t = 0  and  sum_{t in T_S} def_t = 0

- **T_S** is the 20 teams of the target fixture list.
- **F_S(d)** is every fold-history match plus every target match dated strictly before d.
- **No prior** is placed on mu, h, or the effects of teams that appear only in the history.

### Parameterisation and identifiability (exact)

**Team sets at refit date d.**

- **T_S:** the 20 target teams, from the fixture list.
- **D(d):** the teams with at least one match in F_S(d).
- **Q_S(d) = D(d) \ T_S:** the *history-only* teams, which appear in the fitting data but not in the target season (for
  example teams relegated before S). Under arm S1 this set also contains the pre-absence identity of a returning team.
- **The parameter set** is U = T_S ∪ Q_S(d).

**Parameters.**

- mu (intercept) and h (home advantage);
- att_t and def_t for every t in U, which is 2|U| effects.

Effect coding:

- For the 20 target teams, the two linear constraints

      (C1)  Σ_{t ∈ T_S} att_t = 0          (C2)  Σ_{t ∈ T_S} def_t = 0

  are imposed by elimination. The target team that is last alphabetically, t*, is not a free parameter:
  att_{t*} = −Σ_{t ∈ T_S, t ≠ t*} att_t, and likewise for def. Its prior still applies to this implied value.
- **History-only teams (Q_S(d)) are not constrained and get no prior.** Their att and def are free parameters measured on
  the same scale, relative to the average of the 20 current teams. Their matches inform mu, h and the target teams'
  effects through shared opponents, with the usual time-decay weights. They are never predicted.
- The free vector is θ = (mu, h, att and def of the 19 free target teams, att and def of every history-only team), of
  dimension 2 + 38 + 2|Q_S(d)|.

**Why the constraints are needed.** The likelihood depends on θ only through the linear predictors (one row per goal
count, as in the long format of `poisson_static_v1`):

    eta_home(m) = mu + h + att_home + def_away,    eta_away(m) = mu + att_away + def_home

It is unchanged by either shift:

- (S_att) att_t → att_t + c for **every** t in U, with mu → mu − c;
- (S_def) def_t → def_t + c for every t in U, with mu → mu − c.

So without constraints mu and the effect levels are not identified. C1 and C2 remove exactly these two directions. A
shift by c changes Σ_{t∈T_S} att_t by c·(number of target teams in U) = 20c ≠ 0, so only c = 0 satisfies C1.

- **The intercept is identified by C1 and C2.** Once the effect levels are fixed, mu is the log goal rate of an
  average current team (away from home), and h the log home/away ratio.
- **The constraints change the model, not just its coordinates.** The priors are not invariant to S_att and S_def. The
  constraints define the scale on which the prior means are stated: effects relative to the average of the 20 teams
  of the season. This is the scale of the single-season EB fits, which also sum to zero over that season's teams.

**Uniqueness (strict concavity on the constrained space).** Write the objective as f(θ) = ℓ_w(Xθ) − ½(θ − m)ᵀP(θ − m).
Here P is the diagonal prior precision, with 1/τ² on each penalised target-team effect, including the implied one,
and 0 elsewhere. Then

    −∇²f(θ) = Xᵀ W(θ) X + P,   W = diag(w_m λ_m) > 0,

so for any free direction v ≠ 0, vᵀ(−∇²f)v = Σ_m w_m λ_m (x_mᵀv)² + vᵀPv. This is zero only if Xv = 0 **and** Pv = 0.
Three facts show that no such v exists:

1. **Connected comparison graph.** Every season in the data is a complete double round robin, consecutive seasons share
   17 teams, and the continuing target teams played S − 1. So all teams in D(d) are linked by matches. Hence the
   only directions with Xv = 0 that move teams with data are the two shifts S_att and S_def. Directions can also move
   the effects of target teams **without** any match in F_S(d) (never-seen promoted teams before their first match),
   since those effects appear in no row.
2. **h is separable.** Every pair of teams in a season meets home and away, so changing h cannot be offset by team
   effects. h enters Xv = 0 only with coefficient 0.
3. **Every target team without a match has a prior** in M1, M2 and S1, because such a team was not in S − 1 and is
   therefore promoted. So Pv = 0 forces the components of these teams to 0. What remains is c₁·S_att + c₂·S_def, and C1
   (C2) then force c₁ = c₂ = 0, because at least 17 target teams with data enter the sums.

So the Hessian is negative definite on the constrained space, and there is **at most one** maximiser.

**Existence.** A maximiser exists if no unpenalised effect can drift to ±∞ while improving the likelihood. This
happens, for example, if a team never scored. Before each fit the implementation checks that every **unpenalised**
team (history-only teams; continuing teams in M1) has positive weighted goals scored and conceded in F_S(d), and that
the fitting set has positive total weighted goals. Penalised effects are bounded by their priors. If a check fails, or
the solver fails to reach max |gradient| ≤ 1e-9 with finite parameters, the stage aborts with no fallback (`[solver]`).

**Arm M0** (no priors, unseen-team matches excluded) has the same constraints over the target teams present in its
data. Its predictions do not depend on this choice, because the maximum-likelihood fit is invariant to the shifts. It
is therefore identical to the treatment-coded fits of Experiment 13, which is what the anchor test checks.

## 4. Season-start priors (sections B, C)

### Groups (fixtures only)

| Group | Rule | Prior mean (attack, defence) |
|---|---|---|
| continuing | played the PL in S − 1 | (a_C, b_C) = −n_P/(20 − n_P) × (a_P, b_P) |
| promoted | did not play the PL in S − 1: newly promoted, returning after an absence, or never in the PL | (a_P, b_P) from EB |

Both groups use the same variances (tau_att, tau_def).

- **Why the continuing mean is derived.** The continuing mean is defined so that the prior means sum to zero over the
  20 target teams, matching the constraint. Every season in the data has n_P = 3, so this equals the EB mean of
  continuing team-seasons.

### Empirical-Bayes estimation, nested in each fold

For target S, the EB seasons are every data season s with 2014-15 < s < S. Membership of s − 1 must be observable, so
2014-15 itself is excluded.

1. For each EB season s, fit the unweighted single-season Poisson model by maximum likelihood on that season's 380
   matches only. Effects sum to zero over its 20 teams, and standard errors come from the inverse Fisher information
   (including the implied 20th team).
2. Label each team-season promoted (not in s − 1) or continuing.
3. a_P (b_P) = the mean of the promoted team-seasons' attack (defence) estimates.
4. tau_att^2 = max(pooled within-group variance of the attack estimates − mean squared SE, 0.05^2). The pooled variance
   uses ddof = 2, one for each group mean; the same rule gives tau_def. Subtracting the squared SEs removes estimation
   noise, so tau describes the spread of true team strengths, not of noisy estimates. The floor 0.05 (about 5% in
   goal rate) is fixed a priori and is reported whenever it binds.
5. Attack and defence priors are independent. Their empirical correlation is reported as a diagnostic.

Only results of seasons before S enter. The EB values are computed once, before the first target match, and held fixed
for the whole season.

### How partial history is blended with the prior

No threshold or schedule is added. The blend follows from the posterior. Approximately, for one team:

    posterior effect ≈ (prior mean / tau^2 + I_t × data estimate) / (1 / tau^2 + I_t)

Here I_t ≈ Σ_m w_m(d) λ_m is the team's weighted information (roughly the weighted expected goals in its fitted
matches).

- **Lots of recent PL matches.** The team's data dominate.
- **Few or old matches.** The prior dominates.
- **During the season.** Each new match adds weight close to 1, so the data take over within a few rounds.
- **Example.** With tau ≈ 0.2, the prior is worth about 1/0.04 = 25 units of information, roughly 20-25 recent matches.

### Promoted, returning and new teams

| Case | Primary rule (M2) |
|---|---|
| Never in the fold history (newly promoted, never-PL or long absence before the window) | Promoted prior only; no data until its first target match |
| Returning: absent from S − 1, present earlier in the fold history | Promoted prior **plus** its earlier PL matches at their decayed weights. The team is not treated as continuously observed: there are no matches from its absence, and the latest PL match is already ≥ about 15 months old at the season start |
| Promoted last season, now continuing (one PL season) | Continuing prior; its 38 recent matches dominate |
| Continuing with long history | Continuing prior; data dominate |

Approximate weight at the season start of a returning team's most recent PL match (age ≈ 365k + 90 days after k
seasons out):

| Seasons out | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Weight | 0.65 | 0.46 | 0.32 | 0.23 |

Older matches weigh up to about 25% less again. After one season out the stale history and the prior have similar
weight; after three or more the prior dominates.

Returning and newly promoted teams use the same prior. The only difference is the decayed data that a returning team
actually has. Sensitivity arm S1 ("identity break") tests the alternative in which those earlier matches are moved to a
separate history-only identity. That choice is set in advance and does not depend on any 2024-25 observation (for
example Experiment 13's 2024-25 returning-team pattern, which must not motivate anything here).

## 5. Online updating (section H)

- **Cadence.** One full refit per distinct target date d, over F_S(d). This equals a refit per match, because no team
  plays twice on a date.
- **What is re-estimated.** All parameters: mu, h, and every attack and defence effect, including the promoted teams'.
- **History.** All fold-history matches stay in every fit with their exponential weights; there is no rolling window.
  Target matches get the same weight formula and dominate as they accumulate.
- **Priors.** They are re-applied, unchanged, at every refit. The prior is part of the model, so it acts as continuing
  regularisation whose relative influence shrinks as data accumulate. With full refits and no warm start, an
  "initialisation-only" prior is not defined, so it is not an arm.
- **Starts.** There are no warm starts. Each fit starts deterministically: mu = log of the weighted mean goals of the
  fitting set, h = 0, target-team effects at their prior means, all other effects 0.
- **Solver.** Damped Newton with an exact gradient and Hessian. Convergence requires max |gradient| ≤ 1e-9, with at
  most 200 iterations and 40 step halvings. On failure the stage aborts, with no fallback.

## 6. Information boundary (section D)

Let R(m) be the outcome of match m, Fix(S) the fixture list of S, and TH the team-history table. For target S and a
match on date d:

    eta_S        = EB( { R(m) : season(m) in (2014-15, S) } , Fix(S) )                  (prior hyperparameters)
    theta_S(d)   = argmax objective( { R(m) : season(m) in history(S) } ∪ { R(m) : season(m) = S, date(m) < d } ; eta_S )
    p(match)     = Poisson-grid( theta_S(d) )

There is no term for season(m) > S, for season(m) = S with date(m) ≥ d, or for any season after 2021-22. The
operational guards are:

1. The match table is cut to seasons ≤ 2021-22 straight after reading (`restrict_to_max_season`).
2. Folds come from the harness `build_fold`, whose strict guard admits only the selection targets.
3. The EB function receives only rows with season < S and refuses any other row.
4. Each refit set is built with `rows_strictly_before(history, target, d)`.
5. The fixture frame passed to prediction has no result columns, and forecast frames are checked by
   `check_forecast_frame`.

No 2024-25 information was used to design this protocol.

## 7. Arms and ablation (section J)

| Arm | Priors | Unseen-team matches | Purpose |
|---|---|---|---|
| M0 `poisson_tw_online` | none (MLE) | excluded from fits and scores | Experiment 13 baseline (recorded predictions, common group) |
| M1 `fc_promoted` | promoted prior only; continuing teams unpenalised | fitted and scored | effect of handling new and returning teams |
| **M2 `fc_hier` (primary)** | promoted and continuing priors | fitted and scored | full design |
| S1 `fc_hier_break` | as M2, plus an identity break for returning teams | fitted and scored | sensitivity of the returning-team rule |

The decomposition is descriptive:

- M1 − M0 on the common group: the cost or benefit, for already-scorable matches, of letting promoted teams' matches
  into the fit.
- M2 − M1: the continuing-team shrinkage.
- S1 − M2 on returning-team matches.

No other arm is part of this protocol.

## 8. Specification control (section I)

| Choice | Status |
|---|---|
| H = 730, exponential weights, online cadence, strictly-earlier information, independent Poisson, 10-goal grid, no warm start | **inherited, locked** (Experiments 12-13) |
| Gaussian priors; groups by S − 1 membership; EB from single-season fits on seasons 2015-16 … S − 1; pooled deconvolved variance; floor 0.05; centred continuing mean; sum-to-zero over target teams; weight reference = refit date; no prior on mu, h, history-only teams; returning rule; solver; arms; evidence rules | **new, fixed now** (this document) |
| M1, S1 | **sensitivity / ablation**, never promoted to primary because of their scores |

No value is tuned on a target season, so no selection procedure runs. Any later tunable value would need nested
forward-chaining selection on 2017-18 … 2021-22 with the one-SE rule (as in Experiment 12), registered first.

## 9. Evaluation (sections K, L)

- **Targets and groups.** Targets 2017-18 … 2021-22 (amendment A1) on dev_v2 cut to ≤ 2021-22. Groups and their
  registered sizes are in `[groups]`: full 380 per target, plus common, unseen, promoted, returning and continuing-only.
- **Metrics.** Log loss (primary) and Brier through the harness. Calibration in the large per outcome with
  date-clustered SEs. The Experiment 11 segments (0-9, 10-18, 19-28, 29+), because priors matter most early in a
  season.
- **Uncertainty.** Paired (left − right) differences with naive and date-clustered SEs (clusters (target, date)), and
  per-fold sign counts.
- **Diagnostic taxonomy.** The Experiment 7 taxonomy with yoyo_threshold = 2:
  - **recent yoyo:** ≤ 2 seasons out of the PL, counted from the fold history when the team is in it, otherwise from
    `team_history.csv`;
  - **long absence or newcomer:** more than 2 seasons out, or never in the PL;
  - **returning:** present in the fold history;
  - **continuing:** played the previous season.

  The group lists are registered in the config and are not revised after scoring.
- **Evidence rules (`[evidence]`).**
  - **COV:** 1,900/1,900 valid forecasts.
  - **NI (primary):** pooled common-group M2 − M0, with mean + 2 clustered SE < +0.002.
  - **Key secondary:** M2 vs online Elo and vs the frequency baseline on the 296 unseen-team matches, read with the
    Experiment 13 U rule.
  - **Context only:** M2 − Elo (full group) and M2 − market benchmark (Experiment 14), never selection evidence.
- **One season decides nothing.** Readings use the five folds pooled, with per-fold sign counts.

### Non-inferiority margin: δ = 0.002 log loss (rationale, fixed before any Experiment 15 output)

**Provenance.**

- The value is the project's *practical floor*, first fixed in the Experiment 12 pre-registration (`b94bba1`,
  `configs/time_weighted_poisson_v1.toml [evidence]`, "fixed before any result"), before any time-weighted model was
  fitted. It was reused unchanged as the floor of Experiment 13's criterion U.
- It was written into this protocol at registration (`5034f56`), when no Experiment 15 model, prediction or score
  existed (`[historical_locked] status = "not_run"`; no `results/full_coverage_poisson_*`).
- It was not chosen from, or adjusted to, any Experiment 15 target-season performance.
- When first registered it had **no written rationale**, and it was defined as a floor for calling a difference an
  *improvement*, not as a non-inferiority margin. The rationale below closes that gap without changing the value.

**Why this value, used this way, is appropriate.**

1. **One threshold for "practically meaningful", in both directions.** The project already treats a log-loss difference
   smaller than 0.002 as too small to count as an improvement (criteria D, V and U). Using the same δ as the largest
   acceptable degradation makes the rule symmetric: M2 may not be worse by an amount that would have counted as a real
   gain had the sign been reversed. Reusing the inherited value rather than choosing a new one also leaves no room to
   shop for a margin.
2. **Interpretable size.** A mean log-loss increase of δ means the probability given to the observed outcome falls by
   a factor exp(−0.002) ≈ 0.998, about 0.2% in relative terms, on average.
3. **Small relative to the comparator's own established benefit.**
   - **What M0's advantage is.** M0's defining feature is online refitting. Its recorded, locked historical gain over
     the frozen model on the same 1,604 common matches is −0.0117 log loss (Experiment 13, `[historical_locked]`).
   - **What the margin preserves.** δ is about 17% of that gain. Non-inferiority therefore guarantees that M2 keeps at
     least about 83% of the benefit that made M0 the comparator. This is the usual effect-preservation reading of a
     non-inferiority margin.
   - **Status of this check.** It is a check on the inherited value, not a derivation of it. It uses only a recorded,
     locked comparator result and no Experiment 15 output.
4. **Stated in advance: the test may be inconclusive.**
   - **The bar is strict.** The rule needs the upper 2-SE bound below +0.002, so M2 must be close to M0 with good
     precision.
   - **Why the SE is unknown.** Its size depends on how much M2 and M0 differ match by match, which is unknown before
     the run. For comparison, Experiment 13's online − frozen difference had a clustered SE of 0.0027 on these matches,
     but those arms differ far more per match.
   - **What happens if the SE is large.** The reading is "inconclusive_on_common_group". Neither δ nor the 2-SE
     multiple will be changed after the SE is seen.
5. **Scope.** δ applies to log loss, the primary metric. Brier differences are reported with their SEs but have no
   non-inferiority criterion.
- **After the run.** The historical results are then locked in `[historical_locked]`. A descriptive 2024-25 stage
  would need its own access-log entry; 2025-26 remains the sealed confirmation holdout.

## 10. Planned implementation and tests (section O)

Planned modules (not written yet):

- `eplmodel.models.full_coverage`: penalised objective, solver, and EB estimation;
- `eplmodel.evaluation` harness use only;
- `experiments/full_coverage_poisson.py`.

| Test | What it proves |
|---|---|
| Season-start no-leakage | Changing any target-season outcome leaves every first-date prediction and the EB values unchanged |
| EB uses only pre-target seasons | Altering seasons ≥ S changes nothing; EB refuses rows with season ≥ S; 2014-15 excluded from the EB set |
| Online no-leakage | A prediction on d is unchanged by any outcome dated ≥ d; earlier outcomes do change it (synthetic league, as in Experiment 13 tests) |
| Unseen-team initialisation | At the first date a never-seen team's effects equal (a_P, b_P) exactly (after constraint projection) |
| Returning-team initialisation | Its earlier matches enter with the decayed weights; under S1 they belong to a separate identity |
| Continuing initialisation | Prior mean equals the centred a_C; prior means sum to zero over T_S |
| Weights × priors | Doubling every weight equals halving tau^2 (scale identity); the weight reference is the refit date |
| tau → ∞ limit | Reproduces the unpenalised MLE; with unseen teams excluded it reproduces Experiment 13 online predictions within 1e-7 |
| Home advantage | Estimated from the fitting set only; a single global parameter; changes between refits only through new data |
| Identification | C1/C2 hold exactly at every optimum; the eliminated team's effects equal minus the sum of the other 19; history-only teams are free and unpenalised; the Hessian is negative definite on the constrained space; with no priors the predictions are invariant to the choice of eliminated team (shift invariance) |
| Existence checks | A history-only team with zero weighted goals scored (or conceded) aborts the fit before solving; a matchless target team without a prior is refused |
| Probability validity | Every row finite, ≥ 0 and summing to 1; (H, D, A) order |
| Full coverage | 380 forecasts per target; registered group sizes and online-fit counts hold before scoring |
| Deterministic refitting | Two runs give bit-identical predictions; no warm start |
| Solver | Converges on concave synthetic problems; a forced failure aborts with no fallback |
| Strict guards | Targets outside the selection set refused; holdout seasons refused; no market/xG input accepted |
| Protocol | The config agrees with the code and the Experiment 12 lock; arms not registered for dev-test, exposed-validation or holdout |

## 11. Amendments

### PA1 (2026-10-02): identifiability parameterisation and rationale of the inherited 0.002 criterion

Made after the registration commit `5034f56` and **before any implementation**. At the time of this amendment **no
Experiment 15 model had been implemented, fitted or scored, and no Experiment 15 prediction or result existed**: the
config's `[historical_locked]` status was `not_run`, and there was no `results/full_coverage_poisson_*` directory. No
market-benchmark output and no 2024-25 information were used.

| Item | Change | Effect on the frozen specification |
|---|---|---|
| Identifiability | The section "Parameterisation and identifiability" was added. It covers the team sets, the free parameter vector, the elimination of the alphabetically last target team, history-only teams as free and unpenalised, how mu is identified, the strict-concavity argument and the pre-fit existence checks. The config gains `eliminated_team`, `history_only_teams` and `[implementation_checks] existence_checks` | None: these make explicit the constraints already registered (sum to zero over the 20 target teams; no prior on history-only teams) |
| 0.002 criterion | The value is **inherited** from the earlier pre-registered criterion: the practical floor of `time_weighted_poisson_v1`, fixed in `b94bba1` before any time-weighted fit and reused by Experiment 13. It was registered here unchanged in `5034f56`. Its **written rationale is added now, by this amendment** (section 9, "Non-inferiority margin"). The config gains `non_inferiority_margin_log_loss = 0.002` and its source | None: the value 0.002, the 2-SE multiple and every reading are unchanged; the margin was not chosen from any target-season performance |

No other part of the protocol was changed. Recorded as an amendment of this pre-registration rather than in
`TEST_SET_ACCESS_LOG.md`, whose protocol-change table records changes to season roles or selection rules. This
amendment changes neither and accesses no data.
