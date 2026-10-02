# Roadmap and extension points

Status of the 13 stages in `PROJECT_OVERVIEW.md`. Only stages marked **done** have code. Later stages are
deliberately *not* stubbed out: each will be added when real data and a validation plan exist.

| # | Stage | Status | Where it lives / will live |
|---|---|---|---|
| 1 | Historical data pipeline | **done** (results only, 2014-24) | `eplmodel.data` |
| 2 | Elo baseline | **done** (`elo_k25_logreg_v1`) | `eplmodel.models.elo` |
| 3 | Poisson / Dixon-Coles | **full-coverage dynamic Poisson run** (Experiment 15, `full_coverage_poisson_v1`: 100% coverage; common-group non-inferiority inconclusive; historical results locked; descriptive 2024-25 validation V5 recorded, not selection evidence). **done**: static Poisson and staged DC; time-weighted static Poisson candidate `poisson_time_weighted_v1` (Experiment 12, H* = 730 days locked; 2024-25-specific observation that cannot confirm); online refitted diagnostic arm (Experiment 13: in-season refitting helps historically, same sign in 2024-25; diagnostic only, no spec adopted). Not yet: an adopted dynamic Poisson spec, joint DC MLE | `eplmodel.models.poisson`, `.dixon_coles`, `.scoreline` |
| 4 | xG and team-performance features | **shots tested** (Experiment 16, `shots_information_v1`: historical stage locked, ω\* = 0, reading `no_distinguishable_shot_information`); xG not started | new data source → `eplmodel.data`; features → a future `eplmodel.features` package |
| 5 | Squad / injury / lineup information | not started | needs a timestamped data source (must be known before kickoff) |
| 6 | Tactical / contextual features | not started | `eplmodel.features` |
| 7 | Market odds benchmark | **done** (Experiment 14, `market_benchmark_v1`): Pinnacle closing odds with Shin margin removal as primary; pre-closing snapshot and proportional/power methods as registered sensitivity arms; historical folds only | `eplmodel.market` |
| 8 | Machine-learning models | not started | `eplmodel.models`; must beat stages 2-3 on the same folds |
| 9 | Probability calibration | partial: home-win reliability table only | `eplmodel.evaluation.calibration` |
| 10 | Walk-forward / backtesting | **common evaluation harness done** (2026-10-02): folds and information sets, forecast frames, scoring with clustered SEs, reproduction gate for Experiments 10-13 | `eplmodel.evaluation.folds`, `.forecasts`, `.scoring`, `.segments`, `.reproduction` (see METHODOLOGY §9) |
| 11 | Closing-line value | not started | needs a separately defined pre-match snapshot and an explicit per-prediction information cutoff (not established by the current data) |
| 12 | ROI and proper scoring metrics | partial: log loss and Brier done; ROI not started | `eplmodel.evaluation.metrics` |
| 13 | Structured football knowledge | started: `data/reference/team_history.csv` | `data/reference/` |

## Conventions any new stage must follow

- **Pre-kickoff information only.** Every feature must carry the time it became known. Post-match statistics (shots,
  xG) for a match may only be used for *later* matches.
- **Same interface for every model.** A model produces an (n, 3) array of (H, D, A) probabilities for a set of
  fixtures, keyed by `match_id`, so it can be scored by `eplmodel.evaluation.metrics` and compared on the same matches.
- **Selection on the historical folds only.** New selection work must use `eplmodel.splits.selection_folds()`
  (targets 2017-18 … 2021-22, `SELECTION_TARGET_SEASONS`) and the guard `eplmodel.splits.assert_selection_target`.
  2024-25 was scored in four pre-registered runs (Experiments 10, 11, 12 and 13) and is retired as a selection target
  (amendment A1, [HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md) §9): it is history only for new work, and its results must
  not be used to tune the established specifications, to retune the Experiment 12 candidate (its half-life stays 730
  days) or to select or confirm anything. The development test benchmark and 2024-25 may only be re-scored for
  registered specifications; the 2025-26 holdout is sealed ([HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md)).
- **Each experiment** gets a script in `experiments/`, a results folder, a `RESULTS_LOG.md` entry and, once frozen, a
  spec id in `configs/` and golden values in the tests.

## Notes for stage 7 (market odds), as implemented

- The raw football-data files already contain opening and closing odds from several bookmakers (for example
  `B365H/D/A`, `PSH/D/A`, `PSCH/D/A`, and market averages and maxima). The column set changes between seasons
  (62-106 columns).
- The data build would add a *separate* odds table rather than widening `matches.csv`, so the dataset behind the
  recorded results stays unchanged.
- Implied probabilities need a documented margin-removal method (for example proportional or Shin), chosen before
  any comparison.
- Opening and closing prices must not be mixed. Using closing odds as a *feature* for a pre-kickoff prediction made
  earlier would be leakage.
- Implemented in Experiment 14. The meaning of the columns was checked against the site's notes
  (`data/reference/football_data_odds_columns.md`): `PSH/PSD/PSA` are **pre-closing**, not opening. The odds are read
  directly from the raw files, keyed by `match_id`, so `matches.csv` is unchanged. Proportional, power and Shin
  methods are all implemented; Shin was fixed in advance as primary.
