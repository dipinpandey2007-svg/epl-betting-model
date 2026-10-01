# Roadmap and extension points

Status of the 13 stages in `PROJECT_OVERVIEW.md`. Only stages marked **done** have code. Later stages are
deliberately *not* stubbed out: each will be added when real data and a validation plan exist.

| # | Stage | Status | Where it lives / will live |
|---|---|---|---|
| 1 | Historical data pipeline | **done** (results only, 2014-24) | `eplmodel.data` |
| 2 | Elo baseline | **done** (`elo_k25_logreg_v1`) | `eplmodel.models.elo` |
| 3 | Poisson / Dixon-Coles | **done**: static Poisson and staged DC. Not yet: dynamic Poisson, joint DC MLE | `eplmodel.models.poisson`, `.dixon_coles`, `.scoreline` |
| 4 | xG and team-performance features | not started | new data source → `eplmodel.data`; features → a future `eplmodel.features` package |
| 5 | Squad / injury / lineup information | not started | needs a timestamped data source (must be known before kickoff) |
| 6 | Tactical / contextual features | not started | `eplmodel.features` |
| 7 | Market odds benchmark | not started (the data exist in the raw files) | see below |
| 8 | Machine-learning models | not started | `eplmodel.models`; must beat stages 2-3 on the same folds |
| 9 | Probability calibration | partial: home-win reliability table only | `eplmodel.evaluation.calibration` |
| 10 | Walk-forward / backtesting | partial: folds for Elo K selection and the promoted-team analysis | `eplmodel.splits`, `eplmodel.evaluation.walk_forward` (next milestone: common harness) |
| 11 | Closing-line value | not started | needs stage 7 |
| 12 | ROI and proper scoring metrics | partial: log loss and Brier done; ROI not started | `eplmodel.evaluation.metrics` |
| 13 | Structured football knowledge | started: `data/reference/team_history.csv` | `data/reference/` |

## Conventions any new stage must follow

- **Pre-kickoff information only.** Every feature must carry the time it became known. Post-match statistics (shots,
  xG) for a match may only be used for *later* matches.
- **Same interface for every model.** A model produces an (n, 3) array of (H, D, A) probabilities for a set of
  fixtures, keyed by `match_id`, so it can be scored by `eplmodel.evaluation.metrics` and compared on the same matches.
- **Selection on the selection folds only.** Use `eplmodel.splits.selection_folds` (training folds plus 2024-25).
  The development test benchmark may only be re-scored for registered specifications; the 2025-26 holdout is sealed
  ([HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md)).
- **Each experiment** gets a script in `experiments/`, a results folder, a `RESULTS_LOG.md` entry and, once frozen, a
  spec id in `configs/` and golden values in the tests.

## Notes for stage 7 (market odds), when it starts

- The raw football-data files already contain opening and closing odds from several bookmakers (for example
  `B365H/D/A`, `PSH/D/A`, `PSCH/D/A`, and market averages and maxima). The column set changes between seasons
  (62-106 columns).
- The data build would add a *separate* odds table rather than widening `matches.csv`, so the dataset behind the
  recorded results stays unchanged.
- Implied probabilities need a documented margin-removal method (for example proportional or Shin), chosen before
  any comparison.
- Opening and closing prices must not be mixed. Using closing odds as a *feature* for a pre-kickoff prediction made
  earlier would be leakage.
