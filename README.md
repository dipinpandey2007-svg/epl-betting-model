# EPL match-probability models

Research code for estimating English Premier League match-outcome probabilities, with an emphasis on leakage-safe,
chronological evaluation and proper scoring rules. The long-term question is whether genuine betting-market
inefficiencies can be identified; the project is building and validating baselines before attempting that.

> **Status: research in progress. Nothing here is betting advice.** No model has yet been compared with bookmaker
> prices, and none of the results below say anything about beating the market.

## Results so far

Trained on 2014-15 to 2021-22, scored on 2022-23 and 2023-24. Lower is better for both metrics.

| Model | Matches | Log loss | Brier |
|---|---|---|---|
| Frequency baseline (training H/D/A rates) | 760 | 1.0525 | 0.6355 |
| Elo (K=25) + multinomial logistic regression | 760 | 0.9527 | 0.5642 |
| Elo (same subset) | 648 | 0.9547 | 0.5661 |
| Static Poisson goal model | 648 | 1.0058 | 0.5989 |
| Staged Dixon-Coles (rho = -0.04) | 648 | 1.0072 | 0.5994 |

Read these with three caveats:

1. **The 2022-24 period is an exposed development benchmark, not a clean holdout.** It was evaluated several times
   during exploratory work ([access log](docs/TEST_SET_ACCESS_LOG.md)). The sealed final holdout is 2025-26
   ([holdout protocol](docs/HOLDOUT_PROTOCOL.md)); it has not been scored.
2. **648-match subset.** The static goal models cannot predict for teams absent from training (Nott'm Forest, Luton),
   so 112 matches are excluded from the goal-model comparison.
3. **Elo vs Poisson is not a like-for-like comparison.** Elo updates its ratings through the test seasons; the
   Poisson model is static and assumes constant team strength over 2014-2022. The gap mixes model family with update
   dynamics, so it does not show that rating models are intrinsically better than goal models.

## Methods in brief

- **Chronological splits.** Hyperparameters are chosen on expanding-window walk-forward folds inside the training
  seasons. Every prediction uses only information available before kickoff.
- **Elo.** Ratings updated after each match (K = 25, chosen by walk-forward validation; no home-advantage term in the
  update). A multinomial logistic regression maps the pre-match rating difference to P(home win, draw, away win).
- **Poisson.** A GLM `Goals ~ Team + Opponent + IsHome` gives expected goals for each side. Independent Poisson
  scoreline probabilities are computed on a 0-10 goal grid and summed into H/D/A.
- **Dixon-Coles.** A low-score correction (rho) applied to the Poisson grid. It is estimated in a *staged* way, with the
  Poisson coefficients held fixed, not by joint maximum likelihood.
- **Scoring.** Log loss and the multiclass Brier score (summed over outcomes), plus reliability tables.

Full rules: [docs/METHODOLOGY.md](docs/METHODOLOGY.md). Every experiment: [docs/RESULTS_LOG.md](docs/RESULTS_LOG.md).

## Quickstart

Requires Python 3.11+. The recorded results were produced with Python 3.13 on Windows using the pinned versions in
`requirements-lock.txt`. The full test suite, including golden tests, has also been run against the minimum dependency
versions listed in `pyproject.toml`. CI runs the unit tests on Linux and Windows for Python 3.11-3.13.

The download step fetches ten third-party files from football-data.co.uk. **Check Football-Data's current terms before
running it** (see [Data](#data) below), and please don't run it repeatedly.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"                     # or: pip install -r requirements-lock.txt && pip install -e . --no-deps

# Check that both paths are inside this repository (not another project's environment):
python -c "import sys, eplmodel; print(sys.executable); print(eplmodel.__file__)"

python -m eplmodel.data.download            # fetch season CSVs from football-data.co.uk into data/raw/
python -m eplmodel.data.build               # build + validate data/processed/matches.csv, verify checksum

python -m experiments.run_all               # reproduce every recorded result into results/
python -m pytest                            # unit + golden regression tests (~5 s)
python -m pytest -m "not golden"            # unit tests only (no data needed)
```

## Repository layout

```
src/eplmodel/          reusable library code
  data/                download, build, load, validate, checksums
  models/              elo, poisson, scoreline, dixon_coles
  evaluation/          metrics, calibration, baselines, alignment, walk_forward
  analysis/            promoted/unseen-team fold analysis
  reporting/           results files with provenance, figures
  splits.py            season splits and development-test guards
experiments/           one script per recorded experiment (python -m experiments.<name>)
configs/               frozen model specifications
results/               metrics.json + figures written by experiments
tests/                 unit tests and golden regression tests
data/                  see data/README.md (match data are downloaded, not committed)
docs/                  methodology, results log, project state, access log, roadmap
archive/exploratory/   the original exploratory scripts, kept for reference
```

## Reproducibility

- `data/checksums.json` records the exact dataset behind every result. The build step checks it.
- Each `results/<experiment>/metrics.json` records the git commit, data checksum and package versions.
- `tests/test_golden_results.py` reproduces every number in the table above. Metrics must match within 1e-9 and
  log-likelihoods within 1e-6; the current code reproduces them exactly.
- `requirements-lock.txt` pins the environment used to produce the results.
- Re-running `python -m experiments.run_all` rewrites the tracked `results/*/metrics.json` files. If the code and data
  are unchanged, only the provenance fields (`git_commit`, `git_dirty`) differ.

## Known limitations

- Promoted and unseen teams are not handled properly yet. Goal models exclude them; Elo starts them at the league
  average.
- Team strength is static in the goal models.
- Dixon-Coles is staged, not jointly estimated.
- No uncertainty estimates on differences between models.
- Only match results are used so far.
- The final holdout (2025-26) is sealed by protocol but has not been downloaded or scored yet.

See [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) for the full list and the next milestone, and
[docs/ROADMAP.md](docs/ROADMAP.md) for the planned stages (xG, market odds, calibration, CLV) and how they will plug in.

## Data

- **Match data:** [football-data.co.uk](https://www.football-data.co.uk/), English Premier League (`E0`) season files.
  These are third-party data. They are **not included** in this repository and **not covered by this project's MIT
  licence**. **Before running the download step, read Football-Data's current terms and conditions on their website
  and make sure your intended use complies with them.** This project does not grant any rights to that data. See
  [data/README.md](data/README.md).
- **Reference data:** `data/reference/team_history.csv` is compiled by this project. Each row cites its sources
  (Wikipedia's "List of Premier League clubs" and the relevant season article).

## References

- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present.* Arco.
- Maher, M. J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109-118.
  doi:[10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x)
- Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football
  betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265-280.
  doi:[10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065)

## Citing this project

If you use this code, please cite it using [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## Author

Dipin. Issues and questions are welcome through the repository's issue tracker.

## AI-assisted development

Parts of this codebase and its documentation were written with the help of an AI coding assistant (Claude Code). The
author directs the research and is responsible for the methodology, the results and their interpretation. Every
reported number is reproduced by the golden regression tests.

## Licence

| What | Licence / terms |
|---|---|
| This project's code, tests, configs, documentation, results files and `data/reference/` table | [MIT Licence](LICENSE) |
| Football-Data match files (`data/raw/`, and `data/processed/` derived from them; downloaded, never committed) | Football-Data's own terms. Not covered by the MIT Licence; check their current terms before obtaining the data |
