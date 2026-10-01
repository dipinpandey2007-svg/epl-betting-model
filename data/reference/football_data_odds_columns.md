# Source record: meaning of the football-data.co.uk Pinnacle 1X2 odds columns

| | |
|---|---|
| Source | football-data.co.uk, "Notes for Football Data", <https://www.football-data.co.uk/notes.txt> |
| Accessed | 2026-10-02 (HTTP 200, `text/plain`, 7,686 bytes) |
| SHA-256 of the file as fetched | `6ecd41a98ad2751372817e7e6f1709bfeb433c53dd9aeda330fd926a5471452d` |
| Used by | `configs/market_benchmark_v1.toml`, `eplmodel.market.odds` |

The file itself is third-party text and is not copied into the repository. The lines below are quoted verbatim
because they define the columns the market benchmark uses.

> The following key to betting odds data is described below. These are for pre-closing odds. For the closing odds,
> as below but with an additional "C" character following the bookmaker abbreviation/Max/Avg (e.g. B365CH = closing
> Bet365 home win odds).

> PSH and PH = Pinnacle home win odds
> PSD and PD = Pinnacle draw odds
> PSA and PA = Pinnacle away win odds

> Betting odds for weekend games are collected Friday afternoons, and on Tuesday afternoons for midweek games.

## What this establishes

| Columns | Meaning | Snapshot name used here |
|---|---|---|
| `PSH`, `PSD`, `PSA` | Pinnacle home / draw / away decimal odds, **pre-closing** | `pre_closing` |
| `PSCH`, `PSCD`, `PSCA` | Pinnacle home / draw / away decimal odds, **closing** (`C` suffix) | `closing` |

## What it does not establish

- The notes do not say when the "closing" odds were taken; they are therefore treated only as the last market prices
  before kickoff, with no exact timestamp.
- The collection-time sentence describes the current practice of the site. It does not show, match by match, when
  the pre-closing prices in the historical files were collected (for example for rescheduled matches). No
  information cutoff is inferred from it.
- The columns are not "opening" odds; the project calls them pre-closing.
- The notes are the site's current version. The historical files may have been compiled under earlier wording.

## Verified in the project's raw files (headers and values, no result column read)

`PSH/PSD/PSA` and `PSCH/PSCD/PSCA` are present in every development season file 2014-15 … 2024-25; the aliases
`PH/PD/PA` and `PCH/PCD/PCA` are absent. Values were read for 2014-15 … 2021-22 only (380 rows each, no missing or
non-numeric price, every price > 1.0, book sum 1/H + 1/D + 1/A between 1.0005 and 1.0591, never below 1).
