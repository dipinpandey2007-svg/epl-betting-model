# Data

| Path | Committed? | Contents |
|---|---|---|
| `raw/E0_<season>.csv` | no | Season files from football-data.co.uk (results, match statistics, bookmaker odds) |
| `processed/matches.csv` | no | 3,800 matches: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Season |
| `reference/team_history.csv` | yes | Hand-compiled Premier League history of teams that join the league in the dataset window |
| `checksums.json` | yes | Content SHA-256 of the files behind all recorded results |
| `holdout/` | no | Sealed final-holdout data (2025-26), never in `raw/` or `processed/`; see [HOLDOUT_PROTOCOL.md](../docs/HOLDOUT_PROTOCOL.md). Not acquired yet |

## Why the match data are not in the repository

The match files are third-party data from [football-data.co.uk](https://www.football-data.co.uk/). This repository
never contains them: the CSVs are git-ignored and were left out of the public repository.

| Content | Licence / terms |
|---|---|
| `reference/team_history.csv`, `checksums.json`, this README | this project's [MIT Licence](../LICENSE) |
| `raw/*.csv` (Football-Data files) and `processed/matches.csv` (derived from them) | **Football-Data's own terms.** Not covered by the MIT Licence; this project grants no rights to them |

**Before downloading, read Football-Data's current terms and conditions on their website and make sure your intended
use complies with them.** You are responsible for that compliance. The terms may change, so check the current version
rather than relying on any summary. Then recreate the files with:

```bash
python -m eplmodel.data.download   # skips files that already exist; please don't run it repeatedly
python -m eplmodel.data.build      # validates the result and checks it against checksums.json
```

The download requests ten files, pausing one second between them, with up to three attempts per file. The site can
rate-limit repeated downloads (HTTP 429); if that happens, wait and run the command again, since files already
downloaded are skipped.

The published repository is a fresh export of a private development history; that private history is the only place
copies of the CSVs were ever committed.

`reference/team_history.csv` is this project's own compilation of facts. Each row cites its sources (Wikipedia's
"List of Premier League clubs" and the relevant season article, accessed 2026-10-01) and is covered by the MIT licence.

## Checksums

Hashes are computed after converting CRLF to LF line endings, so they are the same on every platform.
`processed/matches.csv` is the authoritative check. Raw-file hashes are for information only: the download
re-writes files through pandas, so raw bytes can change with pandas version or upstream edits even when the content
is the same. A fresh download on 2026-10-01 rebuilt a byte-identical processed file. Nine of ten raw files were also
identical; `E0_2324.csv` differed only in formatting (same columns and values).

## Processed schema

| Column | Type after `load_matches()` | Notes |
|---|---|---|
| Date | datetime | match date (no kickoff time) |
| HomeTeam, AwayTeam | str | football-data team names |
| FTHG, FTAG | int | full-time goals (stored as floats in the CSV for historical reasons) |
| FTR | str | H / D / A |
| Season | str | e.g. `"1415"` |
| match_id | str | added on load: `YYYY-MM-DD_Home_Away` |
