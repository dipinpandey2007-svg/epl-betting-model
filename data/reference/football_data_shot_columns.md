# Source record: football-data.co.uk shot columns

| | |
|---|---|
| Source | football-data.co.uk, "Notes for Football Data", <https://www.football-data.co.uk/notes.txt> |
| File | the copy fetched on 2026-10-02 for `football_data_odds_columns.md`, SHA-256 `6ecd41a98ad2751372817e7e6f1709bfeb433c53dd9aeda330fd926a5471452d` (re-verified; not re-fetched) |
| Used by | `configs/shots_information_v1.toml` (Experiment 16, pre-registered) |

The notes are third-party text and are not copied into the repository. These are the lines that define the columns
(quoted verbatim):

> HS = Home Team Shots
> AS = Away Team Shots
> HST = Home Team Shots on Target
> AST = Away Team Shots on Target

On sources, the notes say "Match statistics: BBC, Flashscore, ESPN Soccer, Bundesliga.de, Gazzetta.it and Football.fr".

## What the notes do not define

They do not say:

- whether blocked shots count as shots;
- whether goals count as shots on target;
- how own goals or penalties are counted;
- which of the listed providers supplied a given season.

The columns are therefore used only as the counts the files contain. No finer definition is assumed.

## Verified in the project's raw files

**Headers.** `HS`, `AS`, `HST` and `AST` are present in every development season file from 2014-15 to 2024-25.
`HHW`/`AHW` (woodwork) and `HO`/`AO` (offsides) are absent. `HF`, `AF`, `HC`, `AC`, `HY`, `AY`, `HR` and `AR` are
present but are not shot fields.

**Values, 2014-15 to 2021-22 only (3,040 matches).** The values of later seasons were not read.

- No missing, non-numeric, non-integer or negative value in any of the four columns.
- Goals ≤ shots in every match.
- Goals > shots on target in 18 matches (0.6%; one to three per season). This is consistent with shots on target
  generally including goals, with exceptions such as own goals; the source does not say.
- Shots on target > shots in 1 match (2021-22), an inconsistency in the source data. Each arm uses one variable only,
  so the value is kept as recorded.
