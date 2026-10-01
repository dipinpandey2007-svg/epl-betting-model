"""Build the processed match table from raw season CSVs.

Behaviour preserved exactly from the original build_dataset.py so that the
output is identical to the dataset used for every recorded result:

- dates parsed day-first with mixed formats (the raw files mix 2- and 4-digit years);
- only Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR are kept;
- rows missing any of these are dropped (one blank trailing row in E0_1415.csv);
- Season is taken from the file name (e.g. "1415");
- rows are sorted by Date with pandas' default sort.

Within a single date the row order is arbitrary, which is harmless here: no
team plays twice on one date, so sequential Elo updates do not depend on it
(validate_matches checks this).

Usage:  python -m eplmodel.data.build
"""

import argparse
from pathlib import Path

import pandas as pd

from eplmodel.data.checksums import verify_file
from eplmodel.data.download import LEAGUE
from eplmodel.data.validate import validate_matches
from eplmodel.paths import PROCESSED_MATCHES, RAW_DIR

ESSENTIAL_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def load_and_clean_season(path: Path) -> tuple[pd.DataFrame, int]:
    """Return the essential columns of one raw season file and the number of incomplete rows dropped."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df[ESSENTIAL_COLUMNS].copy()
    before = len(df)
    df = df.dropna(subset=ESSENTIAL_COLUMNS)
    return df, before - len(df)


def build_matches(raw_dir: Path = RAW_DIR, league: str = LEAGUE, verbose: bool = True) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob(f"{league}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw files found in {raw_dir}; run `python -m eplmodel.data.download` first.")
    seasons = []
    for f in files:
        season_df, dropped = load_and_clean_season(f)
        if dropped and verbose:
            print(f"{f.name}: dropped {dropped} incomplete row(s)")
        season_df["Season"] = f.stem.split(f"{league}_")[1]
        seasons.append(season_df)
    matches = pd.concat(seasons, ignore_index=True)
    return matches.sort_values("Date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/processed/matches.csv from raw season files.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_MATCHES)
    args = parser.parse_args()

    matches = build_matches(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(f"Saved {len(matches)} rows to {args.output}")

    from eplmodel.data.load import load_matches
    validate_matches(load_matches(args.output, validate=False))
    print("Validation passed.")

    if args.output.resolve() == PROCESSED_MATCHES.resolve():
        if verify_file(args.output):
            print("Checksum matches data/checksums.json: this is the dataset used for all recorded results.")
        else:
            print("WARNING: checksum differs from data/checksums.json. Recorded results may not reproduce "
                  "(upstream data may have been revised).")


if __name__ == "__main__":
    main()
