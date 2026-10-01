"""Build the processed match table from raw season CSVs.

Behaviour preserved exactly from the original build_dataset.py so that the
output is identical to the dataset used for every recorded result:

- dates parsed day-first with mixed formats (the raw files mix 2- and 4-digit years);
- only Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR are kept;
- rows missing any of these are dropped (one blank trailing row in E0_1415.csv);
- Season is taken from the file name (e.g. "1415");
- rows are sorted by Date with pandas' default sort.

The seasons are given explicitly; files for other seasons in the raw
directory are ignored, and holdout seasons are refused
(docs/HOLDOUT_PROTOCOL.md). Two named datasets exist:

- v1 (default): 1415..2324 -> data/processed/matches.csv, behind every recorded result;
- dev_v2: 1415..2425 (adds the 2024-25 validation season) -> data/processed/matches_dev_v2.csv.

Within a single date the row order is arbitrary, which is harmless here: no
team plays twice on one date, so sequential Elo updates do not depend on it
(validate_matches checks this).

Usage:  python -m eplmodel.data.build [--dataset v1|dev_v2]
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from eplmodel.data.checksums import verify_file
from eplmodel.data.download import LEAGUE, raw_season_path
from eplmodel.data.validate import validate_matches, validate_season_dates
from eplmodel.paths import PROCESSED_DEV_V2, PROCESSED_MATCHES, RAW_DIR
from eplmodel.splits import DATASET_DEV_V2_SEASONS, DATASET_V1_SEASONS, assert_not_holdout

ESSENTIAL_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Named datasets: seasons and canonical output file.
DATASETS = {
    "v1": (DATASET_V1_SEASONS, PROCESSED_MATCHES),
    "dev_v2": (DATASET_DEV_V2_SEASONS, PROCESSED_DEV_V2),
}
# Datasets whose recorded checksum must exist; the build fails if it is missing.
CHECKSUM_REQUIRED = frozenset({"v1"})


class DatasetTargetError(ValueError):
    """The requested seasons would be written to another dataset's canonical file."""


def canonical_dataset(output: Path) -> str | None:
    """Name of the dataset whose canonical file `output` is, or None for any other path."""
    for name, (_, path) in DATASETS.items():
        if Path(output).resolve() == path.resolve():
            return name
    return None


def check_output_target(seasons: Sequence[str], output: Path) -> None:
    """Refuse to write anything but a dataset's own seasons to that dataset's canonical file.

    This stops, for example, dev_v2 seasons from replacing data/processed/matches.csv
    (dataset v1, behind every recorded result). Other output paths are unrestricted.
    """
    name = canonical_dataset(output)
    if name is not None and sorted(seasons) != sorted(DATASETS[name][0]):
        raise DatasetTargetError(
            f"{Path(output).name} is the canonical file of dataset {name!r} ({DATASETS[name][0][0]}.."
            f"{DATASETS[name][0][-1]}); refusing to write seasons {sorted(seasons)} to it."
        )


def load_and_clean_season(path: Path) -> tuple[pd.DataFrame, int]:
    """Return the essential columns of one raw season file and the number of incomplete rows dropped."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df[ESSENTIAL_COLUMNS].copy()
    before = len(df)
    df = df.dropna(subset=ESSENTIAL_COLUMNS)
    return df, before - len(df)


def build_matches(
    raw_dir: Path = RAW_DIR,
    league: str = LEAGUE,
    verbose: bool = True,
    seasons: Sequence[str] = DATASET_V1_SEASONS,
) -> pd.DataFrame:
    """Combine the raw files of exactly `seasons`; other files in `raw_dir` are ignored.

    Holdout seasons are refused, so a holdout file placed in the raw directory
    by mistake can never enter a development dataset. Every match must lie in
    its season's date window (validate_season_dates).
    """
    assert_not_holdout(seasons)
    files = [raw_season_path(s, raw_dir, league) for s in sorted(seasons)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Raw files {missing} not found in {raw_dir}; "
                                "run `python -m eplmodel.data.download` first.")
    frames = []
    for f in files:
        season_df, dropped = load_and_clean_season(f)
        if dropped and verbose:
            print(f"{f.name}: dropped {dropped} incomplete row(s)")
        season_df["Season"] = f.stem.split(f"{league}_")[1]
        frames.append(season_df)
    matches = pd.concat(frames, ignore_index=True)
    validate_season_dates(matches)
    return matches.sort_values("Date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a processed match table from raw season files.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="v1",
                        help="v1: 1415..2324 -> matches.csv (default); dev_v2: 1415..2425 -> matches_dev_v2.csv")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, help="override the dataset's output file")
    parser.add_argument("--seasons", nargs="+", help="override the dataset's season codes")
    args = parser.parse_args()
    default_seasons, default_output = DATASETS[args.dataset]
    args.seasons = args.seasons or list(default_seasons)
    args.output = args.output or default_output
    check_output_target(args.seasons, args.output)  # before anything is read or written

    matches = build_matches(args.raw_dir, seasons=args.seasons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(f"Saved {len(matches)} rows to {args.output}")

    from eplmodel.data.load import load_matches
    validate_matches(load_matches(args.output, validate=False))
    print("Validation passed.")

    report_checksum(args.output)


def report_checksum(output: Path) -> None:
    """Compare a canonical dataset file with data/checksums.json and print the outcome.

    A missing record raises KeyError for datasets in CHECKSUM_REQUIRED (v1);
    for other datasets it is reported, since a new dataset has no record until
    it is first built.
    """
    name = canonical_dataset(output)
    if name is None:
        return
    try:
        matches_record = verify_file(output)
    except KeyError:
        if name in CHECKSUM_REQUIRED:
            raise
        print(f"No checksum recorded yet for {Path(output).name} in data/checksums.json.")
        return
    if matches_record:
        print(f"Checksum matches data/checksums.json: this is the recorded {Path(output).name}.")
    else:
        print("WARNING: checksum differs from data/checksums.json. Recorded results may not reproduce "
              "(upstream data may have been revised).")


if __name__ == "__main__":
    main()
