"""Load the processed match table with consistent dtypes."""

from pathlib import Path

import pandas as pd

from eplmodel.data.validate import DataValidationError, validate_matches, validate_season_dates
from eplmodel.paths import PROCESSED_DEV_V2, PROCESSED_MATCHES
from eplmodel.splits import DATASET_DEV_V2_SEASONS, assert_not_holdout


def load_matches(path: Path = PROCESSED_MATCHES, validate: bool = True) -> pd.DataFrame:
    """Load matches with Date as datetime, Season as str (e.g. "1415") and goals as int.

    Adds `match_id` ("YYYY-MM-DD_Home_Away"), a stable key used to align
    predictions from different models instead of relying on row positions.
    The processed CSV stores goals as floats (an artefact of the NaN row in
    the raw data); they are cast to int here, which does not change any result.

    This is the loader for development work: it raises HoldoutAccessError if
    the file contains any sealed holdout season. Holdout data are loaded only
    through eplmodel.holdout.load_holdout_matches.
    """
    df = _read_matches(path)
    assert_not_holdout(df["Season"].unique())
    if validate:
        validate_matches(df)
    return df


def load_dev_matches(path: Path = PROCESSED_DEV_V2, validate: bool = True) -> pd.DataFrame:
    """Load the development dataset dev_v2 (2014-15 .. 2024-25, including the 2024-25 validation season).

    Raises HoldoutAccessError if the file holds any holdout season, and
    DataValidationError unless it holds exactly the dev_v2 seasons, each match
    dated inside its season's window (so no later-season rows can hide under a
    development label).
    """
    df = load_matches(path, validate=validate)
    seasons = tuple(sorted(df["Season"].unique()))
    if seasons != DATASET_DEV_V2_SEASONS:
        raise DataValidationError(f"{Path(path).name} holds seasons {list(seasons)}, "
                                  f"expected {list(DATASET_DEV_V2_SEASONS)}")
    validate_season_dates(df)
    return df


def _read_matches(path: Path) -> pd.DataFrame:
    """Read and type a processed match file WITHOUT the holdout guard. Internal: use load_matches."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m eplmodel.data.download` then `python -m eplmodel.data.build`."
        )
    df = pd.read_csv(path, dtype={"Season": str})
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["match_id"] = df["Date"].dt.strftime("%Y-%m-%d") + "_" + df["HomeTeam"] + "_" + df["AwayTeam"]
    return df
