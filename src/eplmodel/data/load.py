"""Load the processed match table with consistent dtypes."""

from pathlib import Path

import pandas as pd

from eplmodel.data.validate import validate_matches
from eplmodel.paths import PROCESSED_MATCHES


def load_matches(path: Path = PROCESSED_MATCHES, validate: bool = True) -> pd.DataFrame:
    """Load matches with Date as datetime, Season as str (e.g. "1415") and goals as int.

    Adds `match_id` ("YYYY-MM-DD_Home_Away"), a stable key used to align
    predictions from different models instead of relying on row positions.
    The processed CSV stores goals as floats (an artefact of the NaN row in
    the raw data); they are cast to int here, which does not change any result.
    """
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
    if validate:
        validate_matches(df)
    return df
