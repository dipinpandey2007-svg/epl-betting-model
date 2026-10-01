import pandas as pd
import pytest

from eplmodel.data.build import build_matches
from eplmodel.data.checksums import content_sha256
from eplmodel.data.validate import DataValidationError, validate_matches
from eplmodel.paths import RAW_DIR


def test_processed_dataset_passes_validation(matches):
    validate_matches(matches)
    assert len(matches) == 3800
    assert matches["FTHG"].dtype.kind == "i"
    assert matches["match_id"].is_unique


def test_build_reproduces_processed_dataset(tmp_path, golden):
    if not list(RAW_DIR.glob("E0_*.csv")):
        pytest.skip("raw data missing: run `python -m eplmodel.data.download`")
    out = tmp_path / "matches.csv"
    build_matches(RAW_DIR, verbose=False).to_csv(out, index=False)
    assert content_sha256(out) == golden["data_sha256"]


def _frame(**overrides):
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "HomeTeam": ["A", "C"], "AwayTeam": ["B", "D"],
        "FTHG": [1, 0], "FTAG": [0, 0], "FTR": ["H", "D"], "Season": ["x", "x"],
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def test_validation_accepts_clean_frame():
    validate_matches(_frame(), matches_per_season=2)


@pytest.mark.parametrize("overrides,message", [
    ({"FTR": ["A", "D"]}, "disagrees"),
    ({"AwayTeam": ["A", "D"]}, "plays itself"),
    ({"Date": pd.to_datetime(["2020-01-02", "2020-01-01"])}, "chronological"),
    ({"Date": pd.to_datetime(["2020-01-01", "2020-01-01"]), "HomeTeam": ["A", "A"], "AwayTeam": ["B", "D"]},
     "more than once"),
])
def test_validation_catches_problems(overrides, message):
    with pytest.raises(DataValidationError, match=message):
        validate_matches(_frame(**overrides), matches_per_season=2)
