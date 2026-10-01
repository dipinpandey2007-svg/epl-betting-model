import inspect

import pandas as pd
import pytest

from eplmodel.data import download
from eplmodel.data.build import build_matches
from eplmodel.data.checksums import content_sha256
from eplmodel.data.load import load_matches
from eplmodel.data.validate import DataValidationError, validate_matches
from eplmodel.paths import RAW_DIR
from eplmodel.splits import DATASET_V1_SEASONS, RESERVED_HOLDOUT_SEASONS, HoldoutAccessError


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


# --- Holdout guards in the data pipeline (docs/HOLDOUT_PROTOCOL.md) ----------------

def _write_raw(raw_dir, season, home="A", away="B"):
    pd.DataFrame({"Date": ["01/08/2020"], "HomeTeam": [home], "AwayTeam": [away],
                  "FTHG": [1], "FTAG": [0], "FTR": ["H"]}).to_csv(raw_dir / f"E0_{season}.csv", index=False)


def test_default_download_and_build_cover_only_the_recorded_dataset():
    for func in (download.download_all, build_matches):
        default = inspect.signature(func).parameters[
            "season_codes" if func is download.download_all else "seasons"].default
        assert tuple(default) == DATASET_V1_SEASONS


def test_build_reads_only_the_requested_seasons(tmp_path):
    _write_raw(tmp_path, "1415")
    _write_raw(tmp_path, "2526", home="Sealed", away="Holdout")  # e.g. a holdout file placed here by mistake
    built = build_matches(tmp_path, verbose=False, seasons=["1415"])
    assert list(built["Season"]) == ["1415"]
    assert "Sealed" not in set(built["HomeTeam"])


@pytest.mark.parametrize("season", RESERVED_HOLDOUT_SEASONS)
def test_build_refuses_holdout_seasons(tmp_path, season):
    _write_raw(tmp_path, season)
    with pytest.raises(HoldoutAccessError):
        build_matches(tmp_path, verbose=False, seasons=["1415", season])


def test_build_fails_on_missing_requested_season(tmp_path):
    _write_raw(tmp_path, "1415")
    with pytest.raises(FileNotFoundError, match="E0_1516"):
        build_matches(tmp_path, verbose=False, seasons=["1415", "1516"])


@pytest.mark.parametrize("season", RESERVED_HOLDOUT_SEASONS)
def test_download_refuses_holdout_seasons_before_any_request(tmp_path, monkeypatch, season):
    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(download.pd, "read_csv", no_network)
    with pytest.raises(HoldoutAccessError):
        download.download_season(season, tmp_path)
    with pytest.raises(HoldoutAccessError):
        download.download_all(["2425", season], tmp_path, pause=0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("season", RESERVED_HOLDOUT_SEASONS)
def test_development_loader_refuses_holdout_rows(tmp_path, season):
    path = tmp_path / "matches.csv"
    pd.DataFrame({"Date": ["2020-08-01", "2025-08-16"], "HomeTeam": ["A", "C"], "AwayTeam": ["B", "D"],
                  "FTHG": [1.0, 2.0], "FTAG": [0.0, 2.0], "FTR": ["H", "D"],
                  "Season": ["2021", season]}).to_csv(path, index=False)
    with pytest.raises(HoldoutAccessError):
        load_matches(path, validate=False)
