import inspect
import sys

import pandas as pd
import pytest

from eplmodel.data import download
from eplmodel.data import build as build_module
from eplmodel.data.build import DATASETS, DatasetTargetError, build_matches, check_output_target, report_checksum
from eplmodel.data.checksums import content_sha256, load_manifest
from eplmodel.data.load import load_dev_matches, load_matches
from eplmodel.data.validate import DataValidationError, season_window, validate_matches, validate_season_dates
from eplmodel.paths import PROCESSED_DEV_V2, PROCESSED_MATCHES, RAW_DIR
from eplmodel.splits import (
    DATASET_DEV_V2_SEASONS,
    DATASET_V1_SEASONS,
    DEVELOPMENT_SEASONS,
    RESERVED_HOLDOUT_SEASONS,
    HoldoutAccessError,
)


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

def _write_raw(raw_dir, season, home="A", away="B", date=None):
    date = date or f"01/09/20{season[:2]}"  # inside the season's window
    pd.DataFrame({"Date": [date], "HomeTeam": [home], "AwayTeam": [away],
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


# --- Development dataset dev_v2 (adds the 2024-25 validation season) ---------------

def test_named_datasets_never_include_holdout_seasons():
    assert DATASETS["v1"] == (DATASET_V1_SEASONS, PROCESSED_MATCHES)
    assert DATASETS["dev_v2"] == (DATASET_DEV_V2_SEASONS, PROCESSED_DEV_V2)
    assert DATASET_DEV_V2_SEASONS == DATASET_V1_SEASONS + ("2425",) == DEVELOPMENT_SEASONS
    for seasons, _ in DATASETS.values():
        assert set(seasons).isdisjoint(RESERVED_HOLDOUT_SEASONS)


@pytest.mark.parametrize("season,start,end", [
    ("1415", "2014-08-01", "2015-07-31"), ("1920", "2019-08-01", "2020-07-31"), ("2425", "2024-08-01", "2025-07-31"),
])
def test_season_window(season, start, end):
    assert season_window(season) == (pd.Timestamp(start), pd.Timestamp(end))


def test_season_windows_do_not_overlap():
    """Consecutive windows abut exactly, so every date belongs to one season only."""
    for this, nxt in zip(DATASET_DEV_V2_SEASONS, DATASET_DEV_V2_SEASONS[1:]):
        assert season_window(this)[1] + pd.Timedelta(days=1) == season_window(nxt)[0]


@pytest.mark.parametrize("date,ok", [
    ("2024-07-31", False),  # last day of 2023-24's window
    ("2024-08-01", True),   # first day of 2024-25's window
    ("2025-07-31", True),   # last day of 2024-25's window
    ("2025-08-01", False),  # first day of 2025-26's window
])
def test_season_window_boundaries(date, ok):
    df = pd.DataFrame({"Season": ["2425"], "Date": [pd.Timestamp(date)]})
    if ok:
        validate_season_dates(df)
    else:
        with pytest.raises(DataValidationError, match="outside 2024-08-01..2025-07-31"):
            validate_season_dates(df)


@pytest.mark.parametrize("season", ["x", "2426", "24-25", "242"])
def test_season_window_rejects_malformed_codes(season):
    with pytest.raises(DataValidationError):
        season_window(season)


def test_season_dates_catch_a_later_match_under_an_earlier_label():
    df = pd.DataFrame({"Season": ["2425", "2425"], "Date": pd.to_datetime(["2025-05-25", "2025-08-16"])})
    with pytest.raises(DataValidationError, match="1 2425 matches outside"):
        validate_season_dates(df)
    validate_season_dates(df.iloc[:1])


def test_build_rejects_a_raw_file_with_matches_from_another_season(tmp_path):
    _write_raw(tmp_path, "2425", date="16/08/2025")  # a 2025-26 date inside the 2024-25 file
    with pytest.raises(DataValidationError, match="2425 matches outside"):
        build_matches(tmp_path, verbose=False, seasons=["2425"])


# --- Canonical output files cannot be overwritten with another dataset's seasons ------

@pytest.mark.parametrize("seasons,output", [
    (DATASET_DEV_V2_SEASONS, PROCESSED_MATCHES),   # dev_v2 must never replace dataset v1
    (DATASET_V1_SEASONS, PROCESSED_DEV_V2),
    (DATASET_V1_SEASONS[:-1], PROCESSED_MATCHES),  # any other season set, too
])
def test_canonical_outputs_refuse_other_seasons(seasons, output):
    with pytest.raises(DatasetTargetError, match="canonical file"):
        check_output_target(seasons, output)


def test_canonical_outputs_accept_their_own_seasons_in_any_order(tmp_path):
    for seasons, output in DATASETS.values():
        check_output_target(list(reversed(seasons)), output)
    check_output_target(DATASET_DEV_V2_SEASONS, tmp_path / "scratch.csv")  # other paths are unrestricted


@pytest.mark.parametrize("argv", [
    ["--seasons", *DATASET_DEV_V2_SEASONS],                          # --dataset defaults to v1 -> matches.csv
    ["--dataset", "dev_v2", "--output", str(PROCESSED_MATCHES)],
    ["--dataset", "v1", "--output", str(PROCESSED_DEV_V2)],
])
def test_build_cli_refuses_before_reading_or_writing(monkeypatch, argv):
    def must_not_build(*args, **kwargs):
        raise AssertionError("build_matches was called")
    monkeypatch.setattr(build_module, "build_matches", must_not_build)
    monkeypatch.setattr(sys, "argv", ["build", *argv])
    before = {p: p.stat().st_mtime_ns for p in (PROCESSED_MATCHES, PROCESSED_DEV_V2) if p.exists()}
    with pytest.raises(DatasetTargetError):
        build_module.main()
    assert {p: p.stat().st_mtime_ns for p in before} == before


def test_missing_v1_checksum_fails_closed(monkeypatch, capsys):
    def no_record(path, *args, **kwargs):
        raise KeyError(f"No checksum recorded for {path}")
    monkeypatch.setattr(build_module, "verify_file", no_record)
    with pytest.raises(KeyError):
        report_checksum(PROCESSED_MATCHES)
    report_checksum(PROCESSED_DEV_V2)  # a new dataset may not have a record yet
    assert "No checksum recorded yet for matches_dev_v2.csv" in capsys.readouterr().out


def _write_processed(path, seasons):
    pd.DataFrame({"Date": [f"20{s[:2]}-09-01" for s in seasons], "HomeTeam": ["A"] * len(seasons),
                  "AwayTeam": ["B"] * len(seasons), "FTHG": 1.0, "FTAG": 0.0, "FTR": "H",
                  "Season": list(seasons)}).to_csv(path, index=False)


def test_dev_loader_requires_exactly_the_dev_v2_seasons(tmp_path):
    path = tmp_path / "dev.csv"
    _write_processed(path, DATASET_V1_SEASONS)  # 2024-25 missing
    with pytest.raises(DataValidationError, match="expected"):
        load_dev_matches(path, validate=False)
    _write_processed(path, DATASET_DEV_V2_SEASONS)
    assert len(load_dev_matches(path, validate=False)) == len(DATASET_DEV_V2_SEASONS)


@pytest.mark.parametrize("season", RESERVED_HOLDOUT_SEASONS)
def test_dev_loader_refuses_holdout_rows(tmp_path, season):
    path = tmp_path / "dev.csv"
    _write_processed(path, DATASET_DEV_V2_SEASONS + (season,))
    with pytest.raises(HoldoutAccessError):
        load_dev_matches(path, validate=False)


@pytest.fixture(scope="module")
def dev_v2():
    if not PROCESSED_DEV_V2.exists():
        pytest.skip("data/processed/matches_dev_v2.csv missing: run `python -m eplmodel.data.download --seasons 2425` "
                    "and `python -m eplmodel.data.build --dataset dev_v2`")
    return load_dev_matches()


def test_dev_v2_is_the_recorded_dataset(dev_v2):
    assert content_sha256(PROCESSED_DEV_V2) == load_manifest()["data/processed/matches_dev_v2.csv"]


def test_dev_v2_coverage(dev_v2):
    assert len(dev_v2) == 4180
    assert dev_v2.groupby("Season").size().to_dict() == {s: 380 for s in DATASET_DEV_V2_SEASONS}
    val = dev_v2[dev_v2["Season"] == "2425"]
    assert (val["Date"].min(), val["Date"].max()) == (pd.Timestamp("2024-08-16"), pd.Timestamp("2025-05-25"))
    assert dev_v2["Date"].max() < pd.Timestamp("2025-07-01")  # nothing from 2025-26 or later
    assert dev_v2["match_id"].is_unique


def test_dev_v2_contains_dataset_v1_unchanged(dev_v2, matches):
    """The 2014-24 rows of dev_v2 equal dataset v1 match for match.

    Row order within a date can differ: the build sorts by Date with pandas'
    default (unstable) sort, so same-day matches may be ordered differently
    once more rows are added. No team plays twice on one date, so sequential
    Elo ratings are unaffected.
    """
    cols = ["match_id", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"]
    old = dev_v2[dev_v2["Season"] != "2425"].reset_index(drop=True)
    assert old["Date"].equals(matches["Date"])
    assert (old[cols].sort_values("match_id").reset_index(drop=True)
            .equals(matches[cols].sort_values("match_id").reset_index(drop=True)))


def test_build_reproduces_dev_v2(tmp_path):
    if not all((RAW_DIR / f"E0_{s}.csv").exists() for s in DATASET_DEV_V2_SEASONS):
        pytest.skip("raw data for dev_v2 missing")
    out = tmp_path / "matches_dev_v2.csv"
    build_matches(RAW_DIR, verbose=False, seasons=DATASET_DEV_V2_SEASONS).to_csv(out, index=False)
    assert content_sha256(out) == load_manifest()["data/processed/matches_dev_v2.csv"]
