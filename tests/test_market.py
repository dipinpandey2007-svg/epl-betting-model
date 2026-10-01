"""Market benchmark (protocol market_benchmark_v1): odds mapping, margin removal, validity, protocol guards.

Unit tests use synthetic odds files. The only real-data test (golden) reruns the recorded experiment and
compares it with the committed metrics.json.
"""

import json

import numpy as np
import pandas as pd
import pytest

from eplmodel.config import load_config
from eplmodel.evaluation.forecasts import ForecastFormatError, check_forecast_frame, prob_columns
from eplmodel.market import coverage as cov
from eplmodel.market import devig
from eplmodel.market.benchmark import MarketArm, arm_from_name, market_forecasts
from eplmodel.market.odds import (
    PINNACLE_CLOSING,
    PINNACLE_PRE_CLOSING,
    SNAPSHOTS,
    read_season_odds,
    snapshot_columns,
)
from eplmodel.paths import PROCESSED_DEV_V2, RESULTS_DIR
from eplmodel.splits import HoldoutAccessError, SplitAccessError
from experiments import market_benchmark as mb

CFG = load_config(mb.MARKET_CONFIG)
BOTH = [PINNACLE_CLOSING, PINNACLE_PRE_CLOSING]


def _raw_file(tmp_path, season="1718", rows=None):
    rows = rows or [
        {"Date": "12/08/2017", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 0, "FTR": "H",
         "PSH": 1.80, "PSD": 3.80, "PSA": 5.00, "PSCH": 1.75, "PSCD": 3.90, "PSCA": 5.40},
        {"Date": "13/08/2017", "HomeTeam": "C", "AwayTeam": "D", "FTHG": 1, "FTAG": 1, "FTR": "D",
         "PSH": 2.60, "PSD": 3.30, "PSA": 2.90, "PSCH": 2.70, "PSCD": 3.25, "PSCA": 2.85},
    ]
    path = tmp_path / f"E0_{season}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _bisect(f, lo, hi, n=200):
    """An independent root finder (plain bisection) to check the brentq solutions."""
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


ODDS = np.array([1.75, 3.90, 5.40])


# --- Implied probabilities and margin removal -----------------------------------------------------

def test_implied_probabilities_and_booksum():
    pi = devig.implied_probabilities(ODDS)
    assert pi == pytest.approx(1 / ODDS)
    assert pi.sum() > 1


def test_proportional_normalisation():
    pi = 1 / ODDS
    p, info = devig.proportional(pi)
    assert p == pytest.approx(pi / pi.sum(), abs=0)
    assert info["booksum"] == pytest.approx(pi.sum())


def test_power_normalisation_solves_its_equation():
    pi = 1 / ODDS
    p, info = devig.power(pi)
    c = _bisect(lambda k: np.sum(pi ** k) - 1, 1.0, 100.0)
    assert info["power_c"] == pytest.approx(c, abs=1e-12) and c > 1
    assert p == pytest.approx(pi ** c, abs=1e-12)


def test_shin_solves_its_equation():
    pi = 1 / ODDS
    p, info = devig.shin(pi)
    z = _bisect(lambda x: devig.shin_probabilities(pi, x).sum() - 1, 0.0, 1 - 1e-12)
    assert 0 < info["shin_z"] < 1
    assert info["shin_z"] == pytest.approx(z, abs=1e-12)
    expected = (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / pi.sum()) - z) / (2 * (1 - z))
    assert p == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("method", sorted(devig.METHODS))
def test_probabilities_sum_to_one_and_keep_hda_order(method):
    p, _ = devig.remove_margin(ODDS, method)
    assert p.sum() == pytest.approx(1.0, abs=1e-15)
    assert p[0] > p[1] > p[2]                     # favourite home side stays first


def test_shin_and_power_remove_more_margin_from_the_longshot():
    prop, _ = devig.remove_margin(ODDS, "proportional")
    for method in ("shin", "power"):
        p, _ = devig.remove_margin(ODDS, method)
        assert p[2] < prop[2] and p[0] > prop[0]


@pytest.mark.parametrize("method", sorted(devig.METHODS))
def test_a_fair_book_is_returned_unchanged(method):
    odds = np.array([2.0, 4.0, 4.0])              # 1/2 + 1/4 + 1/4 = 1
    p, _ = devig.remove_margin(odds, method)
    assert p == pytest.approx([0.5, 0.25, 0.25], abs=1e-15)


@pytest.mark.parametrize("method", sorted(devig.METHODS))
def test_equal_odds_give_equal_probabilities(method):
    p, _ = devig.remove_margin([2.8, 2.8, 2.8], method)
    assert p == pytest.approx([1 / 3] * 3, abs=1e-14)


@pytest.mark.parametrize("bad", [[1.0, 3.0, 4.0], [0.9, 3.0, 4.0], [np.nan, 3.0, 4.0], [np.inf, 3, 4], [2.0, 3.0],
                                 [3.0, 4.0, 5.0]])
@pytest.mark.parametrize("method", sorted(devig.METHODS))
def test_invalid_odds_are_refused(bad, method):
    with pytest.raises(devig.MarginRemovalError):
        devig.remove_margin(bad, method)          # [3, 4, 5]: booksum < 1


def test_unknown_method_is_refused():
    with pytest.raises(devig.MarginRemovalError):
        devig.remove_margin(ODDS, "best_of_three")


def test_margin_removal_is_deterministic():
    for method in devig.METHODS:
        a, ia = devig.remove_margin(ODDS, method)
        b, ib = devig.remove_margin(ODDS, method)
        assert np.array_equal(a, b) and ia == ib


# --- Odds ingestion and snapshot separation --------------------------------------------------------

def test_hda_columns_map_to_the_verified_snapshots():
    assert PINNACLE_CLOSING.columns == ("PSCH", "PSCD", "PSCA")
    assert PINNACLE_PRE_CLOSING.columns == ("PSH", "PSD", "PSA")
    assert set(PINNACLE_CLOSING.columns).isdisjoint(PINNACLE_PRE_CLOSING.columns)
    assert snapshot_columns(PINNACLE_CLOSING) == ["closing_odds_H", "closing_odds_D", "closing_odds_A"]


def test_read_season_odds_maps_columns_and_reads_no_result(tmp_path):
    _raw_file(tmp_path)
    odds = read_season_odds("1718", BOTH, ["1718"], raw_dir=tmp_path)
    assert list(odds["match_id"]) == ["2017-08-12_A_B", "2017-08-13_C_D"]
    assert odds.loc[0, snapshot_columns(PINNACLE_CLOSING)].astype(float).tolist() == [1.75, 3.90, 5.40]
    assert odds.loc[0, snapshot_columns(PINNACLE_PRE_CLOSING)].astype(float).tolist() == [1.80, 3.80, 5.00]
    assert not {"FTR", "FTHG", "FTAG"} & set(odds.columns)


def test_outcomes_cannot_change_market_probabilities(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir(), b_dir.mkdir()
    _raw_file(a_dir)
    flipped = pd.read_csv(a_dir / "E0_1718.csv").assign(FTHG=[0, 3], FTAG=[4, 0], FTR=["A", "H"])
    flipped.to_csv(b_dir / "E0_1718.csv", index=False)
    out = []
    for d in (a_dir, b_dir):
        odds = read_season_odds("1718", BOTH, ["1718"], raw_dir=d)
        labels = cov.classify(odds, PINNACLE_CLOSING, 1.0, 1.10)
        out.append(market_forecasts(odds, labels, MarketArm(PINNACLE_CLOSING, "shin"))[0])
    pd.testing.assert_frame_equal(out[0], out[1], check_exact=True)


def test_snapshots_are_computed_independently(tmp_path):
    _raw_file(tmp_path)
    base = read_season_odds("1718", BOTH, ["1718"], raw_dir=tmp_path)
    changed = base.copy()
    changed[snapshot_columns(PINNACLE_CLOSING)] = [[1.5, 4.5, 7.0], [2.0, 3.6, 3.9]]
    for odds_table in (base, changed):
        labels = cov.classify(odds_table, PINNACLE_PRE_CLOSING, 1.0, 1.10)
        frame, _ = market_forecasts(odds_table, labels, MarketArm(PINNACLE_PRE_CLOSING, "shin"))
        if odds_table is base:
            first = frame
    pd.testing.assert_frame_equal(first, frame, check_exact=True)   # closing prices never reach the pre arm


@pytest.mark.parametrize("season", ["2526", "2627"])
def test_holdout_seasons_are_refused_before_any_file_is_read(tmp_path, season):
    _raw_file(tmp_path, season=season)
    with pytest.raises(HoldoutAccessError):
        read_season_odds(season, BOTH, [season], raw_dir=tmp_path)


@pytest.mark.parametrize("season", ["2223", "2324", "2425"])
def test_seasons_outside_the_protocol_are_refused(tmp_path, season):
    _raw_file(tmp_path, season=season)
    with pytest.raises(SplitAccessError):
        read_season_odds(season, BOTH, CFG["protocol"]["coverage_seasons"], raw_dir=tmp_path)


# --- Validity and coverage -------------------------------------------------------------------------

def _table(rows):
    df = pd.DataFrame(rows, columns=snapshot_columns(PINNACLE_CLOSING), dtype=object)
    df["Season"] = "1718"
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    return df


def test_validity_reasons_in_registered_order():
    table = _table([[1.75, 3.9, 5.4], [None, 3.9, 5.4], ["n/a", 3.9, 5.4], [1.0, 3.9, 5.4],
                    [3.0, 4.0, 5.0], [1.2, 3.0, 4.0], [2.0, 4.0, 4.0]])
    labels = cov.classify(table, PINNACLE_CLOSING, 1.0, 1.10)
    assert list(labels) == ["valid", "missing", "non_numeric", "odds_not_above_1", "booksum_below_min",
                            "booksum_above_max", "valid"]
    t = cov.coverage_table(table, labels).iloc[0]
    assert (t["n_matches"], t["valid"], t["missing"], t["booksum_above_max"]) == (7, 2, 1, 1)


def test_missing_prices_are_excluded_not_imputed():
    table = _table([[1.75, 3.9, 5.4], [None, 3.9, 5.4], [2.6, 3.3, 2.9]])
    table["Date"], table["HomeTeam"], table["AwayTeam"] = pd.Timestamp("2017-08-12"), "A", "B"
    labels = cov.classify(table, PINNACLE_CLOSING, 1.0, 1.10)
    frame, _ = market_forecasts(table, labels, MarketArm(PINNACLE_CLOSING, "proportional"))
    assert list(frame.index) == ["m0", "m2"]
    check_forecast_frame(frame, ["market_close_proportional"])


def test_market_frames_never_carry_results():
    frame = pd.DataFrame([[0.5, 0.3, 0.2]], columns=prob_columns("m"), index=pd.Index(["x"], name="match_id"))
    with pytest.raises(ForecastFormatError):
        check_forecast_frame(frame.assign(FTR="H"), ["m"])


def test_arm_names():
    assert MarketArm(PINNACLE_CLOSING, "shin").name == "market_close_shin"
    assert arm_from_name("market_pre_power") == MarketArm(SNAPSHOTS["pre_closing"], "power")
    with pytest.raises(ValueError):
        arm_from_name("elo_close_shin")


# --- Protocol --------------------------------------------------------------------------------------

def test_protocol_agrees_with_the_code_and_splits():
    mb.check_protocol(CFG)
    assert CFG["margin"]["primary"] == "shin" and set(CFG["margin"]["sensitivity"]) == {"proportional", "power"}
    assert CFG["protocol"]["scored_seasons"] == ["1718", "1819", "1920", "2021", "2122"]


@pytest.mark.parametrize("section,key,value", [
    ("margin", "primary", "power"),
    ("protocol", "scored_seasons", ["1718", "1819", "1920", "2021", "2122", "2223"]),
    ("snapshot_rules", "mix_snapshots", True),
    ("validity", "impute", True),
    ("evaluation", "ranking_claims", True),
    ("clv", "status", "implemented"),
])
def test_protocol_mismatch_is_refused(section, key, value):
    cfg = json.loads(json.dumps(CFG))
    cfg[section][key] = value
    with pytest.raises(mb.ProtocolMismatchError):
        mb.check_protocol(cfg)


def test_benchmark_is_not_in_run_all():
    from experiments import run_all
    assert mb not in run_all.EXPERIMENTS


# --- Recorded result (needs the data) ---------------------------------------------------------------

@pytest.mark.golden
def test_recorded_market_benchmark_reproduces():
    metrics = RESULTS_DIR / mb.NAME / "metrics.json"
    if not PROCESSED_DEV_V2.exists() or not metrics.exists():
        pytest.skip("data/processed/matches_dev_v2.csv or the recorded market metrics missing")
    recorded = json.loads(metrics.read_text(encoding="utf-8"))["results"]
    recorded.pop("predictions")
    from eplmodel.reporting.results import _jsonable
    assert json.loads(json.dumps(_jsonable(mb.run(write=False)))) == recorded
