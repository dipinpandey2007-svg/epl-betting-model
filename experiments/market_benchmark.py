"""Experiment 14 (protocol market_benchmark_v1): a market-implied probability benchmark from Pinnacle 1X2 odds.

Pre-registered in configs/market_benchmark_v1.toml (commit ac60693) before any market probability was
computed. A benchmark only: nothing is fitted, tuned or selected, and no ranking is claimed.

- Arms: Pinnacle closing (primary snapshot) and pre-closing (secondary) odds, each with Shin (primary),
  proportional and power margin removal; six arms, all reported separately (eplmodel.market).
- Odds values are read for 2014-15 .. 2021-22 (coverage); 2017-18 .. 2021-22 are scored, each a selection
  target (assert_selection_target). Exposed, retired and holdout seasons are neither read nor scored.
- Outcomes come from dev_v2 cut to seasons <= 2021-22 straight after reading.
- Scoring uses the common evaluation harness (eplmodel.evaluation.scoring); paired differences are
  (left - right) per match with naive and date-clustered SEs.
- Context: the primary arm against the recorded predictions of Experiments 10 and 13 (historical folds,
  checksum-verified); no model is refitted.

    python -m experiments.market_benchmark

Not part of experiments.run_all.
"""

import numpy as np
import pandas as pd

from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.data.checksums import verify_file
from eplmodel.data.download import raw_season_path
from eplmodel.data.validate import validate_matches, validate_season_dates
from eplmodel.evaluation.calibration import calibration_table
from eplmodel.evaluation.folds import restrict_to_max_season
from eplmodel.evaluation.forecasts import check_forecast_frame, outcomes_for, probabilities
from eplmodel.evaluation.reproduction import RECORDED, load_recorded_predictions
from eplmodel.evaluation.scoring import EvaluationSet, evaluation_set
from eplmodel.market import coverage as cov
from eplmodel.market import devig
from eplmodel.market.benchmark import MarketArm, arm_from_name, market_forecasts
from eplmodel.market.odds import SNAPSHOTS, read_odds
from eplmodel.paths import CONFIG_DIR, HOLDOUT_CONFIG, PROCESSED_DEV_V2, RESULTS_DIR
from eplmodel.reporting.locks import refuse_locked_overwrite
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import (
    REGISTERED_DEV_TEST_SPECS,
    REGISTERED_EXPOSED_VALIDATION_SPECS,
    SELECTION_TARGET_SEASONS,
    TRAIN_SEASONS,
    assert_selection_target,
)

NAME = "market_benchmark"
MARKET_CONFIG = CONFIG_DIR / "market_benchmark_v1.toml"
PRIMARY = "market_close_shin"


class ProtocolMismatchError(RuntimeError):
    """The pre-registration disagrees with the code, the splits or the data."""


def check_protocol(cfg: dict) -> None:
    p, m, v, a = cfg["protocol"], cfg["margin"], cfg["validity"], cfg["arms"]
    expected_arms = {MarketArm(s, meth).name for s in SNAPSHOTS.values() for meth in devig.METHODS}
    expected = {
        "scored_seasons": (tuple(p["scored_seasons"]), SELECTION_TARGET_SEASONS),
        "coverage_seasons": (tuple(p["coverage_seasons"]), TRAIN_SEASONS),
        "outcomes_max_season": (p["outcomes_max_season"], SELECTION_TARGET_SEASONS[-1]),
        "benchmark_only": (p["benchmark_only"], True),
        "selection_candidate": (p["selection_candidate"], False),
        "closing.columns": (tuple(cfg["snapshots"]["closing"]["columns"]), SNAPSHOTS["closing"].columns),
        "pre_closing.columns": (tuple(cfg["snapshots"]["pre_closing"]["columns"]), SNAPSHOTS["pre_closing"].columns),
        "closing.role": (cfg["snapshots"]["closing"]["role"], "primary"),
        "mix_snapshots": (cfg["snapshot_rules"]["mix_snapshots"], False),
        "margin.primary": (m["primary"], "shin"),
        "margin.methods": ({m["primary"], *m["sensitivity"]}, set(devig.METHODS)),
        "margin.solver": ((m["solver_xtol"], m["solver_maxiter"], m["power_c_upper"]),
                          (devig.XTOL, devig.MAXITER, devig.POWER_C_UPPER)),
        "margin.selection_by_score": (m["selection_by_score"], False),
        "validity.reasons": (tuple(v["reasons"]), cov.REASONS),
        "validity.impute": (v["impute"], False),
        "arms": (set(a), expected_arms),
        "primary_arm": (a[PRIMARY], "market_pinnacle_close_shin_v1"),
        "ranking_claims": (cfg["evaluation"]["ranking_claims"], False),
        "clv": (cfg["clv"]["status"], "not_implemented"),
        "in_run_all": (cfg["outputs"]["in_run_all"], False),
    }
    wrong = {k: val for k, val in expected.items() if val[0] != val[1]}
    if wrong:
        raise ProtocolMismatchError(f"Pre-registration disagrees with the code or splits: {wrong}")
    for season in p["scored_seasons"]:
        assert_selection_target(season)
    holdout_specs = set(load_config(HOLDOUT_CONFIG)["registration"]["spec_ids"])
    if set(a.values()) & (REGISTERED_DEV_TEST_SPECS | REGISTERED_EXPOSED_VALIDATION_SPECS | holdout_specs):
        raise ProtocolMismatchError("Market arms must not be registered for dev-test, exposed-validation or holdout.")


def load_outcomes(max_season: str) -> pd.DataFrame:
    """dev_v2, checksum-verified, cut to seasons <= max_season straight after reading."""
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    matches = restrict_to_max_season(load_matches(PROCESSED_DEV_V2, validate=False), max_season)
    validate_matches(matches)
    validate_season_dates(matches)
    return matches


def _pooled(sets: list[EvaluationSet]) -> EvaluationSet:
    return EvaluationSet(pd.concat([s.preds for s in sets]), pd.concat([s.results for s in sets]),
                         np.concatenate([s.clusters for s in sets]), sets[0].arms)


def _arm_block(es: EvaluationSet, arm: str, n_matches: int) -> dict:
    return {"n_scored": int(len(es.preds)), "n_matches": int(n_matches),
            "coverage": len(es.preds) / n_matches if n_matches else float("nan"),
            **es.scores()[arm], "calibration": es.calibration(arm)}


def _paired(frames: dict[str, pd.DataFrame], matches: pd.DataFrame, season_of: pd.Series, left: str, right: str,
            seasons) -> dict:
    """(left - right) per season and pooled, on the matches both arms cover, aligned by match_id."""
    ids = frames[left].index.intersection(frames[right].index, sort=False)
    joined = frames[left].loc[ids].join(frames[right].loc[ids, [c for c in frames[right].columns
                                                                 if c.startswith(f"{right}_")]])
    out = {}
    per_season = []
    for s in seasons:
        part = joined[season_of.loc[joined.index].to_numpy() == s]
        es = evaluation_set(part, matches, s, [left, right])
        per_season.append(es)
        out[s] = es.differences([f"{left}_minus_{right}"])[f"{left}_minus_{right}"]
    out["pooled"] = _pooled(per_season).differences([f"{left}_minus_{right}"])[f"{left}_minus_{right}"]
    out["n_matches"] = int(len(ids))
    return out


def run(write: bool = True) -> dict:
    cfg = load_config(MARKET_CONFIG)
    out_dir = RESULTS_DIR / cfg["outputs"]["results_name"]
    if write:
        # The historical results are recorded: refuse before anything is checked, loaded or scored.
        refuse_locked_overwrite(out_dir, cfg.get("locked", {}))
    check_protocol(cfg)
    p, v = cfg["protocol"], cfg["validity"]
    coverage_seasons, scored = list(p["coverage_seasons"]), list(p["scored_seasons"])
    if p["require_raw_checksums"]:
        bad = [s for s in coverage_seasons if not verify_file(raw_season_path(s))]
        if bad:
            raise ProtocolMismatchError(f"raw files {bad} differ from data/checksums.json")

    # 1. Odds (no result column is read) and validity, for the coverage seasons.
    snapshots = [SNAPSHOTS["closing"], SNAPSHOTS["pre_closing"]]
    odds = read_odds(coverage_seasons, snapshots, allowed_seasons=coverage_seasons)
    labels = {s.name: cov.classify(odds, s, v["min_booksum"], v["max_booksum"]) for s in snapshots}
    coverage = {s.name: {"by_season": cov.coverage_table(odds, labels[s.name]),
                         "booksum": cov.booksum_summary(odds, s, labels[s.name])} for s in snapshots}

    # 2. Forecast frames of every arm on the scored seasons, before any outcome is read.
    in_scored = odds["Season"].isin(scored).to_numpy()
    scored_odds = odds[in_scored].reset_index(drop=True)
    frames, diagnostics = {}, {}
    for name in cfg["arms"]:
        arm = arm_from_name(name)
        lab = labels[arm.snapshot.name][in_scored].reset_index(drop=True)
        frames[name], diag = market_forecasts(scored_odds, lab, arm)
        diag["Season"] = frames[name]["Season"].to_numpy()
        param = "shin_z" if arm.method == "shin" else "power_c" if arm.method == "power" else "booksum"
        diagnostics[name] = diag.groupby("Season")[param].agg(["min", "median", "max"]).reset_index()

    # 3. Outcomes (seasons <= 2021-22 only), joined by match_id.
    matches = load_outcomes(p["outcomes_max_season"])
    scored_matches = matches[matches["Season"].isin(scored)]
    if set(scored_odds["match_id"]) != set(scored_matches["match_id"]):
        raise ProtocolMismatchError("odds and processed matches do not cover the same scored matches")
    n_by_season = scored_matches.groupby("Season").size()
    season_of = scored_odds.set_index("match_id")["Season"]

    arms = {}
    for name, frame in frames.items():
        sets = [evaluation_set(frame[frame["Season"] == s], matches, s, [name]) for s in scored]
        arms[name] = {"spec_id": cfg["arms"][name],
                      "by_season": {s: _arm_block(es, name, int(n_by_season[s])) for s, es in zip(scored, sets)},
                      "pooled": _arm_block(_pooled(sets), name, int(n_by_season.sum()))}

    paired = {f"{a}_minus_{b}": _paired(frames, matches, season_of, a, b, scored)
              for a, b in cfg["evaluation"]["paired"]}

    # 4. Context: the primary arm against recorded predictions (historical folds; checksum-verified).
    rec10 = load_recorded_predictions(RECORDED["10"], scored)
    rec13 = load_recorded_predictions(RECORDED["13-historical"], scored)
    context = {"full_group_experiment_10": {}, "common_group_experiment_13": {}}
    for key, rec, models in (("full_group_experiment_10", rec10, cfg["evaluation"]["context_full"]),
                             ("common_group_experiment_13", rec13, cfg["evaluation"]["context_common"])):
        for model in models:
            ref = rec[[f"{model}_{o}" for o in "HDA"]].copy()
            ref = ref[ref.notna().all(axis=1)]
            ref.index.name = "match_id"
            check_forecast_frame(ref, [model])
            context[key][f"{PRIMARY}_minus_{model}"] = _paired({PRIMARY: frames[PRIMARY], model: ref}, matches,
                                                               season_of, PRIMARY, model, scored)

    primary_frame = frames[PRIMARY]
    home_won = outcomes_for(matches, primary_frame.index).to_numpy() == "H"
    result = {
        "protocol_id": p["protocol_id"],
        "benchmark_only": True,
        "ranking_claims": False,
        "primary_arm": PRIMARY,
        "seasons": {"coverage": coverage_seasons, "scored": scored},
        "sign_convention": "differences are (left - right) per match; negative = left has the lower loss",
        "coverage": coverage,
        "solver_diagnostics": diagnostics,
        "arms": arms,
        "paired": paired,
        "context": context,
        "primary_home_win_reliability": calibration_table(probabilities(primary_frame, PRIMARY)[:, 0], home_won, 10)
        .reset_index().astype({"bin": str}),
        "note": "Benchmark only. The specification was fixed before scoring; no arm is selected or ranked. "
                "Context comparisons use recorded predictions and are not model-selection evidence.",
    }
    if write:
        refuse_locked_overwrite(out_dir, cfg.get("locked", {}))   # and again just before writing
        joined = frames[PRIMARY][["Date", "Season", "HomeTeam", "AwayTeam"]].copy()
        for name, frame in frames.items():
            joined = joined.join(frame[[c for c in frame.columns if c.startswith(f"{name}_")]])
        result["predictions"] = write_predictions(NAME, joined)
        write_results(NAME, result, data_path=PROCESSED_DEV_V2)
    return result


if __name__ == "__main__":
    out = run()
    for name, block in out["arms"].items():
        print(f"{name:28s} pooled n={block['pooled']['n_scored']}  log loss {block['pooled']['log_loss']:.4f}  "
              f"Brier {block['pooled']['brier']:.4f}")
