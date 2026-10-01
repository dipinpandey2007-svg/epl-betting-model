"""Exposed descriptive validation (protocol exposed_validation_descriptive_v1, access-log entry V5).

The frozen Experiment 15 arms (M1, M2, S1; M0 = recorded Experiment 13 online predictions) and the frozen
Experiment 14 market benchmark, run unchanged on the exposed validation season (2024-25). Evidence class:
exposed/descriptive only; nothing is selected, tuned or confirmed, and no criterion is applied.

    python -m experiments.exposed_validation_descriptive

- The season is opened only by eplmodel.evaluation.folds.build_exposed_validation_fold for logged entry V5 and
  exactly its specs. History = 2014-15 .. 2023-24.
- Stage 1, before any outcome is joined: the frozen configs of Experiments 14-15 and their locks, the fixture-level
  facts in the config, the EB team-season counts, the M0 anchor (1e-7), coverage and refit counts, odds validity.
- Stage 2: scoring through the common harness; paired differences reported descriptively.

Not part of experiments.run_all.
"""

import json

import numpy as np
import pandas as pd

from eplmodel.analysis.promoted_teams import categorize_team, load_team_history
from eplmodel.config import load_config
from eplmodel.data import load_dev_matches
from eplmodel.data.checksums import content_sha256, verify_file
from eplmodel.data.download import raw_season_path
from eplmodel.evaluation.alignment import involves_teams, teams_in
from eplmodel.evaluation.folds import build_exposed_validation_fold
from eplmodel.evaluation.forecasts import check_forecast_frame, prob_columns
from eplmodel.evaluation.reproduction import RECORDED, compare_predictions, load_recorded_predictions
from eplmodel.evaluation.segments import assign_segments, prior_games_played
from eplmodel.market import coverage as mcov
from eplmodel.market.benchmark import arm_from_name, market_forecasts
from eplmodel.market.odds import SNAPSHOTS, read_odds
from eplmodel.models import full_coverage as fc
from eplmodel.models.full_coverage_spec import FULL_COVERAGE_CONFIG, load_full_coverage_spec
from eplmodel.paths import CONFIG_DIR, PROCESSED_DEV_V2, RESULTS_DIR
from eplmodel.reporting.results import write_predictions, write_results
from experiments import full_coverage_poisson as fcp
from experiments import market_benchmark as mb

STAGE_CONFIG = CONFIG_DIR / "exposed_validation_descriptive_v1.toml"
EXP14_METRICS_SHA256 = "55a77b4f57a0f55a8a6daf68aabba12bc552972bda98e837ea05ccac6eebcf56"   # RESULTS_LOG Exp 14
SEGMENT_EDGES, SEGMENT_LABELS = [0, 10, 19, 29], ["0-9", "10-18", "19-28", "29+"]


class ProtocolMismatchError(RuntimeError):
    """The data, the frozen specifications or the locks disagree with the registration."""


def check_frozen_inputs(scfg: dict, fcfg: dict, mcfg: dict) -> dict:
    """Experiment 15 config frozen and locked; Experiment 14 protocol unchanged and its metrics as recorded."""
    fcp.check_frozen(fcfg)
    lock = fcfg["historical_locked"]
    m15 = RESULTS_DIR / fcfg["outputs"]["results_name"] / "metrics.json"
    if lock["status"] != "locked" or content_sha256(m15) != lock["historical_metrics_sha256"]:
        raise ProtocolMismatchError("Experiment 15 historical results are not locked as recorded")
    mb.check_protocol(mcfg)
    if content_sha256(RESULTS_DIR / mcfg["outputs"]["results_name"] / "metrics.json") != EXP14_METRICS_SHA256:
        raise ProtocolMismatchError("Experiment 14 metrics differ from the recorded result")
    spec_ids = set(scfg["access"]["spec_ids"])
    expected = {fcfg["arms"][a] for a in scfg["experiment_15"]["arms"]} | {mcfg["arms"][a] for a in
                                                                           scfg["experiment_14"]["arms"]}
    if spec_ids != expected:
        raise ProtocolMismatchError("authorised specs differ from the frozen arms")
    return {"experiment_15_metrics_sha256": lock["historical_metrics_sha256"],
            "experiment_14_metrics_sha256": EXP14_METRICS_SHA256,
            "stage_config_sha256": content_sha256(STAGE_CONFIG)}


def taxonomy(fold, structure, yoyo_threshold: int) -> dict:
    """Experiment 7 categories of the promoted teams; an unseen team missing from team_history.csv is unclassified."""
    team_history = load_team_history()
    out = {}
    seasons = sorted(set(fold.history_rows["Season"]))
    for team in structure.promoted:
        if team in structure.returning:
            last = max(s for s in seasons if team in teams_in(fold.history_rows[fold.history_rows["Season"] == s]))
            out_seasons = int(fc.SEASON_ORDER.index(fold.target) - fc.SEASON_ORDER.index(last) - 1)
            out[team] = "recent_yoyo" if out_seasons <= yoyo_threshold else "long_absence_or_newcomer"
        elif team in team_history:
            out[team] = categorize_team(team, team_history, yoyo_threshold)
        else:
            out[team] = "unclassified"
    return out


def stage_one(matches, scfg, fcfg, mcfg, spec):
    e15 = scfg["experiment_15"]
    fold = build_exposed_validation_fold(matches, scfg["protocol"]["access_entry"], scfg["access"]["spec_ids"])
    target = fold.target
    structure = fc.target_structure(fold.history_rows, fold.target_rows)
    tgt = fold.target_rows.sort_values("Date", kind="stable")
    facts = (len(tgt), list(structure.promoted), list(structure.unseen), list(structure.returning))
    expected = (e15["expected_full"], e15["expected_promoted_teams"], e15["expected_unseen_teams"],
                e15["expected_returning_teams"])
    if facts != expected:
        raise ProtocolMismatchError(f"fixture facts {facts} differ from the registration {expected}")
    cats = taxonomy(fold, structure, fcfg["groups"]["yoyo_threshold"])
    by_cat = lambda c: [t for t, v in cats.items() if v == c]
    flags = pd.DataFrame({
        "full": True,
        "common": ~involves_teams(tgt, structure.unseen).to_numpy(),
        "unseen": involves_teams(tgt, structure.unseen).to_numpy(),
        "promoted": involves_teams(tgt, structure.promoted).to_numpy(),
        "returning": involves_teams(tgt, structure.returning).to_numpy(),
        "continuing_only": ~involves_teams(tgt, structure.promoted).to_numpy(),
        "recent_yoyo": involves_teams(tgt, by_cat("recent_yoyo")).to_numpy(),
        "long_absence_or_newcomer": involves_teams(tgt, by_cat("long_absence_or_newcomer")).to_numpy(),
        "unclassified": involves_teams(tgt, by_cat("unclassified")).to_numpy(),
    }, index=pd.Index(tgt["match_id"], name="match_id"))
    if int(flags["common"].sum()) != e15["expected_common"] or int(flags["unseen"].sum()) != e15["expected_unseen"]:
        raise ProtocolMismatchError("common/unseen group sizes differ from the registration")
    flags["prior_games"] = prior_games_played(tgt).reindex(flags.index).to_numpy()
    flags["segment"] = assign_segments(flags["prior_games"], SEGMENT_EDGES, SEGMENT_LABELS)

    # Experiment 15 arms, frozen code path.
    settings = fc.SolverSettings(spec.solver.gradient_tolerance, spec.solver.max_iterations, spec.solver.step_halvings)
    eb = fc.empirical_bayes(fold.history_rows, target, spec.prior.variance_floor, settings)
    if [eb.n_promoted, eb.n_continuing] != e15["expected_eb_team_seasons"]:
        raise ProtocolMismatchError("EB team-season counts differ from the registration")
    anchor, _ = fc.online_fold(fold.history_rows, fold.target_rows, fc.ANCHOR, None, spec.half_life_days,
                               spec.max_goals, settings)
    rec13 = load_recorded_predictions(RECORDED["13-validation"])
    anchor_diff = compare_predictions(anchor, rec13, ["anchor"], tolerance=e15["anchor_tolerance"],
                                      rename={"anchor": e15["m0"]})
    preds, records = flags.copy(), {}
    n_dates = tgt["Date"].nunique()
    for arm in e15["arms"]:
        p, rec = fc.online_fold(fold.history_rows, fold.target_rows, arm, eb, spec.half_life_days, spec.max_goals,
                                settings)
        check_forecast_frame(p, [arm])
        if list(p.index) != list(flags.index) or rec["n_fits"] != n_dates:
            raise ProtocolMismatchError(f"{arm}: coverage or refit count differs")
        preds = preds.join(p[prob_columns(arm)])
        records[arm] = rec

    # Experiment 14 market arms, frozen code path and validity rules.
    p14, v14 = mcfg["protocol"], mcfg["validity"]
    if p14["require_raw_checksums"] and not verify_file(raw_season_path(target)):
        raise ProtocolMismatchError("raw odds file differs from data/checksums.json")
    snapshots = [SNAPSHOTS["closing"], SNAPSHOTS["pre_closing"]]
    odds = read_odds([target], snapshots, allowed_seasons=[target])
    labels = {s.name: mcov.classify(odds, s, v14["min_booksum"], v14["max_booksum"]) for s in snapshots}
    market_coverage = {s.name: {"by_season": mcov.coverage_table(odds, labels[s.name]),
                                "booksum": mcov.booksum_summary(odds, s, labels[s.name])} for s in snapshots}
    if set(odds["match_id"]) != set(flags.index):
        raise ProtocolMismatchError("odds and fixtures do not cover the same matches")
    for name in scfg["experiment_14"]["arms"]:
        arm = arm_from_name(name)
        frame, _ = market_forecasts(odds, labels[arm.snapshot.name], arm)
        preds = preds.join(frame[prob_columns(name)], how="left")
    preds = fold.fixtures().join(preds)
    return {"fold": fold, "structure": structure, "categories": cats, "eb": eb, "anchor_max_abs_diff": anchor_diff,
            "preds": preds, "records": records, "market_coverage": market_coverage, "n_dates": n_dates}


def _valid(frame: pd.DataFrame, arm: str) -> np.ndarray:
    return frame[prob_columns(arm)].notna().all(axis=1).to_numpy()


def paired_descriptive(frame, matches, left, right, group, target, segment=None) -> dict:
    """(left - right) on the group's matches where both arms have a forecast; no reading, no label."""
    f = frame.copy()
    f["_mask"] = f[group].to_numpy(dtype=bool) & _valid(f, left) & _valid(f, right)
    return fcp.paired(f, matches, left, right, "_mask", [target], segment) | {"group": group}


def run(write: bool = True) -> dict:
    scfg = load_config(STAGE_CONFIG)
    fcfg, mcfg = load_config(FULL_COVERAGE_CONFIG), load_config(mb.MARKET_CONFIG)
    frozen = check_frozen_inputs(scfg, fcfg, mcfg)
    spec = load_full_coverage_spec()
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    matches = load_dev_matches()

    s1 = stage_one(matches, scfg, fcfg, mcfg, spec)          # every forecast and check before any outcome
    target, preds = s1["fold"].target, s1["preds"]
    rec10 = load_recorded_predictions(RECORDED["10"], [target])
    rec13 = load_recorded_predictions(RECORDED["13-validation"], [target])
    comp = rec10[[*prob_columns("elo"), *prob_columns("frequency_baseline")]]
    comp = comp.join(rec13[[*prob_columns("poisson_tw_online"), *prob_columns("poisson_tw"),
                            *prob_columns("poisson")]], how="left")
    frame = preds.join(comp)
    if not np.array_equal(_valid(frame, "poisson_tw_online"), frame["common"].to_numpy(dtype=bool)):
        raise ProtocolMismatchError("recorded M0 does not cover exactly the common group")

    # Scoring (outcomes joined by match_id inside the harness).
    fc_arms, mk_arms = scfg["experiment_15"]["arms"], scfg["experiment_14"]["arms"]
    all_arms = [*fc_arms, "poisson_tw_online", *mk_arms, "elo", "frequency_baseline", "poisson", "poisson_tw"]
    groups = ["full", "common", "unseen", "promoted", "returning", "continuing_only", "recent_yoyo",
              "long_absence_or_newcomer", "unclassified"]
    scores = {}
    for g in groups:
        scores[g] = {}
        for a in all_arms:
            f = frame.copy()
            f["_mask"] = f[g].to_numpy(dtype=bool) & _valid(f, a)
            if f["_mask"].any():
                scores[g][a] = fcp.scores(f, matches, a, "_mask", [target])["pooled"]
    coverage = {a: {"n_forecasts": int(_valid(frame, a).sum()), "n_matches": int(len(frame))} for a in all_arms}

    e15 = {f"{l}_minus_{r}__{g}": paired_descriptive(frame, matches, l, r, g, target)
           for l, r, g in scfg["experiment_15"]["comparisons"]}
    e15_segments = {f"{l}_minus_{r}__{g}": {lab: paired_descriptive(frame, matches, l, r, g, target, lab)
                                            for lab in SEGMENT_LABELS}
                    for l, r, g in scfg["experiment_15"]["comparisons"]}
    e14 = {f"{l}_minus_{r}__full": paired_descriptive(frame, matches, l, r, "full", target)
           for l, r in mcfg["evaluation"]["paired"]}
    e14_context = {f"market_close_shin_minus_{m}__full": paired_descriptive(frame, matches, "market_close_shin", m,
                                                                             "full", target)
                   for m in scfg["experiment_14"]["context_full"]}
    e14_context |= {f"market_close_shin_minus_{m}__common": paired_descriptive(frame, matches, "market_close_shin",
                                                                                m, "common", target)
                    for m in scfg["experiment_14"]["context_common"]}
    calib = {}
    for a in [*fc_arms, "market_close_shin"]:
        f = frame.copy()
        f["_mask"] = _valid(f, a)
        calib[a] = fcp._pool(fcp._sets(f, matches, [a], "_mask", [target])).calibration(a)

    hist15 = json.loads((RESULTS_DIR / fcfg["outputs"]["results_name"] / "metrics.json").read_text(encoding="utf-8"))
    hist14 = json.loads((RESULTS_DIR / mcfg["outputs"]["results_name"] / "metrics.json").read_text(encoding="utf-8"))
    eb = s1["eb"]
    result = {
        "protocol_id": scfg["protocol"]["protocol_id"],
        "access_entry": scfg["protocol"]["access_entry"],
        "evidence_class": scfg["protocol"]["evidence_class"],
        "criteria_applied": False,
        "target": target,
        "history": f"{s1['fold'].history[0]}..{s1['fold'].history[-1]}",
        "frozen_inputs": frozen,
        "structure": {"promoted": list(s1["structure"].promoted), "returning": list(s1["structure"].returning),
                      "unseen": list(s1["structure"].unseen), "taxonomy": s1["categories"]},
        "coverage": coverage,
        "experiment_15": {
            "anchor_m0_max_abs_diff": s1["anchor_max_abs_diff"],
            "n_target_dates": s1["n_dates"],
            "empirical_bayes": {"seasons": list(eb.seasons), "a_promoted": eb.a_promoted, "b_promoted": eb.b_promoted,
                                "tau_att": eb.tau_att, "tau_def": eb.tau_def, "n_promoted": eb.n_promoted,
                                "n_continuing": eb.n_continuing, "floor_binding_att": eb.floor_binding_att,
                                "floor_binding_def": eb.floor_binding_def, "corr_att_def": eb.corr_att_def},
            "fits": {a: {k: v for k, v in r.items() if k != "fits"} | {
                "home_advantage_first": r["fits"][0]["home_advantage"],
                "home_advantage_last": r["fits"][-1]["home_advantage"], "per_fit": r["fits"]}
                for a, r in s1["records"].items()},
            "paired_descriptive": e15,
            "segments_descriptive": e15_segments,
        },
        "experiment_14": {"coverage": s1["market_coverage"], "paired_descriptive": e14,
                          "context_descriptive": e14_context},
        "scores": scores,
        "calibration_in_the_large": calib,
        "historical_side_by_side": {
            "experiment_15_pooled_full": {a: hist15["results"]["scores"]["full"][a]["pooled"] for a in fc_arms},
            "experiment_15_primary_m2_minus_m0": hist15["results"]["primary_non_inferiority"],
            "experiment_14_pooled": {a: hist14["results"]["arms"][a]["pooled"]["log_loss"] for a in mk_arms},
        },
        "note": "Exposed descriptive validation only: cannot select, tune or confirm anything; no criterion applied. "
                "Differences are (left - right) per match.",
    }
    if write:
        result["predictions"] = write_predictions(scfg["outputs"]["results_name"], frame)
        write_results(scfg["outputs"]["results_name"], result, data_path=PROCESSED_DEV_V2)
    return result


if __name__ == "__main__":
    out = run()
    print("coverage:", {a: v["n_forecasts"] for a, v in out["coverage"].items()})
