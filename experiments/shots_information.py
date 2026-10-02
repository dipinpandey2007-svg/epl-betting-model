"""Experiment 16 (protocol shots_information_v1): do historical shots add information to the online goal model?

Pre-registered in configs/shots_information_v1.toml and docs/preregistration/shots_information_v1.md (commit
6dd216b) before any implementation.

    python -m experiments.shots_information

- Targets 2017-18 .. 2021-22 only (selection_folds(), strict guard via build_fold); dev_v2 and the raw shot
  columns are cut to seasons <= 2021-22.
- Stage 1, before any outcome of a target is joined: frozen-config check, registered common-group sizes,
  unseen-team exclusions and refit counts, and omega = 0 reproducing the recorded Experiment 13 online
  predictions (1e-12), for both shot arms.
- Stage 2: scoring through the common harness; omega selected per arm by the one-SE rule towards smaller omega
  (all five targets -> the value to lock), the nested estimate on 2018-19 .. 2021-22 and criterion S.
- omega* is locked in [locked] by a separate commit. No other season is scored.

Not part of experiments.run_all.
"""

import subprocess
import tomllib

import numpy as np
import pandas as pd

from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.data.checksums import content_sha256, verify_file
from eplmodel.data.download import raw_season_path
from eplmodel.data.shots import attach_shots, read_shots
from eplmodel.data.validate import validate_matches, validate_season_dates
from eplmodel.evaluation import shots_selection as sel
from eplmodel.evaluation.alignment import involves_teams, teams_in
from eplmodel.evaluation.folds import build_fold, restrict_to_max_season
from eplmodel.evaluation.forecasts import check_forecast_frame
from eplmodel.evaluation.reproduction import RECORDED, compare_predictions, load_recorded_predictions
from eplmodel.evaluation.scoring import EvaluationSet, evaluation_set, paired_difference_clustered
from eplmodel.evaluation.segments import assign_segments, prior_games_played
from eplmodel.models import shots as sm
from eplmodel.models.shots_spec import SHOTS_CONFIG, load_shots_spec
from eplmodel.paths import PROCESSED_DEV_V2, PROJECT_ROOT, RESULTS_DIR
from eplmodel.reporting.locks import refuse_locked_overwrite
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import SEASON_ORDER, TRAIN_SEASONS, selection_folds

PREREG_COMMIT = "6dd216b4de719e138263b00ec3c500122c7a67c3"
PREREG_DOC = "docs/preregistration/shots_information_v1.md"
CONFIG_FILE = "configs/shots_information_v1.toml"
B0_RECORDED = "poisson_tw_online"
SEGMENT_EDGES, SEGMENT_LABELS = [0, 10, 19, 29], ["0-9", "10-18", "19-28", "29+"]


class ProtocolMismatchError(RuntimeError):
    """The code, the data or the config disagree with the frozen pre-registration."""


# --- Frozen state and data ---------------------------------------------------------------------------

def _git_show(path: str) -> str:
    out = subprocess.run(["git", "show", f"{PREREG_COMMIT}:{path}"], cwd=PROJECT_ROOT, capture_output=True,
                         text=True, encoding="utf-8")
    if out.returncode != 0:
        raise ProtocolMismatchError(f"cannot read {path} at the pre-registration commit")
    return out.stdout


def check_frozen(cfg: dict) -> dict:
    """The config equals the pre-registered one except [locked]; the document is unchanged."""
    frozen = tomllib.loads(_git_show(CONFIG_FILE))
    strip = lambda c: {k: v for k, v in c.items() if k != "locked"}
    if strip(frozen) != strip(cfg):
        raise ProtocolMismatchError(f"{CONFIG_FILE} differs from the pre-registration")
    doc = (PROJECT_ROOT / PREREG_DOC).read_text(encoding="utf-8").replace("\r\n", "\n")
    if doc != _git_show(PREREG_DOC).replace("\r\n", "\n"):
        raise ProtocolMismatchError(f"{PREREG_DOC} differs from the pre-registration")
    load_shots_spec()
    return {"prereg_commit": PREREG_COMMIT, "config_sha256": content_sha256(PROJECT_ROOT / CONFIG_FILE),
            "prereg_doc_sha256": content_sha256(PROJECT_ROOT / PREREG_DOC)}


def load_data(max_season: str, require_raw_checksums: bool) -> pd.DataFrame:
    """dev_v2 cut to seasons <= max_season, with the raw shot columns of those seasons attached by match_id."""
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    matches = restrict_to_max_season(load_matches(PROCESSED_DEV_V2, validate=False), max_season)
    validate_matches(matches)
    validate_season_dates(matches)
    seasons = [s for s in TRAIN_SEASONS if SEASON_ORDER.index(s) <= SEASON_ORDER.index(max_season)]
    if require_raw_checksums:
        bad = [s for s in seasons if not verify_file(raw_season_path(s))]
        if bad:
            raise ProtocolMismatchError(f"raw files {bad} differ from data/checksums.json")
    return attach_shots(matches, read_shots(seasons, allowed_seasons=seasons))


# --- Stage 1: forecasts and pre-scoring checks ------------------------------------------------------------

def stage_one(data: pd.DataFrame, cfg: dict, spec, recorded_b0: pd.DataFrame) -> list[dict]:
    g = cfg["groups"]
    staged = []
    for history, target in selection_folds():
        fold = build_fold(data, history, target)
        preds, record = sm.online_fold(fold.history_rows, fold.target_rows, spec.omega_grid, spec.half_life_days,
                                       spec.max_goals, 100, 1000)
        actual = (record["n_common"], record["n_unseen_excluded"], record["n_fits"])
        registered = (g["expected_common"][target], g["expected_unseen_excluded"][target],
                      g["expected_online_fits"][target])
        if actual != registered:
            raise ProtocolMismatchError(f"{target}: (common, excluded, fits) {actual} differ from {registered}")
        check_forecast_frame(preds, [sm.arm_name(a, w) for a in sm.SHOT_ARMS for w in spec.omega_grid])
        rec = recorded_b0[recorded_b0["fold_target"] == target]
        repro = {arm: compare_predictions(preds, rec, [sm.arm_name(arm, 0.0)], spec.b0_reproduction_tolerance,
                                          rename={sm.arm_name(arm, 0.0): B0_RECORDED}) for arm in sm.SHOT_ARMS}
        staged.append({"target": target, "history": history, "fold": fold, "preds": preds, "record": record,
                       "b0_reproduction_max_abs_diff": repro})
    return staged


# --- Stage 2: scoring --------------------------------------------------------------------------------------

def _pool(sets: list[EvaluationSet]) -> EvaluationSet:
    return EvaluationSet(pd.concat([s.preds for s in sets]), pd.concat([s.results for s in sets]),
                         np.concatenate([s.clusters for s in sets]), sets[0].arms)


def returning_teams(fold) -> set[str]:
    """Target teams present in the fold history but absent from the previous season."""
    prev = SEASON_ORDER[SEASON_ORDER.index(fold.target) - 1]
    returning = (teams_in(fold.target_rows) & teams_in(fold.history_rows)) - teams_in(
        fold.history_rows[fold.history_rows["Season"] == prev])
    return returning


def evaluate(staged: list[dict], matches: pd.DataFrame, omegas, outer_targets, se_multiple: float, floor: float,
             k: float, min_negative: int) -> dict:
    """Grid scores, omega selection per arm (all targets and nested), criterion S and descriptive blocks."""
    targets = [s["target"] for s in staged]
    names = {a: {w: sm.arm_name(a, w) for w in omegas} for a in sm.SHOT_ARMS}
    sets, losses_all = {}, {}
    for s in staged:
        es = evaluation_set(s["preds"], matches, s["target"], [n for a in names for n in names[a].values()])
        sets[s["target"]] = es
        losses_all[s["target"]] = es.losses()
    clusters = {t: sets[t].clusters for t in targets}

    out = {"grid": {}, "selection": {}, "nested": {}, "criterion_s": {}}
    nested_omega = {}
    for arm in sm.SHOT_ARMS:
        ll = {t: {w: losses_all[t][names[arm][w]]["log_loss"] for w in omegas} for t in targets}
        out["grid"][arm] = {t: {sm.omega_label(w): {m: float(losses_all[t][names[arm][w]][m].mean())
                                                     for m in ("log_loss", "brier")} for w in omegas} for t in targets}
        out["selection"][arm] = sel.select_omega(ll, clusters, se_multiple)
        nested = sel.nested_selection(ll, clusters, outer_targets, se_multiple)
        nested_omega[arm] = {t: v["omega_selected"] for t, v in nested.items()}
        out["nested"][arm] = {"selected": nested_omega[arm], "tables": nested}
        base = names[arm][0.0]
        diffs = {t: {m: losses_all[t][names[arm][nested_omega[arm][t]]][m] - losses_all[t][base][m]
                     for m in ("log_loss", "brier")} for t in outer_targets}
        res = sel.criterion_s(diffs, {t: clusters[t] for t in outer_targets}, floor, k, min_negative)
        res["omega_star"] = out["selection"][arm]["omega_selected"]
        res["no_shot_weight_selected"] = res["omega_star"] == 0.0
        out["criterion_s"][arm] = res

    # Descriptive: B1 - S1 at their nested omegas on the outer folds.
    d_bs = {m: np.concatenate([losses_all[t][names["b1"][nested_omega["b1"][t]]][m]
                               - losses_all[t][names["s1"][nested_omega["s1"][t]]][m] for t in outer_targets])
            for m in ("log_loss", "brier")}
    cl = np.concatenate([clusters[t] for t in outer_targets])
    out["b1_minus_s1_nested"] = {m: paired_difference_clustered(d_bs[m], np.zeros(len(cl)), cl) for m in d_bs}

    # Descriptive, in-sample at the omega* to be locked: returning vs continuing, segments, calibration.
    b1_star = names["b1"][out["selection"]["b1"]["omega_selected"]]
    b0 = names["b1"][0.0]
    frames = []
    for s in staged:
        p = s["preds"].copy()
        p["returning"] = involves_teams(p, returning_teams(s["fold"])).to_numpy()
        # Experiment 11 segments, from the full target fixture list (fixtures only).
        prior = prior_games_played(s["fold"].target_rows).reindex(p.index).to_numpy()
        p["segment"] = assign_segments(prior, SEGMENT_EDGES, SEGMENT_LABELS)
        frames.append(p)
    pooled = _pool([sets[t] for t in targets])
    flags = pd.concat(frames)
    losses = pooled.losses()
    groups = {"returning": flags["returning"].to_numpy(), "continuing": ~flags["returning"].to_numpy()}
    groups |= {f"segment_{lab}": flags["segment"].to_numpy() == lab for lab in SEGMENT_LABELS}
    name = f"{b1_star}_minus_{b0}"
    out["descriptive_b1_star_minus_b0"] = {
        g: {m: paired_difference_clustered(losses[b1_star][m][mask], losses[b0][m][mask], pooled.clusters[mask])
            for m in ("log_loss", "brier")} | {"n_matches": int(mask.sum())}
        for g, mask in {"all": np.ones(len(flags), dtype=bool), **groups}.items()} | {"comparison": name}
    out["calibration_in_the_large"] = {n: pooled.calibration(n) for n in (b0, b1_star)}
    out["n_scored_per_target"] = {t: int(len(sets[t].preds)) for t in targets}
    return out


def run(write: bool = True) -> dict:
    cfg = load_config(SHOTS_CONFIG)
    out_dir = RESULTS_DIR / cfg["outputs"]["results_name"]
    if write:
        refuse_locked_overwrite(out_dir, cfg["locked"])   # before anything is loaded or computed
    frozen = check_frozen(cfg)
    spec = load_shots_spec()
    data = load_data(spec.max_season, cfg["source"]["require_raw_checksums"])
    recorded_b0 = load_recorded_predictions(RECORDED["13-historical"], spec.targets)

    staged = stage_one(data, cfg, spec, recorded_b0)          # every forecast and check before any outcome
    ev = cfg["evidence"]
    evaluation = evaluate(staged, data, spec.omega_grid, cfg["selection"]["nested_outer_targets"], spec.se_multiple,
                          spec.practical_floor_log_loss, spec.clustered_se_multiple, ev["min_negative_outer_folds"])
    result = {
        "protocol_id": spec.protocol_id, "stage": "historical", "frozen_state": frozen,
        "seasons_loaded": sorted(data["Season"].unique(), key=SEASON_ORDER.index), "targets": list(spec.targets),
        "omega_grid": list(spec.omega_grid), "arms": cfg["arms"],
        "folds": [{"target": s["target"], "history": f"{s['history'][0]}..{s['history'][-1]}",
                   "b0_reproduction_max_abs_diff": s["b0_reproduction_max_abs_diff"],
                   **{k: v for k, v in s["record"].items() if k != "fits"}, "per_fit": s["record"]["fits"]}
                  for s in staged],
        **evaluation,
        "reading_primary": evaluation["criterion_s"]["b1"]["reading"],
        "note": "omega = 0 is B0 (Experiment 13 online). In-sample grid tables are optimistically biased; the nested "
                "estimate is the evidence. Differences are (left - right) per match.",
    }
    if write:
        refuse_locked_overwrite(out_dir, cfg["locked"])   # and again just before writing
        frame = pd.concat([s["preds"].assign(fold_target=s["target"]) for s in staged])
        result["predictions"] = write_predictions(cfg["outputs"]["results_name"], frame)
        write_results(cfg["outputs"]["results_name"], result, data_path=PROCESSED_DEV_V2)
    return result


if __name__ == "__main__":
    out = run()
    print({a: (out["selection"][a]["omega_selected"], out["criterion_s"][a]["reading"]) for a in out["selection"]})
