"""Time-weighted static Poisson (protocol time_weighted_poisson_v1), in two separately run stages.

Pre-registered in configs/time_weighted_poisson_v1.toml and docs/TEST_SET_ACCESS_LOG.md entry V3
before any time-weighted model was fitted on real data.

    python -m experiments.time_weighted_poisson --stage development
    python -m experiments.time_weighted_poisson --stage validation

Development stage:
- The match table is cut to seasons <= 2021-22 straight after reading, before anything else, and
  the stage refuses any later season.
- Every grid half-life is fitted on each development fold (targets 2017-18 .. 2021-22).
- H = inf must reproduce the recorded poisson_static_v1 predictions (Experiment 10).
- H* is selected by the one-SE rule on all five targets, the nested estimate re-selects H for each
  outer target from earlier targets only, and criterion D is evaluated.
- The selected H* is NOT written to the config by this code: it is locked by hand, in a separate
  commit ([locked] in the config).

Validation stage:
- Refuses to run until [locked] holds the development stage's H*, the development metrics match
  their recorded checksum and were produced from a clean commit that HEAD descends from, and the
  working tree is clean outside results/.
- Scores 2024-25 once for the locked H* only; the grid is never scored on 2024-25.
- If H* = inf, nothing new is scored.
- The established arms must first reproduce the Experiment 10 and 11 predictions.

Not part of experiments.run_all.
"""

import argparse
import json
import math
import subprocess

import numpy as np
import pandas as pd

from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.data.checksums import content_sha256, load_manifest, verify_file
from eplmodel.data.validate import validate_matches, validate_season_dates
from eplmodel.evaluation import time_weighting as tw
from eplmodel.evaluation import update_policy as up
from eplmodel.evaluation.validation import outcomes_for, prob_columns
from eplmodel.models.poisson import FORMULA
from eplmodel.paths import CONFIG_DIR, HOLDOUT_CONFIG, PROCESSED_DEV_V2, PROJECT_ROOT, RESULTS_DIR
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import REGISTERED_DEV_TEST_SPECS, SEASON_ORDER, SELECTION_VALIDATION_SEASONS, selection_folds
from experiments import update_policy_diagnostic, validation_2425

TW_CONFIG = CONFIG_DIR / "time_weighted_poisson_v1.toml"
TW_ARM = "poisson_tw"


class ProtocolMismatchError(RuntimeError):
    """The pre-registration disagrees with the code, the frozen specs, the splits or the data."""


class LockError(RuntimeError):
    """The validation stage may not run: H* is not locked in a committed, clean state."""


# --- Protocol ----------------------------------------------------------------------------------

def check_protocol(tcfg: dict, vcfg: dict, dcfg: dict, base: dict) -> None:
    """Refuse to run unless the pre-registration agrees with the established protocols and the code."""
    p, c, s, n, e, r = (tcfg["protocol"], tcfg["candidate"], tcfg["selection"], tcfg["nested"],
                        tcfg["evidence"], tcfg["reproduction"])
    grid = [float(h) for h in c["half_life_grid_days"]]
    dev = list(p["development_targets"])
    expected = {
        "development_targets": (tuple(dev), SELECTION_VALIDATION_SEASONS[:5]),
        "validation_target": (p["validation_target"], SELECTION_VALIDATION_SEASONS[5]),
        "score_grid_on_validation": (p["score_grid_on_validation"], False),
        "data_sha256": (p["data_sha256"], vcfg["protocol"]["data_sha256"]),
        "data_sha256_manifest": (p["data_sha256"], load_manifest()[PROCESSED_DEV_V2.relative_to(PROJECT_ROOT).as_posix()]),
        "base_spec": (c["base_spec"], base["poisson"]["spec_id"]),
        "formula": (c["formula"], base["poisson"]["formula"]),
        "formula_in_code": (c["formula"], FORMULA),
        "max_goals": (c["max_goals"], base["poisson"]["max_goals"]),
        "fit": (c["fit"], "once_on_history"),
        "in_season_updates": (c["in_season_updates"], False),
        "unseen_team_rule": (c["unseen_team_rule"], vcfg["poisson"]["unseen_team_rule"]),
        "weighting": (c["weighting"], "exponential_half_life_days"),
        "reference_date": (c["reference_date"], "latest_history_date"),
        "age_unit": (c["age_unit"], "days"),
        "weights_apply_to": (c["weights_apply_to"], "both_goal_rows_of_a_match_equally"),
        "glm_weights": (c["glm_weights"], "var_weights"),
        "truncation": (c["truncation"], "none"),
        "grid_has_static": (math.inf in grid, True),
        "grid_sorted_unique": (grid, sorted(set(grid))),
        "selection.group": (s["group"], "common"),
        "selection.metric": (s["metric"], "log_loss"),
        "selection.criterion": (s["criterion"], "mean_of_target_mean_log_loss"),
        "selection.tie_break": (s["tie_break"], "longer_half_life"),
        "selection.rule": (s["rule"], "one_se_longest"),
        "selection.se_type": (s["se_type"], "clustered"),
        "nested.outer_targets": (tuple(n["outer_targets"]), tuple(dev[1:])),
        "nested.inner_targets": (n["inner_targets"], "development_targets_strictly_before_outer"),
        "evidence.primary_difference": (e["primary_difference"], f"{TW_ARM}_minus_{tw.STATIC_ARM}"),
        "development_stage.max_season": (tcfg["development_stage"]["max_season"], dev[-1]),
        "reproduction.experiment_10_sha256": (r["experiment_10_sha256"], dcfg["reproduction"]["predictions_sha256"]),
        "reproduction.experiment_10_file": (r["experiment_10_predictions"], dcfg["reproduction"]["predictions_file"]),
        "reproduction.experiment_11_file": (r["experiment_11_predictions"], dcfg["outputs"]["predictions_file"]),
        "validation_stage.arms": (tuple(tcfg["validation_stage"]["arms"]),
                                  ("elo", "elo_f1", "elo_f2", tw.STATIC_ARM, TW_ARM, "dixon_coles", "frequency_baseline")),
        "outputs.in_run_all": (tcfg["outputs"]["in_run_all"], False),
    }
    wrong = {k: v for k, v in expected.items() if v[0] != v[1]}
    if wrong:
        raise ProtocolMismatchError(f"Pre-registration disagrees with frozen values or code: {wrong}")
    for h in grid:
        tw.half_life_label(h)  # positive whole days or inf
    new_specs = {c["spec_id"], *(f"{c['grid_arm_prefix']}{tw.half_life_label(h)}{c['grid_arm_suffix']}" for h in grid
                                 if not math.isinf(h))}
    holdout_specs = set(load_config(HOLDOUT_CONFIG)["registration"]["spec_ids"])
    if new_specs & (REGISTERED_DEV_TEST_SPECS | holdout_specs):
        raise ProtocolMismatchError("Time-weighted specs must not be registered for dev-test or holdout scoring.")


def load_all_configs() -> tuple[dict, dict, dict, dict]:
    tcfg = load_config(TW_CONFIG)
    vcfg = load_config(validation_2425.VALIDATION_CONFIG)
    dcfg = load_config(update_policy_diagnostic.DIAGNOSTIC_CONFIG)
    base = load_config()
    check_protocol(tcfg, vcfg, dcfg, base)
    return tcfg, vcfg, dcfg, base


def check_groups(target: str, fitted: dict, tcfg: dict) -> None:
    """Registered common-group size, full size and unseen teams must hold before scoring (fixtures only)."""
    g = tcfg["groups"]
    actual = (fitted["n_full"], fitted["n_common"], fitted["unseen_teams"])
    registered = (g["expected_full"][target], g["expected_common"][target], g["expected_unseen_teams"][target])
    if actual != registered:
        raise ProtocolMismatchError(f"{target}: groups {actual} differ from the registration {registered}")


def load_recorded(path_key: str, sha_key: str, tcfg: dict, targets) -> pd.DataFrame:
    """Recorded predictions, checksum-verified, restricted to the given fold targets straight away."""
    r = tcfg["reproduction"]
    path = PROJECT_ROOT / r[path_key]
    if not path.exists() or content_sha256(path) != r[sha_key]:
        raise ProtocolMismatchError(f"{r[path_key]} is missing or differs from its registered checksum")
    recorded = pd.read_csv(path, index_col="match_id", dtype={"Season": str, "fold_target": str})
    return recorded[recorded["fold_target"].isin(list(targets))]


def check_reproduction(target: str, preds: pd.DataFrame, recorded: pd.DataFrame, models, tolerance: float,
                       common_only: bool) -> float:
    """Arms in `preds` must equal the recorded predictions for the same matches; returns the largest gap."""
    rec = recorded[recorded["fold_target"] == target]
    if common_only:
        rec = rec[rec["in_common"].astype(bool)]
    if list(rec.index) != list(preds.index):
        raise ProtocolMismatchError(f"{target}: matches differ from the recorded predictions")
    cols = [c for m in models for c in prob_columns(m)]
    new, old = preds[cols].to_numpy(dtype=float), rec[cols].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(new), np.isnan(old)):
        raise ProtocolMismatchError(f"{target}: missing predictions differ from the recorded ones")
    worst = float(np.nanmax(np.abs(new - old))) if new.size else 0.0
    if worst > tolerance:
        raise ProtocolMismatchError(f"{target}: predictions differ from the record by {worst:.3e}")
    return worst


def date_clusters(preds: pd.DataFrame, target: str) -> np.ndarray:
    return (target + "_" + pd.to_datetime(preds["Date"]).dt.strftime("%Y-%m-%d")).to_numpy()


# --- Development stage -------------------------------------------------------------------------

def load_development_matches(max_season: str) -> pd.DataFrame:
    """dev_v2, checksum-verified, cut to seasons <= max_season before any validation or use.

    The file is read from disk as a whole (it is one CSV), but rows of later seasons are dropped
    immediately after parsing; nothing is computed on them.
    """
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    matches = tw.restrict_to_development(load_matches(PROCESSED_DEV_V2, validate=False), max_season)
    validate_matches(matches)
    validate_season_dates(matches)
    return matches


def run_development(write: bool = True) -> dict:
    tcfg, vcfg, _, _ = load_all_configs()
    p, c, ds = tcfg["protocol"], tcfg["candidate"], tcfg["development_stage"]
    max_season = ds["max_season"]
    matches = load_development_matches(max_season)
    tw.assert_development_only(matches, max_season)
    targets = list(p["development_targets"])
    recorded = load_recorded("experiment_10_predictions", "experiment_10_sha256", tcfg, targets)
    grid = [float(h) for h in c["half_life_grid_days"]]
    arms = {tw.grid_arm(h): h for h in grid}

    # Stage 1: every prediction and every pre-scoring check, before any outcome is joined.
    staged = []
    for history, target in selection_folds(targets):
        tw.assert_development_only(matches[matches["Season"].isin([*history, target])], max_season)
        preds, fitted = tw.tw_fold_predictions(matches, history, target, arms, c["max_goals"])
        check_groups(target, fitted, tcfg)
        fitted["reproduction_max_abs_diff"] = check_reproduction(
            target, preds, recorded, [tw.STATIC_ARM], tcfg["reproduction"]["tolerance"], common_only=True)
        staged.append((history, target, preds, fitted))

    # Stage 2: scoring and selection.
    se_multiple, floor = tcfg["selection"]["se_multiple"], tcfg["evidence"]["practical_floor_log_loss"]
    losses, clusters, folds, all_preds = {}, {}, [], []
    for history, target, preds, fitted in staged:
        results = outcomes_for(matches, preds.index)
        clusters[target] = date_clusters(preds, target)
        per_arm = tw.per_match_losses(preds, results, list(arms))
        losses[target] = {h: per_arm[arm]["log_loss"] for arm, h in arms.items()}
        returning = preds["involves_returning_team"].to_numpy(dtype=bool)
        folds.append({
            "target": target,
            "history": f"{history[0]}..{history[-1]}",
            "n_history_seasons": len(history),
            "fitted": fitted,
            "scores": {tw.half_life_label(h): {m: float(per_arm[arm][m].mean()) for m in tw.METRICS}
                       for arm, h in arms.items()},
            "paired_vs_static": {
                tw.half_life_label(h): {m: up.paired_difference_clustered(per_arm[arm][m], per_arm[tw.STATIC_ARM][m],
                                                                          clusters[target]) for m in tw.METRICS}
                for arm, h in arms.items()},
            "sharpness_and_calibration": {tw.half_life_label(h): tw.sharpness_and_calibration(preds, results, arm)
                                          for arm, h in arms.items()},
            "returning_vs_continuous_log_loss_vs_static": {
                tw.half_life_label(h): {
                    "returning": up.paired_difference_clustered(per_arm[arm]["log_loss"][returning],
                                                                per_arm[tw.STATIC_ARM]["log_loss"][returning],
                                                                clusters[target][returning]),
                    "continuous": up.paired_difference_clustered(per_arm[arm]["log_loss"][~returning],
                                                                 per_arm[tw.STATIC_ARM]["log_loss"][~returning],
                                                                 clusters[target][~returning])}
                for arm, h in arms.items()},
        })
        all_preds.append((preds.assign(fold_target=target), per_arm))

    selection = tw.select_half_life(losses, clusters, se_multiple)
    nested = tw.nested_selection(losses, clusters, tcfg["nested"]["outer_targets"], se_multiple)
    outer = {}
    for target, sel in nested.items():
        per_arm = next(pa for pr, pa in all_preds if pr["fold_target"].iloc[0] == target)
        arm = tw.grid_arm(sel["h_selected_days"])
        outer[target] = {m: per_arm[arm][m] - per_arm[tw.STATIC_ARM][m] for m in tw.METRICS}
    criterion_d = tw.development_criterion(outer, {t: clusters[t] for t in outer}, floor,
                                           tcfg["evidence"]["clustered_se_multiple"],
                                           tcfg["evidence"]["development_min_negative_outer_folds"])
    h_star = selection["h_selected"]
    result = {
        "protocol_id": p["protocol_id"],
        "stage": "development",
        "seasons_loaded": sorted(matches["Season"].unique(), key=SEASON_ORDER.index),
        "grid": [tw.half_life_label(h) for h in grid],
        "selection": {**selection, "h_star": h_star,
                      "note": "In-sample: H* is selected on the same five targets, so its scores here are "
                              "optimistically biased."},
        "nested": {
            "selected": {t: s["h_selected"] for t, s in nested.items()},
            "selection_tables": nested,
            "outer_differences": {t: {m: up.paired_difference_clustered(d[m], np.zeros(len(d[m])), clusters[t])
                                      for m in tw.METRICS} for t, d in outer.items()},
        },
        "leave_one_target_out": tw.leave_one_target_out(losses, clusters, se_multiple),
        "criterion_d": criterion_d,
        "reading": tw.reading(criterion_d["met"], None, h_star == "inf"),
        "folds": folds,
    }
    if write:
        result["predictions"] = write_predictions(ds["results_name"], pd.concat([pr for pr, _ in all_preds]))
        write_results(ds["results_name"], result, data_path=PROCESSED_DEV_V2)
    return result


# --- Validation stage ----------------------------------------------------------------------------

def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)


def working_tree_clean() -> bool:
    """No uncommitted change outside results/ (tracked or untracked)."""
    out = _git("status", "--porcelain", "--", ".", ":(exclude)results")
    return out.returncode == 0 and out.stdout.strip() == ""


def is_ancestor_of_head(commit: str) -> bool:
    return bool(commit) and _git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0


def check_lock(tcfg: dict) -> float:
    """Return the locked H* (days) or raise LockError. Reads only the development metrics and git state."""
    lock = tcfg["locked"]
    if lock.get("status") != "locked":
        raise LockError("H* is not locked: [locked] status is not 'locked'.")
    try:
        h_star = float(lock["half_life_days"])
    except (TypeError, ValueError):
        raise LockError("[locked] half_life_days is not a number.") from None
    if h_star not in [float(h) for h in tcfg["candidate"]["half_life_grid_days"]]:
        raise LockError(f"Locked H* = {h_star} is not on the registered grid.")
    path = RESULTS_DIR / tcfg["development_stage"]["results_name"] / "metrics.json"
    if not path.exists() or content_sha256(path) != lock["development_metrics_sha256"]:
        raise LockError("Development metrics are missing or differ from [locked] development_metrics_sha256.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["results"]["selection"]["h_star"] != tw.half_life_label(h_star):
        raise LockError("Locked H* differs from the half-life selected by the development stage.")
    prov = payload["provenance"]
    if prov.get("git_dirty") is not False or prov.get("git_commit") != lock["development_commit"]:
        raise LockError("The development stage did not run from the clean commit recorded in [locked].")
    if not is_ancestor_of_head(lock["development_commit"]):
        raise LockError("HEAD does not descend from the development-stage commit.")
    if not working_tree_clean():
        raise LockError("Working tree has uncommitted changes outside results/: commit the lock first.")
    return h_star


def run_validation(write: bool = True) -> dict:
    tcfg, vcfg, dcfg, _ = load_all_configs()
    h_star = check_lock(tcfg)
    p, vs = tcfg["protocol"], tcfg["validation_stage"]
    target = p["validation_target"]
    dev_metrics = json.loads((RESULTS_DIR / tcfg["development_stage"]["results_name"] / "metrics.json")
                             .read_text(encoding="utf-8"))["results"]
    d_met = dev_metrics["criterion_d"]["met"]
    if math.isinf(h_star):
        result = {"protocol_id": p["protocol_id"], "stage": "validation", "h_star": "inf",
                  "reading": tw.reading(d_met, None, True),
                  "note": "No weighting selected: nothing new is scored on 2024-25."}
        if write:
            write_results(vs["results_name"], result, data_path=PROCESSED_DEV_V2)
        return result

    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    from eplmodel.data import load_dev_matches
    matches = load_dev_matches()
    (history, tgt), = selection_folds([target])

    # Predictions and every check before any outcome is joined.
    preds, fitted = up.diagnostic_fold_predictions(matches, history, tgt, vcfg, dcfg)
    r = tcfg["reproduction"]
    rec10 = load_recorded("experiment_10_predictions", "experiment_10_sha256", tcfg, [tgt])
    rec11 = load_recorded("experiment_11_predictions", "experiment_11_sha256", tcfg, [tgt])
    repro = {"experiment_10": check_reproduction(tgt, preds, rec10, r["experiment_10_models"], r["tolerance"], False),
             "experiment_11": check_reproduction(tgt, preds, rec11, r["experiment_11_models"], r["tolerance"], False)}
    tw_preds, tw_fitted = tw.tw_fold_predictions(matches, history, tgt, {TW_ARM: h_star},
                                                 tcfg["candidate"]["max_goals"])
    check_groups(tgt, tw_fitted, tcfg)
    common = preds[preds["in_common"].astype(bool)]
    if list(common.index) != list(tw_preds.index):
        raise ProtocolMismatchError("time-weighted predictions and the common group differ")
    common = common.join(tw_preds[prob_columns(TW_ARM) + ["involves_returning_team"]])

    # Scoring.
    results = outcomes_for(matches, common.index)
    clusters = date_clusters(common, tgt)
    arms = vs["arms"]
    losses = tw.per_match_losses(common, results, arms)
    diff = {m: losses[TW_ARM][m] - losses[tw.STATIC_ARM][m] for m in tw.METRICS}
    ev = tcfg["evidence"]
    criterion_v = tw.validation_criterion(diff["log_loss"], diff["brier"], clusters,
                                          ev["practical_floor_log_loss"], ev["clustered_se_multiple"])
    residual = {m: up.check_decomposition({a: losses[a][m] for a in arms}, vs["components"], "poisson_minus_elo",
                                          vs["identity_tolerance"]) for m in tw.METRICS}

    def differences(names, mask=None):
        mask = np.ones(len(common), dtype=bool) if mask is None else mask
        out = {}
        for name in names:
            left, right = up.split_difference(name)
            out[name] = {m: up.paired_difference_clustered(losses[left][m][mask], losses[right][m][mask],
                                                           clusters[mask]) for m in tw.METRICS}
        return out

    segments = common["segment"].to_numpy()
    returning = common["involves_returning_team"].to_numpy(dtype=bool)
    result = {
        "protocol_id": p["protocol_id"],
        "stage": "validation",
        "target": tgt,
        "h_star": tw.half_life_label(h_star),
        "n_common": int(len(common)),
        "reproduction_max_abs_diff": repro,
        "fitted": tw_fitted,
        "scores": {a: {m: float(losses[a][m].mean()) for m in tw.METRICS} for a in arms},
        "primary_tw_minus_static": criterion_v,
        "criterion_d_from_development": d_met,
        "reading": tw.reading(d_met, criterion_v["met"], False),
        "decomposition": {"identity": vs["identity"], "max_identity_residual": residual,
                          "components": differences([*vs["components"], "poisson_minus_elo"]),
                          "experiment_11_split": differences(["poisson_minus_elo_f2", "poisson_minus_poisson_tw",
                                                              "poisson_tw_minus_elo_f2"])},
        "context_comparisons": differences([f"{a}_minus_{b}" for a, b in vs["context_pairs"]]),
        "segments": {label: differences([f"{TW_ARM}_minus_{tw.STATIC_ARM}"], segments == label)
                     for label in vs["segments_labels"]},
        "returning_vs_continuous": {
            "returning": differences([f"{TW_ARM}_minus_{tw.STATIC_ARM}"], returning),
            "continuous": differences([f"{TW_ARM}_minus_{tw.STATIC_ARM}"], ~returning)},
        "sharpness_and_calibration": {a: tw.sharpness_and_calibration(common, results, a) for a in arms},
    }
    if write:
        result["predictions"] = write_predictions(vs["results_name"], common.assign(fold_target=tgt))
        write_results(vs["results_name"], result, data_path=PROCESSED_DEV_V2)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("development", "validation"))
    args = parser.parse_args(argv)
    if args.stage == "development":
        out = run_development()
        print(f"Selected H* = {out['selection']['h_star']} (H_min {out['selection']['h_min']}); "
              f"criterion D met: {out['criterion_d']['met']}. Lock H* in a separate commit before validation.")
    else:
        out = run_validation()
        print(f"2024-25, H* = {out['h_star']}: reading {out['reading']}")


if __name__ == "__main__":
    main()
