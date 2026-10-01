"""Online vs frozen time-weighted Poisson (protocol online_tw_poisson_diagnostic_v1), in two stages.

Pre-registered in configs/online_tw_poisson_diagnostic_v1.toml and docs/TEST_SET_ACCESS_LOG.md entry V4
before any online weighted Poisson model was fitted on real data. A diagnostic only: nothing is
selected or changed because of its results.

    python -m experiments.online_tw_poisson_diagnostic --stage historical
    python -m experiments.online_tw_poisson_diagnostic --stage validation

Historical stage:
- The match table is cut to seasons <= 2021-22 straight after reading; later seasons are refused.
- For each target 2017-18 .. 2021-22: the established arms are recomputed and must reproduce the
  Experiment 11 predictions, the frozen arm (H = 730) must reproduce Experiment 12, the online arm
  is refitted once per target date, and online must equal frozen on the first target date.
- All of that happens before any outcome is joined; then criterion U is evaluated.
- The historical results are locked by hand, in a separate commit ([historical_locked]).

Validation stage (2024-25, diagnostic/descriptive only):
- Refuses to run, before loading any data, until [historical_locked] matches the committed
  historical metrics from a clean commit that HEAD descends from, and the tree is clean.
- Scores the online arm once at H = 730. It cannot select, tune or confirm anything.

Not part of experiments.run_all.
"""

import argparse
import json

import numpy as np
import pandas as pd

from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.data.checksums import content_sha256, load_manifest, verify_file
from eplmodel.data.validate import validate_matches, validate_season_dates
from eplmodel.evaluation import online_poisson as op
from eplmodel.evaluation import time_weighting as tw
from eplmodel.evaluation import update_policy as up
from eplmodel.evaluation.validation import outcomes_for, prob_columns
from eplmodel.models.poisson import FORMULA
from eplmodel.paths import CONFIG_DIR, HOLDOUT_CONFIG, PROCESSED_DEV_V2, PROJECT_ROOT, RESULTS_DIR
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import REGISTERED_DEV_TEST_SPECS, SEASON_ORDER, SELECTION_VALIDATION_SEASONS, selection_folds
from experiments import time_weighted_poisson as twp
from experiments import update_policy_diagnostic, validation_2425

ONLINE_CONFIG = CONFIG_DIR / "online_tw_poisson_diagnostic_v1.toml"

working_tree_clean = twp.working_tree_clean
is_ancestor_of_head = twp.is_ancestor_of_head


class ProtocolMismatchError(RuntimeError):
    """The pre-registration disagrees with the code, the locked candidate, the splits or the data."""


class LockError(RuntimeError):
    """The validation stage may not run: the historical results are not locked in a committed, clean state."""


# --- Protocol ----------------------------------------------------------------------------------

def _recorded_predictions_sha(results_name: str) -> str | None:
    path = RESULTS_DIR / results_name / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["results"]["predictions"]["sha256"]


def check_protocol(ocfg: dict, tcfg: dict, vcfg: dict, dcfg: dict, base: dict) -> None:
    """Refuse to run unless the pre-registration agrees with the locked candidate, the frozen specs and the code."""
    p, a, o, f, g = ocfg["protocol"], ocfg["arms"], ocfg["online"], ocfg["fitting"], ocfg["groups"]
    u, dec, seg, r = ocfg["criterion_u"], ocfg["decomposition"], ocfg["segments"], ocfg["reproduction"]
    c, lock = tcfg["candidate"], tcfg["locked"]
    targets = [*p["historical_targets"], p["validation_target"]]
    expected = {
        "historical_targets": (tuple(p["historical_targets"]), SELECTION_VALIDATION_SEASONS[:5]),
        "validation_target": (p["validation_target"], SELECTION_VALIDATION_SEASONS[5]),
        "validation_evidence_class": (p["validation_evidence_class"], "diagnostic_descriptive_cannot_confirm"),
        "pool_validation_with_historical": (p["pool_validation_with_historical"], False),
        "data_sha256": (p["data_sha256"], tcfg["protocol"]["data_sha256"]),
        "data_sha256_manifest": (p["data_sha256"],
                                 load_manifest()[PROCESSED_DEV_V2.relative_to(PROJECT_ROOT).as_posix()]),
        "arms.poisson_tw": (a["poisson_tw"], c["spec_id"]),
        "arms.poisson": (a["poisson"], base["poisson"]["spec_id"]),
        "arms.elo": (a["elo"], base["elo"]["spec_id"]),
        "arms.dixon_coles": (a["dixon_coles"], base["dixon_coles"]["spec_id"]),
        "arms.frequency_baseline": (a["frequency_baseline"], base["baseline"]["spec_id"]),
        "arms.elo_f1": (a["elo_f1"], dcfg["arms"]["elo_f1"]),
        "arms.elo_f2": (a["elo_f2"], dcfg["arms"]["elo_f2"]),
        "arm_names": (set(a), set(op.ARMS)),
        "candidate_locked": (lock["status"], "locked"),
        "half_life_days": (float(o["half_life_days"]), float(lock["half_life_days"])),
        "half_life_is_730": (float(o["half_life_days"]), 730.0),
        "formula": (o["formula"], c["formula"]),
        "formula_in_code": (o["formula"], FORMULA),
        "max_goals": (o["max_goals"], c["max_goals"]),
        "weighting": (o["weighting"], c["weighting"]),
        "age_unit": (o["age_unit"], c["age_unit"]),
        "glm_weights": (o["glm_weights"], c["glm_weights"]),
        "truncation": (o["truncation"], c["truncation"]),
        "fit_set": (o["fit_set"], "history_plus_common_target_matches_strictly_before_date"),
        "reference_date": (o["reference_date"], "latest_date_in_fit_set"),
        "refit_cadence": (o["refit_cadence"], "once_per_target_date"),
        "information_policy": (o["information_policy"], "strictly_earlier_dates"),
        "refit_parameters": (o["refit_parameters"], "all"),
        "warm_start": (o["warm_start"], False),
        "unseen_team_rule": (o["unseen_team_rule"], c["unseen_team_rule"]),
        "unseen_team_matches_in_online_fit": (o["unseen_team_matches_in_online_fit"], False),
        "team_parameter_space": (o["team_parameter_space"], "fixed_history_teams"),
        "fitting": ((f["method"], f["maxiter"], f["tol"], f["retry_maxiter"], f["on_failure"]),
                    ("IRLS", 100, 1e-8, 1000, "abort_stage_no_fallback")),
        "groups.expected_common": (g["expected_common"], dcfg["groups"]["expected_common"]),
        "groups.expected_full": (g["expected_full"], dcfg["groups"]["expected_full"]),
        "groups.expected_unseen_teams": (g["expected_unseen_teams"], dcfg["groups"]["expected_unseen_teams"]),
        "groups.expected_online_fits": (sorted(g["expected_online_fits"]), sorted(targets)),
        "primary.difference": (ocfg["primary"]["difference"], op.PRIMARY),
        "criterion_u": ((u["practical_floor_log_loss"], u["clustered_se_multiple"], u["min_folds_same_sign"]),
                        (tcfg["evidence"]["practical_floor_log_loss"], tcfg["evidence"]["clustered_se_multiple"], 4)),
        "segments": ((seg["lower_edges"], seg["labels"]),
                     (dcfg["segments"]["lower_edges"], dcfg["segments"]["labels"])),
        "decomposition.components": (tuple(dec["components"]),
                                     (f"poisson_minus_{op.FROZEN_ARM}", f"{op.FROZEN_ARM}_minus_{op.ONLINE_ARM}",
                                      f"{op.ONLINE_ARM}_minus_elo")),
        "decomposition.total": (dec["total"], "poisson_minus_elo"),
        "reproduction.experiment_11_sha256": (r["experiment_11_sha256"], tcfg["reproduction"]["experiment_11_sha256"]),
        "reproduction.experiment_11_file": (r["experiment_11_predictions"], dcfg["outputs"]["predictions_file"]),
        "reproduction.experiment_12_development_sha256": (
            r["experiment_12_development_sha256"],
            _recorded_predictions_sha(tcfg["development_stage"]["results_name"])),
        "reproduction.experiment_12_validation_sha256": (
            r["experiment_12_validation_sha256"], _recorded_predictions_sha(tcfg["validation_stage"]["results_name"])),
        "reproduction.experiment_12_development_column": (r["experiment_12_development_column"],
                                                          tw.grid_arm(float(lock["half_life_days"]))),
        "reproduction.experiment_12_validation_column": (r["experiment_12_validation_column"], twp.TW_ARM),
        "historical_stage.max_season": (ocfg["historical_stage"]["max_season"], p["historical_targets"][-1]),
        "outputs.in_run_all": (ocfg["outputs"]["in_run_all"], False),
    }
    wrong = {k: v for k, v in expected.items() if v[0] != v[1]}
    if wrong:
        raise ProtocolMismatchError(f"Pre-registration disagrees with locked or frozen values or code: {wrong}")
    holdout_specs = set(load_config(HOLDOUT_CONFIG)["registration"]["spec_ids"])
    if a["poisson_tw_online"] in REGISTERED_DEV_TEST_SPECS | holdout_specs | {a[k] for k in a if k != "poisson_tw_online"}:
        raise ProtocolMismatchError("The online arm must be a new spec, not registered for dev-test or holdout scoring.")


def load_all_configs() -> tuple[dict, dict, dict, dict, dict]:
    ocfg = load_config(ONLINE_CONFIG)
    tcfg = load_config(twp.TW_CONFIG)
    vcfg = load_config(validation_2425.VALIDATION_CONFIG)
    dcfg = load_config(update_policy_diagnostic.DIAGNOSTIC_CONFIG)
    base = load_config()
    check_protocol(ocfg, tcfg, vcfg, dcfg, base)
    return ocfg, tcfg, vcfg, dcfg, base


def check_groups(target: str, fitted: dict, ocfg: dict) -> None:
    """Registered group sizes, unseen teams and number of online fits (fixtures only), before scoring."""
    g = ocfg["groups"]
    actual = (fitted["n_full"], fitted["n_common"], fitted["unseen_teams"], fitted["n_fits"])
    registered = (g["expected_full"][target], g["expected_common"][target], g["expected_unseen_teams"][target],
                  g["expected_online_fits"][target])
    if actual != registered:
        raise ProtocolMismatchError(f"{target}: (full, common, unseen, fits) {actual} differ from the "
                                    f"registration {registered}")


def load_recorded(ocfg: dict, path_key: str, sha_key: str, targets) -> pd.DataFrame:
    """Recorded predictions, checksum-verified, restricted to the given fold targets straight away."""
    r = ocfg["reproduction"]
    path = PROJECT_ROOT / r[path_key]
    if not path.exists() or content_sha256(path) != r[sha_key]:
        raise ProtocolMismatchError(f"{r[path_key]} is missing or differs from its registered checksum")
    recorded = pd.read_csv(path, index_col="match_id", dtype={"Season": str, "fold_target": str})
    return recorded[recorded["fold_target"].isin(list(targets))]


def as_frozen_arm(recorded: pd.DataFrame, column: str) -> pd.DataFrame:
    """Recorded Experiment 12 predictions with the frozen arm's columns renamed to the frozen-arm prefix."""
    renamed = recorded.rename(columns=dict(zip(prob_columns(column), prob_columns(op.FROZEN_ARM))))
    keep = [*prob_columns(op.FROZEN_ARM), "fold_target"]
    return renamed[keep].assign(in_common=True)


def date_clusters(preds: pd.DataFrame, target: str) -> np.ndarray:
    return twp.date_clusters(preds, target)


# --- One fold: every prediction and pre-scoring check ----------------------------------------------

def fold_predictions(matches: pd.DataFrame, history, target: str, ocfg: dict, vcfg: dict, dcfg: dict,
                     rec11: pd.DataFrame, rec12: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Common-group predictions of every arm, after every registered check. No outcome is joined here."""
    o, f, tol = ocfg["online"], ocfg["fitting"], ocfg["reproduction"]["tolerance"]
    h = float(o["half_life_days"])
    diag, diag_fitted = up.diagnostic_fold_predictions(matches, history, target, vcfg, dcfg)
    repro11 = twp.check_reproduction(target, diag, rec11, ocfg["reproduction"]["experiment_11_models"], tol,
                                     common_only=False)
    frozen, frozen_fitted = tw.tw_fold_predictions(matches, history, target, {op.FROZEN_ARM: h}, o["max_goals"])
    repro12 = twp.check_reproduction(target, frozen, rec12, [op.FROZEN_ARM], tol, common_only=True)
    online, online_fitted = op.online_fold_predictions(matches, history, target, h, o["max_goals"],
                                                       f["maxiter"], f["retry_maxiter"])
    check_groups(target, online_fitted, ocfg)

    common = diag[diag["in_common"].astype(bool)]
    if not (list(common.index) == list(frozen.index) == list(online.index)):
        raise ProtocolMismatchError(f"{target}: the arms' common groups differ")
    common = common.join(frozen[[*prob_columns(op.FROZEN_ARM), "involves_returning_team"]])
    common = common.join(online[[*prob_columns(op.ONLINE_ARM), "on_first_target_date"]])
    gap = op.first_date_gap(common)
    if gap > ocfg["segments"]["first_date_invariant_tolerance"]:
        raise op.OnlineCheckError(f"{target}: online differs from frozen on the first target date by {gap:.3e}")
    fitted = {"target": target, "history": list(history), "n_full": diag_fitted["n_full"],
              "n_common": int(len(common)), "unseen_teams": online_fitted["unseen_teams"],
              "returning_teams": frozen_fitted["returning_teams"],
              "reproduction_max_abs_diff": {"experiment_11": repro11, "experiment_12": repro12},
              "first_date_max_abs_diff": gap, "frozen_fit": frozen_fitted["arms"][op.FROZEN_ARM],
              "online": online_fitted}
    return common, fitted


def score_fold(common: pd.DataFrame, results, target: str, ocfg: dict) -> tuple[dict, dict, np.ndarray]:
    clusters = date_clusters(common, target)
    losses = op.losses_for(common, results)
    returning = common["involves_returning_team"].to_numpy(dtype=bool)
    block = op.score_block(clusters, losses, ocfg)
    block["segments"] = op.segment_blocks(common, clusters, losses, ocfg)
    block["returning_vs_continuous"] = {
        "returning": op.differences(losses, [op.PRIMARY], clusters, returning)[op.PRIMARY],
        "continuous": op.differences(losses, [op.PRIMARY], clusters, ~returning)[op.PRIMARY]}
    block["mean_abs_prob_change_vs_frozen"] = op.mean_abs_prob_change(common)
    block["sharpness_and_calibration"] = {a: tw.sharpness_and_calibration(common, results, a) for a in op.ARMS}
    return block, losses, clusters


# --- Historical stage ----------------------------------------------------------------------------

def load_historical_matches(max_season: str) -> pd.DataFrame:
    """dev_v2, checksum-verified, cut to seasons <= max_season before any validation or use."""
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    matches = tw.restrict_to_development(load_matches(PROCESSED_DEV_V2, validate=False), max_season)
    validate_matches(matches)
    validate_season_dates(matches)
    return matches


def run_historical(write: bool = True) -> dict:
    ocfg, _, vcfg, dcfg, _ = load_all_configs()
    p, r, u = ocfg["protocol"], ocfg["reproduction"], ocfg["criterion_u"]
    max_season = ocfg["historical_stage"]["max_season"]
    matches = load_historical_matches(max_season)
    tw.assert_development_only(matches, max_season)
    targets = list(p["historical_targets"])
    rec11 = load_recorded(ocfg, "experiment_11_predictions", "experiment_11_sha256", targets)
    rec12 = as_frozen_arm(load_recorded(ocfg, "experiment_12_development_predictions",
                                        "experiment_12_development_sha256", targets),
                          r["experiment_12_development_column"])

    # Stage 1: every prediction and every pre-scoring check, before any outcome is joined.
    staged = []
    for history, target in selection_folds(targets):
        tw.assert_development_only(matches[matches["Season"].isin([*history, target])], max_season)
        staged.append((history, target, *fold_predictions(matches, history, target, ocfg, vcfg, dcfg, rec11, rec12)))

    # Stage 2: scoring.
    folds, all_losses, all_clusters, clusters_by_target, primary, frames = [], [], [], {}, {}, []
    for history, target, common, fitted in staged:
        results = outcomes_for(matches, common.index)
        block, losses, clusters = score_fold(common, results, target, ocfg)
        folds.append({"target": target, "history": f"{history[0]}..{history[-1]}",
                      "n_history_seasons": len(history), "fitted": fitted, **block})
        all_losses.append(losses)
        all_clusters.append(clusters)
        clusters_by_target[target] = clusters
        primary[target] = {m: losses[op.ONLINE_ARM][m] - losses[op.FROZEN_ARM][m] for m in op.METRICS}
        frames.append(common.assign(fold_target=target))

    pooled_preds = pd.concat(frames)
    pooled_losses = op.concat_losses(all_losses)
    pooled_clusters = np.concatenate(all_clusters)
    pooled = op.score_block(pooled_clusters, pooled_losses, ocfg)
    pooled["segments"] = op.segment_blocks(pooled_preds, pooled_clusters, pooled_losses, ocfg)
    returning = pooled_preds["involves_returning_team"].to_numpy(dtype=bool)
    pooled["returning_vs_continuous"] = {
        "returning": op.differences(pooled_losses, [op.PRIMARY], pooled_clusters, returning)[op.PRIMARY],
        "continuous": op.differences(pooled_losses, [op.PRIMARY], pooled_clusters, ~returning)[op.PRIMARY]}
    u_result = op.criterion_u(primary, clusters_by_target, u["practical_floor_log_loss"], u["clustered_se_multiple"],
                              u["min_folds_same_sign"])
    labels = ocfg["segments"]["labels"]
    accumulation = op.accumulation_pattern({lab: pooled["segments"][lab][op.PRIMARY]["log_loss"] for lab in labels},
                                           labels, u["clustered_se_multiple"])
    result = {
        "protocol_id": p["protocol_id"],
        "stage": "historical",
        "seasons_loaded": sorted(matches["Season"].unique(), key=SEASON_ORDER.index),
        "half_life_days": float(ocfg["online"]["half_life_days"]),
        "criterion_u": u_result,
        "reading": u_result["reading"],
        "accumulation_pattern": accumulation,
        "pooled": pooled,
        "fits_total": int(sum(f["fitted"]["online"]["n_fits"] for f in folds)),
        "retries_total": int(sum(f["fitted"]["online"]["n_retried"] for f in folds)),
        "note": "H = 730 was selected on these folds for the frozen arm; any selection bias favours the frozen arm.",
        "folds": folds,
    }
    if write:
        result["predictions"] = write_predictions(ocfg["historical_stage"]["results_name"], pooled_preds)
        write_results(ocfg["historical_stage"]["results_name"], result, data_path=PROCESSED_DEV_V2)
    return result


# --- Validation stage (2024-25, diagnostic only) ---------------------------------------------------

def check_historical_lock(ocfg: dict) -> dict:
    """Return the locked historical results or raise LockError. Reads only the historical metrics and git state."""
    lock = ocfg["historical_locked"]
    if lock.get("status") != "locked":
        raise LockError("The historical results are not locked: [historical_locked] status is not 'locked'.")
    path = RESULTS_DIR / ocfg["historical_stage"]["results_name"] / "metrics.json"
    if not path.exists() or content_sha256(path) != lock["historical_metrics_sha256"]:
        raise LockError("Historical metrics are missing or differ from [historical_locked] historical_metrics_sha256.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["results"].get("stage") != "historical":
        raise LockError("The locked metrics file is not a historical-stage result.")
    prov = payload["provenance"]
    if prov.get("git_dirty") is not False or prov.get("git_commit") != lock["historical_commit"]:
        raise LockError("The historical stage did not run from the clean commit recorded in [historical_locked].")
    if not is_ancestor_of_head(lock["historical_commit"]):
        raise LockError("HEAD does not descend from the historical-stage commit.")
    if not working_tree_clean():
        raise LockError("Working tree has uncommitted changes outside results/: commit the lock first.")
    return payload["results"]


def run_validation(write: bool = True) -> dict:
    ocfg, _, vcfg, dcfg, _ = load_all_configs()
    historical = check_historical_lock(ocfg)
    p, r = ocfg["protocol"], ocfg["reproduction"]
    target = p["validation_target"]
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    from eplmodel.data import load_dev_matches
    matches = load_dev_matches()
    (history, tgt), = selection_folds([target])
    rec11 = load_recorded(ocfg, "experiment_11_predictions", "experiment_11_sha256", [tgt])
    rec12 = as_frozen_arm(load_recorded(ocfg, "experiment_12_validation_predictions",
                                        "experiment_12_validation_sha256", [tgt]),
                          r["experiment_12_validation_column"])

    # Predictions and every check before any outcome is joined.
    common, fitted = fold_predictions(matches, history, tgt, ocfg, vcfg, dcfg, rec11, rec12)

    # Scoring.
    results = outcomes_for(matches, common.index)
    block, _, _ = score_fold(common, results, tgt, ocfg)
    historical_mean = historical["criterion_u"]["pooled"]["log_loss"]["mean"]
    result = {
        "protocol_id": p["protocol_id"],
        "stage": "validation",
        "target": tgt,
        "evidence_class": p["validation_evidence_class"],
        "half_life_days": float(ocfg["online"]["half_life_days"]),
        "label": op.validation_label(block["primary"]["log_loss"], historical_mean,
                                     ocfg["validation"]["clustered_se_multiple"]),
        "historical_reading": historical["reading"],
        "historical_pooled_mean_log_loss": historical_mean,
        "fitted": fitted,
        **block,
        "note": "Diagnostic observation only: 2024-25 cannot select, tune or confirm anything and is not "
                "pooled with the historical folds.",
    }
    if write:
        result["predictions"] = write_predictions(ocfg["validation_stage"]["results_name"],
                                                  common.assign(fold_target=tgt))
        write_results(ocfg["validation_stage"]["results_name"], result, data_path=PROCESSED_DEV_V2)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("historical", "validation"))
    args = parser.parse_args(argv)
    if args.stage == "historical":
        out = run_historical()
        print(f"Historical reading: {out['reading']} ({out['fits_total']} online fits). "
              "Lock the historical results in a separate commit before the validation stage.")
    else:
        out = run_validation()
        print(f"2024-25 (diagnostic only): {out['label']}")


if __name__ == "__main__":
    main()
