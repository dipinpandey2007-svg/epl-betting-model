"""Update-policy diagnostic (protocol update_policy_diagnostic_v1): online vs season-start Elo vs goal models.

Pre-registered in configs/update_policy_diagnostic_v1.toml and docs/TEST_SET_ACCESS_LOG.md entry V2
before any diagnostic prediction was generated.

- The six selection folds of validation_2425_v1; 2024-25 is the primary result, folds 2017-18 .. 2021-22
  historical context, and the pooled figure descriptive only (K = 25 was selected on folds 1-5).
- Arms: online Elo, F1 (frozen ratings, online layer), F2 (frozen ratings, own season-start layer),
  static Poisson, staged Dixon-Coles, frequency baseline (eplmodel.evaluation.update_policy).
- Decomposition on the common group:
  poisson - elo = (poisson - elo_f2) + (elo_f2 - elo_f1) + (elo_f1 - elo),
  with elo_f1 - elo the updating effect.
- Before anything is scored: the protocol is checked against the frozen specs, the established
  arms must reproduce the recorded validation_2425_v1 predictions, and every fold's registered
  group sizes must hold.
- Diagnostic only: nothing is selected, tuned or changed because of these results.

Not part of experiments.run_all.
"""

import numpy as np
import pandas as pd

from eplmodel.config import load_config
from eplmodel.data import load_dev_matches
from eplmodel.data.checksums import content_sha256, load_manifest, verify_file
from eplmodel.evaluation import update_policy as up
from eplmodel.evaluation.validation import outcomes_for, prob_columns
from eplmodel.paths import CONFIG_DIR, HOLDOUT_CONFIG, PROCESSED_DEV_V2, PROJECT_ROOT
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import REGISTERED_DEV_TEST_SPECS, SELECTION_VALIDATION_SEASONS, selection_folds
from experiments import validation_2425

NAME = "update_policy_diagnostic"
DIAGNOSTIC_CONFIG = CONFIG_DIR / "update_policy_diagnostic_v1.toml"
DIAGNOSTIC_ARMS = ("elo_f1", "elo_f2")


class ProtocolMismatchError(RuntimeError):
    """The pre-registration disagrees with the frozen specifications, the code, the splits or the data."""


def check_protocol(dcfg: dict, vcfg: dict, base: dict) -> None:
    """Refuse to run unless the pre-registration agrees with validation_2425_v1, baselines_v1 and the code."""
    validation_2425.check_protocol(vcfg, base)
    p, arms, ss, f2, dec = (dcfg["protocol"], dcfg["arms"], dcfg["season_start_elo"], dcfg["elo_f2"],
                            dcfg["decomposition"])
    elo = vcfg["elo"]
    expected = {
        "established_protocol": (p["established_protocol"], vcfg["protocol"]["protocol_id"]),
        "diagnostic_only": (p["diagnostic_only"], True),
        "targets": (tuple(p["targets"]), SELECTION_VALIDATION_SEASONS),
        "targets_v1": (tuple(p["targets"]), tuple(vcfg["protocol"]["targets"])),
        "primary_target": (p["primary_target"], vcfg["protocol"]["primary_target"]),
        "pooled_is_unbiased": (p["pooled_is_unbiased"], False),
        "data_sha256": (p["data_sha256"], vcfg["protocol"]["data_sha256"]),
        "data_sha256_manifest": (p["data_sha256"],
                                 load_manifest()[PROCESSED_DEV_V2.relative_to(PROJECT_ROOT).as_posix()]),
        "arms.elo": (arms["elo"], base["elo"]["spec_id"]),
        "arms.poisson": (arms["poisson"], base["poisson"]["spec_id"]),
        "arms.dixon_coles": (arms["dixon_coles"], base["dixon_coles"]["spec_id"]),
        "arms.frequency_baseline": (arms["frequency_baseline"], base["baseline"]["spec_id"]),
        "arms.order": (tuple(arms), up.ARMS),
        "season_start_elo.k": (ss["k"], elo["k"]),
        "season_start_elo.initial_rating": (ss["initial_rating"], elo["initial_rating"]),
        "season_start_elo.update_home_advantage": (ss["update_home_advantage"], elo["update_home_advantage"]),
        "elo_f1.layer": (dcfg["elo_f1"]["layer"], "identical_to_online_fold_layer"),
        "elo_f2.layer": (f2["layer"], "fit_on_season_start_features"),
        "elo_f2.apply_home_shift": (f2["apply_home_shift"], elo["apply_home_shift"]),
        "elo_f2.logistic_C": (f2["logistic_C"], elo["logistic_C"]),
        "elo_f2.logistic_tol": (f2["logistic_tol"], elo["logistic_tol"]),
        "groups.full": (tuple(dcfg["groups"]["full_models"]), up.GROUP_MODELS["full"]),
        "groups.common": (tuple(dcfg["groups"]["common_models"]), up.GROUP_MODELS["common"]),
        "groups.unseen": (tuple(dcfg["groups"]["unseen_models"]), up.GROUP_MODELS["unseen"]),
        "decomposition.components": (tuple(dec["components"]), up.COMPONENTS),
        "decomposition.total": (dec["total"], up.TOTAL),
        "decomposition.full_group_components": (tuple(dec["full_group_components"]), up.FULL_GROUP_COMPONENTS),
        "decomposition.derived_subtotal": (dec["derived_subtotal"], up.SUBTOTAL),
        "decomposition.updating_effect": (dec["label_elo_f1_minus_elo"], up.LABELS["elo_f1_minus_elo"]),
        "decomposition.layer_effect": (dec["label_elo_f2_minus_elo_f1"], up.LABELS["elo_f2_minus_elo_f1"]),
        "decomposition.family": (dec["label_poisson_minus_elo_f2"], up.LABELS["poisson_minus_elo_f2"]),
        "decomposition.metrics": (tuple(dec["metrics"]), up.METRICS),
        "outputs.in_run_all": (dcfg["outputs"]["in_run_all"], False),
    }
    wrong = {k: v for k, v in expected.items() if v[0] != v[1]}
    if wrong:
        raise ProtocolMismatchError(f"Pre-registration disagrees with frozen values or code: {wrong}")
    if ss["between_season_regression"] != "none" or ss["target_ratings"] != "end_of_history_frozen_all_season":
        raise ProtocolMismatchError("Only the registered season-start rating policy is implemented.")
    if f2["exclude_first_history_season"] is not True:
        raise ProtocolMismatchError("Only the registered F2 layer rule (first history season excluded) is implemented.")
    diag_specs = {arms[a] for a in DIAGNOSTIC_ARMS}
    holdout_specs = set(load_config(HOLDOUT_CONFIG)["registration"]["spec_ids"])
    if diag_specs & (REGISTERED_DEV_TEST_SPECS | holdout_specs):
        raise ProtocolMismatchError("Diagnostic arms must not be registered for dev-test or holdout scoring.")


def check_fold_counts(target: str, preds: pd.DataFrame, fitted: dict, dcfg: dict) -> None:
    """The registered group sizes and unseen teams must hold before anything is scored (fixtures only)."""
    g = dcfg["groups"]
    actual = (fitted["n_full"], fitted["n_common"], fitted["goal_models"]["unseen_teams"])
    registered = (g["expected_full"][target], g["expected_common"][target], g["expected_unseen_teams"][target])
    if actual != registered or fitted["n_unseen"] != fitted["n_full"] - fitted["n_common"]:
        raise ProtocolMismatchError(f"{target}: groups {actual} differ from the registration {registered}")


def load_recorded_predictions(dcfg: dict) -> pd.DataFrame:
    """The validation_2425_v1 predictions, verified against their registered checksum."""
    rep = dcfg["reproduction"]
    path = PROJECT_ROOT / rep["predictions_file"]
    if not path.exists() or content_sha256(path) != rep["predictions_sha256"]:
        raise ProtocolMismatchError(f"{rep['predictions_file']} is missing or differs from its registered checksum")
    return pd.read_csv(path, index_col="match_id", dtype={"Season": str, "fold_target": str})


def check_reproduction(target: str, preds: pd.DataFrame, recorded: pd.DataFrame, dcfg: dict) -> float:
    """Established arms must equal the recorded predictions row for row; returns the largest difference."""
    rep = dcfg["reproduction"]
    rec = recorded[recorded["fold_target"] == target]
    if list(rec.index) != list(preds.index) or not np.array_equal(rec["in_common"].to_numpy(dtype=bool),
                                                                   preds["in_common"].to_numpy(dtype=bool)):
        raise ProtocolMismatchError(f"{target}: matches or groups differ from the recorded predictions")
    cols = [c for m in rep["models"] for c in prob_columns(m)]
    new, old = preds[cols].to_numpy(dtype=float), rec[cols].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(new), np.isnan(old)):
        raise ProtocolMismatchError(f"{target}: missing predictions differ from the recorded ones")
    worst = float(np.nanmax(np.abs(new - old))) if new.size else 0.0
    if worst > rep["tolerance"]:
        raise ProtocolMismatchError(f"{target}: established predictions differ from the record by {worst:.3e}")
    return worst


def date_clusters(preds: pd.DataFrame, target: str) -> np.ndarray:
    return (target + "_" + pd.to_datetime(preds["Date"]).dt.strftime("%Y-%m-%d")).to_numpy()


def run(write: bool = True) -> dict:
    dcfg = load_config(DIAGNOSTIC_CONFIG)
    vcfg = load_config(validation_2425.VALIDATION_CONFIG)
    check_protocol(dcfg, vcfg, load_config())
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    recorded = load_recorded_predictions(dcfg)
    matches = load_dev_matches()
    protocol = dcfg["protocol"]

    # Stage 1: predictions and every pre-scoring check, for all folds, before any outcome is joined.
    staged = []
    for history, target in selection_folds(protocol["targets"]):
        preds, fitted = up.diagnostic_fold_predictions(matches, history, target, vcfg, dcfg)
        check_fold_counts(target, preds, fitted, dcfg)
        fitted["reproduction_max_abs_diff"] = check_reproduction(target, preds, recorded, dcfg)
        staged.append((history, target, preds, fitted))

    # Stage 2: scoring.
    folds, all_preds, all_results, all_clusters = [], [], [], []
    for history, target, preds, fitted in staged:
        results = outcomes_for(matches, preds.index)
        clusters = date_clusters(preds, target)
        folds.append({
            "target": target,
            "role": ("primary_validation" if target == protocol["primary_target"]
                     else "historical_fold_used_for_development_and_k_selection"),
            "history": f"{history[0]}..{history[-1]}",
            "fitted": fitted,
            "groups": up.score_fold(preds, results, clusters, dcfg),
        })
        all_preds.append(preds.assign(fold_target=target))
        all_results.append(results)
        all_clusters.append(clusters)

    pooled_preds = pd.concat(all_preds)
    primary = next(f for f in folds if f["target"] == protocol["primary_target"])
    result = {
        "protocol_id": protocol["protocol_id"],
        "diagnostic_only": True,
        "primary_target": protocol["primary_target"],
        "sign_convention": dcfg["decomposition"]["sign"],
        "identity": dcfg["decomposition"]["identity"],
        "labels": up.LABELS,
        "primary_2425": {"groups": primary["groups"], "fitted": primary["fitted"]},
        "folds": folds,
        "pooled_six_folds": {
            "descriptive_only": True,
            "unbiased_estimate": protocol["pooled_is_unbiased"],
            "note": "K = 25 was selected on folds 2017-18..2021-22, so this aggregate is not an unbiased "
                    "out-of-sample estimate. Clusters are (fold target, date).",
            "groups": up.score_fold(pooled_preds, pd.concat(all_results), np.concatenate(all_clusters), dcfg),
        },
    }
    if write:
        result["predictions"] = write_predictions(NAME, pooled_preds)
        write_results(NAME, result, data_path=PROCESSED_DEV_V2)
    return result


if __name__ == "__main__":
    out = run()
    for name, block in out["primary_2425"]["groups"]["common"]["decomposition"].items():
        ll = block["log_loss"]
        print(f"2024-25 common {name:24s} {str(block['label']):36s} log loss {ll['mean']:+.4f} "
              f"(clustered SE {ll['clustered_se']:.4f})")
