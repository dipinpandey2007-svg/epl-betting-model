"""Experiment 10 (protocol validation_2425_v1): the established specs on the six selection folds.

Pre-registered in configs/validation_2425_v1.toml and docs/TEST_SET_ACCESS_LOG.md entry V1
before any 2024-25 prediction was generated.

- Targets 2017-18 .. 2021-22 and 2024-25, each predicted from every development season
  before it (eplmodel.splits.selection_folds). 2022-24 is history only.
- 2024-25 is the primary new validation result. Folds 1-5 were used during development and
  for selecting K = 25, so the pooled six-fold figures are descriptive context, not an
  unbiased estimate.
- Information policies and groups: eplmodel.evaluation.validation.
- Nothing about the established specifications may be changed on the basis of these results.

Not part of experiments.run_all until its first run has been reviewed.
"""

import pandas as pd

from eplmodel.config import load_config
from eplmodel.data import load_dev_matches
from eplmodel.data.checksums import load_manifest, verify_file
from eplmodel.evaluation.validation import (
    fold_predictions,
    home_win_calibration,
    outcomes_for,
    score_groups,
)
from eplmodel.models.poisson import FORMULA
from eplmodel.paths import CONFIG_DIR, PROCESSED_DEV_V2, PROJECT_ROOT
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import SELECTION_VALIDATION_SEASONS, selection_folds

NAME = "validation_2425"
VALIDATION_CONFIG = CONFIG_DIR / "validation_2425_v1.toml"


class ProtocolMismatchError(RuntimeError):
    """The pre-registered protocol disagrees with the frozen specifications, the splits or the data."""


def check_protocol(vcfg: dict, base: dict) -> None:
    """Refuse to run unless the pre-registration agrees with baselines_v1, the selection folds and the data."""
    elo, b_elo = vcfg["elo"], base["elo"]
    dc, b_dc = vcfg["dixon_coles"], base["dixon_coles"]
    expected = {
        "elo.spec_id": (elo["spec_id"], b_elo["spec_id"]),
        "elo.k": (elo["k"], b_elo["selected_k"]),
        "elo.initial_rating": (elo["initial_rating"], b_elo["initial_rating"]),
        "elo.update_home_advantage": (elo["update_home_advantage"], b_elo["update_home_advantage"]),
        "elo.apply_home_shift": (elo["apply_home_shift"], b_elo["apply_home_shift"]),
        "elo.logistic_C": (elo["logistic_C"], b_elo["logistic_C"]),
        "elo.logistic_tol": (elo["logistic_tol"], 1e-4),
        "poisson.spec_id": (vcfg["poisson"]["spec_id"], base["poisson"]["spec_id"]),
        "poisson.formula": (vcfg["poisson"]["formula"], base["poisson"]["formula"]),
        "poisson.formula_in_code": (vcfg["poisson"]["formula"], FORMULA),
        "poisson.max_goals": (vcfg["poisson"]["max_goals"], base["poisson"]["max_goals"]),
        "dixon_coles.spec_id": (dc["spec_id"], b_dc["spec_id"]),
        "dixon_coles.fit": (dc["fit"], b_dc["fit"]),
        "dixon_coles.rho_grid": ((dc["rho_grid_start"], dc["rho_grid_stop"], dc["rho_grid_step"]),
                                 (b_dc["rho_grid_start"], b_dc["rho_grid_stop"], b_dc["rho_grid_step"])),
        "dixon_coles.rho_reference": (dc["rho_reference_experiment_5"], b_dc["selected_rho"]),
        "baseline.spec_id": (vcfg["baseline"]["spec_id"], base["baseline"]["spec_id"]),
        "targets": (tuple(vcfg["protocol"]["targets"]), SELECTION_VALIDATION_SEASONS),
        "data_sha256": (vcfg["protocol"]["data_sha256"],
                        load_manifest()[PROCESSED_DEV_V2.relative_to(PROJECT_ROOT).as_posix()]),
    }
    wrong = {k: v for k, v in expected.items() if v[0] != v[1]}
    if wrong:
        raise ProtocolMismatchError(f"Pre-registration disagrees with frozen values: {wrong}")
    if vcfg["dixon_coles"]["rho_estimation"] != "reestimate_on_history":
        raise ProtocolMismatchError("Only the registered rho policy (reestimate_on_history) is implemented.")


def check_primary_counts(fitted: dict, vcfg: dict) -> None:
    """The registered 2024-25 group sizes must hold before anything is scored."""
    g = vcfg["groups"]
    actual = (fitted["n_full"], fitted["n_common"], fitted["goal_models"]["unseen_teams"])
    registered = (g["expected_2425_full"], g["expected_2425_common"], g["expected_2425_unseen_teams"])
    if actual != registered:
        raise ProtocolMismatchError(f"2024-25 groups {actual} differ from the registration {registered}")


def run(write: bool = True) -> dict:
    vcfg = load_config(VALIDATION_CONFIG)
    check_protocol(vcfg, load_config())
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    matches = load_dev_matches()
    protocol = vcfg["protocol"]
    refs = vcfg["metrics"]["paired_reference_models"]

    folds, all_preds = [], []
    for history, target in selection_folds(protocol["targets"]):
        preds, fitted = fold_predictions(matches, history, target, vcfg)
        if target == protocol["primary_target"]:
            check_primary_counts(fitted, vcfg)
        results = outcomes_for(matches, preds.index)
        folds.append({
            "target": target,
            "role": ("primary_validation" if target == protocol["primary_target"]
                     else "historical_fold_used_for_development_and_k_selection"),
            "history": f"{history[0]}..{history[-1]}",
            "fitted": fitted,
            "groups": score_groups(preds, results, refs),
            "_preds": preds,
            "_results": results,
        })
        all_preds.append(preds.assign(fold_target=target))

    primary = next(f for f in folds if f["target"] == protocol["primary_target"])
    pooled_preds = pd.concat([f["_preds"] for f in folds])
    pooled_results = pd.concat([f["_results"] for f in folds])
    bins = vcfg["metrics"]["calibration_bins"]

    result = {
        "protocol_id": protocol["protocol_id"],
        "primary_target": protocol["primary_target"],
        "primary_2425": {
            "groups": primary["groups"],
            "fitted": primary["fitted"],
            "home_win_calibration": home_win_calibration(primary["_preds"], primary["_results"], bins),
        },
        "folds": [{k: v for k, v in f.items() if not k.startswith("_")} for f in folds],
        "pooled_six_folds": {
            "descriptive_only": True,
            "unbiased_estimate": protocol["pooled_is_unbiased"],
            "note": "K = 25 was selected on folds 2017-18..2021-22, so this aggregate is not an unbiased "
                    "out-of-sample estimate. Common groups are the union of each fold's common matches.",
            "groups": score_groups(pooled_preds, pooled_results, refs),
            "home_win_calibration": home_win_calibration(pooled_preds, pooled_results, bins),
        },
    }
    if write:
        result["predictions"] = write_predictions(NAME, pd.concat(all_preds))
        write_results(NAME, result, data_path=PROCESSED_DEV_V2)
    return result


if __name__ == "__main__":
    out = run()
    for group, block in out["primary_2425"]["groups"].items():
        for model, s in block["scores"].items():
            print(f"2024-25 {group:6s} {model:20s} n={s['n_matches']:3d}  "
                  f"log loss {s['log_loss']:.4f}  Brier {s['brier']:.4f}")
