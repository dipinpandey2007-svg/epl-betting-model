"""Synthetic-data snapshot of the library code behind the recorded Experiments 10-13.

compute_snapshot() runs the prediction and scoring functions used by the recorded protocols on
the synthetic league of test_validation (no real data) and returns every output at full
precision. tests/golden/recorded_synthetic_snapshot.json holds the snapshot captured from the
code as it was BEFORE the common evaluation harness was introduced; test_evaluation_harness.py
requires the current code to reproduce it exactly. This is the reproduction gate that CI can run.

Regenerate only for a deliberate, documented methodological change:
    python tests/recorded_snapshot.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eplmodel.config import load_config
from eplmodel.evaluation import online_poisson as op
from eplmodel.evaluation import time_weighting as tw
from eplmodel.evaluation import update_policy as up
from eplmodel.evaluation.validation import fold_predictions, home_win_calibration, outcomes_for, score_groups
from eplmodel.reporting.results import _jsonable

SNAPSHOT_FILE = Path(__file__).parent / "golden" / "recorded_synthetic_snapshot.json"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
VCFG = load_config(CONFIG_DIR / "validation_2425_v1.toml")
DCFG = load_config(CONFIG_DIR / "update_policy_diagnostic_v1.toml")
OCFG = load_config(CONFIG_DIR / "online_tw_poisson_diagnostic_v1.toml")
# Two synthetic folds: an unseen team in the first target, the same team as history in the second.
FOLDS = ((("1415", "1516", "1617"), "1718"), (("1415", "1516", "1617", "1718"), "1819"))
TW_ARMS = {"poisson": math.inf, "poisson_tw_h365": 365.0, "poisson_tw_h730": 730.0}
NONDETERMINISTIC = {"runtime_seconds"}


def _clusters(preds: pd.DataFrame, target: str) -> np.ndarray:
    return (target + "_" + pd.to_datetime(preds["Date"]).dt.strftime("%Y-%m-%d")).to_numpy()


def _frame(df: pd.DataFrame) -> dict:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return {"index": [str(i) for i in df.index], "columns": [str(c) for c in df.columns],
            "data": df.to_numpy(dtype=object).tolist()}


def normalise(obj):
    """JSON-compatible form in which equal values compare equal (NaN and inf become strings)."""
    if isinstance(obj, pd.DataFrame):
        return normalise(_frame(obj))
    if isinstance(obj, dict):
        return {str(k): normalise(v) for k, v in obj.items() if k not in NONDETERMINISTIC}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [normalise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return normalise(obj.tolist())
    if isinstance(obj, (np.generic,)):
        return normalise(obj.item())
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    return _jsonable(obj)


# Floats are compared with math.isclose at the registered reproduction tolerance (1e-12). The fits behind the
# snapshot go through BLAS, whose CPU-specific code paths and library versions change results in the last bits
# (observed: up to ~1e-13); everything that is not a float must match exactly.
FLOAT_REL_TOL = 1e-12
FLOAT_ABS_TOL = 1e-12


def snapshot_mismatches(actual, expected, rel_tol: float = FLOAT_REL_TOL, abs_tol: float = FLOAT_ABS_TOL,
                        path: str = "") -> list[str]:
    """Every difference between two snapshot structures, as 'path: description' strings.

    Structure (dict keys, list lengths), types and every non-float value must be identical; floats must agree
    within math.isclose(rel_tol, abs_tol). No type is coerced: an int never equals a float, a bool never an int.
    """
    where = path or "<root>"
    if type(actual) is not type(expected):
        return [f"{where}: type {type(actual).__name__} != {type(expected).__name__} "
                f"({actual!r:.80} vs {expected!r:.80})"]
    if isinstance(actual, dict):
        out = []
        missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
        if missing:
            out.append(f"{where}: missing keys {missing}")
        if extra:
            out.append(f"{where}: unexpected keys {extra}")
        for key in expected:
            if key in actual:
                out += snapshot_mismatches(actual[key], expected[key], rel_tol, abs_tol, f"{path}/{key}")
        return out
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return [f"{where}: length {len(actual)} != {len(expected)}"]
        out = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            out += snapshot_mismatches(a, e, rel_tol, abs_tol, f"{path}[{i}]")
        return out
    if isinstance(actual, float):
        if math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            return []
        return [f"{where}: {actual!r} != {expected!r} (|diff| {abs(actual - expected):.3e})"]
    return [] if actual == expected else [f"{where}: {actual!r:.80} != {expected!r:.80}"]


def assert_snapshot_close(actual, expected, max_reported: int = 5) -> None:
    """Raise AssertionError listing the number of mismatches and only the first `max_reported` of them."""
    mismatches = snapshot_mismatches(actual, expected)
    if mismatches:
        shown = "\n  ".join(mismatches[:max_reported])
        more = len(mismatches) - max_reported
        raise AssertionError(f"{len(mismatches)} snapshot mismatch(es):\n  {shown}"
                             + (f"\n  ... and {more} more" if more > 0 else ""))


def compute_snapshot() -> dict:
    from test_validation import synthetic_league

    league = synthetic_league()
    out = {}
    for history, target in FOLDS:
        fold = {}
        # Experiment 10
        preds, fitted = fold_predictions(league, history, target, VCFG)
        results = outcomes_for(league, preds.index)
        fold["exp10"] = {"preds": preds, "fitted": fitted,
                         "groups": score_groups(preds, results, VCFG["metrics"]["paired_reference_models"]),
                         "calibration": home_win_calibration(preds, results, 10)}
        # Experiment 11
        diag, dfitted = up.diagnostic_fold_predictions(league, history, target, VCFG, DCFG)
        clusters = _clusters(diag, target)
        fold["exp11"] = {"preds": diag, "fitted": dfitted,
                         "scored": up.score_fold(diag, outcomes_for(league, diag.index), clusters, DCFG)}
        # Experiment 12
        twp, twf = tw.tw_fold_predictions(league, history, target, TW_ARMS, 10)
        tres = outcomes_for(league, twp.index)
        losses = tw.per_match_losses(twp, tres, list(TW_ARMS))
        fold["exp12"] = {"preds": twp, "fitted": twf, "losses": losses,
                         "sharpness": {a: tw.sharpness_and_calibration(twp, tres, a) for a in TW_ARMS}}
        # Experiment 13
        online, ofitted = op.online_fold_predictions(league, history, target, 730.0, 10, 100, 1000)
        frozen, _ = tw.tw_fold_predictions(league, history, target, {op.FROZEN_ARM: 730.0}, 10)
        common = diag[diag["in_common"].astype(bool)]
        common = common.join(frozen[[*[f"{op.FROZEN_ARM}_{o}" for o in "HDA"], "involves_returning_team"]])
        common = common.join(online[[*[f"{op.ONLINE_ARM}_{o}" for o in "HDA"], "on_first_target_date"]])
        cres = outcomes_for(league, common.index)
        cclusters = _clusters(common, target)
        closses = op.losses_for(common, cres)
        block = op.score_block(cclusters, closses, OCFG)
        fold["exp13"] = {"online": online, "fitted": ofitted, "first_date_gap": op.first_date_gap(common),
                         "block": block, "segments": op.segment_blocks(common, cclusters, closses, OCFG),
                         "mean_abs_prob_change": op.mean_abs_prob_change(common),
                         "sharpness": {a: tw.sharpness_and_calibration(common, cres, a) for a in op.ARMS}}
        out[target] = fold
    return normalise(out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    SNAPSHOT_FILE.write_text(json.dumps(compute_snapshot(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {SNAPSHOT_FILE}")
