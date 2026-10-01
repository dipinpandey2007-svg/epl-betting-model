"""Experiment 2a: choose the Elo K factor by walk-forward validation inside the training seasons."""

from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.evaluation.walk_forward import select_elo_k
from eplmodel.reporting.results import write_results
from eplmodel.splits import TRAIN_SEASONS, expanding_window_folds

NAME = "elo_k_selection"


def run(write: bool = True) -> dict:
    cfg = load_config()["elo"]
    matches = load_matches()
    folds = expanding_window_folds(TRAIN_SEASONS, cfg["min_train_seasons"])
    table, best_k = select_elo_k(matches, cfg["candidate_k"], folds, cfg["logistic_C"])
    result = {
        "folds": [{"train": list(tr), "validate": val} for tr, val in folds],
        "k_table": table,
        "best_k": int(best_k),
        "configured_k": cfg["selected_k"],
        "configured_k_matches_selection": int(best_k) == cfg["selected_k"],
    }
    if write:
        write_results(NAME, result)
    return result


if __name__ == "__main__":
    out = run()
    print(out["k_table"].to_string(index=False))
    print(f"Best K: {out['best_k']} (configured: {out['configured_k']})")
