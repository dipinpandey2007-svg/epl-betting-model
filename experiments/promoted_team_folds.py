"""Experiment 7: size the unseen/promoted-team problem inside the training-season walk-forward folds."""

from eplmodel.analysis.promoted_teams import fold_summary, load_team_history
from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.reporting.results import write_results
from eplmodel.splits import TRAIN_SEASONS, expanding_window_folds

NAME = "promoted_team_folds"

COUNT_COLUMNS = ["n_recent_yoyo", "n_long_absence_or_newcomer", "n_mixed", "n_total_affected"]


def run(write: bool = True) -> dict:
    cfg = load_config()["promoted_teams"]
    matches = load_matches()
    folds = expanding_window_folds(TRAIN_SEASONS, cfg["min_train_seasons"])
    summary = fold_summary(matches, folds, load_team_history(), cfg["yoyo_threshold"])
    result = {
        "yoyo_threshold": cfg["yoyo_threshold"],
        "folds": summary,
        "pooled": {c: int(summary[c].sum()) for c in COUNT_COLUMNS},
        "n_val_matches": int(summary["n_val_matches"].sum()),
    }
    if write:
        write_results(NAME, result)
    return result


if __name__ == "__main__":
    out = run()
    print(out["folds"].to_string(index=False))
    print("Pooled:", out["pooled"])
