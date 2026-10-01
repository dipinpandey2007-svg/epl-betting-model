"""Walk-forward selection of the Elo K factor (training seasons only)."""

from collections.abc import Sequence

import pandas as pd

from eplmodel.evaluation.metrics import multiclass_log_loss
from eplmodel.models.elo import EloOutcomeModel, elo_difference, home_shift_from_results, run_elo
from eplmodel.splits import assert_no_dev_test


def elo_fold_log_loss(
    matches: pd.DataFrame, k: float, train_seasons: Sequence[str], val_season: str, logistic_C: float = 1.0
) -> float:
    """Validation log loss of the Elo outcome model for one fold.

    Ratings are run through the full match table with home_adv=0. This is
    leakage-safe because only pre-match ratings are used, and the ratings of
    fold rows depend only on earlier matches. The home shift and the logistic
    regression are fitted on the fold's training seasons only.
    """
    assert_no_dev_test([*train_seasons, val_season])
    hist = run_elo(matches, k=k, home_adv=0.0)
    train = hist["Season"].isin(train_seasons).to_numpy()
    val = (hist["Season"] == val_season).to_numpy()
    shift = home_shift_from_results(hist.loc[train, "FTR"])
    diff = elo_difference(hist, shift)
    model = EloOutcomeModel(C=logistic_C).fit(diff[train], hist.loc[train, "FTR"])
    return multiclass_log_loss(hist.loc[val, "FTR"], model.predict_proba(diff[val]))


def select_elo_k(
    matches: pd.DataFrame,
    candidate_ks: Sequence[float],
    folds: Sequence[tuple[Sequence[str], str]],
    logistic_C: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    """Mean fold log loss for each K, and the K with the lowest mean."""
    rows = []
    for k in candidate_ks:
        scores = [elo_fold_log_loss(matches, k, tr, val, logistic_C) for tr, val in folds]
        rows.append({"K": k, "mean_log_loss": sum(scores) / len(scores),
                     **{f"val_{val}": s for (_, val), s in zip(folds, scores)}})
    table = pd.DataFrame(rows)
    return table, table.loc[table["mean_log_loss"].idxmin(), "K"]
