"""Selection of the blend weight omega and the evidence rule of protocol shots_information_v1.

- select_omega: on the given targets, criterion(w) = mean over targets of each target's mean log loss
  (equal target weights); w_min = argmin (exact ties -> smaller w); one-SE rule towards the baseline: the
  SMALLEST w whose pooled per-match paired difference to w_min is <= se_multiple x its date-clustered SE.
- nested_selection: for each outer target, w is selected from the targets strictly before it.
- criterion_s: on the nested outer folds, (blend at nested w - baseline) per match: helps if the pooled
  log-loss mean < -floor, |mean| > k x clustered SE, negative in >= min_negative folds and pooled Brier < 0;
  the mirror reverses every sign.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from eplmodel.evaluation.scoring import paired_difference_clustered
from eplmodel.splits import SEASON_ORDER

HELPS = "historical_evidence_shots_on_target_add_information"
HURTS = "historical_evidence_shots_on_target_hurt"
NEITHER = "no_distinguishable_shot_information"


def select_omega(losses: Mapping[str, Mapping[float, np.ndarray]], clusters: Mapping[str, np.ndarray],
                 se_multiple: float) -> dict:
    """losses[target][w] = per-match log loss on the target's common group; clusters[target] = cluster labels."""
    targets = list(losses)
    if not targets:
        raise ValueError("at least one target is needed")
    grid = sorted(losses[targets[0]])
    if any(sorted(losses[t]) != grid for t in targets):
        raise ValueError("every target must be scored on the same omega grid")
    criterion = {w: float(np.mean([np.mean(losses[t][w]) for t in targets])) for w in grid}
    best = min(criterion.values())
    w_min = min(w for w in grid if criterion[w] == best)
    pooled_clusters = np.concatenate([np.asarray(clusters[t]) for t in targets])
    table = []
    for w in grid:
        d = paired_difference_clustered(np.concatenate([losses[t][w] for t in targets]),
                                        np.concatenate([losses[t][w_min] for t in targets]), pooled_clusters)
        eligible = w == w_min or bool(d["mean"] <= se_multiple * d["clustered_se"])
        table.append({"omega": w, "criterion_mean_log_loss": criterion[w], "diff_vs_omega_min": d["mean"],
                      "diff_vs_omega_min_clustered_se": d["clustered_se"], "within_one_se": eligible})
    w_sel = min(row["omega"] for row in table if row["within_one_se"])
    return {"targets": targets, "omega_min": w_min, "omega_selected": w_sel, "table": table}


def nested_selection(losses, clusters, outer_targets: Sequence[str], se_multiple: float) -> dict[str, dict]:
    order = sorted(losses, key=SEASON_ORDER.index)
    out = {}
    for outer in outer_targets:
        inner = [t for t in order if SEASON_ORDER.index(t) < SEASON_ORDER.index(outer)]
        out[outer] = select_omega({t: losses[t] for t in inner}, {t: clusters[t] for t in inner}, se_multiple)
    return out


def criterion_s(outer: Mapping[str, Mapping[str, np.ndarray]], clusters: Mapping[str, np.ndarray], floor: float,
                se_multiple: float, min_negative_folds: int) -> dict:
    """outer[target] = {'log_loss': d, 'brier': d}: per-match (blend at nested omega - baseline)."""
    targets = list(outer)
    cl = np.concatenate([np.asarray(clusters[t]) for t in targets])
    zeros = np.zeros(len(cl))
    pooled = {m: paired_difference_clustered(np.concatenate([outer[t][m] for t in targets]), zeros, cl)
              for m in ("log_loss", "brier")}
    means = {t: float(np.mean(outer[t]["log_loss"])) for t in targets}
    n_neg, n_pos = sum(v < 0 for v in means.values()), sum(v > 0 for v in means.values())
    ll, br = pooled["log_loss"], pooled["brier"]
    beyond = bool(abs(ll["mean"]) > se_multiple * ll["clustered_se"])
    helps = {"below_minus_floor": bool(ll["mean"] < -floor), "beyond_se_multiple": beyond,
             "enough_negative_folds": n_neg >= min_negative_folds, "brier_negative": bool(br["mean"] < 0)}
    hurts = {"above_floor": bool(ll["mean"] > floor), "beyond_se_multiple": beyond,
             "enough_positive_folds": n_pos >= min_negative_folds, "brier_positive": bool(br["mean"] > 0)}
    met, mirror = all(helps.values()), all(hurts.values())
    return {"pooled": pooled, "fold_means": means, "n_negative_folds": int(n_neg), "n_positive_folds": int(n_pos),
            "checks_helps": helps, "checks_hurts": hurts, "met": met, "mirror_met": mirror,
            "reading": HELPS if met else HURTS if mirror else NEITHER}
