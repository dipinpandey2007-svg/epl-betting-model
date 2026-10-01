"""Experiment 15 (protocol full_coverage_poisson_v1): full-coverage dynamic Poisson with empirical-Bayes priors.

Pre-registered in configs/full_coverage_poisson_v1.toml and docs/preregistration/full_coverage_poisson_v1.md
(commit 5034f56, amendment PA1 in 10e99d1) before any implementation.

    python -m experiments.full_coverage_poisson

- Targets 2017-18 .. 2021-22 only (selection_folds() default, strict guard via build_fold); the match table
  is cut to seasons <= 2021-22 straight after reading.
- Stage 1, before any outcome of a target is joined: frozen-config check, registered group sizes, EB
  team-season counts, the M0 anchor (no prior, unseen excluded) against the recorded Experiment 13 online
  predictions, forecasts of M1, M2 and S1 for all 380 matches per target, and registered refit counts.
- Stage 2: scoring through the common harness and the registered evidence rules.
- The historical results are locked in [historical_locked] by a separate commit. No other season is scored.

Not part of experiments.run_all.
"""

import subprocess
import tomllib

import numpy as np
import pandas as pd

from eplmodel.analysis.promoted_teams import load_team_history
from eplmodel.config import load_config
from eplmodel.data import load_matches
from eplmodel.data.checksums import content_sha256, verify_file
from eplmodel.data.validate import validate_matches, validate_season_dates
from eplmodel.evaluation.alignment import involves_teams
from eplmodel.evaluation.folds import build_fold, restrict_to_max_season
from eplmodel.evaluation.forecasts import check_forecast_frame, prob_columns
from eplmodel.evaluation.reproduction import (
    RECORDED,
    RecordedPredictions,
    compare_predictions,
    load_recorded_predictions,
)
from eplmodel.evaluation.scoring import EvaluationSet, evaluation_set
from eplmodel.evaluation.segments import assign_segments, prior_games_played
from eplmodel.models import full_coverage as fc
from eplmodel.models.full_coverage_spec import (
    FULL_COVERAGE_CONFIG,
    M0_ONLINE,
    M1_PROMOTED,
    M2_HIERARCHICAL,
    S1_IDENTITY_BREAK,
    load_full_coverage_spec,
)
from eplmodel.paths import PROCESSED_DEV_V2, PROJECT_ROOT
from eplmodel.reporting.results import write_predictions, write_results
from eplmodel.splits import SELECTION_TARGET_SEASONS, selection_folds

PREREG_COMMIT = "10e99d1429ae9af4ad26dfdf8cb019fdcd27dfa2"
PREREG_DOC = "docs/preregistration/full_coverage_poisson_v1.md"
CONFIG_FILE = "configs/full_coverage_poisson_v1.toml"
MARKET = RecordedPredictions("14", "market_benchmark", ("market_close_shin",))
GROUP_FLAGS = ("full", "common", "unseen", "promoted", "returning", "continuing_only",
               "recent_yoyo", "long_absence_or_newcomer")


class ProtocolMismatchError(RuntimeError):
    """The code, the data or the config disagree with the frozen pre-registration."""


# --- Frozen state --------------------------------------------------------------------------------------

def _git_show(path: str) -> str:
    out = subprocess.run(["git", "show", f"{PREREG_COMMIT}:{path}"], cwd=PROJECT_ROOT, capture_output=True,
                         text=True, encoding="utf-8")
    if out.returncode != 0:
        raise ProtocolMismatchError(f"cannot read {path} at the pre-registration commit")
    return out.stdout


def check_frozen(cfg: dict) -> dict:
    """The config equals the pre-registered one (except [historical_locked]); the document is unchanged."""
    frozen = tomllib.loads(_git_show(CONFIG_FILE))
    strip = lambda c: {k: v for k, v in c.items() if k != "historical_locked"}
    if strip(frozen) != strip(cfg):
        raise ProtocolMismatchError("configs/full_coverage_poisson_v1.toml differs from the pre-registration")
    doc = (PROJECT_ROOT / PREREG_DOC).read_text(encoding="utf-8").replace("\r\n", "\n")
    if doc != _git_show(PREREG_DOC).replace("\r\n", "\n"):
        raise ProtocolMismatchError(f"{PREREG_DOC} differs from the pre-registration")
    load_full_coverage_spec()   # validated typed view
    return {"prereg_commit": PREREG_COMMIT, "config_sha256": content_sha256(PROJECT_ROOT / CONFIG_FILE),
            "prereg_doc_sha256": content_sha256(PROJECT_ROOT / PREREG_DOC)}


def load_outcome_table(max_season: str) -> pd.DataFrame:
    if not verify_file(PROCESSED_DEV_V2):
        raise ProtocolMismatchError("matches_dev_v2.csv does not match its recorded checksum")
    matches = restrict_to_max_season(load_matches(PROCESSED_DEV_V2, validate=False), max_season)
    validate_matches(matches)
    validate_season_dates(matches)
    return matches


# --- Stage 1: forecasts and pre-scoring checks ----------------------------------------------------------

def fold_groups(fold, structure, yoyo: list[str], cfg: dict) -> pd.DataFrame:
    """Fixture-only flags per target match, plus the registered segment; checked against [groups]."""
    g, target = cfg["groups"], fold.target
    tgt = fold.target_rows.sort_values("Date", kind="stable")
    long_absence = sorted(set(structure.promoted) - set(yoyo))
    flags = pd.DataFrame({
        "full": True,
        "common": ~involves_teams(tgt, structure.unseen).to_numpy(),
        "unseen": involves_teams(tgt, structure.unseen).to_numpy(),
        "promoted": involves_teams(tgt, structure.promoted).to_numpy(),
        "returning": involves_teams(tgt, structure.returning).to_numpy(),
        "continuing_only": ~involves_teams(tgt, structure.promoted).to_numpy(),
        "recent_yoyo": involves_teams(tgt, yoyo).to_numpy(),
        "long_absence_or_newcomer": involves_teams(tgt, long_absence).to_numpy(),
    }, index=pd.Index(tgt["match_id"], name="match_id"))
    seg = cfg["evaluation"]
    flags["prior_games"] = prior_games_played(tgt).reindex(flags.index).to_numpy()
    flags["segment"] = assign_segments(flags["prior_games"], seg["segments_lower_edges"], seg["segments_labels"])
    actual = {k: int(flags[k].sum()) for k in ("full", "common", "unseen", "promoted", "returning",
                                               "continuing_only")}
    registered = {k: g[k][target] for k in actual}
    if (actual != registered or list(structure.promoted) != g["promoted_teams"][target]
            or yoyo != g["recent_yoyo_teams"][target]):
        raise ProtocolMismatchError(f"{target}: groups {actual}, promoted {structure.promoted}, yoyo {yoyo} "
                                    f"differ from the registration")
    return flags


def stage_one(matches: pd.DataFrame, cfg: dict, spec) -> list[dict]:
    rec13 = load_recorded_predictions(RECORDED["13-historical"], spec.targets)
    settings = fc.SolverSettings(spec.solver.gradient_tolerance, spec.solver.max_iterations,
                                 spec.solver.step_halvings)
    team_history = load_team_history()
    staged = []
    for history, target in selection_folds():
        fold = build_fold(matches, history, target)
        structure = fc.target_structure(fold.history_rows, fold.target_rows)
        yoyo = fc.recent_yoyo_teams(fold.history_rows, structure, team_history, cfg["groups"]["yoyo_threshold"])
        flags = fold_groups(fold, structure, yoyo, cfg)

        eb = fc.empirical_bayes(fold.history_rows, target, spec.prior.variance_floor, settings)
        if [eb.n_promoted, eb.n_continuing] != cfg["empirical_bayes"]["expected_team_seasons"][target]:
            raise ProtocolMismatchError(f"{target}: EB team-season counts differ from the registration")

        anchor, anchor_rec = fc.online_fold(fold.history_rows, fold.target_rows, fc.ANCHOR, None,
                                            spec.half_life_days, spec.max_goals, settings)
        rec = rec13[rec13["fold_target"] == target]
        anchor_diff = compare_predictions(anchor, rec, ["anchor"], tolerance=spec.anchor_tolerance,
                                          rename={"anchor": M0_ONLINE})

        preds, records = flags.copy(), {}
        for arm in fc.ARMS:
            arm_preds, arm_rec = fc.online_fold(fold.history_rows, fold.target_rows, arm, eb,
                                                spec.half_life_days, spec.max_goals, settings)
            check_forecast_frame(arm_preds, [arm])
            if list(arm_preds.index) != list(flags.index) or arm_rec["n_fits"] != cfg["groups"]["online_fits"][target]:
                raise ProtocolMismatchError(f"{target} {arm}: coverage or refit count differs from the registration")
            preds = preds.join(arm_preds[prob_columns(arm)])
            records[arm] = arm_rec
        preds = fold.fixtures().join(preds)
        staged.append({"target": target, "history": history, "structure": structure, "yoyo": yoyo, "eb": eb,
                       "anchor_max_abs_diff": anchor_diff, "anchor_fits": anchor_rec["n_fits"],
                       "preds": preds, "records": records})
    return staged


# --- Stage 2: scoring ---------------------------------------------------------------------------------------

def comparators(targets) -> pd.DataFrame:
    """Recorded, checksum-verified comparators: M0 (Exp 13, common), Elo and baseline (Exp 10), market (Exp 14)."""
    rec13 = load_recorded_predictions(RECORDED["13-historical"], targets)
    rec10 = load_recorded_predictions(RECORDED["10"], targets)
    rec14 = load_recorded_predictions(MARKET)                 # Experiment 14 has no folds: filter by season
    rec14 = rec14[rec14["Season"].isin(list(targets))]
    out = rec10[[*prob_columns("elo"), *prob_columns("frequency_baseline")]]
    out = out.join(rec13[prob_columns(M0_ONLINE)], how="left").join(rec14[prob_columns("market_close_shin")],
                                                                     how="left")
    return out


def _sets(frame: pd.DataFrame, matches: pd.DataFrame, arms, mask_col: str, targets) -> list[EvaluationSet]:
    out = []
    for t in targets:
        part = frame[(frame["Season"] == t) & frame[mask_col].to_numpy(dtype=bool)]
        out.append(evaluation_set(part, matches, t, arms))
    return out


def _pool(sets: list[EvaluationSet]) -> EvaluationSet:
    return EvaluationSet(pd.concat([s.preds for s in sets]), pd.concat([s.results for s in sets]),
                         np.concatenate([s.clusters for s in sets]), sets[0].arms)


def paired(frame, matches, left, right, group, targets, segment: str | None = None) -> dict:
    """(left - right) per target and pooled on one group (and optionally one segment)."""
    f = frame if segment is None else frame[frame["segment"] == segment]
    sets = _sets(f, matches, [left, right], group, targets)
    name = f"{left}_minus_{right}"
    per = {t: s.differences([name])[name] for s, t in zip(sets, targets) if len(s.preds)}
    nonempty = [s for s in sets if len(s.preds)]
    out = {"group": group, "n_matches": int(sum(len(s.preds) for s in nonempty)), "per_target": per}
    if nonempty:
        out["pooled"] = _pool(nonempty).differences([name])[name]
    return out


def scores(frame, matches, arm, group, targets) -> dict:
    sets = _sets(frame, matches, [arm], group, targets)
    out = {t: {"n": int(len(s.preds)), **s.scores()[arm]} for t, s in zip(targets, sets) if len(s.preds)}
    pooled = _pool(sets)
    out["pooled"] = {"n": int(len(pooled.preds)), **pooled.scores()[arm]}
    return out


def non_inferiority(pooled_ll: dict, margin: float, k: float) -> dict:
    upper = pooled_ll["mean"] + k * pooled_ll["clustered_se"]
    lower = pooled_ll["mean"] - k * pooled_ll["clustered_se"]
    reading = ("non_inferior_on_common_group" if upper < margin else
               "inferior_on_common_group" if lower > margin else "inconclusive_on_common_group")
    return {"mean": pooled_ll["mean"], "clustered_se": pooled_ll["clustered_se"], "upper_bound": upper,
            "lower_bound": lower, "margin": margin, "se_multiple": k, "reading": reading}


def u_rule(result: dict, floor: float, k: float, min_folds: int) -> dict:
    """Experiment 13's U rule with neutral labels, on a paired (left - right) result."""
    ll, br = result["pooled"]["log_loss"], result["pooled"]["brier"]
    folds = [v["log_loss"]["mean"] for v in result["per_target"].values()]
    beyond = abs(ll["mean"]) > k * ll["clustered_se"]
    helps = ll["mean"] < -floor and beyond and sum(m < 0 for m in folds) >= min_folds and br["mean"] < 0
    hurts = ll["mean"] > floor and beyond and sum(m > 0 for m in folds) >= min_folds and br["mean"] > 0
    return {"n_negative_folds": int(sum(m < 0 for m in folds)), "n_positive_folds": int(sum(m > 0 for m in folds)),
            "reading": "helps" if helps else "hurts" if hurts else "not_distinguishable"}


def evaluate(frame: pd.DataFrame, matches: pd.DataFrame, cfg: dict, targets) -> dict:
    ev = cfg["evidence"]
    floor, k, min_folds = ev["practical_floor_log_loss"], ev["clustered_se_multiple"], ev["min_folds_same_sign"]
    margin = ev["non_inferiority_margin_log_loss"]
    frame = frame.copy()
    frame["m0_available"] = frame[prob_columns(M0_ONLINE)].notna().all(axis=1).to_numpy()
    if not np.array_equal(frame["m0_available"].to_numpy(), frame["common"].to_numpy(dtype=bool)):
        raise ProtocolMismatchError("recorded M0 predictions do not cover exactly the common group")

    arms = [M1_PROMOTED, M2_HIERARCHICAL, S1_IDENTITY_BREAK]
    comparisons = {
        "primary_ni": (M2_HIERARCHICAL, M0_ONLINE, "common"),
        "key_unseen_vs_elo": (M2_HIERARCHICAL, "elo", "unseen"),
        "key_unseen_vs_baseline": (M2_HIERARCHICAL, "frequency_baseline", "unseen"),
        "decomp_m1_vs_m0_common": (M1_PROMOTED, M0_ONLINE, "common"),
        "decomp_m2_vs_m1_full": (M2_HIERARCHICAL, M1_PROMOTED, "full"),
        "decomp_m2_vs_m1_common": (M2_HIERARCHICAL, M1_PROMOTED, "common"),
        "decomp_m2_vs_m1_promoted": (M2_HIERARCHICAL, M1_PROMOTED, "promoted"),
        "decomp_m2_vs_m1_continuing_only": (M2_HIERARCHICAL, M1_PROMOTED, "continuing_only"),
        "decomp_s1_vs_m2_returning": (S1_IDENTITY_BREAK, M2_HIERARCHICAL, "returning"),
        "context_m2_vs_elo_full": (M2_HIERARCHICAL, "elo", "full"),
        "context_m2_vs_market_full": (M2_HIERARCHICAL, "market_close_shin", "full"),
    }
    paired_results = {name: paired(frame, matches, l, r, g, targets) for name, (l, r, g) in comparisons.items()}
    segments = {name: {lab: paired(frame, matches, l, r, g, targets, lab)
                       for lab in cfg["evaluation"]["segments_labels"]}
                for name, (l, r, g) in comparisons.items()}
    group_scores = {g: {a: scores(frame, matches, a, g, targets) for a in [*arms, "elo", "frequency_baseline",
                                                                           "market_close_shin"]}
                    for g in GROUP_FLAGS}
    group_scores["common"][M0_ONLINE] = scores(frame, matches, M0_ONLINE, "common", targets)
    calibration = {a: _pool(_sets(frame, matches, [a], "full", targets)).calibration(a) for a in arms}

    n_valid = int(frame[prob_columns(M2_HIERARCHICAL)].notna().all(axis=1).sum())
    cov_met = n_valid == sum(cfg["groups"]["full"].values())
    ni = non_inferiority(paired_results["primary_ni"]["pooled"]["log_loss"], margin, k)
    reading = ("coverage_failed" if not cov_met else
               {"non_inferior_on_common_group": "full_coverage_non_inferior",
                "inconclusive_on_common_group": "full_coverage_inconclusive",
                "inferior_on_common_group": "full_coverage_inferior"}[ni["reading"]])
    return {
        "coverage": {"n_valid_fc_hier": n_valid, "n_target_matches": int(len(frame)), "met": cov_met,
                     "per_target": {t: int((frame["Season"] == t).sum()) for t in targets}},
        "primary_non_inferiority": ni,
        "reading": reading,
        "key_secondary": {name: u_rule(paired_results[name], floor, k, min_folds)
                          for name in ("key_unseen_vs_elo", "key_unseen_vs_baseline")},
        "paired": paired_results,
        "segments": segments,
        "scores": group_scores,
        "calibration_full_group": calibration,
    }


def run(write: bool = True) -> dict:
    cfg = load_config(FULL_COVERAGE_CONFIG)
    frozen = check_frozen(cfg)
    spec = load_full_coverage_spec()
    if spec.targets != SELECTION_TARGET_SEASONS:
        raise ProtocolMismatchError("targets differ from the selection targets")
    matches = load_outcome_table(spec.max_season)

    staged = stage_one(matches, cfg, spec)                    # every forecast and check before any outcome
    targets = list(spec.targets)
    frame = pd.concat([s["preds"] for s in staged]).join(comparators(targets))
    evaluation = evaluate(frame, matches, cfg, targets)       # outcomes joined here, by match_id

    folds = []
    for s in staged:
        eb = s["eb"]
        folds.append({
            "target": s["target"], "history": f"{s['history'][0]}..{s['history'][-1]}",
            "promoted": list(s["structure"].promoted), "returning": list(s["structure"].returning),
            "unseen": list(s["structure"].unseen), "recent_yoyo": s["yoyo"],
            "empirical_bayes": {"seasons": list(eb.seasons), "a_promoted": eb.a_promoted,
                                "b_promoted": eb.b_promoted, "tau_att": eb.tau_att, "tau_def": eb.tau_def,
                                "n_promoted": eb.n_promoted, "n_continuing": eb.n_continuing,
                                "pooled_var_att": eb.pooled_var_att, "pooled_var_def": eb.pooled_var_def,
                                "mean_se2_att": eb.mean_se2_att, "mean_se2_def": eb.mean_se2_def,
                                "floor_binding_att": eb.floor_binding_att, "floor_binding_def": eb.floor_binding_def,
                                "corr_att_def": eb.corr_att_def},
            "anchor_m0_max_abs_diff": s["anchor_max_abs_diff"], "anchor_fits": s["anchor_fits"],
            "fits": {arm: {k: v for k, v in rec.items() if k != "fits"} |
                     {"home_advantage_first": rec["fits"][0]["home_advantage"],
                      "home_advantage_last": rec["fits"][-1]["home_advantage"],
                      "per_fit": rec["fits"]} for arm, rec in s["records"].items()},
        })
    result = {
        "protocol_id": cfg["protocol"]["protocol_id"],
        "stage": "historical",
        "frozen_state": frozen,
        "seasons_loaded": sorted(matches["Season"].unique()),
        "targets": targets,
        "arms": {k: v for k, v in cfg["arms"].items()},
        "aborted_fits": 0,
        "folds": folds,
        **evaluation,
        "note": "Differences are (left - right) per match; negative = left has the lower loss. Context "
                "comparisons (Elo, market) are descriptive, never selection evidence.",
    }
    if write:
        result["predictions"] = write_predictions(cfg["outputs"]["results_name"], frame)
        write_results(cfg["outputs"]["results_name"], result, data_path=PROCESSED_DEV_V2)
    return result


if __name__ == "__main__":
    out = run()
    print(f"Coverage {out['coverage']['n_valid_fc_hier']}/{out['coverage']['n_target_matches']}; "
          f"primary: {out['primary_non_inferiority']['reading']}; reading: {out['reading']}")
