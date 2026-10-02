"""Typed, validated view of the pre-registered protocol shots_information_v1 (planned Experiment 16).

Constants and the specification object that the future implementation will consume. No model is implemented
here (docs/preregistration/shots_information_v1.md). load_shots_spec() refuses a config that disagrees with the
locked Experiment 12-13 choices, the amendment-A1 selection targets or itself.
"""

import math
from dataclasses import dataclass

from eplmodel.config import load_config
from eplmodel.paths import CONFIG_DIR, HOLDOUT_CONFIG
from eplmodel.splits import (
    LOGGED_EXPOSED_VALIDATION_ACCESSES,
    REGISTERED_DEV_TEST_SPECS,
    REGISTERED_EXPOSED_VALIDATION_SPECS,
    SELECTION_TARGET_SEASONS,
    assert_selection_target,
)

SHOTS_CONFIG = CONFIG_DIR / "shots_information_v1.toml"
PROTOCOL_ID = "shots_information_v1"
PRIMARY_COLUMNS = ("HST", "AST")      # shots on target (home, away)
SENSITIVITY_COLUMNS = ("HS", "AS")    # total shots (home, away)
RESULT_COLUMNS = frozenset({"FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"})
OMEGA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = ("B0", "B1", "S1")
FORBIDDEN_INPUTS = frozenset({"market_odds", "xg", "lineups", "injuries", "squad", "tactics", "ml_features"})


class ShotsSpecError(ValueError):
    """The shots_information_v1 config disagrees with the locked choices, the splits or itself."""


@dataclass(frozen=True)
class ShotsSpec:
    protocol_id: str
    status: str
    targets: tuple[str, ...]
    max_season: str
    half_life_days: float
    max_goals: int
    primary_columns: tuple[str, str]
    sensitivity_columns: tuple[str, str]
    omega_grid: tuple[float, ...]
    arms: dict[str, str]
    se_multiple: float
    practical_floor_log_loss: float
    clustered_se_multiple: float
    b0_reproduction_tolerance: float


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ShotsSpecError(message)


def validate(cfg: dict) -> None:
    p, src, b, sm, bl, sel, ev, g = (cfg["protocol"], cfg["source"], cfg["baseline"], cfg["shot_model"],
                                    cfg["blend"], cfg["selection"], cfg["evidence"], cfg["groups"])
    online = load_config(CONFIG_DIR / "online_tw_poisson_diagnostic_v1.toml")
    tw_lock = load_config(CONFIG_DIR / "time_weighted_poisson_v1.toml")["locked"]
    tw_ev = load_config(CONFIG_DIR / "time_weighted_poisson_v1.toml")["evidence"]

    _check(p["protocol_id"] == PROTOCOL_ID, "protocol id")
    _check(tuple(p["targets"]) == SELECTION_TARGET_SEASONS, "targets must be the amendment-A1 selection targets")
    for season in p["targets"]:
        assert_selection_target(season)
    _check(p["max_season"] == SELECTION_TARGET_SEASONS[-1], "data must be cut at the last selection target")
    _check(set(p["forbidden_inputs"]) == FORBIDDEN_INPUTS and not FORBIDDEN_INPUTS & set(p["inputs"]),
           "forbidden inputs (market, xG, ML, ...) must be excluded")

    _check(tuple(src["primary_columns"]) == PRIMARY_COLUMNS, "primary shot columns are HST/AST")
    _check(tuple(src["sensitivity_columns"]) == SENSITIVITY_COLUMNS, "sensitivity shot columns are HS/AS")
    _check(not RESULT_COLUMNS & set(src["read_columns"]), "the shots table must not read result columns")

    _check(b["spec_id"] == online["arms"]["poisson_tw_online"], "B0 is the Experiment 13 online arm")
    _check(float(b["half_life_days"]) == float(tw_lock["half_life_days"]) == float(online["online"]["half_life_days"])
           == 730.0, "H is the locked 730 days")
    _check((b["refit_cadence"], b["information_policy"], b["fit_set"], b["unseen_team_rule"]) ==
           (online["online"]["refit_cadence"], online["online"]["information_policy"], online["online"]["fit_set"],
            online["online"]["unseen_team_rule"]), "B0's online policy is inherited unchanged")
    _check(b["max_goals"] == online["online"]["max_goals"], "scoreline grid inherited")
    _check(sm["home"] == "global_home_term" and sm["weights"] == "inherited_h730", "shot model venue and weights")

    grid = tuple(float(x) for x in bl["omega_grid"])
    _check(grid == OMEGA_GRID and grid[0] == 0.0, "omega grid is registered and contains B0 (omega = 0)")
    _check(bl["level_and_home_from"] == "goal_model", "the goal level and home advantage come from the goal model")

    _check(sel["rule"] == "one_se_smallest_omega" and sel["tie_break"] == "smaller_omega"
           and float(sel["se_multiple"]) == 1.0 and sel["se_type"] == "clustered", "selection rule")
    _check(tuple(sel["nested_outer_targets"]) == SELECTION_TARGET_SEASONS[1:], "nested outer targets")
    _check(float(ev["practical_floor_log_loss"]) == float(tw_ev["practical_floor_log_loss"]) == 0.002,
           "the practical floor is the inherited 0.002")
    _check(g["expected_common"] == {t: online["groups"]["expected_common"][t] for t in p["targets"]}
           and g["expected_online_fits"] == {t: online["groups"]["expected_online_fits"][t] for t in p["targets"]},
           "groups and refit counts are inherited from Experiment 13")
    for t in p["targets"]:
        _check(g["expected_common"][t] + g["expected_unseen_excluded"][t] == 380, f"{t}: common + excluded = 380")

    arms = cfg["arms"]
    _check(set(a for a in arms if a != "primary") == set(ARMS) and arms["primary"] == "B1", "arms")
    _check(arms["B0"] == b["spec_id"], "B0 spec")
    new = {arms["B1"], arms["S1"]}
    logged = frozenset().union(*LOGGED_EXPOSED_VALIDATION_ACCESSES.values())
    holdout = set(load_config(HOLDOUT_CONFIG)["registration"]["spec_ids"])
    _check(not new & (REGISTERED_DEV_TEST_SPECS | REGISTERED_EXPOSED_VALIDATION_SPECS | logged | holdout),
           "new arms must not be registered for dev-test, exposed-validation or holdout scoring")
    _check(cfg["outputs"]["in_run_all"] is False, "not part of run_all")


def load_shots_spec(path=SHOTS_CONFIG) -> ShotsSpec:
    cfg = load_config(path)
    validate(cfg)
    p, src, b, bl, sel, ev = (cfg["protocol"], cfg["source"], cfg["baseline"], cfg["blend"], cfg["selection"],
                              cfg["evidence"])
    tol = float(cfg["implementation_checks"]["b0_reproduction_tolerance"])
    if not (math.isfinite(tol) and tol > 0):
        raise ShotsSpecError("reproduction tolerance")
    return ShotsSpec(
        protocol_id=p["protocol_id"], status=p["status"], targets=tuple(p["targets"]), max_season=p["max_season"],
        half_life_days=float(b["half_life_days"]), max_goals=int(b["max_goals"]),
        primary_columns=tuple(src["primary_columns"]), sensitivity_columns=tuple(src["sensitivity_columns"]),
        omega_grid=tuple(float(x) for x in bl["omega_grid"]),
        arms={a: cfg["arms"][a] for a in ARMS}, se_multiple=float(sel["se_multiple"]),
        practical_floor_log_loss=float(ev["practical_floor_log_loss"]),
        clustered_se_multiple=float(ev["clustered_se_multiple"]), b0_reproduction_tolerance=tol)
