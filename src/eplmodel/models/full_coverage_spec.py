"""Typed, validated view of the pre-registered protocol full_coverage_poisson_v1 (planned Experiment 15).

This module holds the constants and the specification object that the future implementation will consume.
It contains no model: the penalised fit, the empirical-Bayes estimation and the online loop are not written
yet (docs/preregistration/full_coverage_poisson_v1.md). load_full_coverage_spec() refuses a config that
disagrees with the locked choices of Experiments 12-13, the selection targets of amendment A1, or the
recorded group sizes.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from eplmodel.analysis.promoted_teams import LONG_ABSENCE, RECENT_YOYO
from eplmodel.config import load_config
from eplmodel.paths import CONFIG_DIR, HOLDOUT_CONFIG
from eplmodel.splits import (
    REGISTERED_DEV_TEST_SPECS,
    REGISTERED_EXPOSED_VALIDATION_SPECS,
    SELECTION_TARGET_SEASONS,
    assert_selection_target,
)

FULL_COVERAGE_CONFIG = CONFIG_DIR / "full_coverage_poisson_v1.toml"
PROTOCOL_ID = "full_coverage_poisson_v1"

# Arm prefixes (config [arms]).
M0_ONLINE = "poisson_tw_online"
M1_PROMOTED = "fc_promoted"
M2_HIERARCHICAL = "fc_hier"
S1_IDENTITY_BREAK = "fc_hier_break"
ARMS = (M0_ONLINE, M1_PROMOTED, M2_HIERARCHICAL, S1_IDENTITY_BREAK)
PRIMARY_ARM = M2_HIERARCHICAL

# Inputs that no arm of this protocol may use.
FORBIDDEN_INPUTS = frozenset({"market_odds", "shots", "xg", "lineups", "injuries", "squad", "tactics",
                              "championship_data"})


class PriorGroup(StrEnum):
    """Prior group of a target team, from fixtures only: did it play the PL in the season before the target?"""

    CONTINUING = "continuing"
    PROMOTED = "promoted"


class DiagnosticCategory(StrEnum):
    """Reporting taxonomy (Experiment 7 thresholds); never used to fit anything."""

    CONTINUING = "continuing"
    RECENT_YOYO = RECENT_YOYO
    LONG_ABSENCE_OR_NEWCOMER = LONG_ABSENCE


class FullCoverageSpecError(ValueError):
    """The full_coverage_poisson_v1 config disagrees with the locked choices, the splits or itself."""


@dataclass(frozen=True)
class PriorSpec:
    group_rule: str
    promoted_mean: str
    continuing_mean: str
    variance: str
    variance_floor: float
    returning_rule: str


@dataclass(frozen=True)
class SolverSpec:
    method: str
    start: str
    gradient_tolerance: float
    max_iterations: int
    step_halvings: int


@dataclass(frozen=True)
class FullCoverageSpec:
    protocol_id: str
    status: str
    targets: tuple[str, ...]
    max_season: str
    half_life_days: float
    max_goals: int
    weight_reference: str
    identification: str
    prior: PriorSpec
    solver: SolverSpec
    arms: dict[str, str]
    primary_arm: str
    practical_floor_log_loss: float
    clustered_se_multiple: float
    anchor_tolerance: float


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise FullCoverageSpecError(message)


def validate(cfg: dict) -> None:
    """Raise FullCoverageSpecError unless the config agrees with the locked choices and is internally consistent."""
    p, inh, m, pr, eb, g = (cfg["protocol"], cfg["inherited"], cfg["model"], cfg["prior"], cfg["empirical_bayes"],
                            cfg["groups"])
    tw_lock = load_config(CONFIG_DIR / "time_weighted_poisson_v1.toml")["locked"]
    online = load_config(CONFIG_DIR / "online_tw_poisson_diagnostic_v1.toml")
    base = load_config()

    _check(p["protocol_id"] == PROTOCOL_ID, "protocol id")
    _check(tuple(p["targets"]) == SELECTION_TARGET_SEASONS, "targets must be the amendment-A1 selection targets")
    for season in p["targets"]:
        assert_selection_target(season)
    _check(p["max_season"] == SELECTION_TARGET_SEASONS[-1], "data must be cut at the last selection target")
    _check(p["hyperparameter_selection"] == "none" and cfg["selection"]["tuned_values"] == [],
           "no hyperparameter may be tuned under this protocol")
    _check(set(p["forbidden_inputs"]) == FORBIDDEN_INPUTS and not FORBIDDEN_INPUTS & set(p["inputs"]),
           "forbidden inputs (market, xG, squad, ...) must be excluded")

    _check(tw_lock["status"] == "locked" and float(inh["half_life_days"]) == float(tw_lock["half_life_days"]) == 730.0,
           "H must be the Experiment 12 lock (730 days)")
    _check(float(inh["half_life_days"]) == float(online["online"]["half_life_days"]), "H must equal Experiment 13's")
    _check(inh["refit_cadence"] == online["online"]["refit_cadence"]
           and inh["information_policy"] == online["online"]["information_policy"]
           and inh["warm_start"] is False, "online cadence, information policy and cold starts are inherited")
    _check(inh["max_goals"] == base["poisson"]["max_goals"], "scoreline grid is inherited")
    _check(m["form"] == "independent_poisson_map" and m["home_advantage"] == "global", "model form")
    _check(m["priors_on_every_refit"] is True and m["hyperparameters_frozen_within_season"] is True, "prior use")

    _check(pr["group_rule"] == "membership_of_previous_season", "prior groups come from fixtures")
    _check(math.isfinite(pr["variance_floor"]) and pr["variance_floor"] > 0, "variance floor must be positive")
    for target in p["targets"]:
        k = SELECTION_TARGET_SEASONS.index(target) + 2            # EB seasons: 2015-16 .. season before target
        _check(eb["expected_team_seasons"][target] == [3 * k, 17 * k], f"{target}: EB team-season counts")

    for target in p["targets"]:
        _check(g["full"][target] == 380, f"{target}: full coverage is 380 matches")
        _check(g["common"][target] + g["unseen"][target] == g["full"][target], f"{target}: common + unseen")
        _check(g["promoted"][target] + g["continuing_only"][target] == g["full"][target],
               f"{target}: promoted + continuing-only")
        # A returning team's matches are promoted-team matches (some also involve an unseen team).
        _check(g["returning"][target] <= g["promoted"][target] and g["unseen"][target] <= g["promoted"][target],
               f"{target}: returning and unseen matches are subsets of the promoted-team matches")
        _check(len(g["promoted_teams"][target]) == 3, f"{target}: three promoted teams")
        _check(set(g["recent_yoyo_teams"][target]) <= set(g["promoted_teams"][target]), f"{target}: yoyo teams")
    _check(g["common"] == {t: online["groups"]["expected_common"][t] for t in p["targets"]},
           "common groups must equal Experiments 11-13")
    _check(g["yoyo_threshold"] == base["promoted_teams"]["yoyo_threshold"], "taxonomy threshold of Experiment 7")

    arms = cfg["arms"]
    _check(set(a for a in arms if a != "primary") == set(ARMS) and arms["primary"] == PRIMARY_ARM, "arms")
    _check(arms[M0_ONLINE] == online["arms"]["poisson_tw_online"], "M0 is the Experiment 13 online arm")
    new_specs = {arms[a] for a in (M1_PROMOTED, M2_HIERARCHICAL, S1_IDENTITY_BREAK)}
    holdout_specs = set(load_config(HOLDOUT_CONFIG)["registration"]["spec_ids"])
    _check(not new_specs & (REGISTERED_DEV_TEST_SPECS | REGISTERED_EXPOSED_VALIDATION_SPECS | holdout_specs),
           "new arms must not be registered for dev-test, exposed-validation or holdout scoring")
    _check(cfg["outputs"]["in_run_all"] is False, "not part of run_all")


def load_full_coverage_spec(path: Path = FULL_COVERAGE_CONFIG) -> FullCoverageSpec:
    cfg = load_config(path)
    validate(cfg)
    p, inh, m, pr, s, e = (cfg["protocol"], cfg["inherited"], cfg["model"], cfg["prior"], cfg["solver"],
                           cfg["evidence"])
    return FullCoverageSpec(
        protocol_id=p["protocol_id"], status=p["status"], targets=tuple(p["targets"]), max_season=p["max_season"],
        half_life_days=float(inh["half_life_days"]), max_goals=int(inh["max_goals"]),
        weight_reference=m["weight_reference"], identification=m["identification"],
        prior=PriorSpec(pr["group_rule"], pr["promoted_mean"], pr["continuing_mean"], pr["variance"],
                        float(pr["variance_floor"]), pr["returning_rule"]),
        solver=SolverSpec(s["method"], s["start"], float(s["gradient_tolerance"]), int(s["max_iterations"]),
                          int(s["step_halvings"])),
        arms={a: cfg["arms"][a] for a in ARMS}, primary_arm=cfg["arms"]["primary"],
        practical_floor_log_loss=float(e["practical_floor_log_loss"]),
        clustered_se_multiple=float(e["clustered_se_multiple"]),
        anchor_tolerance=float(cfg["implementation_checks"]["anchor_tolerance"]),
    )
