import ast
import tomllib

import pytest

from eplmodel.paths import CONFIG_DIR, PROJECT_ROOT
from eplmodel.splits import (
    DATASET_V1_SEASONS,
    DEV_TEST_SEASONS,
    DEVELOPMENT_SEASONS,
    EXPOSED_DEV_SEASONS,
    EXPOSED_VALIDATION_SEASONS,
    FINAL_HOLDOUT_SEASONS,
    NEXT_HOLDOUT_SEASONS,
    RECORDED_EXPOSED_VALIDATION_PROTOCOLS,
    REGISTERED_DEV_TEST_SPECS,
    REGISTERED_EXPOSED_VALIDATION_SPECS,
    RESERVED_HOLDOUT_SEASONS,
    SEASON_ORDER,
    SEASON_ROLES,
    SELECTION_TARGET_SEASONS,
    SELECTION_VALIDATION_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    DevTestAccessError,
    ExposedValidationError,
    HoldoutAccessError,
    SeasonRole,
    SplitAccessError,
    assert_history_precedes,
    assert_no_dev_test,
    assert_not_holdout,
    assert_selection_target,
    assert_valid_selection_target,
    expanding_window_folds,
    require_registered_dev_test_spec,
    require_registered_exposed_validation_spec,
    season_role,
    selection_folds,
)


def test_train_and_dev_test_partition_seasons():
    assert set(TRAIN_SEASONS).isdisjoint(DEV_TEST_SEASONS)
    assert TRAIN_SEASONS + DEV_TEST_SEASONS == DATASET_V1_SEASONS
    assert DEV_TEST_SEASONS == EXPOSED_DEV_SEASONS == ("2223", "2324")


def test_season_roles_follow_the_holdout_protocol():
    assert VALIDATION_SEASONS == EXPOSED_VALIDATION_SEASONS == ("2425",)
    assert FINAL_HOLDOUT_SEASONS == ("2526",)
    assert NEXT_HOLDOUT_SEASONS == ("2627",)
    assert list(SEASON_ROLES) == list(SEASON_ORDER)  # every season has exactly one role
    assert DEVELOPMENT_SEASONS + RESERVED_HOLDOUT_SEASONS == SEASON_ORDER
    assert set(DEVELOPMENT_SEASONS).isdisjoint(RESERVED_HOLDOUT_SEASONS)
    assert season_role("2223") is SeasonRole.EXPOSED_DEV
    assert season_role("2425") is SeasonRole.EXPOSED_VALIDATION
    assert season_role("2526") is SeasonRole.HOLDOUT
    with pytest.raises(ValueError):
        season_role("2728")


def test_season_order_is_chronological():
    starts = [int(s[:2]) for s in SEASON_ORDER]
    assert starts == list(range(14, 14 + len(SEASON_ORDER)))
    assert all(int(s[2:]) == int(s[:2]) + 1 for s in SEASON_ORDER)


@pytest.mark.parametrize("min_train,first_val,n_folds", [(3, "1718", 5), (1, "1516", 7)])
def test_folds_used_by_recorded_experiments(min_train, first_val, n_folds):
    folds = expanding_window_folds(TRAIN_SEASONS, min_train)
    assert len(folds) == n_folds
    assert folds[0][1] == first_val
    assert folds[-1] == (TRAIN_SEASONS[:-1], "2122")


def test_folds_are_strictly_chronological():
    for train, val in expanding_window_folds(TRAIN_SEASONS, 1):
        assert val not in train
        assert all(SEASON_ORDER.index(s) < SEASON_ORDER.index(val) for s in train)
        assert set(train).isdisjoint(DEV_TEST_SEASONS) and val not in DEV_TEST_SEASONS


def test_dev_test_seasons_cannot_be_used_for_tuning():
    assert_no_dev_test(TRAIN_SEASONS)
    with pytest.raises(DevTestAccessError):
        assert_no_dev_test(["2122", "2223"])


def test_unregistered_specs_cannot_score_dev_test():
    require_registered_dev_test_spec("elo_k25_logreg_v1")
    with pytest.raises(DevTestAccessError):
        require_registered_dev_test_spec("some_new_model")


# --- Holdout protocol (docs/HOLDOUT_PROTOCOL.md) ---------------------------------

@pytest.mark.parametrize("season", RESERVED_HOLDOUT_SEASONS)
def test_holdout_seasons_are_refused_by_every_guard(season):
    with pytest.raises(HoldoutAccessError):
        assert_not_holdout(["2122", season])
    with pytest.raises(HoldoutAccessError):
        assert_no_dev_test([season])
    with pytest.raises(HoldoutAccessError):
        assert_valid_selection_target(season)
    with pytest.raises(HoldoutAccessError):
        assert_selection_target(season)
    with pytest.raises(HoldoutAccessError):
        selection_folds([season])
    with pytest.raises(HoldoutAccessError):
        assert_history_precedes(["2324", season], "2627")
    with pytest.raises(HoldoutAccessError):
        expanding_window_folds(("2324", "2425", season))


def test_development_seasons_pass_the_holdout_guard():
    assert_not_holdout(DEVELOPMENT_SEASONS)


@pytest.mark.parametrize("season", EXPOSED_DEV_SEASONS)
def test_exposed_seasons_are_never_selection_targets(season):
    with pytest.raises(DevTestAccessError):
        assert_valid_selection_target(season)


def test_exposed_seasons_may_be_history_for_later_seasons():
    assert_history_precedes(TRAIN_SEASONS + EXPOSED_DEV_SEASONS, "2425")
    with pytest.raises(SplitAccessError):
        assert_history_precedes(["2122", "2425"], "2425")


def test_selection_folds_default_excludes_2425():
    folds = selection_folds()
    assert [val for _, val in folds] == list(SELECTION_TARGET_SEASONS)
    assert SELECTION_TARGET_SEASONS == ("1718", "1819", "1920", "2021", "2122")
    assert "2425" not in SELECTION_TARGET_SEASONS
    # The training-season folds are exactly those of the recorded Elo K selection.
    assert folds == expanding_window_folds(TRAIN_SEASONS, 3)


def test_legacy_selection_folds_still_build_the_recorded_2425_fold():
    # Frozen for the recorded protocols of Experiments 10-13 (amendment A1 keeps it unchanged).
    assert SELECTION_VALIDATION_SEASONS == ("1718", "1819", "1920", "2021", "2122", "2425")
    folds = selection_folds(SELECTION_VALIDATION_SEASONS)
    assert folds[:5] == selection_folds()
    # 2024-25 is predicted from everything before it, including the exposed seasons as history.
    assert folds[5] == (TRAIN_SEASONS + EXPOSED_DEV_SEASONS, "2425")


def test_selection_folds_never_target_exposed_or_holdout_seasons():
    for history, target in selection_folds():
        assert target not in EXPOSED_DEV_SEASONS + RESERVED_HOLDOUT_SEASONS
        assert set(history).isdisjoint(RESERVED_HOLDOUT_SEASONS)
        assert all(SEASON_ORDER.index(s) < SEASON_ORDER.index(target) for s in history)
    with pytest.raises(DevTestAccessError):
        selection_folds(["2324"])
    with pytest.raises(HoldoutAccessError):
        selection_folds(["2526"])


# --- Amendment A1 / access-log entry P1: 2024-25 retired from selection ----------

@pytest.mark.parametrize("season", SELECTION_TARGET_SEASONS)
def test_new_selection_guard_accepts_the_historical_targets(season):
    assert_selection_target(season)


def test_new_selection_guard_refuses_2425():
    with pytest.raises(ExposedValidationError):
        assert_selection_target("2425")
    assert issubclass(ExposedValidationError, SplitAccessError)


@pytest.mark.parametrize("season,error", [("2223", DevTestAccessError), ("2324", DevTestAccessError),
                                          ("1415", SplitAccessError), ("1516", SplitAccessError),
                                          ("1617", SplitAccessError), ("2728", ValueError)])
def test_new_selection_guard_refuses_every_other_season(season, error):
    with pytest.raises(error):
        assert_selection_target(season)


# Recorded protocols that scored 2024-25 (Experiments 10-13) and their configs.
RECORDED_PROTOCOL_CONFIGS = {
    "validation_2425_v1": "validation_2425_v1.toml",
    "update_policy_diagnostic_v1": "update_policy_diagnostic_v1.toml",
    "time_weighted_poisson_v1": "time_weighted_poisson_v1.toml",
    "online_tw_poisson_diagnostic_v1": "online_tw_poisson_diagnostic_v1.toml",
}


def test_exposed_2425_specs_are_exactly_the_registered_set():
    assert REGISTERED_EXPOSED_VALIDATION_SPECS == frozenset({
        "elo_k25_logreg_v1", "frequency_baseline_v1", "poisson_static_v1", "dixon_coles_staged_v1",
        "elo_k25_frozen_ratings_online_layer_v1_diag", "elo_k25_season_start_v1_diag",
        "poisson_time_weighted_v1", "poisson_tw_online_h730_v1_diag",
    })
    assert set(RECORDED_EXPOSED_VALIDATION_PROTOCOLS) == set(RECORDED_PROTOCOL_CONFIGS)
    for spec in REGISTERED_EXPOSED_VALIDATION_SPECS:
        require_registered_exposed_validation_spec(spec)
    for spec in ("some_new_model", "elo_k20_exploratory", "poisson_tw_h365_dev"):
        with pytest.raises(ExposedValidationError):
            require_registered_exposed_validation_spec(spec)


def _recorded_config(protocol_id: str) -> dict:
    with open(CONFIG_DIR / RECORDED_PROTOCOL_CONFIGS[protocol_id], "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["protocol"]["protocol_id"] == protocol_id
    return cfg


def _recorded_targets(cfg: dict) -> tuple[str, ...]:
    p = cfg["protocol"]
    if "targets" in p:
        return tuple(p["targets"])
    if "development_targets" in p:
        return (*p["development_targets"], p["validation_target"])
    return (*p["historical_targets"], p["validation_target"])


def _new_2425_specs(protocol_id: str, cfg: dict) -> set[str]:
    """The spec ids each recorded config names as its newly exposed arms on 2024-25."""
    if protocol_id == "validation_2425_v1":
        return {cfg[m]["spec_id"] for m in ("elo", "baseline", "poisson", "dixon_coles")}
    if protocol_id == "update_policy_diagnostic_v1":
        return {cfg["arms"]["elo_f1"], cfg["arms"]["elo_f2"]}
    if protocol_id == "time_weighted_poisson_v1":
        return {cfg["candidate"]["spec_id"]}
    return {cfg["arms"]["poisson_tw_online"]}


@pytest.mark.parametrize("protocol_id", list(RECORDED_PROTOCOL_CONFIGS))
def test_recorded_protocols_keep_their_registered_targets_and_specs(protocol_id):
    cfg = _recorded_config(protocol_id)
    targets = _recorded_targets(cfg)
    assert targets == SELECTION_VALIDATION_SEASONS
    # The legacy path still builds every recorded fold, including 2024-25 as the last target.
    assert [t for _, t in selection_folds(targets)] == list(targets)
    assert _new_2425_specs(protocol_id, cfg) == set(RECORDED_EXPOSED_VALIDATION_PROTOCOLS[protocol_id])


def test_only_the_established_specs_are_registered_for_both_exposed_benchmarks():
    assert (REGISTERED_EXPOSED_VALIDATION_SPECS & REGISTERED_DEV_TEST_SPECS
            == RECORDED_EXPOSED_VALIDATION_PROTOCOLS["validation_2425_v1"])


# --- Static check: new experiments must not reach 2024-25 through the legacy paths --

# Modules of the recorded protocols (Experiments 10-13), frozen before amendment A1.
RECORDED_2425_MODULES = frozenset({
    "validation_2425", "update_policy_diagnostic", "time_weighted_poisson", "online_tw_poisson_diagnostic"})
LEGACY_NAMES = frozenset({"SELECTION_VALIDATION_SEASONS", "VALIDATION_SEASONS", "EXPOSED_VALIDATION_SEASONS",
                          "assert_valid_selection_target", "fold_data", *RECORDED_2425_MODULES})
FORBIDDEN_LITERALS = frozenset({"2425", "2024-25"})


def legacy_2425_violations(source: str) -> list[str]:
    """Names, imports and string literals through which code could target 2024-25 by the legacy paths.

    Docstrings are skipped, so prose may mention 2024-25; string values used in code may not.
    """
    tree = ast.parse(source)
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in LEGACY_NAMES:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in LEGACY_NAMES:
            found.append(node.attr)
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in LEGACY_NAMES:
            found.append(node.name)
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
              and any(lit in node.value for lit in FORBIDDEN_LITERALS)):
            found.append(repr(node.value))
    return found


def _toml_strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _toml_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _toml_strings(v)
    elif isinstance(value, str):
        yield value


def test_new_experiment_modules_do_not_use_the_legacy_2425_paths():
    modules = sorted((PROJECT_ROOT / "experiments").glob("*.py"))
    assert {m.stem for m in modules} >= RECORDED_2425_MODULES
    offenders = {m.name: v for m in modules if m.stem not in RECORDED_2425_MODULES
                 and (v := legacy_2425_violations(m.read_text(encoding="utf-8")))}
    assert offenders == {}, (
        "New experiments must not target 2024-25 (retired by amendment A1). Use SELECTION_TARGET_SEASONS, "
        f"selection_folds() and assert_selection_target instead: {offenders}")


def test_new_configs_do_not_name_2425():
    recorded = set(RECORDED_PROTOCOL_CONFIGS.values())
    offenders = []
    for path in sorted(CONFIG_DIR.glob("*.toml")):
        if path.name in recorded:
            continue
        with open(path, "rb") as f:
            if any(lit in s for s in _toml_strings(tomllib.load(f)) for lit in FORBIDDEN_LITERALS):
                offenders.append(path.name)
    assert offenders == []


@pytest.mark.parametrize("source", [
    "from eplmodel.splits import SELECTION_VALIDATION_SEASONS",
    "import eplmodel.splits as s\nfolds = s.selection_folds(s.SELECTION_VALIDATION_SEASONS)",
    "from eplmodel.splits import selection_folds\nfolds = selection_folds(['1718', '2425'])",
    "from eplmodel.evaluation.validation import fold_data",
    "from eplmodel.evaluation import validation\nvalidation.fold_data(m, h, t)",
    "from experiments import time_weighted_poisson",
    "TARGET = '2024-25'",
    "PATH = 'results/validation_2425/predictions.csv'",
])
def test_static_check_catches_legacy_2425_use(source):
    assert legacy_2425_violations(source)


def test_static_check_accepts_the_new_selection_path():
    source = ('"""Selection on the historical folds; does not use 2024-25 or the 2025-26 holdout."""\n'
              "from eplmodel.splits import SELECTION_TARGET_SEASONS, assert_selection_target, selection_folds\n"
              "folds = selection_folds()\n")
    assert legacy_2425_violations(source) == []


def test_amendment_a1_is_recorded_before_use():
    protocol = (PROJECT_ROOT / "docs" / "HOLDOUT_PROTOCOL.md").read_text(encoding="utf-8")
    log = (PROJECT_ROOT / "docs" / "TEST_SET_ACCESS_LOG.md").read_text(encoding="utf-8")
    assert "Amendment A1" in protocol
    assert "| P1 |" in log
