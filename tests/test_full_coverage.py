"""Full-coverage dynamic Poisson (protocol full_coverage_poisson_v1): the pre-registered test plan.

Every fit uses a synthetic league built to contain each team type:

    1415: A B C D E F      1516: A B C D E G (G promoted)      1617: A B C D G F (F returns after one season)
    1718 (target): A B C D E N   -> continuing A-D; promoted E (returning, absent 1617) and N (never seen);
                                    F and G are history-only teams
    1819: a later season that must never enter a 1718 fit

No test fits a model on real data.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from eplmodel.config import load_config
from eplmodel.evaluation import online_poisson as op
from eplmodel.evaluation.folds import build_fold
from eplmodel.evaluation.forecasts import check_forecast_frame, prob_columns
from eplmodel.models import full_coverage as fc
from eplmodel.models.full_coverage_spec import M1_PROMOTED, M2_HIERARCHICAL, S1_IDENTITY_BREAK
from eplmodel.models.time_weights import exponential_decay_weights
from eplmodel.splits import DevTestAccessError, ExposedValidationError, HoldoutAccessError, SplitAccessError
from experiments import full_coverage_poisson as fcp
from test_validation import _reverse_scores, _round_robin, _with_results

H = 730.0
FLOOR = 0.05
SEASON_TEAMS = {"1415": "ABCDEF", "1516": "ABCDEG", "1617": "ABCDGF", "1718": "ABCDEN", "1819": "ABCDEN"}
STRENGTH = {"A": 0.4, "B": 0.2, "C": 0.0, "D": 0.0, "E": -0.2, "F": -0.3, "G": -0.25, "N": -0.3}
HISTORY, TARGET = ("1415", "1516", "1617"), "1718"


def league(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for season, teams in SEASON_TEAMS.items():
        start = pd.Timestamp(2000 + int(season[:2]), 8, 15)
        for r, fixtures in enumerate(_round_robin(list(teams))):
            for home, away in fixtures:
                rows.append({"Date": start + pd.Timedelta(weeks=r), "HomeTeam": home, "AwayTeam": away,
                             "FTHG": int(rng.poisson(np.exp(0.35 + STRENGTH[home] - STRENGTH[away]))),
                             "FTAG": int(rng.poisson(np.exp(0.10 + STRENGTH[away] - STRENGTH[home]))),
                             "Season": season})
    return _with_results(pd.DataFrame(rows).sort_values("Date", kind="stable").reset_index(drop=True))


@pytest.fixture(scope="module")
def lg():
    return league()


@pytest.fixture(scope="module")
def fold(lg):
    return build_fold(lg, HISTORY, TARGET)


@pytest.fixture(scope="module")
def eb(fold):
    return fc.empirical_bayes(fold.history_rows, TARGET, FLOOR)


@pytest.fixture(scope="module")
def runs(fold, eb):
    return {arm: fc.online_fold(fold.history_rows, fold.target_rows, arm, eb, H, 10) for arm in fc.ARMS}


def _first_date_fit(fold, eb, arm):
    """The fit used for the first target date, with its parameterisation."""
    s = fc.target_structure(fold.history_rows, fold.target_rows)
    hist = fc.break_identities(fold.history_rows, s.returning) if arm == S1_IDENTITY_BREAK else fold.history_rows
    par = fc.Parameterisation(s.teams, tuple(sorted((set(hist["HomeTeam"]) | set(hist["AwayTeam"])) - set(s.teams))))
    prior = fc.arm_prior(arm, s, eb)
    d = fold.target_rows["Date"].min()
    theta, _ = fc.fit(hist, exponential_decay_weights(hist["Date"], d, H), par, prior)
    return theta, par, prior, s


# --- Structure and taxonomy --------------------------------------------------------------------------

def test_target_structure_from_fixtures(fold):
    s = fc.target_structure(fold.history_rows, fold.target_rows)
    assert s.teams == tuple("ABCDEN") and s.promoted == ("E", "N") and s.continuing == tuple("ABCD")
    assert s.returning == ("E",) and s.unseen == ("N",)


def test_recent_yoyo_uses_the_experiment_7_threshold(fold):
    s = fc.target_structure(fold.history_rows, fold.target_rows)
    # E is one season out (in the fold history); N is unseen and must be looked up in the team history table.
    assert fc.recent_yoyo_teams(fold.history_rows, s, {"N": None}, 2) == ["E"]
    assert fc.recent_yoyo_teams(fold.history_rows, s, {"N": 1}, 2) == ["E", "N"]
    assert fc.recent_yoyo_teams(fold.history_rows, s, {"N": None}, 0) == []


# --- Parameterisation and identification ------------------------------------------------------------

def test_parameterisation_eliminates_the_alphabetically_last_target_team():
    par = fc.Parameterisation(tuple("ABCDEN"), ("F", "G"))
    assert par.eliminated == "N" and par.free == ("A", "B", "C", "D", "E", "F", "G")
    theta = np.arange(par.n_params, dtype=float)
    eff = par.effects(theta)
    assert eff.loc[list("ABCDEN"), "attack"].sum() == pytest.approx(0, abs=1e-12)
    assert eff.loc[list("ABCDEN"), "defence"].sum() == pytest.approx(0, abs=1e-12)
    assert eff.loc["F", "attack"] == theta[2 + par.free.index("F")]          # history-only: free, unconstrained
    with pytest.raises(ValueError):
        fc.Parameterisation(("B", "A"), ())


@pytest.mark.parametrize("arm", fc.ARMS)
def test_constraints_hold_and_hessian_is_negative_definite(fold, eb, arm):
    theta, par, prior, _ = _first_date_fit(fold, eb, arm)
    eff = par.effects(theta)
    assert abs(eff.loc[list(par.constrained), "attack"].sum()) < 1e-12
    assert abs(eff.loc[list(par.constrained), "defence"].sum()) < 1e-12
    hist = fold.history_rows if arm != S1_IDENTITY_BREAK else fc.break_identities(fold.history_rows, ("E",))
    w2 = np.repeat(exponential_decay_weights(hist["Date"], fold.target_rows["Date"].min(), H), 2)
    _, g, Hm = fc.objective(theta, fc.design(hist, par), fc.goals(hist), w2, *prior.matrices(par))
    assert np.max(np.abs(g)) <= 1e-9
    assert np.max(np.linalg.eigvalsh(Hm)) < 0


def test_without_priors_predictions_do_not_depend_on_the_eliminated_team(fold):
    s = fc.target_structure(fold.history_rows, fold.target_rows)
    hist = fold.history_rows
    w = exponential_decay_weights(hist["Date"], fold.target_rows["Date"].min(), H)
    common = tuple(sorted(set(s.teams) - set(s.unseen)))       # A B C D E
    tgt = fold.target_rows[~fold.target_rows["HomeTeam"].isin(s.unseen) & ~fold.target_rows["AwayTeam"].isin(s.unseen)]
    out = []
    for constrained, extra in ((common, ("F", "G")), (("A", "B", "C", "D", "E", "F"), ("G",))):
        par = fc.Parameterisation(constrained, extra)
        theta, _ = fc.fit(hist, w, par, fc.NO_PRIOR)
        out.append(np.column_stack(fc.rates(theta, tgt, par)))
    assert np.max(np.abs(out[0] - out[1])) < 1e-9


# --- Empirical-Bayes prior: pre-target information only --------------------------------------------

def test_single_season_fit_matches_statsmodels_sum_coding(lg):
    season = lg[lg["Season"] == "1516"]
    eff = fc.single_season_effects(season).set_index("team")
    long = pd.concat([pd.DataFrame({"Team": season["HomeTeam"], "Opp": season["AwayTeam"], "Goals": season["FTHG"],
                                    "IsHome": 1}),
                      pd.DataFrame({"Team": season["AwayTeam"], "Opp": season["HomeTeam"], "Goals": season["FTAG"],
                                    "IsHome": 0})], ignore_index=True)
    res = smf.glm("Goals ~ C(Team, Sum) + C(Opp, Sum) + IsHome", long, family=sm.families.Poisson()).fit(tol=1e-12)
    teams = sorted(set(season["HomeTeam"]))
    sm_att = [res.params[f"C(Team, Sum)[S.{t}]"] for t in teams[:-1]]
    assert eff.loc[teams[:-1], "attack"].to_numpy() == pytest.approx(sm_att, abs=1e-7)
    assert eff.loc[teams[:-1], "se_attack"].to_numpy() == pytest.approx(
        [res.bse[f"C(Team, Sum)[S.{t}]"] for t in teams[:-1]], rel=1e-5)
    assert eff["attack"].sum() == pytest.approx(0, abs=1e-12) and eff["defence"].sum() == pytest.approx(0, abs=1e-12)


def test_empirical_bayes_formula(fold, eb):
    ts = eb.team_seasons
    assert eb.seasons == ("1516", "1617")                       # 2014-15 excluded
    assert set(ts.loc[ts["promoted"], "team"] + ts.loc[ts["promoted"], "season"]) == {"G1516", "F1617"}
    assert (eb.n_promoted, eb.n_continuing) == (2, 10)
    assert eb.a_promoted == pytest.approx(ts.loc[ts["promoted"], "attack"].mean())
    resid = ts["attack"] - ts.groupby("promoted")["attack"].transform("mean")
    pooled = float((resid ** 2).sum()) / (len(ts) - 2)
    assert eb.pooled_var_att == pytest.approx(pooled)
    assert eb.tau_att ** 2 == pytest.approx(max(pooled - float((ts["se_attack"] ** 2).mean()), FLOOR ** 2))


def test_variance_floor_binds_when_the_spread_is_below_the_noise(fold):
    big = fc.empirical_bayes(fold.history_rows, TARGET, 10.0)
    assert big.tau_att == big.tau_def == 10.0 and big.floor_binding_att and big.floor_binding_def


def test_empirical_bayes_ignores_target_and_later_outcomes_and_refuses_them(lg, fold, eb):
    changed = _reverse_scores(lg, lg["Season"].isin(["1718", "1819"]))
    eb2 = fc.empirical_bayes(build_fold(changed, HISTORY, TARGET).history_rows, TARGET, FLOOR)
    assert (eb2.a_promoted, eb2.b_promoted, eb2.tau_att, eb2.tau_def) == (eb.a_promoted, eb.b_promoted,
                                                                           eb.tau_att, eb.tau_def)
    with pytest.raises(SplitAccessError):
        fc.empirical_bayes(lg[lg["Season"].isin([*HISTORY, TARGET])], TARGET, FLOOR)


def test_prior_means_and_centring(fold, eb):
    s = fc.target_structure(fold.history_rows, fold.target_rows)
    m2 = fc.arm_prior(M2_HIERARCHICAL, s, eb)
    a_c, b_c = fc.continuing_means(eb, 6, 2)
    assert a_c == pytest.approx(-2 * eb.a_promoted / 4)
    assert m2.means["E"] == m2.means["N"] == (eb.a_promoted, eb.b_promoted)
    assert m2.means["A"] == (a_c, b_c)
    assert sum(a for a, _ in m2.means.values()) == pytest.approx(0, abs=1e-12)
    m1 = fc.arm_prior(M1_PROMOTED, s, eb)
    assert set(m1.means) == {"E", "N"}                            # continuing teams unpenalised in M1


# --- Season-start initialisation ---------------------------------------------------------------------

@pytest.mark.parametrize("arm", [M2_HIERARCHICAL, S1_IDENTITY_BREAK])
def test_never_seen_team_starts_exactly_at_the_promoted_prior(fold, eb, arm):
    theta, par, _, _ = _first_date_fit(fold, eb, arm)
    # With centred prior means the constraint's Lagrange multiplier is 0, so a matchless team sits at its mean.
    assert par.effects(theta).loc["N"].to_numpy() == pytest.approx([eb.a_promoted, eb.b_promoted], abs=1e-9)


def test_in_m1_the_constraint_moves_a_matchless_team_off_its_prior_mean(fold, eb):
    theta, par, prior, _ = _first_date_fit(fold, eb, M1_PROMOTED)
    eff = par.effects(theta)
    # Stationarity: att_N = a_P - tau^2 * lambda, with lambda the same for every constrained team.
    assert np.isfinite(eff.loc["N"]).all()
    assert abs(eff.loc["N", "attack"] - eb.a_promoted) > 1e-6


def test_s1_returning_team_starts_at_the_prior_and_m2_uses_its_decayed_history(fold, eb):
    theta_s1, par_s1, _, _ = _first_date_fit(fold, eb, S1_IDENTITY_BREAK)
    assert par_s1.effects(theta_s1).loc["E"].to_numpy() == pytest.approx([eb.a_promoted, eb.b_promoted], abs=1e-9)
    assert "E" + fc.PRE_ABSENCE_SUFFIX in par_s1.unconstrained
    theta_m2, par_m2, _, _ = _first_date_fit(fold, eb, M2_HIERARCHICAL)
    assert abs(par_m2.effects(theta_m2).loc["E", "attack"] - eb.a_promoted) > 1e-4   # history moves it
    # Its history enters with weight 2^(-age/730), age measured to the refit date.
    d = fold.target_rows["Date"].min()
    e_rows = fold.history_rows[(fold.history_rows["HomeTeam"] == "E") | (fold.history_rows["AwayTeam"] == "E")]
    w = exponential_decay_weights(e_rows["Date"], d, H)
    assert w.max() == pytest.approx(2 ** (-(d - e_rows["Date"].max()).days / H))
    assert w.max() < 0.7                                           # one season out: stale, not continuous


def test_continuing_team_prior_is_centred_and_data_dominate(fold, eb):
    theta, par, prior, _ = _first_date_fit(fold, eb, M2_HIERARCHICAL)
    a_c, _ = fc.continuing_means(eb, 6, 2)
    assert prior.means["A"][0] == pytest.approx(a_c)
    assert par.effects(theta).loc["A", "attack"] > par.effects(theta).loc["D", "attack"]   # A is truly stronger


# --- Weights x priors, home advantage, solver -------------------------------------------------------

def test_doubling_weights_equals_doubling_prior_variance(fold, eb):
    """f(theta; 2w, tau) = 2 [sum w l - 1/2 sum (theta - m)^2 / (2 tau^2)]: doubling the weights is the same as
    doubling tau^2 (halving the prior precision). The pre-registered test plan wrote 'halving tau^2', a slip in the
    test description; this is the identity implied by the registered objective."""
    s = fc.target_structure(fold.history_rows, fold.target_rows)
    hist = fold.history_rows
    par = fc.Parameterisation(s.teams, ("F", "G"))
    w = exponential_decay_weights(hist["Date"], fold.target_rows["Date"].min(), H)
    prior = fc.arm_prior(M2_HIERARCHICAL, s, eb)
    wider = fc.Prior(prior.means, prior.tau_att * math.sqrt(2), prior.tau_def * math.sqrt(2))
    t1, _ = fc.fit(hist, 2 * w, par, prior)
    t2, _ = fc.fit(hist, w, par, wider)
    assert np.max(np.abs(t1 - t2)) < 1e-8
    narrower = fc.Prior(prior.means, prior.tau_att / math.sqrt(2), prior.tau_def / math.sqrt(2))
    t3, _ = fc.fit(hist, w, par, narrower)
    assert np.max(np.abs(t1 - t3)) > 1e-4


def test_weights_are_measured_to_the_refit_date(fold, eb):
    """Without priors the reference does not matter; with priors it does, so the registered one is used."""
    s = fc.target_structure(fold.history_rows, fold.target_rows)
    hist, par = fold.history_rows, fc.Parameterisation(s.teams, ("F", "G"))
    prior = fc.arm_prior(M2_HIERARCHICAL, s, eb)
    d = fold.target_rows["Date"].min()
    t_refit, _ = fc.fit(hist, exponential_decay_weights(hist["Date"], d, H), par, prior)
    t_latest, _ = fc.fit(hist, exponential_decay_weights(hist["Date"], hist["Date"].max(), H), par, prior)
    assert np.max(np.abs(t_refit - t_latest)) > 1e-6


def test_home_advantage_is_global_and_learned_from_the_fit_set_only(lg, fold, eb, runs):
    preds, rec = runs[M2_HIERARCHICAL]
    hs = [f["home_advantage"] for f in rec["fits"]]
    assert hs[0] > 0 and len(set(np.round(hs, 12))) > 1                # re-estimated as matches arrive
    changed = _reverse_scores(lg, lg["Season"] == "1819")              # outside every fit set
    f2 = build_fold(changed, HISTORY, TARGET)
    _, rec2 = fc.online_fold(f2.history_rows, f2.target_rows, M2_HIERARCHICAL, eb, H, 10)
    assert [f["home_advantage"] for f in rec2["fits"]] == hs


def test_solver_failure_aborts_without_fallback(fold, eb):
    with pytest.raises(fc.FitFailure):
        fc.online_fold(fold.history_rows, fold.target_rows, M2_HIERARCHICAL, eb, H, 10,
                       fc.SolverSettings(max_iterations=1))


def test_existence_checks(fold):
    hist = fold.history_rows.copy()
    g_home, g_away = hist["HomeTeam"] == "G", hist["AwayTeam"] == "G"
    hist.loc[g_home, "FTHG"], hist.loc[g_away, "FTAG"] = 0, 0                # G never scores
    par = fc.Parameterisation(tuple("ABCDE"), ("F", "G"))
    with pytest.raises(fc.ExistenceCheckError, match="'G'"):
        fc.fit(hist, np.ones(len(hist)), par, fc.NO_PRIOR)
    with pytest.raises(fc.ExistenceCheckError, match="no prior"):
        fc.fit(fold.history_rows, np.ones(len(fold.history_rows)), fc.Parameterisation(tuple("ABCDEN"), ("F", "G")),
               fc.NO_PRIOR)


# --- Online information policy, coverage, validity, determinism ----------------------------------------

@pytest.mark.parametrize("arm", fc.ARMS)
def test_full_coverage_and_valid_probabilities(fold, runs, arm):
    preds, rec = runs[arm]
    assert len(preds) == len(fold.target_rows) and set(preds.index) == set(fold.target_rows["match_id"])
    check_forecast_frame(preds, [arm], atol=1e-12)
    assert rec["n_fits"] == fold.target_rows["Date"].nunique()
    assert {"FTR", "FTHG", "FTAG"}.isdisjoint(preds.columns)


def test_strong_home_side_gets_the_highest_home_win_probability(runs):
    preds, _ = runs[M2_HIERARCHICAL]
    p = preds.loc[(preds["HomeTeam"] == "A") & (preds["AwayTeam"] == "N"), prob_columns(M2_HIERARCHICAL)].iloc[0]
    assert p.iloc[0] > p.iloc[2]


@pytest.mark.parametrize("arm", fc.ARMS)
def test_season_start_forecasts_ignore_every_target_outcome(lg, fold, eb, runs, arm):
    preds, _ = runs[arm]
    changed = _reverse_scores(lg, lg["Season"] == TARGET)
    f2 = build_fold(changed, HISTORY, TARGET)
    eb2 = fc.empirical_bayes(f2.history_rows, TARGET, FLOOR)
    p2, _ = fc.online_fold(f2.history_rows, f2.target_rows, arm, eb2, H, 10)
    first = preds["Date"] == preds["Date"].min()
    assert np.array_equal(p2.loc[first, prob_columns(arm)].to_numpy(), preds.loc[first, prob_columns(arm)].to_numpy())


@pytest.mark.parametrize("round_index", [3, 6])
def test_online_forecasts_use_only_strictly_earlier_matches(lg, fold, eb, runs, round_index):
    preds, _ = runs[M2_HIERARCHICAL]
    cut = sorted(preds["Date"].unique())[round_index]
    changed = _reverse_scores(lg, (lg["Season"] == TARGET) & (lg["Date"] >= cut))
    f2 = build_fold(changed, HISTORY, TARGET)
    p2, _ = fc.online_fold(f2.history_rows, f2.target_rows, M2_HIERARCHICAL, eb, H, 10)
    cols = prob_columns(M2_HIERARCHICAL)
    upto = (preds["Date"] <= cut).to_numpy()
    assert np.array_equal(p2.loc[upto, cols].to_numpy(), preds.loc[upto, cols].to_numpy())
    later = (preds["Date"] > cut).to_numpy()
    assert not np.allclose(p2.loc[later, cols].to_numpy(), preds.loc[later, cols].to_numpy())


def test_refitting_is_deterministic(fold, eb, runs):
    again, _ = fc.online_fold(fold.history_rows, fold.target_rows, M2_HIERARCHICAL, eb, H, 10)
    pd.testing.assert_frame_equal(again, runs[M2_HIERARCHICAL][0], check_exact=True)


def test_later_seasons_never_enter_and_history_must_precede(lg, fold, eb):
    with pytest.raises(SplitAccessError):
        fc.online_fold(lg[lg["Season"].isin(["1617", "1718"])], lg[lg["Season"] == "1617"], M2_HIERARCHICAL, eb,
                       H, 10)


# --- M0 anchor -----------------------------------------------------------------------------------------

def test_anchor_reproduces_the_experiment_13_online_arm(lg, fold):
    anchor, _ = fc.online_fold(fold.history_rows, fold.target_rows, fc.ANCHOR, None, H, 10)
    ref, _ = op.online_fold_predictions(lg, HISTORY, TARGET, H, 10, 100, 1000)
    assert list(anchor.index) == list(ref.index)
    diff = np.abs(anchor[prob_columns("anchor")].to_numpy() - ref[prob_columns(op.ONLINE_ARM)].to_numpy())
    assert diff.max() < 1e-7


# --- Guards and protocol -----------------------------------------------------------------------------------

@pytest.mark.parametrize("history,target,error", [
    (("2122",), "2425", ExposedValidationError), (("2122",), "2223", DevTestAccessError),
    (("2324",), "2526", HoldoutAccessError), (("2425",), "2627", HoldoutAccessError)])
def test_target_season_guards(lg, history, target, error):
    with pytest.raises(error):
        build_fold(lg, history, target)


def test_holdout_rows_are_refused(fold, eb):
    hold = fold.target_rows.assign(Season="2526")
    with pytest.raises(HoldoutAccessError):
        fc.online_fold(fold.history_rows, hold, M2_HIERARCHICAL, eb, H, 10)


# Canonical SHA-256 (sorted-key JSON of the parsed TOML) of the config as frozen at the pre-registration commit
# 10e99d1, without [historical_locked], the only section written after registration. Checkable without git history.
FROZEN_CONFIG_CANONICAL_SHA256 = "4eb3dda75b774137e592c997472f47731f8d84d78cd46ca1bb3a90b92481f968"


def test_config_equals_the_frozen_preregistration_without_git_history():
    import hashlib
    import json

    cfg = dict(load_config(fcp.FULL_COVERAGE_CONFIG))
    cfg.pop("historical_locked")
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == FROZEN_CONFIG_CANONICAL_SHA256


def test_config_is_the_frozen_preregistration():
    """The run-time check (git show of the pre-registration commit); needs history, so CI's shallow checkout skips it."""
    import subprocess

    from eplmodel.paths import PROJECT_ROOT

    present = subprocess.run(["git", "cat-file", "-e", f"{fcp.PREREG_COMMIT}^{{commit}}"], cwd=PROJECT_ROOT,
                             capture_output=True).returncode == 0
    if not present:
        pytest.skip("pre-registration commit not in this checkout (shallow clone)")
    frozen = fcp.check_frozen(load_config(fcp.FULL_COVERAGE_CONFIG))
    assert frozen["prereg_commit"] == fcp.PREREG_COMMIT


def test_evidence_rules():
    ni = fcp.non_inferiority({"mean": -0.001, "clustered_se": 0.001}, 0.002, 2.0)
    assert ni["reading"] == "non_inferior_on_common_group"
    assert fcp.non_inferiority({"mean": 0.001, "clustered_se": 0.001}, 0.002, 2.0)["reading"] == \
        "inconclusive_on_common_group"
    assert fcp.non_inferiority({"mean": 0.01, "clustered_se": 0.001}, 0.002, 2.0)["reading"] == \
        "inferior_on_common_group"
    fold_ll = lambda m: {"log_loss": {"mean": m}}
    helps = {"pooled": {"log_loss": {"mean": -0.01, "clustered_se": 0.002}, "brier": {"mean": -0.004}},
             "per_target": {str(i): fold_ll(-0.01) for i in range(4)} | {"x": fold_ll(0.01)}}
    assert fcp.u_rule(helps, 0.002, 2.0, 4)["reading"] == "helps"
    helps["per_target"]["y"] = fold_ll(0.02)
    helps["per_target"].pop("0")
    assert fcp.u_rule(helps, 0.002, 2.0, 4)["reading"] == "not_distinguishable"


def test_experiment_is_not_in_run_all():
    from experiments import run_all
    assert fcp not in run_all.EXPERIMENTS


# --- Recorded Experiment 15 (needs the data) ---------------------------------------------------------------

@pytest.mark.golden
def test_recorded_experiment_15_reproduces():
    """Rerunning the locked historical stage (without writing) gives the recorded predictions and metrics."""
    import json

    from eplmodel.data.checksums import content_sha256
    from eplmodel.paths import PROCESSED_DEV_V2, RESULTS_DIR
    from eplmodel.reporting.results import _jsonable

    out_dir = RESULTS_DIR / load_config(fcp.FULL_COVERAGE_CONFIG)["outputs"]["results_name"]
    if not PROCESSED_DEV_V2.exists() or not (out_dir / "predictions.csv").exists():
        pytest.skip("dev_v2 or the recorded Experiment 15 predictions missing")
    lock = load_config(fcp.FULL_COVERAGE_CONFIG)["historical_locked"]
    assert content_sha256(out_dir / "predictions.csv") == lock["historical_predictions_sha256"]
    recorded = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["results"]
    recorded.pop("predictions")
    again = json.loads(json.dumps(_jsonable(fcp.run(write=False))))
    # The config hash changes only because [historical_locked] was written after the run; check_frozen (run
    # inside fcp.run) has already verified that every other section equals the pre-registration.
    for result in (recorded, again):
        result["frozen_state"].pop("config_sha256")
    assert again == recorded
