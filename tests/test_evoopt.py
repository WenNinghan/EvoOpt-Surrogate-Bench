"""
Comprehensive Unit & Integration Test Suite for EvoOpt-Surrogate-Bench.

Verifies:
  1. Engineering & AOP reaction benchmark formulations and analytical global optima
  2. Deb's Feasibility Rules, Adaptive Penalty, and Epsilon-Constraint dominance
  3. Constrained non-dominated sorting, crowding distance, exact Hypervolume, IGD, Knee-Point
  4. HEOA, DE, and PSO convergence on constrained structural design benchmarks
  5. Kriging (GP) & PyTorch Deep Ensemble surrogates, CEI = EI * PoF infill criteria,
     and strict evaluation budget adherence of SA-HEOA and SA-NSGA-II
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import evoopt as eo


# ==============================================================================
# 1. Benchmark Formulations & Analytical Optima
# ==============================================================================


def test_three_bar_truss_analytical_optimum():
    prob = eo.ThreeBarTruss()
    f_val, g_val = prob.evaluate(prob.x_opt)
    assert np.isclose(f_val, 263.895843376, atol=1e-6)
    assert np.all(g_val <= 1e-9), f"Expected feasible x_opt, got g={g_val}"
    # Active stress constraint g_1(x*) == 0
    assert np.isclose(g_val[0], 0.0, atol=1e-8)


def test_spring_vessel_and_welded_beam_optima():
    spring = eo.TensionCompressionSpring()
    f_s, g_s = spring.evaluate(spring.x_opt)
    assert np.isclose(f_s, 0.0126652328, atol=1e-8)
    assert np.all(g_s <= 0.0), f"Spring x_opt must be strictly feasible, got g={g_s}"
    assert eo.compute_cv(g_s) == 0.0

    vessel_cont = eo.PressureVessel(discrete_thickness=False)
    f_vc, g_vc = vessel_cont.evaluate(vessel_cont.x_opt)
    assert np.isclose(f_vc, 5885.3327736, atol=1e-6)
    assert np.all(g_vc <= 0.0)
    assert eo.compute_cv(g_vc) == 0.0

    vessel_disc = eo.PressureVessel(discrete_thickness=True)
    f_vd, g_vd = vessel_disc.evaluate(vessel_disc.x_opt)
    assert np.isclose(f_vd, 6059.71433505, atol=1e-6)
    assert np.isclose(f_vd, vessel_disc.f_opt, atol=1e-10)
    assert np.all(g_vd <= 0.0)
    assert eo.compute_cv(g_vd) == 0.0

    beam = eo.WeldedBeam()
    f_b, g_b = beam.evaluate(beam.x_opt)
    assert np.isclose(f_b, 1.7248523086, atol=1e-8)
    assert np.all(g_b <= 0.0)
    assert eo.compute_cv(g_b) == 0.0


def test_aop_ozonation_kinetics_and_bromate_constraint():
    aop = eo.AOPOzonationKineticsMOO()
    # High O3 without H2O2 at alkaline pH should trigger bromate > 10 ug/L (g1 > 0)
    x_unsafe = np.array([8.6, 10.5, 0.0, 20.0])
    f_u, g_u = aop.evaluate(x_unsafe)
    diag_u = aop.evaluate_diagnostics(x_unsafe)
    assert diag_u["bromate_ug_L"] > 10.0
    assert g_u[0] > 0.0

    # Excess H2O2 (r = 1.2 > 0.60 stoichiometric limit) at high O3 persists in effluent (g3 > 0)
    x_excess_h2o2 = np.array([8.5, 11.5, 1.2, 30.0])
    _, g_ex = aop.evaluate(x_excess_h2o2)
    diag_ex = aop.evaluate_diagnostics(x_excess_h2o2)
    assert diag_ex["h2o2_res_mg_L"] > 3.5
    assert g_ex[2] > 0.0

    # Moderate O3 with peroxone H2O2 dosing at neutral pH satisfies all constraints
    x_safe = np.array([7.2, 4.2, 0.55, 16.0])
    f_s, g_s = aop.evaluate(x_safe)
    diag_s = aop.evaluate_diagnostics(x_safe)
    assert diag_s["bromate_ug_L"] <= 10.0
    assert np.all(g_s <= 0.0)
    assert 0.65 <= diag_s["eta_removal"] < 1.0
    # Verify exact Bolton (2001) IUPAC EE/O = E_elec / log10(C0 / Ct)
    log10_removal = -np.log10(diag_s["f1_residual"])
    assert np.isclose(diag_s["eeo_kwh_m3_order"], diag_s["energy_kwh_m3"] / log10_removal, rtol=1e-10)

    pf = aop.pareto_front(n_points=60)
    assert pf.shape == (60, 2)
    assert np.all(np.diff(pf[:, 0]) > 0)
    assert np.all(np.diff(pf[:, 1]) <= 1e-8)

    # Verify OSY 5-region Deb (2002) analytical Pareto front coverage
    osy = eo.OSY()
    pf_osy = osy.pareto_front(n_points=120)
    assert pf_osy[0, 0] <= -273.5 and pf_osy[-1, 0] >= -43.0
    assert pf_osy[-1, 1] <= 4.1 and pf_osy[0, 1] >= 75.5


# ==============================================================================
# 2. Constraint-Handling Mechanisms
# ==============================================================================


def test_deb_feasibility_rules():
    deb = eo.DebFeasibilityRules()
    # Rule 1: Both feasible -> smaller objective wins
    assert deb.is_better(10.0, np.array([-0.1, -0.2]), 12.0, np.array([-0.5, -0.5]))
    assert not deb.is_better(15.0, np.array([-0.1, -0.2]), 12.0, np.array([-0.5, -0.5]))

    # Rule 2: Feasible beats infeasible even if infeasible has tiny objective
    assert deb.is_better(500.0, np.array([0.0, -0.1]), 1.0, np.array([0.05, -0.2]))
    assert not deb.is_better(1.0, np.array([0.05, -0.2]), 500.0, np.array([0.0, -0.1]))

    # Rule 3: Both infeasible -> smaller total CV wins regardless of objective
    assert deb.is_better(999.0, np.array([0.1, 0.0]), 1.0, np.array([0.2, 0.2]))

    # Population ranking check
    f_pop = np.array([1.0, 200.0, 150.0, 50.0])
    G_pop = np.array([
        [0.5, 0.5],   # CV = 1.0 (infeasible, worst)
        [-0.1, 0.0],  # CV = 0.0 (feasible, f=200)
        [-0.2, -0.1], # CV = 0.0 (feasible, f=150 -> best!)
        [0.2, 0.0],   # CV = 0.2 (infeasible, 3rd)
    ])
    ranked = deb.rank_population(f_pop, G_pop)
    assert list(ranked) == [2, 1, 3, 0]


def test_epsilon_and_penalty_handlers():
    f_init = np.array([10.0, 20.0, 30.0, 40.0])
    G_init = np.array([[0.0, -0.1], [0.2, 0.0], [0.6, 0.2], [1.2, 0.3]])

    eps_h = eo.EpsilonConstraintHandler(theta_quantile=0.5, cp=3.0, cutoff_ratio=0.6)
    eps_h.initialize(f_init, G_init)
    assert eps_h.epsilon_level(0.0) > 0.0
    assert eps_h.epsilon_level(0.7) == 0.0  # Strictly zero after cutoff_ratio

    pen_h = eo.AdaptivePenaltyConstraintHandler()
    pen_h.initialize(f_init, G_init)
    pen_early = pen_h.penalized_objective(f_init, G_init, progress=0.1)
    pen_late = pen_h.penalized_objective(f_init, G_init, progress=0.9)
    # Infeasible solutions incur strictly higher penalty at late generations
    assert pen_late[3] > pen_early[3]
    assert np.isclose(pen_late[0], pen_early[0])


# ==============================================================================
# 3. Non-Dominated Sorting, Hypervolume, IGD, and Knee-Point
# ==============================================================================


def test_fast_non_dominated_sort_and_crowding():
    F = np.array([
        [0.1, 0.9],  # Front 0 (feasible)
        [0.5, 0.5],  # Front 0 (feasible)
        [0.9, 0.1],  # Front 0 (feasible)
        [0.6, 0.7],  # Front 1 (dominated by [0.5, 0.5])
        [0.0, 0.0],  # Infeasible -> pushed to last front despite utopia objective!
    ])
    G = np.array([
        [-1.0],
        [-0.5],
        [0.0],
        [-0.2],
        [0.4],
    ])
    fronts = eo.fast_non_dominated_sort(F, G)
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]
    assert fronts[2] == [4]

    cd = eo.calculate_crowding_distance(F, fronts[0])
    assert np.isinf(cd[0]) and np.isinf(cd[2])
    assert np.isfinite(cd[1]) and cd[1] > 0.0


def test_exact_hypervolume_and_igd():
    # Single point [0.2, 0.4] w.r.t. ref [1.0, 1.0] -> area = 0.8 * 0.6 = 0.48
    hv_single = eo.compute_hypervolume(np.array([[0.2, 0.4]]), np.array([1.0, 1.0]))
    assert np.isclose(hv_single, 0.48, atol=1e-12)

    # Analytical ZDT1 HV w.r.t. [1.1, 1.1]: 1.1*1.1 - int_0^1 (1 - sqrt(x)) dx = 1.21 - 1/3 = 0.876667
    zdt1 = eo.ZDT1(dim=10)
    pf_dense = zdt1.pareto_front(n_points=4000)
    hv_zdt1 = eo.compute_hypervolume(pf_dense, zdt1.ref_point)
    assert np.isclose(hv_zdt1, 1.21 - 1.0 / 3.0, atol=5e-4)

    # IGD of true PF against itself is zero
    assert np.isclose(eo.compute_igd(pf_dense, pf_dense), 0.0, atol=1e-12)

    # Knee point on convex curve f2 = (1 - f1)^3 lies near symmetric bend f1 ~ 0.25-0.40
    t = np.linspace(0.0, 1.0, 101)
    curve = np.column_stack([t, (1.0 - t) ** 3])
    knee_idx = eo.find_knee_point(curve)
    assert 20 <= knee_idx <= 45


# ==============================================================================
# 4. Evolutionary & Surrogate-Assisted Optimizers
# ==============================================================================


def test_heoa_convergence_on_three_bar_truss():
    prob = eo.ThreeBarTruss()
    opt = eo.HEOA(pop_size=40, max_evals=1200, seed=42)
    res = opt.optimize(prob)
    assert res.is_feasible
    assert res.cv_best == 0.0
    assert res.n_evals == 1200
    assert abs(res.f_best - prob.f_opt) < 0.05


def test_surrogates_and_acquisition_criteria():
    rng = np.random.default_rng(7)
    lb, ub = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    X_train = rng.uniform(0.0, 1.0, size=(25, 2))
    Y_train = np.column_stack([
        np.sin(3.0 * X_train[:, 0]) + X_train[:, 1] ** 2,
        X_train[:, 0] + X_train[:, 1] - 1.0,
    ])

    for surr in [
        eo.GaussianProcessSurrogate(lb, ub, n_restarts=2, seed=7),
        eo.DeepEnsembleSurrogate(lb, ub, n_members=4, epochs=120, seed=7),
    ]:
        surr.fit(X_train, Y_train)
        mu, std = surr.predict(X_train[:5])
        assert mu.shape == (5, 2)
        assert std.shape == (5, 2)
        assert np.all(std > 0.0)

        cei = eo.constrained_expected_improvement(
            mu[:, 0], std[:, 0], mu[:, 1:], std[:, 1:], f_best=0.2
        )
        assert cei.shape == (5,)
        assert np.all(cei >= 0.0)


def test_de_pso_and_moead_baselines():
    truss = eo.ThreeBarTruss()
    de_res = eo.DE(pop_size=40, max_evals=1200, seed=5).optimize(truss)
    pso_res = eo.PSO(pop_size=40, max_evals=1200, seed=5).optimize(truss)
    assert de_res.is_feasible and abs(de_res.f_best - truss.f_opt) < 0.05
    assert pso_res.is_feasible and abs(pso_res.f_best - truss.f_opt) < 0.25

    zdt1 = eo.ZDT1(dim=8)
    moead_res = eo.MOEAD(pop_size=60, max_evals=1800, seed=3).optimize(zdt1, track_hv=False)
    hv_m = eo.compute_hypervolume(moead_res.F_pareto, zdt1.ref_point)
    assert hv_m > 0.78
    # Verify MOEA/D covers the full [0, 1] Pareto front including f1 > 0.85
    assert np.max(moead_res.F_pareto[:, 0]) > 0.85


def test_sa_heoa_deep_ensemble_pure_budget():
    prob = eo.ThreeBarTruss()
    budget = 72
    sa_dens = eo.SurrogateAssistedHEOA(
        max_evals=budget,
        n_init=24,
        surrogate_type="deep_ensemble",
        infill_per_iter=12,
        inner_pop_size=24,
        inner_max_evals=192,
        seed=11,
    )
    res = sa_dens.optimize(prob)
    assert res.algorithm == "SA-HEOA-DEns"
    assert res.n_evals == budget
    assert len(res.evaluated_X) == budget
    assert res.is_feasible


def test_sa_heoa_strict_budget_and_efficiency():
    prob = eo.ThreeBarTruss()
    budget = 90
    sa_heoa = eo.SurrogateAssistedHEOA(
        max_evals=budget, n_init=20, surrogate_type="gp", seed=12
    )
    res = sa_heoa.optimize(prob)
    assert res.n_evals == budget
    assert len(res.evaluated_X) == budget
    assert res.is_feasible
    assert abs(res.f_best - prob.f_opt) < 0.05


def test_sa_nsga2_budget_and_bromate_compliance():
    aop = eo.AOPOzonationKineticsMOO()
    budget = 96
    sa_nsga2 = eo.SurrogateAssistedNSGA2(
        max_evals=budget,
        n_init=32,
        infill_per_iter=8,
        inner_pop_size=40,
        inner_max_evals=400,
        seed=21,
    )
    res = sa_nsga2.optimize(aop)
    assert res.n_evals == budget
    assert len(res.evaluated_X) == budget
    assert len(res.F_pareto) >= 5
    # Every returned Pareto solution must satisfy [BrO3-] <= 10 ug/L (g_1 <= 0)
    assert np.all(res.G_pareto <= 1e-9)
    hv = eo.compute_hypervolume(res.F_pareto, aop.ref_point)
    assert hv > 0.03


def test_held_out_benchmark_protocol_and_archived_fronts():
    path = Path(__file__).resolve().parents[1] / "results" / "small_budget_tuning.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report["test_seeds"]).isdisjoint(report["tuning_seeds"])
    problems = {"aop_ozonation": eo.AOPOzonationKineticsMOO(), "zdt1": eo.ZDT1(dim=8), "zdt3": eo.ZDT3(dim=8)}
    fronts = {k: prob.pareto_front(n_points=250) for k, prob in problems.items()}
    for problem_name, selections in report["selections"].items():
        for alg, selection in selections.items():
            grid = report["candidate_grid"][alg]
            tuning = [r for r in report["tuning_runs"] if r["problem"] == problem_name and r["algorithm"] == alg]
            scores = []
            for candidate in grid:
                matches = [r for r in tuning if r["parameters"] == candidate]
                assert sorted(r["seed"] for r in matches) == report["tuning_seeds"]
                scores.append(np.mean([r["hv_ratio_pct"] for r in matches]))
            assert grid[int(np.argmax(scores))] == selection["parameters"]
    for row in report["test_runs"]:
        problem = problems[row["problem"]]
        if row["configuration"] == "tuned":
            assert row["parameters"] == report["selections"][row["problem"]][row["algorithm"]]["parameters"]
        assert row["seed"] in report["test_seeds"]
        assert row["n_evals"] == report["budget_per_run"]
        if not row["n_feasible_pareto"]:
            assert row["hv"] == 0 and row["igd"] is None
            continue
        objectives, constraints = problem.evaluate(np.asarray(row["front_x"]))
        np.testing.assert_allclose(objectives, row["front_f"], rtol=1e-10, atol=1e-12)
        assert np.all(eo.compute_cv(constraints) <= 0)
        assert eo.compute_hypervolume(objectives, problem.ref_point) == pytest.approx(row["hv"])
        assert eo.compute_igd(objectives, fronts[row["problem"]]) == pytest.approx(row["igd"])
