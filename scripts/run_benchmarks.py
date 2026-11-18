"""
Reproducible 20-Seed Benchmark Runner for EvoOpt-Surrogate-Bench.

Executes:
  1. Constrained Engineering Optimization (20 seeds):
     HEOA, DE, PSO (1,500 true evaluations) vs. SA-HEOA (GP Kriging, 160 evaluations)
     and SA-HEOA-DEns (PyTorch Deep Ensemble, 180 evaluations) across:
       - Three-Bar Truss Design (D=2, 3 stress constraints)
       - Tension/Compression Spring Design (D=3, 4 nonlinear constraints)
       - Cylindrical Pressure Vessel Design (D=4, 4 nonlinear constraints)
  2. Constraint-Handling Mechanism Ablation (20 seeds):
     Deb's Feasibility Rules vs. Epsilon-Constraint vs. Adaptive Dynamic Penalty.
  3. Multi-Objective & Constrained Environmental Reaction Optimization (20 seeds):
     NSGA-II & MOEA/D (2,400 evaluations) vs. SA-NSGA-II (220 evaluations) on
     ZDT1, ZDT3, and Peroxone (O3/H2O2) AOP Ozonation Reaction Kinetics (with
     WHO/EPA [BrO3-] <= 10 ug/L regulatory constraint).

Outputs:
  - results/engineering_opt_benchmark.json
  - results/pareto_metrics.csv
  - results/benchmark_trajectories.npz
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(platform, "_wmi"):
    platform._wmi = None

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evoopt as eo


N_SEEDS = 20
EA_BUDGET = 1500
SA_GP_BUDGET = 160
SA_DENS_BUDGET = 180

MOO_EA_BUDGET = 2400
MOO_SA_BUDGET = 220


def _interp_curve(eval_grid: np.ndarray, hist_evals: np.ndarray, hist_y: np.ndarray) -> np.ndarray:
    """Right-continuous step interpolation of best-so-far trajectory onto common eval grid."""
    idx = np.searchsorted(hist_evals, eval_grid, side="right") - 1
    idx = np.clip(idx, 0, len(hist_y) - 1)
    return hist_y[idx]


def _run_single_so_job(args: Tuple[str, str, int]) -> Dict[str, Any]:
    prob_key, alg_name, seed = args
    prob = eo.get_problem(prob_key)

    if alg_name == "HEOA":
        opt = eo.HEOA(pop_size=50, max_evals=EA_BUDGET, constraint_handler="deb", seed=seed)
        grid = np.linspace(50, EA_BUDGET, 60)
    elif alg_name == "DE":
        opt = eo.DE(pop_size=50, max_evals=EA_BUDGET, constraint_handler="deb", seed=seed)
        grid = np.linspace(50, EA_BUDGET, 60)
    elif alg_name == "PSO":
        opt = eo.PSO(pop_size=50, max_evals=EA_BUDGET, constraint_handler="deb", seed=seed)
        grid = np.linspace(50, EA_BUDGET, 60)
    elif alg_name == "SA-HEOA":
        opt = eo.SurrogateAssistedHEOA(
            max_evals=SA_GP_BUDGET,
            n_init=24,
            surrogate_type="gp",
            infill_per_iter=6,
            seed=seed,
        )
        grid = np.linspace(24, SA_GP_BUDGET, 35)
    elif alg_name == "SA-HEOA-DEns":
        opt = eo.SurrogateAssistedHEOA(
            max_evals=SA_DENS_BUDGET,
            n_init=24,
            surrogate_type="deep_ensemble",
            infill_per_iter=12,
            inner_pop_size=28,
            inner_max_evals=224,
            seed=seed,
        )
        grid = np.linspace(24, SA_DENS_BUDGET, 40)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

    res = opt.optimize(prob)
    feasible_history = np.where(res.history_cv <= 0.0, res.history_f, np.nan)
    curve = _interp_curve(grid, res.history_evals, feasible_history)
    rel_err = abs(res.f_best - prob.f_opt) / max(abs(prob.f_opt), 1e-12)

    return {
        "problem": prob.name,
        "prob_key": prob_key,
        "algorithm": alg_name,
        "seed": seed,
        "f_best": float(res.f_best),
        "f_opt": float(prob.f_opt),
        "abs_error": float(abs(res.f_best - prob.f_opt)),
        "rel_error_pct": float(100.0 * rel_err),
        "cv_best": float(res.cv_best),
        "is_feasible": bool(res.is_feasible),
        "n_evals": int(res.n_evals),
        "x_best": res.x_best.tolist(),
        "grid": grid,
        "curve": curve,
    }


def _run_ablation_job(args: Tuple[str, str, int]) -> Dict[str, Any]:
    prob_key, handler_name, seed = args
    prob = eo.get_problem(prob_key)
    opt = eo.HEOA(
        pop_size=50,
        max_evals=EA_BUDGET,
        constraint_handler=handler_name,
        seed=seed,
    )
    res = opt.optimize(prob)
    rel_err = abs(res.f_best - prob.f_opt) / max(abs(prob.f_opt), 1e-12)
    return {
        "problem": prob.name,
        "prob_key": prob_key,
        "handler": handler_name,
        "seed": seed,
        "f_best": float(res.f_best),
        "rel_error_pct": float(100.0 * rel_err),
        "cv_best": float(res.cv_best),
        "is_feasible": bool(res.is_feasible),
    }


def _build_moo_problem(prob_key: str) -> eo.MultiObjectiveProblem:
    if prob_key == "zdt1":
        return eo.ZDT1(dim=8)
    if prob_key == "zdt3":
        return eo.ZDT3(dim=8)
    if prob_key == "aop_ozonation":
        return eo.AOPOzonationKineticsMOO()
    raise ValueError(f"Unknown MOO problem: {prob_key}")


def _run_moo_job(args: Tuple[str, str, int, int]) -> Dict[str, Any]:
    prob_key, alg_name, budget, seed = args
    prob = _build_moo_problem(prob_key)
    pf_ref = prob.pareto_front(n_points=250)

    if alg_name == "NSGA-II":
        opt = eo.NSGA2(pop_size=80, max_evals=budget, seed=seed)
    elif alg_name == "MOEA/D":
        opt = eo.MOEAD(pop_size=80, max_evals=budget, seed=seed)
    elif alg_name == "SA-NSGA-II":
        opt = eo.SurrogateAssistedNSGA2(
            max_evals=budget,
            n_init=40,
            infill_per_iter=12,
            inner_pop_size=52,
            inner_max_evals=728,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown MOO algorithm: {alg_name}")

    res = opt.optimize(prob, track_hv=False)
    feasible_front = res.F_pareto[eo.compute_cv(res.G_pareto) <= 0.0]
    hv = eo.compute_hypervolume(feasible_front, prob.ref_point)
    igd = eo.compute_igd(feasible_front, pf_ref, normalize=False)

    return {
        "problem": prob.name,
        "prob_key": prob_key,
        "algorithm": alg_name,
        "seed": seed,
        "n_evals": int(res.n_evals),
        "n_pareto": int(len(feasible_front)),
        "hv": float(hv),
        "igd": float(igd),
        "feasible": bool(len(feasible_front)),
    }


def _feasible_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    good = [r for r in runs if r["is_feasible"]]
    f = [r["f_best"] for r in good]
    err = [r["rel_error_pct"] for r in good]
    return {
        "feasibility_rate_pct": 100.0 * len(good) / len(runs),
        "f_best_min": float(np.min(f)) if good else None,
        "f_best_median": float(np.median(f)) if good else None,
        "f_best_mean": float(np.mean(f)) if good else None,
        "f_best_std": float(np.std(f)) if good else None,
        "rel_error_mean_pct": float(np.mean(err)) if good else None,
        "rel_error_median_pct": float(np.median(err)) if good else None,
    }


def main() -> None:
    t_start = time.time()
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_workers = min(8, os.cpu_count() or 4)
    print(f"[EvoOpt-Bench] Running 20-seed benchmarks across {n_workers} parallel workers...")

    so_problems = ["three_bar_truss", "tension_compression_spring", "pressure_vessel"]
    so_algs = ["HEOA", "DE", "PSO", "SA-HEOA", "SA-HEOA-DEns"]
    so_jobs = [
        (p, a, s) for p in so_problems for a in so_algs for s in range(N_SEEDS)
    ]

    handlers = ["deb", "epsilon", "penalty"]
    ablation_jobs = [
        (p, h, s) for p in so_problems for h in handlers for s in range(N_SEEDS)
    ]

    moo_problems = ["aop_ozonation", "zdt1", "zdt3"]
    moo_configs = [
        ("NSGA-II", MOO_EA_BUDGET), ("MOEA/D", MOO_EA_BUDGET),
        ("NSGA-II", MOO_SA_BUDGET), ("MOEA/D", MOO_SA_BUDGET),
        ("SA-NSGA-II", MOO_SA_BUDGET),
    ]
    moo_jobs = [
        (p, a, b, s) for p in moo_problems for a, b in moo_configs for s in range(N_SEEDS)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        so_raw = list(pool.map(_run_single_so_job, so_jobs))
        print(f"  [1/3] Single-objective engineering runs complete ({len(so_raw)} runs, {time.time()-t_start:.1f}s).")

        ablation_raw = list(pool.map(_run_ablation_job, ablation_jobs))
        print(f"  [2/3] Constraint-handling ablation runs complete ({len(ablation_raw)} runs, {time.time()-t_start:.1f}s).")

        moo_raw = list(pool.map(_run_moo_job, moo_jobs))
        print(f"  [3/3] Multi-objective Pareto runs complete ({len(moo_raw)} runs, {time.time()-t_start:.1f}s).")

    # ------------------------------------------------------------------
    # Aggregate & Save Structured JSON + CSV + Trajectory Cache
    # ------------------------------------------------------------------
    so_summary: Dict[str, Any] = {}
    traj_npz: Dict[str, np.ndarray] = {}

    for p_key in so_problems:
        prob_obj = eo.get_problem(p_key)
        so_summary[prob_obj.name] = {
            "dim": prob_obj.dim,
            "n_constraints": prob_obj.n_constraints,
            "f_opt": float(prob_obj.f_opt),
            "x_opt": prob_obj.x_opt.tolist(),
            "algorithms": {},
        }
        for alg in so_algs:
            sub = [r for r in so_raw if r["prob_key"] == p_key and r["algorithm"] == alg]
            f_vals = np.array([r["f_best"] for r in sub], dtype=np.float64)
            feas_flags = np.array([r["is_feasible"] for r in sub], dtype=bool)
            n_evals = int(sub[0]["n_evals"])
            eval_savings = 100.0 * (1.0 - n_evals / EA_BUDGET)

            curves = np.vstack([r["curve"] for r in sub])
            grid = sub[0]["grid"]
            key_prefix = f"{p_key}__{alg.replace('-', '_')}"
            traj_npz[f"{key_prefix}__grid"] = grid
            traj_npz[f"{key_prefix}__curves"] = curves
            traj_npz[f"{key_prefix}__f_vals"] = np.where(feas_flags, f_vals, np.nan)

            best = min((r for r in sub if r["is_feasible"]), key=lambda r: r["f_best"], default=None)
            so_summary[prob_obj.name]["algorithms"][alg] = {
                "n_seeds": N_SEEDS,
                "n_evals": n_evals,
                "eval_savings_pct": round(eval_savings, 2),
                **_feasible_stats(sub),
                "best_x": best["x_best"] if best else None,
            }

    ablation_summary: Dict[str, Any] = {}
    handler_labels = {
        "deb": "Deb's Feasibility Rules",
        "epsilon": "Epsilon-Constraint",
        "penalty": "Adaptive Penalty",
    }
    for p_key in so_problems:
        prob_obj = eo.get_problem(p_key)
        ablation_summary[prob_obj.name] = {}
        for h_key in handlers:
            sub = [
                r
                for r in ablation_raw
                if r["prob_key"] == p_key and r["handler"] == h_key
            ]
            f_vals = np.array([r["f_best"] for r in sub], dtype=np.float64)
            rel_errs = np.array([r["rel_error_pct"] for r in sub], dtype=np.float64)
            feas_flags = np.array([r["is_feasible"] for r in sub], dtype=bool)
            traj_npz[f"ablation__{p_key}__{h_key}__f_vals"] = np.where(feas_flags, f_vals, np.nan)
            traj_npz[f"ablation__{p_key}__{h_key}__rel_err"] = np.where(feas_flags, rel_errs, np.nan)

            ablation_summary[prob_obj.name][handler_labels[h_key]] = {
                "n_seeds": N_SEEDS,
                "n_evals": EA_BUDGET,
                **_feasible_stats(sub),
                "seed_f_values": [r["f_best"] if r["is_feasible"] else None for r in sub],
            }

    # Compute Knee-Point engineering diagnostics on AOP Ozonation Reaction Kinetics (seed=0)
    aop_prob = eo.AOPOzonationKineticsMOO()
    aop_sa_rep = eo.SurrogateAssistedNSGA2(
        max_evals=MOO_SA_BUDGET,
        n_init=40,
        infill_per_iter=12,
        inner_pop_size=52,
        inner_max_evals=728,
        seed=0,
    ).optimize(aop_prob, track_hv=False)
    knee_idx = eo.find_knee_point(aop_sa_rep.F_pareto)
    x_knee = aop_sa_rep.X_pareto[knee_idx]
    knee_diag = aop_prob.evaluate_diagnostics(x_knee)
    traj_npz["aop_rep__SA_NSGA_II__X_pareto"] = aop_sa_rep.X_pareto
    traj_npz["aop_rep__SA_NSGA_II__F_pareto"] = aop_sa_rep.F_pareto
    traj_npz["aop_rep__knee_idx"] = np.array([knee_idx], dtype=np.int64)

    moo_rows: List[Dict[str, Any]] = []
    for p_key in moo_problems:
        prob_obj = _build_moo_problem(p_key)
        pf_ref = prob_obj.pareto_front(250)
        hv_ref = eo.compute_hypervolume(pf_ref, prob_obj.ref_point)
        for alg, budget in moo_configs:
            sub = [r for r in moo_raw if r["prob_key"] == p_key and r["algorithm"] == alg and r["n_evals"] == budget]
            hvs = np.array([r["hv"] for r in sub], dtype=np.float64)
            igds = np.array([r["igd"] for r in sub], dtype=np.float64)
            feas = np.array([r["feasible"] for r in sub], dtype=bool)
            n_evals = int(sub[0]["n_evals"])
            savings = 100.0 * (1.0 - n_evals / MOO_EA_BUDGET)

            moo_rows.append(
                {
                    "Problem": prob_obj.name,
                    "Algorithm": alg,
                    "Seeds": N_SEEDS,
                    "True_Evaluations": n_evals,
                    "Eval_Savings_Pct": round(savings, 2),
                    "Reference_HV": round(hv_ref, 6),
                    "HV_Mean": round(float(np.mean(hvs)), 6),
                    "HV_Std": round(float(np.std(hvs)), 6),
                    "HV_Median": round(float(np.median(hvs)), 6),
                    "HV_Best": round(float(np.max(hvs)), 6),
                    "HV_Ratio_Pct": round(100.0 * float(np.mean(hvs)) / max(hv_ref, 1e-12), 2),
                    "IGD_Mean": round(float(np.mean(igds)), 6),
                    "IGD_Std": round(float(np.std(igds)), 6),
                    "IGD_Median": round(float(np.median(igds)), 6),
                    "IGD_Best": round(float(np.min(igds)), 6),
                    "Feasibility_Rate_Pct": round(100.0 * float(np.mean(feas)), 2),
                }
            )

    moo_df = pd.DataFrame(moo_rows)
    csv_path = results_dir / "pareto_metrics.csv"
    moo_df.to_csv(csv_path, index=False)

    full_report = {
        "metadata": {
            "framework": "EvoOpt-Surrogate-Bench",
            "n_seeds": N_SEEDS,
            "ea_budget": EA_BUDGET,
            "sa_heoa_gp_budget": SA_GP_BUDGET,
            "sa_heoa_deep_ensemble_budget": SA_DENS_BUDGET,
            "moo_ea_budget": MOO_EA_BUDGET,
            "moo_sa_budget": MOO_SA_BUDGET,
            "elapsed_seconds": round(time.time() - t_start, 2),
        },
        "single_objective_benchmarks": so_summary,
        "constraint_handling_ablation": ablation_summary,
        "multi_objective_benchmarks": moo_rows,
        "multi_objective_runs": moo_raw,
        "single_objective_runs": [{k: v for k, v in r.items() if k not in ("grid", "curve")} for r in so_raw],
        "constraint_handling_runs": ablation_raw,
        "aop_ozonation_knee_point": {
            "pH": round(float(x_knee[0]), 3),
            "O3_dose_mg_L": round(float(x_knee[1]), 3),
            "H2O2_O3_molar_ratio": round(float(x_knee[2]), 3),
            "HRT_min": round(float(x_knee[3]), 2),
            "micropollutant_removal_pct": round(100.0 * knee_diag["eta_removal"], 2),
            "EEO_kWh_m3_order": round(knee_diag["eeo_kwh_m3_order"], 4),
            "specific_energy_kWh_m3": round(knee_diag["energy_kwh_m3"], 4),
            "bromate_ug_L": round(knee_diag["bromate_ug_L"], 3),
            "residual_H2O2_mg_L": round(knee_diag["h2o2_res_mg_L"], 3),
        },
    }

    json_path = results_dir / "engineering_opt_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(full_report, fp, indent=2)

    npz_path = results_dir / "benchmark_trajectories.npz"
    np.savez_compressed(npz_path, **traj_npz)

    print(f"\n[EvoOpt-Bench] Completed in {full_report['metadata']['elapsed_seconds']}s.")
    print(f"  Saved JSON -> {json_path}")
    print(f"  Saved CSV  -> {csv_path}")
    print("\n--- Single-Objective Summary (20 Seeds) ---")
    for p_name, p_data in so_summary.items():
        print(f"\n{p_name} (f* = {p_data['f_opt']:.6f}):")
        for alg, st in p_data["algorithms"].items():
            if st["f_best_mean"] is None:
                print(f"  {alg:14s} | no feasible run")
                continue
            print(
                f"  {alg:14s} | Evals={st['n_evals']:4d} (Save {st['eval_savings_pct']:5.1f}%) "
                f"| Mean={st['f_best_mean']:.6f} +/- {st['f_best_std']:.6f} "
                f"| Best={st['f_best_min']:.6f} | RelErr={st['rel_error_mean_pct']:.4f}%"
            )

    print("\n--- Multi-Objective Summary (20 Seeds) ---")
    print(moo_df.to_string(index=False))
    print("\n--- AOP Ozonation Knee-Point Operating Condition ---")
    print(json.dumps(full_report["aop_ozonation_knee_point"], indent=2))


if __name__ == "__main__":
    main()
