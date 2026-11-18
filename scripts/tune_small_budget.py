"""Select small-budget direct optimizers on development seeds, then test frozen configurations."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from run_benchmarks import ROOT, _build_moo_problem
import evoopt as eo
import numpy as np
from threadpoolctl import threadpool_limits


BUDGET = 220
PROBLEMS = ("aop_ozonation", "zdt1", "zdt3")
BASELINES = ("NSGA-II", "MOEA/D")
TUNING_SEEDS = tuple(range(100, 108))
TEST_SEEDS = tuple(range(1000, 1020))
SA_CONFIG = {"n_init": 40, "infill_per_iter": 12, "inner_pop_size": 52, "inner_max_evals": 728}


def candidates(algorithm):
    options = [{"pop_size": 80}]  # Original benchmark, also eligible for selection.
    for pop in (10, 20, 40):
        for eta_m in (5, 20):
            if algorithm == "NSGA-II":
                options.extend({"pop_size": pop, "eta_m": eta_m, "eta_c": eta_c} for eta_c in (5, 20))
            else:
                options.extend(
                    {"pop_size": pop, "eta_m": eta_m, "n_neighbors": max(3, int(frac * pop))}
                    for frac in (0.25, 0.75)
                )
    return options


@lru_cache(maxsize=None)
def reference(problem_key):
    problem = _build_moo_problem(problem_key)
    front = problem.pareto_front(n_points=250)
    return front, eo.compute_hypervolume(front, problem.ref_point)


def run(job):
    problem_key, algorithm, config, seed = job
    problem = _build_moo_problem(problem_key)
    front_ref, hv_ref = reference(problem_key)
    optimizer = {"NSGA-II": eo.NSGA2, "MOEA/D": eo.MOEAD, "SA-NSGA-II": eo.SurrogateAssistedNSGA2}[algorithm]
    result = optimizer(max_evals=BUDGET, seed=seed, **config).optimize(problem, track_hv=False)
    if result.n_evals != BUDGET or len(result.evaluated_X) != BUDGET:
        raise RuntimeError(f"Evaluation archive does not match budget for {job}")
    feasible = eo.compute_cv(result.G_pareto) <= 0.0
    front = result.F_pareto[feasible]
    hv = eo.compute_hypervolume(front, problem.ref_point)
    record = {
        "problem": problem_key, "algorithm": algorithm, "parameters": config, "seed": seed,
        "n_evals": result.n_evals, "n_feasible_pareto": len(front), "hv": hv,
        "hv_ratio_pct": 100 * hv / hv_ref,
        "igd": eo.compute_igd(front, front_ref, normalize=False) if len(front) else None,
    }
    if seed in TEST_SEEDS:
        record["front_x"] = result.X_pareto[feasible].tolist()
        record["front_f"] = front.tolist()
    return record


def summarize(rows):
    hv = [r["hv_ratio_pct"] for r in rows]
    igd = [r["igd"] for r in rows if r["igd"] is not None]
    return {
        "n": len(rows), "feasible_runs": len(igd), "mean_hv_ratio_pct": float(np.mean(hv)),
        "sd_hv_ratio_pct": float(np.std(hv, ddof=1)), "mean_igd_feasible": float(np.mean(igd)) if igd else None,
    }


def paired_difference(sa, other):
    first = {r["seed"]: r["hv_ratio_pct"] for r in sa}
    second = {r["seed"]: r["hv_ratio_pct"] for r in other}
    if first.keys() != second.keys() or len(first) != len(sa) or len(second) != len(other):
        raise ValueError("Unpaired or duplicate test seeds")
    diffs = np.array([first[s] - second[s] for s in sorted(first)])
    rng = np.random.default_rng(20261119)
    sample = rng.choice(diffs, (10000, len(diffs)), replace=True).mean(axis=1)
    return {
        "mean_delta_pp": float(np.mean(diffs)),
        "paired_percentile95_pp": np.quantile(sample, [0.025, 0.975]).tolist(),
        "sa_wins": int(np.count_nonzero(diffs > 0)),
        "ties": int(np.count_nonzero(diffs == 0)),
    }


def limit_threads():
    threadpool_limits(limits=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    grid = {a: candidates(a) for a in BASELINES}
    tuning_jobs = [(p, a, cfg, seed) for p in PROBLEMS for a in BASELINES for cfg in grid[a] for seed in TUNING_SEEDS]
    selections = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=limit_threads) as pool:
        tuning = list(pool.map(run, tuning_jobs, chunksize=4))
        for problem in PROBLEMS:
            selections[problem] = {}
            for algorithm in BASELINES:
                sub = [r for r in tuning if r["problem"] == problem and r["algorithm"] == algorithm]
                scores = [summarize([r for r in sub if r["parameters"] == cfg]) for cfg in grid[algorithm]]
                # First maximum in the declared grid resolves ties, including ties with the default.
                ix = max(range(len(scores)), key=lambda i: scores[i]["mean_hv_ratio_pct"])
                selections[problem][algorithm] = {"parameters": grid[algorithm][ix], "tuning_scores_in_grid_order": scores}
        print(f"Tuning complete: {len(tuning)} runs. Testing frozen configurations...", flush=True)
        test_jobs = []
        labels = []
        for problem in PROBLEMS:
            for algorithm in BASELINES:
                for label, cfg in [("default", grid[algorithm][0]), ("tuned", selections[problem][algorithm]["parameters"])]:
                    test_jobs.extend((problem, algorithm, cfg, seed) for seed in TEST_SEEDS)
                    labels.extend([label] * len(TEST_SEEDS))
            test_jobs.extend((problem, "SA-NSGA-II", SA_CONFIG, seed) for seed in TEST_SEEDS)
            labels.extend(["fixed"] * len(TEST_SEEDS))
        test = list(pool.map(run, test_jobs, chunksize=1))

    for row, label in zip(test, labels):
        row["configuration"] = label
    summaries = {}
    for problem in PROBLEMS:
        groups = {}
        for row in test:
            if row["problem"] == problem:
                groups.setdefault(row["algorithm"] + " " + row["configuration"], []).append(row)
        summaries[problem] = {}
        for name, rows in groups.items():
            summary = summarize(rows)
            if name.endswith("tuned"):
                summary["sa_minus_this"] = paired_difference(groups["SA-NSGA-II fixed"], rows)
            summaries[problem][name] = summary
    payload = {
        "budget_per_run": BUDGET, "tuning_seeds": TUNING_SEEDS, "test_seeds": TEST_SEEDS,
        "candidate_grid": grid, "fixed_surrogate": SA_CONFIG, "selection": "Maximum mean feasible-front HV ratio by problem and algorithm; empty feasible front receives zero HV; first candidate wins exact ties.",
        "reference_hv": {p: reference(p)[1] for p in PROBLEMS},
        "evaluation_costs": {"tuning": BUDGET * len(tuning), "test": BUDGET * len(test)},
        "selections": selections, "summaries": summaries, "tuning_runs": tuning, "test_runs": test,
    }
    root = ROOT / "results"
    root.mkdir(exist_ok=True)
    path = root / "small_budget_tuning.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
