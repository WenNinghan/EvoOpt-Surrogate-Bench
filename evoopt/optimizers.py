"""
Evolutionary Single-Objective and Multi-Objective Optimization Algorithms.

Implements:
  1. HEOA: Human Evolutionary Optimization Algorithm (Logistic chaotic initialization,
     Lévy-flight Human Exploration phase, and 4-role Human Development phase:
     Leaders, Explorers, Followers, and Losers).
  2. DifferentialEvolution (DE): Classic DE/rand/1/bin & adaptive pbest mutation.
  3. ParticleSwarmOptimization (PSO): Inertia-annealed constrained swarm search.
  4. NSGA2: Constrained Non-Dominated Sorting Genetic Algorithm II (SBX + PM).
  5. MOEAD: Multi-Objective Evolutionary Algorithm Based on Decomposition (Tchebycheff).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gamma
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from evoopt.constraints import (
    ConstraintHandler,
    DebFeasibilityRules,
    compute_cv,
    get_constraint_handler,
)
from evoopt.metrics import compute_hypervolume, is_pareto_efficient
from evoopt.problems import BenchmarkProblem, MultiObjectiveProblem


@dataclass
class OptimizationResult:
    """Result container for single-objective constrained optimization."""

    algorithm: str
    x_best: np.ndarray
    f_best: float
    g_best: np.ndarray
    cv_best: float
    is_feasible: bool
    n_evals: int
    history_evals: np.ndarray
    history_f: np.ndarray
    history_cv: np.ndarray
    evaluated_X: np.ndarray
    evaluated_f: np.ndarray
    evaluated_G: np.ndarray
    n_init: int = 0
    population_snapshots: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class MultiObjectiveResult:
    """Result container for multi-objective (constrained) optimization."""

    algorithm: str
    X_pareto: np.ndarray
    F_pareto: np.ndarray
    G_pareto: np.ndarray
    X_pop: np.ndarray
    F_pop: np.ndarray
    G_pop: np.ndarray
    n_evals: int
    history_evals: np.ndarray
    history_hv: np.ndarray
    evaluated_X: np.ndarray
    evaluated_F: np.ndarray
    evaluated_G: np.ndarray
    n_init: int = 0


def _logistic_chaotic_init(
    pop_size: int,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    mu: float = 4.0,
) -> np.ndarray:
    """Generate chaotic initial population via Logistic map z_{k+1} = mu * z_k * (1 - z_k)."""
    dim = len(lb)
    z = rng.uniform(0.05, 0.95, size=(dim,))
    # Avoid unstable fixed / periodic points {0.25, 0.50, 0.75}
    for fixed in (0.25, 0.50, 0.75):
        mask = np.abs(z - fixed) < 1e-3
        z[mask] += 0.03

    # Burn-in iterations for ergodic chaotic mixing
    for _ in range(15):
        z = mu * z * (1.0 - z)

    Z = np.empty((pop_size, dim), dtype=np.float64)
    for i in range(pop_size):
        z = mu * z * (1.0 - z)
        Z[i] = z
    return lb + Z * (ub - lb)


def _levy_flight(
    shape: Tuple[int, ...],
    rng: np.random.Generator,
    beta: float = 1.5,
) -> np.ndarray:
    """Sample Mantegna's heavy-tailed Lévy flight steps."""
    num = gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=shape)
    v = rng.normal(0.0, 1.0, size=shape)
    step = u / (np.abs(v) ** (1.0 / beta) + 1e-12)
    return np.clip(step, -5.0, 5.0)


def _repair_bounds(
    X_cand: np.ndarray,
    X_ref: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Midpoint bounce-back boundary repair preserving interior search feasibility."""
    repaired = np.where(X_cand < lb, 0.5 * (X_ref + lb), X_cand)
    repaired = np.where(repaired > ub, 0.5 * (X_ref + ub), repaired)
    return np.clip(repaired, lb, ub)


def _select_best_archive_solution(
    X_all: np.ndarray,
    f_all: np.ndarray,
    G_all: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray, float, bool]:
    """Select global best solution from evaluated archive under strict feasibility rules."""
    cv_all = compute_cv(G_all)
    feas_mask = cv_all <= 0.0
    if np.any(feas_mask):
        feas_indices = np.where(feas_mask)[0]
        best_idx = feas_indices[int(np.argmin(f_all[feas_indices]))]
        return (
            X_all[best_idx].copy(),
            float(f_all[best_idx]),
            G_all[best_idx].copy(),
            0.0,
            True,
        )
    best_idx = int(np.lexsort((f_all, cv_all))[0])
    return (
        X_all[best_idx].copy(),
        float(f_all[best_idx]),
        G_all[best_idx].copy(),
        float(cv_all[best_idx]),
        False,
    )


class HEOA:
    """Human Evolutionary Optimization Algorithm (HEOA).

    Combines Logistic chaotic initialization, Lévy-driven Human Exploration,
    and four-role Human Development (Leaders, Explorers, Followers, Losers)
    with pluggable constraint-handling rules.
    """

    def __init__(
        self,
        pop_size: int = 50,
        max_evals: int = 2000,
        exploration_ratio: float = 0.35,
        leader_ratio: float = 0.20,
        explorer_ratio: float = 0.30,
        follower_ratio: float = 0.30,
        constraint_handler: Union[str, ConstraintHandler] = "deb",
        seed: Optional[int] = None,
    ) -> None:
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evals)
        self.exploration_ratio = float(exploration_ratio)
        self.leader_ratio = float(leader_ratio)
        self.explorer_ratio = float(explorer_ratio)
        self.follower_ratio = float(follower_ratio)
        if isinstance(constraint_handler, str):
            self.handler = get_constraint_handler(constraint_handler)
        else:
            self.handler = constraint_handler
        self.seed = seed

    def optimize(self, problem: BenchmarkProblem) -> OptimizationResult:
        rng = np.random.default_rng(self.seed)
        lb, ub = problem.lb, problem.ub
        dim = problem.dim
        span = ub - lb

        n_init = min(self.pop_size, self.max_evals)
        X = _logistic_chaotic_init(n_init, lb, ub, rng)
        f, G = problem.evaluate(X)
        n_evals = n_init

        self.handler.initialize(f, G)
        order = self.handler.rank_population(f, G, progress=0.0)
        X, f, G = X[order], f[order], G[order]

        eval_X_list = [X.copy()]
        eval_f_list = [f.copy()]
        eval_G_list = [G.copy()]

        _, f_best_cur, _, cv_best_cur, is_feas_cur = _select_best_archive_solution(X, f, G)
        history_evals = [n_evals]
        history_f = [f_best_cur if is_feas_cur else float(np.min(f) + 1e3 * cv_best_cur)]
        history_cv = [cv_best_cur]

        snapshots: Dict[int, np.ndarray] = {0: X.copy()}
        total_gens = max(1, (self.max_evals - n_init) // n_init)
        snap_gens = {
            max(1, total_gens // 4),
            max(1, total_gens // 2),
            total_gens,
        }

        gen = 0
        while n_evals < self.max_evals:
            gen += 1
            batch = min(n_init, self.max_evals - n_evals)
            progress = min(1.0, n_evals / max(1, self.max_evals))

            order = self.handler.rank_population(f, G, progress=progress)
            X, f, G = X[order], f[order], G[order]
            x_best = X[0]
            x_mean = np.mean(X, axis=0)

            X_new = np.empty((batch, dim), dtype=np.float64)
            levy_steps = _levy_flight((batch, dim), rng)

            if progress <= self.exploration_ratio:
                # Stage 1: Human Exploration Phase
                decay = 1.0 - progress / max(1e-6, self.exploration_ratio)
                jump_scale = 0.25 * (0.4 + 0.6 * decay)
                for i in range(batch):
                    r1, r2 = rng.choice(n_init, size=2, replace=False)
                    if rng.random() < 0.55:
                        beta_step = rng.uniform(0.2, 0.9) * (1.0 - 0.7 * progress)
                        cand = (
                            x_best
                            + beta_step * (X[i] - x_best)
                            + jump_scale * levy_steps[i] * (X[r1] - x_mean)
                        )
                    else:
                        F_exp = rng.uniform(0.4, 0.85)
                        cand = X[i] + F_exp * (x_best - X[i]) + F_exp * (X[r1] - X[r2])
                        mask = rng.random(dim) < 0.85
                        cand = np.where(mask, cand, X[i])
                    X_new[i] = cand
            else:
                # Stage 2: Human Development Phase (Leaders, Explorers, Followers, Losers)
                n_leaders = max(2, int(round(self.leader_ratio * n_init)))
                n_explorers = max(1, int(round(self.explorer_ratio * n_init)))
                n_followers = max(1, int(round(self.follower_ratio * n_init)))
                anneal = max(1e-4, (1.0 - progress) ** 1.6)

                for i in range(batch):
                    r1, r2 = rng.choice(n_init, size=2, replace=False)
                    if i < n_leaders:
                        # Leaders: local exploitation & active-constraint differential refinement
                        if rng.random() < 0.65:
                            p_lead = rng.integers(0, n_leaders)
                            F_loc = rng.uniform(0.35, 0.75)
                            cand = X[i] + F_loc * (X[p_lead] - X[i]) + F_loc * (X[r1] - X[r2])
                            mask = rng.random(dim) < 0.8
                            cand = np.where(mask, cand, X[i])
                        else:
                            sigma_loc = 0.05 * anneal * span + 0.25 * np.abs(x_best - X[r1])
                            cand = x_best + rng.normal(0.0, 1.0, size=dim) * sigma_loc
                    elif i < n_leaders + n_explorers:
                        # Explorers: Lévy-assisted boundary traversal
                        cand = (
                            x_best
                            + rng.normal(0.0, 0.35, size=dim) * (x_mean - X[i])
                            + 0.08 * anneal * levy_steps[i] * (X[r1] - X[r2])
                        )
                    elif i < n_leaders + n_explorers + n_followers:
                        # Followers: guided social learning toward Leaders
                        lead_idx = int(rng.integers(0, n_leaders))
                        rd1 = rng.uniform(0.3, 1.0, size=dim)
                        rd2 = rng.normal(0.0, 0.35 * (1.0 - progress), size=dim)
                        cand = X[i] + rd1 * (X[lead_idx] - X[i]) + rd2 * (x_best - X[r1])
                    else:
                        # Losers: replacement around elite feasible subspace
                        lead_idx = int(rng.integers(0, n_leaders))
                        cand = x_best + rng.normal(0.0, 0.5, size=dim) * (
                            X[lead_idx] - X[r1]
                        ) + 0.02 * anneal * rng.normal(0.0, 1.0, size=dim) * span
                    X_new[i] = cand

            X_new = _repair_bounds(X_new, X[:batch], lb, ub)
            f_new, G_new = problem.evaluate(X_new)
            n_evals += batch

            eval_X_list.append(X_new.copy())
            eval_f_list.append(f_new.copy())
            eval_G_list.append(G_new.copy())

            for i in range(batch):
                if self.handler.is_better(f_new[i], G_new[i], f[i], G[i], progress=progress):
                    X[i] = X_new[i]
                    f[i] = f_new[i]
                    G[i] = G_new[i]

            if gen in snap_gens:
                snapshots[gen] = X.copy()

            X_arch = np.vstack(eval_X_list)
            f_arch = np.concatenate(eval_f_list)
            G_arch = np.vstack(eval_G_list)
            _, f_b, _, cv_b, is_feas = _select_best_archive_solution(X_arch, f_arch, G_arch)
            history_evals.append(n_evals)
            history_f.append(f_b if is_feas else float(np.min(f_arch) + 1e3 * cv_b))
            history_cv.append(cv_b)

        snapshots[gen] = X.copy()
        X_all = np.vstack(eval_X_list)
        f_all = np.concatenate(eval_f_list)
        G_all = np.vstack(eval_G_list)
        x_best, f_best, g_best, cv_best, is_feas = _select_best_archive_solution(
            X_all, f_all, G_all
        )

        return OptimizationResult(
            algorithm="HEOA",
            x_best=x_best,
            f_best=f_best,
            g_best=g_best,
            cv_best=cv_best,
            is_feasible=is_feas,
            n_evals=n_evals,
            history_evals=np.asarray(history_evals, dtype=np.int64),
            history_f=np.asarray(history_f, dtype=np.float64),
            history_cv=np.asarray(history_cv, dtype=np.float64),
            evaluated_X=X_all,
            evaluated_f=f_all,
            evaluated_G=G_all,
            n_init=n_init,
            population_snapshots=snapshots,
        )


class DifferentialEvolution:
    """Constrained Differential Evolution (DE/rand/1/bin with SHADE-style F/CR jitter)."""

    def __init__(
        self,
        pop_size: int = 50,
        max_evals: int = 2000,
        F: float = 0.65,
        CR: float = 0.85,
        adaptive: bool = True,
        constraint_handler: Union[str, ConstraintHandler] = "deb",
        seed: Optional[int] = None,
    ) -> None:
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evals)
        self.F = float(F)
        self.CR = float(CR)
        self.adaptive = bool(adaptive)
        if isinstance(constraint_handler, str):
            self.handler = get_constraint_handler(constraint_handler)
        else:
            self.handler = constraint_handler
        self.seed = seed

    def optimize(self, problem: BenchmarkProblem) -> OptimizationResult:
        rng = np.random.default_rng(self.seed)
        lb, ub = problem.lb, problem.ub
        dim = problem.dim

        n_init = min(self.pop_size, self.max_evals)
        X = rng.uniform(lb, ub, size=(n_init, dim))
        f, G = problem.evaluate(X)
        n_evals = n_init
        self.handler.initialize(f, G)

        eval_X_list = [X.copy()]
        eval_f_list = [f.copy()]
        eval_G_list = [G.copy()]

        _, f_b, _, cv_b, is_feas = _select_best_archive_solution(X, f, G)
        history_evals = [n_evals]
        history_f = [f_b if is_feas else float(np.min(f) + 1e3 * cv_b)]
        history_cv = [cv_b]

        mu_F = self.F
        mu_CR = self.CR

        while n_evals < self.max_evals:
            batch = min(n_init, self.max_evals - n_evals)
            progress = min(1.0, n_evals / max(1, self.max_evals))
            ranked = self.handler.rank_population(f, G, progress=progress)
            p_top = max(2, int(0.2 * n_init))

            U = np.empty((batch, dim), dtype=np.float64)
            F_vals = np.empty(batch, dtype=np.float64)
            CR_vals = np.empty(batch, dtype=np.float64)

            for i in range(batch):
                if self.adaptive:
                    Fi = float(np.clip(rng.normal(mu_F, 0.1), 0.2, 1.0))
                    CRi = float(np.clip(rng.normal(mu_CR, 0.1), 0.1, 1.0))
                else:
                    Fi, CRi = self.F, self.CR
                F_vals[i] = Fi
                CR_vals[i] = CRi

                candidates = [idx for idx in range(n_init) if idx != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                if self.adaptive and rng.random() < 0.5:
                    pbest_idx = int(ranked[rng.integers(0, p_top)])
                    mutant = X[i] + Fi * (X[pbest_idx] - X[i]) + Fi * (X[r1] - X[r2])
                else:
                    mutant = X[r1] + Fi * (X[r2] - X[r3])

                cross = rng.random(dim) < CRi
                cross[rng.integers(0, dim)] = True
                U[i] = np.where(cross, mutant, X[i])

            U = _repair_bounds(U, X[:batch], lb, ub)
            f_u, G_u = problem.evaluate(U)
            n_evals += batch

            eval_X_list.append(U.copy())
            eval_f_list.append(f_u.copy())
            eval_G_list.append(G_u.copy())

            succ_F: List[float] = []
            succ_CR: List[float] = []
            for i in range(batch):
                if self.handler.is_better(f_u[i], G_u[i], f[i], G[i], progress=progress):
                    X[i] = U[i]
                    f[i] = f_u[i]
                    G[i] = G_u[i]
                    succ_F.append(F_vals[i])
                    succ_CR.append(CR_vals[i])

            if self.adaptive and succ_F:
                mu_F = 0.85 * mu_F + 0.15 * float(np.mean(succ_F))
                mu_CR = 0.85 * mu_CR + 0.15 * float(np.mean(succ_CR))

            X_arch = np.vstack(eval_X_list)
            f_arch = np.concatenate(eval_f_list)
            G_arch = np.vstack(eval_G_list)
            _, f_b, _, cv_b, is_feas = _select_best_archive_solution(X_arch, f_arch, G_arch)
            history_evals.append(n_evals)
            history_f.append(f_b if is_feas else float(np.min(f_arch) + 1e3 * cv_b))
            history_cv.append(cv_b)

        X_all = np.vstack(eval_X_list)
        f_all = np.concatenate(eval_f_list)
        G_all = np.vstack(eval_G_list)
        x_best, f_best, g_best, cv_best, is_feas = _select_best_archive_solution(
            X_all, f_all, G_all
        )

        return OptimizationResult(
            algorithm="DE",
            x_best=x_best,
            f_best=f_best,
            g_best=g_best,
            cv_best=cv_best,
            is_feasible=is_feas,
            n_evals=n_evals,
            history_evals=np.asarray(history_evals, dtype=np.int64),
            history_f=np.asarray(history_f, dtype=np.float64),
            history_cv=np.asarray(history_cv, dtype=np.float64),
            evaluated_X=X_all,
            evaluated_f=f_all,
            evaluated_G=G_all,
            n_init=n_init,
        )


class ParticleSwarmOptimization:
    """Constrained Particle Swarm Optimization (PSO) with inertia decay."""

    def __init__(
        self,
        pop_size: int = 50,
        max_evals: int = 2000,
        w_max: float = 0.90,
        w_min: float = 0.40,
        c1: float = 1.70,
        c2: float = 1.70,
        constraint_handler: Union[str, ConstraintHandler] = "deb",
        seed: Optional[int] = None,
    ) -> None:
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evals)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        if isinstance(constraint_handler, str):
            self.handler = get_constraint_handler(constraint_handler)
        else:
            self.handler = constraint_handler
        self.seed = seed

    def optimize(self, problem: BenchmarkProblem) -> OptimizationResult:
        rng = np.random.default_rng(self.seed)
        lb, ub = problem.lb, problem.ub
        dim = problem.dim
        v_max = 0.20 * (ub - lb)

        n_init = min(self.pop_size, self.max_evals)
        X = rng.uniform(lb, ub, size=(n_init, dim))
        V = rng.uniform(-v_max, v_max, size=(n_init, dim))
        f, G = problem.evaluate(X)
        n_evals = n_init
        self.handler.initialize(f, G)

        pbest_X = X.copy()
        pbest_f = f.copy()
        pbest_G = G.copy()

        ranked = self.handler.rank_population(pbest_f, pbest_G, progress=0.0)
        gbest_idx = int(ranked[0])
        gbest_X = pbest_X[gbest_idx].copy()
        gbest_f = float(pbest_f[gbest_idx])
        gbest_G = pbest_G[gbest_idx].copy()

        eval_X_list = [X.copy()]
        eval_f_list = [f.copy()]
        eval_G_list = [G.copy()]

        _, f_b, _, cv_b, is_feas = _select_best_archive_solution(X, f, G)
        history_evals = [n_evals]
        history_f = [f_b if is_feas else float(np.min(f) + 1e3 * cv_b)]
        history_cv = [cv_b]

        while n_evals < self.max_evals:
            batch = min(n_init, self.max_evals - n_evals)
            progress = min(1.0, n_evals / max(1, self.max_evals))
            w = self.w_max - (self.w_max - self.w_min) * progress

            r1 = rng.random(size=(batch, dim))
            r2 = rng.random(size=(batch, dim))
            V[:batch] = (
                w * V[:batch]
                + self.c1 * r1 * (pbest_X[:batch] - X[:batch])
                + self.c2 * r2 * (gbest_X - X[:batch])
            )
            V[:batch] = np.clip(V[:batch], -v_max, v_max)
            X_next = X[:batch] + V[:batch]

            # Reflect velocity when hitting box bounds
            hit_low = X_next < lb
            hit_high = X_next > ub
            V[:batch] = np.where(hit_low | hit_high, -0.5 * V[:batch], V[:batch])
            X[:batch] = np.clip(X_next, lb, ub)

            f_cur, G_cur = problem.evaluate(X[:batch])
            n_evals += batch

            eval_X_list.append(X[:batch].copy())
            eval_f_list.append(f_cur.copy())
            eval_G_list.append(G_cur.copy())

            for i in range(batch):
                if self.handler.is_better(
                    f_cur[i], G_cur[i], pbest_f[i], pbest_G[i], progress=progress
                ):
                    pbest_X[i] = X[i].copy()
                    pbest_f[i] = f_cur[i]
                    pbest_G[i] = G_cur[i].copy()
                    if self.handler.is_better(
                        f_cur[i], G_cur[i], gbest_f, gbest_G, progress=progress
                    ):
                        gbest_X = X[i].copy()
                        gbest_f = float(f_cur[i])
                        gbest_G = G_cur[i].copy()

            X_arch = np.vstack(eval_X_list)
            f_arch = np.concatenate(eval_f_list)
            G_arch = np.vstack(eval_G_list)
            _, f_b, _, cv_b, is_feas = _select_best_archive_solution(X_arch, f_arch, G_arch)
            history_evals.append(n_evals)
            history_f.append(f_b if is_feas else float(np.min(f_arch) + 1e3 * cv_b))
            history_cv.append(cv_b)

        X_all = np.vstack(eval_X_list)
        f_all = np.concatenate(eval_f_list)
        G_all = np.vstack(eval_G_list)
        x_best, f_best, g_best, cv_best, is_feas = _select_best_archive_solution(
            X_all, f_all, G_all
        )

        return OptimizationResult(
            algorithm="PSO",
            x_best=x_best,
            f_best=f_best,
            g_best=g_best,
            cv_best=cv_best,
            is_feasible=is_feas,
            n_evals=n_evals,
            history_evals=np.asarray(history_evals, dtype=np.int64),
            history_f=np.asarray(history_f, dtype=np.float64),
            history_cv=np.asarray(history_cv, dtype=np.float64),
            evaluated_X=X_all,
            evaluated_f=f_all,
            evaluated_G=G_all,
            n_init=n_init,
        )


# ==============================================================================
# Multi-Objective Operators (Constrained Non-Dominated Sorting, SBX, PM)
# ==============================================================================


def _constrained_dominates(
    f_a: np.ndarray,
    cv_a: float,
    f_b: np.ndarray,
    cv_b: float,
) -> bool:
    """Deb's Constrained Pareto-Dominance relation between solutions a and b."""
    feas_a = cv_a <= 0.0
    feas_b = cv_b <= 0.0
    if feas_a and feas_b:
        return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))
    if feas_a and not feas_b:
        return True
    if not feas_a and feas_b:
        return False
    return cv_a < cv_b


def fast_non_dominated_sort(
    F: np.ndarray,
    G: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """Vectorized Fast Constrained Non-Dominated Sorting (Deb et al., IEEE TEVC 2002)."""
    F_arr = np.asarray(F, dtype=np.float64)
    n_pop = len(F_arr)
    if n_pop == 0:
        return []
    cv = np.zeros(n_pop, dtype=np.float64) if G is None else compute_cv(G)
    feas = cv <= 0.0

    # dom[p, q] is True iff individual p constrained-dominates individual q
    f_leq = np.all(F_arr[:, np.newaxis, :] <= F_arr[np.newaxis, :, :], axis=-1)
    f_lt = np.any(F_arr[:, np.newaxis, :] < F_arr[np.newaxis, :, :], axis=-1)
    dom_both_feas = feas[:, np.newaxis] & feas[np.newaxis, :] & f_leq & f_lt
    dom_feas_infeas = feas[:, np.newaxis] & (~feas[np.newaxis, :])
    dom_both_infeas = (
        (~feas[:, np.newaxis])
        & (~feas[np.newaxis, :])
        & (cv[:, np.newaxis] < cv[np.newaxis, :])
    )
    dom = dom_both_feas | dom_feas_infeas | dom_both_infeas
    np.fill_diagonal(dom, False)

    domination_counts = np.sum(dom, axis=0)
    remaining = np.ones(n_pop, dtype=bool)
    fronts: List[List[int]] = []

    while np.any(remaining):
        front_idx = np.where(remaining & (domination_counts == 0))[0]
        if len(front_idx) == 0:
            # Numerical safeguard against degenerate cycles
            front_idx = np.where(remaining)[0][:1]
        fronts.append(front_idx.tolist())
        remaining[front_idx] = False
        domination_counts = domination_counts - np.sum(dom[front_idx], axis=0)

    return fronts


def calculate_crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
    """Compute NSGA-II crowding distance for indices in `front`."""
    k = len(front)
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    if k <= 2:
        return np.full(k, np.inf, dtype=np.float64)

    F_sub = F[front]
    n_obj = F_sub.shape[1]
    distances = np.zeros(k, dtype=np.float64)

    for m in range(n_obj):
        order = np.argsort(F_sub[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        f_span = float(F_sub[order[-1], m] - F_sub[order[0], m])
        if f_span > 1e-12:
            distances[order[1:-1]] += (
                F_sub[order[2:], m] - F_sub[order[:-2], m]
            ) / f_span

    return distances


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    eta_c: float = 20.0,
    prob_c: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX) respecting box bounds."""
    if rng.random() > prob_c:
        return p1.copy(), p2.copy()

    dim = len(lb)
    c1 = p1.copy()
    c2 = p2.copy()
    for j in range(dim):
        if rng.random() <= 0.5 and abs(p1[j] - p2[j]) > 1e-14:
            y1 = min(p1[j], p2[j])
            y2 = max(p1[j], p2[j])
            yl, yu = lb[j], ub[j]
            rand_u = rng.random()

            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand_u <= (1.0 / alpha):
                betaq = (rand_u * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand_u * alpha)) ** (1.0 / (eta_c + 1.0))
            ch1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand_u <= (1.0 / alpha):
                betaq = (rand_u * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand_u * alpha)) ** (1.0 / (eta_c + 1.0))
            ch2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            ch1 = float(np.clip(ch1, yl, yu))
            ch2 = float(np.clip(ch2, yl, yu))
            if rng.random() < 0.5:
                c1[j], c2[j] = ch2, ch1
            else:
                c1[j], c2[j] = ch1, ch2
    return c1, c2


def polynomial_mutation(
    x: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    eta_m: float = 20.0,
    prob_m: Optional[float] = None,
) -> np.ndarray:
    """Polynomial Mutation (PM) respecting box bounds."""
    dim = len(lb)
    pm = (1.0 / dim) if prob_m is None else prob_m
    y = x.copy()
    for j in range(dim):
        if rng.random() <= pm:
            yl, yu = lb[j], ub[j]
            span = yu - yl
            if span <= 1e-14:
                continue
            delta1 = (y[j] - yl) / span
            delta2 = (yu - y[j]) / span
            rnd = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1.0))
                deltaq = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - (val ** mut_pow)
            y[j] = float(np.clip(y[j] + deltaq * span, yl, yu))
    return y


def _extract_feasible_pareto(
    X_all: np.ndarray,
    F_all: np.ndarray,
    G_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract feasible non-dominated Pareto set from archive."""
    cv = compute_cv(G_all)
    feas = cv <= 0.0
    if np.any(feas):
        Xf, Ff, Gf = X_all[feas], F_all[feas], G_all[feas]
        nd = is_pareto_efficient(Ff)
        Xp, Fp, Gp = Xf[nd], Ff[nd], Gf[nd]
        order = np.argsort(Fp[:, 0])
        return Xp[order], Fp[order], Gp[order]

    # Fallback if no feasible solution found yet: least-violating point
    best_idx = int(np.argmin(cv))
    return (
        X_all[[best_idx]].copy(),
        F_all[[best_idx]].copy(),
        G_all[[best_idx]].copy(),
    )


class NSGA2:
    """Constrained Elitist Non-Dominated Sorting Genetic Algorithm II (NSGA-II)."""

    def __init__(
        self,
        pop_size: int = 80,
        max_evals: int = 3200,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        crossover_prob: float = 0.90,
        seed: Optional[int] = None,
    ) -> None:
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evals)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.crossover_prob = float(crossover_prob)
        self.seed = seed

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        track_hv: bool = True,
    ) -> MultiObjectiveResult:
        rng = np.random.default_rng(self.seed)
        lb, ub = problem.lb, problem.ub
        dim = problem.dim

        n_pop = min(self.pop_size, self.max_evals)
        X = rng.uniform(lb, ub, size=(n_pop, dim))
        F, G = problem.evaluate(X)
        n_evals = n_pop

        eval_X_list = [X.copy()]
        eval_F_list = [F.copy()]
        eval_G_list = [G.copy()]

        history_evals = [n_evals]
        if track_hv:
            _, F_p0, _ = _extract_feasible_pareto(X, F, G)
            cv0 = compute_cv(G)
            hv0 = compute_hypervolume(F_p0, problem.ref_point) if np.any(cv0 <= 0.0) else 0.0
            history_hv = [hv0]
        else:
            history_hv = [0.0]

        while n_evals < self.max_evals:
            n_off = min(n_pop, self.max_evals - n_evals)

            # Assign rank & crowding distance to current population for tournament mating
            fronts = fast_non_dominated_sort(F, G)
            rank_arr = np.empty(n_pop, dtype=np.int64)
            crowd_arr = np.zeros(n_pop, dtype=np.float64)
            for r_idx, front in enumerate(fronts):
                rank_arr[front] = r_idx
                crowd_arr[front] = calculate_crowding_distance(F, front)

            offspring_list: List[np.ndarray] = []
            while len(offspring_list) < n_off:
                # Binary crowded tournament selection
                parents: List[np.ndarray] = []
                for _ in range(2):
                    i1, i2 = rng.integers(0, n_pop, size=2)
                    if rank_arr[i1] < rank_arr[i2]:
                        winner = i1
                    elif rank_arr[i2] < rank_arr[i1]:
                        winner = i2
                    else:
                        winner = i1 if crowd_arr[i1] >= crowd_arr[i2] else i2
                    parents.append(X[winner])

                c1, c2 = sbx_crossover(
                    parents[0],
                    parents[1],
                    lb,
                    ub,
                    rng,
                    eta_c=self.eta_c,
                    prob_c=self.crossover_prob,
                )
                c1 = polynomial_mutation(c1, lb, ub, rng, eta_m=self.eta_m)
                offspring_list.append(c1)
                if len(offspring_list) < n_off:
                    c2 = polynomial_mutation(c2, lb, ub, rng, eta_m=self.eta_m)
                    offspring_list.append(c2)

            X_off = np.asarray(offspring_list[:n_off], dtype=np.float64)
            F_off, G_off = problem.evaluate(X_off)
            n_evals += n_off

            eval_X_list.append(X_off.copy())
            eval_F_list.append(F_off.copy())
            eval_G_list.append(G_off.copy())

            # (mu + lambda) elitist environmental survival
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            G_comb = np.vstack([G, G_off])

            comb_fronts = fast_non_dominated_sort(F_comb, G_comb)
            next_indices: List[int] = []
            for front in comb_fronts:
                if len(next_indices) + len(front) <= n_pop:
                    next_indices.extend(front)
                else:
                    rem = n_pop - len(next_indices)
                    cd = calculate_crowding_distance(F_comb, front)
                    top_cd = np.argsort(-cd)[:rem]
                    next_indices.extend([front[k] for k in top_cd])
                    break

            X = X_comb[next_indices]
            F = F_comb[next_indices]
            G = G_comb[next_indices]

            history_evals.append(n_evals)
            if track_hv:
                X_arch = np.vstack(eval_X_list)
                F_arch = np.vstack(eval_F_list)
                G_arch = np.vstack(eval_G_list)
                cv_arch = compute_cv(G_arch)
                if np.any(cv_arch <= 0.0):
                    _, F_pf, _ = _extract_feasible_pareto(X_arch, F_arch, G_arch)
                    history_hv.append(compute_hypervolume(F_pf, problem.ref_point))
                else:
                    history_hv.append(0.0)
            else:
                history_hv.append(0.0)

        X_all = np.vstack(eval_X_list)
        F_all = np.vstack(eval_F_list)
        G_all = np.vstack(eval_G_list)
        X_pf, F_pf, G_pf = _extract_feasible_pareto(X_all, F_all, G_all)

        return MultiObjectiveResult(
            algorithm="NSGA-II",
            X_pareto=X_pf,
            F_pareto=F_pf,
            G_pareto=G_pf,
            X_pop=X,
            F_pop=F,
            G_pop=G,
            n_evals=n_evals,
            history_evals=np.asarray(history_evals, dtype=np.int64),
            history_hv=np.asarray(history_hv, dtype=np.float64),
            evaluated_X=X_all,
            evaluated_F=F_all,
            evaluated_G=G_all,
            n_init=n_pop,
        )


class MOEAD:
    """Multi-Objective Evolutionary Algorithm Based on Decomposition (MOEA/D).

    Uses Tchebycheff scalarization with uniform weight vectors, neighborhood
    mating, and Deb's feasibility-first replacement rule (Jain & Deb, 2014).
    """

    def __init__(
        self,
        pop_size: int = 80,
        max_evals: int = 3200,
        n_neighbors: int = 15,
        max_replace: int = 2,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        seed: Optional[int] = None,
    ) -> None:
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evals)
        self.n_neighbors = int(n_neighbors)
        self.max_replace = int(max_replace)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.seed = seed

    @staticmethod
    def _generate_weights(n_weights: int, n_obj: int) -> np.ndarray:
        if n_obj == 2:
            w1 = np.linspace(0.0, 1.0, n_weights)
            W = np.column_stack([w1, 1.0 - w1])
        else:
            rng = np.random.default_rng(0)
            W = rng.dirichlet(np.ones(n_obj), size=n_weights)
        return np.maximum(W, 1e-4)

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        track_hv: bool = True,
    ) -> MultiObjectiveResult:
        rng = np.random.default_rng(self.seed)
        lb, ub = problem.lb, problem.ub
        dim = problem.dim
        n_obj = problem.n_obj

        n_pop = min(self.pop_size, self.max_evals)
        W = self._generate_weights(n_pop, n_obj)
        T = min(self.n_neighbors, n_pop)
        dist_W = np.linalg.norm(W[:, np.newaxis, :] - W[np.newaxis, :, :], axis=-1)
        B = np.argsort(dist_W, axis=1)[:, :T]

        X = rng.uniform(lb, ub, size=(n_pop, dim))
        F, G = problem.evaluate(X)
        cv = compute_cv(G)
        n_evals = n_pop

        feas = cv <= 0.0
        if np.any(feas):
            z_ideal = np.min(F[feas], axis=0)
            z_nadir = np.max(F[feas], axis=0)
        else:
            z_ideal = np.min(F, axis=0)
            z_nadir = np.max(F, axis=0)

        eval_X_list = [X.copy()]
        eval_F_list = [F.copy()]
        eval_G_list = [G.copy()]

        history_evals = [n_evals]
        if track_hv and np.any(feas):
            _, F_p0, _ = _extract_feasible_pareto(X, F, G)
            history_hv = [compute_hypervolume(F_p0, problem.ref_point)]
        else:
            history_hv = [0.0]

        while n_evals < self.max_evals:
            batch = min(n_pop, self.max_evals - n_evals)
            perm = rng.permutation(n_pop)[:batch]

            X_children = np.empty((batch, dim), dtype=np.float64)
            parent_pools: List[np.ndarray] = []

            for k, i in enumerate(perm):
                if rng.random() < 0.85:
                    pool = B[i]
                else:
                    pool = np.arange(n_pop)
                parent_pools.append(pool)
                p1_idx, p2_idx = rng.choice(pool, size=2, replace=False)
                c1, _ = sbx_crossover(
                    X[p1_idx], X[p2_idx], lb, ub, rng, eta_c=self.eta_c, prob_c=1.0
                )
                c1 = polynomial_mutation(c1, lb, ub, rng, eta_m=self.eta_m)
                X_children[k] = c1

            F_children, G_children = problem.evaluate(X_children)
            cv_children = compute_cv(G_children)
            n_evals += batch

            eval_X_list.append(X_children.copy())
            eval_F_list.append(F_children.copy())
            eval_G_list.append(G_children.copy())

            for k in range(batch):
                f_y = F_children[k]
                g_y = G_children[k]
                cv_y = float(cv_children[k])
                if cv_y <= 0.0:
                    z_ideal = np.minimum(z_ideal, f_y)
                    z_nadir = np.maximum(z_nadir, f_y)
                else:
                    z_ideal = np.minimum(z_ideal, f_y)

                scale = np.maximum(z_nadir - z_ideal, 1e-6)
                pool = rng.permutation(parent_pools[k])
                n_rep = 0
                for j in pool:
                    if n_rep >= self.max_replace:
                        break
                    cv_j = float(cv[j])
                    replace = False
                    if cv_y <= 0.0 and cv_j > 0.0:
                        replace = True
                    elif cv_y > 0.0 and cv_j > 0.0:
                        replace = cv_y < cv_j
                    elif cv_y <= 0.0 and cv_j <= 0.0:
                        tch_y = float(np.max(W[j] * np.abs(f_y - z_ideal) / scale))
                        tch_j = float(np.max(W[j] * np.abs(F[j] - z_ideal) / scale))
                        replace = tch_y <= tch_j
                    if replace:
                        X[j] = X_children[k]
                        F[j] = f_y
                        G[j] = g_y
                        cv[j] = cv_y
                        n_rep += 1

            history_evals.append(n_evals)
            if track_hv:
                X_arch = np.vstack(eval_X_list)
                F_arch = np.vstack(eval_F_list)
                G_arch = np.vstack(eval_G_list)
                cv_arch = compute_cv(G_arch)
                if np.any(cv_arch <= 0.0):
                    _, F_pf, _ = _extract_feasible_pareto(X_arch, F_arch, G_arch)
                    history_hv.append(compute_hypervolume(F_pf, problem.ref_point))
                else:
                    history_hv.append(0.0)
            else:
                history_hv.append(0.0)

        X_all = np.vstack(eval_X_list)
        F_all = np.vstack(eval_F_list)
        G_all = np.vstack(eval_G_list)
        X_pf, F_pf, G_pf = _extract_feasible_pareto(X_all, F_all, G_all)

        return MultiObjectiveResult(
            algorithm="MOEA/D",
            X_pareto=X_pf,
            F_pareto=F_pf,
            G_pareto=G_pf,
            X_pop=X,
            F_pop=F,
            G_pop=G,
            n_evals=n_evals,
            history_evals=np.asarray(history_evals, dtype=np.int64),
            history_hv=np.asarray(history_hv, dtype=np.float64),
            evaluated_X=X_all,
            evaluated_F=F_all,
            evaluated_G=G_all,
            n_init=n_pop,
        )


DE = DifferentialEvolution
PSO = ParticleSwarmOptimization
