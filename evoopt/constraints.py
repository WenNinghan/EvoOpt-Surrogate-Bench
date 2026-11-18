"""
Canonical Constraint-Handling Mechanisms for Evolutionary Optimization.

Implements:
  1. Deb's Feasibility Rules (Deb, 2000): Parameter-free tournament dominance
     comparing constraint violation CV(x) first and objective value f(x) second.
  2. Adaptive Penalty Function: Dynamic annealing penalty coefficient scaling
     with evolutionary generation t / T_max.
  3. Epsilon-Constraint Method (Takahama & Sakai, 2006): Decayed epsilon(t)
     relaxation schedule enabling smooth traversal of narrow feasible regions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


def compute_cv(G: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Compute total inequality constraint violation CV(x) = sum_j max(0, g_j(x))."""
    G_arr = np.asarray(G, dtype=np.float64)
    if G_arr.size == 0:
        return np.zeros(G_arr.shape[:-1], dtype=np.float64)
    return np.sum(np.maximum(0.0, G_arr - tol), axis=-1)


class ConstraintHandler(ABC):
    """Abstract interface for constraint-handling strategies in evolutionary search."""

    name: str = "BaseConstraintHandler"

    def initialize(self, f_init: np.ndarray, G_init: np.ndarray) -> None:
        """Optional hook called on initial population."""

    @abstractmethod
    def is_better(
        self,
        f_a: float,
        g_a: np.ndarray,
        f_b: float,
        g_b: np.ndarray,
        progress: float = 0.0,
    ) -> bool:
        """Return True if candidate A is strictly better than candidate B."""

    @abstractmethod
    def rank_population(
        self,
        f: np.ndarray,
        G: np.ndarray,
        progress: float = 0.0,
    ) -> np.ndarray:
        """Return indices sorting population from best (index 0) to worst."""

    def penalized_objective(
        self,
        f: np.ndarray,
        G: np.ndarray,
        progress: float = 0.0,
    ) -> np.ndarray:
        """Return scalarized fitness values suitable for continuous ranking."""
        cv = compute_cv(G)
        feasible = cv <= 0.0
        f_max = np.max(f[feasible]) if np.any(feasible) else np.max(f)
        return np.where(feasible, f, f_max + 1e4 * (1.0 + cv))


class DebFeasibilityRules(ConstraintHandler):
    """Deb's Parameter-Free Feasibility Rules (Deb, CMAME 2000).

    Pairwise comparison between solutions x_a and x_b:
      1. If both are feasible (CV == 0), the one with smaller f(x) wins.
      2. If one is feasible and the other is infeasible, the feasible one wins.
      3. If both are infeasible (CV > 0), the one with smaller CV(x) wins.
    """

    name = "DebFeasibilityRules"

    def __init__(self, feas_tol: float = 0.0) -> None:
        self.feas_tol = float(feas_tol)

    def is_better(
        self,
        f_a: float,
        g_a: np.ndarray,
        f_b: float,
        g_b: np.ndarray,
        progress: float = 0.0,
    ) -> bool:
        cv_a = float(compute_cv(g_a, self.feas_tol))
        cv_b = float(compute_cv(g_b, self.feas_tol))
        feas_a = cv_a <= 0.0
        feas_b = cv_b <= 0.0

        if feas_a and feas_b:
            return f_a < f_b
        if feas_a and not feas_b:
            return True
        if not feas_a and feas_b:
            return False
        if abs(cv_a - cv_b) > 1e-15:
            return cv_a < cv_b
        return f_a < f_b

    def rank_population(
        self,
        f: np.ndarray,
        G: np.ndarray,
        progress: float = 0.0,
    ) -> np.ndarray:
        cv = compute_cv(G, self.feas_tol)
        # Lexicographic sort: primary key is cv (0 for all feasible), secondary key is f
        return np.lexsort((f, cv))


class AdaptivePenaltyConstraintHandler(ConstraintHandler):
    """Dynamic Annealing Penalty Constraint Handler.

    Computes penalized fitness:
        F_pen(x, t) = f(x) + lambda(t) * (CV(x) + CV(x)^2)
    where lambda(t) = base_penalty * (1 + alpha * (t / T_max)^beta) * scale_f
    adapts automatically to objective scale and evolutionary progress.
    """

    name = "AdaptivePenalty"

    def __init__(
        self,
        base_penalty: float = 50.0,
        alpha: float = 200.0,
        beta: float = 2.0,
    ) -> None:
        self.base_penalty = float(base_penalty)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._f_scale = 1.0

    def initialize(self, f_init: np.ndarray, G_init: np.ndarray) -> None:
        span = float(np.percentile( np.abs(f_init), 75 ) - np.percentile( np.abs(f_init), 25 ))
        med = float(np.median(np.abs(f_init)))
        self._f_scale = max(med, span, 1.0)

    def _penalty_factor(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return self._f_scale * self.base_penalty * (1.0 + self.alpha * (p ** self.beta))

    def penalized_objective(
        self,
        f: np.ndarray,
        G: np.ndarray,
        progress: float = 0.0,
    ) -> np.ndarray:
        cv = compute_cv(G)
        lam = self._penalty_factor(progress)
        return f + lam * (cv + np.square(cv))

    def is_better(
        self,
        f_a: float,
        g_a: np.ndarray,
        f_b: float,
        g_b: np.ndarray,
        progress: float = 0.0,
    ) -> bool:
        cv_a = float(compute_cv(g_a))
        cv_b = float(compute_cv(g_b))
        lam = self._penalty_factor(progress)
        pen_a = f_a + lam * (cv_a + cv_a * cv_a)
        pen_b = f_b + lam * (cv_b + cv_b * cv_b)
        return pen_a < pen_b

    def rank_population(
        self,
        f: np.ndarray,
        G: np.ndarray,
        progress: float = 0.0,
    ) -> np.ndarray:
        pen = self.penalized_objective(f, G, progress)
        return np.argsort(pen)


class EpsilonConstraintHandler(ConstraintHandler):
    """Takahama & Sakai's Decayed Epsilon-Constraint Method.

    Relaxes feasibility up to threshold epsilon(t):
        epsilon(t) = epsilon_0 * (1 - t / T_c)^cp   if t / T_max < cutoff_ratio
        epsilon(t) = 0                              otherwise
    where epsilon_0 is the theta-quantile of initial population constraint violations.
    """

    name = "EpsilonConstraint"

    def __init__(
        self,
        theta_quantile: float = 0.25,
        cp: float = 5.0,
        cutoff_ratio: float = 0.65,
    ) -> None:
        self.theta_quantile = float(theta_quantile)
        self.cp = float(cp)
        self.cutoff_ratio = float(cutoff_ratio)
        self.eps_0: float = 0.1

    def initialize(self, f_init: np.ndarray, G_init: np.ndarray) -> None:
        cv = compute_cv(G_init)
        positive_cv = cv[cv > 0.0]
        if len(positive_cv) > 0:
            self.eps_0 = float(np.quantile(positive_cv, self.theta_quantile))
        else:
            self.eps_0 = 0.0

    def epsilon_level(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        if p >= self.cutoff_ratio or self.eps_0 <= 0.0:
            return 0.0
        return self.eps_0 * ((1.0 - p / self.cutoff_ratio) ** self.cp)

    def is_better(
        self,
        f_a: float,
        g_a: np.ndarray,
        f_b: float,
        g_b: np.ndarray,
        progress: float = 0.0,
    ) -> bool:
        eps = self.epsilon_level(progress)
        cv_a = float(compute_cv(g_a))
        cv_b = float(compute_cv(g_b))

        eff_cv_a = 0.0 if cv_a <= eps else cv_a
        eff_cv_b = 0.0 if cv_b <= eps else cv_b

        if eff_cv_a == 0.0 and eff_cv_b == 0.0:
            return f_a < f_b
        if eff_cv_a != eff_cv_b:
            return eff_cv_a < eff_cv_b
        return f_a < f_b

    def rank_population(
        self,
        f: np.ndarray,
        G: np.ndarray,
        progress: float = 0.0,
    ) -> np.ndarray:
        eps = self.epsilon_level(progress)
        cv = compute_cv(G)
        eff_cv = np.where(cv <= eps, 0.0, cv)
        return np.lexsort((f, eff_cv))


def get_constraint_handler(name: str) -> ConstraintHandler:
    """Factory function for constraint-handling mechanisms."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in ("deb", "deb_feasibility", "deb_feasibility_rules", "feasibility"):
        return DebFeasibilityRules()
    if key in ("penalty", "adaptive_penalty"):
        return AdaptivePenaltyConstraintHandler()
    if key in ("epsilon", "epsilon_constraint", "eps"):
        return EpsilonConstraintHandler()
    raise ValueError(f"Unknown constraint handler: '{name}'")
