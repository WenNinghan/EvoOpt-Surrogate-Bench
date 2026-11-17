"""
Surrogate-Assisted Evolutionary Algorithms (SAEA) with Kriging & Deep Ensembles.

Implements:
  1. GaussianProcessSurrogate: ARD Matérn 5/2 Kriging with input/output scaling.
  2. DeepEnsembleSurrogate: Vectorized PyTorch Deep Ensemble MLP providing
     predictive mean mu(x) and epistemic uncertainty sigma(x).
  3. Constrained Acquisition Criteria:
     - Probability of Feasibility (PoF)
     - Expected Improvement (EI) & Constrained Expected Improvement (CEI = EI * PoF)
     - Feasibility-Aware Lower Confidence Bound (LCB)
  4. SurrogateAssistedHEOA (SA-HEOA): Hybrid global CEI active learning + local
     trust-region surrogate refinement.
  5. SurrogateAssistedNSGA2 (SA-NSGA-II): Multi-objective LCB + PoF prescreened
     evolutionary Pareto search.
"""

from __future__ import annotations

import platform
import warnings
from typing import List, Literal, Optional, Tuple

if hasattr(platform, "_wmi"):
    platform._wmi = None

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc
import torch
import torch.nn as nn

from evoopt.constraints import DebFeasibilityRules, compute_cv
from evoopt.metrics import compute_hypervolume
from evoopt.optimizers import (
    HEOA,
    NSGA2,
    MultiObjectiveResult,
    OptimizationResult,
    _extract_feasible_pareto,
    _select_best_archive_solution,
)
from evoopt.problems import BenchmarkProblem, MultiObjectiveProblem


# ==============================================================================
# Constrained Acquisition / Infill Functions
# ==============================================================================


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / np.sqrt(2.0 * np.pi)


def probability_of_feasibility(
    mu_g: np.ndarray,
    sigma_g: np.ndarray,
) -> np.ndarray:
    """Compute Probability of Feasibility PoF(x) = prod_j Phi(-mu_g_j / sigma_g_j)."""
    mu_arr = np.asarray(mu_g, dtype=np.float64)
    sig_arr = np.asarray(sigma_g, dtype=np.float64)
    is_1d = mu_arr.ndim == 1
    if is_1d:
        mu_arr = mu_arr[np.newaxis, :]
        sig_arr = sig_arr[np.newaxis, :]

    if mu_arr.shape[1] == 0:
        out = np.ones(mu_arr.shape[0], dtype=np.float64)
        return float(out[0]) if is_1d else out

    safe_sig = np.maximum(sig_arr, 1e-9)
    z = -mu_arr / safe_sig
    pof = np.prod(ndtr(z), axis=-1)
    return float(pof[0]) if is_1d else pof


def log_probability_of_feasibility(
    mu_g: np.ndarray,
    sigma_g: np.ndarray,
) -> np.ndarray:
    """Numerically stable log PoF(x) for deeply infeasible regions."""
    mu_arr = np.asarray(mu_g, dtype=np.float64)
    sig_arr = np.asarray(sigma_g, dtype=np.float64)
    if mu_arr.shape[-1] == 0:
        return np.zeros(mu_arr.shape[:-1], dtype=np.float64)
    safe_sig = np.maximum(sig_arr, 1e-9)
    z = np.clip(-mu_arr / safe_sig, -35.0, 35.0)
    probs = np.maximum(ndtr(z), 1e-300)
    return np.sum(np.log(probs), axis=-1)


def expected_improvement(
    mu_f: np.ndarray,
    sigma_f: np.ndarray,
    f_best: float,
    xi: float = 1e-4,
) -> np.ndarray:
    """Compute Expected Improvement EI(x) for minimization."""
    mu_arr = np.asarray(mu_f, dtype=np.float64)
    sig_arr = np.asarray(sigma_f, dtype=np.float64)
    is_scalar = mu_arr.ndim == 0
    mu_1d = np.atleast_1d(mu_arr)
    sig_1d = np.maximum(np.atleast_1d(sig_arr), 1e-12)

    imp = f_best - mu_1d - xi
    z = imp / sig_1d
    ei = imp * ndtr(z) + sig_1d * _norm_pdf(z)
    ei = np.where(sig_arr.ravel() <= 1e-12, np.maximum(0.0, imp), ei)
    ei = np.maximum(0.0, ei)
    return float(ei[0]) if is_scalar else ei


def constrained_expected_improvement(
    mu_f: np.ndarray,
    sigma_f: np.ndarray,
    mu_g: np.ndarray,
    sigma_g: np.ndarray,
    f_best: float,
    xi: float = 1e-4,
) -> np.ndarray:
    """Compute Constrained Expected Improvement CEI(x) = EI(x) * PoF(x)."""
    ei = expected_improvement(mu_f, sigma_f, f_best=f_best, xi=xi)
    pof = probability_of_feasibility(mu_g, sigma_g)
    return ei * pof


def feasibility_aware_lcb(
    mu_f: np.ndarray,
    sigma_f: np.ndarray,
    mu_g: np.ndarray,
    sigma_g: np.ndarray,
    kappa: float = 2.0,
    beta_g: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Lower Confidence Bound LCB_f = mu_f - kappa * sigma_f and constraint margin."""
    lcb_f = np.asarray(mu_f, dtype=np.float64) - kappa * np.asarray(sigma_f, dtype=np.float64)
    eff_g = np.asarray(mu_g, dtype=np.float64) + beta_g * np.asarray(sigma_g, dtype=np.float64)
    return lcb_f, eff_g


# ==============================================================================
# Surrogate Models: Kriging (Matérn 5/2 GP) & PyTorch Deep Ensemble MLP
# ==============================================================================


class GaussianProcessSurrogate:
    """Exact Vectorized Multi-Output ARD Matérn 5/2 Kriging Surrogate (EGO Formulation).

    Computes concentrated Maximum Likelihood ARD length-scales and exact output-specific
    process variances sigma_k^2 = (1/N) y_k^T (R + eta I)^(-1) y_k via a shared Cholesky
    factorization, enabling simultaneous O(M * N) batch posterior inference across all
    objectives and nonlinear constraints.
    """

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        n_restarts: int = 0,
        nugget: float = 1e-6,
        seed: int = 42,
    ) -> None:
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.span = np.where((self.ub - self.lb) > 1e-12, self.ub - self.lb, 1.0)
        self.n_restarts = int(n_restarts)
        self.nugget = float(nugget)
        self.seed = int(seed)

        self.X_train_: np.ndarray = np.empty((0, len(self.lb)))
        self.length_scales_: np.ndarray = np.ones(len(self.lb))
        self.L_: np.ndarray = np.eye(1)
        self.alpha_: np.ndarray = np.zeros((1, 1))
        self.sigma2_proc_: np.ndarray = np.ones(1)
        self.const_mask_: np.ndarray = np.zeros(1, dtype=bool)
        self.y_mean: np.ndarray = np.zeros(1)
        self.y_std: np.ndarray = np.ones(1)

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.lb) / self.span

    @staticmethod
    def _matern52_cov(XA: np.ndarray, XB: np.ndarray, ls: np.ndarray) -> np.ndarray:
        diff = (XA[:, np.newaxis, :] - XB[np.newaxis, :, :]) / ls[np.newaxis, np.newaxis, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        sqrt5_d = np.sqrt(5.0) * dist
        return (1.0 + sqrt5_d + (5.0 / 3.0) * (dist * dist)) * np.exp(-sqrt5_d)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GaussianProcessSurrogate":
        X_scaled = self._scale_X(X)
        Y_arr = np.asarray(Y, dtype=np.float64)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr[:, np.newaxis]

        q_low = np.percentile(Y_arr, 2, axis=0)
        q_high = np.percentile(Y_arr, 98, axis=0)
        iqr = np.maximum(q_high - q_low, 1e-6)
        Y_clipped = np.clip(Y_arr, q_low - 10.0 * iqr, q_high + 10.0 * iqr)

        self.y_mean = np.mean(Y_clipped, axis=0)
        raw_std = np.std(Y_clipped, axis=0)
        self.const_mask_ = raw_std < 1e-11
        self.y_std = np.where(self.const_mask_, 1.0, raw_std)
        Y_norm = (Y_clipped - self.y_mean) / self.y_std

        n_samples, dim = X_scaled.shape
        self.X_train_ = X_scaled
        active_Y = Y_norm[:, ~self.const_mask_]
        if active_Y.shape[1] == 0:
            active_Y = np.zeros((n_samples, 1), dtype=np.float64)

        def _neg_concentrated_ll(log_ls: np.ndarray) -> float:
            ls = np.exp(log_ls)
            R = self._matern52_cov(X_scaled, X_scaled, ls)
            R[np.diag_indices_from(R)] += self.nugget
            try:
                L = cholesky(R, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e12
            V = solve_triangular(L, active_Y, lower=True, check_finite=False)
            sig2 = np.maximum(np.mean(V * V, axis=0), 1e-10)
            log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
            return 0.5 * n_samples * float(np.sum(np.log(sig2))) + 0.5 * active_Y.shape[1] * log_det

        # Coarse isotropic warm-start + fast ARD L-BFGS-B polish
        grid_candidates = [np.full(dim, np.log(v)) for v in (0.12, 0.30, 0.70, 1.50)]
        best_log_ls = min(grid_candidates, key=_neg_concentrated_ll)
        res = minimize(
            _neg_concentrated_ll,
            best_log_ls,
            method="L-BFGS-B",
            bounds=[(-3.2, 1.8)] * dim,
            options={"maxiter": 10, "ftol": 1e-4},
        )
        if np.all(np.isfinite(res.x)):
            best_log_ls = res.x

        self.length_scales_ = np.exp(best_log_ls)
        R_opt = self._matern52_cov(X_scaled, X_scaled, self.length_scales_)
        R_opt[np.diag_indices_from(R_opt)] += self.nugget
        self.L_ = cholesky(R_opt, lower=True, check_finite=False)
        V_all = solve_triangular(self.L_, Y_norm, lower=True, check_finite=False)
        self.sigma2_proc_ = np.maximum(np.mean(V_all * V_all, axis=0), 1e-9)
        self.alpha_ = solve_triangular(
            self.L_.T, V_all, lower=False, check_finite=False
        )
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) arrays of shape (N, n_targets)."""
        X_scaled = self._scale_X(X)
        if X_scaled.ndim == 1:
            X_scaled = X_scaled[np.newaxis, :]

        K_star = self._matern52_cov(X_scaled, self.X_train_, self.length_scales_)
        mu_norm = K_star @ self.alpha_

        v = solve_triangular(self.L_, K_star.T, lower=True, check_finite=False)
        mse = np.maximum(1.0 + self.nugget - np.sum(v * v, axis=0), 1e-12)
        std_norm = np.sqrt(mse[:, np.newaxis] * self.sigma2_proc_[np.newaxis, :])

        mu = mu_norm * self.y_std + self.y_mean
        std = np.where(self.const_mask_, 1e-9, np.maximum(std_norm * self.y_std, 1e-9))
        return mu, std


class _VectorizedEnsembleMLP(nn.Module):
    """Parallelized M-member MLP ensemble executed via batched tensor contractions."""

    def __init__(
        self,
        n_members: int,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_members = n_members
        self.w1 = nn.Parameter(torch.empty(n_members, in_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(n_members, 1, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(n_members, hidden_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_members, 1, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(n_members, hidden_dim, out_dim))
        self.b3 = nn.Parameter(torch.zeros(n_members, 1, out_dim))
        self.act = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in range(self.n_members):
            nn.init.kaiming_normal_(self.w1.data[m], nonlinearity="relu")
            nn.init.kaiming_normal_(self.w2.data[m], nonlinearity="relu")
            nn.init.xavier_normal_(self.w3.data[m], gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1 and self.n_members > 1:
            x = x.expand(self.n_members, -1, -1)
        h1 = self.act(torch.bmm(x, self.w1) + self.b1)
        h2 = self.act(torch.bmm(h1, self.w2) + self.b2)
        return torch.bmm(h2, self.w3) + self.b3


class DeepEnsembleSurrogate:
    """PyTorch Deep Ensemble Neural Surrogate with Epistemic Uncertainty Quantification."""

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        n_members: int = 5,
        hidden_dim: int = 64,
        epochs: int = 120,
        lr: float = 8e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.span = np.where((self.ub - self.lb) > 1e-12, self.ub - self.lb, 1.0)
        self.n_members = int(n_members)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.net: Optional[_VectorizedEnsembleMLP] = None
        self.y_mean: np.ndarray = np.zeros(1)
        self.y_std: np.ndarray = np.ones(1)

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(X, dtype=np.float64) - self.lb) / self.span - 1.0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DeepEnsembleSurrogate":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X_scaled = self._scale_X(X)
        Y_arr = np.asarray(Y, dtype=np.float64)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr[:, np.newaxis]

        q_low = np.percentile(Y_arr, 2, axis=0)
        q_high = np.percentile(Y_arr, 98, axis=0)
        iqr = np.maximum(q_high - q_low, 1e-6)
        Y_clipped = np.clip(Y_arr, q_low - 10.0 * iqr, q_high + 10.0 * iqr)

        self.y_mean = np.mean(Y_clipped, axis=0)
        raw_std = np.std(Y_clipped, axis=0)
        self.y_std = np.where(raw_std < 1e-11, 1.0, raw_std)
        Y_norm = (Y_clipped - self.y_mean) / self.y_std

        n_samples, in_dim = X_scaled.shape
        out_dim = Y_norm.shape[1]

        self.net = _VectorizedEnsembleMLP(
            n_members=self.n_members,
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.net.train()

        boot_weights = rng.dirichlet(np.ones(n_samples), size=self.n_members) * n_samples
        w_t = torch.tensor(
            boot_weights[:, :, np.newaxis], dtype=torch.float32, device=self.device
        )
        x_t = torch.tensor(
            X_scaled[np.newaxis, :, :], dtype=torch.float32, device=self.device
        )
        y_t = torch.tensor(
            Y_norm[np.newaxis, :, :], dtype=torch.float32, device=self.device
        ).expand(self.n_members, -1, -1)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.net(x_t)
            loss = torch.mean(
                w_t
                * nn.functional.smooth_l1_loss(
                    preds, y_t, reduction="none", beta=0.5
                )
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

        self.net.eval()
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.net is not None, "DeepEnsembleSurrogate must be fitted before predict()."
        X_scaled = self._scale_X(X)
        if X_scaled.ndim == 1:
            X_scaled = X_scaled[np.newaxis, :]

        with torch.no_grad():
            x_t = torch.tensor(
                X_scaled[np.newaxis, :, :], dtype=torch.float32, device=self.device
            )
            preds = self.net(x_t).cpu().numpy()

        mu_norm = np.mean(preds, axis=0)
        std_norm = np.std(preds, axis=0, ddof=1 if self.n_members > 1 else 0)

        mu = mu_norm * self.y_std + self.y_mean
        std = np.maximum(std_norm * self.y_std, 1e-6 * self.y_std)
        return mu.astype(np.float64), std.astype(np.float64)


# ==============================================================================
# Surrogate-Assisted Single-Objective HEOA (SA-HEOA)
# ==============================================================================


class _AcquisitionProblem(BenchmarkProblem):
    """Internal cheap wrapper turning Constrained Expected Improvement into a BenchmarkProblem."""

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        surrogate,
        f_best: Optional[float],
    ) -> None:
        super().__init__(
            name="SurrogateAcquisition",
            dim=len(lb),
            n_constraints=0,
            lb=lb,
            ub=ub,
            f_opt=0.0,
            x_opt=0.5 * (lb + ub),
        )
        self.surrogate = surrogate
        self.f_best = f_best

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, std = self.surrogate.predict(X)
        mu_f, std_f = mu[:, 0], std[:, 0]
        mu_g, std_g = mu[:, 1:], std[:, 1:]

        if self.f_best is not None:
            ei = expected_improvement(mu_f, std_f, f_best=self.f_best, xi=1e-4)
            log_pof = log_probability_of_feasibility(mu_g, std_g)
            score = -(np.log(ei + 1e-30) + log_pof)
        else:
            log_pof = log_probability_of_feasibility(mu_g, std_g)
            exp_cv = np.sum(np.maximum(0.0, mu_g - 0.5 * std_g), axis=-1)
            score = -log_pof + 10.0 * exp_cv

        return score, np.zeros((len(X), 0), dtype=np.float64)


class _TrustRegionSurrogateProblem(BenchmarkProblem):
    """Local trust-region surrogate constrained subproblem around current incumbent."""

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        n_constraints: int,
        surrogate,
        kappa: float = 0.20,
    ) -> None:
        super().__init__(
            name="LocalTrustRegionSurrogate",
            dim=len(lb),
            n_constraints=n_constraints,
            lb=lb,
            ub=ub,
            f_opt=0.0,
            x_opt=0.5 * (lb + ub),
        )
        self.surrogate = surrogate
        self.kappa = kappa

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, std = self.surrogate.predict(X)
        lcb_f = mu[:, 0] - self.kappa * std[:, 0]
        pred_G = mu[:, 1:]
        return lcb_f, pred_G


class SurrogateAssistedHEOA:
    """Surrogate-Assisted Human Evolutionary Optimization Algorithm (SA-HEOA).

    Combines Latin Hypercube Design of Experiments (DoE), global Constrained
    Expected Improvement (CEI = EI * PoF) active learning via HEOA, and local
    trust-region surrogate active-boundary exploitation.
    """

    def __init__(
        self,
        max_evals: int = 160,
        n_init: Optional[int] = None,
        surrogate_type: Literal["gp", "deep_ensemble"] = "gp",
        infill_per_iter: int = 4,
        inner_pop_size: int = 32,
        inner_max_evals: int = 384,
        seed: Optional[int] = None,
    ) -> None:
        self.max_evals = int(max_evals)
        self.n_init = n_init
        self.surrogate_type = surrogate_type
        self.infill_per_iter = int(max(1, infill_per_iter))
        self.inner_pop_size = int(inner_pop_size)
        self.inner_max_evals = int(inner_max_evals)
        self.seed = seed

    def _build_surrogate(self, lb: np.ndarray, ub: np.ndarray, seed: int):
        if self.surrogate_type == "gp":
            return GaussianProcessSurrogate(lb, ub, n_restarts=0, seed=seed)
        if self.surrogate_type == "deep_ensemble":
            return DeepEnsembleSurrogate(lb, ub, n_members=5, hidden_dim=48, epochs=65, lr=1e-2, seed=seed)
        raise ValueError(f"Unknown surrogate_type: {self.surrogate_type}")

    @staticmethod
    def _ensure_novel(
        x_cand: np.ndarray,
        X_arch: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        rng: np.random.Generator,
        min_dist: float = 1e-6,
    ) -> np.ndarray:
        span = np.maximum(ub - lb, 1e-12)
        dists = np.linalg.norm((X_arch - x_cand) / span, axis=1)
        if np.min(dists) >= min_dist:
            return np.clip(x_cand, lb, ub)
        jitter = rng.normal(0.0, 4e-4, size=len(lb)) * span
        return np.clip(x_cand + jitter, lb, ub)

    def optimize(self, problem: BenchmarkProblem) -> OptimizationResult:
        base_seed = 42 if self.seed is None else int(self.seed)
        rng = np.random.default_rng(base_seed)
        lb, ub = problem.lb, problem.ub
        dim = problem.dim
        span = ub - lb

        n_init_default = max(20, 6 * dim)
        n_init = min(
            self.max_evals,
            n_init_default if self.n_init is None else int(self.n_init),
        )

        # Phase 1: Space-filling Latin Hypercube Design of Experiments
        sampler = qmc.LatinHypercube(d=dim, seed=base_seed)
        X_arch = qmc.scale(sampler.random(n=n_init), lb, ub)
        f_arch, G_arch = problem.evaluate(X_arch)
        n_evals = n_init

        _, f_b, _, cv_b, is_feas = _select_best_archive_solution(X_arch, f_arch, G_arch)
        history_evals = [n_evals]
        history_f = [f_b if is_feas else float(np.min(f_arch) + 1e3 * cv_b)]
        history_cv = [cv_b]

        step = 0
        while n_evals < self.max_evals:
            step += 1
            progress = n_evals / max(1, self.max_evals)
            rem_budget = self.max_evals - n_evals
            k_batch = min(self.infill_per_iter, rem_budget)
            Y_arch = np.column_stack([f_arch, G_arch])

            # 1. Fit Global Surrogate & maximize CEI = EI * PoF via HEOA
            global_surr = self._build_surrogate(lb, ub, seed=base_seed + step * 7)
            global_surr.fit(X_arch, Y_arch)

            x_inc, f_inc, _, _, has_feas = _select_best_archive_solution(
                X_arch, f_arch, G_arch
            )
            f_target = f_inc if has_feas else None

            acq_prob = _AcquisitionProblem(lb, ub, global_surr, f_best=f_target)
            inner_heoa = HEOA(
                pop_size=self.inner_pop_size,
                max_evals=self.inner_max_evals,
                seed=base_seed + step * 13,
            )
            acq_res = inner_heoa.optimize(acq_prob)

            cand_list: List[np.ndarray] = []
            c1 = self._ensure_novel(acq_res.x_best, X_arch, lb, ub, rng, min_dist=1e-5)
            cand_list.append(c1)

            # 2. Local Trust-Region Surrogate Active-Boundary Exploitation
            if len(cand_list) < k_batch:
                radius = max(0.015, 0.35 * ((1.0 - progress) ** 1.4))
                tr_lb = np.maximum(lb, x_inc - radius * span)
                tr_ub = np.minimum(ub, x_inc + radius * span)

                k_loc = min(len(X_arch), max(18, 8 * dim))
                norm_dists = np.linalg.norm(
                    (X_arch - x_inc) / np.maximum(span, 1e-12), axis=1
                )
                loc_idx = np.argsort(norm_dists)[:k_loc]
                local_surr = GaussianProcessSurrogate(
                    tr_lb, tr_ub, n_restarts=0, seed=base_seed + step * 19
                )
                local_surr.fit(X_arch[loc_idx], Y_arch[loc_idx])

                tr_prob = _TrustRegionSurrogateProblem(
                    tr_lb,
                    tr_ub,
                    problem.n_constraints,
                    local_surr,
                    kappa=0.15 * (1.0 - progress),
                )
                tr_heoa = HEOA(
                    pop_size=28,
                    max_evals=336,
                    constraint_handler=DebFeasibilityRules(),
                    seed=base_seed + step * 23,
                )
                tr_res = tr_heoa.optimize(tr_prob)
                X_so_far = np.vstack([X_arch] + cand_list)
                c2 = self._ensure_novel(
                    tr_res.x_best, X_so_far, lb, ub, rng, min_dist=1e-6
                )
                cand_list.append(c2)

                # 3. Micro Trust-Region Active-Boundary Polish (pure vectorized GP exploitation)
                if len(cand_list) < k_batch:
                    if problem.n_constraints > 0 and progress >= 0.40:
                        micro_r = max(2e-3, 0.22 * radius)
                        m_lb = np.maximum(lb, tr_res.x_best - micro_r * span)
                        m_ub = np.minimum(ub, tr_res.x_best + micro_r * span)
                        m_ub = np.maximum(m_ub, m_lb + 1e-6 * span)
                        m_prob = _TrustRegionSurrogateProblem(
                            m_lb, m_ub, problem.n_constraints, local_surr, kappa=0.0
                        )
                        m_res = HEOA(
                            pop_size=24,
                            max_evals=288,
                            constraint_handler=DebFeasibilityRules(),
                            seed=base_seed + step * 31,
                        ).optimize(m_prob)
                        x_pol = m_res.x_best
                    else:
                        glob_lcb = _TrustRegionSurrogateProblem(
                            lb, ub, problem.n_constraints, global_surr, kappa=1.0
                        )
                        lcb_res = HEOA(
                            pop_size=24,
                            max_evals=240,
                            seed=base_seed + step * 29,
                        ).optimize(glob_lcb)
                        x_pol = lcb_res.x_best

                    X_so_far = np.vstack([X_arch] + cand_list)
                    c3 = self._ensure_novel(x_pol, X_so_far, lb, ub, rng, min_dist=1e-6)
                    cand_list.append(c3)

                # Fill any remaining slot in batch from diverse high-CEI archive of inner HEOA
                while len(cand_list) < k_batch:
                    X_so_far = np.vstack([X_arch] + cand_list)
                    idx_pick = int(rng.integers(0, min(10, len(acq_res.evaluated_X))))
                    cx = self._ensure_novel(
                        acq_res.evaluated_X[-1 - idx_pick],
                        X_so_far,
                        lb,
                        ub,
                        rng,
                        min_dist=1e-4,
                    )
                    cand_list.append(cx)

            X_new = np.asarray(cand_list[:k_batch], dtype=np.float64)
            f_new, G_new = problem.evaluate(X_new)
            n_evals += len(X_new)

            X_arch = np.vstack([X_arch, X_new])
            f_arch = np.concatenate([f_arch, f_new])
            G_arch = np.vstack([G_arch, G_new])

            _, f_b, _, cv_b, is_feas = _select_best_archive_solution(
                X_arch, f_arch, G_arch
            )
            history_evals.append(n_evals)
            history_f.append(f_b if is_feas else float(np.min(f_arch) + 1e3 * cv_b))
            history_cv.append(cv_b)

        x_best, f_best, g_best, cv_best, is_feas = _select_best_archive_solution(
            X_arch, f_arch, G_arch
        )
        alg_label = "SA-HEOA" if self.surrogate_type == "gp" else "SA-HEOA-DEns"

        return OptimizationResult(
            algorithm=alg_label,
            x_best=x_best,
            f_best=f_best,
            g_best=g_best,
            cv_best=cv_best,
            is_feasible=is_feas,
            n_evals=n_evals,
            history_evals=np.asarray(history_evals, dtype=np.int64),
            history_f=np.asarray(history_f, dtype=np.float64),
            history_cv=np.asarray(history_cv, dtype=np.float64),
            evaluated_X=X_arch,
            evaluated_f=f_arch,
            evaluated_G=G_arch,
            n_init=n_init,
        )


# ==============================================================================
# Surrogate-Assisted Multi-Objective NSGA-II (SA-NSGA-II)
# ==============================================================================


class _SurrogateMOOProblem(MultiObjectiveProblem):
    """Internal surrogate multi-objective problem evaluating LCB objectives & constraints."""

    def __init__(
        self,
        base_problem: MultiObjectiveProblem,
        surrogate,
        kappa: float,
        beta_g: float,
    ) -> None:
        super().__init__(
            name=f"Surrogate_{base_problem.name}",
            dim=base_problem.dim,
            n_obj=base_problem.n_obj,
            n_constraints=base_problem.n_constraints,
            lb=base_problem.lb,
            ub=base_problem.ub,
            ref_point=base_problem.ref_point,
        )
        self.surrogate = surrogate
        self.kappa = kappa
        self.beta_g = beta_g

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, std = self.surrogate.predict(X)
        m = self.n_obj
        lcb_F = mu[:, :m] - self.kappa * std[:, :m]
        if self.n_constraints > 0:
            eff_G = mu[:, m:] + self.beta_g * std[:, m:]
        else:
            eff_G = np.zeros((len(X), 0), dtype=np.float64)
        return lcb_F, eff_G

    def pareto_front(self, n_points: int = 50) -> np.ndarray:
        return np.zeros((n_points, self.n_obj))


class SurrogateAssistedNSGA2:
    """Surrogate-Assisted NSGA-II (SA-NSGA-II) for Expensive Multi-Objective Problems."""

    def __init__(
        self,
        max_evals: int = 220,
        n_init: Optional[int] = None,
        surrogate_type: Literal["gp", "deep_ensemble"] = "gp",
        infill_per_iter: int = 10,
        inner_pop_size: int = 56,
        inner_max_evals: int = 896,
        seed: Optional[int] = None,
    ) -> None:
        self.max_evals = int(max_evals)
        self.n_init = n_init
        self.surrogate_type = surrogate_type
        self.infill_per_iter = int(max(1, infill_per_iter))
        self.inner_pop_size = int(inner_pop_size)
        self.inner_max_evals = int(inner_max_evals)
        self.seed = seed

    def _build_surrogate(self, lb: np.ndarray, ub: np.ndarray, seed: int):
        if self.surrogate_type == "gp":
            return GaussianProcessSurrogate(lb, ub, n_restarts=0, seed=seed)
        if self.surrogate_type == "deep_ensemble":
            return DeepEnsembleSurrogate(lb, ub, n_members=5, hidden_dim=48, epochs=65, lr=1e-2, seed=seed)
        raise ValueError(f"Unknown surrogate_type: {self.surrogate_type}")

    @staticmethod
    def _select_diverse_infill(
        X_cand: np.ndarray,
        F_cand: np.ndarray,
        X_arch: np.ndarray,
        F_arch_feas: Optional[np.ndarray],
        lb: np.ndarray,
        ub: np.ndarray,
        k_select: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        span_x = np.maximum(ub - lb, 1e-12)
        X_cand_norm = (X_cand - lb) / span_x
        X_arch_norm = (X_arch - lb) / span_x

        dist_to_arch = np.min(
            np.linalg.norm(
                X_cand_norm[:, np.newaxis, :] - X_arch_norm[np.newaxis, :, :], axis=-1
            ),
            axis=1,
        )
        valid_idx = np.where(dist_to_arch > 1e-4)[0]
        if len(valid_idx) == 0:
            jitter = rng.normal(0.0, 0.02, size=(k_select, len(lb))) * span_x
            base = X_cand[rng.integers(0, len(X_cand), size=k_select)]
            return np.clip(base + jitter, lb, ub)

        Xc = X_cand[valid_idx]
        Fc = F_cand[valid_idx]
        if len(Xc) <= k_select:
            return Xc

        f_min = np.min(Fc, axis=0)
        f_max = np.max(Fc, axis=0)
        f_span = np.maximum(f_max - f_min, 1e-8)
        Fc_norm = (Fc - f_min) / f_span

        selected: List[int] = []
        for m in range(Fc.shape[1]):
            ext_idx = int(np.argmin(Fc_norm[:, m]))
            if ext_idx not in selected and len(selected) < k_select:
                selected.append(ext_idx)

        if F_arch_feas is not None and len(F_arch_feas) > 0:
            Fa_norm = (F_arch_feas - f_min) / f_span
            ref_F = np.vstack([Fa_norm, Fc_norm[selected]]) if selected else Fa_norm
        else:
            ref_F = Fc_norm[selected] if selected else Fc_norm[[0]]

        while len(selected) < k_select:
            dists = np.min(
                np.linalg.norm(
                    Fc_norm[:, np.newaxis, :] - ref_F[np.newaxis, :, :], axis=-1
                ),
                axis=1,
            )
            for idx in selected:
                dists[idx] = -1.0
            best_next = int(np.argmax(dists))
            selected.append(best_next)
            ref_F = np.vstack([ref_F, Fc_norm[[best_next]]])

        return Xc[selected]

    def optimize(
        self,
        problem: MultiObjectiveProblem,
        track_hv: bool = True,
    ) -> MultiObjectiveResult:
        base_seed = 42 if self.seed is None else int(self.seed)
        rng = np.random.default_rng(base_seed)
        lb, ub = problem.lb, problem.ub
        dim = problem.dim

        n_init_default = min(max(36, 5 * dim), max(20, self.max_evals // 3))
        n_init = min(
            self.max_evals,
            n_init_default if self.n_init is None else int(self.n_init),
        )

        sampler = qmc.LatinHypercube(d=dim, seed=base_seed)
        X_arch = qmc.scale(sampler.random(n=n_init), lb, ub)
        F_arch, G_arch = problem.evaluate(X_arch)
        n_evals = n_init

        history_evals = [n_evals]
        cv0 = compute_cv(G_arch)
        if track_hv and np.any(cv0 <= 0.0):
            _, F_p0, _ = _extract_feasible_pareto(X_arch, F_arch, G_arch)
            history_hv = [compute_hypervolume(F_p0, problem.ref_point)]
        else:
            history_hv = [0.0]

        step = 0
        while n_evals < self.max_evals:
            step += 1
            progress = n_evals / max(1, self.max_evals)
            Y_arch = np.hstack([F_arch, G_arch])

            surrogate = self._build_surrogate(lb, ub, seed=base_seed + step * 11)
            surrogate.fit(X_arch, Y_arch)

            kappa = 1.2 * ((1.0 - progress) ** 1.2) + 0.10
            beta_g = -0.15 * (1.0 - progress) + 0.05 * progress

            surr_prob = _SurrogateMOOProblem(
                problem, surrogate, kappa=kappa, beta_g=beta_g
            )
            inner_nsga2 = NSGA2(
                pop_size=self.inner_pop_size,
                max_evals=self.inner_max_evals,
                seed=base_seed + step * 17,
            )
            surr_res = inner_nsga2.optimize(surr_prob, track_hv=False)

            k_batch = min(self.infill_per_iter, self.max_evals - n_evals)
            cv_arch = compute_cv(G_arch)
            F_feas_cur = None
            if np.any(cv_arch <= 0.0):
                _, F_feas_cur, _ = _extract_feasible_pareto(X_arch, F_arch, G_arch)

            if len(surr_res.X_pareto) >= k_batch:
                X_pool, F_pool = surr_res.X_pareto, surr_res.F_pareto
            else:
                X_pool, F_pool = surr_res.X_pop, surr_res.F_pop

            X_new = self._select_diverse_infill(
                X_pool, F_pool, X_arch, F_feas_cur, lb, ub, k_batch, rng
            )
            F_new, G_new = problem.evaluate(X_new)
            n_evals += len(X_new)

            X_arch = np.vstack([X_arch, X_new])
            F_arch = np.vstack([F_arch, F_new])
            G_arch = np.vstack([G_arch, G_new])

            history_evals.append(n_evals)
            if track_hv:
                cv_cur = compute_cv(G_arch)
                if np.any(cv_cur <= 0.0):
                    _, F_pf_cur, _ = _extract_feasible_pareto(X_arch, F_arch, G_arch)
                    history_hv.append(compute_hypervolume(F_pf_cur, problem.ref_point))
                else:
                    history_hv.append(0.0)
            else:
                history_hv.append(0.0)

        X_pf, F_pf, G_pf = _extract_feasible_pareto(X_arch, F_arch, G_arch)
        alg_label = "SA-NSGA-II" if self.surrogate_type == "gp" else "SA-NSGA-II-DEns"

        return MultiObjectiveResult(
            algorithm=alg_label,
            X_pareto=X_pf,
            F_pareto=F_pf,
            G_pareto=G_pf,
            X_pop=X_pf,
            F_pop=F_pf,
            G_pop=G_pf,
            n_evals=n_evals,
            history_evals=np.asarray(history_evals, dtype=np.int64),
            history_hv=np.asarray(history_hv, dtype=np.float64),
            evaluated_X=X_arch,
            evaluated_F=F_arch,
            evaluated_G=G_arch,
            n_init=n_init,
        )
