"""
Constrained Engineering Design Benchmarks and Multi-Objective Test Suites.

Includes exact mathematical formulations, nonlinear stress/safety constraints,
and reference global optima / Pareto frontiers from the optimization literature:
  1. Three-Bar Truss Design Problem (2D, 3 nonlinear stress constraints)
  2. Tension/Compression Spring Design Problem (3D, 4 nonlinear constraints)
  3. Cylindrical Pressure Vessel Design Problem (4D, 4 nonlinear constraints)
  4. Welded Beam Structural Design Problem (4D, 7 nonlinear constraints)
  5. Peroxone (O3/H2O2) Advanced Oxidation Process (AOP) Reaction Kinetics
     Bi-Objective Optimization (4D, 3 regulatory/safety constraints incl. Bromate)
  6. ZDT1, ZDT2, ZDT3, and OSY Multi-Objective Benchmarks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np


class BenchmarkProblem(ABC):
    """Base class for single-objective constrained optimization problems.

    Convention:
      Minimize f(x) subject to g_j(x) <= 0 for j = 1, ..., m
      and lb <= x <= ub.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        n_constraints: int,
        lb: np.ndarray,
        ub: np.ndarray,
        f_opt: float,
        x_opt: np.ndarray,
    ) -> None:
        self.name = name
        self.dim = dim
        self.n_constraints = n_constraints
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.f_opt = float(f_opt)
        self.x_opt = np.asarray(x_opt, dtype=np.float64)

    @abstractmethod
    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch X of shape (N, dim). Returns (f, G) of shapes (N,) and (N, m)."""

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate single point (dim,) or batch (N, dim).

        Returns:
            f: Objective value(s), float if input is 1D else (N,) ndarray.
            g: Constraint value(s) g(x) <= 0, (m,) if 1D else (N, m) ndarray.
        """
        arr = np.asarray(x, dtype=np.float64)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr[np.newaxis, :]
        f, g = self._evaluate_batch(arr)
        if is_1d:
            return float(f[0]), g[0]
        return f, g

    @staticmethod
    def total_constraint_violation(g: np.ndarray) -> np.ndarray:
        """Compute total constraint violation CV(x) = sum_j max(0, g_j(x))."""
        g_arr = np.asarray(g, dtype=np.float64)
        return np.sum(np.maximum(0.0, g_arr), axis=-1)


class ThreeBarTruss(BenchmarkProblem):
    """Three-Bar Truss Structural Design Problem.

    Minimizes the volume of a statically loaded 3-bar planar truss subject to
    stress constraints on each of the three members.

    Formulation:
        min f(x) = (2 * sqrt(2) * x_1 + x_2) * L
        s.t.
            g_1(x) = (sqrt(2)*x_1 + x_2) / (sqrt(2)*x_1^2 + 2*x_1*x_2) * P - sigma <= 0
            g_2(x) = x_2 / (sqrt(2)*x_1^2 + 2*x_1*x_2) * P - sigma <= 0
            g_3(x) = 1 / (sqrt(2)*x_2 + x_1) * P - sigma <= 0
        where L = 100 cm, P = 2 kN/cm^2, sigma = 2 kN/cm^2,
        and x_1, x_2 in [1e-5, 1.0].

    Exact analytical optimum:
        x_1* = 0.5 + 1 / (2 * sqrt(3)) = 0.7886751345948129
        x_2* = 1 / sqrt(6)             = 0.4082482904638630
        f(x*) = 263.89584337646845
    """

    def __init__(self) -> None:
        x1_opt = 0.5 + 1.0 / (2.0 * np.sqrt(3.0))
        x2_opt = 1.0 / np.sqrt(6.0)
        f_opt = (2.0 * np.sqrt(2.0) * x1_opt + x2_opt) * 100.0
        super().__init__(
            name="Three-Bar Truss",
            dim=2,
            n_constraints=3,
            lb=np.array([1e-5, 1e-5]),
            ub=np.array([1.0, 1.0]),
            f_opt=f_opt,
            x_opt=np.array([x1_opt, x2_opt]),
        )
        self.L = 100.0
        self.P = 2.0
        self.sigma = 2.0

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1 = np.clip(X[:, 0], 1e-8, None)
        x2 = np.clip(X[:, 1], 1e-8, None)
        sqrt2 = np.sqrt(2.0)

        f = (2.0 * sqrt2 * x1 + x2) * self.L
        denom12 = sqrt2 * (x1 ** 2) + 2.0 * x1 * x2
        g1 = ((sqrt2 * x1 + x2) / denom12) * self.P - self.sigma
        g2 = (x2 / denom12) * self.P - self.sigma
        g3 = (1.0 / (sqrt2 * x2 + x1)) * self.P - self.sigma

        G = np.column_stack([g1, g2, g3])
        return f, G


class TensionCompressionSpring(BenchmarkProblem):
    """Tension/Compression Spring Design Problem.

    Minimizes the weight of a tension/compression spring subject to constraints on
    minimum deflection, shear stress, surge frequency, and outside diameter.

    Variables:
        x_1 = d in [0.05, 2.0]   (wire diameter)
        x_2 = D in [0.25, 1.3]   (mean coil diameter)
        x_3 = N in [2.0, 15.0]   (number of active coils)

    Literature optimum:
        x* = (0.0516890614, 0.3567177469, 11.2889653382)
        f(x*) = 0.0126652328
    """

    def __init__(self) -> None:
        x_opt = np.array([0.0516890614, 0.3567177469, 11.2889653382])
        f_opt = (x_opt[2] + 2.0) * x_opt[1] * (x_opt[0] ** 2)
        super().__init__(
            name="Tension/Compression Spring",
            dim=3,
            n_constraints=4,
            lb=np.array([0.05, 0.25, 2.0]),
            ub=np.array([2.0, 1.3, 15.0]),
            f_opt=f_opt,
            x_opt=x_opt,
        )

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        f = (x3 + 2.0) * x2 * (x1 ** 2)
        g1 = 1.0 - ((x2 ** 3) * x3) / (71785.0 * (x1 ** 4))
        denom_g2 = 12566.0 * (x2 * (x1 ** 3) - (x1 ** 4))
        denom_g2 = np.where(np.abs(denom_g2) < 1e-12, 1e-12, denom_g2)
        g2 = (4.0 * (x2 ** 2) - x1 * x2) / denom_g2 + 1.0 / (5108.0 * (x1 ** 2)) - 1.0
        g3 = 1.0 - (140.45 * x1) / ((x2 ** 2) * x3)
        g4 = (x1 + x2) / 1.5 - 1.0

        G = np.column_stack([g1, g2, g3, g4])
        return f, G


class PressureVessel(BenchmarkProblem):
    """Cylindrical Pressure Vessel Design Problem.

    Minimizes the total manufacturing cost (material, forming, and welding)
    of a cylindrical pressure vessel capped at both ends by hemispherical heads.

    Variables:
        x_1 = T_s in [0.0625, 6.1875] (shell thickness, in.)
        x_2 = T_h in [0.0625, 6.1875] (head thickness, in.)
        x_3 = R   in [10.0, 200.0]    (inner radius, in.)
        x_4 = L   in [10.0, 200.0]    (cylindrical shell length, in.)

    Formulation:
        min f(x) = 0.6224*x_1*x_3*x_4 + 1.7781*x_2*x_3^2 + 3.1661*x_1^2*x_4 + 19.84*x_1^2*x_3
        s.t.
            g_1(x) = -x_1 + 0.0193*x_3 <= 0
            g_2(x) = -x_2 + 0.00954*x_3 <= 0
            g_3(x) = -pi*x_3^2*x_4 - (4/3)*pi*x_3^3 + 1296000 <= 0
            g_4(x) = x_4 - 240 <= 0
    """

    def __init__(self, discrete_thickness: bool = False) -> None:
        self.discrete_thickness = discrete_thickness
        if discrete_thickness:
            x_opt = np.array([0.8125, 0.4375, 42.0984456, 176.6365958])
            f_opt = 6059.714335
        else:
            x4_opt = 200.0
            roots = np.roots([4.0 / 3.0 * np.pi, 200.0 * np.pi, 0.0, -1296000.0])
            x3_opt = float(np.real(roots[np.isreal(roots) & (np.real(roots) > 0)][0]))
            x1_opt = 0.0193 * x3_opt
            x2_opt = 0.00954 * x3_opt
            x_opt = np.array([x1_opt, x2_opt, x3_opt, x4_opt])
            f_opt = (
                0.6224 * x1_opt * x3_opt * x4_opt
                + 1.7781 * x2_opt * (x3_opt ** 2)
                + 3.1661 * (x1_opt ** 2) * x4_opt
                + 19.84 * (x1_opt ** 2) * x3_opt
            )

        super().__init__(
            name="Pressure Vessel",
            dim=4,
            n_constraints=4,
            lb=np.array([0.0625, 0.0625, 10.0, 10.0]),
            ub=np.array([6.1875, 6.1875, 200.0, 200.0]),
            f_opt=f_opt,
            x_opt=x_opt,
        )

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1 = X[:, 0]
        x2 = X[:, 1]
        if self.discrete_thickness:
            x1 = np.round(x1 / 0.0625) * 0.0625
            x2 = np.round(x2 / 0.0625) * 0.0625
        x3 = X[:, 2]
        x4 = X[:, 3]

        f = (
            0.6224 * x1 * x3 * x4
            + 1.7781 * x2 * (x3 ** 2)
            + 3.1661 * (x1 ** 2) * x4
            + 19.84 * (x1 ** 2) * x3
        )
        g1 = -x1 + 0.0193 * x3
        g2 = -x2 + 0.00954 * x3
        g3 = -np.pi * (x3 ** 2) * x4 - (4.0 / 3.0) * np.pi * (x3 ** 3) + 1296000.0
        g4 = x4 - 240.0

        G = np.column_stack([g1, g2, g3, g4])
        return f, G


class WeldedBeam(BenchmarkProblem):
    """Welded Beam Structural Design Problem (Coello / Deb Formulation).

    Minimizes fabrication cost of a welded cantilever beam subject to constraints
    on weld shear stress, beam bending stress, buckling load, end deflection,
    and geometric weld limits.

    Variables:
        x_1 = h in [0.1, 2.0]    (weld thickness, in.)
        x_2 = l in [0.1, 10.0]   (weld length, in.)
        x_3 = t in [0.1, 10.0]   (beam height, in.)
        x_4 = b in [0.1, 2.0]    (beam width, in.)

    Literature optimum:
        x* = (0.2057296398, 3.4704886656, 9.0366239104, 0.2057296398)
        f(x*) = 1.7248523086
    """

    def __init__(self) -> None:
        x_opt = np.array([0.2057296398, 3.4704886656, 9.0366239104, 0.2057296398])
        f_opt = 1.10471 * (x_opt[0] ** 2) * x_opt[1] + 0.04811 * x_opt[2] * x_opt[3] * (14.0 + x_opt[1])
        super().__init__(
            name="Welded Beam",
            dim=4,
            n_constraints=7,
            lb=np.array([0.1, 0.1, 0.1, 0.1]),
            ub=np.array([2.0, 10.0, 10.0, 2.0]),
            f_opt=f_opt,
            x_opt=x_opt,
        )

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, l, t, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        P = 6000.0
        L = 14.0
        E = 30.0e6
        G_mod = 12.0e6
        tau_max = 13600.0
        sigma_max = 30000.0
        delta_max = 0.25

        f = 1.10471 * (h ** 2) * l + 0.04811 * t * b * (14.0 + l)

        tau_p = P / (np.sqrt(2.0) * h * l)
        M = P * (L + l / 2.0)
        R = np.sqrt((l ** 2) / 4.0 + ((h + t) / 2.0) ** 2)
        J = 2.0 * (np.sqrt(2.0) * h * l * ((l ** 2) / 12.0 + ((h + t) / 2.0) ** 2))
        tau_pp = M * R / J
        tau = np.sqrt(tau_p ** 2 + 2.0 * tau_p * tau_pp * (l / (2.0 * R)) + tau_pp ** 2)

        sigma = 6.0 * P * L / (b * (t ** 2))
        delta = 4.0 * P * (L ** 3) / (E * (t ** 3) * b)
        P_c = (
            (4.013 * E * np.sqrt((t ** 2) * (b ** 6) / 36.0))
            / (L ** 2)
            * (1.0 - (t / (2.0 * L)) * np.sqrt(E / (4.0 * G_mod)))
        )

        g1 = tau - tau_max
        g2 = sigma - sigma_max
        g3 = h - b
        g4 = 0.10471 * (h ** 2) + 0.04811 * t * b * (14.0 + l) - 5.0
        g5 = 0.125 - h
        g6 = delta - delta_max
        g7 = P - P_c

        G = np.column_stack([g1, g2, g3, g4, g5, g6, g7])
        return f, G


class MultiObjectiveProblem(ABC):
    """Base class for multi-objective (optionally constrained) optimization problems.

    Convention:
        Minimize F(x) = (f_1(x), ..., f_M(x))
        subject to g_j(x) <= 0, j = 1, ..., m
        and lb <= x <= ub.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        n_obj: int,
        n_constraints: int,
        lb: np.ndarray,
        ub: np.ndarray,
        ref_point: np.ndarray,
    ) -> None:
        self.name = name
        self.dim = dim
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.ref_point = np.asarray(ref_point, dtype=np.float64)

    @abstractmethod
    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return F of shape (N, n_obj) and G of shape (N, n_constraints)."""

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x, dtype=np.float64)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr[np.newaxis, :]
        F, G = self._evaluate_batch(arr)
        if is_1d:
            return F[0], G[0]
        return F, G

    @abstractmethod
    def pareto_front(self, n_points: int = 200) -> np.ndarray:
        """Return reference Pareto front of shape (n_points, n_obj)."""


class ChemicalReactionKineticsMOO(MultiObjectiveProblem):
    """Peroxone (O3/H2O2) Advanced Oxidation Process (AOP) Reaction Kinetics MOO.

    Models ozonation / peroxone water treatment of recalcitrant micropollutants
    (e.g., pharmaceuticals in bromide-containing surface water), simultaneously
    maximizing micropollutant removal eta and minimizing Electrical/Oxidant Energy
    per Order (EE/O) subject to the WHO/EPA carcinogenic bromate byproduct limit
    ([BrO3-] <= 10 ug/L).

    Decision Variables (D = 4):
        x_1 = pH     in [5.5, 9.0]    Solution pH
        x_2 = [O3]_0 in [0.8, 12.0]   Applied ozone dose (mg/L)
        x_3 = r_H2O2 in [0.0, 1.2]    Molar H2O2 / O3 ratio (mol/mol)
        x_4 = t      in [2.0, 30.0]   Hydraulic retention time, HRT (min)

    Objectives (both minimized):
        f_1(x) = 1 - eta(x) = C_t / C_0
                 (Residual micropollutant fraction; minimizing f_1 maximizes removal eta)
        f_2(x) = EE/O(x) [kWh / m^3 / order]
                 (Electrical & oxidant energy cost per log-order contaminant removal)

    Regulatory & Process Constraints (g_j(x) <= 0):
        g_1(x) = [BrO3-](x) - 10.0 <= 0
                 (WHO / US EPA drinking water bromate standard <= 10 ug/L)
        g_2(x) = 0.65 - eta(x) <= 0
                 (Minimum required micropollutant removal efficiency >= 65%)
        g_3(x) = [H2O2]_res(x) - 3.5 <= 0
                 (Effluent residual hydrogen peroxide quenching limit <= 3.5 mg/L)
    """

    _CLASS_PF_CACHE: Dict[Tuple[bool, int], np.ndarray] = {}

    def __init__(self) -> None:
        super().__init__(
            name="AOP Ozonation Kinetics MOO",
            dim=4,
            n_obj=2,
            n_constraints=3,
            lb=np.array([5.5, 0.8, 0.0, 2.0]),
            ub=np.array([9.0, 12.0, 1.2, 30.0]),
            ref_point=np.array([0.38, 0.34]),
        )

    def evaluate_diagnostics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute mechanistic kinetic exposures, removal, EE/O, and bromate formation."""
        arr = np.asarray(X, dtype=np.float64)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr[np.newaxis, :]

        pH = arr[:, 0]
        O3 = np.clip(arr[:, 1], 0.1, None)
        r = np.clip(arr[:, 2], 0.0, None)
        t = np.clip(arr[:, 3], 0.1, None)

        # OH- and HO2- accelerated ozone decomposition rate constant k_d (min^-1)
        k_d = 0.045 * (10.0 ** (0.22 * (pH - 7.0))) * (1.0 + 2.4 * r)

        # Effective oxidant exposure with NOM/carbonate radical scavenging saturation
        ox_eff = O3 / (1.0 + 0.28 * O3)
        E_O3 = ox_eff * (1.0 - np.exp(-k_d * t)) / (k_d * 4.5)

        # Peroxone radical yield (peaks near r ~ 0.45-0.75; excess H2O2 scavenges *OH)
        phi_peroxone = 1.0 + 1.8 * r * np.exp(-1.3 * r)
        phi_pH = (10.0 ** (0.18 * (pH - 7.0))) * np.exp(-0.07 * ((pH - 8.1) ** 2))

        psi = (
            0.22 * E_O3
            + 0.44 * ox_eff * (1.0 - np.exp(-1.25 * k_d * t)) * phi_peroxone * phi_pH
        )
        residual_fraction = np.exp(-psi)
        eta = 1.0 - residual_fraction

        # Carcinogenic bromate [BrO3-] (ug/L): promoted by O3 & alkaline OBr-, suppressed by H2O2
        bromate_ug_L = (
            2.4
            * (O3 ** 1.48)
            * (10.0 ** (0.42 * (pH - 7.0)))
            * (1.0 - np.exp(-0.90 * k_d * t))
            * np.exp(-2.30 * r)
        )

        # Residual H2O2 in reactor effluent (mg/L)
        h2o2_res_mg_L = (34.0 / 48.0) * r * O3 * np.exp(-0.35 * k_d * t)

        # Specific electrical + chemical oxidant energy (kWh / m^3) & EE/O (kWh / m^3 / order)
        E_elec = (
            0.015 * (O3 ** 1.18)
            + 0.019 * r * O3
            + 0.014 * ((pH - 7.5) ** 2)
            + 0.00035 * t
        )
        log_orders = np.maximum(psi / np.log(10.0), 0.10)
        eeo = E_elec / log_orders

        out = {
            "f1_residual": residual_fraction,
            "eta_removal": eta,
            "eeo_kwh_m3_order": eeo,
            "energy_kwh_m3": E_elec,
            "bromate_ug_L": bromate_ug_L,
            "h2o2_res_mg_L": h2o2_res_mg_L,
        }
        if is_1d:
            return {k: float(v[0]) for k, v in out.items()}
        return out

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        diag = self.evaluate_diagnostics(X)
        f1 = diag["f1_residual"]
        f2 = diag["eeo_kwh_m3_order"]

        g1 = diag["bromate_ug_L"] - 10.0
        g2 = 0.65 - diag["eta_removal"]
        g3 = diag["h2o2_res_mg_L"] - 3.5

        F = np.column_stack([f1, f2])
        G = np.column_stack([g1, g2, g3])
        return F, G

    def _compute_grid_pf(self, enforce_bromate: bool, n_points: int) -> np.ndarray:
        ph_grid = np.linspace(5.5, 8.8, 28)
        o3_grid = np.linspace(0.8, 12.0, 40)
        r_grid = np.linspace(0.0, 1.2, 30)
        t_grid = np.linspace(4.0, 30.0, 22)

        PP, OO, RR, TT = np.meshgrid(ph_grid, o3_grid, r_grid, t_grid, indexing="ij")
        X_grid = np.column_stack([PP.ravel(), OO.ravel(), RR.ravel(), TT.ravel()])
        F, G = self._evaluate_batch(X_grid)
        if enforce_bromate:
            mask = np.all(G <= 0.0, axis=1)
        else:
            mask = G[:, 1] <= 0.0  # Only require eta >= 0.65, ignore bromate limit
        F_feas = F[mask]

        order = np.argsort(F_feas[:, 0])
        F_sorted = F_feas[order]
        keep = [0]
        min_f2 = F_sorted[0, 1]
        for idx in range(1, len(F_sorted)):
            if F_sorted[idx, 1] < min_f2 - 1e-6:
                keep.append(idx)
                min_f2 = F_sorted[idx, 1]
        pf = F_sorted[keep]

        f1_interp = np.linspace(pf[0, 0], pf[-1, 0], n_points)
        f2_interp = np.interp(f1_interp, pf[:, 0], pf[:, 1])
        return np.column_stack([f1_interp, f2_interp])

    def pareto_front(self, n_points: int = 200) -> np.ndarray:
        """Compute high-density constrained reference Pareto front ([BrO3-] <= 10 ug/L)."""
        key = (True, int(n_points))
        if key not in ChemicalReactionKineticsMOO._CLASS_PF_CACHE:
            ChemicalReactionKineticsMOO._CLASS_PF_CACHE[key] = self._compute_grid_pf(
                enforce_bromate=True, n_points=n_points
            )
        return ChemicalReactionKineticsMOO._CLASS_PF_CACHE[key]

    def unconstrained_pareto_front(self, n_points: int = 200) -> np.ndarray:
        """Compute reference Pareto front ignoring the bromate regulatory limit."""
        key = (False, int(n_points))
        if key not in ChemicalReactionKineticsMOO._CLASS_PF_CACHE:
            ChemicalReactionKineticsMOO._CLASS_PF_CACHE[key] = self._compute_grid_pf(
                enforce_bromate=False, n_points=n_points
            )
        return ChemicalReactionKineticsMOO._CLASS_PF_CACHE[key]


AOPOzonationKineticsMOO = ChemicalReactionKineticsMOO


class ZDT1(MultiObjectiveProblem):
    """ZDT1 Bi-Objective Benchmark with Convex Pareto Front."""

    def __init__(self, dim: int = 30) -> None:
        super().__init__(
            name="ZDT1",
            dim=dim,
            n_obj=2,
            n_constraints=0,
            lb=np.zeros(dim),
            ub=np.ones(dim),
            ref_point=np.array([1.1, 1.1]),
        )

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        h = 1.0 - np.sqrt(np.clip(f1 / g, 0.0, None))
        f2 = g * h
        F = np.column_stack([f1, f2])
        G = np.zeros((len(X), 0), dtype=np.float64)
        return F, G

    def pareto_front(self, n_points: int = 200) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT2(MultiObjectiveProblem):
    """ZDT2 Bi-Objective Benchmark with Non-Convex (Concave) Pareto Front."""

    def __init__(self, dim: int = 30) -> None:
        super().__init__(
            name="ZDT2",
            dim=dim,
            n_obj=2,
            n_constraints=0,
            lb=np.zeros(dim),
            ub=np.ones(dim),
            ref_point=np.array([1.1, 1.1]),
        )

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        h = 1.0 - (f1 / g) ** 2
        f2 = g * h
        F = np.column_stack([f1, f2])
        G = np.zeros((len(X), 0), dtype=np.float64)
        return F, G

    def pareto_front(self, n_points: int = 200) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - f1 ** 2
        return np.column_stack([f1, f2])


class ZDT3(MultiObjectiveProblem):
    """ZDT3 Bi-Objective Benchmark with Disconnected Pareto Front."""

    def __init__(self, dim: int = 30) -> None:
        super().__init__(
            name="ZDT3",
            dim=dim,
            n_obj=2,
            n_constraints=0,
            lb=np.zeros(dim),
            ub=np.ones(dim),
            ref_point=np.array([1.1, 1.1]),
        )

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        ratio = np.clip(f1 / g, 0.0, None)
        h = 1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1)
        f2 = g * h
        F = np.column_stack([f1, f2])
        G = np.zeros((len(X), 0), dtype=np.float64)
        return F, G

    def pareto_front(self, n_points: int = 300) -> np.ndarray:
        f1_dense = np.linspace(0.0, 1.0, 5000)
        f2_dense = 1.0 - np.sqrt(f1_dense) - f1_dense * np.sin(10.0 * np.pi * f1_dense)
        keep = [0]
        min_f2 = f2_dense[0]
        for idx in range(1, len(f1_dense)):
            if f2_dense[idx] < min_f2:
                keep.append(idx)
                min_f2 = f2_dense[idx]
        pf = np.column_stack([f1_dense[keep], f2_dense[keep]])
        indices = np.linspace(0, len(pf) - 1, n_points, dtype=int)
        return pf[indices]


class OSY(MultiObjectiveProblem):
    """Osyczka and Kundu (OSY) 6-Variable, 6-Constraint Multi-Objective Problem."""

    def __init__(self) -> None:
        super().__init__(
            name="OSY",
            dim=6,
            n_obj=2,
            n_constraints=6,
            lb=np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            ub=np.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0]),
            ref_point=np.array([0.0, 80.0]),
        )

    def _evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1, x2, x3, x4, x5, x6 = (
            X[:, 0],
            X[:, 1],
            X[:, 2],
            X[:, 3],
            X[:, 4],
            X[:, 5],
        )
        f1 = -(
            25.0 * ((x1 - 2.0) ** 2)
            + (x2 - 2.0) ** 2
            + (x3 - 1.0) ** 2
            + (x4 - 4.0) ** 2
            + (x5 - 1.0) ** 2
        )
        f2 = x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2

        g1 = (2.0 - x1 - x2) / 2.0
        g2 = (x1 + x2 - 6.0) / 6.0
        g3 = (x2 - x1 - 2.0) / 2.0
        g4 = (x1 - 3.0 * x2 - 2.0) / 2.0
        g5 = ((x3 - 3.0) ** 2 + x4 - 4.0) / 4.0
        g6 = (4.0 - (x5 - 3.0) ** 2 - x6) / 4.0

        F = np.column_stack([f1, f2])
        G = np.column_stack([g1, g2, g3, g4, g5, g6])
        return F, G

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        x3 = np.linspace(1.0, 5.0, n_points)
        x1 = np.full_like(x3, 5.0)
        x2 = np.full_like(x3, 1.0)
        x4 = np.zeros_like(x3)
        x5 = np.ones_like(x3)
        x6 = np.zeros_like(x3)
        X = np.column_stack([x1, x2, x3, x4, x5, x6])
        F, _ = self._evaluate_batch(X)
        return F[np.argsort(F[:, 0])]


def get_problem(name: str):
    """Factory function for retrieving single- or multi-objective benchmark problems."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "three_bar_truss": ThreeBarTruss,
        "truss": ThreeBarTruss,
        "pressure_vessel": PressureVessel,
        "vessel": PressureVessel,
        "welded_beam": WeldedBeam,
        "beam": WeldedBeam,
        "tension_compression_spring": TensionCompressionSpring,
        "spring": TensionCompressionSpring,
        "chemical_reaction": ChemicalReactionKineticsMOO,
        "chemical_reaction_kinetics_moo": ChemicalReactionKineticsMOO,
        "aop_ozonation": AOPOzonationKineticsMOO,
        "aop_ozonation_kinetics_moo": AOPOzonationKineticsMOO,
        "zdt1": ZDT1,
        "zdt2": ZDT2,
        "zdt3": ZDT3,
        "osy": OSY,
    }
    if key not in mapping:
        raise ValueError(f"Unknown problem '{name}'. Available: {list(mapping.keys())}")
    return mapping[key]()
