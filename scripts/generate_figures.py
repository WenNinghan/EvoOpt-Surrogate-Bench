"""
Publication-Grade Figure Generator for EvoOpt-Surrogate-Bench (300 DPI).

Generates 4 multi-panel scientific figures in assets/:
  1. assets/three_bar_truss_landscape_and_trajectory.png
  2. assets/surrogate_efficiency_convergence.png
  3. assets/pareto_frontier_chemical_and_zdt.png
  4. assets/constraint_handling_ablation.png
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

if hasattr(platform, "_wmi"):
    platform._wmi = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evoopt as eo


# Clean academic typography & colorblind-safe Okabe-Ito-inspired palette
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "legend.fontsize": 8.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

PALETTE = {
    "SA-HEOA": "#D55E00",       # Vermilion / prominent highlight
    "SA-HEOA-DEns": "#CC79A7",  # Reddish purple
    "HEOA": "#0072B2",          # Deep ocean blue
    "DE": "#009E73",            # Bluish green
    "PSO": "#E69F00",           # Golden orange
    "SA-NSGA-II": "#D55E00",
    "NSGA-II": "#0072B2",
    "MOEA/D": "#009E73",
}


def plot_three_bar_truss_landscape(assets_dir: Path) -> None:
    """Figure 1: 2D Feasible Landscape, Stress Boundaries, HEOA vs SA-HEOA Queries."""
    prob = eo.ThreeBarTruss()
    n_grid = 320
    x1 = np.linspace(0.02, 1.0, n_grid)
    x2 = np.linspace(0.02, 1.0, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    pts = np.column_stack([X1.ravel(), X2.ravel()])
    f_raw, g_raw = prob.evaluate(pts)
    F_map = f_raw.reshape(n_grid, n_grid)
    G1 = g_raw[:, 0].reshape(n_grid, n_grid)
    G2 = g_raw[:, 1].reshape(n_grid, n_grid)
    G3 = g_raw[:, 2].reshape(n_grid, n_grid)
    feas_mask = (G1 <= 0.0) & (G2 <= 0.0) & (G3 <= 0.0)

    # Run representative HEOA (1,500 evals) & SA-HEOA (160 evals)
    heoa_res = eo.HEOA(pop_size=45, max_evals=1500, seed=7).optimize(prob)
    sa_res = eo.SurrogateAssistedHEOA(
        max_evals=160, n_init=24, surrogate_type="gp", infill_per_iter=4, seed=7
    ).optimize(prob)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)

    for idx_ax, ax in enumerate(axes):
        # Objective contours inside feasible region, muted outside
        F_feas = np.where(feas_mask, F_map, np.nan)
        cf = ax.contourf(
            X1, X2, F_feas, levels=22, cmap="YlGnBu_r", alpha=0.88
        )
        ax.contour(X1, X2, F_map, levels=14, colors="#555555", linewidths=0.45, alpha=0.35)

        # Infeasible zone hatching / shading
        infeas_field = np.where(~feas_mask, 1.0, np.nan)
        ax.contourf(X1, X2, infeas_field, levels=[0.5, 1.5], colors=["#F2D7D5"], alpha=0.55)

        # Nonlinear stress constraint boundaries g1=0, g3=0 (g2 is strictly inactive inside g1<=0)
        ax.contour(X1, X2, G1, levels=[0.0], colors="#C0392B", linewidths=2.0, linestyles="-")
        ax.contour(X1, X2, G3, levels=[0.0], colors="#8E44AD", linewidths=1.8, linestyles="--")

        ax.text(
            0.55, 0.72, "Infeasible Stress Zone\n($g_1(\\mathbf{x}) > 0$)",
            color="#922B21", fontsize=8.8, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#C0392B", alpha=0.82),
        )
        ax.text(
            0.88, 0.86, "Feasible Region\n($g_1, g_2, g_3 \\leq 0$)",
            color="#1B4F72", fontsize=8.6, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#1B4F72", alpha=0.82),
        )

        if idx_ax == 0:
            snaps = sorted(heoa_res.population_snapshots.items(), key=lambda kv: kv[0])
            pop_init = snaps[0][1]
            pop_mid = snaps[len(snaps) // 2][1]
            pop_final = snaps[-1][1]

            ax.scatter(
                pop_init[:, 0], pop_init[:, 1], s=26, c="#7F8C8D", marker="o",
                alpha=0.55, label="Initial Chaotic Pop ($t=0$)"
            )
            ax.scatter(
                pop_mid[:, 0], pop_mid[:, 1], s=32, c="#2980B9", marker="^",
                alpha=0.75, label="Exploration/Dev Pop ($t=0.5 T_{\\max}$)"
            )
            ax.scatter(
                pop_final[:, 0], pop_final[:, 1], s=42, c="#004D40", marker="s",
                edgecolor="white", linewidth=0.5, alpha=0.92,
                label=f"Converged Leaders (1,500 evals, $f={heoa_res.f_best:.4f}$)"
            )
            ax.set_title("(a) Standard HEOA Population Dynamics (1,500 Evals)")
        else:
            n_init = sa_res.n_init
            X_doe = sa_res.evaluated_X[:n_init]
            X_inf = sa_res.evaluated_X[n_init:]
            steps = np.arange(1, len(X_inf) + 1)

            ax.scatter(
                X_doe[:, 0], X_doe[:, 1], s=38, facecolor="white", edgecolor="#34495E",
                marker="D", linewidth=1.1, zorder=4, label=f"LHS Initial DoE ($N_{{\\mathrm{{init}}}}={n_init}$)"
            )
            sc = ax.scatter(
                X_inf[:, 0], X_inf[:, 1], s=40, c=steps, cmap="plasma",
                edgecolor="black", linewidth=0.35, zorder=5,
                label=f"CEI + Trust-Region Infill ({len(X_inf)} queries)"
            )
            cbar_inf = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cbar_inf.set_label("Active-Learning Query Index", fontsize=8.8)
            ax.set_title(f"(b) SA-HEOA Active-Learning Queries (160 Evals, $f={sa_res.f_best:.4f}$)")

        # Global Analytical Optimum Star
        ax.scatter(
            [prob.x_opt[0]], [prob.x_opt[1]], s=230, c="#FFD700", edgecolor="#111111",
            marker="*", linewidth=1.2, zorder=10,
            label=f"Analytical Optimum $\\mathbf{{x}}^*$ ($f^*={prob.f_opt:.4f}$)"
        )

        # Custom legend entries for constraint curves
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([
            Line2D([0], [0], color="#C0392B", lw=2.0, ls="-", label="Active Stress $g_1(\\mathbf{x})=0$"),
            Line2D([0], [0], color="#8E44AD", lw=1.8, ls="--", label="Member Stress $g_3(\\mathbf{x})=0$"),
        ])
        ax.legend(handles=handles, loc="lower left", framealpha=0.92)
        ax.set_xlabel("Cross-Sectional Area $x_1 = A_1 = A_3$ ($\\mathrm{cm}^2$)")
        ax.set_ylabel("Cross-Sectional Area $x_2 = A_2$ ($\\mathrm{cm}^2$)")
        ax.set_xlim(0.02, 1.0)
        ax.set_ylim(0.02, 1.0)

        if idx_ax == 0:
            cbar_f = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.02)
            cbar_f.set_label("Feasible Truss Volume $f(\\mathbf{x})$ ($\\mathrm{cm}^3$)", fontsize=8.8)

    fig.suptitle(
        "Three-Bar Truss Constrained Structural Optimization: Evolutionary Swarm vs. Kriging Surrogate Active Learning",
        fontsize=12.5, fontweight="bold",
    )
    out_path = assets_dir / "three_bar_truss_landscape_and_trajectory.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def plot_surrogate_efficiency_convergence(
    assets_dir: Path,
    report: dict,
    traj_data: np.lib.npyio.NpzFile,
) -> None:
    """Figure 2: Convergence vs True Function Evaluations Highlighting 88-89% Savings."""
    problems = [
        ("three_bar_truss", "Three-Bar Truss ($D=2, m=3$)"),
        ("tension_compression_spring", "Tension/Compression Spring ($D=3, m=4$)"),
        ("pressure_vessel", "Cylindrical Pressure Vessel ($D=4, m=4$)"),
    ]
    algs = [
        ("SA-HEOA", "SA-HEOA (GP Kriging, 160 evals)", "-", 2.5),
        ("SA-HEOA-DEns", "SA-HEOA (PyTorch Ensemble, 180 evals)", "-.", 2.1),
        ("HEOA", "HEOA (1,500 evals)", "-", 1.9),
        ("DE", "DE (1,500 evals)", "--", 1.8),
        ("PSO", "PSO (1,500 evals)", ":", 1.9),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)

    for idx, (p_key, title) in enumerate(problems):
        ax = axes[idx]
        prob = eo.get_problem(p_key)
        f_opt = prob.f_opt

        for alg_key, label, ls, lw in algs:
            prefix = f"{p_key}__{alg_key.replace('-', '_')}"
            grid = traj_data[f"{prefix}__grid"]
            curves = traj_data[f"{prefix}__curves"]
            # Plot relative optimality gap (%) = max(1e-5, (f - f*) / f* * 100)
            gaps = np.maximum(1e-5, (curves - f_opt) / f_opt * 100.0)
            med = np.median(gaps, axis=0)
            q25 = np.percentile(gaps, 25, axis=0)
            q75 = np.percentile(gaps, 75, axis=0)

            color = PALETTE[alg_key]
            ax.plot(grid, med, label=label, color=color, linestyle=ls, linewidth=lw, zorder=4 if "SA" in alg_key else 3)
            ax.fill_between(grid, q25, q75, color=color, alpha=0.14)
            ax.scatter([grid[-1]], [med[-1]], color=color, s=36, zorder=6, edgecolor="white", linewidth=0.6)

        # Annotate 89.3% evaluation reduction from 1,500 -> 160
        sa_med_end = float(
            np.median(
                np.maximum(
                    1e-5,
                    (traj_data[f"{p_key}__SA_HEOA__curves"][:, -1] - f_opt) / f_opt * 100.0,
                )
            )
        )
        ax.axvline(160, color=PALETTE["SA-HEOA"], linestyle="--", linewidth=1.0, alpha=0.65)
        ax.annotate(
            "89.3% Fewer\nTrue Evals\n(160 vs. 1,500)",
            xy=(160, max(sa_med_end, 1e-4)),
            xytext=(470, max(sa_med_end * 8.0, 0.08)),
            arrowprops=dict(arrowstyle="->", color="#B03A2E", lw=1.4),
            fontsize=8.5,
            fontweight="bold",
            color="#922B21",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#FDEDEC", edgecolor="#D55E00", alpha=0.92),
        )

        ax.set_yscale("log")
        ax.set_xlim(0, 1530)
        ax.set_title(f"({chr(97 + idx)}) {title}")
        ax.set_xlabel("True Expensive Function Evaluations")
        if idx == 0:
            ax.set_ylabel("Relative Optimality Gap $(f - f^*) / f^*$ [%] (log)")
            ax.legend(loc="upper right", framealpha=0.93)

    fig.suptitle(
        "Sample Efficiency Across Constrained Engineering Benchmarks (20 Seeds: Median & Interquartile Band)",
        fontsize=12.5, fontweight="bold",
    )
    out_path = assets_dir / "surrogate_efficiency_convergence.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def plot_pareto_frontiers(assets_dir: Path) -> None:
    """Figure 3: Constrained AOP Ozonation Reaction Kinetics + ZDT1 & ZDT3 Pareto Fronts."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True)

    # --- Panel (a): AOP Ozonation Reaction Kinetics with Bromate Boundary & Knee Point ---
    ax0 = axes[0]
    aop = eo.AOPOzonationKineticsMOO()
    pf_feas = aop.pareto_front(200)
    pf_uncons = aop.unconstrained_pareto_front(200)

    r_nsga = eo.NSGA2(pop_size=80, max_evals=2400, seed=2).optimize(aop, track_hv=False)
    r_moead = eo.MOEAD(pop_size=80, max_evals=2400, seed=2).optimize(aop, track_hv=False)
    r_sansga = eo.SurrogateAssistedNSGA2(
        max_evals=220, n_init=40, infill_per_iter=10, seed=2
    ).optimize(aop, track_hv=False)

    # Convert f1 = 1 - eta into Removal Efficiency eta (%) = 100 * (1 - f1)
    eta_feas = 100.0 * (1.0 - pf_feas[:, 0])
    eta_uncons = 100.0 * (1.0 - pf_uncons[:, 0])

    # Shade Bromate Regulatory Violation Zone between unconstrained PF and feasible PF
    eta_common = np.linspace(float(np.max(eta_feas)), float(np.max(eta_uncons)), 80)
    eeo_uncons_tail = np.interp(1.0 - eta_common / 100.0, pf_uncons[:, 0], pf_uncons[:, 1])
    ax0.fill_betweenx(
        eeo_uncons_tail, eta_feas[0], eta_common,
        color="#FADBD8", alpha=0.65, label="Bromate Violation ($[\\mathrm{BrO}_3^-] > 10\\;\\mu\\mathrm{g/L}$)"
    )
    ax0.plot(
        eta_uncons, pf_uncons[:, 1], color="#C0392B", linestyle="--", linewidth=1.6,
        label="Unconstrained PF (Violates $\\mathrm{BrO}_3^-$)"
    )
    ax0.plot(
        eta_feas, pf_feas[:, 1], color="#111111", linestyle="-", linewidth=2.2,
        label="Feasible Reference PF ($[\\mathrm{BrO}_3^-] \\leq 10\\;\\mu\\mathrm{g/L}$)"
    )

    ax0.scatter(
        100.0 * (1.0 - r_nsga.F_pareto[:, 0]), r_nsga.F_pareto[:, 1],
        s=26, c=PALETTE["NSGA-II"], alpha=0.72, marker="o", label="NSGA-II (2,400 evals)"
    )
    ax0.scatter(
        100.0 * (1.0 - r_moead.F_pareto[:, 0]), r_moead.F_pareto[:, 1],
        s=26, c=PALETTE["MOEA/D"], alpha=0.72, marker="^", label="MOEA/D (2,400 evals)"
    )
    ax0.scatter(
        100.0 * (1.0 - r_sansga.F_pareto[:, 0]), r_sansga.F_pareto[:, 1],
        s=42, c=PALETTE["SA-NSGA-II"], edgecolor="black", linewidth=0.45,
        marker="D", zorder=6, label="SA-NSGA-II (220 evals, 90.8% saved)"
    )

    # Highlight Knee-Point Operating Condition
    knee_idx = eo.find_knee_point(r_sansga.F_pareto)
    xk = r_sansga.X_pareto[knee_idx]
    dk = aop.evaluate_diagnostics(xk)
    eta_k = 100.0 * dk["eta_removal"]
    eeo_k = dk["eeo_kwh_m3_order"]
    ax0.scatter(
        [eta_k], [eeo_k], s=220, c="#FFD700", edgecolor="#111111",
        marker="*", linewidth=1.2, zorder=10, label="Knee-Point Design"
    )
    ax0.annotate(
        f"Knee-Point Reactor Regime:\n"
        f"pH={xk[0]:.2f}, $[\\mathrm{{O}}_3]_0$={xk[1]:.2f} mg/L\n"
        f"$\\mathrm{{H}}_2\\mathrm{{O}}_2/\\mathrm{{O}}_3$={xk[2]:.2f}, HRT={xk[3]:.1f} min\n"
        f"$\\eta$={eta_k:.1f}%, $[\\mathrm{{BrO}}_3^-]$={dk['bromate_ug_L']:.1f} $\\mu$g/L",
        xy=(eta_k, eeo_k),
        xytext=(72.0, 0.138),
        arrowprops=dict(arrowstyle="->", color="#111111", lw=1.3),
        fontsize=8.0,
        zorder=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", edgecolor="#B7950B", alpha=0.95),
    )

    ax0.set_title("(a) Peroxone ($\\mathrm{O}_3/\\mathrm{H}_2\\mathrm{O}_2$) AOP Reaction Kinetics")
    ax0.set_xlabel("Micropollutant Removal Efficiency $\\eta$ [%]")
    ax0.set_ylabel("Specific Energy Cost $\\mathrm{EE/O}$ [$\\mathrm{kWh}\\cdot\\mathrm{m}^{-3}\\cdot\\mathrm{order}^{-1}$]")
    ax0.legend(loc="upper left", fontsize=7.8, framealpha=0.92)

    # --- Panel (b): ZDT1 Convex Benchmark & Panel (c): ZDT3 Disconnected Benchmark ---
    for idx_panel, (ProbCls, p_title) in enumerate([
        (lambda: eo.ZDT1(dim=8), "(b) ZDT1 Convex Pareto Front ($D=8$)"),
        (lambda: eo.ZDT3(dim=8), "(c) ZDT3 Disconnected Pareto Front ($D=8$)"),
    ], start=1):
        ax = axes[idx_panel]
        prob = ProbCls()
        pf = prob.pareto_front(300)

        res_n = eo.NSGA2(pop_size=80, max_evals=2400, seed=3).optimize(prob, track_hv=False)
        res_m = eo.MOEAD(pop_size=80, max_evals=2400, seed=3).optimize(prob, track_hv=False)
        res_s = eo.SurrogateAssistedNSGA2(
            max_evals=220, n_init=40, infill_per_iter=10, seed=3
        ).optimize(prob, track_hv=False)

        if "ZDT3" in p_title:
            ax.scatter(pf[:, 0], pf[:, 1], s=8, c="#111111", label="Analytical True PF")
        else:
            ax.plot(pf[:, 0], pf[:, 1], color="#111111", lw=2.0, label="Analytical True PF")

        hv_n = eo.compute_hypervolume(res_n.F_pareto, prob.ref_point)
        hv_m = eo.compute_hypervolume(res_m.F_pareto, prob.ref_point)
        hv_s = eo.compute_hypervolume(res_s.F_pareto, prob.ref_point)

        ax.scatter(
            res_n.F_pareto[:, 0], res_n.F_pareto[:, 1], s=26, c=PALETTE["NSGA-II"],
            alpha=0.70, marker="o", label=f"NSGA-II (2,400 evals, HV={hv_n:.3f})"
        )
        ax.scatter(
            res_m.F_pareto[:, 0], res_m.F_pareto[:, 1], s=26, c=PALETTE["MOEA/D"],
            alpha=0.70, marker="^", label=f"MOEA/D (2,400 evals, HV={hv_m:.3f})"
        )
        ax.scatter(
            res_s.F_pareto[:, 0], res_s.F_pareto[:, 1], s=42, c=PALETTE["SA-NSGA-II"],
            edgecolor="black", linewidth=0.45, marker="D", zorder=6,
            label=f"SA-NSGA-II (220 evals, HV={hv_s:.3f})"
        )

        ax.set_title(p_title)
        ax.set_xlabel("Objective $f_1(\\mathbf{x})$")
        ax.set_ylabel("Objective $f_2(\\mathbf{x})$")
        ax.legend(loc="upper right", fontsize=8.1, framealpha=0.92)

    fig.suptitle(
        "Multi-Objective Pareto Frontiers: Environmental Reaction Kinetics (Bromate-Constrained) & ZDT Suite",
        fontsize=12.5, fontweight="bold",
    )
    out_path = assets_dir / "pareto_frontier_chemical_and_zdt.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def plot_constraint_handling_ablation(
    assets_dir: Path,
    report: dict,
    traj_data: np.lib.npyio.NpzFile,
) -> None:
    """Figure 4: 20-Seed Statistical Ablation of Constraint-Handling Mechanisms."""
    problems = [
        ("three_bar_truss", "Three-Bar Truss", "Volume $f(\\mathbf{x})$ [$\\mathrm{cm}^3$]"),
        ("tension_compression_spring", "Tension/Compression Spring", "Spring Weight $f(\\mathbf{x})$"),
        ("pressure_vessel", "Cylindrical Pressure Vessel", "Manufacturing Cost $f(\\mathbf{x})$ [USD]"),
    ]
    handlers = [
        ("deb", "Deb's Feasibility\nRules", "#0072B2"),
        ("epsilon", "$\\epsilon$-Constraint\nRelaxation", "#009E73"),
        ("penalty", "Adaptive Dynamic\nPenalty", "#E69F00"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), constrained_layout=True)
    rng = np.random.default_rng(2026)

    for idx, (p_key, p_name, y_label) in enumerate(problems):
        ax = axes[idx]
        prob = eo.get_problem(p_key)
        data_list = []
        colors = []
        tick_labels = []

        for pos, (h_key, h_label, col) in enumerate(handlers, start=1):
            vals = traj_data[f"ablation__{p_key}__{h_key}__f_vals"]
            data_list.append(vals)
            colors.append(col)
            tick_labels.append(h_label)

        parts = ax.violinplot(
            data_list, positions=[1, 2, 3], widths=0.58, showmeans=False, showextrema=False
        )
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_edgecolor("black")
            pc.set_alpha(0.28)

        bp = ax.boxplot(
            data_list,
            positions=[1, 2, 3],
            widths=0.24,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#111111", linewidth=2.0),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
            patch.set_edgecolor("#111111")

        # Overlay individual 20-seed results & feasibility badges
        ab_dict = report["constraint_handling_ablation"][prob.name]
        json_names = ["Deb's Feasibility Rules", "Epsilon-Constraint", "Adaptive Penalty"]
        for pos, (vals, col, j_name) in enumerate(zip(data_list, colors, json_names), start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full_like(vals, pos) + jitter,
                vals,
                s=26,
                c=col,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.88,
                zorder=5,
            )
            feas_pct = ab_dict[j_name]["feasibility_rate_pct"]
            med_err = ab_dict[j_name]["rel_error_median_pct"]
            ax.text(
                pos,
                float(np.max(vals)) + 0.04 * max(float(np.ptp( np.concatenate(data_list) )), 1e-5),
                f"Feas: {feas_pct:.0f}%\nMed $\\Delta$: {med_err:.3f}%",
                ha="center",
                va="bottom",
                fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.88),
            )

        ax.axhline(
            prob.f_opt, color="#C0392B", linestyle="--", linewidth=1.6,
            label=f"Global Optimum $f^* = {prob.f_opt:.4f}$"
        )
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(tick_labels)
        ax.set_title(f"({chr(97 + idx)}) {p_name}")
        ax.set_ylabel(y_label)
        ax.legend(loc="upper left", framealpha=0.92)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + 0.16 * max(y_max - y_min, 1e-5))

    fig.suptitle(
        "Constraint-Handling Mechanism Ablation inside HEOA (20 Independent Seeds, 1,500 Evaluations)",
        fontsize=12.5, fontweight="bold",
    )
    out_path = assets_dir / "constraint_handling_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def main() -> None:
    assets_dir = ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"

    with open(results_dir / "engineering_opt_benchmark.json", "r", encoding="utf-8") as fp:
        report = json.load(fp)
    traj_data = np.load(results_dir / "benchmark_trajectories.npz")

    print("[EvoOpt-Figures] Rendering 4 publication-grade figures at 300 DPI...")
    plot_three_bar_truss_landscape(assets_dir)
    plot_surrogate_efficiency_convergence(assets_dir, report, traj_data)
    plot_pareto_frontiers(assets_dir)
    plot_constraint_handling_ablation(assets_dir, report, traj_data)
    print("[EvoOpt-Figures] All 4 figures generated successfully.")


if __name__ == "__main__":
    main()
