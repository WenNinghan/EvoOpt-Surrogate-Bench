"""
Top-Journal / CCF-A Camera-Ready Figure Generator for EvoOpt-Surrogate-Bench (300 DPI).

Generates the main benchmark figures and the held-out tuning comparison in assets/:
  1. assets/three_bar_truss_landscape_and_trajectory.png
  2. assets/surrogate_efficiency_convergence.png
  3. assets/pareto_frontier_chemical_and_zdt.png
  4. assets/constraint_handling_ablation.png
  5. assets/small_budget_tuning.png (when the tuning experiment has run)
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


# ==============================================================================
# Top-Journal / CCF-A Typography, Frame & Academic Palette Configuration
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.weight": "bold",
    "mathtext.fontset": "stix",
    "font.size": 10.2,
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.3,
    "ytick.labelsize": 9.3,
    "legend.fontsize": 8.4,
    "figure.titlesize": 12.6,
    "figure.titleweight": "bold",
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "axes.edgecolor": "#111111",
    "axes.linewidth": 1.30,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.70,
    "grid.alpha": 0.35,
    "grid.color": "#94A3B8",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# High-contrast journal palette (Navy, Emerald, Crimson, Ochre, Plum, Slate)
PALETTE = {
    "SA-HEOA": "#B03A2E",       # Crimson (primary surrogate-assisted method)
    "SA-HEOA-DEns": "#5B2C6F",  # Plum (deep ensemble surrogate)
    "HEOA": "#1B4F72",          # Navy (standard HEOA)
    "DE": "#0B6E4F",            # Emerald (Differential Evolution)
    "PSO": "#C55A11",           # Ochre (Particle Swarm Optimization)
    "SA-NSGA-II": "#B03A2E",    # Crimson
    "NSGA-II": "#1B4F72",       # Navy
    "MOEA/D": "#0B6E4F",        # Emerald
}


def _style_continuous_axes(ax: plt.Axes) -> None:
    """Apply 4-sided closed #111111 frame, 4-sided inward ticks, and subtle dotted grid."""
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.30)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=4.5,
        width=1.2,
        color="#111111",
        labelcolor="#111111",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=2.5,
        width=0.9,
        color="#111111",
    )
    ax.grid(True, which="major", linestyle=":", linewidth=0.70, alpha=0.35, color="#94A3B8")


def _style_categorical_axes(ax: plt.Axes) -> None:
    """Apply 4-sided closed #111111 frame, inward y-ticks, and outward bottom x-ticks."""
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.30)
    ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        left=True,
        right=True,
        length=4.5,
        width=1.2,
        color="#111111",
        labelcolor="#111111",
    )
    ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        bottom=True,
        top=False,
        length=4.2,
        width=1.2,
        color="#111111",
        labelcolor="#111111",
    )
    ax.grid(True, axis="y", linestyle=":", linewidth=0.70, alpha=0.35, color="#94A3B8")
    ax.grid(False, axis="x")


def _finalize_journal_figure(fig: plt.Figure) -> None:
    """Enforce bold Times New Roman across every axis, colorbar, tick, annotation, and legend."""
    fig.canvas.draw()
    for text_obj in fig.texts:
        text_obj.set_fontfamily("Times New Roman")
        text_obj.set_fontweight("bold")

    for ax in fig.axes:
        ax.title.set_fontfamily("Times New Roman")
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontfamily("Times New Roman")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontfamily("Times New Roman")
        ax.yaxis.label.set_fontweight("bold")
        ax.xaxis.get_offset_text().set_fontfamily("Times New Roman")
        ax.xaxis.get_offset_text().set_fontweight("bold")
        ax.yaxis.get_offset_text().set_fontfamily("Times New Roman")
        ax.yaxis.get_offset_text().set_fontweight("bold")

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")
        for lbl in ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True):
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")

        for txt in ax.texts:
            txt.set_fontfamily("Times New Roman")
            txt.set_fontweight("bold")

        leg = ax.get_legend()
        if leg is not None:
            for ltxt in leg.get_texts():
                ltxt.set_fontfamily("Times New Roman")
                ltxt.set_fontweight("bold")
            if leg.get_title() is not None:
                leg.get_title().set_fontfamily("Times New Roman")
                leg.get_title().set_fontweight("bold")
            frame = leg.get_frame()
            frame.set_linewidth(1.1)
            frame.set_edgecolor("#111111")
            frame.set_facecolor("#FFFFFF")


def plot_three_bar_truss_landscape(assets_dir: Path) -> None:
    """Figure 1: 2D Feasible Landscape, Stress Boundaries, HEOA vs. SA-HEOA Queries."""
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

    # Run representative HEOA (1,500 evals) & SA-HEOA (160 evals) matching benchmark hyperparameters
    heoa_res = eo.HEOA(pop_size=50, max_evals=1500, seed=7).optimize(prob)
    sa_res = eo.SurrogateAssistedHEOA(
        max_evals=160, n_init=24, surrogate_type="gp", infill_per_iter=6, seed=7
    ).optimize(prob)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), constrained_layout=True)

    for idx_ax, ax in enumerate(axes):
        _style_continuous_axes(ax)

        # Objective contours inside feasible region, muted isolines outside
        F_feas = np.where(feas_mask, F_map, np.nan)
        cf = ax.contourf(
            X1, X2, F_feas, levels=22, cmap="YlGnBu_r", alpha=0.88, zorder=1
        )
        ax.contour(
            X1, X2, F_map, levels=14, colors="#5D6D7E", linewidths=0.48, alpha=0.36, zorder=2
        )

        # Infeasible zone shading
        infeas_field = np.where(~feas_mask, 1.0, np.nan)
        ax.contourf(
            X1, X2, infeas_field, levels=[0.5, 1.5], colors=["#F8E8E6"], alpha=0.60, zorder=1
        )

        # Nonlinear stress constraint boundaries g1=0, g3=0
        ax.contour(
            X1, X2, G1, levels=[0.0], colors="#B03A2E", linewidths=2.15, linestyles="-", zorder=6
        )
        ax.contour(
            X1, X2, G3, levels=[0.0], colors="#5B2C6F", linewidths=1.85, linestyles="--", zorder=6
        )

        # Placed cleanly in open region away from active boundary query clusters
        ax.text(
            0.28, 0.77, "Infeasible Stress Zone\n($g_1(\\mathbf{x}) > 0$)",
            color="#B03A2E", fontsize=8.8, fontweight="bold", ha="center", va="center", zorder=9,
            bbox=dict(boxstyle="square,pad=0.30", facecolor="#FFFFFF", edgecolor="#B03A2E", lw=1.0, alpha=0.94),
        )
        ax.text(
            0.875, 0.895, "Feasible Region\n($g_1, g_2, g_3 \\leq 0$)",
            color="#1B4F72", fontsize=8.6, fontweight="bold", ha="center", va="center", zorder=9,
            bbox=dict(boxstyle="square,pad=0.28", facecolor="#FFFFFF", edgecolor="#1B4F72", lw=1.0, alpha=0.94),
        )

        if idx_ax == 0:
            snaps = sorted(heoa_res.population_snapshots.items(), key=lambda kv: kv[0])
            pop_init = snaps[0][1]
            pop_mid = snaps[len(snaps) // 2][1]
            pop_final = snaps[-1][1]

            ax.scatter(
                pop_init[:, 0], pop_init[:, 1], s=28, c="#5D6D7E", marker="o",
                edgecolor="#111111", linewidth=0.35, alpha=0.62, zorder=4,
                label="Initial Chaotic Pop. ($t = 0$)"
            )
            ax.scatter(
                pop_mid[:, 0], pop_mid[:, 1], s=34, c="#1B4F72", marker="^",
                edgecolor="#FFFFFF", linewidth=0.35, alpha=0.82, zorder=5,
                label="Exploration Pop. ($t = 0.5\\,T_{\\max}$)"
            )
            ax.scatter(
                pop_final[:, 0], pop_final[:, 1], s=44, c="#0B6E4F", marker="s",
                edgecolor="#FFFFFF", linewidth=0.55, alpha=0.95, zorder=7,
                label=f"Converged Leaders ($1{{,}}500$ evals,  $f = {heoa_res.f_best:.4f}$)"
            )
            ax.set_title("(a) Standard HEOA Population Dynamics ($1{,}500$ Evals)", pad=8)
        else:
            n_init = sa_res.n_init
            X_doe = sa_res.evaluated_X[:n_init]
            X_inf = sa_res.evaluated_X[n_init:]
            steps = np.arange(1, len(X_inf) + 1)

            ax.scatter(
                X_doe[:, 0], X_doe[:, 1], s=40, facecolor="#FFFFFF", edgecolor="#1B4F72",
                marker="D", linewidth=1.25, zorder=4,
                label=f"LHS Initial DoE ($N_{{\\mathrm{{init}}}} = {n_init}$)"
            )
            sc = ax.scatter(
                X_inf[:, 0], X_inf[:, 1], s=42, c=steps, cmap="plasma",
                edgecolor="#111111", linewidth=0.45, zorder=7,
                label=f"CEI + Trust-Region Infill (${len(X_inf)}$ queries)"
            )
            cbar_inf = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.025)
            cbar_inf.set_label("Active-Learning Query Index $k$", fontsize=9.2)
            cbar_inf.outline.set_edgecolor("#111111")
            cbar_inf.outline.set_linewidth(1.15)
            cbar_inf.ax.tick_params(direction="in", length=3.5, width=1.0, color="#111111")
            ax.set_title(
                f"(b) SA-HEOA Active-Learning Queries ($160$ Evals,  $f = {sa_res.f_best:.4f}$)",
                pad=8,
            )

        # Global Analytical Optimum Star
        ax.scatter(
            [prob.x_opt[0]], [prob.x_opt[1]], s=240, c="#F4D03F", edgecolor="#111111",
            marker="*", linewidth=1.25, zorder=10,
            label=f"Analytical Optimum $\\mathbf{{x}}^*$ ($f^* = {prob.f_opt:.4f}$)"
        )

        # Custom legend entries for constraint curves
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([
            Line2D([0], [0], color="#B03A2E", lw=2.15, ls="-", label="Active Stress $g_1(\\mathbf{x}) = 0$"),
            Line2D([0], [0], color="#5B2C6F", lw=1.85, ls="--", label="Member Stress $g_3(\\mathbf{x}) = 0$"),
        ])
        ax.legend(
            handles=handles,
            loc="lower left",
            fancybox=False,
            edgecolor="#111111",
            framealpha=0.96,
            fontsize=8.1,
        )
        ax.set_xlabel("Cross-Sectional Area $x_1 = A_1 = A_3$ [$\\mathrm{cm}^2$]")
        ax.set_ylabel("Cross-Sectional Area $x_2 = A_2$ [$\\mathrm{cm}^2$]")
        ax.set_xlim(0.015, 1.015)
        ax.set_ylim(0.015, 1.015)

        if idx_ax == 0:
            cbar_f = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.025)
            cbar_f.set_label("Feasible Truss Volume $f(\\mathbf{x})$ [$\\mathrm{cm}^3$]", fontsize=9.2)
            cbar_f.outline.set_edgecolor("#111111")
            cbar_f.outline.set_linewidth(1.15)
            cbar_f.ax.tick_params(direction="in", length=3.5, width=1.0, color="#111111")

    fig.suptitle(
        "Three-Bar Truss Constrained Structural Optimization: Evolutionary Swarm vs. Kriging Active Learning",
        fontsize=12.6, fontweight="bold",
    )
    _finalize_journal_figure(fig)
    out_path = assets_dir / "three_bar_truss_landscape_and_trajectory.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def plot_surrogate_efficiency_convergence(
    assets_dir: Path,
    report: dict,
    traj_data: np.lib.npyio.NpzFile,
) -> None:
    """Figure 2: objective gap as a function of evaluations under fixed budgets."""
    problems = [
        ("three_bar_truss", "Three-Bar Truss ($D = 2,\\; m = 3$)"),
        ("tension_compression_spring", "Tension/Compression Spring ($D = 3,\\; m = 4$)"),
        ("pressure_vessel", "Cylindrical Pressure Vessel ($D = 4,\\; m = 4$)"),
    ]
    algs = [
        ("SA-HEOA", "SA-HEOA (GP Kriging, $160$ evals)", "-", 2.55),
        ("SA-HEOA-DEns", "SA-HEOA (Deep Ensemble, $180$ evals)", "-.", 2.15),
        ("HEOA", "HEOA ($1{,}500$ evals)", "-", 1.95),
        ("DE", "DE ($1{,}500$ evals)", "--", 1.85),
        ("PSO", "PSO ($1{,}500$ evals)", ":", 2.00),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), constrained_layout=True)

    # Custom non-overlapping callout text positions in the open lower pocket below evolutionary curves
    callout_cfg = {
        "three_bar_truss": (410, 2.2e-4, 7e-5),
        "tension_compression_spring": (420, 0.48, 0.12),
        "pressure_vessel": (410, 1.8, 0.70),
    }

    for idx, (p_key, title) in enumerate(problems):
        ax = axes[idx]
        _style_continuous_axes(ax)
        prob = eo.get_problem(p_key)
        f_opt = prob.f_opt
        max_q75 = 10.0

        for alg_key, label, ls, lw in algs:
            prefix = f"{p_key}__{alg_key.replace('-', '_')}"
            grid = traj_data[f"{prefix}__grid"]
            curves = traj_data[f"{prefix}__curves"]
            gaps = np.maximum(1e-5, (curves - f_opt) / f_opt * 100.0)
            has_feasible_run = np.any(np.isfinite(gaps), axis=0)
            grid, gaps = grid[has_feasible_run], gaps[:, has_feasible_run]
            q25, med, q75 = np.nanpercentile(gaps, [25, 50, 75], axis=0)
            max_q75 = max(max_q75, float(np.max(q75)))

            color = PALETTE[alg_key]
            ax.plot(
                grid, med, label=label, color=color, linestyle=ls, linewidth=lw,
                zorder=5 if "SA" in alg_key else 3
            )
            ax.fill_between(grid, q25, q75, color=color, alpha=0.13, zorder=2)
            ax.scatter(
                [grid[-1]], [med[-1]], color=color, s=40, zorder=6,
                edgecolor="#111111", linewidth=0.75
            )

        # Mark the prespecified GP evaluation budget.
        sa_med_end = float(
            np.nanmedian(
                np.maximum(
                    1e-5,
                    (traj_data[f"{p_key}__SA_HEOA__curves"][:, -1] - f_opt) / f_opt * 100.0,
                )
            )
        )
        tx, ty, y_lo = callout_cfg[p_key]
        y_lo_eff = min(y_lo, 0.35 * max(sa_med_end, 1e-4))
        y_hi_eff = max_q75 * 2.2
        ax.axvline(160, color=PALETTE["SA-HEOA"], linestyle="--", linewidth=1.1, alpha=0.65, zorder=2)
        ax.annotate(
            "Evaluation budgets\n($160$ vs. $1{,}500$)",
            xy=(160, max(sa_med_end, 1e-4)),
            xytext=(tx, ty),
            arrowprops=dict(arrowstyle="-|>", color="#B03A2E", lw=1.35, mutation_scale=11),
            fontsize=8.4,
            fontweight="bold",
            color="#B03A2E",
            zorder=8,
            bbox=dict(boxstyle="square,pad=0.30", facecolor="#FFFFFF", edgecolor="#B03A2E", lw=1.0, alpha=0.96),
        )

        ax.set_yscale("log")
        ax.set_xlim(0, 1550)
        ax.set_ylim(y_lo_eff, y_hi_eff)
        _style_continuous_axes(ax)
        ax.set_title(f"({chr(97 + idx)}) {title}", pad=8)
        ax.set_xlabel("True Expensive Function Evaluations $N_{\\mathrm{eval}}$")
        if idx == 0:
            ax.set_ylabel("Relative Optimality Gap $(f - f^*) / f^*$ [%]")
            ax.legend(
                loc="upper right",
                fancybox=False,
                edgecolor="#111111",
                framealpha=0.96,
                fontsize=8.1,
            )

    fig.suptitle(
        "Feasible Objective Gap vs. Evaluation Budget (Median and IQR Across Runs With a Feasible Incumbent)",
        fontsize=12.6, fontweight="bold",
    )
    _finalize_journal_figure(fig)
    out_path = assets_dir / "surrogate_efficiency_convergence.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def plot_pareto_frontiers(
    assets_dir: Path,
    traj_data: np.lib.npyio.NpzFile,
) -> None:
    """Figure 3: Constrained AOP Ozonation Reaction Kinetics + ZDT1 & ZDT3 Pareto Fronts."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.35), constrained_layout=True)

    # --- Panel (a): AOP Ozonation Reaction Kinetics with Bromate Boundary & Knee Point ---
    ax0 = axes[0]
    _style_continuous_axes(ax0)
    aop = eo.AOPOzonationKineticsMOO()
    pf_feas = aop.pareto_front(250)
    pf_uncons = aop.unconstrained_pareto_front(250)

    r_nsga = eo.NSGA2(pop_size=80, max_evals=2400, seed=0).optimize(aop, track_hv=False)
    r_moead = eo.MOEAD(pop_size=80, max_evals=2400, seed=0).optimize(aop, track_hv=False)
    if "aop_rep__SA_NSGA_II__X_pareto" in traj_data:
        sa_X_pf = traj_data["aop_rep__SA_NSGA_II__X_pareto"]
        sa_F_pf = traj_data["aop_rep__SA_NSGA_II__F_pareto"]
        knee_idx = int(traj_data["aop_rep__knee_idx"][0])
    else:
        r_sansga = eo.SurrogateAssistedNSGA2(
            max_evals=220, n_init=40, infill_per_iter=12, inner_pop_size=52, inner_max_evals=728, seed=0
        ).optimize(aop, track_hv=False)
        sa_X_pf = r_sansga.X_pareto
        sa_F_pf = r_sansga.F_pareto
        knee_idx = eo.find_knee_point(sa_F_pf)

    eta_feas = 100.0 * (1.0 - pf_feas[:, 0])
    eta_uncons = 100.0 * (1.0 - pf_uncons[:, 0])

    # Shade Bromate Regulatory Violation Zone between Feasible PF and Unconstrained PF
    eeo_grid = np.linspace(
        float(max(np.min(pf_feas[:, 1]), np.min(pf_uncons[:, 1]))),
        float(min(np.max(pf_feas[:, 1]), np.max(pf_uncons[:, 1]))),
        120,
    )
    # Sort ascending in EEO for horizontal fill_betweenx
    ord_f = np.argsort(pf_feas[:, 1])
    ord_u = np.argsort(pf_uncons[:, 1])
    eta_f_of_eeo = np.interp(eeo_grid, pf_feas[ord_f, 1], eta_feas[ord_f])
    eta_u_of_eeo = np.interp(eeo_grid, pf_uncons[ord_u, 1], eta_uncons[ord_u])
    ax0.fill_betweenx(
        eeo_grid,
        eta_f_of_eeo,
        np.maximum(eta_f_of_eeo, eta_u_of_eeo),
        color="#FADBD8",
        alpha=0.72,
        zorder=1,
        label="Bromate Violation ($[\\mathrm{BrO}_3^-] > 10\\;\\mu\\mathrm{g\\,L^{-1}}$)",
    )
    ax0.plot(
        eta_uncons, pf_uncons[:, 1], color="#B03A2E", linestyle="--", linewidth=1.75, zorder=2,
        label="Unconstrained PF (Violates $\\mathrm{BrO}_3^-$)"
    )
    ax0.plot(
        eta_feas, pf_feas[:, 1], color="#111111", linestyle="-", linewidth=2.25, zorder=3,
        label="Feasible PF ($[\\mathrm{BrO}_3^-] \\leq 10\\;\\mu\\mathrm{g\\,L^{-1}}$)"
    )

    ax0.scatter(
        100.0 * (1.0 - r_nsga.F_pareto[:, 0]), r_nsga.F_pareto[:, 1],
        s=30, c=PALETTE["NSGA-II"], edgecolor="#FFFFFF", linewidth=0.35,
        alpha=0.80, marker="o", zorder=4, label="NSGA-II ($2{,}400$ evals)"
    )
    ax0.scatter(
        100.0 * (1.0 - r_moead.F_pareto[:, 0]), r_moead.F_pareto[:, 1],
        s=32, c=PALETTE["MOEA/D"], edgecolor="#FFFFFF", linewidth=0.35,
        alpha=0.80, marker="^", zorder=5, label="MOEA/D ($2{,}400$ evals)"
    )
    ax0.scatter(
        100.0 * (1.0 - sa_F_pf[:, 0]), sa_F_pf[:, 1],
        s=44, c=PALETTE["SA-NSGA-II"], edgecolor="#111111", linewidth=0.60,
        marker="D", zorder=6, label="SA-NSGA-II ($220$ evals)"
    )

    # Highlight Knee-Point Operating Condition (exact match to JSON & README)
    xk = sa_X_pf[knee_idx]
    dk = aop.evaluate_diagnostics(xk)
    eta_k = 100.0 * dk["eta_removal"]
    eeo_k = dk["eeo_kwh_m3_order"]
    ax0.scatter(
        [eta_k], [eeo_k], s=230, c="#F4D03F", edgecolor="#111111",
        marker="*", linewidth=1.25, zorder=10, label="Knee-Point Design"
    )
    y_max_aop = max(float(np.max(pf_feas[:, 1])), float(np.max(sa_F_pf[:, 1]))) + 0.018
    ax0.annotate(
        f"Knee-Point Reactor Regime:\n"
        f"$\\mathrm{{pH}} = {xk[0]:.2f}$, $[\\mathrm{{O}}_3]_0 = {xk[1]:.2f}\\;\\mathrm{{mg\\,L^{{-1}}}}$\n"
        f"$\\mathrm{{H}}_2\\mathrm{{O}}_2/\\mathrm{{O}}_3 = {xk[2]:.2f}$, $\\mathrm{{HRT}} = {xk[3]:.2f}\\;\\mathrm{{min}}$\n"
        f"$\\eta = {eta_k:.2f}\\%$, $[\\mathrm{{BrO}}_3^-] = {dk['bromate_ug_L']:.2f}\\;\\mu\\mathrm{{g\\,L^{{-1}}}}$",
        xy=(eta_k, eeo_k),
        xytext=(70.2, 0.103),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.3, mutation_scale=11),
        fontsize=7.8,
        zorder=11,
        bbox=dict(boxstyle="square,pad=0.28", facecolor="#FEF9E7", edgecolor="#C55A11", lw=1.0, alpha=0.96),
    )

    ax0.set_xlim(69.5, 100.4)
    ax0.set_ylim(0.056, y_max_aop)
    ax0.set_title("(a) Peroxone ($\\mathrm{O}_3/\\mathrm{H}_2\\mathrm{O}_2$) AOP Reaction Kinetics", pad=8)
    ax0.set_xlabel("Micropollutant Removal Efficiency $\\eta$ [%]")
    ax0.set_ylabel("Specific Energy Cost $\\mathrm{EE/O}$ [$\\mathrm{kWh\\,m^{-3}\\,order^{-1}}$]")
    ax0.legend(
        loc="upper left",
        fancybox=False,
        edgecolor="#111111",
        framealpha=0.96,
        fontsize=7.8,
    )

    # --- Panel (b): ZDT1 Convex Benchmark & Panel (c): ZDT3 Disconnected Benchmark ---
    for idx_panel, (ProbCls, p_title) in enumerate([
        (lambda: eo.ZDT1(dim=8), "(b) ZDT1 Convex Pareto Front ($D = 8$)"),
        (lambda: eo.ZDT3(dim=8), "(c) ZDT3 Disconnected Pareto Front ($D = 8$)"),
    ], start=1):
        ax = axes[idx_panel]
        _style_continuous_axes(ax)
        prob = ProbCls()
        pf = prob.pareto_front(300)

        res_n = eo.NSGA2(pop_size=80, max_evals=2400, seed=3).optimize(prob, track_hv=False)
        res_m = eo.MOEAD(pop_size=80, max_evals=2400, seed=3).optimize(prob, track_hv=False)
        res_s = eo.SurrogateAssistedNSGA2(
            max_evals=220,
            n_init=40,
            infill_per_iter=12,
            inner_pop_size=52,
            inner_max_evals=728,
            seed=3,
        ).optimize(prob, track_hv=False)

        if "ZDT3" in p_title:
            ax.scatter(pf[:, 0], pf[:, 1], s=10, c="#111111", zorder=3, label="Analytical True PF")
        else:
            ax.plot(pf[:, 0], pf[:, 1], color="#111111", lw=2.2, zorder=3, label="Analytical True PF")

        hv_n = eo.compute_hypervolume(res_n.F_pareto, prob.ref_point)
        hv_m = eo.compute_hypervolume(res_m.F_pareto, prob.ref_point)
        hv_s = eo.compute_hypervolume(res_s.F_pareto, prob.ref_point)

        ax.scatter(
            res_n.F_pareto[:, 0], res_n.F_pareto[:, 1], s=30, c=PALETTE["NSGA-II"],
            edgecolor="#FFFFFF", linewidth=0.35, alpha=0.80, marker="o", zorder=4,
            label=f"NSGA-II ($2{{,}}400$ evals, $\\mathrm{{HV}} = {hv_n:.3f}$)"
        )
        ax.scatter(
            res_m.F_pareto[:, 0], res_m.F_pareto[:, 1], s=32, c=PALETTE["MOEA/D"],
            edgecolor="#FFFFFF", linewidth=0.35, alpha=0.80, marker="^", zorder=5,
            label=f"MOEA/D ($2{{,}}400$ evals, $\\mathrm{{HV}} = {hv_m:.3f}$)"
        )
        ax.scatter(
            res_s.F_pareto[:, 0], res_s.F_pareto[:, 1], s=44, c=PALETTE["SA-NSGA-II"],
            edgecolor="#111111", linewidth=0.60, marker="D", zorder=6,
            label=f"SA-NSGA-II ($220$ evals, $\\mathrm{{HV}} = {hv_s:.3f}$)"
        )

        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + 0.12 * (y_hi - y_lo))
        ax.set_title(p_title, pad=8)
        ax.set_xlabel("Objective $f_1(\\mathbf{x})$")
        ax.set_ylabel("Objective $f_2(\\mathbf{x})$")
        ax.legend(
            loc="upper right",
            fancybox=False,
            edgecolor="#111111",
            framealpha=0.96,
            fontsize=8.0,
        )

    fig.suptitle(
        "Multi-Objective Pareto Frontiers: Bromate-Constrained Environmental Kinetics & ZDT Suite",
        fontsize=12.6, fontweight="bold",
    )
    _finalize_journal_figure(fig)
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
        ("three_bar_truss", "Three-Bar Truss", "Structural Volume $f(\\mathbf{x})$ [$\\mathrm{cm}^3$]"),
        ("tension_compression_spring", "Tension/Compression Spring", "Spring Weight $f(\\mathbf{x})$"),
        ("pressure_vessel", "Cylindrical Pressure Vessel", "Manufacturing Cost $f(\\mathbf{x})$ [USD]"),
    ]
    handlers = [
        ("deb", "Deb's Feasibility\nRules", "#1B4F72", ""),
        ("epsilon", "$\\epsilon$-Constraint\nRelaxation", "#0B6E4F", "//"),
        ("penalty", "Adaptive Dynamic\nPenalty", "#C55A11", "//"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), constrained_layout=True)
    rng = np.random.default_rng(2026)

    for idx, (p_key, p_name, y_label) in enumerate(problems):
        ax = axes[idx]
        _style_categorical_axes(ax)
        prob = eo.get_problem(p_key)
        data_list = []
        colors = []
        hatches = []
        tick_labels = []

        for h_key, h_label, col, h_pat in handlers:
            vals = traj_data[f"ablation__{p_key}__{h_key}__f_vals"]
            data_list.append(vals[np.isfinite(vals)])
            colors.append(col)
            hatches.append(h_pat)
            tick_labels.append(h_label)

        parts = ax.violinplot(
            data_list, positions=[1, 2, 3], widths=0.58, showmeans=False, showextrema=False
        )
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_edgecolor("#111111")
            pc.set_linewidth(0.85)
            pc.set_alpha(0.22)

        bp = ax.boxplot(
            data_list,
            positions=[1, 2, 3],
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#111111", linewidth=2.2),
            whiskerprops=dict(color="#111111", linewidth=1.15),
            capprops=dict(color="#111111", linewidth=1.15),
        )
        for patch, col, h_pat in zip(bp["boxes"], colors, hatches):
            patch.set_facecolor(col)
            patch.set_alpha(0.62)
            patch.set_edgecolor("#111111")
            patch.set_linewidth(0.95)
            if h_pat:
                patch.set_hatch(h_pat)

        # Overlay individual 20-seed results & feasibility badges
        ab_dict = report["constraint_handling_ablation"][prob.name]
        json_names = ["Deb's Feasibility Rules", "Epsilon-Constraint", "Adaptive Penalty"]
        all_concat = np.concatenate(data_list)
        data_span = max(float(np.ptp(all_concat)), 1e-5)

        for pos, (vals, col, j_name) in enumerate(zip(data_list, colors, json_names), start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full_like(vals, pos) + jitter,
                vals,
                s=28,
                c=col,
                edgecolor="#111111",
                linewidth=0.55,
                alpha=0.90,
                zorder=5,
            )
            feas_pct = ab_dict[j_name]["feasibility_rate_pct"]
            med_err = ab_dict[j_name]["rel_error_median_pct"]
            ax.text(
                pos,
                float(np.max(vals)) + 0.045 * data_span,
                f"Feas.: {feas_pct:.0f}%\nMed. $\\Delta$: {med_err:.3f}%",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#111111",
                zorder=7,
                bbox=dict(boxstyle="square,pad=0.24", facecolor="#FFFFFF", edgecolor=col, lw=1.05, alpha=0.95),
            )

        ax.axhline(
            prob.f_opt, color="#B03A2E", linestyle="--", linewidth=1.75, zorder=4,
            label=f"Global Optimum  $f^* = {prob.f_opt:.4f}$"
        )
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(0.45, 3.55)
        ax.set_title(f"({chr(97 + idx)}) {p_name}", pad=8)
        ax.set_ylabel(y_label)
        # Generous upper headroom so legend sits cleanly above all statistical badges
        y_min_val = min(float(np.min(all_concat)), prob.f_opt) - 0.05 * data_span
        y_max_val = float(np.max(all_concat)) + 0.34 * data_span
        ax.set_ylim(y_min_val, y_max_val)
        ax.legend(
            loc="upper left",
            fancybox=False,
            edgecolor="#111111",
            framealpha=0.96,
            fontsize=8.3,
        )

    fig.suptitle(
        "Constraint-Handling Mechanism Ablation Inside HEOA ($20$ Independent Seeds, $1{,}500$ Evaluations)",
        fontsize=12.6, fontweight="bold",
    )
    _finalize_journal_figure(fig)
    out_path = assets_dir / "constraint_handling_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def plot_small_budget(assets_dir: Path, report: dict) -> None:
    """Draw all paired test runs. Lines connect baseline settings for one seed."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), layout="constrained")
    settings = [("NSGA-II", "default"), ("NSGA-II", "tuned"),
                ("MOEA/D", "default"), ("MOEA/D", "tuned"), ("SA-NSGA-II", "fixed")]
    names = {"aop_ozonation": "Simulated water treatment", "zdt1": "ZDT1", "zdt3": "ZDT3"}
    seeds = report["test_seeds"]
    jitter = np.linspace(-0.11, 0.11, len(seeds))
    for ax, (problem, title) in zip(axes, names.items()):
        subset = [r for r in report["test_runs"] if r["problem"] == problem]
        previous = None
        for i, (algorithm, label) in enumerate(settings):
            values = {r["seed"]: r["hv_ratio_pct"] for r in subset
                      if r["algorithm"] == algorithm and r["configuration"] == label}
            y = np.array([values[s] for s in seeds])
            color = PALETTE[algorithm]
            if label == "tuned":
                ax.plot([i - 1 + jitter, i + jitter], [previous, y],
                        color=color, alpha=0.22, linewidth=0.7)
            ax.scatter(i + jitter, y, s=12, color=color, alpha=0.48, zorder=3)
            ax.scatter(i, y.mean(), s=58, marker="D", color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)
            previous = y
        ax.axhline(100, color="#555555", linewidth=0.9, linestyle="--")
        ax.set_xticks(range(5), ["NSGA-II\ndefault", "NSGA-II\ntuned", "MOEA/D\ndefault",
                                 "MOEA/D\ntuned", "SA\nfixed"], fontsize=7.9)
        ax.set_title(title)
        ax.set_ylabel("Hypervolume / reference (%)")
        ax.set_xlim(-0.4, 4.4)
        ax.grid(axis="x", visible=False)
    fig.suptitle("220 true evaluations per run · 20 held-out seeds · diamonds show means")
    fig.savefig(assets_dir / "small_budget_tuning.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    assets_dir = ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"

    with open(results_dir / "engineering_opt_benchmark.json", "r", encoding="utf-8") as fp:
        report = json.load(fp)
    traj_data = np.load(results_dir / "benchmark_trajectories.npz")

    print("[EvoOpt-Figures] Rendering benchmark figures at 300 DPI...")
    plot_three_bar_truss_landscape(assets_dir)
    plot_surrogate_efficiency_convergence(assets_dir, report, traj_data)
    plot_pareto_frontiers(assets_dir, traj_data)
    plot_constraint_handling_ablation(assets_dir, report, traj_data)
    tuning_path = results_dir / "small_budget_tuning.json"
    if tuning_path.exists():
        plot_small_budget(assets_dir, json.loads(tuning_path.read_text(encoding="utf-8")))
    print("[EvoOpt-Figures] Figures generated successfully.")


if __name__ == "__main__":
    main()
