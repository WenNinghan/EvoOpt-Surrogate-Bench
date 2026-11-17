"""
Multi-Objective Optimization Performance Indicators and Pareto Geometry Utilities.

Implements:
  - Non-dominated (Pareto-efficient) filtering
  - Exact 2D Sweep-Line & Recursive Hypervolume (HV)
  - Inverted Generational Distance (IGD)
  - Maximum-Marginal-Utility Knee-Point Identification
"""

from __future__ import annotations

import numpy as np


def is_pareto_efficient(F: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal (non-dominated) rows under minimization.

    A point u dominates v iff all(u <= v) and any(u < v).
    """
    pts = np.asarray(F, dtype=np.float64)
    n_points = pts.shape[0]
    if n_points == 0:
        return np.zeros(0, dtype=bool)

    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not is_efficient[i]:
            continue
        # Keep points that are NOT dominated by pts[i], plus keep pts[i] itself
        dominated = np.all(pts[i] <= pts, axis=1) & np.any(pts[i] < pts, axis=1)
        is_efficient[dominated] = False
        # Deduplicate exact ties after index i
        ties = np.all(pts == pts[i], axis=1)
        ties[: i + 1] = False
        is_efficient[ties] = False
    return is_efficient


def compute_hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute exact Hypervolume (HV) dominated by solution set F w.r.t. ref_point.

    Args:
        F: Objective array of shape (N, M), minimization assumed.
        ref_point: Nadir reference point of shape (M,).

    Returns:
        Scalar hypervolume >= 0.0.
    """
    pts = np.asarray(F, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    if pts.size == 0:
        return 0.0

    # Retain only points strictly dominating the reference point in all objectives
    valid = np.all(pts < ref, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    pts = pts[is_pareto_efficient(pts)]
    n_obj = pts.shape[1]

    if n_obj == 1:
        return float(ref[0] - np.min(pts[:, 0]))

    if n_obj == 2:
        # Exact O(N log N) sweep-line algorithm in 2D
        order = np.argsort(pts[:, 0])
        sorted_pts = pts[order]
        hv = 0.0
        current_f2_upper = ref[1]
        for f1, f2 in sorted_pts:
            if f2 < current_f2_upper:
                hv += (ref[0] - f1) * (current_f2_upper - f2)
                current_f2_upper = f2
        return float(hv)

    # Recursive inclusion-exclusion slicing along the last objective for M >= 3
    order = np.argsort(pts[:, -1])
    sorted_pts = pts[order]
    hv = 0.0
    prev_z = ref[-1]
    for idx in range(len(sorted_pts) - 1, -1, -1):
        z = sorted_pts[idx, -1]
        dz = prev_z - z
        if dz > 0.0:
            slice_hv = compute_hypervolume(sorted_pts[: idx + 1, :-1], ref[:-1])
            hv += dz * slice_hv
            prev_z = z
    return float(hv)


def compute_igd(
    F: np.ndarray,
    pf_ref: np.ndarray,
    normalize: bool = False,
) -> float:
    """Compute Inverted Generational Distance (IGD) from reference Pareto front pf_ref to F.

    IGD(F, P*) = (1 / |P*|) * sum_{v in P*} min_{u in F} ||v - u||_2
    """
    approx = np.asarray(F, dtype=np.float64)
    ref = np.asarray(pf_ref, dtype=np.float64)
    if approx.ndim == 1:
        approx = approx[np.newaxis, :]
    if approx.size == 0 or len(approx) == 0:
        return float("inf")

    if normalize:
        f_min = np.min(ref, axis=0)
        f_max = np.max(ref, axis=0)
        span = np.where((f_max - f_min) > 1e-12, f_max - f_min, 1.0)
        approx = (approx - f_min) / span
        ref = (ref - f_min) / span

    # Pairwise Euclidean distance matrix: shape (|P*|, |F|)
    diff = ref[:, np.newaxis, :] - approx[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    min_dists = np.min(dists, axis=1)
    return float(np.mean(min_dists))


def find_knee_point(F: np.ndarray) -> int:
    """Return index of the knee point on a 2D/MD Pareto front via normalized chord distance.

    Normalizes objectives to [0, 1] and selects the point maximizing perpendicular
    distance below the hyperplane/line connecting extreme anchor points.
    """
    pts = np.asarray(F, dtype=np.float64)
    if len(pts) <= 2:
        return 0

    f_min = np.min(pts, axis=0)
    f_max = np.max(pts, axis=0)
    span = np.where((f_max - f_min) > 1e-12, f_max - f_min, 1.0)
    norm_pts = (pts - f_min) / span

    if pts.shape[1] == 2:
        idx_a = int(np.argmin(norm_pts[:, 0]))
        idx_b = int(np.argmin(norm_pts[:, 1]))
        p_a = norm_pts[idx_a]
        p_b = norm_pts[idx_b]
        line_vec = p_b - p_a
        norm_vec = float(np.linalg.norm(line_vec))
        if norm_vec < 1e-12:
            return 0
        # Signed distance below the chord from (min f1, max f2) to (max f1, min f2)
        # Normal pointing toward utopia (0, 0)
        normal = np.array([line_vec[1], -line_vec[0]]) / norm_vec
        if np.dot(-p_a, normal) < 0:
            normal = -normal
        scores = (norm_pts - p_a) @ normal
        return int(np.argmax(scores))

    # General M-objective fallback: minimum L2 distance to ideal utopia (0, ..., 0)
    utopia_dists = np.linalg.norm(norm_pts, axis=1)
    return int(np.argmin(utopia_dists))
