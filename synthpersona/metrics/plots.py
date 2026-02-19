"""Visualization of diversity metrics (inspired by paper Figure 6, Appendix C)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull
from scipy.stats.qmc import Sobol


def _project_2d(embeddings: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Project to 2D for visualization. Returns (points_2d, xlabel, ylabel)."""
    if embeddings.shape[1] == 2:
        return embeddings, "Dimension 1", "Dimension 2"
    # Use first two dimensions for 3+ dim data
    return embeddings[:, :2], "Dimension 1", "Dimension 2"


def plot_convex_hull(
    embeddings: np.ndarray,
    ax: plt.Axes,
    dim_names: list[str] | None = None,
) -> None:
    """Plot convex hull volume visualization."""
    pts, xl, yl = _project_2d(embeddings)
    if dim_names and len(dim_names) >= 2:
        xl, yl = dim_names[0], dim_names[1]

    try:
        hull = ConvexHull(pts)
        # Fill hull
        for simplex in hull.simplices:
            ax.plot(pts[simplex, 0], pts[simplex, 1], "k-", alpha=0.3, linewidth=0.8)
        hull_pts = pts[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon
        ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.15, color="teal")
        volume = hull.volume
    except Exception:
        volume = 0.0

    # Draw edges between all points (wireframe effect)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < 0.6:  # Only draw nearby connections
                ax.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    "teal",
                    alpha=0.1,
                    linewidth=0.5,
                )

    ax.scatter(pts[:, 0], pts[:, 1], c="darkslategray", s=40, zorder=5, edgecolors="white", linewidth=0.5)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"Convex Hull Volume = {volume:.4f}")


def plot_coverage(
    embeddings: np.ndarray,
    radius: float,
    ax: plt.Axes,
    n_mc: int = 5000,
    dim_names: list[str] | None = None,
) -> None:
    """Plot Monte Carlo coverage estimate."""
    pts, xl, yl = _project_2d(embeddings)
    if dim_names and len(dim_names) >= 2:
        xl, yl = dim_names[0], dim_names[1]

    k = pts.shape[1]
    mc_points = np.random.uniform(0, 1, size=(n_mc, k))

    # Check which MC points are covered
    diffs = mc_points[:, np.newaxis, :] - pts[np.newaxis, :, :]
    min_dists = np.min(np.sqrt(np.sum(diffs**2, axis=2)), axis=1)
    covered = min_dists <= radius

    coverage = np.mean(covered)

    ax.scatter(
        mc_points[covered, 0], mc_points[covered, 1],
        c="green", s=2, alpha=0.4, label="Covered",
    )
    ax.scatter(
        mc_points[~covered, 0], mc_points[~covered, 1],
        c="red", s=2, alpha=0.4, label="Not covered",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"Coverage = {coverage:.3f} (r={radius:.3f})")
    ax.legend(markerscale=4, fontsize=7, loc="upper right")


def plot_dispersion(
    embeddings: np.ndarray,
    ax: plt.Axes,
    n_mc: int = 5000,
    dim_names: list[str] | None = None,
) -> None:
    """Plot dispersion (largest empty ball)."""
    pts, xl, yl = _project_2d(embeddings)
    if dim_names and len(dim_names) >= 2:
        xl, yl = dim_names[0], dim_names[1]

    k = pts.shape[1]
    mc_points = np.random.uniform(0, 1, size=(n_mc, k))

    # Find min distance from each MC point to population
    diffs = mc_points[:, np.newaxis, :] - pts[np.newaxis, :, :]
    min_dists = np.min(np.sqrt(np.sum(diffs**2, axis=2)), axis=1)

    # Find the farthest MC point (center of largest empty ball)
    worst_idx = np.argmax(min_dists)
    dispersion = min_dists[worst_idx]
    center = mc_points[worst_idx]

    # Find nearest data point to the worst MC point
    nearest_idx = np.argmin(np.linalg.norm(pts - center, axis=1))

    ax.scatter(pts[:, 0], pts[:, 1], c="navy", s=40, zorder=5, label="Personas", edgecolors="white", linewidth=0.5)
    ax.scatter(
        mc_points[:, 0], mc_points[:, 1],
        c="gray", s=1, alpha=0.15, label="Search points",
    )

    # Draw the largest empty circle
    circle = Circle(center, dispersion, fill=False, color="red", linewidth=2, linestyle="-", label=f"Dispersion = {dispersion:.3f}")
    ax.add_patch(circle)

    ax.scatter(center[0], center[1], c="red", s=100, marker="x", zorder=6, linewidths=2, label="Farthest point")
    ax.plot(
        [center[0], pts[nearest_idx, 0]],
        [center[1], pts[nearest_idx, 1]],
        "r--", linewidth=1.5, alpha=0.7,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"Dispersion = {dispersion:.3f}")
    ax.legend(fontsize=6, loc="upper right")


def plot_kl_divergence(
    embeddings: np.ndarray,
    ax: plt.Axes,
    dim_names: list[str] | None = None,
) -> None:
    """Plot KL divergence: actual vs ideal Sobol distribution."""
    pts, xl, yl = _project_2d(embeddings)
    if dim_names and len(dim_names) >= 2:
        xl, yl = dim_names[0], dim_names[1]

    k = pts.shape[1]
    n = len(pts)

    # Generate Sobol reference
    sampler = Sobol(d=k, scramble=True)
    m = 1
    while m < n:
        m *= 2
    ref = sampler.random(m)[:n]

    ax.scatter(
        ref[:, 0], ref[:, 1],
        facecolors="none", edgecolors="green", s=50, linewidths=1.2,
        label="Ideal (Sobol)", zorder=4,
    )
    ax.scatter(
        pts[:, 0], pts[:, 1],
        facecolors="none", edgecolors="blue", s=30, linewidths=1.2,
        label="Actual", zorder=5,
    )

    # Compute KL for title
    n_bins = 20
    kl_sum = 0.0
    for dim in range(k):
        bins = np.linspace(0, 1, n_bins + 1)
        p_hist, _ = np.histogram(pts[:, dim], bins=bins, density=True)
        q_hist, _ = np.histogram(ref[:, dim], bins=bins, density=True)
        eps = 1e-10
        p_hist = (p_hist + eps) / (p_hist + eps).sum()
        q_hist = (q_hist + eps) / (q_hist + eps).sum()
        kl_sum += float(np.sum(p_hist * np.log(p_hist / q_hist)))
    kl = kl_sum / k

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"KL Divergence = {kl:.3f}")
    ax.legend(fontsize=7, loc="upper right")


def plot_mean_pairwise_distance(
    embeddings: np.ndarray,
    ax: plt.Axes,
) -> None:
    """Plot histogram of pairwise distances with mean line."""
    diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
    all_dists = np.sqrt(np.sum(diff**2, axis=2))
    # Upper triangle only (exclude diagonal and duplicates)
    n = len(embeddings)
    upper_dists = all_dists[np.triu_indices(n, k=1)]

    mean_dist = np.mean(upper_dists)

    ax.hist(upper_dists, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(mean_dist, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_dist:.3f}")
    ax.set_xlabel("Pairwise Distance")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Mean Pairwise Distance = {mean_dist:.3f}")
    ax.legend(fontsize=8)


def plot_min_pairwise_distance(
    embeddings: np.ndarray,
    ax: plt.Axes,
    dim_names: list[str] | None = None,
) -> None:
    """Plot minimum pairwise distance with closest pair highlighted."""
    pts, xl, yl = _project_2d(embeddings)
    if dim_names and len(dim_names) >= 2:
        xl, yl = dim_names[0], dim_names[1]

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)

    min_dist = np.min(dists)
    i, j = np.unravel_index(np.argmin(dists), dists.shape)

    ax.scatter(pts[:, 0], pts[:, 1], c="navy", s=40, zorder=5, label="Personas", edgecolors="white", linewidth=0.5)
    ax.scatter(
        pts[[i, j], 0], pts[[i, j], 1],
        c="red", s=80, zorder=6, label="Closest pair", edgecolors="white", linewidth=0.5,
    )
    ax.plot(
        [pts[i, 0], pts[j, 0]],
        [pts[i, 1], pts[j, 1]],
        "r--", linewidth=2, label=f"Min dist = {min_dist:.3f}",
    )

    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"Min Pairwise Distance = {min_dist:.3f}")
    ax.legend(fontsize=7, loc="upper right")


def plot_all_metrics(
    embeddings: np.ndarray,
    radius: float,
    output_path: Path | str,
    dim_names: list[str] | None = None,
    questionnaire_name: str = "",
) -> None:
    """Generate a 2x3 figure with all 6 metric visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Diversity Metrics — {questionnaire_name}" if questionnaire_name else "Diversity Metrics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plot_convex_hull(embeddings, axes[0, 0], dim_names)
    plot_coverage(embeddings, radius, axes[0, 1], dim_names=dim_names)
    plot_dispersion(embeddings, axes[0, 2], dim_names=dim_names)
    plot_kl_divergence(embeddings, axes[1, 0], dim_names=dim_names)
    plot_mean_pairwise_distance(embeddings, axes[1, 1])
    plot_min_pairwise_distance(embeddings, axes[1, 2], dim_names)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
