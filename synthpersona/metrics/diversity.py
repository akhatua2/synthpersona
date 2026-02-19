"""Six diversity metrics + Monte Carlo coverage calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats.qmc import Sobol


@dataclass
class DiversityMetrics:
    coverage: float
    convex_hull_volume: float
    min_pairwise_distance: float
    mean_pairwise_distance: float
    dispersion: float
    kl_divergence: float


def normalize_likert(
    embeddings: np.ndarray,
    likert_min: float = 1.0,
    likert_max: float = 5.0,
) -> np.ndarray:
    """Normalize Likert scores from [min, max] to [0, 1]."""
    return (embeddings - likert_min) / (likert_max - likert_min)


def calibrate_radius(
    n: int,
    k: int,
    target_coverage: float = 0.99,
    n_trials: int = 1000,
    n_mc: int = 10_000,
) -> float:
    """Calibrate the coverage radius using Sobol reference populations.

    For each trial, generate a Sobol population of size n in k dimensions,
    then find the smallest radius where target_coverage fraction of MC
    points are covered. Average over n_trials.
    """
    radii = []
    for _ in range(n_trials):
        # Generate a Sobol reference population
        sampler = Sobol(d=k, scramble=True)
        m = 1
        while m < n:
            m *= 2
        ref_pop = sampler.random(m)[:n]

        # Generate MC test points uniformly in [0,1]^k
        mc_points = np.random.uniform(0, 1, size=(n_mc, k))

        # Compute distances from each MC point to nearest population member
        min_dists = _min_distances_to_set(mc_points, ref_pop)

        # Find radius for target coverage
        sorted_dists = np.sort(min_dists)
        idx = int(np.ceil(target_coverage * n_mc)) - 1
        radii.append(sorted_dists[idx])

    return float(np.mean(radii))


def _min_distances_to_set(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute minimum Euclidean distance from each query point to reference."""
    # query: (Q, K), reference: (R, K) -> distances: (Q,)
    # Use chunked computation to avoid memory issues
    chunk_size = 1000
    min_dists = np.full(len(query), np.inf)
    for i in range(0, len(query), chunk_size):
        chunk = query[i : i + chunk_size]
        # (chunk_size, 1, K) - (1, R, K) -> (chunk_size, R)
        diffs = chunk[:, np.newaxis, :] - reference[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        min_dists[i : i + chunk_size] = np.min(dists, axis=1)
    return min_dists


def _coverage(
    embeddings: np.ndarray,
    radius: float,
    n_mc: int = 10_000,
) -> float:
    """Estimate coverage via Monte Carlo sampling."""
    k = embeddings.shape[1]
    mc_points = np.random.uniform(0, 1, size=(n_mc, k))
    min_dists = _min_distances_to_set(mc_points, embeddings)
    return float(np.mean(min_dists <= radius))


def _convex_hull_volume(embeddings: np.ndarray) -> float:
    """Compute convex hull volume of the population embeddings."""
    n, k = embeddings.shape
    if n <= k:
        return 0.0
    try:
        hull = ConvexHull(embeddings)
        return float(hull.volume)
    except Exception:
        return 0.0


def _pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Compute all pairwise Euclidean distances."""
    diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return dists


def _min_pairwise_distance(embeddings: np.ndarray) -> float:
    """Minimum pairwise Euclidean distance (excluding self-pairs)."""
    dists = _pairwise_distances(embeddings)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def _mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """Mean pairwise Euclidean distance."""
    dists = _pairwise_distances(embeddings)
    n = len(embeddings)
    # Sum upper triangle, divide by n*(n-1)/2
    upper = np.triu(dists, k=1)
    return float(np.sum(upper) / (n * (n - 1) / 2))


def _dispersion(
    embeddings: np.ndarray,
    n_mc: int = 10_000,
) -> float:
    """Dispersion: max of min-distances from random points to population.

    This is the radius of the largest empty ball.
    """
    k = embeddings.shape[1]
    mc_points = np.random.uniform(0, 1, size=(n_mc, k))
    min_dists = _min_distances_to_set(mc_points, embeddings)
    return float(np.max(min_dists))


def _kl_divergence(
    embeddings: np.ndarray,
    n_reference: int = 1000,
    n_bins: int = 20,
) -> float:
    """KL divergence between population and Sobol reference, averaged.

    Uses histogram-based estimation, averaged over n_reference Sobol
    reference distributions.
    """
    k = embeddings.shape[1]
    kl_values = []

    for _ in range(n_reference):
        sampler = Sobol(d=k, scramble=True)
        m = 1
        while m < len(embeddings):
            m *= 2
        ref = sampler.random(m)[: len(embeddings)]

        kl_sum = 0.0
        for dim in range(k):
            # Histogram for both distributions along this dimension
            bins = np.linspace(0, 1, n_bins + 1)
            p_hist, _ = np.histogram(embeddings[:, dim], bins=bins, density=True)
            q_hist, _ = np.histogram(ref[:, dim], bins=bins, density=True)

            # Add small epsilon to avoid log(0)
            eps = 1e-10
            p_hist = p_hist + eps
            q_hist = q_hist + eps

            # Normalize
            p_hist = p_hist / p_hist.sum()
            q_hist = q_hist / q_hist.sum()

            kl_sum += float(np.sum(p_hist * np.log(p_hist / q_hist)))

        kl_values.append(kl_sum / k)

    return float(np.mean(kl_values))


def compute_all_metrics(
    embeddings: np.ndarray,
    radius: float | None = None,
    n_mc: int = 10_000,
    calibration_trials: int = 1000,
    coverage_target: float = 0.99,
) -> DiversityMetrics:
    """Compute all 6 diversity metrics on normalized embeddings.

    Args:
        embeddings: (N, K) array of population embeddings, already
            normalized to [0, 1].
        radius: Pre-calibrated coverage radius. If None, will calibrate.
        n_mc: Number of Monte Carlo samples for coverage and dispersion.
        calibration_trials: Number of trials for radius calibration.
        coverage_target: Target coverage for radius calibration.
    """
    n, k = embeddings.shape

    if radius is None:
        radius = calibrate_radius(n, k, coverage_target, calibration_trials, n_mc)

    return DiversityMetrics(
        coverage=_coverage(embeddings, radius, n_mc),
        convex_hull_volume=_convex_hull_volume(embeddings),
        min_pairwise_distance=_min_pairwise_distance(embeddings),
        mean_pairwise_distance=_mean_pairwise_distance(embeddings),
        dispersion=_dispersion(embeddings, n_mc),
        kl_divergence=_kl_divergence(embeddings),
    )
