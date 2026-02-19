"""Island model with per-metric elites for evolutionary optimization."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from synthpersona.metrics.diversity import DiversityMetrics

# Metrics where lower is better
_MINIMIZE_METRICS = {"dispersion", "kl_divergence"}


@dataclass
class GeneratorCandidate:
    source_code: str
    metrics: DiversityMetrics
    iteration: int


class Island:
    def __init__(self, island_id: int) -> None:
        self.island_id = island_id
        # Per-metric elites: metric_name -> best candidate for that metric
        self.elites: dict[str, GeneratorCandidate] = {}

    def submit(self, candidate: GeneratorCandidate) -> list[str]:
        """Submit a candidate and return list of metrics where it became elite."""
        improved: list[str] = []
        metrics_dict = _metrics_to_dict(candidate.metrics)

        for metric_name, value in metrics_dict.items():
            minimize = metric_name in _MINIMIZE_METRICS
            current = self.elites.get(metric_name)

            if current is None:
                self.elites[metric_name] = candidate
                improved.append(metric_name)
            else:
                current_value = _metrics_to_dict(current.metrics)[metric_name]
                if (minimize and value < current_value) or (
                    not minimize and value > current_value
                ):
                    self.elites[metric_name] = candidate
                    improved.append(metric_name)

        return improved

    def get_random_elite(self) -> GeneratorCandidate | None:
        if not self.elites:
            return None
        return random.choice(list(self.elites.values()))

    def average_score(self) -> float:
        """Compute average normalized score across all metric elites."""
        if not self.elites:
            return 0.0
        scores = []
        for metric_name, candidate in self.elites.items():
            value = _metrics_to_dict(candidate.metrics)[metric_name]
            if metric_name in _MINIMIZE_METRICS:
                scores.append(-value)  # Negate so higher is better
            else:
                scores.append(value)
        return sum(scores) / len(scores)

    def reset_from(self, other: Island) -> None:
        """Reset this island's elites from another (extinction event)."""
        self.elites = dict(other.elites)


def _metrics_to_dict(m: DiversityMetrics) -> dict[str, float]:
    return {
        "coverage": m.coverage,
        "convex_hull_volume": m.convex_hull_volume,
        "min_pairwise_distance": m.min_pairwise_distance,
        "mean_pairwise_distance": m.mean_pairwise_distance,
        "dispersion": m.dispersion,
        "kl_divergence": m.kl_divergence,
    }


@dataclass
class IslandPool:
    islands: list[Island] = field(default_factory=list)

    def create_islands(self, n: int) -> None:
        self.islands = [Island(i) for i in range(n)]

    def get_round_robin(self, iteration: int) -> Island:
        return self.islands[iteration % len(self.islands)]

    def run_extinction(self, bottom_pct: float = 0.30, top_pct: float = 0.30) -> None:
        """Reset bottom-performing islands from top-performing ones."""
        sorted_islands = sorted(self.islands, key=lambda i: i.average_score())
        n = len(sorted_islands)
        n_bottom = max(1, int(n * bottom_pct))
        n_top = max(1, int(n * top_pct))

        top_islands = sorted_islands[-n_top:]
        bottom_islands = sorted_islands[:n_bottom]

        for bottom in bottom_islands:
            donor = random.choice(top_islands)
            bottom.reset_from(donor)
