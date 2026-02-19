"""Main evolutionary loop for optimizing persona generators."""

from __future__ import annotations

import json
import random

import numpy as np
import structlog

from synthpersona.config import Settings, get_settings
from synthpersona.evolution.code_loader import load_generator_from_source
from synthpersona.evolution.island import (
    GeneratorCandidate,
    IslandPool,
)
from synthpersona.evolution.mutator import Mutator
from synthpersona.generator.base import PersonaGenerator
from synthpersona.generator.two_stage import TwoStageGenerator
from synthpersona.llm import LLMClient
from synthpersona.metrics.diversity import (
    DiversityMetrics,
    compute_all_metrics,
    normalize_likert,
)
from synthpersona.models.questionnaire import Questionnaire
from synthpersona.simulation.runner import SimulationRunner

logger = structlog.get_logger()


class EvolutionLoop:
    def __init__(
        self,
        training_questionnaires: list[Questionnaire],
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(self.settings)
        self.training_qs = training_questionnaires
        self.mutator = Mutator(client=self.client, settings=self.settings)
        self.runner = SimulationRunner(client=self.client, settings=self.settings)
        self.pool = IslandPool()
        self._calibrated_radii: dict[str, float] = {}

    async def run(self) -> GeneratorCandidate | None:
        """Run the full evolutionary loop."""
        n_islands = self.settings.num_islands
        n_iter = self.settings.evolution_iterations

        # Initialize islands
        self.pool.create_islands(n_islands)

        # Seed with the default TwoStageGenerator
        seed_gen = TwoStageGenerator(client=self.client, settings=self.settings)
        seed_source = seed_gen.get_source_code()

        logger.info("evaluating_seed_generator")
        seed_metrics = await self._evaluate(seed_gen)
        if seed_metrics is None:
            logger.error("seed_evaluation_failed")
            return None

        seed_candidate = GeneratorCandidate(
            source_code=seed_source,
            metrics=seed_metrics,
            iteration=0,
        )

        # Seed all islands with the initial generator
        for island in self.pool.islands:
            island.submit(seed_candidate)

        best_candidate = seed_candidate
        logger.info("evolution_started", islands=n_islands, iterations=n_iter)

        for iteration in range(1, n_iter + 1):
            # Round-robin island selection
            island = self.pool.get_round_robin(iteration)

            # Get a parent from this island
            parent = island.get_random_elite()
            if parent is None:
                continue

            # Generate feedback from a sample evaluation
            feedback = await self._generate_feedback(parent)

            # Mutate
            new_source = await self.mutator.mutate(parent.source_code, feedback)
            if new_source is None:
                logger.debug("mutation_returned_none", iteration=iteration)
                continue

            # Load the mutated generator
            gen = load_generator_from_source(new_source, self.client, self.settings)
            if gen is None:
                logger.debug("load_failed", iteration=iteration)
                continue

            # Evaluate
            metrics = await self._evaluate(gen)
            if metrics is None:
                logger.debug("evaluation_failed", iteration=iteration)
                continue

            candidate = GeneratorCandidate(
                source_code=new_source,
                metrics=metrics,
                iteration=iteration,
            )

            improved = island.submit(candidate)
            if improved:
                logger.info(
                    "new_elite",
                    iteration=iteration,
                    island=island.island_id,
                    improved_metrics=improved,
                )

            # Extinction event
            if iteration % self.settings.extinction_interval == 0 and iteration > 0:
                logger.info("extinction_event", iteration=iteration)
                self.pool.run_extinction(
                    bottom_pct=self.settings.extinction_bottom_pct,
                    top_pct=self.settings.extinction_top_pct,
                )

            # Track overall best
            for isl in self.pool.islands:
                for cand in isl.elites.values():
                    if _overall_score(cand.metrics) > _overall_score(
                        best_candidate.metrics
                    ):
                        best_candidate = cand

            if iteration % 10 == 0:
                logger.info(
                    "progress",
                    iteration=iteration,
                    best_score=round(_overall_score(best_candidate.metrics), 4),
                )

        logger.info(
            "evolution_complete",
            best_score=round(_overall_score(best_candidate.metrics), 4),
            best_iteration=best_candidate.iteration,
        )
        return best_candidate

    async def _evaluate(self, generator: PersonaGenerator) -> DiversityMetrics | None:
        """Evaluate a generator by averaging metrics across training Qs."""
        all_metrics: list[DiversityMetrics] = []

        for q in self.training_qs:
            try:
                personas = await generator.generate(
                    q.context, q.dimensions, self.settings.population_size
                )
                embeddings = await self.runner.simulate(personas, q)

                # Build (N, K) array
                emb_array = np.array([e.to_array(q.dimensions) for e in embeddings])
                emb_norm = normalize_likert(emb_array)

                metrics = compute_all_metrics(
                    emb_norm,
                    n_mc=self.settings.mc_samples,
                    calibration_trials=min(self.settings.calibration_trials, 100),
                )
                all_metrics.append(metrics)
            except Exception:
                logger.exception("evaluation_error", questionnaire=q.name)
                continue

        if not all_metrics:
            return None

        return DiversityMetrics(
            coverage=_mean([m.coverage for m in all_metrics]),
            convex_hull_volume=_mean([m.convex_hull_volume for m in all_metrics]),
            min_pairwise_distance=_mean([m.min_pairwise_distance for m in all_metrics]),
            mean_pairwise_distance=_mean(
                [m.mean_pairwise_distance for m in all_metrics]
            ),
            dispersion=_mean([m.dispersion for m in all_metrics]),
            kl_divergence=_mean([m.kl_divergence for m in all_metrics]),
        )

    async def _generate_feedback(self, candidate: GeneratorCandidate) -> str:
        """Generate feedback by showing sample persona profiles + scores."""
        q = random.choice(self.training_qs)
        try:
            gen = load_generator_from_source(
                candidate.source_code, self.client, self.settings
            )
            if gen is None:
                return "Could not load generator for feedback."

            personas = await gen.generate(
                q.context, q.dimensions, min(5, self.settings.population_size)
            )
            embeddings = await self.runner.simulate(personas, q)

            lines = [f"Sample evaluation on '{q.name}':"]
            for p, e in zip(personas, embeddings, strict=True):
                scores_str = json.dumps({k: round(v, 2) for k, v in e.scores.items()})
                lines.append(f"- {p.name}: {scores_str}")
                lines.append(f"  Description: {p.full_description[:200]}...")

            return "\n".join(lines)
        except Exception:
            return "Feedback generation failed."


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _overall_score(m: DiversityMetrics) -> float:
    """Compute a single overall score (higher is better)."""
    return (
        m.coverage
        + m.convex_hull_volume
        + m.min_pairwise_distance
        + m.mean_pairwise_distance
        - m.dispersion
        - m.kl_divergence
    )
