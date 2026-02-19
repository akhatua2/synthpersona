"""Command-line interface for synthpersona."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import numpy as np
import structlog

from synthpersona.config import get_settings

logger = structlog.get_logger()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Synthpersona - Generate diverse synthetic personas at scale."""


@cli.command()
@click.argument("description")
@click.option(
    "-o", "--output", type=click.Path(), default=None, help="Output JSON path"
)
def generate_questionnaire(description: str, output: str | None) -> None:
    """Generate a questionnaire from a short description."""
    from synthpersona.llm import LLMClient
    from synthpersona.questionnaire.generator import QuestionnaireGenerator

    async def _run() -> None:
        settings = get_settings()
        client = LLMClient(settings)
        gen = QuestionnaireGenerator(client=client, settings=settings)

        click.echo(f"Generating questionnaire for: {description}")
        q = await gen.generate(description)

        data = q.model_dump()
        if output:
            Path(output).write_text(json.dumps(data, indent=2))
            click.echo(f"Saved to {output}")
        else:
            click.echo(json.dumps(data, indent=2))

    asyncio.run(_run())


@cli.command()
@click.argument("questionnaire_path", type=click.Path(exists=True))
@click.option("-n", "--num-personas", type=int, default=None)
@click.option(
    "-o", "--output", type=click.Path(), default=None, help="Output JSON path"
)
def generate_personas(
    questionnaire_path: str, num_personas: int | None, output: str | None
) -> None:
    """Generate personas for a questionnaire."""
    from synthpersona.generator.two_stage import TwoStageGenerator
    from synthpersona.llm import LLMClient
    from synthpersona.models.questionnaire import Questionnaire

    async def _run() -> None:
        settings = get_settings()
        if num_personas is not None:
            settings.population_size = num_personas
        client = LLMClient(settings)

        q_data = json.loads(Path(questionnaire_path).read_text())
        q = Questionnaire(**q_data)

        gen = TwoStageGenerator(client=client, settings=settings)
        click.echo(f"Generating {settings.population_size} personas for '{q.name}'...")
        personas = await gen.generate(q.context, q.dimensions, settings.population_size)

        data = [p.model_dump() for p in personas]
        if output:
            Path(output).write_text(json.dumps(data, indent=2))
            click.echo(f"Saved {len(personas)} personas to {output}")
        else:
            click.echo(json.dumps(data, indent=2))

    asyncio.run(_run())


@cli.command()
@click.argument("questionnaire_path", type=click.Path(exists=True))
@click.argument("personas_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output", type=click.Path(), default=None, help="Output JSON path"
)
def simulate(questionnaire_path: str, personas_path: str, output: str | None) -> None:
    """Run simulation: personas answer questionnaire items."""
    from synthpersona.llm import LLMClient
    from synthpersona.models.persona import Persona
    from synthpersona.models.questionnaire import Questionnaire
    from synthpersona.simulation.runner import SimulationRunner

    async def _run() -> None:
        settings = get_settings()
        client = LLMClient(settings)

        q = Questionnaire(**json.loads(Path(questionnaire_path).read_text()))
        personas_data = json.loads(Path(personas_path).read_text())
        personas = [Persona(**p) for p in personas_data]

        runner = SimulationRunner(client=client, settings=settings)
        click.echo(f"Simulating {len(personas)} personas on '{q.name}'...")
        embeddings = await runner.simulate(personas, q)

        data = [e.model_dump() for e in embeddings]
        if output:
            Path(output).write_text(json.dumps(data, indent=2))
            click.echo(f"Saved embeddings to {output}")
        else:
            click.echo(json.dumps(data, indent=2))

    asyncio.run(_run())


@cli.command()
@click.argument("embeddings_path", type=click.Path(exists=True))
@click.option("--dimensions", "-d", multiple=True, required=True)
def metrics(embeddings_path: str, dimensions: tuple[str, ...]) -> None:
    """Compute diversity metrics on population embeddings."""
    from synthpersona.metrics.diversity import compute_all_metrics, normalize_likert
    from synthpersona.models.persona import PopulationEmbedding

    data = json.loads(Path(embeddings_path).read_text())
    embeddings = [PopulationEmbedding(**e) for e in data]
    dims = list(dimensions)

    emb_array = np.array([e.to_array(dims) for e in embeddings])
    emb_norm = normalize_likert(emb_array)

    settings = get_settings()
    result = compute_all_metrics(
        emb_norm,
        n_mc=settings.mc_samples,
        calibration_trials=settings.calibration_trials,
    )

    click.echo("Diversity Metrics:")
    click.echo(f"  Coverage:               {result.coverage:.4f}")
    click.echo(f"  Convex Hull Volume:     {result.convex_hull_volume:.4f}")
    click.echo(f"  Min Pairwise Distance:  {result.min_pairwise_distance:.4f}")
    click.echo(f"  Mean Pairwise Distance: {result.mean_pairwise_distance:.4f}")
    click.echo(f"  Dispersion:             {result.dispersion:.4f}")
    click.echo(f"  KL Divergence:          {result.kl_divergence:.4f}")


@cli.command()
@click.option("--iterations", "-i", type=int, default=None, help="Override iterations")
@click.option("--islands", type=int, default=None, help="Override number of islands")
@click.option(
    "--questionnaires",
    "-q",
    type=click.Path(exists=True),
    multiple=True,
    help="Training questionnaire JSON files",
)
def evolve(
    iterations: int | None,
    islands: int | None,
    questionnaires: tuple[str, ...],
) -> None:
    """Run the evolutionary optimization loop."""
    from synthpersona.evolution.loop import EvolutionLoop
    from synthpersona.llm import LLMClient
    from synthpersona.models.questionnaire import Questionnaire
    from synthpersona.questionnaire.examples import EXAMPLE_QUESTIONNAIRES

    async def _run() -> None:
        settings = get_settings()
        if iterations is not None:
            settings.evolution_iterations = iterations
        if islands is not None:
            settings.num_islands = islands

        client = LLMClient(settings)

        if questionnaires:
            training_qs = [
                Questionnaire(**json.loads(Path(p).read_text())) for p in questionnaires
            ]
        else:
            training_qs = EXAMPLE_QUESTIONNAIRES
            click.echo("Using built-in example questionnaires for training.")

        click.echo(
            f"Starting evolution: {settings.evolution_iterations} iterations, "
            f"{settings.num_islands} islands, "
            f"{len(training_qs)} questionnaires"
        )

        loop = EvolutionLoop(
            training_questionnaires=training_qs,
            client=client,
            settings=settings,
        )
        best = await loop.run()

        if best:
            click.echo(f"\nBest generator found at iteration {best.iteration}")
            output_path = Path("best_generator.py")
            output_path.write_text(best.source_code)
            click.echo(f"Saved to {output_path}")

    asyncio.run(_run())


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
