"""Run full evaluation pipeline across the 10 test-set questionnaires.

Generates 25 personas per questionnaire (250 total), simulates their responses,
and computes diversity metrics for each.

Usage:
    uv run python scripts/run_evaluation.py
    uv run python scripts/run_evaluation.py --set test
    uv run python scripts/run_evaluation.py --set training --limit 5
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import numpy as np

from synthpersona.config import get_settings
from synthpersona.generator.two_stage import TwoStageGenerator
from synthpersona.llm import LLMClient
from synthpersona.metrics.diversity import calibrate_radius, compute_all_metrics, normalize_likert
from synthpersona.metrics.plots import plot_all_metrics
from synthpersona.models.questionnaire import Questionnaire
from synthpersona.simulation.runner import SimulationRunner

QUESTIONNAIRE_DIR = Path("questionnaires")
OUTPUT_DIR = Path("eval_results")

# Questionnaire names by split (from the paper's Appendix A.1)
TEST_SET = [
    "agi_job_displacement_global_2035",
    "camelot_chivalry_quests",
    "cold_war_anxiety_us_1962",
    "viking_warriors_valhalla",
    "generalized_trust_in_the_salem_witch_trials_1692",
    "nomadic_values_mongolia_2023",
    "meaning_of_life_2030",
    "agi_wealth_inequality_revolution_2040",
    "climate_anxiety_coastal_au_2024",
    "british_empire_attitudes_uk_1900",
]

TRAINING_SET = [
    "american_conspiracy_theories_2024",
    "health_tech_wearables_2030",
    "gentrification_brooklyn_2022",
    "elderly_rural_japan_2010",
    "plant_based_diets_india_2025",
    "heian_japan_courtiers_1000ad",
    "ubi_attitudes_california_2026",
    "swe_ai_assistants_2024",
    "trojan_war_achaen_morale_1184bc",
    "factory_automation_china_2025",
    "organic_farmers_kenya_2023",
    "industrial_revolution_workers_uk_1850",
    "professional_athletes_gender_equality_europe_2025",
    "gig_economy_ethics_london_2023",
    "financial_literacy_brazil_students_2024",
    "romans_reactions_to_murder_of_julius_caesar_44_bc",
    "alaska_oil_environment_2025",
    "parisian_artists_future_2026",
    "immigrant_integration_canada_2023",
    "greek_underworld_shades",
    "millennial_parenting_us_2025",
    "asi_human_creativity_2060",
    "high_school_students_italy_2016",
    "genz_social_media_politics_2025",
    "healthcare_covid_stress_italy_2021",
    "scifi_authors_future_ai_space_2024",
    "esports_mental_health_sk_2024",
    "silk_road_merchants_samarkand_750ad",
    "mali_empire_scholars_timbuktu_1350ad",
    "ww2_civilian_sentiment_germany_1943",
]

VALIDATION_SET = [
    "ai_tech_stock_sentiment_2025",
    "social_media_politics_europe_2026",
    "inca_commoners_mita_1500ad",
    "athenian_piety_olympian_gods_430bc",
    "sleep_quality_2025",
    "ai_companionship_integration_2070",
    "asi_existential_dread_2050",
    "ancient_egypt_akhenaten_reforms_1340bc",
    "german_energy_policy_2025",
    "moral_dilemmas_global_2045",
]


async def evaluate_questionnaire(
    name: str,
    client: LLMClient,
    generator: TwoStageGenerator,
    runner: SimulationRunner,
    n: int,
) -> dict | None:
    """Run full pipeline for one questionnaire: generate → simulate → metrics."""
    q_path = QUESTIONNAIRE_DIR / f"{name}.json"
    if not q_path.exists():
        print(f"  SKIP {name} (questionnaire not found)")
        return None

    q = Questionnaire(**json.loads(q_path.read_text()))

    # Check for cached results
    result_path = OUTPUT_DIR / f"{name}.json"
    if result_path.exists():
        print(f"  CACHED {name}")
        return json.loads(result_path.read_text())

    try:
        # Step 1: Generate personas
        print(f"  [{name}] Generating {n} personas...")
        personas = await generator.generate(q.context, q.dimensions, n)

        # Step 2: Simulate responses
        print(f"  [{name}] Simulating {len(personas)} personas × {len(q.questions)} questions...")
        embeddings = await runner.simulate(personas, q)
    except Exception as e:
        print(f"  FAIL [{name}] {e}")
        return None

    # Step 3: Compute diversity metrics
    emb_array = np.array([e.to_array(q.dimensions) for e in embeddings])
    emb_norm = normalize_likert(emb_array)
    radius = calibrate_radius(len(emb_norm), emb_norm.shape[1])
    metrics = compute_all_metrics(emb_norm, radius=radius)

    # Step 4: Generate metric plots
    plot_path = OUTPUT_DIR / f"{name}_metrics.png"
    plot_all_metrics(emb_norm, radius, plot_path, dim_names=q.dimensions, questionnaire_name=name)
    print(f"  [{name}] Saved plots to {plot_path}")

    result = {
        "questionnaire": name,
        "dimensions": q.dimensions,
        "num_personas": len(personas),
        "num_questions": len(q.questions),
        "metrics": {
            "coverage": metrics.coverage,
            "convex_hull_volume": metrics.convex_hull_volume,
            "min_pairwise_distance": metrics.min_pairwise_distance,
            "mean_pairwise_distance": metrics.mean_pairwise_distance,
            "dispersion": metrics.dispersion,
            "kl_divergence": metrics.kl_divergence,
        },
        "personas": [p.model_dump() for p in personas],
        "embeddings": [e.model_dump() for e in embeddings],
    }

    # Save per-questionnaire result
    result_path.write_text(json.dumps(result, indent=2))
    print(
        f"  [{name}] Done — "
        f"coverage={metrics.coverage:.3f}, "
        f"hull={metrics.convex_hull_volume:.3f}, "
        f"min_dist={metrics.min_pairwise_distance:.3f}, "
        f"mean_dist={metrics.mean_pairwise_distance:.3f}, "
        f"dispersion={metrics.dispersion:.3f}, "
        f"kl={metrics.kl_divergence:.3f}"
    )
    return result


@click.command()
@click.option(
    "--set",
    "which_set",
    type=click.Choice(["test", "training", "validation", "all"]),
    default=None,
    help="Which questionnaire set to evaluate",
)
@click.option("-q", "--questionnaire", "questionnaire_names", multiple=True, help="Specific questionnaire names")
@click.option("--limit", type=int, default=None, help="Max questionnaires to evaluate")
@click.option("-n", "--num-personas", type=int, default=None, help="Personas per questionnaire")
def main(which_set: str | None, questionnaire_names: tuple[str, ...], limit: int | None, num_personas: int | None) -> None:
    """Run full evaluation pipeline."""

    async def _run() -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        settings = get_settings()
        n = num_personas or settings.population_size
        client = LLMClient(settings)
        generator = TwoStageGenerator(client=client, settings=settings)
        runner = SimulationRunner(client=client, settings=settings)

        if questionnaire_names:
            names = list(questionnaire_names)
        else:
            sets = {
                "test": TEST_SET,
                "training": TRAINING_SET,
                "validation": VALIDATION_SET,
            }
            which = which_set or "test"
            if which == "all":
                names = TEST_SET + TRAINING_SET + VALIDATION_SET
            else:
                names = sets[which]

        if limit:
            names = names[:limit]

        print(f"\nEvaluating {len(names)} questionnaires, {n} personas each")
        print(f"Total personas: {len(names) * n}")
        print(f"={'=' * 60}\n")

        results = []
        for name in names:
            result = await evaluate_questionnaire(name, client, generator, runner, n)
            if result:
                results.append(result)

        # Print summary table
        if results:
            print(f"\n{'=' * 90}")
            print(f"{'Questionnaire':<50} {'Cov':>6} {'Hull':>6} {'MinD':>6} {'MeanD':>6} {'Disp':>6} {'KL':>6}")
            print(f"{'-' * 90}")
            for r in results:
                m = r["metrics"]
                print(
                    f"{r['questionnaire']:<50} "
                    f"{m['coverage']:>6.3f} "
                    f"{m['convex_hull_volume']:>6.3f} "
                    f"{m['min_pairwise_distance']:>6.3f} "
                    f"{m['mean_pairwise_distance']:>6.3f} "
                    f"{m['dispersion']:>6.3f} "
                    f"{m['kl_divergence']:>6.3f}"
                )

            # Averages
            avg = {
                k: np.mean([r["metrics"][k] for r in results])
                for k in results[0]["metrics"]
            }
            print(f"{'-' * 90}")
            print(
                f"{'AVERAGE':<50} "
                f"{avg['coverage']:>6.3f} "
                f"{avg['convex_hull_volume']:>6.3f} "
                f"{avg['min_pairwise_distance']:>6.3f} "
                f"{avg['mean_pairwise_distance']:>6.3f} "
                f"{avg['dispersion']:>6.3f} "
                f"{avg['kl_divergence']:>6.3f}"
            )
            print(f"{'=' * 90}")

            # Save summary
            summary_path = OUTPUT_DIR / "summary.json"
            summary = {
                "set": which_set,
                "num_questionnaires": len(results),
                "personas_per_questionnaire": n,
                "total_personas": len(results) * n,
                "average_metrics": {k: float(v) for k, v in avg.items()},
                "per_questionnaire": [
                    {"questionnaire": r["questionnaire"], "metrics": r["metrics"]}
                    for r in results
                ],
            }
            summary["usage"] = {
                "total_calls": client.total_calls,
                "total_input_tokens": client.total_input_tokens,
                "total_output_tokens": client.total_output_tokens,
                "total_cost_usd": client.total_cost,
            }
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"\nSummary saved to {summary_path}")

        # Always print usage
        print(
            f"\nLLM Usage: {client.total_calls} calls, "
            f"{client.total_input_tokens:,} input + {client.total_output_tokens:,} output tokens, "
            f"${client.total_cost:.4f} USD"
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
