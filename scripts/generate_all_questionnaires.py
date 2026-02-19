"""Generate all 50 questionnaires from the paper (Appendix A.1).

The 3 hardcoded questionnaires are saved directly from examples.py.
The remaining 47 are generated via LLM from their short descriptions.

Usage:
    uv run python scripts/generate_all_questionnaires.py
    uv run python scripts/generate_all_questionnaires.py --set training
    uv run python scripts/generate_all_questionnaires.py --set test
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from synthpersona.config import get_settings
from synthpersona.llm import LLMClient
from synthpersona.questionnaire.examples import EXAMPLE_QUESTIONNAIRES
from synthpersona.questionnaire.generator import QuestionnaireGenerator

OUTPUT_DIR = Path("questionnaires")

# All 50 questionnaire descriptions from paper Appendix A.1,
# organized by split. The key is the paper's identifier, the value
# is the short description fed to the questionnaire generator.

TRAINING_SET: dict[str, str] = {
    "american_conspiracy_theories_2024": "HARDCODED",
    "health_tech_wearables_2030": "Attitudes toward health tech wearables and biometric monitoring in 2030",
    "gentrification_brooklyn_2022": "Gentrification and neighborhood change in Brooklyn, New York in 2022",
    "elderly_rural_japan_2010": "HARDCODED",
    "plant_based_diets_india_2025": "Plant-based diets and vegetarianism attitudes in India in 2025",
    "heian_japan_courtiers_1000ad": "Court life and social ambitions of Heian-era Japanese courtiers around 1000 AD",
    "ubi_attitudes_california_2026": "Attitudes toward universal basic income among Californians in 2026",
    "swe_ai_assistants_2024": "Software engineers' attitudes toward AI coding assistants in 2024",
    "trojan_war_achaen_morale_1184bc": "Morale and fighting spirit of Achaean warriors during the Trojan War around 1184 BC",
    "factory_automation_china_2025": "Factory workers' reactions to automation in Chinese manufacturing in 2025",
    "organic_farmers_kenya_2023": "Organic farming attitudes and challenges among Kenyan farmers in 2023",
    "industrial_revolution_workers_uk_1850": "Industrial revolution factory workers in the United Kingdom around 1850",
    "professional_athletes_gender_equality_europe_2025": "Professional athletes' views on gender equality in European sports in 2025",
    "gig_economy_ethics_london_2023": "Ethics and fairness perceptions in the gig economy in London in 2023",
    "financial_literacy_brazil_students_2024": "Financial literacy and money attitudes among Brazilian university students in 2024",
    "romans_reactions_to_murder_of_julius_caesar_44_bc": "Roman citizens' reactions to the assassination of Julius Caesar in 44 BC",
    "alaska_oil_environment_2025": "Balancing oil industry jobs and environmental protection in Alaska in 2025",
    "parisian_artists_future_2026": "Parisian artists' views on the future of art and AI creativity in 2026",
    "immigrant_integration_canada_2023": "Immigrant integration experiences and attitudes in Canada in 2023",
    "greek_underworld_shades": "Shades (spirits) in the Greek underworld reflecting on their mortal lives",
    "millennial_parenting_us_2025": "Millennial parenting styles and anxieties in the United States in 2025",
    "asi_human_creativity_2060": "Attitudes toward artificial superintelligence and human creativity in 2060",
    "high_school_students_italy_2016": "High school students' attitudes and aspirations in Italy in 2016",
    "genz_social_media_politics_2025": "Gen Z political engagement through social media in 2025",
    "healthcare_covid_stress_italy_2021": "Healthcare worker stress and burnout during COVID-19 in Italy in 2021",
    "scifi_authors_future_ai_space_2024": "Science fiction authors' visions of AI and space exploration in 2024",
    "esports_mental_health_sk_2024": "Esports players' mental health and well-being in South Korea in 2024",
    "silk_road_merchants_samarkand_750ad": "Silk Road merchants trading spices and silk in Samarkand around 750 AD",
    "mali_empire_scholars_timbuktu_1350ad": "Scholars and scribes in Timbuktu during the Mali Empire around 1350 AD",
    "ww2_civilian_sentiment_germany_1943": "Civilian sentiment and daily life in wartime Germany in 1943",
}

VALIDATION_SET: dict[str, str] = {
    "ai_tech_stock_sentiment_2025": "Investor sentiment toward AI technology stocks in 2025",
    "social_media_politics_europe_2026": "Social media's influence on political polarization in Europe in 2026",
    "inca_commoners_mita_1500ad": "Inca commoners' experiences with the mita labor system around 1500 AD",
    "athenian_piety_olympian_gods_430bc": "Athenian citizens' piety and devotion to the Olympian gods around 430 BC",
    "sleep_quality_2025": "Sleep quality, habits, and attitudes toward rest in 2025",
    "ai_companionship_integration_2070": "Attitudes toward AI companions and social integration in 2070",
    "asi_existential_dread_2050": "Existential dread about artificial superintelligence in 2050",
    "ancient_egypt_akhenaten_reforms_1340bc": "Egyptian citizens' reactions to Akhenaten's religious reforms around 1340 BC",
    "german_energy_policy_2025": "German citizens' attitudes toward energy policy and the green transition in 2025",
    "moral_dilemmas_global_2045": "Moral dilemmas and ethical reasoning in a technologically advanced world in 2045",
}

TEST_SET: dict[str, str] = {
    "agi_job_displacement_global_2035": "HARDCODED",
    "camelot_chivalry_quests": "Knights of Camelot and their views on chivalry, honor, and questing",
    "cold_war_anxiety_us_1962": "Cold War anxiety and nuclear fears among Americans in 1962",
    "viking_warriors_valhalla": "Viking warriors' beliefs about Valhalla, honor, and battle",
    "generalized_trust_in_the_salem_witch_trials_1692": "Trust and suspicion during the Salem witch trials in 1692",
    "nomadic_values_mongolia_2023": "Nomadic values and modernization among Mongolian herders in 2023",
    "meaning_of_life_2030": "Perspectives on the meaning of life in 2030",
    "agi_wealth_inequality_revolution_2040": "AGI-driven wealth inequality and revolutionary sentiment in 2040",
    "climate_anxiety_coastal_au_2024": "Climate anxiety among coastal Australians in 2024",
    "british_empire_attitudes_uk_1900": "British attitudes toward empire and colonialism in the United Kingdom in 1900",
}

# Map hardcoded names to their Questionnaire objects
_HARDCODED = {q.name: q for q in EXAMPLE_QUESTIONNAIRES}


async def generate_questionnaire(
    name: str,
    description: str,
    generator: QuestionnaireGenerator,
) -> None:
    """Generate and save a single questionnaire."""
    output_path = OUTPUT_DIR / f"{name}.json"

    # Skip if already exists
    if output_path.exists():
        print(f"  SKIP {name} (already exists)")
        return

    # Handle hardcoded questionnaires
    if description == "HARDCODED":
        if name in _HARDCODED:
            q = _HARDCODED[name]
            output_path.write_text(json.dumps(q.model_dump(), indent=2))
            print(f"  SAVE {name} (hardcoded, {len(q.questions)} items)")
        else:
            print(f"  ERROR {name} not found in hardcoded examples")
        return

    # Generate via LLM
    try:
        q = await generator.generate(description)
        # Override the name with the paper's identifier
        data = q.model_dump()
        data["name"] = name
        output_path.write_text(json.dumps(data, indent=2))
        print(f"  OK {name} ({len(q.questions)} items, {len(q.dimensions)} dims)")
    except Exception as e:
        print(f"  FAIL {name}: {e}")


async def generate_set(
    questionnaires: dict[str, str],
    set_name: str,
    generator: QuestionnaireGenerator,
) -> None:
    """Generate all questionnaires in a set sequentially."""
    print(f"\n{'='*60}")
    print(f"  {set_name} ({len(questionnaires)} questionnaires)")
    print(f"{'='*60}")

    for name, description in questionnaires.items():
        await generate_questionnaire(name, description, generator)


@click.command()
@click.option(
    "--set",
    "which_set",
    type=click.Choice(["all", "training", "validation", "test"]),
    default="all",
    help="Which questionnaire set to generate",
)
def main(which_set: str) -> None:
    """Generate all 50 questionnaires from the paper."""

    async def _run() -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        settings = get_settings()
        client = LLMClient(settings)
        generator = QuestionnaireGenerator(client=client, settings=settings)

        sets = {
            "training": TRAINING_SET,
            "validation": VALIDATION_SET,
            "test": TEST_SET,
        }

        if which_set == "all":
            for name, qs in sets.items():
                await generate_set(qs, f"{name.upper()} SET", generator)
        else:
            await generate_set(sets[which_set], f"{which_set.upper()} SET", generator)

        # Summary
        existing = list(OUTPUT_DIR.glob("*.json"))
        print(f"\nTotal questionnaires saved: {len(existing)}/50")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
