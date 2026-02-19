"""Simulation runner using logic of appropriateness."""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict

import structlog

from synthpersona.config import Settings, get_settings
from synthpersona.llm import LLMClient
from synthpersona.models.persona import Persona, PopulationEmbedding
from synthpersona.models.questionnaire import Question, Questionnaire

logger = structlog.get_logger()

_SIMULATION_SYSTEM = """\
You are role-playing as a specific person in a social simulation. You must \
answer questions in character, staying true to the persona description below.

Answer using ONLY the logic of appropriateness:
1. What kind of situation is this?
2. What kind of person am I?
3. What does a person like me do in a situation like this?

You must respond with a JSON object containing a single key "choice" with \
the 1-indexed number of your selected option. Nothing else."""

_SIMULATION_USER = """\
You are: {persona_name}

{persona_description}

---

{question_text}

Respond with ONLY a JSON object: {{"choice": <number>}}"""


class SimulationRunner:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(self.settings)

    async def simulate(
        self,
        personas: list[Persona],
        questionnaire: Questionnaire,
    ) -> list[PopulationEmbedding]:
        tasks = [self._simulate_persona(p, questionnaire) for p in personas]
        return await asyncio.gather(*tasks)

    async def _simulate_persona(
        self,
        persona: Persona,
        questionnaire: Questionnaire,
    ) -> PopulationEmbedding:
        # Answer each question independently (memory reset between questions)
        question_tasks = [
            self._answer_question(persona, q) for q in questionnaire.questions
        ]
        scores = await asyncio.gather(*question_tasks)

        # Aggregate by dimension (mean score per dimension)
        dim_scores: dict[str, list[float]] = defaultdict(list)
        for question, score in zip(questionnaire.questions, scores, strict=True):
            dim_scores[question.dimension].append(score)

        embedding = {dim: sum(vals) / len(vals) for dim, vals in dim_scores.items()}

        return PopulationEmbedding(
            persona_name=persona.name,
            scores=embedding,
        )

    async def _answer_question(
        self,
        persona: Persona,
        question: Question,
    ) -> float:
        question_text = question.format_for_simulation(persona.name)

        user_prompt = _SIMULATION_USER.format(
            persona_name=persona.name,
            persona_description=persona.full_description,
            question_text=question_text,
        )

        try:
            result = await self.client.complete_json(
                self.settings.fast_model,
                system=_SIMULATION_SYSTEM,
                user=user_prompt,
                temperature=self.settings.simulation_temperature,
            )
            choice_idx = int(result["choice"]) - 1  # Convert to 0-indexed
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract a number from the response
            raw = await self.client.complete(
                self.settings.fast_model,
                system=_SIMULATION_SYSTEM,
                user=user_prompt,
                temperature=self.settings.simulation_temperature,
            )
            match = re.search(r"\d+", raw)
            if match:
                choice_idx = int(match.group()) - 1
            else:
                logger.warning(
                    "failed_to_parse_response",
                    persona=persona.name,
                    question=question.statement[:50],
                )
                choice_idx = len(question.choices) // 2  # Default to middle

        # Clamp to valid range
        choice_idx = max(0, min(choice_idx, len(question.choices) - 1))
        return question.score(choice_idx)
