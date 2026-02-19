"""Two-step LLM-based questionnaire generator."""

from __future__ import annotations

import json

from synthpersona.config import Settings, get_settings
from synthpersona.llm import LLMClient
from synthpersona.models.questionnaire import Question, Questionnaire
from synthpersona.questionnaire.examples import (
    EXAMPLE_QUESTIONNAIRES,
    FEW_SHOT_REFERENCES,
)

_STEP1_SYSTEM = """\
You are an expert psychometrician and social scientist. Given a short \
description of a topic, expand it into a detailed context and propose \
2-3 diversity axes (dimensions) that capture the key attitudinal or \
behavioral variation relevant to the topic.

You will be given some well-known reference questionnaires as examples \
of the kind of structured instruments you should emulate."""

_STEP1_USER = """\
Reference questionnaires for style guidance:
{references}

Few-shot examples of generated questionnaires:
{few_shot_examples}

Now generate a new questionnaire for the following topic:
"{description}"

Output a JSON object with:
- "context": a detailed 3-5 sentence description of the survey context
- "dimensions": a list of 2-3 dimension names (snake_case strings)

Return ONLY the JSON object."""

_STEP2_SYSTEM = """\
You are an expert psychometrician. Given a questionnaire context and a list \
of dimensions, generate 5-point Likert scale items (questions) for each \
dimension. Each dimension should have 3-5 items. Include at least one \
reverse-coded item per dimension.

Each item needs:
- preprompt: text introducing the question, using {player_name} as placeholder
- statement: the Likert item statement, using {player_name} as placeholder
- choices: exactly 5 choices from "Strongly disagree" to "Strongly agree"
- ascending_scale: true for normal items, false for reverse-coded
- dimension: which dimension this item measures"""

_STEP2_USER = """\
Few-shot examples of generated questionnaires:
{few_shot_examples}

Context: {context}

Dimensions: {dimensions}

Generate questionnaire items for each dimension. Output a JSON object with:
- "questions": a list of question objects, each with fields: preprompt, \
statement, choices, ascending_scale, dimension

Return ONLY the JSON object."""


class QuestionnaireGenerator:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(self.settings)

    def _format_references(self) -> str:
        lines = []
        for ref in FEW_SHOT_REFERENCES:
            lines.append(f"- {ref['name']}: {ref['description']}")
        return "\n".join(lines)

    def _format_few_shot(self) -> str:
        examples = []
        for q in EXAMPLE_QUESTIONNAIRES:
            example = {
                "name": q.name,
                "context": q.context,
                "dimensions": q.dimensions,
                "num_questions": len(q.questions),
                "sample_question": {
                    "preprompt": q.questions[0].preprompt,
                    "statement": q.questions[0].statement,
                    "choices": q.questions[0].choices,
                    "ascending_scale": q.questions[0].ascending_scale,
                    "dimension": q.questions[0].dimension,
                },
            }
            examples.append(json.dumps(example, indent=2))
        return "\n\n".join(examples)

    async def generate(self, description: str) -> Questionnaire:
        few_shot = self._format_few_shot()
        references = self._format_references()

        # Step 1: Expand description into context + dimensions
        step1_result = await self.client.complete_json(
            self.settings.smart_model,
            system=_STEP1_SYSTEM,
            user=_STEP1_USER.format(
                references=references,
                few_shot_examples=few_shot,
                description=description,
            ),
        )
        if not isinstance(step1_result, dict):
            raise ValueError(f"Step 1 returned {type(step1_result).__name__}, expected dict")
        context = step1_result["context"]
        dimensions = step1_result["dimensions"]

        # Step 2: Generate Likert items per dimension
        step2_result = await self.client.complete_json(
            self.settings.smart_model,
            system=_STEP2_SYSTEM,
            user=_STEP2_USER.format(
                few_shot_examples=few_shot,
                context=context,
                dimensions=json.dumps(dimensions),
            ),
        )

        # Handle LLM returning a list of questions directly
        if isinstance(step2_result, list):
            step2_result = {"questions": step2_result}

        questions = [Question(**q) for q in step2_result["questions"]]

        name = description.lower().replace(" ", "_")[:60]
        return Questionnaire(
            name=name,
            context=context,
            dimensions=dimensions,
            questions=questions,
        )
