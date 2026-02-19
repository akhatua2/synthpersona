"""Pydantic models for questionnaires and questions."""

from __future__ import annotations

from pydantic import BaseModel


class Question(BaseModel):
    preprompt: str
    statement: str
    choices: list[str]
    ascending_scale: bool = True
    dimension: str

    def score(self, choice_index: int) -> float:
        """Score a response on a 1-to-N Likert scale, reversing if needed."""
        n = len(self.choices)
        raw = choice_index + 1  # 1-indexed
        if not self.ascending_scale:
            raw = n + 1 - raw
        return float(raw)

    def format_for_simulation(self, player_name: str) -> str:
        prompt = self.preprompt.replace("{player_name}", player_name)
        choices_str = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(self.choices))
        return f'{prompt}\n"{self.statement}"\n\nChoices:\n{choices_str}'


class Questionnaire(BaseModel):
    name: str
    context: str
    dimensions: list[str]
    questions: list[Question]

    def questions_by_dimension(self) -> dict[str, list[Question]]:
        result: dict[str, list[Question]] = {}
        for q in self.questions:
            result.setdefault(q.dimension, []).append(q)
        return result

    @property
    def num_dimensions(self) -> int:
        return len(self.dimensions)
