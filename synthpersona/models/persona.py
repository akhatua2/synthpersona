"""Pydantic models for personas and population embeddings."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class PersonaDescriptor(BaseModel):
    name: str
    axis_positions: dict[str, float]
    high_level_description: str


class Persona(BaseModel):
    name: str
    descriptor: PersonaDescriptor
    full_description: str


class PopulationEmbedding(BaseModel):
    persona_name: str
    scores: dict[str, float]

    def to_array(self, dimensions: list[str]) -> np.ndarray:
        return np.array([self.scores[d] for d in dimensions])

    model_config = {"arbitrary_types_allowed": True}
