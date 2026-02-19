"""Protocol for persona generators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from synthpersona.models.persona import Persona


@runtime_checkable
class PersonaGenerator(Protocol):
    async def generate(
        self,
        context: str,
        dimensions: list[str],
        n: int,
    ) -> list[Persona]: ...

    def get_source_code(self) -> str: ...
