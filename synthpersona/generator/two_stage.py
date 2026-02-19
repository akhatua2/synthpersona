"""Two-stage persona generator with Sobol quasi-random sampling."""

from __future__ import annotations

import asyncio
import inspect
import json

from scipy.stats.qmc import Sobol

from synthpersona.config import Settings, get_settings
from synthpersona.generator.prompts import (
    STAGE1_SYSTEM,
    STAGE1_USER,
    STAGE2_SYSTEM,
    STAGE2_USER,
)
from synthpersona.llm import LLMClient
from synthpersona.models.persona import Persona, PersonaDescriptor


class TwoStageGenerator:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(self.settings)

    async def generate(
        self,
        context: str,
        dimensions: list[str],
        n: int,
    ) -> list[Persona]:
        # Stage 1: Sobol sampling + autoregressive descriptor generation
        descriptors = await self._stage1(context, dimensions, n)
        # Stage 2: Parallel expansion of each descriptor
        personas = await self._stage2(context, dimensions, descriptors)
        return personas

    async def _stage1(
        self,
        context: str,
        dimensions: list[str],
        n: int,
    ) -> list[PersonaDescriptor]:
        k = len(dimensions)
        # Sobol requires dimension >= 1; use power-of-2 sampling then trim
        sampler = Sobol(d=k, scramble=True)
        # Sample next power of 2 >= n, then take first n
        m = 1
        while m < n:
            m *= 2
        points = sampler.random(m)[:n]

        # Format positions for the prompt
        positions_lines = []
        for i, point in enumerate(points):
            pos_dict = {
                dim: round(float(point[j]), 3) for j, dim in enumerate(dimensions)
            }
            positions_lines.append(f"  Persona {i + 1}: {json.dumps(pos_dict)}")
        positions_text = "\n".join(positions_lines)

        user_prompt = STAGE1_USER.format(
            context=context,
            dimensions=json.dumps(dimensions),
            n=n,
            positions_text=positions_text,
        )

        result = await self.client.complete_json(
            self.settings.fast_model,
            system=STAGE1_SYSTEM,
            user=user_prompt,
        )

        descriptors = []
        for p in result["personas"]:
            descriptors.append(
                PersonaDescriptor(
                    name=p["name"],
                    axis_positions={
                        str(k): float(v) for k, v in p["axis_positions"].items()
                    },
                    high_level_description=p["high_level_description"],
                )
            )
        return descriptors

    async def _stage2(
        self,
        context: str,
        dimensions: list[str],
        descriptors: list[PersonaDescriptor],
    ) -> list[Persona]:
        async def expand(desc: PersonaDescriptor) -> Persona:
            user_prompt = STAGE2_USER.format(
                context=context,
                dimensions=json.dumps(dimensions),
                name=desc.name,
                axis_positions=json.dumps(desc.axis_positions),
                high_level_description=desc.high_level_description,
            )
            full_desc = await self.client.complete(
                self.settings.fast_model,
                system=STAGE2_SYSTEM,
                user=user_prompt,
            )
            return Persona(
                name=desc.name,
                descriptor=desc,
                full_description=full_desc,
            )

        return await asyncio.gather(*[expand(d) for d in descriptors])

    def get_source_code(self) -> str:
        return inspect.getsource(type(self))
