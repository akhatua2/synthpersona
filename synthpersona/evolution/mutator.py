"""LLM-based code mutation for evolutionary optimization."""

from __future__ import annotations

import random
import re

import structlog

from synthpersona.config import Settings, get_settings
from synthpersona.evolution.prompts import EVOLUTION_SYSTEM_PROMPT, MUTATION_PROMPTS
from synthpersona.llm import LLMClient

logger = structlog.get_logger()

_MUTATION_USER = """\
Here is the current Persona Generator code:

```python
{source_code}
```

{feedback_section}

Mutation instruction: {mutation_prompt}

Produce the improved Python code. The code must define a class with an async \
`generate(self, context: str, dimensions: list[str], n: int) -> list[Persona]` \
method and a `get_source_code(self) -> str` method. The class constructor \
must accept `client` (LLMClient) and `settings` (Settings) keyword arguments.

Return the COMPLETE modified code inside a single ```python ... ``` block."""


class Mutator:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or LLMClient(self.settings)

    async def mutate(
        self,
        source_code: str,
        feedback: str | None = None,
    ) -> str | None:
        """Mutate source code using a random mutation prompt.

        Returns the new source code, or None if mutation failed.
        """
        mutation_prompt = random.choice(MUTATION_PROMPTS)

        feedback_section = ""
        if feedback:
            feedback_section = f"Feedback from recent evaluation:\n{feedback}\n"

        user_prompt = _MUTATION_USER.format(
            source_code=source_code,
            feedback_section=feedback_section,
            mutation_prompt=mutation_prompt,
        )

        try:
            response = await self.client.complete(
                self.settings.smart_model,
                system=EVOLUTION_SYSTEM_PROMPT,
                user=user_prompt,
            )
            return _extract_code(response)
        except Exception:
            logger.exception("mutation_failed")
            return None


def _extract_code(response: str) -> str | None:
    """Extract Python code from markdown code block."""
    match = re.search(r"```python\s*([\s\S]*?)```", response)
    if match:
        return match.group(1).strip()
    # Fallback: try any code block
    match = re.search(r"```\s*([\s\S]*?)```", response)
    if match:
        return match.group(1).strip()
    return None
