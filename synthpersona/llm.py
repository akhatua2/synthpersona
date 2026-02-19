"""Async LLM client wrapping LiteLLM with semaphore and retries."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import litellm
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from synthpersona.config import Settings, get_settings

logger = structlog.get_logger()

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True  # type: ignore[assignment]


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrency)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def complete(
        self,
        model: str,
        *,
        system: str | None = None,
        user: str,
        temperature: float = 1.0,
        response_json: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}

        async with self._semaphore:
            response = await litellm.acompletion(**kwargs)

        content = response.choices[0].message.content or ""
        return content

    async def complete_json(
        self,
        model: str,
        *,
        system: str | None = None,
        user: str,
        temperature: float = 1.0,
    ) -> Any:
        raw = await self.complete(
            model,
            system=system,
            user=user,
            temperature=temperature,
            response_json=True,
        )
        return _parse_json(raw)

    async def complete_batch(
        self,
        model: str,
        *,
        prompts: list[dict[str, str]],
        temperature: float = 1.0,
        response_json: bool = False,
    ) -> list[str]:
        tasks = [
            self.complete(
                model,
                system=p.get("system"),
                user=p["user"],
                temperature=temperature,
                response_json=response_json,
            )
            for p in prompts
        ]
        return await asyncio.gather(*tasks)


def _fix_json(text: str) -> str:
    """Fix common JSON issues from LLM output (trailing commas, etc.)."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _parse_json(raw: str) -> Any:
    """Parse JSON from LLM output with fallback strategies."""
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try fixing common issues
    try:
        return json.loads(_fix_json(raw))
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        block = match.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return json.loads(_fix_json(block))

    raise json.JSONDecodeError("Could not parse JSON from LLM output", raw, 0)
