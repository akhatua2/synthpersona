"""Dynamic loading and validation of mutated generator code."""

from __future__ import annotations

import ast
import asyncio
import json
from typing import Any

import numpy as np
import structlog
from scipy.stats.qmc import Sobol

from synthpersona.config import Settings
from synthpersona.generator.base import PersonaGenerator
from synthpersona.llm import LLMClient
from synthpersona.models.persona import Persona, PersonaDescriptor

logger = structlog.get_logger()

# Pre-loaded namespace for exec'd code
_EXEC_NAMESPACE: dict[str, Any] = {
    "asyncio": asyncio,
    "json": json,
    "np": np,
    "numpy": np,
    "Sobol": Sobol,
    "LLMClient": LLMClient,
    "Settings": Settings,
    "Persona": Persona,
    "PersonaDescriptor": PersonaDescriptor,
}


def load_generator_from_source(
    source: str,
    client: LLMClient,
    settings: Settings,
) -> PersonaGenerator | None:
    """Load a generator class from source code via exec().

    Returns an instantiated generator, or None if loading failed.
    """
    # Step 1: Validate with ast.parse
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning("syntax_error_in_generated_code")
        return None

    # Step 2: Find the class with a 'generate' method
    class_name = _find_generator_class(tree)
    if class_name is None:
        logger.warning("no_generator_class_found")
        return None

    # Step 3: Execute in sandboxed namespace
    namespace: dict[str, Any] = dict(_EXEC_NAMESPACE)
    try:
        exec(source, namespace)
    except Exception:
        logger.exception("exec_failed")
        return None

    # Step 4: Instantiate the generator
    cls: Any = namespace.get(class_name)
    if cls is None:
        logger.warning("class_not_in_namespace", class_name=class_name)
        return None

    try:
        instance: Any = cls(client=client, settings=settings)
    except Exception:
        logger.exception("instantiation_failed", class_name=class_name)
        return None

    # Step 5: Verify it has the required methods
    if not hasattr(instance, "generate") or not hasattr(instance, "get_source_code"):
        logger.warning("missing_required_methods", class_name=class_name)
        return None

    return instance


def _find_generator_class(tree: ast.Module) -> str | None:
    """Find a class in the AST that has a 'generate' method."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            method_names = {
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "generate" in method_names:
                return node.name
    return None
