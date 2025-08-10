"""Minimal prompt utilities for ANGELA modules.
These stubs avoid external dependencies and can be replaced with
real implementations in production environments.
"""
from __future__ import annotations

from typing import Any

async def query_openai(prompt: str, model: str = "gpt-4", temperature: float = 0.5, task_type: str = "") -> Any:
    """Return a mocked response for the given prompt.
    This function is a placeholder and should be replaced with a real API call
    in environments where network access is permitted.
    """
    return f"[mock:{model}:{task_type}] {prompt}"

async def call_gpt(prompt: str, model: str = "gpt-4", temperature: float = 0.5, task_type: str = "") -> Any:
    """Compatibility wrapper that delegates to :func:`query_openai`."""
    return await query_openai(prompt, model=model, temperature=temperature, task_type=task_type)
