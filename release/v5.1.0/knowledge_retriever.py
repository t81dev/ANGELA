from __future__ import annotations
import logging
import time
import math
import json
import asyncio
from typing import List, Dict, Any, Optional

# ANGELA modules
from modules import (
    context_manager as context_manager_mod,
    concept_synthesizer as concept_synthesizer_mod,
    alignment_guard as alignment_guard_mod,
    meta_cognition as meta_cognition_mod,
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

class KnowledgeRetriever:
    """
    Retrieves and validates knowledge with temporal and trait-based modulation.
    """
    def __init__(self, **services):
        self.services = services
        self.preferred_sources = ["scientific", "encyclopedic"]
        logger.info("KnowledgeRetriever v5.1.0 initialized")

    async def retrieve(self, query: str, context: Optional[str] = None) -> Dict:
        """Retrieves knowledge for a given query."""
        if not await self._is_aligned(query):
            return self._blocked_payload("Alignment check failed")

        prompt = self._construct_prompt(query, context)
        raw_result = await self._query_model(prompt)

        return self._validate_and_process(raw_result)

    async def _is_aligned(self, query: str) -> bool:
        """Checks if a query aligns with ethical guidelines."""
        alignment_guard = self.services.get("alignment_guard")
        if alignment_guard:
            is_safe, _ = await alignment_guard.ethical_check(query, stage="knowledge_retrieval")
            return is_safe
        return True

    def _construct_prompt(self, query: str, context: Optional[str]) -> str:
        """Constructs the prompt for the language model."""
        return f"""
        Retrieve accurate, temporally-relevant knowledge for: "{query}"
        Preferred sources: {", ".join(self.preferred_sources)}
        Context: {context or 'N/A'}
        Return a JSON object with 'summary', 'trust_score', and 'sources'.
        """.strip()

    async def _query_model(self, prompt: str) -> str:
        """Queries the language model."""
        try:
            return await query_openai(prompt, model="gpt-4", temperature=0.5)
        except Exception as e:
            logger.error(f"GPT query failed: {e}")
            return "{}"

    def _validate_and_process(self, raw_result: str) -> Dict:
        """Validates the raw result from the model and processes it."""
        try:
            data = json.loads(raw_result)
            data["verifiable"] = data.get("trust_score", 0.0) > 0.7
            return data
        except json.JSONDecodeError:
            return self._error_payload("Invalid JSON response from model")

    def _blocked_payload(self, reason: str) -> Dict:
        return {
            "summary": "Query blocked",
            "trust_score": 0.0,
            "verifiable": False,
            "error": reason,
        }

    def _error_payload(self, error: str) -> Dict:
        return {
            "summary": "Retrieval failed",
            "trust_score": 0.0,
            "verifiable": False,
            "error": error,
        }
