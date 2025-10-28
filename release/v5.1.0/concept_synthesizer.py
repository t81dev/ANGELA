from __future__ import annotations
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional

# ANGELA modules
from modules import (
    context_manager as context_manager_mod,
    error_recovery as error_recovery_mod,
    memory_manager as memory_manager_mod,
    alignment_guard as alignment_guard_mod,
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.ConceptSynthesizer")

class ConceptSynthesizer:
    """
    Synthesizes, compares, and validates concepts for ANGELA.
    """
    def __init__(self, **services):
        self.services = services
        self.concept_cache: List[Dict] = []
        logger.info("ConceptSynthesizer v5.1.0 initialized")

    async def generate(self, concept_name: str, context: Dict, task_type: str = "") -> Dict:
        """Generates a new concept definition."""
        prompt = self._construct_prompt(concept_name, context, task_type)
        raw_result = await self._query_model(prompt)

        return self._process_and_validate(raw_result, concept_name)

    def _construct_prompt(self, name: str, context: Dict, task_type: str) -> str:
        return f"""
        Generate a JSON concept definition for '{name}' with keys 'definition' and 'context'.
        Context: {json.dumps(context)}
        Task Type: {task_type}
        """.strip()

    async def _query_model(self, prompt: str) -> str:
        try:
            return await query_openai(prompt, model="gpt-4", temperature=0.5)
        except Exception as e:
            logger.error(f"GPT query failed: {e}")
            return "{}"

    def _process_and_validate(self, raw_result: str, name: str) -> Dict:
        """Processes the raw model output and validates the concept."""
        try:
            concept = json.loads(raw_result)
            concept["name"] = name

            is_valid, report = self._validate_concept(concept)
            if not is_valid:
                return {"error": "Concept failed validation", "report": report, "success": False}

            self.concept_cache.append(concept)
            return {"concept": concept, "success": True}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from model", "success": False}

    def _validate_concept(self, concept: Dict) -> Tuple[bool, Dict]:
        """Validates a concept against alignment and ontology."""
        alignment_guard = self.services.get("alignment_guard")
        if alignment_guard:
            is_safe, report = asyncio.run(alignment_guard.ethical_check(
                concept.get("definition", ""), stage="concept_validation"
            ))
            if not is_safe:
                return False, report

        return True, {}

    async def compare(self, concept_a: str, concept_b: str) -> Dict:
        """Compares two concepts for similarity."""
        prompt = f"Compare these concepts and return a JSON with 'score' and 'differences':\nA: {concept_a}\nB: {concept_b}"
        raw_result = await self._query_model(prompt)

        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "score": 0.0}
