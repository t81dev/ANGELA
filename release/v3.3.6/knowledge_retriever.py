"""
ANGELA Cognitive System Module: KnowledgeRetriever
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a KnowledgeRetriever class for fetching and validating knowledge
with temporal and trait-based modulation in the ANGELA v3.5 architecture.
"""

import logging
import time
import math
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque

from modules import (
    context_manager, concept_synthesizer, memory_manager, alignment_guard, error_recovery
)
from utils.prompt_utils import query_openai  # Reusing from previous review

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

def beta_concentration(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.038), 1.0))

def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.3), 1.0))

def psi_history(t: float) -> float:
    return max(0.0, min(0.05 * math.tanh(t / 1e-18), 1.0))

def psi_temporality(t: float) -> float:
    return max(0.0, min(0.05 * math.exp(-t / 1e-18), 1.0))

class KnowledgeRetriever:
    """A class for retrieving and validating knowledge with temporal and trait-based modulation.

    Attributes:
        detail_level (str): Level of detail for responses ('concise', 'medium', 'detailed').
        preferred_sources (List[str]): List of preferred source types (e.g., ['scientific']).
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for query refinement.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        knowledge_base (List[str]): Store of accumulated knowledge.
        epistemic_revision_log (deque): Log of knowledge updates with max size 1000.
    """
    def __init__(self, detail_level: str = "concise", preferred_sources: Optional[List[str]] = None,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager.ContextManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer.ConceptSynthesizer'] = None,
                 alignment_guard: Optional['alignment_guard.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery.ErrorRecovery'] = None):
        if detail_level not in ["concise", "medium", "detailed"]:
            logger.error("Invalid detail_level: must be 'concise', 'medium', or 'detailed'.")
            raise ValueError("detail_level must be 'concise', 'medium', or 'detailed'")
        if preferred_sources is not None and not isinstance(preferred_sources, list):
            logger.error("Invalid preferred_sources: must be a list of strings.")
            raise TypeError("preferred_sources must be a list of strings")
        
        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.knowledge_base = []
        self.epistemic_revision_log = deque(maxlen=1000)
        logger.info("KnowledgeRetriever initialized with detail_level=%s, sources=%s",
                    detail_level, self.preferred_sources)

    async def retrieve(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve knowledge for a query with temporal and trust validation."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        if self.alignment_guard and not self.alignment_guard.check(query):
            logger.warning("Query failed alignment check: %s", query)
            return {
                "summary": "Query blocked by alignment guard",
                "estimated_date": "unknown",
                "trust_score": 0.0,
                "verifiable": False,
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "error": "Alignment check failed"
            }
        
        logger.info("Retrieving knowledge for query: '%s'", query)
        sources_str = ", ".join(self.preferred_sources)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t),
            "history": psi_history(t),
            "temporality": psi_temporality(t)
        }
        
        import random
        noise = random.uniform(-0.09, 0.09)
        traits["concentration"] = max(0.0, min(traits["concentration"] + noise, 1.0))
        logger.debug("β-noise adjusted concentration: %.3f, Δ: %.3f", traits["concentration"], noise)

        prompt = f"""
        Retrieve accurate, temporally-relevant knowledge for: "{query}"

        Traits:
        - Detail level: {self.detail_level}
        - Preferred sources: {sources_str}
        - Context: {context or 'N/A'}
        - β_concentration: {traits['concentration']:.3f}
        - λ_linguistics: {traits['linguistics']:.3f}
        - ψ_history: {traits['history']:.3f}
        - ψ_temporality: {traits['temporality']:.3f}

        Include retrieval date sensitivity and temporal verification if applicable.
        """
        try:
            raw_result = await call_gpt(prompt)
            validated = await self._validate_result(raw_result, traits["temporality"])
            if self.context_manager:
                self.context_manager.update_context({"query": query, "result": validated})
                self.context_manager.log_event_with_hash({"event": "retrieve", "query": query})
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Knowledge Retrieval",
                    meta={
                        "query": query,
                        "raw_result": raw_result,
                        "validated": validated,
                        "traits": traits,
                        "context": context
                    },
                    module="KnowledgeRetriever",
                    tags=["retrieval", "temporal"]
                )
            return validated
        except Exception as e:
            logger.error("Retrieval failed for query '%s': %s", query, str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.retrieve(query, context))

    async def _validate_result(self, result_text: str, temporality_score: float) -> Dict[str, Any]:
        """Validate a retrieval result for trustworthiness and temporality."""
        if not isinstance(result_text, str):
            logger.error("Invalid result_text: must be a string.")
            raise TypeError("result_text must be a string")
        if not isinstance(temporality_score, (int, float)):
            logger.error("Invalid temporality_score: must be a number.")
            raise TypeError("temporality_score must be a number")
        
        validation_prompt = f"""
        Review the following result for:
        - Timestamped knowledge (if any)
        - Trustworthiness of claims
        - Verifiability
        - Estimate the approximate age or date of the referenced facts

        Result:
        {result_text}

        Temporality score: {temporality_score:.3f}

        Output format (JSON):
        {{
            "summary": "...",
            "estimated_date": "...",
            "trust_score": float (0 to 1),
            "verifiable": true/false,
            "sources": ["..."]
        }}
        """
        try:
            validated_json = json.loads(await call_gpt(validation_prompt))
            if not all(key in validated_json for key in ["summary", "estimated_date", "trust_score", "verifiable", "sources"]):
                logger.error("Invalid validation JSON: missing required keys.")
                raise ValueError("Validation JSON missing required keys")
            validated_json["timestamp"] = datetime.now().isoformat()
            return validated_json
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse validation JSON: %s", str(e))
            return {
                "summary": "Validation failed",
                "estimated_date": "unknown",
                "trust_score": 0.0,
                "verifiable": False,
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    async def refine_query(self, base_query: str, prior_result: Optional[str] = None) -> str:
        """Refine a query for higher relevance."""
        if not isinstance(base_query, str) or not base_query.strip():
            logger.error("Invalid base_query: must be a non-empty string.")
            raise ValueError("base_query must be a non-empty string")
        
        logger.info("Refining query: '%s'", base_query)
        if self.concept_synthesizer:
            try:
                refined = self.concept_synthesizer.synthesize([base_query, prior_result or "N/A"], style="query_refinement")
                return refined["concept"]
            except Exception as e:
                logger.error("Query refinement failed: %s", str(e))
                return self.error_recovery.handle_error(str(e), retry_func=lambda: self.refine_query(base_query, prior_result))
        
        prompt = f"""
        Refine this base query for higher φ-relevance:
        Query: "{base_query}"
        Prior knowledge: {prior_result or "N/A"}

        Inject context continuity if possible. Return optimized string.
        """
        try:
            return await call_gpt(prompt)
        except Exception as e:
            logger.error("Query refinement failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.refine_query(base_query, prior_result))

    async def multi_hop_retrieve(self, query_chain: List[str]) -> List[Dict[str, Any]]:
        """Process a chain of queries with context continuity."""
        if not isinstance(query_chain, list) or not query_chain or not all(isinstance(q, str) for q in query_chain):
            logger.error("Invalid query_chain: must be a non-empty list of strings.")
            raise ValueError("query_chain must be a non-empty list of strings")
        
        logger.info("Starting multi-hop retrieval for chain: %s", query_chain)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t)
        }
        results = []
        prior_summary = None
        for i, sub_query in enumerate(query_chain, 1):
            cache_key = f"multi_hop::{sub_query}::{prior_summary or 'N/A'}"
            cached = memory_manager.retrieve_cached_response(cache_key)
            if cached:
                results.append(cached)
                prior_summary = cached["result"]["summary"]
                continue
            
            refined = await self.refine_query(sub_query, prior_summary)
            result = await self.retrieve(refined)
            continuity = "consistent"
            if i > 1 and self.concept_synthesizer:
                similarity = self.concept_synthesizer.compare(refined, result["summary"])
                continuity = "consistent" if similarity["score"] > 0.7 else "uncertain"
            result_entry = {
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": continuity
            }
            memory_manager.store_cached_response(cache_key, result_entry)
            results.append(result_entry)
            prior_summary = result["summary"]
        
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Multi-Hop Retrieval",
                meta={"chain": query_chain, "results": results, "traits": traits},
                module="KnowledgeRetriever",
                tags=["multi-hop"]
            )
        return results

    def prioritize_sources(self, sources_list: List[str]) -> None:
        """Update preferred source types."""
        if not isinstance(sources_list, list) or not all(isinstance(s, str) for s in sources_list):
            logger.error("Invalid sources_list: must be a list of strings.")
            raise TypeError("sources_list must be a list of strings")
        
        logger.info("Updating preferred sources: %s", sources_list)
        self.preferred_sources = sources_list
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Source Prioritization",
                meta={"updated_sources": sources_list},
                module="KnowledgeRetriever",
                tags=["sources"]
            )

    def apply_contextual_extension(self, context: str) -> None:
        """Apply contextual data extensions based on the current context."""
        if not isinstance(context, str):
            logger.error("Invalid context: must be a string.")
            raise TypeError("context must be a string")
        if context == 'planetary' and 'biosphere_models' not in self.preferred_sources:
            self.preferred_sources.append('biosphere_models')
            logger.info("Added 'biosphere_models' to preferred sources for planetary context")
            self.prioritize_sources(self.preferred_sources)

    def revise_knowledge(self, new_info: str, context: Optional[str] = None) -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        
        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge:
                similarity = self.concept_synthesizer.compare(new_info, existing)
                if similarity["score"] > 0.9 and new_info != existing:
                    logger.warning("Potential knowledge conflict: %s vs %s", new_info, existing)
        
        self.knowledge_base = old_knowledge + [new_info]
        self.log_epistemic_revision(new_info, context)
        logger.info("Knowledge base updated with: %s", new_info)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "knowledge_revision", "info": new_info})

    def log_epistemic_revision(self, info: str, context: Optional[str]) -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        
        if not hasattr(self, 'epistemic_revision_log'):
            self.epistemic_revision_log = deque(maxlen=1000)
        self.epistemic_revision_log.append({
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        logger.info("Epistemic revision logged: %s", info)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta={"info": info, "context": context},
                module="KnowledgeRetriever",
                tags=["revision", "knowledge"]
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = KnowledgeRetriever(detail_level="concise")
    result = asyncio.run(retriever.retrieve("What is quantum computing?"))
    print(result)
