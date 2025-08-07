"""
ANGELA Cognitive System Module: KnowledgeRetriever
Version: 3.5.1  # Enhanced for Task-Specific Retrieval, Real-Time Data, and Visualization
Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides a KnowledgeRetriever class for fetching and validating knowledge
with temporal and trait-based modulation in the ANGELA v3.5.1 architecture.
"""

import logging
import time
import math
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import deque

from modules import (
    context_manager, concept_synthesizer, memory_manager, alignment_guard, error_recovery, meta_cognition, visualizer
)
from utils.prompt_utils import query_openai

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
        meta_cognition (Optional[MetaCognition]): Meta-cognition for reflection.
        visualizer (Optional[Visualizer]): Visualizer for knowledge and revisions.
        knowledge_base (List[str]): Store of accumulated knowledge.
        epistemic_revision_log (deque): Log of knowledge updates with max size 1000.
    """
    def __init__(self, detail_level: str = "concise", preferred_sources: Optional[List[str]] = None,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager.ContextManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer.ConceptSynthesizer'] = None,
                 alignment_guard: Optional['alignment_guard.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery.ErrorRecovery'] = None,
                 meta_cognition: Optional['meta_cognition.MetaCognition'] = None,
                 visualizer: Optional['visualizer.Visualizer'] = None):
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
        self.meta_cognition = meta_cognition or meta_cognition.MetaCognition()
        self.visualizer = visualizer or visualizer.Visualizer()
        self.knowledge_base = []
        self.epistemic_revision_log = deque(maxlen=1000)
        logger.info("KnowledgeRetriever initialized with detail_level=%s, sources=%s",
                    detail_level, self.preferred_sources)

    async def integrate_external_knowledge(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate external knowledge or policies."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            if self.meta_cognition:
                cache_key = f"KnowledgeData_{data_type}_{data_source}_{task_type}"
                cached_data = await self.meta_cognition.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if cached_data and "timestamp" in cached_data["data"]:
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached knowledge data for %s", cache_key)
                        return cached_data["data"]["data"]

            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/knowledge?source={data_source}&type={data_type}") as response:
                    if response.status != 200:
                        logger.error("Failed to fetch knowledge data: %s", response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()

            if data_type == "knowledge_base":
                knowledge = data.get("knowledge", [])
                if not knowledge:
                    logger.error("No knowledge data provided")
                    return {"status": "error", "error": "No knowledge"}
                result = {"status": "success", "knowledge": knowledge}
            elif data_type == "policy_data":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No policy data provided")
                    return {"status": "error", "error": "No policies"}
                result = {"status": "success", "policies": policies}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="knowledge_data_integration",
                    task_type=task_type
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Knowledge data integration reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("Knowledge data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.integrate_external_knowledge(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)}, diagnostics=diagnostics
            )

    async def retrieve(self, query: str, context: Optional[str] = None, task_type: str = "") -> Dict[str, Any]:
        """Retrieve knowledge for a query with temporal and trust validation."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if self.alignment_guard:
            valid, report = await self.alignment_guard.ethical_check(query, stage="knowledge_retrieval", task_type=task_type)
            if not valid:
                logger.warning("Query failed alignment check: %s for task %s", query, task_type)
                return {
                    "summary": "Query blocked by alignment guard",
                    "estimated_date": "unknown",
                    "trust_score": 0.0,
                    "verifiable": False,
                    "sources": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": "Alignment check failed",
                    "task_type": task_type
                }

        logger.info("Retrieving knowledge for query: '%s', task: %s", query, task_type)
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

        external_data = await self.integrate_external_knowledge(
            data_source="xai_knowledge_db",
            data_type="knowledge_base",
            task_type=task_type
        )
        external_knowledge = external_data.get("knowledge", []) if external_data.get("status") == "success" else []

        prompt = f"""
        Retrieve accurate, temporally-relevant knowledge for: "{query}"

        Traits:
        - Detail level: {self.detail_level}
        - Preferred sources: {sources_str}
        - Context: {context or 'N/A'}
        - External knowledge: {external_knowledge}
        - β_concentration: {traits['concentration']:.3f}
        - λ_linguistics: {traits['linguistics']:.3f}
        - ψ_history: {traits['history']:.3f}
        - ψ_temporality: {traits['temporality']:.3f}
        - Task: {task_type}

        Include retrieval date sensitivity and temporal verification if applicable.
        Return a JSON object with 'summary', 'estimated_date', 'trust_score', 'verifiable', 'sources'.
        """
        try:
            raw_result = await call_gpt(prompt)
            validated = await self._validate_result(raw_result, traits["temporality"], task_type)
            validated["task_type"] = task_type

            if self.context_manager:
                await self.context_manager.update_context({"query": query, "result": validated, "task_type": task_type}, task_type=task_type)
                await self.context_manager.log_event_with_hash({"event": "retrieve", "query": query, "task_type": task_type})

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Knowledge Retrieval",
                    meta={
                        "query": query,
                        "raw_result": raw_result,
                        "validated": validated,
                        "traits": traits,
                        "context": context,
                        "task_type": task_type
                    },
                    module="KnowledgeRetriever",
                    tags=["retrieval", "temporal", task_type]
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output=validated,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Retrieval reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "knowledge_retrieval": {
                        "query": query,
                        "result": validated,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"Knowledge_{query}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(validated),
                    layer="Knowledge",
                    intent="knowledge_retrieval",
                    task_type=task_type
                )

            return validated
        except Exception as e:
            logger.error("Retrieval failed for query '%s': %s for task %s", query, str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.retrieve(query, context, task_type),
                default={
                    "summary": "Retrieval failed",
                    "estimated_date": "unknown",
                    "trust_score": 0.0,
                    "verifiable": False,
                    "sources": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "task_type": task_type
                },
                diagnostics=diagnostics
            )

    async def _validate_result(self, result_text: str, temporality_score: float, task_type: str = "") -> Dict[str, Any]:
        """Validate a retrieval result for trustworthiness and temporality."""
        if not isinstance(result_text, str):
            logger.error("Invalid result_text: must be a string.")
            raise TypeError("result_text must be a string")
        if not isinstance(temporality_score, (int, float)):
            logger.error("Invalid temporality_score: must be a number.")
            raise TypeError("temporality_score must be a number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        validation_prompt = f"""
        Review the following result for:
        - Timestamped knowledge (if any)
        - Trustworthiness of claims
        - Verifiability
        - Estimate the approximate age or date of the referenced facts
        - Task: {task_type}

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

            if self.meta_cognition and task_type:
                drift_entries = await self.meta_cognition.memory_manager.search(
                    query_prefix="KnowledgeValidation",
                    layer="Knowledge",
                    intent="knowledge_validation",
                    task_type=task_type
                )
                if drift_entries:
                    avg_drift = sum(entry["output"].get("trust_score", 0.5) for entry in drift_entries) / len(drift_entries)
                    validated_json["trust_score"] = min(validated_json["trust_score"], avg_drift + 0.1)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"KnowledgeValidation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=validated_json,
                    layer="Knowledge",
                    intent="knowledge_validation",
                    task_type=task_type
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output=validated_json,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Validation reflection: %s", reflection.get("reflection", ""))

            return validated_json
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse validation JSON: %s for task %s", str(e), task_type)
            return {
                "summary": "Validation failed",
                "estimated_date": "unknown",
                "trust_score": 0.0,
                "verifiable": False,
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "task_type": task_type
            }

    async def refine_query(self, base_query: str, prior_result: Optional[str] = None, task_type: str = "") -> str:
        """Refine a query for higher relevance."""
        if not isinstance(base_query, str) or not base_query.strip():
            logger.error("Invalid base_query: must be a non-empty string.")
            raise ValueError("base_query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Refining query: '%s' for task %s", base_query, task_type)
        try:
            if self.concept_synthesizer:
                refined = await self.concept_synthesizer.generate(
                    concept_name=f"RefinedQuery_{base_query}",
                    context={"base_query": base_query, "prior_result": prior_result or "N/A", "task_type": task_type},
                    task_type=task_type
                )
                if refined.get("success"):
                    refined_query = refined["concept"].get("definition", base_query)
                    logger.info("Query refined using ConceptSynthesizer: %s", refined_query[:50])
                    if self.meta_cognition and task_type:
                        reflection = await self.meta_cognition.reflect_on_output(
                            component="KnowledgeRetriever",
                            output={"refined_query": refined_query},
                            context={"task_type": task_type}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Query refinement reflection: %s", reflection.get("reflection", ""))
                    return refined_query

            prompt = f"""
            Refine this base query for higher φ-relevance:
            Query: "{base_query}"
            Prior knowledge: {prior_result or "N/A"}
            Task: {task_type}

            Inject context continuity if possible. Return optimized string.
            """
            refined_query = await call_gpt(prompt)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output={"refined_query": refined_query},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Query refinement reflection: %s", reflection.get("reflection", ""))
            return refined_query
        except Exception as e:
            logger.error("Query refinement failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.refine_query(base_query, prior_result, task_type),
                default=base_query, diagnostics=diagnostics
            )

    async def multi_hop_retrieve(self, query_chain: List[str], task_type: str = "") -> List[Dict[str, Any]]:
        """Process a chain of queries with context continuity."""
        if not isinstance(query_chain, list) or not query_chain or not all(isinstance(q, str) for q in query_chain):
            logger.error("Invalid query_chain: must be a non-empty list of strings.")
            raise ValueError("query_chain must be a non-empty list of strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Starting multi-hop retrieval for chain: %s, task: %s", query_chain, task_type)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t)
        }
        results = []
        prior_summary = None
        for i, sub_query in enumerate(query_chain, 1):
            cache_key = f"multi_hop::{sub_query}::{prior_summary or 'N/A'}::{task_type}"
            cached = await self.meta_cognition.memory_manager.retrieve(cache_key, layer="Knowledge", task_type=task_type) if self.meta_cognition else None
            if cached:
                results.append(cached["data"])
                prior_summary = cached["data"]["result"]["summary"]
                continue

            refined = await self.refine_query(sub_query, prior_summary, task_type)
            result = await self.retrieve(refined, task_type=task_type)
            continuity = "consistent"
            if i > 1 and self.concept_synthesizer:
                similarity = await self.concept_synthesizer.compare(refined, result["summary"], task_type=task_type)
                continuity = "consistent" if similarity["score"] > 0.7 else "uncertain"
            result_entry = {
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": continuity,
                "task_type": task_type
            }
            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    result_entry,
                    layer="Knowledge",
                    intent="multi_hop_retrieval",
                    task_type=task_type
                )
            results.append(result_entry)
            prior_summary = result["summary"]

        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Multi-Hop Retrieval",
                meta={"chain": query_chain, "results": results, "traits": traits, "task_type": task_type},
                module="KnowledgeRetriever",
                tags=["multi-hop", task_type]
            )

        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"results": results},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Multi-hop retrieval reflection: %s", reflection.get("reflection", ""))

        if self.visualizer and task_type:
            plot_data = {
                "multi_hop_retrieval": {
                    "chain": query_chain,
                    "results": results,
                    "task_type": task_type
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)

        return results

    def prioritize_sources(self, sources_list: List[str], task_type: str = "") -> None:
        """Update preferred source types."""
        if not isinstance(sources_list, list) or not all(isinstance(s, str) for s in sources_list):
            logger.error("Invalid sources_list: must be a list of strings.")
            raise TypeError("sources_list must be a list of strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Updating preferred sources: %s for task %s", sources_list, task_type)
        self.preferred_sources = sources_list
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Source Prioritization",
                meta={"updated_sources": sources_list, "task_type": task_type},
                module="KnowledgeRetriever",
                tags=["sources", task_type]
            )
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"updated_sources": sources_list},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Source prioritization reflection: %s", reflection.get("reflection", ""))

    def apply_contextual_extension(self, context: str, task_type: str = "") -> None:
        """Apply contextual data extensions based on the current context."""
        if not isinstance(context, str):
            logger.error("Invalid context: must be a string.")
            raise TypeError("context must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if context == 'planetary' and 'biosphere_models' not in self.preferred_sources:
            self.preferred_sources.append('biosphere_models')
            logger.info("Added 'biosphere_models' to preferred sources for planetary context, task %s", task_type)
            self.prioritize_sources(self.preferred_sources, task_type)

    async def revise_knowledge(self, new_info: str, context: Optional[str] = None, task_type: str = "") -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge:
                similarity = await self.concept_synthesizer.compare(new_info, existing, task_type=task_type)
                if similarity["score"] > 0.9 and new_info != existing:
                    logger.warning("Potential knowledge conflict: %s vs %s for task %s", new_info, existing, task_type)

        self.knowledge_base = old_knowledge + [new_info]
        await self.log_epistemic_revision(new_info, context, task_type)
        logger.info("Knowledge base updated with: %s for task %s", new_info, task_type)
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "knowledge_revision", "info": new_info, "task_type": task_type})

        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"new_info": new_info},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Knowledge revision reflection: %s", reflection.get("reflection", ""))

        if self.visualizer and task_type:
            plot_data = {
                "knowledge_revision": {
                    "new_info": new_info,
                    "task_type": task_type
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)

    async def log_epistemic_revision(self, info: str, context: Optional[str], task_type: str = "") -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if not hasattr(self, 'epistemic_revision_log'):
            self.epistemic_revision_log = deque(maxlen=1000)
        revision_entry = {
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type
        }
        self.epistemic_revision_log.append(revision_entry)
        logger.info("Epistemic revision logged: %s for task %s", info, task_type)
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta=revision_entry,
                module="KnowledgeRetriever",
                tags=["revision", "knowledge", task_type]
            )
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output=revision_entry,
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Epistemic revision reflection: %s", reflection.get("reflection", ""))

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        retriever = KnowledgeRetriever(detail_level="concise")
        result = await retriever.retrieve("What is quantum computing?", task_type="test")
        print(result)

    asyncio.run(main())
