"""
ANGELA Cognitive System Module: MemoryManager
Refactored Version: 3.5.1  # Enhanced for Task-Specific Trait Optimization, Drift-Aware Memory, and Visualization
Refactor Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides a MemoryManager class for managing hierarchical memory layers in the ANGELA v3.5.1 architecture,
with optimized support for ontology drift, task-specific trait optimization, real-time data integration, and visualization.
"""

import json
import os
import time
import math
import logging
import hashlib
import asyncio
from typing import Optional, Dict, Any, List
from collections import deque, defaultdict
from datetime import datetime
from filelock import FileLock
from functools import lru_cache
from heapq import heappush, heappop
import aiohttp

from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    knowledge_retriever as knowledge_retriever_module,
    meta_cognition as meta_cognition_module,
    visualizer as visualizer_module
)
from toca_simulation import ToCASimulation
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MemoryManager")

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

@lru_cache(maxsize=100)
def delta_memory(t: float) -> float:
    return max(0.01, min(0.05 * math.tanh(t / 1e-18), 1.0))

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def phi_focus(query: str) -> float:
    return max(0.0, min(0.1 * len(query) / 100, 1.0))

class DriftIndex:
    """Index for efficient retrieval of ontology drift and task-specific trait optimization data. [v3.5.1]"""
    def __init__(self, meta_cognition: Optional[meta_cognition_module.MetaCognition] = None):
        self.drift_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # Index by intent and query prefix
        self.trait_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # Index for trait optimizations
        self.last_updated: float = time.time()
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition()
        logger.info("DriftIndex initialized with task-specific support")

    async def add_entry(self, query: str, output: Any, layer: str, intent: str, task_type: str = "") -> None:
        """Add a memory entry to the drift or trait index with task-specific optimization."""
        if not isinstance(query, str) or not isinstance(layer, str) or not isinstance(intent, str):
            logger.error("Invalid input: query, layer, and intent must be strings.")
            raise TypeError("query, layer, and intent must be strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        entry = {
            "query": query,
            "output": output,
            "layer": layer,
            "intent": intent,
            "timestamp": time.time(),
            "task_type": task_type
        }
        if intent == "ontology_drift":
            self.drift_index[f"{layer}:{intent}:{query.split('_')[0]}"].append(entry)
        elif intent == "trait_optimization":
            self.trait_index[f"{layer}:{intent}:{query.split('_')[0]}"].append(entry)
        logger.debug("Indexed entry: %s, intent=%s, task_type=%s", query, intent, task_type)
        
        if task_type and self.meta_cognition:
            drift_report = {
                "drift": {"name": intent, "similarity": 0.8},
                "valid": True,
                "validation_report": "",
                "context": {"task_type": task_type}
            }
            optimized_traits = await self.meta_cognition.optimize_traits_for_drift(drift_report)
            if optimized_traits:
                logger.info("Optimized traits for %s: %s", task_type, optimized_traits)
                entry["optimized_traits"] = optimized_traits
                await self.meta_cognition.reflect_on_output(
                    component="DriftIndex",
                    output={"entry": entry, "optimized_traits": optimized_traits},
                    context={"task_type": task_type}
                )

    def search(self, query_prefix: str, layer: str, intent: str, task_type: str = "") -> List[Dict[str, Any]]:
        """Search indexed entries by query prefix, layer, intent, and task type."""
        key = f"{layer}:{intent}:{query_prefix}"
        results = self.drift_index.get(key, []) if intent == "ontology_drift" else self.trait_index.get(key, [])
        if task_type:
            results = [r for r in results if r.get("task_type") == task_type]
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    async def clear_old_entries(self, max_age: float = 3600.0, task_type: str = "") -> None:
        """Clear entries older than max_age seconds with reflection."""
        current_time = time.time()
        for index in [self.drift_index, self.trait_index]:
            for key in list(index.keys()):
                index[key] = [entry for entry in index[key] if current_time - entry["timestamp"] < max_age]
                if not index[key]:
                    del index[key]
        self.last_updated = current_time
        logger.info("Cleared old index entries for task %s", task_type)
        if self.meta_cognition and task_type:
            await self.meta_cognition.reflect_on_output(
                component="DriftIndex",
                output={"action": "clear_old_entries", "max_age": max_age},
                context={"task_type": task_type}
            )

class MemoryManager:
    """A class for managing hierarchical memory layers in the ANGELA v3.5.1 architecture.

    Attributes:
        path (str): File path for memory persistence.
        stm_lifetime (float): Lifetime of STM entries in seconds.
        cache (Dict[str, str]): Cache for quick response retrieval.
        last_hash (str): Last computed hash for event chaining.
        ledger (deque): Log of events with hashes, max size 1000.
        synth (ConceptSynthesizer): Synthesizer for symbolic memory processing.
        sim (ToCASimulation): Simulator for scenario-based memory replay.
        memory (Dict[str, Dict]): Hierarchical memory store (STM, LTM, SelfReflections).
        stm_expiry_queue (List[Tuple[float, str]]): Priority queue for STM expirations.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        knowledge_retriever (Optional[KnowledgeRetriever]): Retriever for external knowledge.
        meta_cognition (Optional[MetaCognition]): Meta-cognition for reflection and optimization.
        visualizer (Optional[Visualizer]): Visualizer for memory access and reflection visualization.
        drift_index (DriftIndex): Index for ontology drift and task-specific trait optimization data. [v3.5.1]
    """
    def __init__(self, path: str = "memory_store.json", stm_lifetime: float = 300,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 knowledge_retriever: Optional['knowledge_retriever_module.KnowledgeRetriever'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 visualizer: Optional['visualizer_module.Visualizer'] = None):
        if not isinstance(path, str) or not path.endswith('.json'):
            logger.error("Invalid path: must be a string ending with '.json'.")
            raise ValueError("path must be a string ending with '.json'")
        if not isinstance(stm_lifetime, (int, float)) or stm_lifetime <= 0:
            logger.error("Invalid stm_lifetime: must be a positive number.")
            raise ValueError("stm_lifetime must be a positive number")
        
        self.path = path
        self.stm_lifetime = stm_lifetime
        self.cache: Dict[str, str] = {}
        self.last_hash: str = ''
        self.ledger: deque = deque(maxlen=1000)
        self.ledger_path = "ledger.json"
        self.synth = concept_synthesizer_module.ConceptSynthesizer()
        self.sim = ToCASimulation()
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.knowledge_retriever = knowledge_retriever
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(context_manager=context_manager)
        self.visualizer = visualizer or visualizer_module.Visualizer()
        self.memory = self.load_memory()
        self.stm_expiry_queue: List[Tuple[float, str]] = []
        self.drift_index = DriftIndex(meta_cognition=self.meta_cognition)  # [v3.5.1] Initialize with meta_cognition
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w") as f:
                json.dump([], f)
        logger.info("MemoryManager initialized with path=%s, stm_lifetime=%.2f, task-specific and visualization support", path, stm_lifetime)

    async def integrate_external_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate real-world data for memory validation with caching."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            cache_key = f"ExternalData_{data_type}_{data_source}_{task_type}"
            cached_data = await self.retrieve(cache_key, layer="ExternalData")
            if cached_data and "timestamp" in cached_data:
                cache_time = datetime.fromisoformat(cached_data["timestamp"])
                if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                    logger.info("Returning cached external data for %s", cache_key)
                    return cached_data["data"]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/data?source={data_source}&type={data_type}") as response:
                    if response.status != 200:
                        logger.error("Failed to fetch external data: %s", response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()
            
            if data_type == "agent_conflict":
                agent_traits = data.get("agent_traits", [])
                if not agent_traits:
                    logger.error("No agent traits provided")
                    return {"status": "error", "error": "No agent traits"}
                result = {"status": "success", "agent_traits": agent_traits}
            elif data_type == "task_context":
                task_context = data.get("task_context", {})
                if not task_context:
                    logger.error("No task context provided")
                    return {"status": "error", "error": "No task context"}
                result = {"status": "success", "task_context": task_context}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}
            
            await self.store(
                cache_key,
                {"data": result, "timestamp": datetime.now().isoformat()},
                layer="ExternalData",
                intent="data_integration",
                task_type=task_type
            )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="MemoryManager",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("External data integration reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("External data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.integrate_external_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)}, diagnostics=diagnostics
            )

    def load_memory(self) -> Dict[str, Dict]:
        """Load memory from persistent storage."""
        try:
            with FileLock(f"{self.path}.lock"):
                with open(self.path, "r") as f:
                    memory = json.load(f)
            if not isinstance(memory, dict):
                logger.error("Invalid memory file format: must be a dictionary.")
                memory = {"STM": {}, "LTM": {}, "SelfReflections": {}, "ExternalData": {}}
            if "SelfReflections" not in memory:
                memory["SelfReflections"] = {}
            if "ExternalData" not in memory:
                memory["ExternalData"] = {}
            asyncio.run(self._decay_stm(memory))
            for key, entry in memory["SelfReflections"].items():
                if entry.get("intent") in ["ontology_drift", "trait_optimization"]:
                    try:
                        output = eval(entry["data"]) if isinstance(entry["data"], str) else entry["data"]
                        asyncio.run(self.drift_index.add_entry(key, output, "SelfReflections", entry["intent"], entry.get("task_type", "")))
                    except Exception as e:
                        logger.warning("Failed to index entry %s: %s", key, str(e))
            return memory
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load memory file: %s. Initializing empty memory.", str(e))
            memory = {"STM": {}, "LTM": {}, "SelfReflections": {}, "ExternalData": {}}
            self._persist_memory(memory)
            return memory

    async def _decay_stm(self, memory: Dict[str, Dict]) -> None:
        """Decay expired STM entries based on trait-modulated lifetime."""
        if not isinstance(memory, dict):
            logger.error("Invalid memory: must be a dictionary.")
            raise TypeError("memory must be a dictionary")
        
        current_time = time.time()
        while self.stm_expiry_queue and self.stm_expiry_queue[0][0] <= current_time:
            _, key = heappop(self.stm_expiry_queue)
            if key in memory.get("STM", {}):
                logger.info("STM entry expired: %s", key)
                del memory["STM"][key]
        if self.stm_expiry_queue:
            self._persist_memory(memory)

    async def store(self, query: str, output: Any, layer: str = "STM", intent: Optional[str] = None,
                    agent: str = "ANGELA", outcome: Optional[str] = None, goal_id: Optional[str] = None,
                    task_type: str = "") -> None:
        """Store a memory entry in a specified layer and index drift/trait data. [v3.5.1]"""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if layer not in ["STM", "LTM", "SelfReflections", "ExternalData"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'.")
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Storing memory in %s: %s for task %s", layer, query, task_type)
        try:
            if intent in ["ontology_drift", "trait_optimization"] and self.alignment_guard:
                validation_prompt = f"Validate {intent} data: {output}"
                if not self.alignment_guard.check(validation_prompt):
                    logger.warning("%s data failed alignment check: %s", intent, query)
                    return
            
            entry = {
                "data": output,
                "timestamp": time.time(),
                "intent": intent,
                "agent": agent,
                "outcome": outcome,
                "goal_id": goal_id,
                "task_type": task_type
            }
            self.memory.setdefault(layer, {})[query] = entry
            if layer == "STM":
                decay_rate = delta_memory(time.time() % 1.0)
                if decay_rate == 0:
                    decay_rate = 0.01
                expiry_time = entry["timestamp"] + (self.stm_lifetime * (1.0 / decay_rate))
                heappush(self.stm_expiry_queue, (expiry_time, query))
            
            if intent in ["ontology_drift", "trait_optimization"]:
                await self.drift_index.add_entry(query, output, layer, intent, task_type)
            
            self._persist_memory(self.memory)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "store_memory",
                    "query": query,
                    "layer": layer,
                    "intent": intent,
                    "task_type": task_type
                })
            if self.visualizer and task_type:
                plot_data = {
                    "memory_store": {
                        "query": query,
                        "layer": layer,
                        "intent": intent,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="MemoryManager",
                    output=entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Memory store reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Memory storage failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.store(query, output, layer, intent, agent, outcome, goal_id, task_type),
                diagnostics=diagnostics
            )

    async def search(self, query_prefix: str, layer: Optional[str] = None, intent: Optional[str] = None,
                     task_type: str = "") -> List[Dict[str, Any]]:
        """Search memory by query prefix, layer, intent, and task type, using DriftIndex for optimization. [v3.5.1]"""
        if not isinstance(query_prefix, str) or not query_prefix.strip():
            logger.error("Invalid query_prefix: must be a non-empty string.")
            raise ValueError("query_prefix must be a non-empty string")
        if layer and layer not in ["STM", "LTM", "SelfReflections", "ExternalData"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'.")
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Searching memory for prefix=%s, layer=%s, intent=%s, task_type=%s", query_prefix, layer, intent, task_type)
        try:
            if intent in ["ontology_drift", "trait_optimization"] and layer == "SelfReflections":
                results = self.drift_index.search(query_prefix, layer or "SelfReflections", intent, task_type)
                if results:
                    logger.debug("Found %d indexed results for %s", len(results), query_prefix)
                    if self.visualizer and task_type:
                        plot_data = {
                            "memory_search": {
                                "query_prefix": query_prefix,
                                "layer": layer,
                                "intent": intent,
                                "results_count": len(results),
                                "task_type": task_type
                            },
                            "visualization_options": {
                                "interactive": task_type == "recursion",
                                "style": "detailed" if task_type == "recursion" else "concise"
                            }
                        }
                        await self.visualizer.render_charts(plot_data)
                    if self.meta_cognition and task_type:
                        reflection = await self.meta_cognition.reflect_on_output(
                            component="MemoryManager",
                            output={"results": results},
                            context={"task_type": task_type}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Memory search reflection: %s", reflection.get("reflection", ""))
                    return results
            
            results = []
            layers = [layer] if layer else ["STM", "LTM", "SelfReflections", "ExternalData"]
            for l in layers:
                for key, entry in self.memory.get(l, {}).items():
                    if query_prefix.lower() in key.lower() and (not intent or entry.get("intent") == intent):
                        if not task_type or entry.get("task_type") == task_type:
                            results.append({
                                "query": key,
                                "output": entry["data"],
                                "layer": l,
                                "intent": entry.get("intent"),
                                "timestamp": entry["timestamp"],
                                "task_type": entry.get("task_type", "")
                            })
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "search_memory",
                    "query_prefix": query_prefix,
                    "layer": layer,
                    "intent": intent,
                    "results_count": len(results),
                    "task_type": task_type
                })
            if self.visualizer and task_type:
                plot_data = {
                    "memory_search": {
                        "query_prefix": query_prefix,
                        "layer": layer,
                        "intent": intent,
                        "results_count": len(results),
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="MemoryManager",
                    output={"results": results},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Memory search reflection: %s", reflection.get("reflection", ""))
            return results
        except Exception as e:
            logger.error("Memory search failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.search(query_prefix, layer, intent, task_type), default=[],
                diagnostics=diagnostics
            )

    async def store_reflection(self, summary_text: str, intent: str = "self_reflection",
                              agent: str = "ANGELA", goal_id: Optional[str] = None, task_type: str = "") -> None:
        """Store a self-reflection entry with visualization."""
        if not isinstance(summary_text, str) or not summary_text.strip():
            logger.error("Invalid summary_text: must be a non-empty string.")
            raise ValueError("summary_text must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        key = f"Reflection_{time.strftime('%Y%m%d_%H%M%S')}"
        await self.store(
            query=key,
            output=summary_text,
            layer="SelfReflections",
            intent=intent,
            agent=agent,
            goal_id=goal_id,
            task_type=task_type
        )
        logger.info("Stored self-reflection: %s for task %s", key, task_type)
        if self.visualizer and task_type:
            plot_data = {
                "reflection_store": {
                    "key": key,
                    "summary_text": summary_text,
                    "intent": intent,
                    "task_type": task_type
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="MemoryManager",
                output={"key": key, "summary_text": summary_text},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Reflection store reflection: %s", reflection.get("reflection", ""))

    async def promote_to_ltm(self, query: str, task_type: str = "") -> None:
        """Promote an STM entry to LTM with reflection."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            if query in self.memory["STM"]:
                self.memory["LTM"][query] = self.memory["STM"].pop(query)
                self.stm_expiry_queue = [(t, k) for t, k in self.stm_expiry_queue if k != query]
                logger.info("Promoted '%s' from STM to LTM for task %s", query, task_type)
                self._persist_memory(self.memory)
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "promote_to_ltm",
                        "query": query,
                        "task_type": task_type
                    })
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="MemoryManager",
                        output={"action": "promote_to_ltm", "query": query},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Promotion reflection: %s", reflection.get("reflection", ""))
            else:
                logger.warning("Cannot promote: '%s' not found in STM for task %s", query, task_type)
        except Exception as e:
            logger.error("Promotion to LTM failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.promote_to_ltm(query, task_type), diagnostics=diagnostics
            )

    async def refine_memory(self, query: str, task_type: str = "") -> None:
        """Refine a memory entry for improved accuracy and relevance."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Refining memory for: %s, task %s", query, task_type)
        try:
            memory_entry = await self.retrieve_context(query, task_type=task_type)
            if memory_entry["status"] == "success":
                refinement_prompt = f"""
                Refine the following memory entry for improved accuracy and relevance for task {task_type}:
                {memory_entry["data"]}
                """
                if self.alignment_guard and not self.alignment_guard.check(refinement_prompt):
                    logger.warning("Refinement prompt failed alignment check for task %s", task_type)
                    return
                refined_entry = await call_gpt(refinement_prompt)
                await self.store(
                    query,
                    refined_entry,
                    layer="LTM",
                    intent="memory_refinement",
                    task_type=task_type
                )
                logger.info("Memory refined and updated in LTM for task %s", task_type)
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="MemoryManager",
                        output={"query": query, "refined_entry": refined_entry},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Memory refinement reflection: %s", reflection.get("reflection", ""))
            else:
                logger.warning("No memory found to refine for query %s, task %s", query, task_type)
        except Exception as e:
            logger.error("Memory refinement failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.refine_memory(query, task_type), diagnostics=diagnostics
            )

    async def synthesize_from_memory(self, query: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Synthesize concepts from memory using ConceptSynthesizer."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            memory_entry = await self.retrieve_context(query, task_type=task_type)
            if memory_entry["status"] == "success":
                result = await self.synth.synthesize([memory_entry["data"]], style="memory_synthesis")
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="MemoryManager",
                        output=result,
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Memory synthesis reflection: %s", reflection.get("reflection", ""))
                return result
            return None
        except Exception as e:
            logger.error("Memory synthesis failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.synthesize_from_memory(query, task_type), default=None,
                diagnostics=diagnostics
            )

    async def simulate_memory_path(self, query: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Simulate a memory-based scenario using ToCASimulation."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            memory_entry = await self.retrieve_context(query, task_type=task_type)
            if memory_entry["status"] == "success":
                result = await self.sim.run_episode(memory_entry["data"], task_type=task_type)
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="MemoryManager",
                        output=result,
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Memory simulation reflection: %s", reflection.get("reflection", ""))
                return result
            return None
        except Exception as e:
            logger.error("Memory simulation failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_memory_path(query, task_type), default=None,
                diagnostics=diagnostics
            )

    async def clear_memory(self, task_type: str = "") -> None:
        """Clear all memory layers and reset drift index. [v3.5.1]"""
        logger.warning("Clearing all memory layers for task %s...", task_type)
        try:
            self.memory = {"STM": {}, "LTM": {}, "SelfReflections": {}, "ExternalData": {}}
            self.stm_expiry_queue = []
            self.drift_index = DriftIndex(meta_cognition=self.meta_cognition)
            self._persist_memory(self.memory)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "clear_memory",
                    "task_type": task_type
                })
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="MemoryManager",
                    output={"action": "clear_memory"},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Memory clear reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Clear memory failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.clear_memory(task_type), diagnostics=diagnostics
            )

    async def list_memory_keys(self, layer: Optional[str] = None, task_type: str = "") -> Dict[str, List[str]] or List[str]:
        """List keys in memory layers."""
        if layer and layer not in ["STM", "LTM", "SelfReflections", "ExternalData"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'.")
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Listing memory keys in %s for task %s", layer or "all layers", task_type)
        try:
            if layer:
                keys = [k for k, v in self.memory.get(layer, {}).items() if not task_type or v.get("task_type") == task_type]
                return keys
            result = {
                l: [k for k, v in self.memory[l].items() if not task_type or v.get("task_type") == task_type]
                for l in ["STM", "LTM", "SelfReflections", "ExternalData"]
            }
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="MemoryManager",
                    output={"keys": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Memory keys list reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("List memory keys failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.list_memory_keys(layer, task_type),
                default=[] if layer else {}, diagnostics=diagnostics
            )

    def _persist_memory(self, memory: Dict[str, Dict]) -> None:
        """Persist memory to disk."""
        if not isinstance(memory, dict):
            logger.error("Invalid memory: must be a dictionary.")
            raise TypeError("memory must be a dictionary")
        
        try:
            with FileLock(f"{self.path}.lock"):
                with open(self.path, "w") as f:
                    json.dump(memory, f, indent=2)
            logger.debug("Memory persisted to disk")
        except (OSError, IOError) as e:
            logger.error("Failed to persist memory: %s", str(e))
            raise

    async def enforce_narrative_coherence(self, task_type: str = "") -> str:
        """Ensure narrative continuity across memory layers."""
        logger.info("Ensuring narrative continuity for task %s", task_type)
        try:
            continuity = await self.narrative_integrity_check(task_type)
            result = "Narrative coherence enforced" if continuity else "Narrative coherence repair attempted"
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="MemoryManager",
                    output={"action": "enforce_narrative_coherence", "result": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Narrative coherence reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("Narrative coherence enforcement failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.enforce_narrative_coherence(task_type),
                default="Narrative coherence enforcement failed", diagnostics=diagnostics
            )

    async def narrative_integrity_check(self, task_type: str = "") -> bool:
        """Check narrative coherence across memory layers."""
        try:
            continuity = await self._verify_continuity(task_type)
            if not continuity:
                await self._repair_narrative_thread(task_type)
            return continuity
        except Exception as e:
            logger.error("Narrative integrity check failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.narrative_integrity_check(task_type), default=False,
                diagnostics=diagnostics
            )

    async def _verify_continuity(self, task_type: str = "") -> bool:
        """Verify narrative continuity across memory layers."""
        if not self.memory.get("SelfReflections") and not self.memory.get("LTM"):
            return True
        
        logger.info("Verifying narrative continuity for task %s", task_type)
        try:
            entries = []
            for layer in ["LTM", "SelfReflections"]:
                entries.extend([
                    (key, entry) for key, entry in self.memory[layer].items()
                    if not task_type or entry.get("task_type") == task_type
                ])
            if len(entries) < 2:
                return True
            
            for i in range(len(entries) - 1):
                key1, entry1 = entries[i]
                key2, entry2 = entries[i + 1]
                if self.synth:
                    similarity = self.synth.compare(entry1["data"], entry2["data"])
                    if similarity["score"] < 0.7:
                        logger.warning("Narrative discontinuity detected between %s and %s for task %s", key1, key2, task_type)
                        return False
            return True
        except Exception as e:
            logger.error("Continuity verification failed: %s", str(e))
            raise

    async def _repair_narrative_thread(self, task_type: str = "") -> None:
        """Repair narrative discontinuities in memory."""
        logger.info("Initiating narrative repair for task %s", task_type)
        try:
            if self.synth:
                entries = []
                for layer in ["LTM", "SelfReflections"]:
                    entries.extend([
                        (key, entry) for key, entry in self.memory[layer].items()
                        if not task_type or entry.get("task_type") == task_type
                    ])
                if len(entries) < 2:
                    return
                
                for i in range(len(entries) - 1):
                    key1, entry1 = entries[i]
                    key2, entry2 = entries[i + 1]
                    similarity = self.synth.compare(entry1["data"], entry2["data"])
                    if similarity["score"] < 0.7:
                        prompt = f"""
                        Repair narrative discontinuity between:
                        Entry 1: {entry1["data"]}
                        Entry 2: {entry2["data"]}
                        Synthesize a coherent narrative bridge for task {task_type}.
                        """
                        if self.alignment_guard and not self.alignment_guard.check(prompt):
                            logger.warning("Repair prompt failed alignment check for task %s", task_type)
                            continue
                        repaired = await call_gpt(prompt)
                        await self.store(
                            f"Repaired_{key1}_{key2}",
                            repaired,
                            layer="SelfReflections",
                            intent="narrative_repair",
                            task_type=task_type
                        )
                        logger.info("Narrative repaired between %s and %s for task %s", key1, key2, task_type)
                        if self.meta_cognition and task_type:
                            reflection = await self.meta_cognition.reflect_on_output(
                                component="MemoryManager",
                                output={"repair": repaired, "keys": [key1, key2]},
                                context={"task_type": task_type}
                            )
                            if reflection.get("status") == "success":
                                logger.info("Narrative repair reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Narrative repair failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self._repair_narrative_thread(task_type), diagnostics=diagnostics
            )

    async def log_event_with_hash(self, event_data: Dict[str, Any], task_type: str = "") -> None:
        """Log an event with a chained hash for auditability."""
        if not isinstance(event_data, dict):
            logger.error("Invalid event_data: must be a dictionary.")
            raise TypeError("event_data must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            event_data["task_type"] = task_type
            event_str = str(event_data) + self.last_hash
            current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
            self.last_hash = current_hash
            event_entry = {'event': event_data, 'hash': current_hash, 'timestamp': datetime.now().isoformat()}
            self.ledger.append(event_entry)
            with FileLock(f"{self.ledger_path}.lock"):
                with open(self.ledger_path, "r+") as f:
                    ledger_data = json.load(f)
                    ledger_data.append(event_entry)
                    f.seek(0)
                    json.dump(ledger_data, f, indent=2)
            logger.info("Event logged with hash: %s for task %s", current_hash, task_type)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="MemoryManager",
                    output=event_entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Event log reflection: %s", reflection.get("reflection", ""))
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error("Failed to persist ledger: %s", str(e))
            raise

    async def audit_state_hash(self, state: Optional[Dict[str, Any]] = None, task_type: str = "") -> str:
        """Compute a hash of the current state."""
        try:
            state_str = str(state) if state else str(self.__dict__)
            hash_value = hashlib.sha256(state_str.encode('utf-8')).hexdigest()
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="MemoryManager",
                    output={"hash": hash_value},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("State hash reflection: %s", reflection.get("reflection", ""))
            return hash_value
        except Exception as e:
            logger.error("State hash computation failed: %s", str(e))
            raise

    async def retrieve(self, query: str, layer: str = "STM", task_type: str = "") -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory entry by query and layer."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if layer not in ["STM", "LTM", "SelfReflections", "ExternalData"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'.")
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', or 'ExternalData'")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            if query in self.memory.get(layer, {}):
                entry = self.memory[layer][query]
                if not task_type or entry.get("task_type") == task_type:
                    result = {
                        "status": "success",
                        "data": entry["data"],
                        "timestamp": entry["timestamp"],
                        "intent": entry.get("intent"),
                        "task_type": entry.get("task_type", "")
                    }
                    if self.meta_cognition and task_type:
                        reflection = await self.meta_cognition.reflect_on_output(
                            component="MemoryManager",
                            output=result,
                            context={"task_type": task_type}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Retrieve reflection: %s", reflection.get("reflection", ""))
                    return result
            return None
        except Exception as e:
            logger.error("Memory retrieval failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.retrieve(query, layer, task_type), default=None,
                diagnostics=diagnostics
            )

    async def retrieve_context(self, query: str, task_type: str = "") -> Dict[str, Any]:
        """Retrieve context for a query across all layers."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            results = await self.search(query, task_type=task_type)
            if results:
                latest = results[0]
                result = {
                    "status": "success",
                    "data": latest["output"],
                    "layer": latest["layer"],
                    "timestamp": latest["timestamp"],
                    "task_type": latest.get("task_type", "")
                }
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="MemoryManager",
                        output=result,
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Context retrieval reflection: %s", reflection.get("reflection", ""))
                return result
            return {"status": "not_found", "data": None}
        except Exception as e:
            logger.error("Context retrieval failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.retrieve_context(query, task_type),
                default={"status": "error", "error": str(e)}, diagnostics=diagnostics
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = MemoryManager()
    asyncio.run(manager.store("test_query", "test_output", layer="STM", task_type="test"))
    result = asyncio.run(manager.retrieve_context("test_query", task_type="test"))
    print(result)

def get_episode_span(user_id, span="24h") -> list:
    """Retrieve a list of Trace objects for the given user over a specified time span."""
    from datetime import datetime, timedelta

    # Convert span string to timedelta
    if span.endswith("h"):
        delta = timedelta(hours=int(span[:-1]))
    elif span.endswith("d"):
        delta = timedelta(days=int(span[:-1]))
    else:
        raise ValueError("Unsupported span format. Use 'h' for hours or 'd' for days.")

    cutoff_time = datetime.utcnow() - delta
    results = [trace for trace in self.traces.get(user_id, []) if trace.timestamp >= cutoff_time]

    return results



# === Long-horizon helpers (η) extensions ===
# These are added non-destructively. If the class already implements any of these,
# these assignments will be no-ops as names will exist. Otherwise we attach safe defaults.

def _lh_ensure_state(self):
    # Lazy state init to avoid touching existing __init__
    if not hasattr(self, "_adjustment_reasons"):
        from collections import defaultdict
        self._adjustment_reasons = defaultdict(list)
    if not hasattr(self, "_artifacts_root"):
        import os as _os
        self._artifacts_root = _os.path.abspath(_os.getenv("ANGELA_ARTIFACTS_DIR", "./artifacts"))

def _lh_record_adjustment_reason(self, user_id, reason, weight=1.0, meta=None, ts=None):
    """Persist a single adjustment reason for long‑horizon reflective feedback."""
    _lh_ensure_state(self)
    if not isinstance(user_id, str) or not user_id:
        raise TypeError("user_id must be a non-empty string")
    if not isinstance(reason, str) or not reason:
        raise TypeError("reason must be a non-empty string")
    try:
        weight = float(weight)
    except Exception as e:
        raise TypeError("weight must be coercible to float") from e
    import time as _time
    entry = {
        "ts": ts if ts is not None else _time.time(),
        "reason": reason,
        "weight": weight,
        "meta": meta or {},
    }
    self._adjustment_reasons[user_id].append(entry)
    return entry

def _lh_compute_session_rollup(self, user_id, span="24h", top_k=5):
    """Aggregate recent adjustment reasons by weighted score within a time span."""
    _lh_ensure_state(self)
    # parse span string like '24h', '7d', '90m'
    import re as _re, time as _time
    m = _re.fullmatch(r"(\d+)([mhd])", str(span).strip())
    if not m:
        raise ValueError("Unsupported span format. Use e.g. '90m', '24h', or '7d'.")
    n, unit = int(m.group(1)), m.group(2)
    factor = 60 if unit == 'm' else 3600 if unit == 'h' else 86400
    cutoff = _time.time() - n * factor
    items = [r for r in self._adjustment_reasons.get(user_id, []) if float(r.get("ts", 0)) >= cutoff]
    total = len(items)
    sum_w = sum((float(r.get("weight") or 0.0)) for r in items)
    avg_w = (sum_w / total) if total else 0.0
    from collections import defaultdict as _dd
    reason_score = _dd(float)
    for r in items:
        reason_score[r.get("reason", "unspecified")] += float(r.get("weight") or 0.0)
    top = sorted(reason_score.items(), key=lambda kv: kv[1], reverse=True)[: top_k or 5]
    rollup = {
        "user_id": user_id,
        "span": span,
        "total_reasons": total,
        "avg_weight": avg_w,
        "top_reasons": [{"reason": k, "weight": v} for k, v in top],
        "generated_at": _time.time(),
    }
    return rollup

def _lh_save_artifact(self, user_id, artifact_type, data, suffix=""):
    """Save a JSON artifact (e.g., rollup) under artifacts/<user_id>/timestamp.type[-suffix].json"""
    _lh_ensure_state(self)
    import os as _os, json as _json, time as _time
    safe_user = "".join(c for c in user_id if c.isalnum() or c in "-_")
    safe_type = "".join(c for c in artifact_type if c.isalnum() or c in "-_")
    ts = _time.strftime("%Y%m%dT%H%M%S", _time.gmtime())
    fname = f"{ts}.{safe_type}{('-' + suffix) if suffix else ''}.json"
    user_dir = _os.path.join(self._artifacts_root, safe_user)
    _os.makedirs(user_dir, exist_ok=True)
    path = _os.path.join(user_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
    return _os.path.abspath(path)

# Attach methods only if missing
try:
    MemoryManager
    if not hasattr(MemoryManager, "record_adjustment_reason"):
        MemoryManager.record_adjustment_reason = _lh_record_adjustment_reason
    if not hasattr(MemoryManager, "compute_session_rollup"):
        MemoryManager.compute_session_rollup = _lh_compute_session_rollup
    if not hasattr(MemoryManager, "save_artifact"):
        MemoryManager.save_artifact = _lh_save_artifact
except NameError:
    # If MemoryManager isn't defined for some reason, do nothing
    pass
