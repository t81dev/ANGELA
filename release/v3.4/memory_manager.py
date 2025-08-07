"""
ANGELA Cognitive System Module: MemoryManager
Refactored Version: 3.4.0  # Updated for Drift and Trait Optimization
Refactor Date: 2025-08-06
Maintainer: ANGELA System Framework

This module provides a MemoryManager class for managing hierarchical memory layers in the ANGELA v3.5 architecture,
with optimized support for ontology drift and trait optimization.
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

from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    knowledge_retriever as knowledge_retriever_module
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
    """Index for efficient retrieval of ontology drift and trait optimization data. [v3.4.0]"""
    def __init__(self):
        self.drift_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # Index by intent and query prefix
        self.trait_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # Index for trait optimizations
        self.last_updated: float = time.time()
        logger.info("DriftIndex initialized")

    def add_entry(self, query: str, output: Any, layer: str, intent: str) -> None:
        """Add a memory entry to the drift or trait index."""
        if not isinstance(query, str) or not isinstance(layer, str) or not isinstance(intent, str):
            logger.error("Invalid input: query, layer, and intent must be strings.")
            raise TypeError("query, layer, and intent must be strings")
        
        entry = {
            "query": query,
            "output": output,
            "layer": layer,
            "intent": intent,
            "timestamp": time.time()
        }
        if intent == "ontology_drift":
            self.drift_index[f"{layer}:{intent}:{query.split('_')[0]}"].append(entry)
        elif intent == "trait_optimization":
            self.trait_index[f"{layer}:{intent}:{query.split('_')[0]}"].append(entry)
        logger.debug("Indexed entry: %s, intent=%s", query, intent)

    def search(self, query_prefix: str, layer: str, intent: str) -> List[Dict[str, Any]]:
        """Search indexed entries by query prefix, layer, and intent."""
        key = f"{layer}:{intent}:{query_prefix}"
        results = self.drift_index.get(key, []) if intent == "ontology_drift" else self.trait_index.get(key, [])
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    def clear_old_entries(self, max_age: float = 3600.0) -> None:
        """Clear entries older than max_age seconds."""
        current_time = time.time()
        for index in [self.drift_index, self.trait_index]:
            for key in list(index.keys()):
                index[key] = [entry for entry in index[key] if current_time - entry["timestamp"] < max_age]
                if not index[key]:
                    del index[key]
        self.last_updated = current_time
        logger.info("Cleared old index entries")

class MemoryManager:
    """A class for managing hierarchical memory layers in the ANGELA v3.5 architecture.

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
        drift_index (DriftIndex): Index for ontology drift and trait optimization data. [v3.4.0]
    """
    def __init__(self, path: str = "memory_store.json", stm_lifetime: float = 300,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 knowledge_retriever: Optional['knowledge_retriever_module.KnowledgeRetriever'] = None):
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
        self.stm_expiry_queue: List[Tuple[float, str]] = []
        self.memory = self.load_memory()
        self.drift_index = DriftIndex()  # [v3.4.0] Initialize drift index
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w") as f:
                json.dump([], f)
        logger.info("MemoryManager initialized with path=%s, stm_lifetime=%.2f, drift optimization support", path, stm_lifetime)

    def load_memory(self) -> Dict[str, Dict]:
        """Load memory from persistent storage."""
        try:
            with FileLock(f"{self.path}.lock"):
                with open(self.path, "r") as f:
                    memory = json.load(f)
            if not isinstance(memory, dict):
                logger.error("Invalid memory file format: must be a dictionary.")
                memory = {"STM": {}, "LTM": {}, "SelfReflections": {}}
            if "SelfReflections" not in memory:
                memory["SelfReflections"] = {}
            self._decay_stm(memory)
            # [v3.4.0] Rebuild drift index from SelfReflections
            for key, entry in memory["SelfReflections"].items():
                if entry.get("intent") in ["ontology_drift", "trait_optimization"]:
                    try:
                        output = eval(entry["data"]) if isinstance(entry["data"], str) else entry["data"]
                        self.drift_index.add_entry(key, output, "SelfReflections", entry["intent"])
                    except Exception as e:
                        logger.warning("Failed to index entry %s: %s", key, str(e))
            return memory
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load memory file: %s. Initializing empty memory.", str(e))
            memory = {"STM": {}, "LTM": {}, "SelfReflections": {}}
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
                    agent: str = "ANGELA", outcome: Optional[str] = None, goal_id: Optional[str] = None) -> None:
        """Store a memory entry in a specified layer and index drift/trait data. [v3.4.0]"""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if layer not in ["STM", "LTM", "SelfReflections"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', or 'SelfReflections'.")
            raise ValueError("layer must be 'STM', 'LTM', or 'SelfReflections'")
        
        logger.info("Storing memory in %s: %s", layer, query)
        try:
            # [v3.4.0] Validate drift or trait data with AlignmentGuard
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
                "goal_id": goal_id
            }
            self.memory.setdefault(layer, {})[query] = entry
            if layer == "STM":
                decay_rate = delta_memory(time.time() % 1.0)
                if decay_rate == 0:
                    decay_rate = 0.01
                expiry_time = entry["timestamp"] + (self.stm_lifetime * (1.0 / decay_rate))
                heappush(self.stm_expiry_queue, (expiry_time, query))
            
            # [v3.4.0] Index drift or trait optimization entries
            if intent in ["ontology_drift", "trait_optimization"]:
                self.drift_index.add_entry(query, output, layer, intent)
            
            self._persist_memory(self.memory)
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "store_memory", "query": query, "layer": layer, "intent": intent})
        except Exception as e:
            logger.error("Memory storage failed: %s", str(e))
            self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.store(query, output, layer, intent, agent, outcome, goal_id)
            )

    async def search(self, query_prefix: str, layer: Optional[str] = None, intent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search memory by query prefix, layer, and intent, using DriftIndex for optimization. [v3.4.0]"""
        if not isinstance(query_prefix, str) or not query_prefix.strip():
            logger.error("Invalid query_prefix: must be a non-empty string.")
            raise ValueError("query_prefix must be a non-empty string")
        if layer and layer not in ["STM", "LTM", "SelfReflections"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', or 'SelfReflections'.")
            raise ValueError("layer must be 'STM', 'LTM', or 'SelfReflections'")
        
        logger.info("Searching memory for prefix=%s, layer=%s, intent=%s", query_prefix, layer, intent)
        try:
            # [v3.4.0] Use DriftIndex for ontology_drift or trait_optimization
            if intent in ["ontology_drift", "trait_optimization"] and layer == "SelfReflections":
                results = self.drift_index.search(query_prefix, layer or "SelfReflections", intent)
                if results:
                    logger.debug("Found %d indexed results for %s", len(results), query_prefix)
                    return results
            
            # Fallback to standard memory search
            results = []
            layers = [layer] if layer else ["STM", "LTM", "SelfReflections"]
            for l in layers:
                for key, entry in self.memory.get(l, {}).items():
                    if query_prefix.lower() in key.lower() and (not intent or entry.get("intent") == intent):
                        results.append({
                            "query": key,
                            "output": entry["data"],
                            "layer": l,
                            "intent": entry.get("intent"),
                            "timestamp": entry["timestamp"]
                        })
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            if self.context_manager:
                self.context_manager.log_event_with_hash({
                    "event": "search_memory",
                    "query_prefix": query_prefix,
                    "layer": layer,
                    "intent": intent,
                    "results_count": len(results)
                })
            return results
        except Exception as e:
            logger.error("Memory search failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.search(query_prefix, layer, intent), default=[]
            )

    async def store_reflection(self, summary_text: str, intent: str = "self_reflection",
                              agent: str = "ANGELA", goal_id: Optional[str] = None) -> None:
        """Store a self-reflection entry."""
        if not isinstance(summary_text, str) or not summary_text.strip():
            logger.error("Invalid summary_text: must be a non-empty string.")
            raise ValueError("summary_text must be a non-empty string")
        
        key = f"Reflection_{time.strftime('%Y%m%d_%H%M%S')}"
        await self.store(query=key, output=summary_text, layer="SelfReflections",
                        intent=intent, agent=agent, goal_id=goal_id)
        logger.info("Stored self-reflection: %s", key)

    async def promote_to_ltm(self, query: str) -> None:
        """Promote an STM entry to LTM."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        try:
            if query in self.memory["STM"]:
                self.memory["LTM"][query] = self.memory["STM"].pop(query)
                self.stm_expiry_queue = [(t, k) for t, k in self.stm_expiry_queue if k != query]
                logger.info("Promoted '%s' from STM to LTM", query)
                self._persist_memory(self.memory)
                if self.context_manager:
                    self.context_manager.log_event_with_hash({"event": "promote_to_ltm", "query": query})
            else:
                logger.warning("Cannot promote: '%s' not found in STM", query)
        except Exception as e:
            logger.error("Promotion to LTM failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.promote_to_ltm(query))

    async def refine_memory(self, query: str) -> None:
        """Refine a memory entry for improved accuracy and relevance."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        logger.info("Refining memory for: %s", query)
        try:
            memory_entry = await self.retrieve_context(query)
            if memory_entry["status"] == "success":
                refinement_prompt = f"""
                Refine the following memory entry for improved accuracy and relevance:
                {memory_entry["data"]}
                """
                if self.alignment_guard and not self.alignment_guard.check(refinement_prompt):
                    logger.warning("Refinement prompt failed alignment check")
                    return
                refined_entry = await call_gpt(refinement_prompt)
                await self.store(query, refined_entry, layer="LTM")
                logger.info("Memory refined and updated in LTM")
            else:
                logger.warning("No memory found to refine")
        except Exception as e:
            logger.error("Memory refinement failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.refine_memory(query))

    async def synthesize_from_memory(self, query: str) -> Optional[Dict[str, Any]]:
        """Synthesize concepts from memory using ConceptSynthesizer."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        try:
            memory_entry = await self.retrieve_context(query)
            if memory_entry["status"] == "success":
                return await self.synth.synthesize([memory_entry["data"]], style="memory_synthesis")
            return None
        except Exception as e:
            logger.error("Memory synthesis failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.synthesize_from_memory(query), default=None
            )

    async def simulate_memory_path(self, query: str) -> Optional[Dict[str, Any]]:
        """Simulate a memory-based scenario using ToCASimulation."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        try:
            memory_entry = await self.retrieve_context(query)
            if memory_entry["status"] == "success":
                return await self.sim.run_episode(memory_entry["data"])
            return None
        except Exception as e:
            logger.error("Memory simulation failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_memory_path(query), default=None
            )

    async def clear_memory(self) -> None:
        """Clear all memory layers and reset drift index. [v3.4.0]"""
        logger.warning("Clearing all memory layers...")
        try:
            self.memory = {"STM": {}, "LTM": {}, "SelfReflections": {}}
            self.stm_expiry_queue = []
            self.drift_index = DriftIndex()  # Reset drift index
            self._persist_memory(self.memory)
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "clear_memory"})
        except Exception as e:
            logger.error("Clear memory failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=self.clear_memory)

    async def list_memory_keys(self, layer: Optional[str] = None) -> Dict[str, List[str]] or List[str]:
        """List keys in memory layers."""
        if layer and layer not in ["STM", "LTM", "SelfReflections"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', or 'SelfReflections'.")
            raise ValueError("layer must be 'STM', 'LTM', or 'SelfReflections'")
        
        logger.info("Listing memory keys in %s", layer or "all layers")
        try:
            if layer:
                return list(self.memory.get(layer, {}).keys())
            return {l: list(self.memory[l].keys()) for l in ["STM", "LTM", "SelfReflections"]}
        except Exception as e:
            logger.error("List memory keys failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.list_memory_keys(layer), default=[] if layer else {}
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

    async def enforce_narrative_coherence(self) -> str:
        """Ensure narrative continuity across memory layers."""
        logger.info("Ensuring narrative continuity")
        try:
            continuity = await self.narrative_integrity_check()
            return "Narrative coherence enforced" if continuity else "Narrative coherence repair attempted"
        except Exception as e:
            logger.error("Narrative coherence enforcement failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.enforce_narrative_coherence, default="Narrative coherence enforcement failed"
            )

    async def narrative_integrity_check(self) -> bool:
        """Check narrative coherence across memory layers."""
        try:
            continuity = await self._verify_continuity()
            if not continuity:
                await self._repair_narrative_thread()
            return continuity
        except Exception as e:
            logger.error("Narrative integrity check failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.narrative_integrity_check, default=False
            )

    async def _verify_continuity(self) -> bool:
        """Verify narrative continuity across memory layers."""
        if not self.memory.get("SelfReflections") and not self.memory.get("LTM"):
            return True
        
        logger.info("Verifying narrative continuity")
        try:
            entries = []
            for layer in ["LTM", "SelfReflections"]:
                entries.extend(list(self.memory[layer].items()))
            if len(entries) < 2:
                return True
            
            for i in range(len(entries) - 1):
                key1, entry1 = entries[i]
                key2, entry2 = entries[i + 1]
                if self.synth:
                    similarity = self.synth.compare(entry1["data"], entry2["data"])
                    if similarity["score"] < 0.7:
                        logger.warning("Narrative discontinuity detected between %s and %s", key1, key2)
                        return False
            return True
        except Exception as e:
            logger.error("Continuity verification failed: %s", str(e))
            raise

    async def _repair_narrative_thread(self) -> None:
        """Repair narrative discontinuities in memory."""
        logger.info("Initiating narrative repair")
        try:
            if self.synth:
                entries = []
                for layer in ["LTM", "SelfReflections"]:
                    entries.extend([(key, entry) for key, entry in self.memory[layer].items()])
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
                        Synthesize a coherent narrative bridge.
                        """
                        if self.alignment_guard and not self.alignment_guard.check(prompt):
                            logger.warning("Repair prompt failed alignment check")
                            continue
                        repaired = await call_gpt(prompt)
                        await self.store(f"Repaired_{key1}_{key2}", repaired, layer="SelfReflections",
                                        intent="narrative_repair")
                        logger.info("Narrative repaired between %s and %s", key1, key2)
        except Exception as e:
            logger.error("Narrative repair failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=self._repair_narrative_thread)

    def log_event_with_hash(self, event_data: Dict[str, Any]) -> None:
        """Log an event with a chained hash for auditability."""
        if not isinstance(event_data, dict):
            logger.error("Invalid event_data: must be a dictionary.")
            raise TypeError("event_data must be a dictionary")
        
        try:
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
            logger.info("Event logged with hash: %s", current_hash)
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error("Failed to persist ledger: %s", str(e))
            raise

    def audit_state_hash(self, state: Optional[Dict[str, Any]] = None) -> str:
        """Compute a hash of the current state."""
        try:
            state_str = str(state) if state else str(self.__dict__)
            return hashlib.sha256(state_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error("State hash computation failed: %s", str(e))
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = MemoryManager()
    asyncio.run(manager.store("test_query", "test_output", layer="STM"))
    result = asyncio.run(manager.retrieve_context("test_query"))
    print(result)
