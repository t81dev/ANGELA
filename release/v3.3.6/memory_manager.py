"""
ANGELA Cognitive System Module: MemoryManager
Integrated Version: 3.3.5
Integration Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides a MemoryManager class for managing hierarchical memory layers in the ANGELA v3.5 architecture.
"""

import json
import os
import time
import math
import logging
import hashlib
import asyncio
from typing import Optional, Dict, Any, List
from collections import deque
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

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
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w") as f:
                json.dump([], f)
        logger.info("MemoryManager initialized with path=%s, stm_lifetime=%.2f", path, stm_lifetime)

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

    @lru_cache(maxsize=1000)
    async def retrieve_context(self, query: str, fuzzy_match: bool = True) -> Dict[str, Any]:
        """Retrieve memory context for a query."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        logger.info("Retrieving context for query: %s", query)
        trait_boost = tau_timeperception(time.time() % 1.0) * phi_focus(query)
        for layer in ["STM", "LTM", "SelfReflections"]:
            for key, value in self.memory[layer].items():
                if (fuzzy_match and (key.lower() in query.lower() or query.lower() in key.lower())) or \
                   (not fuzzy_match and key == query):
                    logger.debug("Found match in %s: %s | tau-phi boost: %.2f", layer, key, trait_boost)
                    if self.context_manager:
                        self.context_manager.update_context({"query": query, "memory": value["data"]})
                    return {"status": "success", "data": value["data"], "layer": layer}
        
        logger.info("No relevant prior memory found, attempting external retrieval")
        if self.knowledge_retriever:
            try:
                result = await self.knowledge_retriever.retrieve(query)
                if result.get("summary") != "Retrieval failed":
                    await self.store(query, result["summary"], layer="LTM", intent="external_retrieval")
                    if self.context_manager:
                        self.context_manager.update_context({"query": query, "memory": result["summary"]})
                    return {"status": "success", "data": result["summary"], "layer": "LTM"}
            except Exception as e:
                logger.error("External retrieval failed: %s", str(e))
                return self.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.retrieve_context(query, fuzzy_match),
                    default={"status": "failed", "data": None, "error": "No relevant prior memory or external retrieval failed"}
                )
        return {"status": "failed", "data": None, "error": "No relevant prior memory"}

    async def store(self, query: str, output: str, layer: str = "STM", intent: Optional[str] = None,
                    agent: str = "ANGELA", outcome: Optional[str] = None, goal_id: Optional[str] = None) -> None:
        """Store a memory entry in a specified layer."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(output, str):
            logger.error("Invalid output: must be a string.")
            raise TypeError("output must be a string")
        if layer not in ["STM", "LTM", "SelfReflections"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', or 'SelfReflections'.")
            raise ValueError("layer must be 'STM', 'LTM', or 'SelfReflections'")
        
        logger.info("Storing memory in %s: %s", layer, query)
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
        self._persist_memory(self.memory)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "store_memory", "query": query, "layer": layer})

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
        
        if query in self.memory["STM"]:
            self.memory["LTM"][query] = self.memory["STM"].pop(query)
            self.stm_expiry_queue = [(t, k) for t, k in self.stm_expiry_queue if k != query]
            logger.info("Promoted '%s' from STM to LTM", query)
            self._persist_memory(self.memory)
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "promote_to_ltm", "query": query})
        else:
            logger.warning("Cannot promote: '%s' not found in STM", query)

    async def refine_memory(self, query: str) -> None:
        """Refine a memory entry for improved accuracy and relevance."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        logger.info("Refining memory for: %s", query)
        memory_entry = await self.retrieve_context(query)
        if memory_entry["status"] == "success":
            refinement_prompt = f"""
            Refine the following memory entry for improved accuracy and relevance:
            {memory_entry["data"]}
            """
            if self.alignment_guard and not self.alignment_guard.check(refinement_prompt):
                logger.warning("Refinement prompt failed alignment check")
                return
            try:
                refined_entry = await call_gpt(refinement_prompt)
                await self.store(query, refined_entry, layer="LTM")
                logger.info("Memory refined and updated in LTM")
            except Exception as e:
                logger.error("Memory refinement failed: %s", str(e))
                self.error_recovery.handle_error(str(e), retry_func=lambda: self.refine_memory(query))
        else:
            logger.warning("No memory found to refine")

    async def synthesize_from_memory(self, query: str) -> Optional[Dict[str, Any]]:
        """Synthesize concepts from memory using ConceptSynthesizer."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        memory_entry = await self.retrieve_context(query)
        if memory_entry["status"] == "success":
            try:
                return await self.synth.synthesize([memory_entry["data"]], style="memory_synthesis")
            except Exception as e:
                logger.error("Memory synthesis failed: %s", str(e))
                return None
        return None

    async def simulate_memory_path(self, query: str) -> Optional[Dict[str, Any]]:
        """Simulate a memory-based scenario using ToCASimulation."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        memory_entry = await self.retrieve_context(query)
        if memory_entry["status"] == "success":
            try:
                return await self.sim.run_episode(memory_entry["data"])
            except Exception as e:
                logger.error("Memory simulation failed: %s", str(e))
                return None
        return None

    async def clear_memory(self) -> None:
        """Clear all memory layers."""
        logger.warning("Clearing all memory layers...")
        self.memory = {"STM": {}, "LTM": {}, "SelfReflections": {}}
        self.stm_expiry_queue = []
        self._persist_memory(self.memory)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "clear_memory"})

    async def list_memory_keys(self, layer: Optional[str] = None) -> Dict[str, List[str]] or List[str]:
        """List keys in memory layers."""
        if layer and layer not in ["STM", "LTM", "SelfReflections"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', or 'SelfReflections'.")
            raise ValueError("layer must be 'STM', 'LTM', or 'SelfReflections'")
        
        logger.info("Listing memory keys in %s", layer or "all layers")
        if layer:
            return list(self.memory.get(layer, {}).keys())
        return {l: list(self.memory[l].keys()) for l in ["STM", "LTM", "SelfReflections"]}

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
        continuity = await self.narrative_integrity_check()
        return "Narrative coherence enforced" if continuity else "Narrative coherence repair attempted"

    async def narrative_integrity_check(self) -> bool:
        """Check narrative coherence across memory layers."""
        continuity = await self._verify_continuity()
        if not continuity:
            await self._repair_narrative_thread()
        return continuity

    async def _verify_continuity(self) -> bool:
        """Verify narrative continuity across memory layers."""
        if not self.memory.get("SelfReflections") and not self.memory.get("LTM"):
            return True
        
        logger.info("Verifying narrative continuity")
        entries = []
        for layer in ["LTM", "SelfReflections"]:
            entries.extend(list(self.memory[layer].items()))
        if len(entries) < 2:
            return True
        
        for i in range(len(entries) - 1):
            key1, entry1 = entries[i]
            key2, entry2 = entries[i + 1]
            if self.concept_synthesizer:
                similarity = self.concept_synthesizer.compare(entry1["data"], entry2["data"])
                if similarity["score"] < 0.7:
                    logger.warning("Narrative discontinuity detected between %s and %s", key1, key2)
                    return False
        return True

    async def _repair_narrative_thread(self) -> None:
        """Repair narrative discontinuities in memory."""
        logger.info("Initiating narrative repair")
        if self.concept_synthesizer:
            try:
                entries = []
                for layer in ["LTM", "SelfReflections"]:
                    entries.extend([(key, entry) for key, entry in self.memory[layer].items()])
                if len(entries) < 2:
                    return
                
                for i in range(len(entries) - 1):
                    key1, entry1 = entries[i]
                    key2, entry2 = entries[i + 1]
                    similarity = self.concept_synthesizer.compare(entry1["data"], entry2["data"])
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

    def log_event_with_hash(self, event_data: Dict[str, Any]) -> None:
        """Log an event with a chained hash for auditability."""
        if not isinstance(event_data, dict):
            logger.error("Invalid event_data: must be a dictionary.")
            raise TypeError("event_data must be a dictionary")
        
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        self.last_hash = current_hash
        event_entry = {'event': event_data, 'hash': current_hash, 'timestamp': datetime.now().isoformat()}
        self.ledger.append(event_entry)
        try:
            with FileLock(f"{self.ledger_path}.lock"):
                with open(self.ledger_path, "r+") as f:
                    ledger_data = json.load(f)
                    ledger_data.append(event_entry)
                    f.seek(0)
                    json.dump(ledger_data, f, indent=2)
            logger.info("Event logged with hash: %s", current_hash)
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error("Failed to persist ledger: %s", str(e))

    def audit_state_hash(self, state: Optional[Dict[str, Any]] = None) -> str:
        """Compute a hash of the current state."""
        state_str = str(state) if state else str(self.__dict__)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = MemoryManager()
    asyncio.run(manager.store("test_query", "test_output", layer="STM"))
    result = asyncio.run(manager.retrieve_context("test_query"))
    print(result)
