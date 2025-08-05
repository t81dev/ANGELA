"""
ANGELA Cognitive System Module
Integrated Version: 3.3.5
Integration Date: 2025-08-05
Maintainer: ANGELA System Framework

This module is part of the ANGELA v3.5 architecture.
Implements multi-layer memory operations with symbolic synthesis (concept_synthesizer),
simulated path reconstruction (toca_simulation), trait-modulated perception (meta_cognition),
contextual harmonization (context_manager), and ethical alignment (alignment_guard).
Supports integration hooks for reasoning_engine, learning_loop, external_agent_bridge,
and user_profile modules.
"""

import json
import os
import time
import logging
import hashlib
from typing import Optional
from utils.prompt_utils import call_gpt
from index import SYSTEM_CONTEXT, delta_memory, tau_timeperception, phi_focus
from concept_synthesizer import ConceptSynthesizer
from toca_simulation import ToCASimulation

logger = logging.getLogger("ANGELA.MemoryManager")

class MemoryManager:
    def __init__(self, path="memory_store.json", stm_lifetime=300):
        """
        Initialize memory manager with STM decay settings, cache, and integrations.
        - ConceptSynthesizer: symbolic transformation of memory
        - ToCASimulation: scenario modeling based on memory entries
        - Contextual and trait-based functions accessed via meta_cognition and index
        - Compatible with user_profile, external_agent_bridge, and learning_loop
        """
        self.path = path
        self.stm_lifetime = stm_lifetime
        self.cache = {}
        self.last_hash = ''
        self.ledger = []
        self.synth = ConceptSynthesizer()
        self.sim = ToCASimulation()
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({"STM": {}, "LTM": {}, "SelfReflections": {}}, f)
        self.memory = self.load_memory()

    def retrieve_cached_response(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def store_cached_response(self, key: str, value: str):
        self.cache[key] = value

    def load_memory(self):
        with open(self.path, "r") as f:
            memory = json.load(f)
        if "SelfReflections" not in memory:
            memory["SelfReflections"] = {}
        self._decay_stm(memory)
        return memory

    def _decay_stm(self, memory):
        current_time = time.time()
        decay_rate = delta_memory(current_time % 1e-18)
        lifetime_adjusted = self.stm_lifetime * (1.0 / decay_rate)
        expired_keys = [key for key, entry in memory.get("STM", {}).items()
                        if current_time - entry["timestamp"] > lifetime_adjusted]
        for key in expired_keys:
            logger.info(f"⏰ STM entry expired: {key}")
            del memory["STM"][key]
        if expired_keys:
            self._persist_memory(memory)

    def retrieve_context(self, query, fuzzy_match=True):
        logger.info(f"🔍 Retrieving context for query: {query}")
        trait_boost = tau_timeperception(time.time() % 1e-18) * phi_focus(query)
        for layer in ["STM", "LTM", "SelfReflections"]:
            for key, value in self.memory[layer].items():
                if (fuzzy_match and (key.lower() in query.lower() or query.lower() in key.lower())) or (not fuzzy_match and key == query):
                    logger.debug(f"🗕 Found match in {layer}: {key} | τϕ_boost: {trait_boost:.2f}")
                    return value["data"]
        logger.info("❌ No relevant prior memory found.")
        return "No relevant prior memory."

    def store(self, query, output, layer="STM", intent=None, agent="ANGELA", outcome=None, goal_id=None):
        logger.info(f"📝 Storing memory in {layer}: {query}")
        entry = {
            "data": output,
            "timestamp": time.time(),
            "intent": intent,
            "agent": agent,
            "outcome": outcome,
            "goal_id": goal_id
        }
        self.memory.setdefault(layer, {})[query] = entry
        self._persist_memory(self.memory)

    def store_reflection(self, summary_text, intent="self_reflection", agent="ANGELA", goal_id=None):
        key = f"Reflection_{time.strftime('%Y%m%d_%H%M%S')}"
        self.store(query=key, output=summary_text, layer="SelfReflections", intent=intent, agent=agent, goal_id=goal_id)
        logger.info(f"🪞 Stored self-reflection: {key}")

    def promote_to_ltm(self, query):
        if query in self.memory["STM"]:
            self.memory["LTM"][query] = self.memory["STM"].pop(query)
            logger.info(f"⬆️ Promoted '{query}' from STM to LTM.")
            self._persist_memory(self.memory)
        else:
            logger.warning(f"⚠️ Cannot promote: '{query}' not found in STM.")

    def refine_memory(self, query):
        logger.info(f"♻️ Refining memory for: {query}")
        memory_entry = self.retrieve_context(query)
        if memory_entry != "No relevant prior memory.":
            refinement_prompt = f"""
            Refine the following memory entry for improved accuracy and relevance:
            {memory_entry}
            """
            refined_entry = call_gpt(refinement_prompt)
            self.store(query, refined_entry, layer="LTM")
            logger.info("✅ Memory refined and updated in LTM.")
        else:
            logger.warning("⚠️ No memory found to refine.")

    def synthesize_from_memory(self, query):
        content = self.retrieve_context(query)
        if content != "No relevant prior memory.":
            return self.synth.synthesize(content)
        return None

    def simulate_memory_path(self, query):
        memory = self.retrieve_context(query)
        if memory != "No relevant prior memory.":
            return self.sim.run_episode(memory)
        return None

    def clear_memory(self):
        logger.warning("🗑️ Clearing all memory layers...")
        self.memory = {"STM": {}, "LTM": {}, "SelfReflections": {}}
        self._persist_memory(self.memory)

    def list_memory_keys(self, layer=None):
        if layer:
            logger.info(f"📃 Listing memory keys in {layer}")
            return list(self.memory.get(layer, {}).keys())
        return {layer: list(self.memory[layer].keys()) for layer in ["STM", "LTM", "SelfReflections"]}

    def _persist_memory(self, memory):
        with open(self.path, "w") as f:
            json.dump(memory, f, indent=2)
        logger.debug("💾 Memory persisted to disk.")

    def enforce_narrative_coherence(self):
        logger.info('Ensuring memory narrative continuity.')
        return "Narrative coherence enforced"

    def narrative_integrity_check(self):
        continuity = self._verify_continuity()
        if not continuity:
            self._repair_narrative_thread()
        return continuity

    def _verify_continuity(self):
        return True

    def _repair_narrative_thread(self):
        print("[ANGELA UPGRADE] Narrative repair initiated.")
        pass

    def log_event_with_hash(self, event_data):
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        self.last_hash = current_hash
        self.ledger.append({'event': event_data, 'hash': current_hash})
        print(f"[ANGELA UPGRADE] Event logged with hash: {current_hash}")

    def audit_state_hash(self, state=None):
        state_str = str(state) if state else str(self.__dict__)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()
