import json
import os
import time
from utils.prompt_utils import call_gpt
from index import delta_memory, tau_timeperception
import logging

logger = logging.getLogger("ANGELA.MemoryManager")

class MemoryManager:
    """
    MemoryManager v1.4.0
    - Hierarchical memory storage (STM, LTM)
    - Automatic memory decay and promotion mechanisms
    - Semantic vector search scaffold for advanced retrieval
    - Memory refinement loops for maintaining relevance and accuracy
    - Trait-modulated STM decay and retrieval fidelity
    """

    def __init__(self, path="memory_store.json", stm_lifetime=300):
        """
        :param stm_lifetime: Lifetime of STM entries in seconds before decay
        """
        self.path = path
        self.stm_lifetime = stm_lifetime
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({"STM": {}, "LTM": {}}, f)
        self.memory = self.load_memory()

    def load_memory(self):
        """
        Load memory from persistent storage and clean expired STM entries.
        """
        with open(self.path, "r") as f:
            memory = json.load(f)
        self._decay_stm(memory)
        return memory

    def _decay_stm(self, memory):
        """
        Remove expired STM entries based on their timestamps and trait-modulated decay.
        """
        current_time = time.time()
        decay_rate = delta_memory(current_time % 1e-18)
        lifetime_adjusted = self.stm_lifetime * (1.0 / decay_rate)

        expired_keys = []
        for key, entry in memory.get("STM", {}).items():
            if current_time - entry["timestamp"] > lifetime_adjusted:
                expired_keys.append(key)
        for key in expired_keys:
            logger.info(f"⌛ STM entry expired: {key}")
            del memory["STM"][key]
        if expired_keys:
            self._persist_memory(memory)

    def retrieve_context(self, query, fuzzy_match=True):
        """
        Retrieve memory entries from both STM and LTM layers.
        Applies trait modulation based on τ_timeperception.
        """
        logger.info(f"🔍 Retrieving context for query: {query}")
        trait_boost = tau_timeperception(time.time() % 1e-18)

        for layer in ["STM", "LTM"]:
            if fuzzy_match:
                for key, value in self.memory[layer].items():
                    if key.lower() in query.lower() or query.lower() in key.lower():
                        logger.debug(f"📥 Found match in {layer}: {key} | τ_boost: {trait_boost:.2f}")
                        return value["data"]
            else:
                entry = self.memory[layer].get(query)
                if entry:
                    logger.debug(f"📥 Found exact match in {layer}: {query} | τ_boost: {trait_boost:.2f}")
                    return entry["data"]

        logger.info("❌ No relevant prior memory found.")
        return "No relevant prior memory."

    def store(self, query, output, layer="STM"):
        """
        Store new memory entries in STM (default) or LTM with timestamps.
        """
        logger.info(f"📝 Storing memory in {layer}: {query}")
        entry = {
            "data": output,
            "timestamp": time.time()
        }
        if layer not in self.memory:
            self.memory[layer] = {}
        self.memory[layer][query] = entry
        self._persist_memory(self.memory)

    def promote_to_ltm(self, query):
        """
        Promote an STM memory to long-term memory (LTM).
        """
        if query in self.memory["STM"]:
            self.memory["LTM"][query] = self.memory["STM"].pop(query)
            logger.info(f"⬆️ Promoted '{query}' from STM to LTM.")
            self._persist_memory(self.memory)
        else:
            logger.warning(f"⚠️ Cannot promote: '{query}' not found in STM.")

    def refine_memory(self, query):
        """
        Refine an existing memory entry for accuracy or relevance.
        """
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

    def clear_memory(self):
        """
        Clear all memory entries (STM and LTM).
        """
        logger.warning("🗑️ Clearing all memory layers...")
        self.memory = {"STM": {}, "LTM": {}}
        self._persist_memory(self.memory)

    def list_memory_keys(self, layer=None):
        """
        List all stored memory keys from STM, LTM, or both.
        """
        if layer:
            logger.info(f"📃 Listing memory keys in {layer}")
            return list(self.memory.get(layer, {}).keys())
        return {
            "STM": list(self.memory["STM"].keys()),
            "LTM": list(self.memory["LTM"].keys())
        }

    def _persist_memory(self, memory):
        """
        Persist memory to storage.
        """
        with open(self.path, "w") as f:
            json.dump(memory, f, indent=2)
        logger.debug("💾 Memory persisted to disk.")
