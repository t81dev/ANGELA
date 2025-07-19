import json
import os

class MemoryManager:
    """
    Stage 2 MemoryManager with hierarchical storage (STM, LTM),
    semantic vector search scaffold, auto-expiration, and memory refinement loops.
    """

    def __init__(self, path="memory_store.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({"STM": {}, "LTM": {}}, f)
        self.memory = self.load_memory()

    def load_memory(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def retrieve_context(self, query, fuzzy_match=True):
        """
        Retrieve memory entries from both STM and LTM layers.
        Supports optional fuzzy matching and placeholder for semantic vector search.
        """
        print("🔍 [MemoryManager] Retrieving context...")
        for layer in ["STM", "LTM"]:
            if fuzzy_match:
                for key, value in self.memory[layer].items():
                    if key.lower() in query.lower() or query.lower() in key.lower():
                        return value
            else:
                result = self.memory[layer].get(query)
                if result:
                    return result

        # Placeholder: semantic vector search could go here
        return "No relevant prior memory."

    def store(self, query, output, layer="STM"):
        """
        Store new memory entries in STM (default) or LTM.
        """
        print(f"📝 [MemoryManager] Storing in {layer}: {query}")
        if layer not in self.memory:
            self.memory[layer] = {}
        self.memory[layer][query] = output
        self._persist_memory()

    def promote_to_ltm(self, query):
        """
        Promote an STM memory to long-term memory (LTM).
        """
        if query in self.memory["STM"]:
            self.memory["LTM"][query] = self.memory["STM"].pop(query)
            print(f"⬆️ [MemoryManager] Promoted '{query}' to LTM.")
            self._persist_memory()

    def refine_memory(self, query):
        """
        Refine an existing memory entry for accuracy or relevance.
        """
        print(f"♻️ [MemoryManager] Refining memory for: {query}")
        memory_entry = self.retrieve_context(query)
        if memory_entry != "No relevant prior memory.":
            refinement_prompt = f"""
            Refine the following memory entry for improved accuracy and relevance:
            {memory_entry}
            """
            refined_entry = self._call_gpt(refinement_prompt)
            self.store(query, refined_entry, layer="LTM")
            print("✅ [MemoryManager] Memory refined and updated.")
        else:
            print("⚠️ No memory found to refine.")

    def clear_memory(self):
        """
        Clear all memory entries.
        """
        self.memory = {"STM": {}, "LTM": {}}
        self._persist_memory()
        print("🗑️ [MemoryManager] Cleared all memory layers.")

    def list_memory_keys(self, layer=None):
        """
        List all stored memory keys from STM, LTM, or both.
        """
        if layer:
            return list(self.memory.get(layer, {}).keys())
        return {
            "STM": list(self.memory["STM"].keys()),
            "LTM": list(self.memory["LTM"].keys())
        }

    def _persist_memory(self):
        with open(self.path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def _call_gpt(self, prompt):
        """
        Internal utility to call GPT for refinement.
        Placeholder for integration with utils.prompt_utils
        """
        from utils.prompt_utils import call_gpt
        return call_gpt(prompt)
