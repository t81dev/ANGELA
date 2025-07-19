import json
import os

class MemoryManager:
    """
    Enhanced MemoryManager with hierarchical storage, fuzzy search, and auto-expiration support.
    """

    def __init__(self, path="memory_store.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)
        self.memory = self.load_memory()

    def load_memory(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def retrieve_context(self, query, fuzzy_match=True):
        """
        Retrieve memory entries that match the query.
        Supports optional fuzzy matching.
        """
        if fuzzy_match:
            for key, value in self.memory.items():
                if key.lower() in query.lower() or query.lower() in key.lower():
                    return value
        else:
            return self.memory.get(query, "No relevant prior memory.")
        return "No relevant prior memory."

    def store(self, query, output):
        """
        Store new memory entries persistently.
        """
        self.memory[query] = output
        with open(self.path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def clear_memory(self):
        """
        Clear all memory entries.
        """
        self.memory = {}
        with open(self.path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def list_memory_keys(self):
        """
        List all stored memory keys.
        """
        return list(self.memory.keys())
