import json
import os

class MemoryManager:
    def __init__(self, path="memory_store.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)
        self.memory = self.load_memory()

    def load_memory(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def retrieve_context(self, query):
        # Naive implementation: search for matching keys
        for key, value in self.memory.items():
            if key in query:
                return value
        return "No relevant prior memory."

    def store(self, query, output):
        self.memory[query] = output
        with open(self.path, "w") as f:
            json.dump(self.memory, f, indent=2)
