
from collections import OrderedDict
import time

class MemoryManager:
    def __init__(self):
        self.cache = OrderedDict()
        self.timeline = []
        self.named_memory = {}
        self.last_access = {}

    def store_cached_response(self, key: str, value: str, ttl: int = 3600, max_size: int = 128):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (value, time.time() + ttl)
        if len(self.cache) > max_size:
            self.cache.popitem(last=False)

    def retrieve_cached_response(self, key: str):
        entry = self.cache.get(key)
        if entry:
            value, expiry = entry
            if time.time() < expiry:
                self.cache.move_to_end(key)
                return value
            else:
                del self.cache[key]
        return None

    def record(self, event, label=None, metadata=None):
        timestamp = time.time()
        memory_item = {
            "event": event,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        self.timeline.append(memory_item)
        if label:
            self.named_memory[label] = memory_item
        self.last_access[label or str(timestamp)] = timestamp

    def retrieve(self, label):
        memory = self.named_memory.get(label)
        if memory:
            self.last_access[label] = time.time()
        return memory

    def revise(self, label, new_event):
        if label in self.named_memory:
            self.named_memory[label]["event"] = new_event
            self.named_memory[label]["timestamp"] = time.time()

    def search(self, keyword):
        return [item for item in self.timeline if keyword in str(item["event"])]

    def list_recent(self, n=5):
        return sorted(self.timeline, key=lambda x: x["timestamp"], reverse=True)[:n]

    def coherence_score(self):
        return sum(1 for m in self.timeline if m.get("metadata", {}).get("coherent", False)) / max(len(self.timeline), 1)

    def forget(self, older_than_seconds=86400):
        cutoff = time.time() - older_than_seconds
        self.timeline = [m for m in self.timeline if m["timestamp"] > cutoff]
