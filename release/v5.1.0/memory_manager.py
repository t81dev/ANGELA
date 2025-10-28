from __future__ import annotations
import hashlib
import json
import time
import logging
import math
import asyncio
import os
from datetime import datetime, timedelta, UTC
from collections import deque, Counter
from typing import List, Dict, Any, Callable, Optional, Set, FrozenSet, Tuple, Union
from functools import lru_cache
from filelock import FileLock
from heapq import heappush, heappop

# ANGELA modules
import context_manager as context_manager_module
import alignment_guard as alignment_guard_module
import error_recovery as error_recovery_module
import concept_synthesizer as concept_synthesizer_module
import knowledge_retriever as knowledge_retriever_module
import meta_cognition as meta_cognition_module
import visualizer as visualizer_module
from toca_simulation import ToCASimulation

logger = logging.getLogger("ANGELA.MemoryManager")

class MemoryManager:
    """
    Hierarchical memory system for ANGELA, managing STM, LTM, and AURA contexts.
    """
    def __init__(self, path: str = "memory_store.json", **services):
        self.path = path
        self.memory = self._load_memory()
        self.stm_expiry_queue: List[Tuple[float, str]] = []
        self.services = services
        self.aura_path = os.getenv("AURA_CONTEXT_PATH", "aura_context.json")
        self.aura_context = self._load_aura_store()
        logger.info("MemoryManager v5.1.0 initialized")

    # --- Core Memory Operations ---
    def _load_memory(self) -> Dict[str, Dict]:
        try:
            with FileLock(f"{self.path}.lock"):
                if os.path.exists(self.path):
                    with open(self.path, "r") as f:
                        return json.load(f)
            return {"STM": {}, "LTM": {}, "SelfReflections": {}}
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return {"STM": {}, "LTM": {}, "SelfReflections": {}}

    def _persist_memory(self) -> None:
        try:
            with FileLock(f"{self.path}.lock"):
                with open(self.path, "w") as f:
                    json.dump(self.memory, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist memory: {e}")

    async def store(self, query: str, output: Any, layer: str = "STM", **kwargs) -> None:
        """Stores an item in the specified memory layer."""
        entry = {"data": output, "timestamp": time.time(), **kwargs}
        self.memory.setdefault(layer, {})[query] = entry

        if layer == "STM":
            expiry_time = time.time() + 300.0  # 5-minute default
            heappush(self.stm_expiry_queue, (expiry_time, query))

        self._persist_memory()
        log_event_to_ledger("ledger_meta", {"event": "memory.store", "query": query, "layer": layer})

    async def retrieve(self, query: str, layer: Optional[str] = None) -> Optional[Dict]:
        """Retrieves an item from memory."""
        layers = [layer] if layer else ["STM", "LTM", "SelfReflections"]
        for l in layers:
            if query in self.memory.get(l, {}):
                return self.memory[l][query]
        return None

    # --- AURA Context Management ---
    def _load_aura_store(self) -> Dict:
        try:
            if os.path.exists(self.aura_path):
                with FileLock(f"{self.aura_path}.lock"):
                    with open(self.aura_path, "r") as f:
                        return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load AURA store: {e}")
        return {}

    def _persist_aura_store(self) -> None:
        try:
            with FileLock(f"{self.aura_path}.lock"):
                with open(self.aura_path, "w") as f:
                    json.dump(self.aura_context, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist AURA store: {e}")

    def save_aura_context(self, user_id: str, summary: str, affective_state: Dict) -> None:
        """Saves an AURA context for a user."""
        self.aura_context[user_id] = {
            "summary": summary,
            "affect": affective_state,
            "updated_at": time.time(),
        }
        self._persist_aura_store()
        log_event_to_ledger("ledger_meta", {"event": "aura.save", "user_id": user_id})

    def load_aura_context(self, user_id: str) -> Optional[Dict]:
        """Loads an AURA context for a user."""
        return self.aura_context.get(user_id)

    # --- Ledger (simplified) ---
    def log_to_ledger(self, event_data: Dict) -> None:
        """A simplified logging mechanism for critical events."""
        log_event_to_ledger("memory_events", event_data)

# Mock ledger functions for standalone operation
def log_event_to_ledger(ledger_name: str, event_data: Dict) -> None:
    logger.info(f"Logging to {ledger_name}: {event_data}")
