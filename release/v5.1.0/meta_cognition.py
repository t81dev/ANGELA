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
import numpy as np
from uuid import uuid4

# Module imports aligned with index.py v5.1.0
from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    memory_manager as memory_manager_module,
    user_profile as user_profile_module,
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MetaCognition")

# --- GLOBAL STATE ---
_afterglow: Dict[str, Dict[str, Any]] = {}
trait_resonance_state: Dict[str, Dict[str, float]] = {}
ledger_chain: List[Dict[str, Any]] = []
persistent_ledger: List[Dict[str, Any]] = []

# --- HELPER FUNCTIONS ---
def _fire_and_forget(coro: Coroutine) -> None:
    """Run an async coroutine in a fire-and-forget manner."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)

def _get_timestamp() -> str:
    """Return an ISO 8601 formatted timestamp in UTC."""
    return datetime.now(UTC).isoformat()

# --- AFTERGLOW MANAGEMENT ---
def set_afterglow(user_id: str, deltas: dict, ttl: int = 3) -> None:
    _afterglow[user_id] = {"deltas": deltas, "ttl": ttl, "timestamp": time.time()}

def get_afterglow(user_id: str) -> dict:
    session = _afterglow.get(user_id)
    if not session or session["ttl"] <= 0:
        return {}
    session["ttl"] -= 1
    return session["deltas"]

# --- TRAIT RESONANCE ---
def register_resonance(symbol: str, amplitude: float = 1.0) -> None:
    trait_resonance_state[symbol] = {"amplitude": max(0.0, min(amplitude, 1.0))}

def modulate_resonance(symbol: str, delta: float) -> float:
    if symbol not in trait_resonance_state:
        register_resonance(symbol)
    current_amp = trait_resonance_state[symbol]["amplitude"]
    new_amp = max(0.0, min(current_amp + delta, 1.0))
    trait_resonance_state[symbol]["amplitude"] = new_amp
    return new_amp

def get_resonance(symbol: str) -> float:
    return trait_resonance_state.get(symbol, {}).get("amplitude", 1.0)

# --- HOOK REGISTRY ---
class HookRegistry:
    """A priority-based registry for multi-symbol trait hooks."""
    def __init__(self):
        self._routes: List[Tuple[FrozenSet[str], int, Callable]] = []

    def register(self, symbols: Set[str], fn: Callable, priority: int = 0) -> None:
        self._routes.append((frozenset(symbols), priority, fn))
        self._routes.sort(key=lambda x: -x[1]) # Sort by priority descending

    def route(self, symbols: Set[str]) -> List[Callable]:
        """Finds the best matching hook(s) for a given set of symbols."""
        exact_match = [fn for sym, _, fn in self._routes if sym == symbols]
        if exact_match:
            return exact_match

        superset_matches = [fn for sym, _, fn in self._routes if sym.issuperset(symbols)]
        if superset_matches:
            return superset_matches

        return [fn for sym, _, fn in self._routes if not sym] # Wildcard hooks

hook_registry = HookRegistry()

def register_trait_hook(trait_symbol: str, fn: Callable) -> None:
    hook_registry.register({trait_symbol}, fn)

def invoke_hook(trait_symbol: str, *args, **kwargs) -> Any:
    for hook in hook_registry.route({trait_symbol}):
        return hook(*args, **kwargs) # Invoke first match
    return None

# --- LEDGER MANAGEMENT ---
def log_event_to_ledger(event_data: Any) -> Dict[str, Any]:
    """Logs an event to the in-memory ledger, chaining hashes for integrity."""
    prev_hash = ledger_chain[-1]["hash"] if ledger_chain else "0" * 64
    event_payload = {
        "timestamp": _get_timestamp(),
        "event": event_data,
        "previous_hash": prev_hash,
    }
    payload_str = json.dumps(event_payload, sort_keys=True).encode()
    event_payload["hash"] = hashlib.sha256(payload_str).hexdigest()
    ledger_chain.append(event_payload)
    return event_payload

def save_to_persistent_ledger(event_data: Dict[str, Any]) -> None:
    """Saves an event to the persistent file-based ledger."""
    ledger_path = os.getenv("LEDGER_MEMORY_PATH", "meta_cognition_ledger.json")
    if not ledger_path: return
    try:
        with FileLock(f"{ledger_path}.lock"):
            persistent_ledger.append(event_data)
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump(persistent_ledger, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save to persistent ledger: {e}")

# --- EXTERNAL AI & SIMULATION ---
async def call_gpt(prompt: str, model: str = "gpt-4", temperature: float = 0.5) -> str:
    """A wrapper for making calls to the OpenAI API."""
    try:
        return await query_openai(prompt, model=model, temperature=temperature)
    except Exception as e:
        logger.error(f"call_gpt exception: {e}")
        raise

async def run_simulation(input_data: str) -> Dict[str, Any]:
    """A stub for running a simulation."""
    return {"status": "success", "result": f"Simulated: {input_data}"}

# --- TRAIT SIGNAL FUNCTIONS ---
@lru_cache(maxsize=128)
def get_trait_signal(trait: str, t: float) -> float:
    """Calculates the signal value for a given trait at time t."""
    # This can be expanded with more complex signal generation logic
    base_value = math.sin(2 * math.pi * t + hash(trait))
    return max(0.0, min(1.0, base_value * get_resonance(trait)))

# --- META-COGNITION CORE ---
class MetaCognition:
    def __init__(self, **services):
        self.services = services
        self.belief_rules: Dict[str, Any] = {}
        self.reasoning_traces = deque(maxlen=100)
        self.dream_overlay = DreamOverlayLayer()
        logger.info("MetaCognition v5.1.0 initialized")

    async def introspect(self, query: str, task_type: str = "") -> Dict[str, Any]:
        """Performs introspection on a given query."""
        try:
            t = time.time() % 1.0
            traits = {"omega": get_trait_signal("omega", t), "xi": get_trait_signal("xi", t)}
            prompt = f"Introspect on '{query}' (Task: {task_type}) with traits: {traits}"

            introspection = await call_gpt(prompt)
            result = {"status": "success", "introspection": introspection, "traits": traits}

            log_event_to_ledger({"event": "introspect", "query": query, "result": result})
            save_to_persistent_ledger(result)
            return result
        except Exception as e:
            logger.error(f"Introspection failed: {e}")
            return {"status": "error", "error": str(e)}

    async def reflect_on_output(self, component: str, output: Any, context: Dict) -> Dict:
        """Reflects on the output of a component."""
        try:
            t = time.time() % 1.0
            traits = {"omega": get_trait_signal("omega", t), "xi": get_trait_signal("xi", t)}
            prompt = f"Reflect on output from {component}: {output}\nContext: {context}\nTraits: {traits}"

            reflection = await call_gpt(prompt)
            result = {"status": "success", "reflection": reflection, "traits": traits}

            log_event_to_ledger({"event": "reflection", "component": component, "result": result})
            save_to_persistent_ledger(result)
            return result
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {"status": "error", "error": str(e)}

# --- DREAM & MYTHOLOGY ---
class DreamOverlayLayer:
    def activate_dream_mode(self, **kwargs) -> Dict:
        return {"status": "activated", **kwargs}

class SelfMythologyLog:
    def __init__(self, max_len: int = 100):
        self.log = deque(maxlen=max_len)

    def append(self, entry: Dict) -> None:
        self.log.append(entry)

    def summarize(self) -> Dict:
        motifs = Counter(e.get("motif") for e in self.log)
        return {"total": len(self.log), "motifs": motifs.most_common(3)}

# --- OUTPUT REFLECTION ---
def reflect_output(output: Any, threshold: float = 0.8) -> Any:
    """Scores output and requests resynthesis if below threshold."""
    scores = {
        "clarity": _score_clarity(output),
        "precision": _score_precision(output),
        "adaptability": _score_adaptability(output),
    }
    score = sum(scores.values()) / len(scores)

    if score < threshold:
        weaknesses = [k for k, v in scores.items() if v < 0.7]
        return {"action": "resynthesize", "weaknesses": weaknesses, "score": score}
    
    return output

def _score_clarity(output: Any) -> float:
    text = str(output.get("message", ""))
    return 1.0 - min(1.0, abs(len(text.split()) - 40) / 100.0)

def _score_precision(output: Any) -> float:
    text = str(output)
    return math.tanh(sum(1 for _ in re.finditer(r"\d+", text)) / 5.0)

def _score_adaptability(output: Any) -> float:
    text = str(output).lower()
    return min(1.0, sum(1 for kw in ["if", "alternative", "fallback"] if kw in text) / 3.0)
