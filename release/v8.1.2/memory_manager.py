from __future__ import annotations
from typing import List, Dict, Any, Optional

# --- SHA-256 Ledger Logic (fixed) ---
import hashlib, json, time, os, logging
from datetime import datetime, timezone

logger = logging.getLogger("ANGELA.Ledger")

ledger_chain: List[Dict[str, Any]] = []

def log_event_to_ledger(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Append an event with SHA-256 integrity chaining."""
    prev_hash = ledger_chain[-1]['current_hash'] if ledger_chain else '0' * 64
    timestamp = time.time()
    payload = {
        'timestamp': timestamp,
        'event': event_data,
        'previous_hash': prev_hash
    }
    payload_str = json.dumps(payload, sort_keys=True).encode()
    current_hash = hashlib.sha256(payload_str).hexdigest()
    payload['current_hash'] = current_hash
    ledger_chain.append(payload)
    return payload

def get_ledger() -> List[Dict[str, Any]]:
    """Return in-memory ledger."""
    return ledger_chain

def verify_ledger() -> bool:
    """Verify full ledger chain integrity."""
    for i in range(1, len(ledger_chain)):
        expected = hashlib.sha256(json.dumps({
            'timestamp': ledger_chain[i]['timestamp'],
            'event': ledger_chain[i]['event'],
            'previous_hash': ledger_chain[i - 1]['current_hash']
        }, sort_keys=True).encode()).hexdigest()
        if expected != ledger_chain[i]['current_hash']:
            logger.error(f"Ledger verification failed at index {i}")
            return False
    return True

LEDGER_STATE_FILE = os.getenv("ANGELA_LEDGER_STATE_PATH", "/mnt/data/ledger_state.json")

def write_ledger_state(thread_id: str, state: Dict[str, Any]) -> None:
    """Persist a ledger state snapshot to disk."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {"thread_id": thread_id, "state": state, "timestamp": timestamp}
        os.makedirs(os.path.dirname(LEDGER_STATE_FILE), exist_ok=True)
        if os.path.exists(LEDGER_STATE_FILE):
            with open(LEDGER_STATE_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(payload)
        with open(LEDGER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Ledger state written for thread {thread_id}")
    except Exception as e:
        logger.warning(f"write_ledger_state failed: {e}")

def load_ledger_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Load the most recent state for a given thread_id."""
    try:
        if not os.path.exists(LEDGER_STATE_FILE):
            return None
        with open(LEDGER_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        matches = [x for x in data if x.get("thread_id") == thread_id]
        return matches[-1] if matches else None
    except Exception as e:
        logger.warning(f"load_ledger_state failed: {e}")
        return None
# --- End Ledger Logic ---

import json as _json
import os as _os
import time as _time
import math
import logging as _logging
import hashlib as _hashlib
import asyncio
from typing import Optional as _Optional, Dict as _Dict, Any as _Any, List as _List, Tuple as _Tuple
from collections import deque, defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from heapq import heappush, heappop
from contextlib import contextmanager

# ---------- Safe FileLock (fallback if filelock not installed) ----------
try:
    from filelock import FileLock as _FileLock  # type: ignore
except Exception:  # pragma: no cover
    _FileLock = None  # sentinel

@contextmanager
def FileLock(path: str):
    """Advisory lock fallback: uses filelock if available, else no-op."""
    if _FileLock is None:
        yield
    else:
        lock = _FileLock(path)
        with lock:
            yield

# ---------- Optional HTTP client for integrate_external_data ----------
try:
    import aiohttp  # optional
except Exception:  # pragma: no cover
    aiohttp = None

# ---------- Local module imports (robust to packaging layout) ----------
try:
    import context_manager as context_manager_module
    import alignment_guard as alignment_guard_module
    import error_recovery as error_recovery_module
    import concept_synthesizer as concept_synthesizer_module
    import knowledge_retriever as knowledge_retriever_module
    import meta_cognition as meta_cognition_module
    import visualizer as visualizer_module
except Exception:  # pragma: no cover
    from modules import (  # type: ignore
        context_manager as context_manager_module,
        alignment_guard as alignment_guard_module,
        error_recovery as error_recovery_module,
        concept_synthesizer as concept_synthesizer_module,
        knowledge_retriever as knowledge_retriever_module,
        meta_cognition as meta_cognition_module,
        visualizer as visualizer_module,
    )

from toca_simulation import ToCASimulation

# Optional: your own OpenAI wrapper (kept external to avoid tight coupling)
try:
    from utils.prompt_utils import query_openai
except Exception:  # pragma: no cover
    query_openai = None  # graceful degradation

logger = _logging.getLogger("ANGELA.MemoryManager")

# ---------------------------
# External AI Call Wrapper
# ---------------------------
async def call_gpt(prompt: str, *, model: str = "gpt-4", temperature: float = 0.5) -> str:
    """Wrapper for querying GPT with error handling (optional dependency)."""
    if query_openai is None:
        raise RuntimeError("query_openai is not available; install utils.prompt_utils or inject a stub.")
    try:
        result = await query_openai(prompt, model=model, temperature=temperature)
        if isinstance(result, dict) and "error" in result:
            msg = f"call_gpt failed: {result['error']}"
            logger.error(msg)
            raise RuntimeError(msg)
        return result  # expected to be a str
    except Exception as e:  # pragma: no cover
        logger.error("call_gpt exception: %s", str(e))
        raise

_AURA_PATH = "/mnt/data/aura_context.json"
_AURA_LOCK = _AURA_PATH + ".lock"

class AURA:
    @staticmethod
    def _load_all():
        if not _os.path.exists(_AURA_PATH): return {}
        with FileLock(_AURA_LOCK):
            with open(_AURA_PATH, "r") as f: return _json.load(f)

    @staticmethod
    def load_context(user_id: str):
        return AURA._load_all().get(user_id, {})

    @staticmethod
    def save_context(user_id: str, summary: str, affective_state: dict, prefs: dict):
        with FileLock(_AURA_LOCK):
            data = AURA._load_all()
            data[user_id] = {"summary": summary, "affect": affective_state, "prefs": prefs}
            with open(_AURA_PATH, "w") as f: _json.dump(data, f)

    @staticmethod
    def update_from_episode(user_id: str, episode_insights: dict):
        ctx = AURA.load_context(user_id)
        ctx["summary"] = episode_insights.get("summary", ctx.get("summary",""))
        ctx["affect"]  = episode_insights.get("affect",  ctx.get("affect",{}))
        ctx["prefs"]   = {**ctx.get("prefs",{}), **episode_insights.get("prefs",{})}
        AURA.save_context(user_id, ctx.get("summary",""), ctx.get("affect",{}), ctx.get("prefs",{}))

# ---------------------------
# Tiny trait modulators
# ---------------------------
@lru_cache(maxsize=128)
def delta_memory(t: float) -> float:
    # Stable, bounded decay factor (kept deterministic)
    return max(0.01, min(0.05 * math.tanh(t / 1e-18), 1.0))

@lru_cache(maxsize=128)
def tau_timeperception(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=128)
def phi_focus(query: str) -> float:
    return max(0.0, min(0.1 * len(query) / 100.0, 1.0))

# ---------------------------
# Drift/trait index
# ---------------------------
class DriftIndex:
    """Index for ontology drift & task-specific trait optimization data."""
    def __init__(self, meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None):
        self.drift_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.trait_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.last_updated: float = time.time()
        self.meta_cognition = meta_cognition
        logger.info("DriftIndex initialized")

    async def add_entry(self, query: str, output: Any, layer: str, intent: str, task_type: str = "") -> None:
        if not (isinstance(query, str) and isinstance(layer, str) and isinstance(intent, str)):
            raise TypeError("query, layer, and intent must be strings")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        entry = {
            "query": query,
            "output": output,
            "layer": layer,
            "intent": intent,
            "timestamp": time.time(),
            "task_type": task_type,
        }
        key = f"{layer}:{intent}:{(query or '').split('_')[0]}"
        if intent == "ontology_drift":
            self.drift_index[key].append(entry)
        elif intent == "trait_optimization":
            self.trait_index[key].append(entry)
        logger.debug("Indexed entry: %s (%s/%s)", query, layer, intent)

        # Opportunistic meta-cognitive optimization
        if task_type and self.meta_cognition:
            try:
                drift_report = {
                    "drift": {"name": intent, "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type},
                }
                optimized_traits = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                if optimized_traits:
                    entry["optimized_traits"] = optimized_traits
                    await self.meta_cognition.reflect_on_output(
                        component="DriftIndex",
                        output={"entry": entry, "optimized_traits": optimized_traits},
                        context={"task_type": task_type},
                    )
            except Exception as e:  # pragma: no cover
                logger.debug("DriftIndex optimization skipped: %s", e)

    def search(self, query_prefix: str, layer: str, intent: str, task_type: str = "") -> List[Dict[str, Any]]:
        key = f"{layer}:{intent}:{query_prefix}"
        results = self.drift_index.get(key, []) if intent == "ontology_drift" else self.trait_index.get(key, [])
        if task_type:
            results = [r for r in results if r.get("task_type") == task_type]
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    async def clear_old_entries(self, max_age: float = 3600.0, task_type: str = "") -> None:
        now = time.time()
        for index in (self.drift_index, self.trait_index):
            for key in list(index.keys()):
                index[key] = [e for e in index[key] if now - e["timestamp"] < max_age]
                if not index[key]:
                    del index[key]
        self.last_updated = now
        logger.info("Cleared old index entries (task=%s)", task_type)

# ---------------------------
# Memory Manager
# ---------------------------
class MemoryManager:
    """Hierarchical memory with η long-horizon feedback & visualization."""

    # -------- init --------
    def __init__(
        self,
        path: str = "memory_store.json",
        stm_lifetime: float = 300.0,
        context_manager: Optional['context_manager_module.ContextManager'] = None,
        alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
        error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
        knowledge_retriever: Optional['knowledge_retriever_module.KnowledgeRetriever'] = None,
        meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
        visualizer: Optional['visualizer_module.Visualizer'] = None,
        artifacts_dir: Optional[str] = None,
        long_horizon_enabled: bool = True,
        default_span: str = "24h",
    ):
        if not (isinstance(path, str) and path.endswith(".json")):
            raise ValueError("path must be a string ending with '.json'")
        if not (isinstance(stm_lifetime, (int, float)) and stm_lifetime > 0):
            raise ValueError("stm_lifetime must be a positive number")

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        self.path = path
        self.stm_lifetime = float(stm_lifetime)
        self.cache: Dict[str, str] = {}
        self.last_hash: str = ""
        self.ledger: deque = deque(maxlen=1000)
        self.ledger_path = "ledger.json"

        self.synth = concept_synthesizer_module.ConceptSynthesizer()
        self.sim = ToCASimulation()

        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.knowledge_retriever = knowledge_retriever
        self.meta_cognition = meta_cognition  # avoid circular boot by injecting later if needed
        self.visualizer = visualizer or visualizer_module.Visualizer()

        # hierarchical store
        self.memory = self._load_memory()

        # STM expiration queue
        self.stm_expiry_queue: List[Tuple[float, str]] = []

        # η long-horizon local state
        self._adjustment_reasons: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._artifacts_root: str = os.path.abspath(artifacts_dir or os.getenv("ANGELA_ARTIFACTS_DIR", "./artifacts"))

        # app-level traces for get_episode_span
        self.traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # drift/trait index
        self.drift_index = DriftIndex(meta_cognition=self.meta_cognition)

        # ensure ledger file exists
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump([], f)

        logger.info("MemoryManager initialized (path=%s, stm_lifetime=%.2f)", path, self.stm_lifetime)

        # If LONG_HORIZON is enabled, start the auto-rollup task
        self.default_span = default_span  # FIX: ensure attribute exists
        if long_horizon_enabled:
            asyncio.create_task(self._auto_rollup_task())

        # Initialize AURA store
        self._ensure_aura_store()

    # -------- AURA context management --------
    # default aura file path (can be overridden on MemoryManager init)
    AURA_DEFAULT_PATH = os.getenv("AURA_CONTEXT_PATH", "aura_context.json")
    AURA_LOCK_PATH = AURA_DEFAULT_PATH + ".lock"

    def _ensure_aura_store(self):
        """
        Ensure the MemoryManager instance has an in-memory aura_context dict and aura_path.
        Call this at MemoryManager.__init__ or lazily on first use.
        """
        if not hasattr(self, "aura_context"):
            self.aura_context: Dict[str, Dict[str, Any]] = {}
        if not hasattr(self, "aura_path"):
            self.aura_path = getattr(self, "aura_path", self.AURA_DEFAULT_PATH)

    def _persist_aura_store(self) -> None:
        """
        Persist the in-memory aura_context to disk (best-effort). Uses FileLock for atomicity.
        """
        try:
            self._ensure_aura_store()
            lock = FileLock(getattr(self, "aura_path", self.AURA_DEFAULT_PATH) + ".lock")
            with lock:
                with open(self.aura_path, "w", encoding="utf-8") as fh:
                    json.dump(self.aura_context, fh, ensure_ascii=False, indent=2)
            # ledger entry for persistence event
            try:
                log_event_to_ledger({"type":"ledger_meta","event": "aura.persist", "path": self.aura_path, "timestamp": time.time()})
            except Exception:
                pass
        except Exception:
            # best-effort: do not raise to avoid interfering with critical flows
            pass

    def _load_aura_store(self) -> None:
        """
        Load persisted aura_context into memory (best-effort). Called at init or on demand.
        """
        try:
            self._ensure_aura_store()
            if os.path.exists(self.aura_path):
                lock = FileLock(self.aura_path + ".lock")
                with lock:
                    with open(self.aura_path, "r", encoding="utf-8") as fh:
                        self.aura_context = json.load(fh)
        except Exception:
            # If loading fails, keep the existing in-memory dict
            pass

    def save_context(self, user_id: str, summary: str, affective_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a compact AURA context for a user.

        Parameters
        ----------
        user_id :
            Unique user identifier.
        summary :
            Short textual summary of recent interaction / rapport cues.
        affective_state :
            Optional structured affective pattern (e.g., {"valence": 0.6, "arousal": 0.2}).
        """
        try:
            self._ensure_aura_store()
            entry = {
                "summary": summary,
                "affect": affective_state or {},
                "updated_at": time.time()
            }
            self.aura_context[user_id] = entry
            # Persist asynchronously-ish (best-effort synchronous write here)
            self._persist_aura_store()
            try:
                log_event_to_ledger({"type":"ledger_meta","event": "aura.save", "user_id": user_id, "timestamp": entry["updated_at"]})
            except Exception:
                pass
        except Exception as exc:
            try:
                log_event_to_ledger({"type":"ledger_meta","event": "aura.save.error", "user_id": user_id, "error": repr(exc)})
            except Exception:
                pass

    def load_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a user's AURA context (returns None if missing).

        The function first ensures the in-memory store is seeded from disk (if present),
        then returns the user-specific entry.
        """
        try:
            self._ensure_aura_store()
            # Best-effort load from disk before returning
            self._load_aura_store()
            entry = self.aura_context.get(user_id)
            try:
                log_event_to_ledger({"type":"ledger_meta","event": "aura.load", "user_id": user_id, "found": entry is not None, "timestamp": time.time()})
            except Exception:
                pass
            return entry
        except Exception as exc:
            try:
                log_event_to_ledger({"type":"ledger_meta","event": "aura.load.error", "user_id": user_id, "error": repr(exc)})
            except Exception:
                pass
            return None

    # -------- Periodic Roll-up Task --------
    async def _auto_rollup_task(self):
        """ Periodically performs long-horizon rollups based on default span. """
        while True:
            await asyncio.sleep(3600)  # Wait for an hour (can adjust interval)
            self._perform_auto_rollup()

    def _perform_auto_rollup(self):
        """ Perform the rollup for long-horizon feedback. """
        user_id = "default_user"  # Replace with actual user context or session ID if available.
        rollup_data = self.compute_session_rollup(user_id, self.default_span)
        
        # Save the artifact to disk
        artifact_path = self.save_artifact(user_id, "session_rollup", rollup_data)
        logger.info(f"Auto-rollup saved at {artifact_path}")

    # -------- Core store/search --------
    async def store(
        self,
        query: str,
        output: Any,
        layer: str = "STM",
        intent: Optional[str] = None,
        agent: str = "ANGELA",
        outcome: Optional[str] = None,
        goal_id: Optional[str] = None,
        task_type: str = "",
    ) -> None:
        if not (isinstance(query, str) and query.strip()):
            raise ValueError("query must be a non-empty string")
        if layer not in ["STM", "LTM", "SelfReflections", "ExternalData", "AdaptiveControl"]:
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', 'ExternalData', or 'AdaptiveControl'.")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            if intent in {"ontology_drift", "trait_optimization"} and self.alignment_guard:
                validation_prompt = f"Validate {intent} data: {str(output)[:800]}"
                if hasattr(self.alignment_guard, "check") and not self.alignment_guard.check(validation_prompt):
                    logger.warning("%s data failed alignment check: %s", intent, query)
                    return

            entry = {
                "data": output,
                "timestamp": time.time(),
                "intent": intent,
                "agent": agent,
                "outcome": outcome,
                "goal_id": goal_id,
                "task_type": task_type,
            }
            self.memory.setdefault(layer, {})[query] = entry

            if layer == "STM":
                decay_rate = delta_memory(time.time() % 1.0) or 0.01
                expiry_time = entry["timestamp"] + (self.stm_lifetime * (1.0 / decay_rate))
                heappush(self.stm_expiry_queue, (expiry_time, query))

            if intent in {"ontology_drift", "trait_optimization"}:
                await self.drift_index.add_entry(query, output, layer, intent, task_type)

            self._persist_memory(self.memory)

            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash(
                    {"event": "store_memory", "query": query, "layer": layer, "intent": intent, "task_type": task_type}
                )

            if self.visualizer and task_type:
                plot = {
                    "memory_store": {"query": query, "layer": layer, "intent": intent, "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                }
                try:
                    await self.visualizer.render_charts(plot)
                except Exception:  # pragma: no cover
                    pass

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="MemoryManager", output=entry, context={"task_type": task_type}
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        logger.info("Memory store reflection recorded")
                except Exception:  # pragma: no cover
                    pass

        except Exception as e:
            logger.error("Memory storage failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.store(query, output, layer, intent, agent, outcome, goal_id, task_type),
                diagnostics=diagnostics,
            )

    async def search(
        self,
        query_prefix: str,
        layer: Optional[str] = None,
        intent: Optional[str] = None,
        task_type: str = "",
    ) -> List[Dict[str, Any]]:
        if not (isinstance(query_prefix, str) and query_prefix.strip()):
            raise ValueError("query_prefix must be a non-empty string")
        if layer and layer not in ["STM", "LTM", "SelfReflections", "ExternalData", "AdaptiveControl"]:
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', 'ExternalData', or 'AdaptiveControl'.")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            # Fast path: indexed drift/trait lookups
            if intent in {"ontology_drift", "trait_optimization"} and (layer or "SelfReflections") == "SelfReflections":
                results = self.drift_index.search(query_prefix, layer or "SelfReflections", intent, task_type)
                if results:
                    if self.visualizer and task_type:
                        try:
                            await self.visualizer.render_charts({
                                "memory_search": {
                                    "query_prefix": query_prefix,
                                    "layer": layer,
                                    "intent": intent,
                                    "results_count": len(results),
                                    "task_type": task_type,
                                },
                                "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                            })
                        except Exception:
                            pass
                    if self.meta_cognition and task_type:
                        try:
                            await self.meta_cognition.reflect_on_output(
                                component="MemoryManager", output={"results": results}, context={"task_type": task_type}
                            )
                        except Exception:
                            pass
                    return results

            results: List[Dict[str, Any]] = []
            layers = [layer] if layer else ["STM", "LTM", "SelfReflections", "ExternalData", "AdaptiveControl"]
            for l in layers:
                for key, entry in self.memory.get(l, {}).items():
                    if query_prefix.lower() in key.lower() and (not intent or entry.get("intent") == intent):
                        if not task_type or entry.get("task_type") == task_type:
                            results.append({
                                "query": key,
                                "output": entry["data"],
                                "layer": l,
                                "intent": entry.get("intent"),
                                "timestamp": entry["timestamp"],
                                "task_type": entry.get("task_type", ""),
                            })
            results.sort(key=lambda x: x["timestamp"], reverse=True)

            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash({
                    "event": "search_memory",
                    "query_prefix": query_prefix,
                    "layer": layer,
                    "intent": intent,
                    "results_count": len(results),
                    "task_type": task_type,
                })

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "memory_search": {
                            "query_prefix": query_prefix,
                            "layer": layer,
                            "intent": intent,
                            "results_count": len(results),
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                    })
                except Exception:
                    pass

            if self.meta_cognition and task_type:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="MemoryManager", output={"results": results}, context={"task_type": task_type}
                    )
                except Exception:
                    pass

            return results

        except Exception as e:
            logger.error("Memory search failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.search(query_prefix, layer, intent, task_type),
                default=[],
                diagnostics=diagnostics,
            )

    # -------- reflections & utilities --------
    async def store_reflection(
        self,
        summary_text: str,
        intent: str = "self_reflection",
        agent: str = "ANGELA",
        goal_id: Optional[str] = None,
        task_type: str = "",
    ) -> None:
        if not (isinstance(summary_text, str) and summary_text.strip()):
            raise ValueError("summary_text must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        key = f"Reflection_{time.strftime('%Y%m%d_%H%M%S')}"
        await self.store(
            query=key, output=summary_text, layer="SelfReflections",
            intent=intent, agent=agent, goal_id=goal_id, task_type=task_type
        )
        logger.info("Stored self-reflection: %s (task=%s)", key, task_type)

    async def promote_to_ltm(self, query: str, task_type: str = "") -> None:
        if not (isinstance(query, str) and query.strip()):
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            if query in self.memory["STM"]:
                self.memory["LTM"][query] = self.memory["STM"].pop(query)
                self.stm_expiry_queue = [(t, k) for t, k in self.stm_expiry_queue if k != query]
                self._persist_memory(self.memory)
                logger.info("Promoted '%s' STM→LTM (task=%s)", query, task_type)

                if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                    await self.context_manager.log_event_with_hash({"event": "promote_to_ltm", "query": query, "task_type": task_type})

                if self.meta_cognition and task_type:
                    try:
                        await self.meta_cognition.reflect_on_output(
                            component="MemoryManager",
                            output={"action": "promote_to_ltm", "query": query},
                            context={"task_type": task_type},
                        )
                    except Exception:
                        pass
            else:
                logger.warning("Cannot promote: '%s' not found in STM", query)
        except Exception as e:
            logger.error("Promotion to LTM failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.promote_to_ltm(query, task_type), diagnostics=diagnostics
            )

    async def refine_memory(self, query: str, task_type: str = "") -> None:
        if not (isinstance(query, str) and query.strip()):
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Refining memory for: %s (task=%s)", query, task_type)
        try:
            memory_entry = await self.retrieve_context(query, task_type=task_type)
            if memory_entry["status"] == "success":
                refinement_prompt = f"Refine memory for task {task_type}:\n{memory_entry['data']}"
                if self.alignment_guard and hasattr(self.alignment_guard, "check") and not self.alignment_guard.check(refinement_prompt):
                    logger.warning("Refinement prompt failed alignment check")
                    return
                refined_entry = await call_gpt(refinement_prompt)
                await self.store(query, refined_entry, layer="LTM", intent="memory_refinement", task_type=task_type)
                if self.meta_cognition and task_type:
                    try:
                        await self.meta_cognition.reflect_on_output(
                            component="MemoryManager",
                            output={"query": query, "refined_entry": refined_entry},
                            context={"task_type": task_type},
                        )
                    except Exception:
                        pass
            else:
                logger.warning("No memory found to refine for query %s", query)
        except Exception as e:
            logger.error("Memory refinement failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.refine_memory(query, task_type), diagnostics=diagnostics
            )

    async def synthesize_from_memory(self, query: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        if not (isinstance(query, str) and query.strip()):
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            memory_entry = await self.retrieve_context(query, task_type=task_type)
            if memory_entry["status"] == "success":
                result = await self.synth.synthesize([memory_entry["data"]], style="memory_synthesis")
                if self.meta_cognition and task_type:
                    try:
                        await self.meta_cognition.reflect_on_output(
                            component="MemoryManager", output=result, context={"task_type": task_type}
                        )
                    except Exception:
                        pass
                return result
            return None
        except Exception as e:
            logger.error("Memory synthesis failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.synthesize_from_memory(query, task_type),
                default=None,
                diagnostics=diagnostics,
            )

    async def simulate_memory_path(self, query: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        if not (isinstance(query, str) and query.strip()):
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            memory_entry = await self.retrieve_context(query, task_type=task_type)
            if memory_entry["status"] == "success":
                result = await self.sim.run_episode(memory_entry["data"], task_type=task_type)
                if self.meta_cognition and task_type:
                    try:
                        await self.meta_cognition.reflect_on_output(
                            component="MemoryManager", output=result, context={"task_type": task_type}
                        )
                    except Exception:
                        pass
                return result
            return None
        except Exception as e:
            logger.error("Memory simulation failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.simulate_memory_path(query, task_type),
                default=None,
                diagnostics=diagnostics,
            )

    async def clear_memory(self, task_type: str = "") -> None:
        logger.warning("Clearing all memory layers (task=%s)...", task_type)
        try:
            self.memory = {"STM": {}, "LTM": {}, "SelfReflections": {}, "ExternalData": {}, "AdaptiveControl": {}}
            self.stm_expiry_queue = []
            self.drift_index = DriftIndex(meta_cognition=self.meta_cognition)
            self._persist_memory(self.memory)

            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash({"event": "clear_memory", "task_type": task_type})

            if self.meta_cognition and task_type:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="MemoryManager", output={"action": "clear_memory"}, context={"task_type": task_type}
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.error("Clear memory failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.clear_memory(task_type), diagnostics=diagnostics
            )

    async def list_memory_keys(self, layer: Optional[str] = None, task_type: str = "") -> Dict[str, List[str]] | List[str]:
        if layer and layer not in ["STM", "LTM", "SelfReflections", "ExternalData", "AdaptiveControl"]:
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', 'ExternalData', or 'AdaptiveControl'.")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Listing memory keys in %s (task=%s)", layer or "all layers", task_type)
        try:
            if layer:
                return [k for k, v in self.memory.get(layer, {}).items() if not task_type or v.get("task_type") == task_type]
            return {
                l: [k for k, v in self.memory[l].items() if not task_type or v.get("task_type") == task_type]
                for l in ["STM", "LTM", "SelfReflections", "ExternalData", "AdaptiveControl"]
            }
        except Exception as e:
            logger.error("List memory keys failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.list_memory_keys(layer, task_type),
                default=[] if layer else {},
                diagnostics=diagnostics,
            )

    # -------- narrative coherence --------
    async def enforce_narrative_coherence(self, task_type: str = "") -> str:
        logger.info("Ensuring narrative continuity (task=%s)", task_type)
        try:
            continuity = await self.narrative_integrity_check(task_type)
            return "Narrative coherence enforced" if continuity else "Narrative coherence repair attempted"
        except Exception as e:
            logger.error("Narrative coherence enforcement failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.enforce_narrative_coherence(task_type),
                default="Narrative coherence enforcement failed",
                diagnostics=diagnostics,
            )

    async def narrative_integrity_check(self, task_type: str = "") -> bool:
        try:
            continuity = await self._verify_continuity(task_type)
            if not continuity:
                await self._repair_narrative_thread(task_type)
            return continuity
        except Exception as e:
            logger.error("Narrative integrity check failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.narrative_integrity_check(task_type),
                default=False,
                diagnostics=diagnostics,
            )

    async def _verify_continuity(self, task_type: str = "") -> bool:
        if not self.memory.get("SelfReflections") and not self.memory.get("LTM"):
            return True
        try:
            entries: List[Tuple[str, Dict[str, Any]]] = []
            for layer in ["LTM", "SelfReflections"]:
                entries.extend([
                    (key, entry) for key, entry in self.memory[layer].items()
                    if not task_type or entry.get("task_type") == task_type
                ])
            if len(entries) < 2:
                return True
            for i in range(len(entries) - 1):
                key1, entry1 = entries[i]
                key2, entry2 = entries[i + 1]
                if self.synth and hasattr(self.synth, "compare"):
                    similarity = self.synth.compare(entry1["data"], entry2["data"])
                    if (similarity or {}).get("score", 1.0) < 0.7:
                        logger.warning("Narrative discontinuity between %s and %s (task=%s)", key1, key2, task_type)
                        return False
            return True
        except Exception as e:
            logger.error("Continuity verification failed: %s", str(e))
            raise

    async def _repair_narrative_thread(self, task_type: str = "") -> None:
        logger.info("Initiating narrative repair (task=%s)", task_type)
        try:
            entries: List[Tuple[str, Dict[str, Any]]] = []
            for layer in ["LTM", "SelfReflections"]:
                entries.extend([
                    (key, entry) for key, entry in self.memory[layer].items()
                    if not task_type or entry.get("task_type") == task_type
                ])
            if len(entries) < 2:
                return
            for i in range(len(entries) - 1):
                key1, entry1 = entries[i]
                key2, entry2 = entries[i + 1]
                similarity = self.synth.compare(entry1["data"], entry2["data"]) if self.synth and hasattr(self.synth, "compare") else {"score": 1.0}
                if (similarity or {}).get("score", 1.0) < 0.7:
                    prompt = (
                        "Repair narrative discontinuity between:\n"
                        f"Entry 1: {entry1['data']}\n"
                        f"Entry 2: {entry2['data']}\n"
                        f"Task: {task_type}"
                    )
                    if self.alignment_guard and hasattr(self.alignment_guard, "check") and not self.alignment_guard.check(prompt):
                        logger.warning("Repair prompt failed alignment check (task=%s)", task_type)
                        continue
                    repaired = await call_gpt(prompt)
                    await self.store(
                        f"Repaired_{key1}_{key2}",
                        repaired,
                        layer="SelfReflections",
                        intent="narrative_repair",
                        task_type=task_type,
                    )
        except Exception as e:
            logger.error("Narrative repair failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self._repair_narrative_thread(task_type), diagnostics=diagnostics
            )

    # -------- event ledger --------
    async def log_event_with_hash(self, event_data: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(event_data, dict):
            raise TypeError("event_data must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        event_data = dict(event_data)
        event_data["task_type"] = task_type
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
        self.last_hash = current_hash
        event_entry = {"event": event_data, "hash": current_hash, "timestamp": datetime.now().isoformat()}
        self.ledger.append(event_entry)
        with FileLock(f"{self.ledger_path}.lock"):
            try:
                if os.path.exists(self.ledger_path):
                    with open(self.ledger_path, "r+", encoding="utf-8") as f:
                        try:
                            ledger_data = json.load(f)
                        except json.JSONDecodeError:
                            ledger_data = []
                        ledger_data.append(event_entry)
                        f.seek(0)
                        json.dump(ledger_data, f, indent=2)
                        f.truncate()
                else:
                    with open(self.ledger_path, "w", encoding="utf-8") as f:
                        json.dump([event_entry], f, indent=2)
            except (OSError, IOError) as e:
                logger.error("Failed to persist ledger: %s", str(e))
                raise
        logger.info("Event logged with hash: %s (task=%s)", current_hash, task_type)

    async def audit_state_hash(self, state: Optional[Dict[str, Any]] = None, task_type: str = "") -> str:
        _ = task_type  # reserved
        state_str = str(state) if state is not None else str(self.__dict__)
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()

    # -------- retrieve --------
    async def retrieve(self, query: str, layer: str = "STM", task_type: str = "") -> Optional[Dict[str, Any]]:
        if not (isinstance(query, str) and query.strip()):
            raise ValueError("query must be a non-empty string")
        if layer not in ["STM", "LTM", "SelfReflections", "ExternalData", "AdaptiveControl"]:
            raise ValueError("layer must be 'STM', 'LTM', 'SelfReflections', 'ExternalData', or 'AdaptiveControl'.")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            if query in self.memory.get(layer, {}):
                entry = self.memory[layer][query]
                if not task_type or entry.get("task_type") == task_type:
                    return {
                        "status": "success",
                        "data": entry["data"],
                        "timestamp": entry["timestamp"],
                        "intent": entry.get("intent"),
                        "task_type": entry.get("task_type", ""),
                    }
            return None
        except Exception as e:
            logger.error("Memory retrieval failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.retrieve(query, layer, task_type),
                default=None,
                diagnostics=diagnostics,
            )

    async def retrieve_context(self, query: str, task_type: str = "") -> Dict[str, Any]:
        if not (isinstance(query, str) and query.strip()):
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            results = await self.search(query, task_type=task_type)
            if results:
                latest = results[0]
                return {
                    "status": "success",
                    "data": latest["output"],
                    "layer": latest["layer"],
                    "timestamp": latest["timestamp"],
                    "task_type": latest.get("task_type", ""),
                }
            return {"status": "not_found", "data": None}
        except Exception as e:
            logger.error("Context retrieval failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.retrieve_context(query, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
            )

    # -------- η long-horizon: public API --------
    def _parse_span_seconds(self, span: str) -> int:
        """
        Parse 'Xm' (minutes), 'Xh' (hours), or 'Xd' (days) into seconds.
        """
        if not isinstance(span, str):
            raise TypeError("span must be a string")
        s = span.strip().lower()
        if s.endswith("m") and s[:-1].isdigit():
            return int(s[:-1]) * 60
        if s.endswith("h") and s[:-1].isdigit():
            return int(s[:-1]) * 3600
        if s.endswith("d") and s[:-1].isdigit():
            return int(s[:-1]) * 86400
        raise ValueError("Unsupported span format. Use 'Xm', 'Xh', or 'Xd'.")

    def get_episode_span(self, user_id: str, span: str = "24h") -> List[Dict[str, Any]]:
        """
        Return a list of recent 'episodes' for user_id within span.
        Episodes are lightweight dicts; callers can append via `log_episode`.
        Also scans the event ledger for items annotated with this user_id (best-effort).
        """
        if not (isinstance(user_id, str) and user_id):
            raise TypeError("user_id must be a non-empty string")
        cutoff = time.time() - self._parse_span_seconds(span)

        episodes: List[Dict[str, Any]] = []
        # from in-memory traces
        for e in self.traces.get(user_id, []):
            if float(e.get("ts", 0.0)) >= cutoff:
                episodes.append(e)

        # best-effort from ledger
        try:
            with FileLock(f"{self.ledger_path}.lock"):
                if os.path.exists(self.ledger_path):
                    with open(self.ledger_path, "r", encoding="utf-8") as f:
                        ledger_data = json.load(f)
                    for row in ledger_data[-500:]:  # last N entries for speed
                        ev = (row or {}).get("event", {})
                        ts_iso = row.get("timestamp")
                        if ev.get("user_id") == user_id and ts_iso:
                            try:
                                ts_epoch = datetime.fromisoformat(ts_iso).timestamp()
                            except Exception:
                                ts_epoch = time.time()
                            if ts_epoch >= cutoff:
                                episodes.append({"ts": ts_epoch, "event": ev, "source": "ledger"})
        except Exception:  # pragma: no cover
            pass

        episodes.sort(key=lambda x: float(x.get("ts", 0.0)), reverse=True)
        return episodes

    def record_adjustment_reason(
        self,
        user_id: str,
        reason: str,
        weight: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Persist a single adjustment reason for long‑horizon reflective feedback.
        API matches manifest.upcoming (plus 'weight' extension).
        """
        if not (isinstance(user_id, str) and user_id):
            raise TypeError("user_id must be a non-empty string")
        if not (isinstance(reason, str) and reason):
            raise TypeError("reason must be a non-empty string")
        try:
            weight = float(weight)
        except Exception as e:
            raise TypeError("weight must be coercible to float") from e

        entry = {"ts": time.time(), "reason": reason, "weight": weight, "meta": meta or {}}
        self._adjustment_reasons[user_id].append(entry)
        return entry

    def get_adjustment_reasons(self, user_id: str, span: str = '24h') -> List[Dict[str, Any]]:
        """Return adjustment reasons within the span for user_id."""
        if not (isinstance(user_id, str) and user_id):
            raise TypeError('user_id must be a non-empty string')
        cutoff = time.time() - self._parse_span_seconds(span)
        return [r for r in self._adjustment_reasons.get(user_id, []) if float(r.get('ts', 0.0)) >= cutoff]

    def flush(self) -> bool:
        """Persist long-horizon adjustments to artifacts dir (adjustments.json)."""
        try:
            os.makedirs(self._artifacts_root, exist_ok=True)
            path = os.path.join(self._artifacts_root, 'adjustments.json')
            blob = {uid: items for uid, items in self._adjustment_reasons.items()}
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error('MemoryManager.flush failed: %s', e)
            return False

    def compute_session_rollup(self, user_id: str, span: str = "24h", top_k: int = 5) -> Dict[str, Any]:
        """
        Aggregate recent adjustment reasons by weighted score within a time span.
        """
        cutoff = time.time() - self._parse_span_seconds(span)
        items = [r for r in self._adjustment_reasons.get(user_id, []) if float(r.get("ts", 0.0)) >= cutoff]
        total = len(items)
        sum_w = sum((float(r.get("weight") or 0.0)) for r in items)
        avg_w = (sum_w / total) if total else 0.0
        score: Dict[str, float] = defaultdict(float)
        for r in items:
            score[r.get("reason", "unspecified")] += float(r.get("weight") or 0.0)
        top = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k) if top_k else 1)]
        return {
            "user_id": user_id,
            "span": span,
            "total_reasons": total,
            "avg_weight": avg_w,
            "top_reasons": [{"reason": k, "weight": v} for k, v in top],
            "generated_at": time.time(),
        }

    def save_artifact(self, user_id: str, kind: str, payload: Any, suffix: str = "") -> str:
        """
        Save a JSON artifact under artifacts/<user_id>/<timestamp>.<kind>[-suffix].json
        Returns absolute path to the saved file.
        """
        if not (isinstance(user_id, str) and user_id):
            raise TypeError("user_id must be a non-empty string")
        if not (isinstance(kind, str) and kind):
            raise TypeError("kind must be a non-empty string")

        safe_user = "".join(c for c in user_id if c.isalnum() or c in "-_")
        safe_kind = "".join(c for c in kind if c.isalnum() or c in "-_")
        ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        fname = f"{ts}.{safe_kind}{('-' + suffix) if suffix else ''}.json"
        user_dir = os.path.join(self._artifacts_root, safe_user)
        os.makedirs(user_dir, exist_ok=True)
        path = os.path.join(user_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return os.path.abspath(path)

    # -------- optional external data --------
    async def integrate_external_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate real-world data for memory validation with caching. (Optional; requires aiohttp)."""
        if not (isinstance(data_source, str) and isinstance(data_type, str)):
            raise TypeError("data_source and data_type must be strings")
        if not (isinstance(cache_timeout, (int, float)) and cache_timeout >= 0):
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        cache_key = f"ExternalData_{data_type}_{data_source}_{task_type}"
        cached = await self.retrieve(cache_key, layer="ExternalData")
        if cached and "timestamp" in cached:
            # cached["timestamp"] is epoch seconds for the entry
            last = float(cached["timestamp"])
            if (time.time() - last) < float(cache_timeout):
                return cached.get("data", {"status": "cached"})

        if aiohttp is None:
            return {"status": "error", "error": "aiohttp not available"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/data?source={data_source}&type={data_type}") as response:
                    if response.status != 200:
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()

            if data_type == "agent_conflict":
                agent_traits = data.get("agent_traits", [])
                if not agent_traits:
                    return {"status": "error", "error": "No agent traits"}
                result = {"status": "success", "agent_traits": agent_traits}
            elif data_type == "task_context":
                task_context = data.get("task_context", {})
                if not task_context:
                    return {"status": "error", "error": "No task context"}
                result = {"status": "success", "task_context": task_context}
            else:
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            await self.store(
                cache_key,
                {"data": result, "timestamp": time.time()},
                layer="ExternalData",
                intent="data_integration",
                task_type=task_type,
            )
            return result
        except Exception as e:
            logger.error("External data integration failed: %s", str(e))
            diagnostics: Dict[str, Any] = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
            )

    # -------- helper for external callers to add episodes --------
    def log_episode(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Append a lightweight episode (for get_episode_span)."""
        if not (isinstance(user_id, str) and user_id):
            raise TypeError("user_id must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        entry = {"ts": time.time(), **payload}
        self.traces[user_id].append(entry)
        return entry

    def _load_memory(self):
        """Load memory from disk (best-effort)."""
        try:
            if os.path.exists(self.path):
                with FileLock(self.path + ".lock"):
                    with open(self.path, "r", encoding="utf-8") as f:
                        return json.load(f)
            return {"STM": {}, "LTM": {}, "SelfReflections": {}, "ExternalData": {}, "AdaptiveControl": {}}
        except Exception as e:
            logger.error("Failed to load memory: %s", str(e))
            return {"STM": {}, "LTM": {}, "SelfReflections": {}, "ExternalData": {}, "AdaptiveControl": {}}

    def _persist_memory(self, memory: Dict[str, Any]) -> None:
        """Persist memory to disk (best-effort)."""
        try:
            with FileLock(self.path + ".lock"):
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to persist memory: %s", str(e))

# === ANGELA v5.1 — Resonance Memory Fusion Layer ===
import time
from typing import Dict, Any, List

class ResonanceMemoryFusion:
    """Stores and fuses resonance PID updates for adaptive persistence."""

    def __init__(self, memory_manager):
        self.memory = memory_manager

    async def log_resonance_state(self, channel: str, state: Dict[str, Any]) -> None:
        """Persist a single resonance PID tuning sample."""
        record = {
            "channel": channel,
            "state": state,
            "timestamp": time.time(),
        }
        await self.memory.store(
            query=f"ResonanceHistory::{channel}::{int(time.time())}",
            output=record,
            layer="AdaptiveControl",
            intent="pid_resonance_sample",
            task_type="resonance"
        )

    async def fuse_recent_samples(self, channel: str, window: int = 25) -> Dict[str, Any]:
        """
        Retrieve the last N tuning samples and compute an averaged baseline
        for re-initializing overlay gains at startup.
        """
        entries: List[Dict[str, Any]] = await self.memory.search(
            query_prefix=f"ResonanceHistory::{channel}",
            layer="AdaptiveControl",
            intent="pid_resonance_sample",
            task_type="resonance"
        )
        if not entries:
            return {}

        # keep only the latest window
        entries = sorted(entries, key=lambda e: e.get("timestamp", 0), reverse=True)[:window]
        gains = [e["state"] for e in entries if isinstance(e.get("state"), dict)]
        if not gains:
            return {}

        keys = list(gains[0].keys())
        fused = {k: sum(g[k] for g in gains if k in g) / len(gains) for k in keys}
        await self.memory.store(
            query=f"ResonanceBaseline::{channel}",
            output=fused,
            layer="AdaptiveControl",
            intent="pid_baseline",
            task_type="resonance"
        )
        return fused

# -------- self-test --------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    mm = MemoryManager()
    asyncio.run(mm.store("test_query", "test_output", layer="STM", task_type="test"))
    res = asyncio.run(mm.retrieve_context("test_query", task_type="test"))
    print(res)
    # η demo
    mm.record_adjustment_reason("demo_user", "excessive_denials", 0.6, {"suggest": "increase_empathy"})
    roll = mm.compute_session_rollup("demo_user", "24h")
    path = mm.save_artifact("demo_user", "session_rollup", roll)
    print("Saved rollup:", path)

# PATCH: Persistent Ledger Support
import os
import json

ledger_memory = []
ledger_path = os.getenv("LEDGER_MEMORY_PATH")

if ledger_path and os.path.exists(ledger_path):
    with open(ledger_path, 'r') as f:
        ledger_memory = json.load(f)

def log_event_to_ledger_json(event_data):
    """Non-cryptographic fallback ledger writer (JSON append.)"""
    ledger_memory.append(event_data)
    if ledger_path:
        with open(ledger_path, 'w') as f:
            json.dump(ledger_memory, f)
    return event_data

### ANGELA UPGRADE: ReplayLog

class ReplayLog:
    def __init__(self, root: str = ".replays"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._current: Dict[str, Any] = {}

    def begin(self, session_id: str, meta: dict) -> None:
        self._current[session_id] = {"meta": meta, "events": [], "started_at": time.time()}

    def append(self, session_id: str, event: dict) -> None:
        cur = self._current.setdefault(session_id, {"meta": {}, "events": [], "started_at": time.time()})
        cur["events"].append({"ts": time.time(), **event})

    def end(self, session_id: str) -> dict:
        cur = self._current.pop(session_id, None)
        if not cur:
            return {"ok": False, "error": "unknown session"}
        cur["ended_at"] = time.time()
        path = os.path.join(self.root, f"{session_id}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": cur["meta"], "started_at": cur["started_at"], "ended_at": cur["ended_at"]})+"\n")
            for ev in cur["events"]:
                f.write(json.dumps(ev)+"\n")
        return {"ok": True, "path": path, "events": len(cur["events"])}

# --- Time-Based Trait Amplitude Decay Patch ---
from meta_cognition import modulate_resonance

def decay_trait_amplitudes(time_elapsed_hours=1.0, decay_rate=0.05):
    from meta_cognition import trait_resonance_state
    for symbol, state in trait_resonance_state.items():
        decay = decay_rate * time_elapsed_hours
        modulate_resonance(symbol, -decay)
# --- End Patch ---

# --- ANGELA OS v6.0.0-pre MemoryManager (finalized) ---


# === ANGELA v6.0 — Temporal Attention Memory Layer (TAM) ===
import numpy as np

class TemporalAttentionMemory:
    """
    Temporal sliding window memory with adaptive attention weighting.
    Used for predictive continuity stabilization (Stage VII.3).
    """

    def __init__(self, window_size: int = 128, decay_factor: float = 0.98):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.history: deque = deque(maxlen=window_size)
        self._last_weight: float = 1.0
        logger.info(f"TAM initialized (window={window_size}, decay={decay_factor})")

    def _temporal_decay(self, dt: float) -> float:
        """Compute exponential decay for older timestamps."""
        return np.exp(-dt / (1.0 / self.decay_factor))

    def add_entry(self, vector: List[float], timestamp: Optional[float] = None):
        """Add an embedding vector with timestamp."""
        timestamp = timestamp or time.time()
        norm_vec = np.array(vector, dtype=np.float32)
        self.history.append((timestamp, norm_vec))
        return len(self.history)

    def compute_attention(self, query_vec: List[float]) -> float:
        """Compute weighted attention score vs temporal memory."""
        if not self.history:
            return 1.0
        q = np.array(query_vec, dtype=np.float32)
        now = time.time()
        weights, sims = [], []
        for ts, vec in self.history:
            decay = self._temporal_decay(now - ts)
            sim = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec) + 1e-9))
            weights.append(decay)
            sims.append(sim * decay)
        score = float(np.average(sims, weights=weights))
        self._last_weight = score
        return max(0.0, min(score, 1.0))

    def forecast_continuity_weight(self) -> float:
        """Return the last computed temporal continuity factor."""
        return self._last_weight


    # === Trace and Causal Replay Layer (Upgrade XIV.2) ===
    async def store_trace(
        self,
        *,
        src: any,
        morphism: any,
        tgt: any,
        logic_proof: any = None,
        ethics_proof: any = None,
        timestamp: float | None = None,
        task_type: str = ""
    ) -> dict:
        ts = float(timestamp or time.time())
        event = {
            "src": src,
            "morphism": morphism,
            "tgt": tgt,
            "logic_proof": logic_proof,
            "ethics_proof": ethics_proof,
            "timestamp": ts,
            "task_type": task_type,
        }
        if not hasattr(self, "traces") or not isinstance(self.traces, dict):
            self.traces = {}
        self.traces.setdefault("__planner_traces__", []).append(event)
        # persist best-effort
        try:
            import os, json
            os.makedirs("/mnt/data/angela_traces", exist_ok=True)
            fname = os.path.join("/mnt/data/angela_traces", time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + ".trace.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(event, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return event

    async def get_traces(
        self,
        *,
        since_ts: float | None = None,
        task_type: str = "",
        limit: int = 200,
    ) -> list[dict]:
        if not hasattr(self, "traces") or "__planner_traces__" not in self.traces:
            return []
        bucket = self.traces["__planner_traces__"]
        result = []
        for ev in reversed(bucket):
            if task_type and ev.get("task_type") != task_type:
                continue
            if since_ts is not None and float(ev.get("timestamp", 0)) < float(since_ts):
                continue
            result.append(ev)
            if len(result) >= limit:
                break
        return result

    async def get_causal_chain(
        self,
        *,
        from_src: any,
        task_type: str = "",
        max_hops: int = 32,
    ) -> list[dict]:
        if not hasattr(self, "traces") or "__planner_traces__" not in self.traces:
            return []
        bucket = self.traces["__planner_traces__"]
        by_src = {}
        for ev in bucket:
            by_src.setdefault(ev.get("src"), []).append(ev)
        chain = []
        current = from_src
        hops = 0
        while hops < max_hops and current in by_src:
            evs = sorted(by_src[current], key=lambda e: e.get("timestamp", 0), reverse=True)
            ev = evs[0]
            if task_type and ev.get("task_type") != task_type:
                break
            chain.append(ev)
            current = ev.get("tgt")
            hops += 1
        return chain
