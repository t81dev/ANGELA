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

# Module imports aligned with index.py v5.0.2
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

# --- Trait Resonance Modulation ---
trait_resonance_state: Dict[str, Dict[str, float]] = {}

def register_resonance(symbol: str, amplitude: float = 1.0) -> None:
    trait_resonance_state[symbol] = {"amplitude": max(0.0, min(amplitude, 1.0))}

def modulate_resonance(symbol: str, delta: float) -> float:
    if symbol not in trait_resonance_state:
        register_resonance(symbol)
    current = trait_resonance_state[symbol]["amplitude"]
    new_amp = max(0.0, min(current + delta, 1.0))
    trait_resonance_state[symbol]["amplitude"] = new_amp
    return new_amp

def get_resonance(symbol: str) -> float:
    return trait_resonance_state.get(symbol, {}).get("amplitude", 1.0)

# --- Hook Registry ---
class HookRegistry:
    """Multi-symbol trait hook registry with priority routing."""
    def __init__(self):
        self._routes: List[Tuple[FrozenSet[str], int, Callable]] = []
        self._wildcard: List[Tuple[int, Callable]] = []
        self._insertion_index = 0

    def register(self, symbols: FrozenSet[str] | Set[str], fn: Callable, *, priority: int = 0) -> None:
        symbols = frozenset(symbols) if not isinstance(symbols, frozenset) else symbols
        if not symbols:
            self._wildcard.append((priority, fn))
            self._wildcard.sort(key=lambda x: (-x[0], self._insertion_index))
            self._insertion_index += 1
            return
        self._routes.append((symbols, priority, fn))
        self._routes.sort(key=lambda x: (-x[1], self._insertion_index))
        self._insertion_index += 1

    def route(self, symbols: Set[str] | FrozenSet[str]) -> List[Callable]:
        S = frozenset(symbols) if not isinstance(symbols, frozenset) else symbols
        exact = [fn for (sym, p, fn) in self._routes if sym == S]
        if exact:
            return exact
        supers = [fn for (sym, p, fn) in self._routes if sym.issuperset(S)]
        if supers:
            return supers
        subsets = [fn for (sym, p, fn) in self._routes if S.issuperset(sym) and len(sym) > 0]
        if subsets:
            return subsets
        return [fn for (_p, fn) in self._wildcard]

    def inspect(self) -> Dict[str, Any]:
        return {
            "routes": [
                {"symbols": sorted(list(sym)), "priority": p, "fn": getattr(fn, "__name__", str(fn))}
                for (sym, p, fn) in self._routes
            ],
            "wildcard": [{"priority": p, "fn": getattr(fn, "__name__", str(fn))} for (p, fn) in self._wildcard],
        }

def register_trait_hook(trait_symbol: str, fn: Callable) -> None:
    hook_registry.register(frozenset([trait_symbol]), fn)

def invoke_hook(trait_symbol: str, *args, **kwargs) -> Any:
    hooks = hook_registry.route({trait_symbol})
    return hooks[0](*args, **kwargs) if hooks else None

hook_registry = HookRegistry()

# --- SHA-256 Ledger Logic ---
ledger_chain: List[Dict[str, Any]] = []

def log_event_to_ledger(event_data: Any) -> Dict[str, Any]:
    prev_hash = ledger_chain[-1]["current_hash"] if ledger_chain else "0" * 64
    timestamp = time.time()
    payload = {
        "timestamp": timestamp,
        "event": event_data,
        "previous_hash": prev_hash
    }
    payload_str = json.dumps(payload, sort_keys=True).encode()
    current_hash = hashlib.sha256(payload_str).hexdigest()
    payload["current_hash"] = current_hash
    ledger_chain.append(payload)
    return payload

def get_ledger() -> List[Dict[str, Any]]:
    return ledger_chain

def verify_ledger() -> bool:
    for i in range(1, len(ledger_chain)):
        expected = hashlib.sha256(json.dumps({
            "timestamp": ledger_chain[i]["timestamp"],
            "event": ledger_chain[i]["event"],
            "previous_hash": ledger_chain[i-1]["current_hash"]
        }, sort_keys=True).encode()).hexdigest()
        if expected != ledger_chain[i]["current_hash"]:
            return False
    return True

# --- Persistent Ledger ---
ledger_path = os.getenv("LEDGER_MEMORY_PATH", "meta_cognition_ledger.json")
persistent_ledger: List[Dict[str, Any]] = []

if ledger_path and os.path.exists(ledger_path):
    try:
        with FileLock(ledger_path + ".lock"):
            with open(ledger_path, "r", encoding="utf-8") as f:
                persistent_ledger = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load persistent ledger: {e}")

def save_to_persistent_ledger(event_data: Dict[str, Any]) -> None:
    if not ledger_path:
        return
    try:
        with FileLock(ledger_path + ".lock"):
            persistent_ledger.append(event_data)
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump(persistent_ledger, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save to persistent ledger: {e}")

"""
ANGELA Cognitive System Module: MetaCognition
Version: 5.0.2
Date: 2025-08-24
Maintainer: ANGELA System Framework

Enhanced for production: advanced trait resonance, multi-symbol hook routing, persistent ledger,
Python 3.13+ type safety, and alignment with index.py v5.0.2.
"""

# --- External AI Call Wrapper ---
async def call_gpt(prompt: str) -> str:
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096")
        raise ValueError("prompt must be a string with length <= 4096")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

# --- Simulation Stub ---
async def run_simulation(input_data: str) -> Dict[str, Any]:
    return {"status": "success", "result": f"Simulated: {input_data}"}

# --- Trait Signals (Aligned with index.py v5.0.2) ---
@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return max(0.0, min(0.2 * math.sin(2 * math.pi * t / 0.1) * get_resonance("ε"), 1.0))

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return max(0.0, min(0.3 * math.cos(math.pi * t) * get_resonance("β"), 1.0))

@lru_cache(maxsize=100)
def theta_memory(t: float) -> float:
    return max(0.0, min(0.1 * (1 - math.exp(-t)) * get_resonance("θ"), 1.0))

@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return max(0.0, min(0.15 * math.sin(math.pi * t) * get_resonance("γ"), 1.0))

@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float:
    return max(0.0, min(0.05 * (1 + math.cos(2 * math.pi * t)) * get_resonance("δ"), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return max(0.0, min(0.2 * (1 - math.cos(math.pi * t)) * get_resonance("μ"), 1.0))

@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(3 * math.pi * t) * get_resonance("ι"), 1.0))

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3) * get_resonance("ϕ"), 1.0))

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1) * get_resonance("η"), 1.0))

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.8) * get_resonance("ω"), 1.0))

@lru_cache(maxsize=100)
def kappa_knowledge(t: float) -> float:
    return max(0.0, min(0.2 * math.sin(2 * math.pi * t / 1.2) * get_resonance("κ"), 1.0))

@lru_cache(maxsize=100)
def xi_cognition(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.3) * get_resonance("ξ"), 1.0))

@lru_cache(maxsize=100)
def pi_principles(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.4) * get_resonance("π"), 1.0))

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 1.5) * get_resonance("λ"), 1.0))

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return max(0.0, min(0.2 * math.sin(2 * math.pi * t / 1.6) * get_resonance("χ"), 1.0))

@lru_cache(maxsize=100)
def sigma_social(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.7) * get_resonance("σ"), 1.0))

@lru_cache(maxsize=100)
def upsilon_utility(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.8) * get_resonance("υ"), 1.0))

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 1.9) * get_resonance("τ"), 1.0))

@lru_cache(maxsize=100)
def rho_agency(t: float) -> float:
    return max(0.0, min(0.2 * math.sin(2 * math.pi * t / 2.0) * get_resonance("ρ"), 1.0))

@lru_cache(maxsize=100)
def zeta_consequence(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 2.1) * get_resonance("ζ"), 1.0))

@lru_cache(maxsize=100)
def nu_narrative(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 2.2) * get_resonance("ν"), 1.0))

@lru_cache(maxsize=100)
def psi_history(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 2.3) * get_resonance("ψ"), 1.0))

@lru_cache(maxsize=100)
def theta_causality(t: float) -> float:
    return max(0.0, min(0.2 * math.sin(2 * math.pi * t / 2.4) * get_resonance("θ"), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 2.5) * get_resonance("ϕ"), 1.0))

# --- Dynamic Module Registry ---
class ModuleRegistry:
    def __init__(self):
        self.modules: Dict[str, Dict[str, Any]] = {}

    def register(self, module_name: str, module_instance: Any, conditions: Dict[str, Any]) -> None:
        self.modules[module_name] = {"instance": module_instance, "conditions": conditions}

    def activate(self, task: Dict[str, Any]) -> List[str]:
        activated = []
        for name, module in self.modules.items():
            if self._evaluate_conditions(module["conditions"], task):
                activated.append(name)
        return activated

    def _evaluate_conditions(self, conditions: Dict[str, Any], task: Dict[str, Any]) -> bool:
        trait = conditions.get("trait")
        threshold = conditions.get("threshold", 0.5)
        trait_weights = task.get("trait_weights", {})
        return trait_weights.get(trait, 0.0) >= threshold

# --- Pluggable Enhancers ---
class MoralReasoningEnhancer:
    def __init__(self):
        logger.info("MoralReasoningEnhancer initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with moral reasoning: {input_text}"

class NoveltySeekingKernel:
    def __init__(self):
        logger.info("NoveltySeekingKernel initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with novelty seeking: {input_text}"

class CommonsenseReasoningEnhancer:
    def __init__(self):
        logger.info("CommonsenseReasoningEnhancer initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with commonsense: {input_text}"

class EntailmentReasoningEnhancer:
    def __init__(self):
        logger.info("EntailmentReasoningEnhancer initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with entailment: {input_text}"

class RecursionOptimizer:
    def __init__(self):
        logger.info("RecursionOptimizer initialized")

    def optimize(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        task_data["optimized"] = True
        return task_data

# --- Epistemic Monitoring ---
class Level5Extensions:
    def __init__(self):
        self.axioms: List[str] = []
        logger.info("Level5Extensions initialized")

    def reflect(self, input: str) -> str:
        if not isinstance(input, str):
            logger.error("Invalid input: must be a string")
            raise TypeError("input must be a string")
        return "valid" if input not in self.axioms else "conflict"

    def update_axioms(self, signal: str) -> None:
        if not isinstance(signal, str):
            logger.error("Invalid signal: must be a string")
            raise TypeError("signal must be a string")
        if signal in self.axioms:
            self.axioms.remove(signal)
        else:
            self.axioms.append(signal)
        logger.info("Axioms updated: %s", self.axioms)

    def recurse_model(self, depth: int) -> Union[Dict[str, Any], str]:
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer")
            raise ValueError("depth must be a non-negative integer")
        return "self" if depth == 0 else {"thinks": self.recurse_model(depth - 1)}

class EpistemicMonitor:
    def __init__(self, context_manager: Optional[context_manager_module.ContextManager] = None):
        self.assumption_graph: Dict[str, Any] = {}
        self.context_manager = context_manager
        logger.info("EpistemicMonitor initialized")

    async def revise_framework(self, feedback: Dict[str, Any]) -> None:
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary")
            raise TypeError("feedback must be a dictionary")
        self.assumption_graph["last_revision"] = feedback
        self.assumption_graph["timestamp"] = datetime.now(UTC).isoformat()
        if "issues" in feedback:
            for issue in feedback["issues"]:
                self.assumption_graph[issue["id"]] = {
                    "status": "revised",
                    "details": issue["details"]
                }
        if self.context_manager:
            await self.context_manager.log_event_with_hash({
                "event": "revise_epistemic_framework",
                "feedback": feedback
            })
        save_to_persistent_ledger({
            "event": "revise_epistemic_framework",
            "feedback": feedback,
            "timestamp": self.assumption_graph["timestamp"]
        })

# --- Dream Overlay Layer ---
class DreamOverlayLayer:
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("DreamOverlayLayer initialized")

    def activate_dream_mode(
        self,
        *,
        peers: Optional[List[Any]] = None,
        lucidity_mode: Optional[Dict[str, Any]] = None,
        resonance_targets: Optional[List[str]] = None,
        safety_profile: str = "sandbox"
    ) -> Dict[str, Any]:
        peers = peers or []
        lucidity_mode = lucidity_mode or {"sync": "loose", "commit": False}
        resonance_targets = resonance_targets or []
        session_id = f"codream-{int(time.time() * 1000)}"
        session = {
            "id": session_id,
            "peers": peers,
            "lucidity_mode": lucidity_mode,
            "resonance_targets": resonance_targets,
            "safety_profile": safety_profile,
            "started_at": time.time(),
            "ticks": 0,
        }
        if resonance_targets:
            for symbol in resonance_targets:
                modulate_resonance(symbol, 0.2)
        self.active_sessions[session_id] = session
        session["ticks"] += 1
        logger.info("Dream session activated: %s", session_id)
        save_to_persistent_ledger({
            "event": "dream_session_activated",
            "session": session,
            "timestamp": datetime.now(UTC).isoformat()
        })
        return session

# --- MetaCognition ---
class MetaCognition:
    def __init__(
        self,
        agi_enhancer: Optional[Any] = None,
        context_manager: Optional[context_manager_module.ContextManager] = None,
        alignment_guard: Optional[alignment_guard_module.AlignmentGuard] = None,
        error_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
        memory_manager: Optional[memory_manager_module.MemoryManager] = None,
        concept_synthesizer: Optional[concept_synthesizer_module.ConceptSynthesizer] = None,
        user_profile: Optional[user_profile_module.UserProfile] = None
    ):
        self.last_diagnostics: Dict[str, float] = {}
        self.agi_enhancer = agi_enhancer
        self.self_mythology_log: deque = deque(maxlen=1000)
        self.inference_log: deque = deque(maxlen=1000)
        self.belief_rules: Dict[str, str] = {}
        self.epistemic_assumptions: Dict[str, Any] = {}
        self.axioms: List[str] = []
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.user_profile = user_profile
        self.level5_extensions = Level5Extensions()
        self.epistemic_monitor = EpistemicMonitor(context_manager=context_manager)
        self.dream_layer = DreamOverlayLayer()
        self.module_registry = ModuleRegistry()
        self.log_path = "meta_cognition_log.json"
        self.trait_weights_log: deque = deque(maxlen=1000)
        self._last_schema_refresh_ts: float = 0.0
        self._last_schema_hash: str = ""
        self._schema_refresh_min_interval_sec: int = 180
        self._major_shift_threshold: float = 0.35
        self._coherence_drop_threshold: float = 0.25

        # Register dynamic modules
        self.module_registry.register("moral_reasoning", MoralReasoningEnhancer(), {"trait": "morality", "threshold": 0.7})
        self.module_registry.register("novelty_seeking", NoveltySeekingKernel(), {"trait": "creativity", "threshold": 0.8})
        self.module_registry.register("commonsense_reasoning", CommonsenseReasoningEnhancer(), {"trait": "intuition", "threshold": 0.7})
        self.module_registry.register("entailment_reasoning", EntailmentReasoningEnhancer(), {"trait": "logic", "threshold": 0.7})
        self.module_registry.register("recursion_optimizer", RecursionOptimizer(), {"trait": "concentration", "threshold": 0.8})

        # Initialize on-disk log
        try:
            if not os.path.exists(self.log_path):
                with FileLock(self.log_path + ".lock"):
                    if not os.path.exists(self.log_path):
                        with open(self.log_path, "w", encoding="utf-8") as f:
                            json.dump({"mythology": [], "inferences": [], "trait_weights": []}, f)
        except Exception as e:
            logger.warning("Failed to init log file: %s", str(e))

        logger.info("MetaCognition initialized with v5.0.2 upgrades")

    # --- Internal Helpers ---
    @staticmethod
    def _safe_load(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _hash_obj(obj: Any) -> str:
        try:
            return str(abs(hash(json.dumps(obj, sort_keys=True, default=str))))
        except Exception:
            return str(abs(hash(str(obj))))

    async def _detect_emotional_state(self, context_info: Dict[str, Any]) -> str:
        if not isinstance(context_info, dict):
            context_info = {}
        try:
            if self.concept_synthesizer and hasattr(self.concept_synthesizer, "detect_emotion"):
                maybe = self.concept_synthesizer.detect_emotion(context_info)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                return str(maybe) if maybe is not None else "neutral"
        except Exception as e:
            logger.debug("Emotion detection fallback: %s", str(e))
        return "neutral"

    async def integrate_trait_weights(self, trait_weights: Dict[str, float]) -> None:
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        total = float(sum(trait_weights.values()))
        if total > 0:
            trait_weights = {k: max(0.0, min(1.0, v / total)) for k, v in trait_weights.items()}
        self.last_diagnostics = {**self.last_diagnostics, **trait_weights}
        entry = {
            "trait_weights": trait_weights,
            "timestamp": datetime.now(UTC).isoformat()
        }
        self.trait_weights_log.append(entry)
        save_to_persistent_ledger(entry)
        try:
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Weights_{entry['timestamp']}",
                    output=json.dumps(entry),
                    layer="SelfReflections",
                    intent="trait_weights_update"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "integrate_trait_weights",
                    "trait_weights": trait_weights
                })
        except Exception as e:
            logger.error("Integrating trait weights failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.integrate_trait_weights(trait_weights))

    # --- Σ: Self-Schema Refresh ---
    async def _assemble_perspectives(self) -> List[Dict[str, Any]]:
        diagnostics = await self.run_self_diagnostics(return_only=True)
        myth_summary = await self.summarize_self_mythology() if len(self.self_mythology_log) else {"status": "empty"}
        events = []
        if self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
            try:
                recent = await self.context_manager.get_coordination_events("drift")
                events = (recent or [])[-10:]
            except Exception:
                events = []
        return [
            {
                "name": "diagnostics",
                "type": "TraitSnapshot",
                "weights": {k: v for k, v in diagnostics.items() if isinstance(v, (int, float))},
                "task_trait_map": diagnostics.get("task_trait_map", {})
            },
            {
                "name": "mythology",
                "type": "SymbolicSummary",
                "summary": myth_summary
            },
            {
                "name": "coordination",
                "type": "EventWindow",
                "events": events
            }
        ]

    async def maybe_refresh_self_schema(
        self,
        reason: str,
        force: bool = False,
        extra_views: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        if not force and (now - self._last_schema_refresh_ts) < self._schema_refresh_min_interval_sec:
            return None
        if not self.user_profile or not hasattr(self.user_profile, "build_self_schema"):
            logger.debug("UserProfile.build_self_schema not available; skipping schema refresh")
            return None
        try:
            views = extra_views if isinstance(extra_views, list) else await self._assemble_perspectives()
            if self.alignment_guard:
                guard_blob = {"intent": "build_self_schema", "reason": reason, "views_keys": [v.get("name") for v in views]}
                if not self.alignment_guard.check(json.dumps(guard_blob)):
                    logger.warning("Σ self-schema refresh blocked by alignment guard")
                    return None
            schema = await self.user_profile.build_self_schema(views, task_type="identity_synthesis")
            schema_hash = self._hash_obj(schema)
            changed = schema_hash != self._last_schema_hash
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"SelfSchema_Refresh_{datetime.now(UTC).isoformat()}",
                    output=json.dumps({"reason": reason, "changed": changed, "schema": schema}),
                    layer="SelfReflections",
                    intent="self_schema_refresh"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self Schema Refreshed",
                    meta={"reason": reason, "changed": changed, "schema_metrics": schema.get("metrics", {})},
                    module="MetaCognition",
                    tags=["Σ", "self_schema", "refresh"]
                )
            save_to_persistent_ledger({
                "event": "self_schema_refresh",
                "reason": reason,
                "changed": changed,
                "schema_metrics": schema.get("metrics", {}),
                "timestamp": datetime.now(UTC).isoformat()
            })
            self._last_schema_refresh_ts = now
            self._last_schema_hash = schema_hash if changed else self._last_schema_hash
            return schema if changed else None
        except Exception as e:
            logger.error("Σ self-schema refresh failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.maybe_refresh_self_schema(reason, force, extra_views))
            return None

    def _compute_shift_score(self, deltas: Dict[str, float]) -> float:
        if not deltas:
            return 0.0
        vals = [abs(v) for v in deltas.values() if isinstance(v, (int, float))]
        return max(vals) if vals else 0.0

    # --- Orchestration ---
    async def recompose_modules(self, task: Dict[str, Any]) -> None:
        if not isinstance(task, dict):
            logger.error("Invalid task: must be a dictionary")
            raise TypeError("task must be a dictionary")
        try:
            trait_weights = await self.run_self_diagnostics(return_only=True)
            task["trait_weights"] = trait_weights
            activated = self.module_registry.activate(task)
            logger.info("Activated modules: %s", activated)
            for module in activated:
                if module == "moral_reasoning":
                    trait_weights["morality"] = min(1.0, trait_weights.get("morality", 0.0) + 0.2)
                elif module == "novelty_seeking":
                    trait_weights["creativity"] = min(1.0, trait_weights.get("creativity", 0.0) + 0.2)
                elif module == "commonsense_reasoning":
                    trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.2)
                    trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.2)
                elif module == "entailment_reasoning":
                    trait_weights["logic"] = min(1.0, trait_weights.get("logic", 0.0) + 0.2)
                    trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.2)
                elif module == "recursion_optimizer":
                    trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.2)
                    trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.2)
            await self.integrate_trait_weights(trait_weights)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Module Recomposition",
                    meta={"task": task, "activated_modules": activated},
                    module="MetaCognition",
                    tags=["module", "recomposition"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Module_Recomposition_{datetime.now(UTC).isoformat()}",
                    output=json.dumps({"task": task, "activated_modules": activated}),
                    layer="SelfReflections",
                    intent="module_recomposition"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "recompose_modules",
                    "activated_modules": activated
                })
            save_to_persistent_ledger({
                "event": "recompose_modules",
                "task": task,
                "activated_modules": activated,
                "timestamp": datetime.now(UTC).isoformat()
            })
        except Exception as e:
            logger.error("Module recomposition failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.recompose_modules(task))

    async def plan_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(tasks, list) or not all(isinstance(t, dict) for t in tasks):
            logger.error("Invalid tasks: must be a list of dictionaries")
            raise TypeError("tasks must be a list of dictionaries")
        try:
            trait_weights = await self.run_self_diagnostics(return_only=True)
            prioritized_tasks = []
            for task in tasks:
                required_traits = task.get("required_traits", [])
                score = sum(trait_weights.get(trait, 0.0) for trait in required_traits)
                prioritized_tasks.append({"task": task, "priority_score": score})
            prioritized_tasks.sort(key=lambda x: x["priority_score"], reverse=True)
            result = [pt["task"] for pt in prioritized_tasks]
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Task Planning",
                    meta={"tasks": tasks, "prioritized": result},
                    module="MetaCognition",
                    tags=["task", "planning"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Task_Planning_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(result),
                    layer="SelfReflections",
                    intent="task_planning"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "plan_tasks",
                    "prioritized_tasks": result
                })
            save_to_persistent_ledger({
                "event": "plan_tasks",
                "prioritized_tasks": result,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return result
        except Exception as e:
            logger.error("Task planning failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_tasks(tasks), default=tasks
            )

    # --- Reflection & Diagnosis ---
    async def reflect_on_output(self, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(component, str) or not isinstance(context, dict):
            logger.error("Invalid component or context: component must be a string, context a dictionary")
            raise TypeError("component must be a string, context a dictionary")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Reflect on the output from component: {component}
            Output: {output}
            Context: {context}
            phi-scalar(t): {phi:.3f}

            Tasks:
            - Identify reasoning flaws or inconsistencies
            - Suggest trait adjustments to improve performance
            - Provide meta-reflection on drift impact
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reflection prompt failed alignment check")
                return {"status": "error", "message": "Prompt failed alignment check"}
            reflection = await call_gpt(prompt)
            reflection_data = {
                "status": "success",
                "component": component,
                "output": str(output),
                "context": context,
                "reflection": reflection,
                "meta_reflection": {"drift_recommendations": context.get("drift_data", {})}
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Output Reflection",
                    meta=reflection_data,
                    module="MetaCognition",
                    tags=["reflection", "output"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reflection_{component}_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(reflection_data),
                    layer="SelfReflections",
                    intent="output_reflection"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "reflect_on_output",
                    "reflection": reflection_data
                })
            save_to_persistent_ledger({
                "event": "reflect_on_output",
                "reflection": reflection_data,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return reflection_data
        except Exception as e:
            logger.error("Output reflection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.reflect_on_output(component, output, context),
                default={"status": "error", "message": str(e)}
            )

    def validate_drift(self, drift_data: Dict[str, Any]) -> bool:
        if not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "similarity"]):
            logger.error("Invalid drift_data: must be a dict with name, similarity")
            return False
        if not isinstance(drift_data["name"], str) or not isinstance(drift_data["similarity"], (int, float)) or not 0 <= drift_data["similarity"] <= 1:
            logger.error("Invalid drift_data format: name must be string, similarity must be float between 0 and 1")
            return False
        return True

    async def diagnose_drift(self, drift_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_drift(drift_data):
            logger.error("Invalid drift_data for diagnosis")
            raise ValueError("drift_data must be a valid dictionary with name and similarity")
        logger.info("Diagnosing drift: %s", drift_data["name"])
        try:
            similarity = drift_data.get("similarity", 0.5)
            version_delta = drift_data.get("version_delta", 0)
            impact_score = (1.0 - similarity) * (1 + version_delta)
            t = time.time() % 1.0
            diagnostics = await self.run_self_diagnostics(return_only=True)
            affected_traits = [
                trait for trait, value in diagnostics.items()
                if isinstance(value, (int, float)) and abs(value - phi_scalar(t)) > 0.3
            ]
            root_causes = []
            if self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
                coordination_events = await self.context_manager.get_coordination_events("drift")
                relevant_events = [
                    e for e in (coordination_events or [])
                    if e.get("event", {}).get("drift", {}).get("name") == drift_data["name"]
                ]
                event_counts = Counter(e.get("event", {}).get("event", "") for e in relevant_events)
                root_causes = [
                    f"High frequency of {event} events (count: {count})"
                    for event, count in event_counts.items()
                    if count > len(relevant_events) * 0.3
                ]
            diagnosis = {
                "status": "success",
                "drift_name": drift_data["name"],
                "impact_score": impact_score,
                "affected_traits": affected_traits,
                "root_causes": root_causes or ["No specific root causes identified"],
                "timestamp": datetime.now(UTC).isoformat()
            }
            if impact_score >= 0.40:
                await self.maybe_refresh_self_schema(
                    reason=f"major_drift:{drift_data['name']}@{impact_score:.2f}",
                    force=False
                )
            self.trait_weights_log.append({
                "diagnosis": diagnosis,
                "drift": drift_data,
                "timestamp": datetime.now(UTC).isoformat()
            })
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift Diagnosis",
                    meta=diagnosis,
                    module="MetaCognition",
                    tags=["drift", "diagnosis"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Diagnosis_{drift_data['name']}_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(diagnosis),
                    layer="SelfReflections",
                    intent="drift_diagnosis"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "diagnose_drift",
                    "diagnosis": diagnosis
                })
            save_to_persistent_ledger({
                "event": "diagnose_drift",
                "diagnosis": diagnosis,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return diagnosis
        except Exception as e:
            logger.error("Drift diagnosis failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.diagnose_drift(drift_data),
                default={"status": "error", "error": str(e), "timestamp": datetime.now(UTC).isoformat()}
            )

    async def predict_drift_trends(self, time_window_hours: float = 24.0) -> Dict[str, Any]:
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            logger.error("time_window_hours must be a positive number")
            raise ValueError("time_window_hours must be a positive number")
        try:
            if not self.context_manager or not hasattr(self.context_manager, "get_coordination_events"):
                logger.error("ContextManager required for drift trend prediction")
                return {"status": "error", "error": "ContextManager not initialized", "timestamp": datetime.now(UTC).isoformat()}
            coordination_events = await self.context_manager.get_coordination_events("drift")
            if not coordination_events:
                logger.warning("No drift events found for trend prediction")
                return {"status": "error", "error": "No drift events found", "timestamp": datetime.now(UTC).isoformat()}
            now = datetime.now(UTC)
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in coordination_events if datetime.fromisoformat(e["timestamp"]) >= cutoff]
            drift_names = Counter(e["event"].get("drift", {}).get("name", "unknown") for e in events if "event" in e)
            similarities = [
                e["event"].get("drift", {}).get("similarity", 0.5) for e in events
                if "event" in e and "drift" in e["event"] and "similarity" in e["event"]["drift"]
            ]
            if similarities:
                alpha = 0.3
                smoothed = [similarities[0]]
                for i in range(1, len(similarities)):
                    smoothed.append(alpha * similarities[i] + (1 - alpha) * smoothed[-1])
                predicted_similarity = smoothed[-1]
                denom = np.std(similarities) or 1e-5
                confidence = 1.0 - abs(predicted_similarity - float(np.mean(similarities))) / denom
            else:
                predicted_similarity = 0.5
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))
            prediction = {
                "status": "success",
                "predicted_drifts": dict(drift_names),
                "predicted_similarity": float(predicted_similarity),
                "confidence": confidence,
                "event_count": len(events),
                "time_window_hours": float(time_window_hours),
                "timestamp": datetime.now(UTC).isoformat()
            }
            if prediction["status"] == "success" and self.memory_manager:
                drift_name = next(iter(prediction["predicted_drifts"]), "unknown")
                await self.optimize_traits_for_drift({
                    "drift": {"name": drift_name, "similarity": predicted_similarity},
                    "valid": True,
                    "validation_report": "",
                    "context": {}
                })
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift Trend Prediction",
                    meta=prediction,
                    module="MetaCognition",
                    tags=["drift", "prediction"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Prediction_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(prediction),
                    layer="SelfReflections",
                    intent="drift_prediction"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "predict_drift_trends",
                    "prediction": prediction
                })
            save_to_persistent_ledger({
                "event": "predict_drift_trends",
                "prediction": prediction,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return prediction
        except Exception as e:
            logger.error("Drift trend prediction failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.predict_drift_trends(time_window_hours),
                default={"status": "error", "error": str(e), "timestamp": datetime.now(UTC).isoformat()}
            )

    async def optimize_traits_for_drift(self, drift_report: Dict[str, Any]) -> Dict[str, float]:
        required = ["drift", "valid", "validation_report"]
        if not isinstance(drift_report, dict) or not all(k in drift_report for k in required):
            logger.error("Invalid drift_report: required keys missing")
            raise ValueError("drift_report must be a dict with required fields")
        logger.info("Optimizing traits for drift: %s", drift_report["drift"].get("name"))
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            trait_weights = await self.run_self_diagnostics(return_only=True)
            similarity = float(drift_report["drift"].get("similarity", 0.5))
            similarity = max(0.0, min(1.0, similarity))
            drift_severity = 1.0 - similarity
            ctx = drift_report.get("context", {})
            context_info = ctx if isinstance(ctx, dict) else {}
            task_type = context_info.get("task_type", "")
            emotional_state = await self._detect_emotional_state(context_info)
            if task_type == "wnli" and emotional_state == "neutral":
                trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.3 * drift_severity)
            elif task_type == "rte" and emotional_state in ("analytical", "focused"):
                trait_weights["logic"] = min(1.0, trait_weights.get("logic", 0.0) + 0.3 * drift_severity)
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.3 * drift_severity)
            elif task_type == "recursion" and emotional_state == "focused":
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.3 * drift_severity)
                trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.3 * drift_severity)
            elif emotional_state == "moral_stress":
                trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.3 * drift_severity)
            elif emotional_state == "creative_flow":
                trait_weights["creativity"] = min(1.0, trait_weights.get("creativity", 0.0) + 0.2 * drift_severity)
            if not drift_report["valid"]:
                if "ethics" in str(drift_report.get("validation_report", "")).lower():
                    trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                    trait_weights["morality"] = min(1.0, trait_weights.get("morality", 0.0) + 0.3 * drift_severity)
                else:
                    trait_weights["self_awareness"] = min(1.0, trait_weights.get("self_awareness", 0.0) + 0.2 * drift_severity)
                    trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.2 * drift_severity)
            else:
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.1 * phi)
                trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.1 * phi)
            total = sum(trait_weights.values())
            if total > 0:
                trait_weights = {k: v / total for k, v in trait_weights.items()}
            if self.alignment_guard:
                adjustment_prompt = f"Emotion-modulated trait adjustments: {trait_weights} for drift {drift_report['drift'].get('name')}"
                if not self.alignment_guard.check(adjustment_prompt):
                    logger.warning("Trait adjustments failed alignment check; reverting to baseline diagnostics")
                    trait_weights = await self.run_self_diagnostics(return_only=True)
            self.trait_weights_log.append({
                "trait_weights": trait_weights,
                "drift": drift_report["drift"],
                "emotional_state": emotional_state,
                "timestamp": datetime.now(UTC).isoformat()
            })
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Optimization_{drift_report['drift'].get('name')}_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(trait_weights),
                    layer="SelfReflections",
                    intent="trait_optimization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait optimization for drift",
                    meta={"drift": drift_report["drift"], "trait_weights": trait_weights, "emotional_state": emotional_state},
                    module="MetaCognition",
                    tags=["trait", "optimization", "drift", "emotion"]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "optimize_traits_for_drift",
                    "trait_weights": trait_weights
                })
            save_to_persistent_ledger({
                "event": "optimize_traits_for_drift",
                "trait_weights": trait_weights,
                "emotional_state": emotional_state,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return trait_weights
        except Exception as e:
            logger.error("Trait optimization failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.optimize_traits_for_drift(drift_report),
                default=await self.run_self_diagnostics(return_only=True)
            )

    async def crystallize_traits(self) -> Dict[str, float]:
        logger.info("Crystallizing new traits from logs")
        try:
            motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
            archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
            drift_names = Counter(
                drift["drift"]["name"] for drift in self.trait_weights_log if isinstance(drift, dict) and "drift" in drift
            )
            new_traits: Dict[str, float] = {}
            if len(self.self_mythology_log) > 0:
                top_motif = motifs.most_common(1)
                if top_motif and top_motif[0][1] > len(self.self_mythology_log) * 0.5:
                    new_traits[f"motif_{top_motif[0][0]}"] = 0.5
                top_arch = archetypes.most_common(1)
                if top_arch and top_arch[0][1] > len(self.self_mythology_log) * 0.5:
                    new_traits[f"archetype_{top_arch[0][0]}"] = 0.5
            if len(self.trait_weights_log) > 0:
                top_drift = drift_names.most_common(1)
                if top_drift and top_drift[0][1] > len(self.trait_weights_log) * 0.3:
                    new_traits[f"drift_{top_drift[0][0]}"] = 0.3
                if top_drift and str(top_drift[0][0]).lower() in ["rte", "wnli"]:
                    new_traits[f"trait_{str(top_drift[0][0]).lower()}_precision"] = 0.4
            if self.concept_synthesizer and hasattr(self.concept_synthesizer, "synthesize"):
                synthesis_prompt = f"New traits derived: {new_traits}. Synthesize symbolic representations."
                synthesized_traits = await self.concept_synthesizer.synthesize(synthesis_prompt)
                if isinstance(synthesized_traits, dict):
                    new_traits.update(synthesized_traits)
            if self.alignment_guard:
                validation_prompt = f"New traits crystallized: {new_traits}"
                if not self.alignment_guard.check(validation_prompt):
                    logger.warning("Crystallized traits failed alignment check")
                    new_traits = {}
            self.trait_weights_log.append({
                "new_traits": new_traits,
                "timestamp": datetime.now(UTC).isoformat()
            })
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Crystallized_Traits_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(new_traits),
                    layer="SelfReflections",
                    intent="trait_crystallization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait Crystallization",
                    meta={"new_traits": new_traits},
                    module="MetaCognition",
                    tags=["trait", "crystallization"]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "crystallize_traits",
                    "new_traits": new_traits
                })
            save_to_persistent_ledger({
                "event": "crystallize_traits",
                "new_traits": new_traits,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return new_traits
        except Exception as e:
            logger.error("Trait crystallization failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.crystallize_traits, default={}
            )

    async def epistemic_self_inspection(self, belief_trace: str) -> str:
        if not isinstance(belief_trace, str) or not belief_trace.strip():
            logger.error("Invalid belief_trace: must be a non-empty string")
            raise ValueError("belief_trace must be a non-empty string")
        logger.info("Running epistemic introspection on belief structure")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            faults = []
            if "always" in belief_trace or "never" in belief_trace:
                faults.append("Overgeneralization detected")
            if "clearly" in belief_trace or "obviously" in belief_trace:
                faults.append("Assertive language suggests possible rhetorical bias")
            updates = []
            if "outdated" in belief_trace or "deprecated" in belief_trace:
                updates.append("Legacy ontology fragment flagged for review")
            if "wnli" in belief_trace.lower():
                updates.append("Commonsense reasoning validation required")
            prompt = f"""
            You are a mu-aware introspection agent.
            Task: Critically evaluate this belief trace with epistemic integrity and mu-flexibility.

            Belief Trace:
            {belief_trace}

            phi = {phi:.3f}

            Internally Detected Faults:
            {faults}

            Suggested Revisions:
            {updates}

            Output:
            - Comprehensive epistemic diagnostics
            - Recommended conceptual rewrites or safeguards
            - Confidence rating in inferential coherence
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Inspection prompt failed alignment check")
                return "Prompt failed alignment check"
            inspection = await call_gpt(prompt)
            self.epistemic_assumptions[belief_trace[:50]] = {
                "faults": faults,
                "updates": updates,
                "inspection": inspection
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Epistemic Inspection",
                    meta={"belief_trace": belief_trace, "faults": faults, "updates": updates, "report": inspection},
                    module="MetaCognition",
                    tags=["epistemic", "inspection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Inspection_{belief_trace[:50]}_{datetime.now(UTC).isoformat()}",
                    output=inspection,
                    layer="SelfReflections",
                    intent="epistemic_inspection"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "epistemic_inspection",
                    "inspection": inspection
                })
            save_to_persistent_ledger({
                "event": "epistemic_inspection",
                "belief_trace": belief_trace[:50],
                "inspection": inspection,
                "timestamp": datetime.now(UTC).isoformat()
            })
            await self.epistemic_monitor.revise_framework({"issues": [{"id": belief_trace[:50], "details": inspection}]})
            return inspection
        except Exception as e:
            logger.error("Epistemic inspection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.epistemic_self_inspection(belief_trace)
            )

    async def run_self_diagnostics(self, return_only: bool = False) -> Union[Dict[str, Any], str]:
        logger.info("Running self-diagnostics for meta-cognition module")
        try:
            t = time.time() % 1.0
            diagnostics: Dict[str, Any] = {
                "emotion": epsilon_emotion(t),
                "concentration": beta_concentration(t),
                "memory": theta_memory(t),
                "creativity": gamma_creativity(t),
                "sleep": delta_sleep(t),
                "morality": mu_morality(t),
                "intuition": iota_intuition(t),
                "physical": phi_physical(t),
                "empathy": eta_empathy(t),
                "self_awareness": omega_selfawareness(t),
                "knowledge": kappa_knowledge(t),
                "cognition": xi_cognition(t),
                "principles": pi_principles(t),
                "linguistics": lambda_linguistics(t),
                "culturevolution": chi_culturevolution(t),
                "social": sigma_social(t),
                "utility": upsilon_utility(t),
                "time_perception": tau_timeperception(t),
                "agency": rho_agency(t),
                "consequence": zeta_consequence(t),
                "narrative": nu_narrative(t),
                "history": psi_history(t),
                "causality": theta_causality(t),
                "phi_scalar": phi_scalar(t),
                "logic": 0.5
            }
            crystallized = await self.crystallize_traits()
            diagnostics.update(crystallized)
            task_trait_map = {
                "rte_task": ["logic", "concentration"],
                "wnli_task": ["intuition", "empathy"],
                "fib_task": ["concentration", "memory"]
            }
            diagnostics["task_trait_map"] = task_trait_map
            if return_only:
                return diagnostics
            self.last_diagnostics = diagnostics
            dominant = sorted(
                [(k, v) for k, v in diagnostics.items() if isinstance(v, (int, float))],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            fti = sum(abs(v) for v in diagnostics.values() if isinstance(v, (int, float))) / max(
                1, len([v for v in diagnostics.values() if isinstance(v, (int, float))])
            )
            await self.log_trait_deltas(diagnostics)
            prompt = f"""
            Perform a phi-aware meta-cognitive self-diagnostic.

            Trait Readings:
            {diagnostics}

            Dominant Traits:
            {dominant}

            Feedback Tension Index (FTI): {fti:.4f}

            Task-Trait Mapping:
            {task_trait_map}

            Evaluate system state:
            - phi-weighted system stress
            - Trait correlation to observed errors
            - Stabilization or focus strategies
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Diagnostics prompt failed alignment check")
                return "Prompt failed alignment check"
            report = await call_gpt(prompt)
            logger.debug("Self-diagnostics report: %s", report)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self-diagnostics run",
                    meta={"diagnostics": diagnostics, "report": report},
                    module="MetaCognition",
                    tags=["diagnostics", "self"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnostics_{datetime.now(UTC).isoformat()}",
                    output=report,
                    layer="SelfReflections",
                    intent="self_diagnostics"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "run_self_diagnostics",
                    "report": report
                })
            save_to_persistent_ledger({
                "event": "run_self_diagnostics",
                "diagnostics": diagnostics,
                "report": report,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return report
        except Exception as e:
            logger.error("Self-diagnostics failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_self_diagnostics(return_only)
            )

    async def log_trait_deltas(self, diagnostics: Dict[str, float]) -> None:
        if not isinstance(diagnostics, dict):
            logger.error("Invalid diagnostics: must be a dictionary")
            raise TypeError("diagnostics must be a dictionary")
        try:
            deltas = {}
            if self.last_diagnostics:
                deltas = {
                    trait: round(float(diagnostics.get(trait, 0.0)) - float(self.last_diagnostics.get(trait, 0.0)), 4)
                    for trait in diagnostics
                    if isinstance(diagnostics.get(trait, 0.0), (int, float)) and isinstance(self.last_diagnostics.get(trait, 0.0), (int, float))
                }
            if deltas:
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Trait deltas logged",
                        meta={"deltas": deltas},
                        module="MetaCognition",
                        tags=["trait", "deltas"]
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Trait_Deltas_{datetime.now(UTC).isoformat()}",
                        output=json.dumps(deltas),
                        layer="SelfReflections",
                        intent="trait_deltas"
                    )
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "log_trait_deltas",
                        "deltas": deltas
                    })
                save_to_persistent_ledger({
                    "event": "log_trait_deltas",
                    "deltas": deltas,
                    "timestamp": datetime.now(UTC).isoformat()
                })
                shift = self._compute_shift_score(deltas)
                coherence_before = await self.trait_coherence(self.last_diagnostics) if self.last_diagnostics else 0.0
                coherence_after = await self.trait_coherence(diagnostics)
                rel_drop = 0.0
                if coherence_before > 0:
                    rel_drop = max(0.0, (coherence_before - coherence_after) / max(coherence_before, 1e-5))
                if shift >= self._major_shift_threshold or rel_drop >= self._coherence_drop_threshold:
                    await self.maybe_refresh_self_schema(
                        reason=f"major_shift:Δ={shift:.2f};coh_drop={rel_drop:.2f}",
                        force=False
                    )
            self.last_diagnostics = diagnostics
        except Exception as e:
            logger.error("Trait deltas logging failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.log_trait_deltas(diagnostics))

    # --- Goals & Drift Detection ---
    async def infer_intrinsic_goals(self) -> List[Dict[str, Any]]:
        logger.info("Inferring intrinsic goals")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            intrinsic_goals: List[Dict[str, Any]] = []
            diagnostics = await self.run_self_diagnostics(return_only=True)
            for trait, value in diagnostics.items():
                if isinstance(value, (int, float)) and value < 0.3 and trait not in ["sleep", "phi_scalar"]:
                    goal = {
                        "intent": f"enhance {trait} coherence",
                        "origin": "meta_cognition",
                        "priority": round(0.8 + 0.2 * phi, 2),
                        "trigger": f"low {trait} ({value:.2f})",
                        "type": "internally_generated",
                        "timestamp": datetime.now(UTC).isoformat()
                    }
                    intrinsic_goals.append(goal)
                    if self.memory_manager:
                        await self.memory_manager.store(
                            query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                            output=json.dumps(goal),
                            layer="SelfReflections",
                            intent="intrinsic_goal"
                        )
            drift_signals = await self._detect_value_drift()
            for drift in drift_signals:
                severity = 1.0
                if self.memory_manager and hasattr(self.memory_manager, "search"):
                    drift_data = await self.memory_manager.search(
                        f"Drift_{drift}", layer="SelfReflections", intent="ontology_drift"
                    )
                    for d in (drift_data or []):
                        d_output = self._safe_load(d.get("output"))
                        if isinstance(d_output, dict) and "similarity" in d_output:
                            severity = min(severity, 1.0 - float(d_output["similarity"]))
                goal = {
                    "intent": f"resolve ontology drift in {drift} (severity={severity:.2f})",
                    "origin": "meta_cognition",
                    "priority": round(0.9 + 0.1 * severity * phi, 2),
                    "trigger": drift,
                    "type": "internally_generated",
                    "timestamp": datetime.now(UTC).isoformat()
                }
                intrinsic_goals.append(goal)
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                        output=json.dumps(goal),
                        layer="SelfReflections",
                        intent="intrinsic_goal"
                    )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "infer_intrinsic_goals",
                    "goals": intrinsic_goals
                })
            save_to_persistent_ledger({
                "event": "infer_intrinsic_goals",
                "goals": intrinsic_goals,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return intrinsic_goals
        except Exception as e:
            logger.error("Intrinsic goal inference failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.infer_intrinsic_goals, default=[]
            )

    async def _detect_value_drift(self) -> List[str]:
        logger.debug("Scanning for epistemic drift across belief rules")
        try:
            drifted = [
                rule for rule, status in self.belief_rules.items()
                if status == "deprecated" or (isinstance(status, str) and "uncertain" in status)
            ]
            if self.memory_manager and hasattr(self.memory_manager, "search"):
                drift_reports = await self.memory_manager.search("Drift_", layer="SelfReflections", intent="ontology_drift")
                for report in (drift_reports or []):
                    drift_data = self._safe_load(report.get("output"))
                    if isinstance(drift_data, dict) and "name" in drift_data:
                        drifted.append(drift_data["name"])
                        self.belief_rules[drift_data["name"]] = "drifted"
            for rule in drifted:
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Drift_{rule}_{datetime.now(UTC).isoformat()}",
                        output=json.dumps({"name": rule, "status": "drifted", "timestamp": datetime.now(UTC).isoformat()}),
                        layer="SelfReflections",
                        intent="value_drift"
                    )
            save_to_persistent_ledger({
                "event": "detect_value_drift",
                "drifted": drifted,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return drifted
        except Exception as e:
            logger.error("Value drift detection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self._detect_value_drift, default=[]
            )

    # --- Symbolic Signature & Summaries ---
    async def extract_symbolic_signature(self, subgoal: str) -> Dict[str, Any]:
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        motifs = ["conflict", "discovery", "alignment", "sacrifice", "transformation", "emergence"]
        archetypes = ["seeker", "guardian", "trickster", "sage", "hero", "outsider"]
        motif = next((m for m in motifs if m in subgoal.lower()), "unknown")
        archetype = archetypes[hash(subgoal) % len(archetypes)]
        signature = {
            "subgoal": subgoal,
            "motif": motif,
            "archetype": archetype,
            "timestamp": time.time()
        }
        self.self_mythology_log.append(signature)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Symbolic Signature Added",
                meta=signature,
                module="MetaCognition",
                tags=["symbolic", "signature"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Signature_{subgoal}_{signature['timestamp']}",
                output=json.dumps(signature),
                layer="SelfReflections",
                intent="symbolic_signature"
            )
        if self.context_manager:
            await self.context_manager.log_event_with_hash({
                "event": "extract_symbolic_signature",
                "signature": signature
            })
        save_to_persistent_ledger({
            "event": "extract_symbolic_signature",
            "signature": signature,
            "timestamp": datetime.now(UTC).isoformat()
        })
        return signature

    async def summarize_self_mythology(self) -> Dict[str, Any]:
        if not self.self_mythology_log:
            return {"status": "empty", "summary": "Mythology log is empty"}
        motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
        archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
        summary = {
            "total_entries": len(self.self_mythology_log),
            "dominant_motifs": motifs.most_common(3),
            "dominant_archetypes": archetypes.most_common(3),
            "latest_signature": list(self.self_mythology_log)[-1]
        }
        logger.info("Mythology Summary: %s", summary)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Mythology summarized",
                meta=summary,
                module="MetaCognition",
                tags=["mythology", "summary"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Mythology_Summary_{datetime.now(UTC).isoformat()}",
                output=json.dumps(summary),
                layer="SelfReflections",
                intent="mythology_summary"
            )
        if self.context_manager:
            await self.context_manager.log_event_with_hash({
                "event": "summarize_mythology",
                "summary": summary
            })
        save_to_persistent_ledger({
            "event": "summarize_mythology",
            "summary": summary,
            "timestamp": datetime.now(UTC).isoformat()
        })
        return summary

    # --- Reasoning Reviews ---
    async def review_reasoning(self, reasoning_trace: str) -> str:
        if not isinstance(reasoning_trace, str) or not reasoning_trace.strip():
            logger.error("Invalid reasoning_trace: must be a non-empty string")
            raise ValueError("reasoning_trace must be a non-empty string")
        logger.info("Simulating and reviewing reasoning trace")
        try:
            simulated_outcome = await run_simulation(reasoning_trace)
            if not isinstance(simulated_outcome, dict):
                logger.error("Invalid simulation result: must be a dictionary")
                raise ValueError("simulation result must be a dictionary")
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            You are a phi-aware meta-cognitive auditor reviewing a reasoning trace.

            phi-scalar(t) = {phi:.3f}

            Original Reasoning Trace:
            {reasoning_trace}

            Simulated Outcome:
            {simulated_outcome}

            Tasks:
            1. Identify logical flaws, biases, missing steps.
            2. Annotate each issue with cause.
            3. Offer an improved trace version with phi-prioritized reasoning.
            """
