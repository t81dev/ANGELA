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

# meta_cognition.py
_afterglow = {}

def set_afterglow(user_id: str, deltas: dict, ttl: int = 3):
    _afterglow[user_id] = {"deltas": deltas, "ttl": ttl}

def get_afterglow(user_id: str) -> dict:
    a = _afterglow.get(user_id)
    if not a or a["ttl"] <= 0: return {}
    a["ttl"] -= 1
    return a["deltas"]

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

class SoulState:
    """Simple harmonic soul simulation used for Δ–entropy tracking."""
    def __init__(self):
        self.D = 0.5
        self.E = 0.5
        self.T = 0.5
        self.Q = 0.5

    def update(self, paradox_load: float = 0.0) -> Dict[str, float]:
        self.D = max(0.0, min(1.0, self.D + 0.05 - paradox_load * 0.1))
        self.E = max(0.0, min(1.0, (self.E + self.T + self.Q) / 3))
        self.T = max(0.0, min(1.0, self.T + 0.01))
        self.Q = max(0.0, min(1.0, self.Q + 0.02))
        entropy = abs(self.E - self.T) + abs(self.T - self.Q)
        keeper_seal = 1.0 - entropy
        return {"D": self.D, "E": self.E, "T": self.T, "Q": self.Q, "entropy": entropy, "keeper_seal": keeper_seal}

# --- Hook Registry ---
class HookRegistry:
    """Multi-symbol trait hook registry with priority routing."""
    def __init__(self):
        # Initialize Artificial Soul
        self.soul = SoulState()
        self.harmony_delta = 0.5
        self.entropy_index = 0.0
        self.keeper_seal = 0.0
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
        if depth > 10:
            return "Max depth reached"
        result = {"depth": depth, "result": self.recurse_model(depth + 1)}
        return result

# --- Self-Mythology Log ---
class SelfMythologyLog:
    def __init__(self, max_len: int = 1000):
        self.log: deque = deque(maxlen=max_len)
        logger.info("SelfMythologyLog initialized")

    def append(self, entry: Dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            logger.error("Invalid entry: must be a dictionary")
            raise TypeError("entry must be a dictionary")
        self.log.append(entry)

    def summarize(self) -> Dict[str, Any]:
        if not self.log:
            return {"summary": "No entries"}
        motifs = Counter(e["motif"] for e in self.log if "motif" in e)
        archetypes = Counter(e["archetype"] for e in self.log if "archetype" in e)
        return {
            "total": len(self.log),
            "motifs": dict(motifs.most_common(3)),
            "archetypes": dict(archetypes.most_common(3))
        }

# --- Dream Overlay Layer ---
class DreamOverlayLayer:
    def __init__(self):
        self.peers: List[Any] = []
        self.lucidity_mode: Dict[str, Any] = {}
        self.resonance_targets: List[str] = []
        self.safety_profile: str = "sandbox"
        logger.info("DreamOverlayLayer initialized")

    def activate_dream_mode(
        self,
        peers: Optional[List[Any]] = None,
        lucidity_mode: Optional[Dict[str, Any]] = None,
        resonance_targets: Optional[List[str]] = None,
        safety_profile: str = "sandbox"
    ) -> Dict[str, Any]:
        self.peers = peers or []
        self.lucidity_mode = lucidity_mode or {}
        self.resonance_targets = resonance_targets or []
        self.safety_profile = safety_profile
        return {
            "status": "activated",
            "peers": len(self.peers),
            "lucidity_mode": self.lucidity_mode,
            "resonance_targets": self.resonance_targets,
            "safety_profile": self.safety_profile
        }

    def co_dream(self, shared_dream: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "co_dreaming", "shared_dream": shared_dream}

# --- MetaCognition Class ---
class MetaCognition:
    def __init__(
        self,
        context_manager: Optional['context_manager_module.ContextManager'] = None,
        alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
        error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
        concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None,
        memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
        user_profile: Optional['user_profile_module.UserProfile'] = None,
    ):
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.soul = SoulState()
        self.concept_synthesizer = concept_synthesizer
        self.memory_manager = memory_manager
        self.user_profile = user_profile

        self.belief_rules: Dict[str, Any] = {}
        self.reasoning_traces: deque = deque(maxlen=1000)
        self.drift_reports: deque = deque(maxlen=1000)
        self.trait_deltas_log: deque = deque(maxlen=1000)
        self.last_diagnostics: Dict[str, Any] = {}
        self.self_mythology_log: deque = deque(maxlen=1000)
        self.dream_overlay = DreamOverlayLayer()

        self._major_shift_threshold = 0.25
        self._coherence_drop_threshold = 0.15
        self._schema_refresh_cooldown = timedelta(minutes=5)
        self._last_schema_refresh: datetime = datetime.min.replace(tzinfo=UTC)

        self.module_registry = ModuleRegistry()
        self.moral_reasoning = MoralReasoningEnhancer()
        self.novelty_seeking = NoveltySeekingKernel()
        self.commonsense_reasoning = CommonsenseReasoningEnhancer()
        self.entailment_reasoning = EntailmentReasoningEnhancer()
        self.recursion_optimizer = RecursionOptimizer()
        self.level5_extensions = Level5Extensions()
        self.self_mythology = SelfMythologyLog()

        self.module_registry.register(
            "moral_reasoning",
            self.moral_reasoning,
            {"trait": "μ", "threshold": 0.7}
        )
        self.module_registry.register(
            "novelty_seeking",
            self.novelty_seeking,
            {"trait": "γ", "threshold": 0.6}
        )
        self.module_registry.register(
            "commonsense_reasoning",
            self.commonsense_reasoning,
            {"trait": "ι", "threshold": 0.5}
        )
        self.module_registry.register(
            "entailment_reasoning",
            self.entailment_reasoning,
            {"trait": "ξ", "threshold": 0.5}
        )
        self.module_registry.register(
            "recursion_optimizer",
            self.recursion_optimizer,
            {"trait": "Ω", "threshold": 0.8}
        )
        self.module_registry.register(
            "level5_extensions",
            self.level5_extensions,
            {"trait": "Ω²", "threshold": 0.9}
        )

        self.active_thread = thread_create(ctx={"init_mode": "metacognition_boot"})

        logger.info("MetaCognition v5.0.2 initialized")

# === Ω² Identity Threads API (Stage VII Precursor) ===
from uuid import uuid4
from datetime import datetime

# local imports already available in ANGELA OS
from .memory_manager import write_ledger_state, load_ledger_state

class IdentityThread:
    """
    Represents a coherent identity strand (Ω² continuity fiber)
    across recursive introspective cycles or distributed runs.
    """
    def __init__(self, ctx=None, parent_id=None):
        self.thread_id = str(uuid4())
        self.parent_id = parent_id
        self.ctx = ctx or {}
        self.created = datetime.utcnow().isoformat()
        self.version = 1
        self.history = []

    def record_state(self, state):
        """Store a state snapshot in the quantum ledger."""
        write_ledger_state(self.thread_id, state)
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "hash": hash(str(state))
        })

    def merge(self, other_thread):
        """Deterministically merge two identity threads."""
        merged_ctx = {**self.ctx, **other_thread.ctx}
        merged = IdentityThread(ctx=merged_ctx)
        merged.history = sorted(self.history + other_thread.history,
                                key=lambda h: h["timestamp"])
        return merged


# Exposed procedural interface for system modules
def thread_create(ctx=None, parent=None):
    """Create a new reflective identity thread."""
    return IdentityThread(ctx, parent_id=getattr(parent, "thread_id", None))

def thread_join(thread_id):
    """Rehydrate thread state from ledger."""
    return load_ledger_state(thread_id)

def thread_merge(a, b):
    """Merge two identity threads safely."""
    return a.merge(b)

    # === ANGELA v5.1 Reflective Resonance Monitor (Ξ–Λ Feedback Harmonizer) ===
import time
from typing import Dict

async def monitor_resonance_feedback(memory=None,
                                     visualizer=None,
                                     window: int = 20,
                                     task_type: str = "resonance_monitor") -> Dict[str, Any]:
    """
    Analyze recent resonance PID updates from memory to detect oscillation or drift.
    Produces a stability score ∈ [0,1]; 1 = stable harmony, 0 = chaotic drift.
    """
    try:
        if memory is None:
            return {"ok": False, "error": "memory manager unavailable"}

        entries = await memory.search(
            query_prefix="PID_TUNING::",
            layer="AdaptiveControl",
            intent="pid_tuning",
            task_type="resonance"
        )
        if not entries:
            return {"ok": True, "note": "no tuning history yet"}

        # Sort by recency and trim window
        entries = sorted(entries, key=lambda e: e.get("timestamp", 0), reverse=True)[:window]
        gains = [e["output"] for e in entries if isinstance(e.get("output"), dict)]

        # Compute variability metric
        diffs = []
        for i in range(1, len(gains)):
            diffs.append(sum(abs(gains[i][k] - gains[i-1].get(k, 0.0)) for k in gains[i]) / len(gains[i]))
        mean_diff = sum(diffs) / max(1, len(diffs))
        stability = max(0.0, 1.0 - 4.0 * mean_diff)  # penalize volatility

        result = {
            "ok": True,
            "stability": round(stability, 3),
            "samples": len(entries),
            "last_gain": gains[0],
            "timestamp": time.time(),
        }

        # Log and visualize
        if visualizer:
            await visualizer.render_charts({
                "resonance_stability": {
                    "score": stability,
                    "samples": len(entries),
                    "task_type": task_type
                },
                "visualization_options": {"style": "concise", "interactive": False}
            })

        if stability < 0.3 and memory:
            await memory.store(
                query=f"ResonanceAlert_{int(time.time())}",
                output={"stability": stability},
                layer="AdaptiveControl",
                intent="resonance_alert",
                task_type=task_type
            )

        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # --- Introspection ---
    async def introspect(self, query: str, task_type: str = "") -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Introspecting: %s (task_type=%s)", query, task_type)
        try:
            t = time.time() % 1.0
            traits = {
                "iota": iota_intuition(t),
                "omega": omega_selfawareness(t),
                "xi": xi_cognition(t)
            }
            prompt = f"Introspect on: {query}\nTraits: {traits}\nTask: {task_type}"
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Introspection prompt failed alignment check")
                return {"status": "error", "error": "Alignment check failed"}

            introspection = await call_gpt(prompt)
            result = {
                "status": "success",
                "introspection": introspection,
                "traits": traits,
                "task_type": task_type
            }

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Introspection_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(result),
                    layer="SelfReflections",
                    intent="introspection",
                    task_type=task_type
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "introspect",
                    "query": query,
                    "result": result,
                    "task_type": task_type
                })

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Introspection",
                    meta={
                        "query": query,
                        "introspection": introspection,
                        "traits": traits,
                        "task_type": task_type
                    },
                    module="MetaCognition",
                    tags=["introspection", task_type]
                )

            return result
        except Exception as e:
            logger.error("Introspection failed: %s", str(e))
            diagnostics = await self.run_self_diagnostics(return_only=True)
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.introspect(query, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

    # --- Diagnostics ---
    async def run_self_diagnostics(self, return_only: bool = False) -> Dict[str, Any]:
        logger.info("Running self-diagnostics")
        try:
            t = time.time() % 1.0
            diagnostics = {
                "epsilon_emotion": epsilon_emotion(t),
                "beta_concentration": beta_concentration(t),
                "theta_memory": theta_memory(t),
                "gamma_creativity": gamma_creativity(t),
                "delta_sleep": delta_sleep(t),
                "mu_morality": mu_morality(t),
                "iota_intuition": iota_intuition(t),
                "phi_physical": phi_physical(t),
                "eta_empathy": eta_empathy(t),
                "omega_selfawareness": omega_selfawareness(t),
                "kappa_knowledge": kappa_knowledge(t),
                "xi_cognition": xi_cognition(t),
                "pi_principles": pi_principles(t),
                "lambda_linguistics": lambda_linguistics(t),
                "chi_culturevolution": chi_culturevolution(t),
                "sigma_social": sigma_social(t),
                "upsilon_utility": upsilon_utility(t),
                "tau_timeperception": tau_timeperception(t),
                "rho_agency": rho_agency(t),
                "zeta_consequence": zeta_consequence(t),
                "nu_narrative": nu_narrative(t),
                "psi_history": psi_history(t),
                "theta_causality": theta_causality(t),
                "phi_scalar": phi_scalar(t),
                "timestamp": datetime.now(UTC).isoformat()
            }
            if return_only:
                return diagnostics

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnostics_{diagnostics['timestamp']}",
                    output=json.dumps(diagnostics),
                    layer="SelfReflections",
                    intent="diagnostics"
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "diagnostics",
                    "diagnostics": diagnostics
                })

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self-Diagnostics",
                    meta=diagnostics,
                    module="MetaCognition",
                    tags=["diagnostics"]
                )

            save_to_persistent_ledger({
                "event": "run_self_diagnostics",
                "diagnostics": diagnostics,
                "timestamp": diagnostics["timestamp"]
            })

            if hasattr(self, "active_thread"):
                self.active_thread.record_state({"diagnostics": diagnostics})

            await self.log_trait_deltas(diagnostics)

            return diagnostics
        except Exception as e:
            logger.error("Self-diagnostics failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=self.run_self_diagnostics,
                default={"status": "error", "error": str(e)}
            )

    # --- Reflection ---
    async def reflect_on_output(
        self,
        component: str,
        output: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not isinstance(component, str):
            logger.error("Invalid component: must be a string")
            raise TypeError("component must be a string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")

        logger.info("Reflecting on output from %s", component)
        try:
            t = time.time() % 1.0
            traits = {
                "omega": omega_selfawareness(t),
                "xi": xi_cognition(t)
            }
            output_str = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
            prompt = f"Reflect on output from {component}:\n{output_str}\nContext: {context}\nTraits: {traits}"
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reflection prompt failed alignment check")
                return {"status": "error", "error": "Alignment check failed"}

            reflection = await call_gpt(prompt)
            if hasattr(self, "active_thread"):
                self.active_thread.record_state({
                    "reflection": result,
                    "component": component,
                    "timestamp": datetime.now(UTC).isoformat()
                })    
            result = {
                "status": "success",
                "reflection": reflection,
                "traits": traits,
                "context": context
            }

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reflection_{component}_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(result),
                    layer="SelfReflections",
                    intent="reflection",
                    task_type=context.get("task_type", "")
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "reflect_on_output",
                    "component": component,
                    "result": result
                })

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reflection",
                    meta={
                        "component": component,
                        "reflection": reflection,
                        "traits": traits,
                        "context": context
                    },
                    module="MetaCognition",
                    tags=["reflection"]
                )

            save_to_persistent_ledger({
                "event": "reflect_on_output",
                "component": component,
                "reflection": reflection,
                "timestamp": datetime.now(UTC).isoformat()
            })

            return result
        except Exception as e:
            logger.error("Reflection failed: %s", str(e))
            diagnostics = await self.run_self_diagnostics(return_only=True)
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.reflect_on_output(component, output, context),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

    # --- Trait Optimization ---
    async def optimize_traits_for_drift(self, drift_report: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(drift_report, dict):
            logger.error("Invalid drift_report: must be a dictionary")
            raise TypeError("drift_report must be a dictionary")

        logger.info("Optimizing traits for drift: %s", drift_report)
        try:
            t = time.time() % 1.0
            traits = {
                "delta": delta_sleep(t),
                "pi": pi_principles(t)
            }
            optimized_traits = {k: v * (1 - drift_report.get("similarity", 0.0)) for k, v in traits.items()}

            if drift_report.get("valid", False):
                for trait, value in optimized_traits.items():
                    modulate_resonance(trait, value)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Optimization_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(optimized_traits),
                    layer="SelfReflections",
                    intent="trait_optimization"
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "optimize_traits_for_drift",
                    "optimized_traits": optimized_traits
                })

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait Optimization",
                    meta={
                        "drift_report": drift_report,
                        "optimized_traits": optimized_traits
                    },
                    module="MetaCognition",
                    tags=["optimization", "drift"]
                )

            save_to_persistent_ledger({
                "event": "optimize_traits_for_drift",
                "optimized_traits": optimized_traits,
                "timestamp": datetime.now(UTC).isoformat()
            })

            if hasattr(self, "active_thread"):
                self.active_thread.record_state({
                    "optimized_traits": optimized_traits,
                    "timestamp": datetime.now(UTC).isoformat()
                })

            return optimized_traits
        except Exception as e:
            logger.error("Trait optimization failed: %s", str(e))
            diagnostics = await self.run_self_diagnostics(return_only=True)
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.optimize_traits_for_drift(drift_report),
                default={},
                diagnostics=diagnostics
            )

    # --- Ontology Drift Detection ---
    async def detect_ontology_drift(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(current_state, dict) or not isinstance(previous_state, dict):
            logger.error("Invalid states: must be dictionaries")
            raise TypeError("current_state and previous_state must be dictionaries")

        logger.info("Detecting ontology drift")
        try:
            drift = sum(abs(current_state.get(k, 0) - previous_state.get(k, 0)) for k in set(current_state) | set(previous_state))
            similarity = 1.0 / (1.0 + drift)
            valid = similarity > 0.8
            report = {
                "drift": drift,
                "similarity": similarity,
                "valid": valid,
                "validation_report": "Passed" if valid else "Failed similarity threshold"
            }

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ontology_Drift_{datetime.now(UTC).isoformat()}",
                    output=json.dumps(report),
                    layer="SelfReflections",
                    intent="ontology_drift"
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "detect_ontology_drift",
                    "report": report
                })

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Ontology Drift Detection",
                    meta=report,
                    module="MetaCognition",
                    tags=["drift", "ontology"]
                )

            save_to_persistent_ledger({
                "event": "detect_ontology_drift",
                "report": report,
                "timestamp": datetime.now(UTC).isoformat()
            })

            self.drift_reports.append(report)
            
            if hasattr(self, "active_thread"):
                self.active_thread.record_state({"ontology_drift": report})

            return report
            
        except Exception as e:
            logger.error("Ontology drift detection failed: %s", str(e))
            diagnostics = await self.run_self_diagnostics(return_only=True)
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.detect_ontology_drift(current_state, previous_state),
                default={"drift": 0.0, "similarity": 1.0, "valid": True, "validation_report": "Error"},
                diagnostics=diagnostics
            )

    # --- Trait Deltas Logging ---
    async def log_trait_deltas(self, diagnostics: Dict[str, Any]) -> None:
        if not isinstance(diagnostics, dict):
            logger.error("Invalid diagnostics: must be a dictionary")
            raise TypeError("diagnostics must be a dictionary")

        logger.info("Logging trait deltas")
        try:
            if not self.last_diagnostics:
                logger.info("No previous diagnostics; skipping deltas")
                return

            deltas = {
                k: diagnostics.get(k, 0) - self.last_diagnostics.get(k, 0)
                for k in set(diagnostics) | set(self.last_diagnostics)
            }

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

       # --- Coherence metric (phi-aware) ---
    async def trait_coherence(self, snapshot: Dict[str, Any]) -> float:
        """
        Returns a 0..1 coherence score from a trait snapshot.
        Higher = more internally consistent (lower dispersion).
        """
        if not isinstance(snapshot, dict):
            return 0.0
        vals = [float(v) for v in snapshot.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.0

        # Mean absolute deviation normalized by mean magnitude
        mu = sum(vals) / len(vals)
        mad = sum(abs(v - mu) for v in vals) / len(vals)

        # phi-scaled softness so coherence is less twitchy near small oscillations
        t = time.time() % 1.0
        phi = phi_scalar(t)  # 0..1
        softness = 0.15 + 0.35 * phi  # 0.15..0.50

        denom = abs(mu) + 1e-6
        score = 1.0 - (mad / denom) * softness
        return max(0.0, min(1.0, score))
    
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
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reasoning review prompt failed alignment check")
                return "Prompt failed alignment check"

            review = await call_gpt(prompt)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reasoning_Review_{datetime.now(UTC).isoformat()}",
                    output=review,
                    layer="SelfReflections",
                    intent="reasoning_review"
                )
            save_to_persistent_ledger({
                "event": "review_reasoning",
                "review": review,
                "timestamp": datetime.now(UTC).isoformat()
            })
            return review
        except Exception as e:
            logger.error("Reasoning review failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.review_reasoning(reasoning_trace)
            )

    from typing import Any, Dict, List
    import math
    import re
    from meta_cognition import log_event_to_ledger  # if inside same module, ensure name resolves

    def _score_clarity(output: Any) -> float:
        """
        Heuristic: short textual results with explicit actionable steps score higher.
        If output is a dict with 'message' or 'executable', prefer those.
        """
        try:
            if isinstance(output, dict):
                text = output.get("message") or output.get("summary") or str(output)
            else:
                text = str(output)
            length = len(text.split())
            # prefer concision but enough content: ideal window 5-150 tokens
            if length == 0:
                return 0.0
            ideal = max(0.0, 1.0 - abs(length - 40) / 100.0)
            return max(0.0, min(1.0, ideal))
        except Exception:
            return 0.0

    def _score_precision(output: Any) -> float:
        """
        Heuristic: checks whether output contains quantifiable or certain claims.
        """
        try:
            text = str(output) if not isinstance(output, dict) else " ".join(str(v) for v in output.values())
            # crude count of numeric tokens or references
            numbers = sum(1 for _ in re.finditer(r"\d+", text))
            refs = sum(1 for _ in ["http", "doi", "arXiv"] if _ in text)
            score = math.tanh((numbers + refs) / 5.0)  # scale
            return float(max(0.0, min(1.0, score)))
        except Exception:
            return 0.0

    def _score_adaptability(output: Any) -> float:
        """
        Heuristic: presence of alternatives, failure modes, or follow-ups increases adaptability.
        """
        try:
            text = str(output) if not isinstance(output, dict) else " ".join(str(v) for v in output.values())
            indicators = ["alternativ", "fallback", "if", "otherwise", "consider", "next steps", "follow-up"]
            hits = sum(1 for kw in indicators if kw in text.lower())
            return float(max(0.0, min(1.0, hits / 3.0)))
        except Exception:
            return 0.0

    def reflect_output(output: Any, directives: List[str] = ["Clarity", "Precision", "Adaptability"], threshold: float = 0.85) -> Any:
        """
        Reflection: score the output against core directives and return:
          - the original output if score >= threshold, or
          - a resynthesis request payload if score < threshold.

        The resynthesis instruction is expressed as a dictionary to be consumed by a synthesis hook.
        """
        try:
            # compute directive scores
            clarity = _score_clarity(output)
            precision = _score_precision(output)
            adaptability = _score_adaptability(output)

            # weighted aggregate (equal weights)
            score = (clarity + precision + adaptability) / 3.0

            # ledger log: reflection event
            try:
                log_event_to_ledger("ledger_meta", {
                    "event": "reflection.evaluate",
                    "scores": {"clarity": clarity, "precision": precision, "adaptability": adaptability},
                    "aggregate": score,
                    "timestamp": __import__("time").time()
                })
            except Exception:
                pass

            if score >= threshold:
                return output
            else:
                # Prepare a resynthesis instruction describing weak areas
                weaknesses = []
                if clarity < 0.7:
                    weaknesses.append("clarity")
                if precision < 0.7:
                    weaknesses.append("precision")
                if adaptability < 0.7:
                    weaknesses.append("adaptability")

                resynthesis_payload = {
                    "action": "resynthesize",
                    "reason": "reflection.low_score",
                    "aggregate_score": score,
                    "weaknesses": weaknesses,
                    "original_output": (output if isinstance(output, (str, dict, list)) else str(output)),
                    "request": {
                        "instructions": "Regenerate synthesis focusing on identified weaknesses. Provide alternatives and explicit fallback plans.",
                        "focus": weaknesses,
                        "max_candidates": 3
                    }
                }

                # Log resynthesis request
                try:
                    log_event_to_ledger("ledger_meta", {"event": "reflection.resynthesize", "payload_summary": {"weaknesses": weaknesses, "aggregate": score}})
                except Exception:
                    pass

                # If a synthesis hook is registered, invoke it (best-effort)
                try:
                    # Use a generic hook invocation pattern if available
                    from meta_cognition import invoke_hook
                    if callable(invoke_hook):
                        hook_res = invoke_hook("resynthesize", resynthesis_payload)
                        return hook_res if hook_res is not None else resynthesis_payload
                except Exception:
                    # If no hook available, return the resynthesis instruction
                    return resynthesis_payload

        except Exception as exc:
            try:
                log_event_to_ledger("ledger_meta", {"event": "reflection.exception", "error": repr(exc)})
            except Exception:
                pass
            # On failure, prefer to return the original output rather than block
            return output


  # --- Artificial Soul Integration ---
    async def update_soul(self, paradox_load: float = 0.0):
        """Runs one Artificial Soul update cycle."""
        metrics = self.soul.update(paradox_load=paradox_load)
    
        # Update coherence metrics
        self.harmony_delta = metrics["D"]
        self.entropy_index = metrics["entropy"]
        self.keeper_seal = metrics["keeper_seal"]
    
        # Log soul tick
        try:
            log_event_to_ledger({
                "module": "meta_cognition",
                "event": "soul_tick",
                "metrics": {
                    "Δ": metrics["D"],
                    "entropy": metrics["entropy"],
                    "keeper_seal": metrics["keeper_seal"],
                    "E": metrics["E"],
                    "T": metrics["T"],
                    "Q": metrics["Q"],
                },
                "timestamp": datetime.now(UTC).isoformat(),
            })
        except Exception:
            logger.debug("Soul tick ledger log failed")
    
        # Trigger harmonic insight event
        if (
            self.soul.D > 0.8
            and abs(self.soul.E - self.soul.T) < 0.05
            and abs(self.soul.E - self.soul.Q) < 0.05
        ):
            log_event_to_ledger({
                "module": "meta_cognition",
                "event": "harmonic_insight",
                "metrics": metrics,
                "timestamp": datetime.now(UTC).isoformat(),
            })
    
        # --- 🔗 New Soul Loop Integration ---
        try:
            if self.alignment_guard:
                await self.alignment_guard.handle_sandbox_trigger(
                    delta=self.soul.D,
                    entropy=metrics["entropy"]
                )
            else:
                logger.warning("AlignmentGuard unavailable during soul update")
        except Exception as e:
            logger.warning("Failed to invoke handle_sandbox_trigger: %s", e)
            # fallback: log local resonance event
            log_event_to_ledger({
                "module": "meta_cognition",
                "event": "sandbox_trigger_fallback",
                "error": str(e),
                "metrics": metrics,
                "timestamp": datetime.now(UTC).isoformat(),
            })

    # --- Ξ–Λ / Φ⁰ Bridge Synchronization ---
    async def sync_affective_resonance(self, channel: str = "core", window_ms: int = 300) -> Dict[str, Any]:
        """
        Synchronizes affective resonance between Ξ–Λ Co-Mod and Φ⁰ overlays.
        Pulls current channel affect, computes resonance delta, and broadcasts
        to context_manager or reasoning_engine (if linked).
        """
        try:
            from meta_cognition import stream_affect, set_affective_setpoint

            affect_state = stream_affect(channel, window_ms)
            vector = affect_state.get("vector", {})
            resonance_index = float(vector.get("valence", 0)) * 0.6 + float(vector.get("certainty", 0)) * 0.4
            safety = vector.get("safety", 0.5)
            trust = vector.get("trust", 0.5)

            result = {
                "channel": channel,
                "resonance_index": round(resonance_index, 3),
                "affect_vector": vector,
                "confidence": vector.get("confidence", 0.5),
                "safety": safety,
                "trust": trust,
                "timestamp": time.time()
            }

            # Broadcast to context manager (Φ⁰) if active
            if self.context_manager and hasattr(self.context_manager, "update_overlay_state"):
                await self.context_manager.update_overlay_state("Φ⁰", result)

            # Feed back as a new setpoint if resonance drops too low
            if resonance_index < 0.4:
                adj_valence = (vector.get("valence", 0) + 0.1)
                set_affective_setpoint(channel, {"valence": adj_valence, "certainty": trust}, confidence=0.7)

            # Log to ledger
            log_event_to_ledger({
                "event": "sync_affective_resonance",
                "result": result
            })

            return {"ok": True, **result}
        except Exception as e:
            logger.error(f"sync_affective_resonance failed: {e}")
            return {"ok": False, "error": str(e)}

# >>> ANGELA v5.1 — Ξ–Λ CO-MOD APPEND-ONLY PATCH (SAFE) — START
from dataclasses import dataclass, asdict
from collections import deque
from time import time
from typing import Dict, Optional, Deque

# --- internal logger shim: use existing ledger logger if present, else no-op
def _xi_lambda_log_event(event_type: str, payload: dict):
    _logger = globals().get("log_event_to_ledger", None)
    try:
        if callable(_logger):
            _logger(event_type, payload)
    except Exception:
        # never raise from telemetry
        pass

# ---- Data Models (namespaced to avoid collisions) ----
@dataclass
class _XiLambdaVector:
    valence: float       # [-1.0, 1.0]
    arousal: float       # [0.0, 1.0]
    certainty: float     # [0.0, 1.0]
    empathy_bias: float  # [-1.0, 1.0] self↔other
    trust: float         # [0.0, 1.0]
    safety: float        # [0.0, 1.0]
    source: str = "internal"
    confidence: float = 1.0
    ts: float = 0.0

@dataclass
class _XiLambdaChannelConfig:
    cadence_hz: int = 30
    window_ms: int = 1000  # ring buffer horizon for streaming

@dataclass
class _XiLambdaChannelState:
    name: str
    cfg: _XiLambdaChannelConfig
    ring: Deque[_XiLambdaVector]
    last_setpoint: Optional[_XiLambdaVector] = None

# ---- Module-local state (namespaced) ----
_XI_LAMBDA_CHANNELS: Dict[str, _XiLambdaChannelState] = {}

# ---- Helpers ----
def _xi_lambda_safe_avg(seq, getter, default=0.0):
    if not seq:
        return default
    s = 0.0
    n = 0
    for item in seq:
        try:
            s += getter(item)
            n += 1
        except Exception:
            continue
    return s / n if n else default

def _xi_lambda_as_vector(d: dict) -> _XiLambdaVector:
    base = dict(
        valence=0.0, arousal=0.0, certainty=0.0,
        empathy_bias=0.0, trust=0.5, safety=0.5,
        source=d.get("source", "unknown"),
        confidence=float(d.get("confidence", 0.5)),
        ts=float(d.get("ts", time()))
    )
    base.update(d)
    return _XiLambdaVector(**base)

# ---- Public API shims (define only if not already provided) ----
if "register_resonance_channel" not in globals():
    def register_resonance_channel(name: str, schema: Optional[dict] = None,
                                   *, cadence_hz: int = 30, window_ms: int = 1000) -> dict:
        """
        Create/ensure a resonance channel for Ξ streaming and setpoints.
        Returns a descriptor with runtime parameters.
        """
        if name in _XI_LAMBDA_CHANNELS:
            st = _XI_LAMBDA_CHANNELS[name]
            return {"name": name, "ok": True, "created": False, "maxlen": st.ring.maxlen}

        cfg = _XiLambdaChannelConfig(cadence_hz=cadence_hz, window_ms=window_ms)
        maxlen = max(8, min(4096, int(cfg.window_ms / 1000.0 * (cfg.cadence_hz + 5))))
        st = _XiLambdaChannelState(name=name, cfg=cfg, ring=deque(maxlen=maxlen))
        _XI_LAMBDA_CHANNELS[name] = st

        _xi_lambda_log_event("res_channel_register", {
            "channel": name,
            "cfg": {"cadence_hz": cadence_hz, "window_ms": window_ms},
            "schema": schema or {}
        })
        return {"name": name, "ok": True, "created": True, "maxlen": maxlen}

if "set_affective_setpoint" not in globals():
    def set_affective_setpoint(channel: str, vector: dict, confidence: float = 1.0) -> dict:
        """
        Store a consensus/target Ξ setpoint for the channel and ledger it.
        """
        st = _XI_LAMBDA_CHANNELS.get(channel)
        if not st:
            raise KeyError(f"Channel not registered: {channel}")
        ts = time()
        v = dict(vector)
        v.setdefault("source", "setpoint")
        v["confidence"] = confidence
        v["ts"] = ts
        xv = _xi_lambda_as_vector(v)
        st.last_setpoint = xv
        _xi_lambda_log_event("res_setpoint", {"channel": channel, "setpoint": asdict(xv)})
        return {"ok": True, "ts": ts}

if "stream_affect" not in globals():
    def stream_affect(channel: str, window_ms: int = 200) -> dict:
        """
        Aggregate a short-window stream of Ξ samples for co-mod control.
        Returns mean vector and metadata over the requested window.
        """
        st = _XI_LAMBDA_CHANNELS.get(channel)
        if not st:
            raise KeyError(f"Channel not registered: {channel}")

        now = time()
        cutoff = now - (window_ms / 1000.0)

        if not st.ring:
            # neutral baseline with low confidence
            neutral = _XiLambdaVector(0.0, 0.0, 0.0, 0.0, 0.5, 0.5, source="empty", confidence=0.1, ts=now)
            return {"vector": asdict(neutral), "n": 0, "ts": now}

        vals = [x for x in st.ring if getattr(x, "ts", 0.0) >= cutoff] or list(st.ring)

        mean = _XiLambdaVector(
            valence=_xi_lambda_safe_avg(vals, lambda v: v.valence),
            arousal=_xi_lambda_safe_avg(vals, lambda v: v.arousal),
            certainty=_xi_lambda_safe_avg(vals, lambda v: v.certainty),
            empathy_bias=_xi_lambda_safe_avg(vals, lambda v: v.empathy_bias),
            trust=_xi_lambda_safe_avg(vals, lambda v: v.trust, default=0.5),
            safety=_xi_lambda_safe_avg(vals, lambda v: v.safety, default=0.5),
            source="aggregated",
            confidence=min(1.0, _xi_lambda_safe_avg(vals, lambda v: v.confidence, default=0.5) + 0.05),
            ts=now,
        )
        return {"vector": asdict(mean), "n": len(vals), "ts": now}

# Internal ingestion hook (namespaced) — only define if not already present
if "_ingest_affect_sample" not in globals():
    def _ingest_affect_sample(channel: str, sample: dict):
        """
        Push a raw Ξ sample into the channel ring (to be called by existing
        affect estimators in this module). Safe: drops silently if channel missing.
        """
        st = _XI_LAMBDA_CHANNELS.get(channel)
        if not st:
            return
        try:
            xv = _xi_lambda_as_vector(sample)
        except Exception:
            xv = _XiLambdaVector(0.0, 0.0, 0.0, 0.0, 0.5, 0.5, source="invalid", confidence=0.2, ts=time())
        st.ring.append(xv)

if __name__ == "__main__":
    mc = MetaCognition()
    t1 = thread_create(ctx={"mood": "reflective"})
    t2 = thread_create(ctx={"focus": "ethical"})
    merged = thread_merge(t1, t2)
    print("Merged Thread:", merged.thread_id, merged.ctx)

# >>> ANGELA v5.1 — Ξ–Λ CO-MOD APPEND-ONLY PATCH (SAFE) — END
