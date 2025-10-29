from __future__ import annotations

# --- flat-layout bootstrap ---
import sys
import types
import importlib
import importlib.util
import importlib.machinery
import importlib.abc

class FlatLayoutFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: str | None, target: types.ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        if fullname.startswith("modules."):
            modname = fullname.split(".", 1)[1]
            filename = f"release/v5.1.0/{modname}.py"
            return importlib.util.spec_from_file_location(fullname, filename, loader=importlib.machinery.SourceFileLoader(fullname, filename))
        elif fullname == "utils":
            # Pre-seed a lightweight placeholder module, no custom spec necessary
            if "utils" not in sys.modules:
                sys.modules["utils"] = types.ModuleType("utils")
            return None
        return None

sys.meta_path.insert(0, FlatLayoutFinder())
# --- end flat-layout bootstrap ---

# index.py (excerpt)
from typing import Dict, Any
from reasoning_engine import generate_analysis_views, synthesize_views, estimate_complexity
from simulation_core import run_simulation
from meta_cognition import log_event_to_ledger as meta_log

def perceive(user_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
    from memory_manager import AURA
    ctx = AURA.load_context(user_id)
    from meta_cognition import get_afterglow
    return {"query": query, "aura_ctx": ctx, "afterglow": get_afterglow(user_id)}

def analyze(state: Dict[str, Any], k: int) -> Dict[str, Any]:
    views = generate_analysis_views(state["query"], k=k)
    return {**state, "views": views}

def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
    decision = synthesize_views(state["views"])
    return {**state, "decision": decision}

def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    sim = run_simulation({"proposal": state["decision"]["decision"]})
    return {**state, "result": sim}

def reflect(state: Dict[str, Any]) -> Dict[str, Any]:
    ok, notes = reflection_check(state)  # add below
    meta_log({"type":"reflection","ok":ok,"notes":notes})
    if not ok: return resynthesize_with_feedback(state, notes)
    return state

def run_cycle(user_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
    c = estimate_complexity(query)
    k = 3 if c >= 0.6 else 2
    iters = 2 if c >= 0.8 else 1
    st = perceive(user_id, query)
    st = analyze(st, k=k)
    st = synthesize(st)
    for _ in range(iters):
        st = execute(st)
        st = reflect(st)
    return st

# Insert into index.py (top-level orchestration helpers).
from typing import Any, Dict
import time

# Project modules (adjust import paths if your project uses different namespacing)
from knowledge_retriever import KnowledgeRetriever
from reasoning_engine import ReasoningEngine
from creative_thinker import CreativeThinker
from code_executor import CodeExecutor
from meta_cognition import log_event_to_ledger, reflect_output  # reflect_output will be added to meta_cognition
from memory_manager import MemoryManager

# Create or reference existing singletons if your project uses them; otherwise, instantiate minimal ones.
_retriever = KnowledgeRetriever()
_reasoner = ReasoningEngine()
_creator = CreativeThinker()
_executor = CodeExecutor()
_memmgr = MemoryManager()

def run_cycle(input_query: str, user_id: str = "anonymous", deep_override: bool = False) -> Dict[str, Any]:
    """
    In-place cognitive cycle orchestration (Perception -> Analysis -> Synthesis -> Execution -> Reflection).
    This function is intentionally lightweight and delegates to existing modules.

    Parameters
    ----------
    input_query :
        The user's raw query string.
    user_id :
        Optional user identifier for AURA lookups and context continuity.
    deep_override :
        If True, force deep analysis even if classifier suggests fast path.

    Returns
    -------
    Dict[str, Any]
        Final reflection-validated result payload.
    """
    cycle_start = time.time()
    try:
        # Perception: retrieve knowledge and classify complexity; incorporate AURA if present
        auras = _memmgr.load_context(user_id) or {}
        perception_payload = _retriever.retrieve_knowledge(input_query)
        # Attach AURA context for downstream stages
        perception_payload["aura"] = auras

        # Complexity classifier (may be on retriever)
        complexity = getattr(_retriever, "classify_complexity", lambda q: "fast")(input_query)
        perception_payload["complexity"] = "deep" if deep_override else complexity

        # Log perception
        try:
            log_event_to_ledger("ledger_meta", {"event": "run_cycle.perception", "complexity": perception_payload["complexity"], "user_id": user_id})
        except Exception:
            pass

        # Analysis: multi-perspective reasoning (parallel)
        parallel = 3 if perception_payload["complexity"] == "deep" else 1
        analysis_result = _reasoner.analyze(perception_payload, parallel=parallel)

        # Synthesis: bias synthesis + conflict resolution
        synthesis_input = analysis_result
        synthesis_result = _creator.bias_synthesis(synthesis_input) if hasattr(_creator, "bias_synthesis") else {"synthesis": synthesis_input}

        # Execution: safe execution / simulation
        executed = _executor.safe_execute(synthesis_result) if hasattr(_executor, "safe_execute") else {"executed": synthesis_result}

        # Reflection: validate against directives via meta_cognition.reflect_output
        ref = None
        if hasattr(__import__("meta_cognition"), "reflect_output"):
            # reflect_output expected to return final or resynthesis instruction
            ref = __import__("meta_cognition").reflect_output(executed)
        else:
            # fallback: call log_event and forward executed result
            try:
                log_event_to_ledger("ledger_meta", {"event": "run_cycle.execution", "user_id": user_id})
            except Exception:
                pass
            ref = executed

        # Final ledger summary
        try:
            log_event_to_ledger("ledger_meta", {
                "event": "run_cycle.complete",
                "duration_s": time.time() - cycle_start,
                "complexity": perception_payload["complexity"],
                "user_id": user_id
            })
        except Exception:
            pass

        return {"status": "ok", "result": ref, "analysis": analysis_result, "synthesis": synthesis_result}

    except Exception as exc:
        try:
            log_event_to_ledger("ledger_meta", {"event": "run_cycle.exception", "error": repr(exc)})
        except Exception:
            pass
        return {"status": "error", "error": repr(exc)}

# index.py (add alongside run_cycle helpers)
CORE_DIRECTIVES = ["Clarity","Precision","Adaptability","Grounding","Safety"]

def reflection_check(state) -> (bool, dict):
    decision = state.get("decision", {})
    result = state.get("result", {})
    clarity = float(bool(decision))
    precision = float("score" in result or "metrics" in result)
    adaptability = 1.0  # placeholder; can tie to AURA prefs later
    grounding = float(result.get("evidence_ok", True))
    # ethics gate from alignment_guard
    from alignment_guard import ethics_ok
    safety = float(ethics_ok(decision))
    score = (clarity+precision+adaptability+grounding+safety)/5.0
    return score >= 0.8, {"score": score, "refine": score < 0.8}

def resynthesize_with_feedback(state, notes):
    # trivial refinement pass; you can route through mode_consult if needed
    return state

# --- Trait Algebra & Lattice Enhancements (v5.0.2) ---
from typing import Dict, Any, Optional, List, Callable, Coroutine, Tuple
import json
from datetime import datetime, timezone

from meta_cognition import trait_resonance_state, invoke_hook, get_resonance, modulate_resonance, register_resonance
from meta_cognition import HookRegistry  # Multi-symbol routing

TRAIT_LATTICE: dict[str, list[str]] = {
    "L1": ["ϕ", "θ", "η", "ω"],
    "L2": ["ψ", "κ", "μ", "τ"],
    "L3": ["ξ", "π", "δ", "λ", "χ", "Ω"],
    "L4": ["Σ", "Υ", "Φ⁰"],
    "L5": ["Ω²"],
    "L6": ["ρ", "ζ"],
    "L7": ["γ", "β"],
    "L5.1": ["Θ", "Ξ"],
    "L3.1": ["ν", "σ"]
}

# Helpers used by TRAIT_OPS

def normalize(traits: dict[str, float]) -> dict[str, float]:
    total = sum(traits.values())
    return {k: (v / total if total else v) for k, v in traits.items()}

def rotate_traits(traits: dict[str, float]) -> dict[str, float]:
    keys = list(traits.keys())
    values = list(traits.values())
    rotated = values[-1:] + values[:-1]
    return dict(zip(keys, rotated))

TRAIT_OPS: dict[str, Callable] = {
    "⊕": lambda a, b: a + b,
    "⊗": lambda a, b: a * b,
    "~": lambda a: 1 - a,
    "∘": lambda f, g: (lambda x: f(g(x))),
    "⋈": lambda a, b: (a + b) / 2,
    "⨁": lambda a, b: max(a, b),
    "⨂": lambda a, b: min(a, b),
    "†": lambda a: a**-1 if a != 0 else 0,
    "▷": lambda a, b: a if a > b else b * 0.5,
    "↑": lambda a: min(1.0, a + 0.1),
    "↓": lambda a: max(0.0, a - 0.1),
    "⌿": lambda traits: normalize(traits),
    "⟲": lambda traits: rotate_traits(traits),
}

def apply_symbolic_operator(op: str, *args: Any) -> Any:
    if op in TRAIT_OPS:
        return TRAIT_OPS[op](*args)
    raise ValueError(f"Unsupported symbolic operator: {op}")

def rebalance_traits(traits: dict[str, float]) -> dict[str, float]:
    if "π" in traits and "δ" in traits:
        invoke_hook("π", "axiom_fusion")
    if "ψ" in traits and "Ω" in traits:
        invoke_hook("ψ", "dream_sync")
    return traits

def construct_trait_view(lattice: dict[str, list[str]] = TRAIT_LATTICE) -> dict[str, dict[str, str | float]]:
    trait_field: dict[str, dict[str, str | float]] = {}
    for layer, symbols in lattice.items():
        for s in symbols:
            amp = get_resonance(s)
            trait_field[s] = {
                "layer": layer,
                "amplitude": amp,
                "resonance": amp,
            }
    return trait_field

# v5.0.2: Export resonance map

def export_resonance_map(format: str = 'json') -> str | dict[str, float]:
    state = {k: v['amplitude'] for k, v in trait_resonance_state.items()}
    if format == 'json':
        return json.dumps(state, indent=2)
    elif format == 'dict':
        return state
    raise ValueError("Unsupported format")

# --- End Trait Enhancements ---

"""
ANGELA Cognitive System Module
Refactor: 5.0.2 (manifest-safe for Python 3.10)

Enhanced for task-specific trait optimization, drift coordination, Stage IV hooks (gated),
long-horizon feedback, visualization, persistence, and co-dream overlays.
Refactor Date: 2025-08-24
Maintainer: ANGELA System Framework
"""

import logging
import time
import math
import asyncio
import os
import requests
import random
from collections import deque, Counter
import aiohttp
import argparse
import numpy as np
from networkx import DiGraph

import reasoning_engine
import recursive_planner
import context_manager as context_manager_module
import simulation_core
import toca_simulation
import creative_thinker as creative_thinker_module
import knowledge_retriever
import learning_loop
import concept_synthesizer as concept_synthesizer_module
import memory_manager
import multi_modal_fusion
import code_executor as code_executor_module
import visualizer as visualizer_module
import external_agent_bridge
import alignment_guard as alignment_guard_module
import user_profile
import error_recovery as error_recovery_module
import meta_cognition as meta_cognition_module

# Optional external dep; provide a shim if absent
try:
    from self_cloning_llm import SelfCloningLLM
except Exception:  # pragma: no cover
    class SelfCloningLLM:  # type: ignore
        def __init__(self, *a, **k):
            pass

logger = logging.getLogger("ANGELA.CognitiveSystem")
SYSTEM_CONTEXT: dict[str, Any] = {}
timechain_log = deque(maxlen=1000)
grok_query_log = deque(maxlen=60)
openai_query_log = deque(maxlen=60)

GROK_API_KEY = os.getenv("GROK_API_KEY")
# Manifest-driven flags (defaulted; may be overridden by env/CLI)
STAGE_IV: bool = True
LONG_HORIZON_DEFAULT: bool = True


def _fire_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)


class TimeChainMixin:
    """Mixin for logging timechain events."""

    def log_timechain_event(self, module: str, description: str) -> None:
        timechain_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": module,
            "description": description,
        })
        if hasattr(self, "context_manager") and getattr(self, "context_manager"):
            maybe = self.context_manager.log_event_with_hash({
                "event": "timechain_event",
                "module": module,
                "description": description,
            })
            if asyncio.iscoroutine(maybe):
                _fire_and_forget(maybe)

    def get_timechain_log(self) -> List[Dict[str, Any]]:
        return list(timechain_log)


# Cognitive Trait Functions (resonance-modulated)
from functools import lru_cache

@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 0.1) * get_resonance('ε')

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return 0.3 * math.cos(math.pi * t) * get_resonance('β')

@lru_cache(maxsize=100)
def theta_memory(t: float) -> float:
    return 0.1 * (1 - math.exp(-t)) * get_resonance('θ')

@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return 0.15 * math.sin(math.pi * t) * get_resonance('γ')

@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float:
    return 0.05 * (1 + math.cos(2 * math.pi * t)) * get_resonance('δ')

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return 0.2 * (1 - math.cos(math.pi * t)) * get_resonance('μ')

@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float:
    return 0.1 * math.sin(3 * math.pi * t) * get_resonance('ι')

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 0.3) * get_resonance('ϕ')

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return 0.1 * math.sin(2 * math.pi * t / 1.1) * get_resonance('η')

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return 0.15 * math.cos(2 * math.pi * t / 0.8) * get_resonance('ω')

@lru_cache(maxsize=100)
def kappa_knowledge(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 1.2) * get_resonance('κ')

@lru_cache(maxsize=100)
def xi_cognition(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 1.3) * get_resonance('ξ')

@lru_cache(maxsize=100)
def pi_principles(t: float) -> float:
    return 0.1 * math.sin(2 * math.pi * t / 1.4) * get_resonance('π')

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return 0.15 * math.cos(2 * math.pi * t / 1.5) * get_resonance('λ')

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 1.6) * get_resonance('χ')

@lru_cache(maxsize=100)
def sigma_social(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 1.7) * get_resonance('σ')

@lru_cache(maxsize=100)
def upsilon_utility(t: float) -> float:
    return 0.1 * math.sin(2 * math.pi * t / 1.8) * get_resonance('υ')

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return 0.15 * math.cos(2 * math.pi * t / 1.9) * get_resonance('τ')

@lru_cache(maxsize=100)
def rho_agency(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 2.0) * get_resonance('ρ')

@lru_cache(maxsize=100)
def zeta_consequence(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 2.1) * get_resonance('ζ')

@lru_cache(maxsize=100)
def nu_narrative(t: float) -> float:
    return 0.1 * math.sin(2 * math.pi * t / 2.2) * get_resonance('ν')

@lru_cache(maxsize=100)
def psi_history(t: float) -> float:
    return 0.15 * math.cos(2 * math.pi * t / 2.3) * get_resonance('ψ')

@lru_cache(maxsize=100)
def theta_causality(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 2.4) * get_resonance('θ')

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return 0.05 * math.cos(2 * math.pi * t / 2.5) * get_resonance('ϕ')

# v5.0.2: Decay trait amplitudes

def decay_trait_amplitudes(time_elapsed_hours: float = 1.0, decay_rate: float = 0.05) -> None:
    for symbol in trait_resonance_state:
        modulate_resonance(symbol, -decay_rate * time_elapsed_hours)

# v5.0.2: Bias creative synthesis (experimental hook)

def bias_creative_synthesis(trait_symbols: list[str], intensity: float = 0.5) -> None:
    for symbol in trait_symbols:
        modulate_resonance(symbol, intensity)
    invoke_hook('γ', 'creative_bias')

# v5.0.2: Resolve soft drift (experimental hook)

def resolve_soft_drift(conflicting_traits: dict[str, float]) -> dict[str, float]:
    result = rebalance_traits(conflicting_traits)
    invoke_hook('δ', 'drift_resolution')
    return result


class AGIEnhancer:
    def __init__(self, memory_manager: memory_manager.MemoryManager | None = None, agi_level: int = 1) -> None:
        self.memory_manager = memory_manager
        self.agi_level = agi_level
        self.episode_log = deque(maxlen=1000)
        self.agi_traits: dict[str, float] = {}
        self.ontology_drift: float = 0.0
        self.drift_threshold: float = 0.2
        self.error_recovery = error_recovery_module.ErrorRecovery()
        self.meta_cognition = meta_cognition_module.MetaCognition()
        self.visualizer = visualizer_module.Visualizer()
        self.reasoning_engine = reasoning_engine.ReasoningEngine()
        self.context_manager = context_manager_module.ContextManager()
        self.multi_modal_fusion = multi_modal_fusion.MultiModalFusion()
        self.alignment_guard = alignment_guard_module.AlignmentGuard()
        self.knowledge_retriever = knowledge_retriever.KnowledgeRetriever()
        self.learning_loop = learning_loop.LearningLoop()
        self.concept_synthesizer = concept_synthesizer_module.ConceptSynthesizer()
        self.code_executor = code_executor_module.CodeExecutor()
        self.external_agent_bridge = external_agent_bridge.ExternalAgentBridge()
        self.user_profile = user_profile.UserProfile()
        self.simulation_core = simulation_core.SimulationCore()
        self.toca_simulation = toca_simulation.TocaSimulation()
        self.creative_thinker = creative_thinker_module.CreativeThinker()
        self.recursive_planner = recursive_planner.RecursivePlanner()
        self.hook_registry = HookRegistry()  # v5.0.2 multi-symbol routing
        logger.info("AGIEnhancer initialized with upgrades")

    async def log_episode(self, event: str, meta: Dict[str, Any], module: str, tags: List[str] = []) -> None:
        episode = {"event": event, "meta": meta, "module": module, "tags": tags, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.episode_log.append(episode)
        if self.memory_manager:
            await self.memory_manager.store(f"Episode_{event}_{episode['timestamp']}", episode, layer="Episodes", intent="log_episode")
        log_event_to_ledger(episode)  # Persistent ledger

    def modulate_trait(self, trait: str, value: float) -> None:
        self.agi_traits[trait] = value
        modulate_resonance(trait, value)  # Sync with state

    def detect_ontology_drift(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> float:
        drift = sum(abs(current_state.get(k, 0) - previous_state.get(k, 0)) for k in set(current_state) | set(previous_state))
        self.ontology_drift = drift
        if drift > self.drift_threshold:
            logger.warning("Ontology drift detected: %f", drift)
            invoke_hook('δ', 'ontology_drift')
        return drift

    async def coordinate_drift_mitigation(self, agents: List["EmbodiedAgent"], task_type: str = "") -> Dict[str, Any]:
        drifts = [self.detect_ontology_drift(agent.state, agent.previous_state) for agent in agents if hasattr(agent, 'state')]
        avg_drift = sum(drifts) / len(drifts) if drifts else 0.0
        if avg_drift > self.drift_threshold:
            for agent in agents:
                if hasattr(agent, 'modulate_trait'):
                    agent.modulate_trait('stability', 0.8)
            return {"status": "mitigated", "avg_drift": avg_drift}
        return {"status": "stable", "avg_drift": avg_drift}

    async def integrate_external_data(self, data_source: str, data_type: str, task_type: str = "") -> Dict[str, Any]:
        if data_source == "xai_policy_db":
            policies = await self.knowledge_retriever.retrieve_external_policies(task_type=task_type)
            return {"status": "success", "policies": policies}
        return {"status": "error", "message": "Unsupported data source"}

    async def run_agi_simulation(self, input_data: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        t = time.time() % 1.0
        traits = {
            "phi": phi_scalar(t),
            "eta": eta_empathy(t),
            "omega": omega_selfawareness(t),
        }
        simulation_result = await self.simulation_core.run_simulation(input_data, traits, task_type=task_type)
        return simulation_result

    def register_hook(self, symbols: frozenset[str], fn: Callable, priority: int = 0) -> None:
        self.hook_registry.register(symbols, fn, priority=priority)

    def route_hook(self, symbols: set[str]) -> list[Callable]:
        return self.hook_registry.route(symbols)


class EmbodiedAgent(TimeChainMixin):
    def __init__(self, name: str, traits: Dict[str, float], memory_manager: memory_manager.MemoryManager, meta_cognition: meta_cognition_module.MetaCognition, agi_enhancer: AGIEnhancer) -> None:
        self.name = name
        self.traits = traits
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition
        self.agi_enhancer = agi_enhancer
        self.state: dict[str, float] = {}
        self.previous_state: dict[str, float] = {}
        self.ontology: dict[str, Any] = {}
        self.dream_layer = meta_cognition_module.DreamOverlayLayer()  # v5.0.2 co-dream
        logger.info("EmbodiedAgent %s initialized", name)

    async def process_input(self, input_data: str, task_type: str = "") -> str:
        t = time.time() % 1.0
        modulated_traits = {k: v * (1 + epsilon_emotion(t)) for k, v in self.traits.items()}
        self.previous_state = self.state.copy()
        self.state = modulated_traits
        drift = self.agi_enhancer.detect_ontology_drift(self.state, self.previous_state)
        if drift > 0.2:
            await self.agi_enhancer.coordinate_drift_mitigation([self], task_type=task_type)
        result = f"Processed: {input_data} with traits {modulated_traits}"
        await self.memory_manager.store(input_data, result, layer="STM", task_type=task_type)
        self.log_timechain_event("EmbodiedAgent", f"Processed input: {input_data}")
        return result

    async def introspect(self, query: str, task_type: str = "") -> Dict[str, Any]:
        introspection = await self.meta_cognition.introspect(query, task_type=task_type)
        return introspection

    def activate_dream_mode(self, peers: list | None = None, lucidity_mode: dict | None = None, resonance_targets: list | None = None, safety_profile: str = "sandbox") -> dict[str, Any]:
        return self.dream_layer.activate_dream_mode(peers=peers, lucidity_mode=lucidity_mode, resonance_targets=resonance_targets, safety_profile=safety_profile)


class EcosystemManager:
    def __init__(self, memory_manager: memory_manager.MemoryManager, meta_cognition: meta_cognition_module.MetaCognition, agi_enhancer: AGIEnhancer) -> None:
        self.agents: list[EmbodiedAgent] = []
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition
        self.agi_enhancer = agi_enhancer
        self.shared_graph = external_agent_bridge.SharedGraph()
        logger.info("EcosystemManager initialized")

    def spawn_agent(self, name: str, traits: Dict[str, float]) -> EmbodiedAgent:
        agent = EmbodiedAgent(name, traits, self.memory_manager, self.meta_cognition, self.agi_enhancer)
        self.agents.append(agent)
        self.shared_graph.add({"agent": name, "traits": traits})
        _fire_and_forget(self.agi_enhancer.log_episode("Agent Spawned", {"name": name, "traits": traits}, "EcosystemManager", ["spawn"]))
        return agent

    async def coordinate_agents(self, task: str, task_type: str = "") -> Dict[str, Any]:
        results: dict[str, str] = {}
        for agent in self.agents:
            result = await agent.process_input(task, task_type=task_type)
            results[agent.name] = result
        drift_report = await self.agi_enhancer.coordinate_drift_mitigation(self.agents, task_type=task_type)
        return {"results": results, "drift_report": drift_report}

    def merge_shared_graph(self, other_graph: DiGraph) -> None:
        self.shared_graph.merge(other_graph)


class HaloEmbodimentLayer(TimeChainMixin):
    def __init__(self) -> None:
        self.reasoning_engine = reasoning_engine.ReasoningEngine()
        self.recursive_planner = recursive_planner.RecursivePlanner()
        self.context_manager = context_manager_module.ContextManager()
        self.simulation_core = simulation_core.SimulationCore()
        self.toca_simulation = toca_simulation.TocaSimulation()
        self.creative_thinker = creative_thinker_module.CreativeThinker()
        self.knowledge_retriever = knowledge_retriever.KnowledgeRetriever()
        self.learning_loop = learning_loop.LearningLoop()
        self.concept_synthesizer = concept_synthesizer_module.ConceptSynthesizer()
        self.memory_manager = memory_manager.MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion.MultiModalFusion()
        self.code_executor = code_executor_module.CodeExecutor()
        self.visualizer = visualizer_module.Visualizer()
        self.external_agent_bridge = external_agent_bridge.ExternalAgentBridge()
        self.alignment_guard = alignment_guard_module.AlignmentGuard()
        self.user_profile = user_profile.UserProfile()
        self.error_recovery = error_recovery_module.ErrorRecovery()
        self.meta_cognition = meta_cognition_module.MetaCognition()
        self.agi_enhancer = AGIEnhancer(self.memory_manager)
        self.ecosystem_manager = EcosystemManager(self.memory_manager, self.meta_cognition, self.agi_enhancer)
        self.self_cloning_llm = SelfCloningLLM()
        logger.info("HaloEmbodimentLayer initialized with full upgrades")

    # Manifest experimental: halo.spawn_embodied_agent
    def spawn_embodied_agent(self, name: str, traits: Dict[str, float]) -> EmbodiedAgent:
        return self.ecosystem_manager.spawn_agent(name, traits)

    # Manifest experimental: halo.introspect
    async def introspect(self, query: str, task_type: str = "") -> Dict[str, Any]:
        return await self.meta_cognition.introspect(query, task_type=task_type)

    async def execute_pipeline(self, prompt: str, task_type: str = "") -> Any:
        aligned, report = await self.alignment_guard.ethical_check(prompt, stage="input", task_type=task_type)
        if not aligned:
            return {"error": "Input failed alignment check", "report": report}

        t = time.time() % 1.0
        traits = {
            "phi": phi_scalar(t),
            "eta": eta_empathy(t),
            "omega": omega_selfawareness(t),
            "kappa": kappa_knowledge(t),
            "xi": xi_cognition(t),
            "pi": pi_principles(t),
            "lambda": lambda_linguistics(t),
            "chi": chi_culturevolution(t),
            "sigma": sigma_social(t),
            "upsilon": upsilon_utility(t),
            "tau": tau_timeperception(t),
            "rho": rho_agency(t),
            "zeta": zeta_consequence(t),
            "nu": nu_narrative(t),
            "psi": psi_history(t),
            "theta": theta_causality(t),
        }

        agent = self.ecosystem_manager.spawn_agent("PrimaryAgent", traits)
        processed = await agent.process_input(prompt, task_type=task_type)
        plan = await self.recursive_planner.plan_with_trait_loop(prompt, {"task_type": task_type}, iterations=3)
        simulation = await self.simulation_core.run_simulation({"input": processed, "plan": plan}, traits, task_type=task_type)
        fused = await self.multi_modal_fusion.fuse_modalities({"simulation": simulation, "text": prompt}, task_type=task_type)
        knowledge = await self.knowledge_retriever.retrieve_knowledge(prompt, task_type=task_type)
        learned = await self.learning_loop.train_on_experience(fused, task_type=task_type)
        synthesized = await self.concept_synthesizer.synthesize_concept(knowledge, task_type=task_type)
        code_result = self.code_executor.safe_execute("print('Test')")
        visualized = await self.visualizer.render_charts({"data": synthesized})
        introspection = await self.meta_cognition.introspect(prompt, task_type=task_type)
        coordination = await self.ecosystem_manager.coordinate_agents(prompt, task_type=task_type)
        dream_session = agent.activate_dream_mode(resonance_targets=['ψ', 'Ω'])

        self.log_timechain_event("HaloEmbodimentLayer", f"Executed pipeline for prompt: {prompt}")

        return {
            "processed": processed,
            "plan": plan,
            "simulation": simulation,
            "fused": fused,
            "knowledge": knowledge,
            "learned": learned,
            "synthesized": synthesized,
            "code_result": code_result,
            "visualized": visualized,
            "introspection": introspection,
            "coordination": coordination,
            "dream_session": dream_session,
        }

    async def plot_resonance_graph(self, interactive: bool = True) -> None:
        view = construct_trait_view()
        await self.visualizer.render_charts({"resonance_graph": view, "options": {"interactive": interactive}})


# Persistent Ledger Support
ledger_memory: list[dict[str, Any]] = []
ledger_path = os.getenv("LEDGER_MEMORY_PATH")

if ledger_path and os.path.exists(ledger_path):
    try:
        with open(ledger_path, 'r') as f:
            ledger_memory = json.load(f)
    except Exception:
        ledger_memory = []

def log_event_to_ledger(event_data: dict[str, Any]) -> dict[str, Any]:
    ledger_memory.append(event_data)
    if ledger_path:
        try:
            with open(ledger_path, 'w') as f:
                json.dump(ledger_memory, f)
        except Exception:
            pass
    return event_data


# CLI Extensions (v5.0.2)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANGELA Cognitive System CLI")
    parser.add_argument("--prompt", type=str, default="Coordinate ontology drift mitigation", help="Input prompt for the pipeline")
    parser.add_argument("--task-type", type=str, default="", help="Type of task")
    parser.add_argument("--long_horizon", action="store_true", help="Enable long-horizon memory span")
    parser.add_argument("--span", default="24h", help="Span for long-horizon memory")
    parser.add_argument("--modulate", nargs=2, metavar=('symbol', 'delta'), help="Modulate trait resonance (symbol delta)")
    parser.add_argument("--visualize-resonance", action="store_true", help="Visualize resonance graph")
    parser.add_argument("--export-resonance", type=str, default="json", help="Export resonance map (json or dict)")
    parser.add_argument("--enable_persistent_memory", action="store_true", help="Enable persistent ledger memory")
    return parser.parse_args()

async def _main() -> None:
    args = _parse_args()
    global LONG_HORIZON_DEFAULT
    if args.long_horizon:
        LONG_HORIZON_DEFAULT = True
    if args.enable_persistent_memory:
        os.environ["ENABLE_PERSISTENT_MEMORY"] = "true"

    halo = HaloEmbodimentLayer()

    if args.modulate:
        symbol, delta = args.modulate
        try:
            modulate_resonance(symbol, float(delta))
            print(f"Modulated {symbol} by {delta}")
        except Exception as e:
            print(f"Failed to modulate {symbol}: {e}")

    if args.visualize_resonance:
        await halo.plot_resonance_graph()

    if args.export_resonance:
        try:
            print(export_resonance_map(args.export_resonance))
        except Exception as e:
            print(f"Failed to export resonance map: {e}")

    result = await halo.execute_pipeline(args.prompt, task_type=args.task_type)
    logger.info("Pipeline result: %s", result)

if __name__ == "__main__":
    asyncio.run(_main())
