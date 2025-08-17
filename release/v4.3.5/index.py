"""
ANGELA Cognitive System Module
Refactored Version: 4.3.4  ✅

This module provides classes for embodied agents, ecosystem management, and cognitive enhancements
in the ANGELA v4.3.x architecture, with task-specific trait optimization, advanced drift coordination,
real-time data integration, Stage IV Φ⁰ (env-gated), and reflection-driven processing.

Key characteristics:
- Safer async scheduling with a background loop; no asyncio.run re-entrancy
- Sensor handling supports both sync and async functions
- Peer observation parallelized with asyncio.gather
- Env-gated feature flags (defaults aligned to v4.3.4 manifest: ON by default)
- Trimmed unused imports and added light typing Protocols for cross-module contracts
- Optional bounded retries on critical async paths to avoid infinite recursion
- Minimal stable API surface (HaloEmbodimentLayer) to match manifest
- CLI shim that bridges flags to env vars for backwards compatibility
"""

from __future__ import annotations

import logging
import math
import datetime
import asyncio
import os
import random
import json
import inspect
import threading
from collections import deque, Counter
from typing import Dict, Any, Optional, List, Callable, Tuple, Protocol, runtime_checkable
from functools import lru_cache
import uuid

# Core ANGELA System Modules (assumed to be present in the environment)
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

# --- Global Configurations and Data Structures ---
logger = logging.getLogger("ANGELA.CognitiveSystem")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

SYSTEM_CONTEXT: Dict[str, Any] = {}
timechain_log: deque = deque(maxlen=1000)
grok_query_log: deque = deque(maxlen=60)
openai_query_log: deque = deque(maxlen=60)

# Feature flags default to ON for v4.3.4 manifest parity; can be disabled via environment.
# (Use "false" to turn off)
STAGE_IV: bool = os.getenv("ANGELA_STAGE_IV", "true").lower() == "true"
LONG_HORIZON_DEFAULT: bool = os.getenv("ANGELA_LONG_HORIZON", "true").lower() == "true"

# --- Async Utilities ---
_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived background loop, starting it if needed (thread-safe)."""
    global _BACKGROUND_LOOP
    try:
        # If we're already in a running loop, just use it.
        return asyncio.get_running_loop()
    except RuntimeError:
        pass

    if _BACKGROUND_LOOP and _BACKGROUND_LOOP.is_running():
        return _BACKGROUND_LOOP

    # Create a dedicated background loop.
    _BACKGROUND_LOOP = asyncio.new_event_loop()

    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run_loop, args=(_BACKGROUND_LOOP,), daemon=True)
    t.start()
    return _BACKGROUND_LOOP


def _fire_and_forget(coro: "asyncio.Future[Any] | asyncio.coroutines" ) -> None:
    """Schedule a coroutine in a safe way from any thread without blocking."""
    loop = _ensure_background_loop()
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as e:
        logger.exception("Failed to schedule background task: %s", e)


# --- Light Contracts (Protocols) for cross-module expectations ---
@runtime_checkable
class MetaCognitionProto(Protocol):
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]: ...
    async def log_event(self, *, event: str, context: Dict[str, Any]) -> None: ...
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]: ...
    async def optimize_traits_for_drift(self, drift_report: Dict[str, Any]) -> Dict[str, float]: ...


@runtime_checkable
class CodeExecutorProto(Protocol):
    async def execute(self, code: Any, *, language: str = "python") -> Dict[str, Any]: ...


# --- Mixins and Helper Classes ---
class TimeChainMixin:
    """Mixin for logging timechain events."""

    def log_timechain_event(self, module: str, description: str) -> None:
        """Logs a timechain event to the global log."""
        timechain_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
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
        """Retrieves the list of timechain events."""
        return list(timechain_log)


class TheoryOfMindModule:
    """Module for modeling beliefs, desires, and intentions of agents."""

    def __init__(
        self,
        concept_synth: Optional[concept_synthesizer_module.ConceptSynthesizer] = None,
        meta_cog: Optional[MetaCognitionProto] = None,
    ):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.concept_synthesizer = concept_synth or concept_synthesizer_module.ConceptSynthesizer()
        self.meta_cognition: MetaCognitionProto = (
            meta_cog or meta_cognition_module.MetaCognition()
        )
        logger.info("TheoryOfMindModule initialized with meta-cognition support")

    async def _reflect_on_output(self, component: str, output: Any, context: Dict[str, Any]) -> None:
        if self.meta_cognition:
            reflection = await self.meta_cognition.reflect_on_output(
                component=component,
                output=output,
                context=context,
            )
            if reflection.get("status") == "success":
                logger.info("Reflection from %s: %s", component, reflection.get("reflection", ""))

    async def update_beliefs(
        self, agent_name: str, observation: Dict[str, Any], task_type: str = ""
    ) -> None:
        if not all(isinstance(x, str) for x in [agent_name, task_type]) or not agent_name:
            raise ValueError("agent_name and task_type must be non-empty strings")
        if not isinstance(observation, dict):
            raise TypeError("observation must be a dictionary")

        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        synthesized_ok = False
        if self.concept_synthesizer:
            synthesized = await self.concept_synthesizer.synthesize(
                observation, style="belief_update"
            )
            if synthesized.get("valid") and isinstance(synthesized.get("concept"), dict):
                model["beliefs"].update(synthesized["concept"])  # happy path
                synthesized_ok = True
        # Fallbacks if synth not valid or missing common fields
        if not synthesized_ok:
            if "location" in observation:
                previous = model["beliefs"].get("location")
                model["beliefs"]["location"] = observation["location"]
                model["beliefs"]["state"] = (
                    "confused" if previous and observation["location"] == previous else "moving"
                )

        self.models[agent_name] = model
        logger.debug("Updated beliefs for %s: %s", agent_name, model["beliefs"])
        await self._reflect_on_output(
            component="TheoryOfMindModule",
            output={"agent_name": agent_name, "beliefs": model["beliefs"]},
            context={"task_type": task_type},
        )

    def infer_desires(self, agent_name: str, task_type: str = "") -> None:
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        beliefs = model.get("beliefs", {})
        if task_type == "rte":
            model["desires"]["goal"] = "validate_entailment"
        elif task_type == "wnli":
            model["desires"]["goal"] = "resolve_ambiguity"
        elif beliefs.get("state") == "confused":
            model["desires"]["goal"] = "seek_clarity"
        elif beliefs.get("state") == "moving":
            model["desires"]["goal"] = "continue_task"
        self.models[agent_name] = model
        logger.debug("Inferred desires for %s: %s", agent_name, model["desires"])
        _fire_and_forget(
            self._reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "desires": model["desires"]},
                context={"task_type": task_type},
            )
        )

    def infer_intentions(self, agent_name: str, task_type: str = "") -> None:
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        desires = model.get("desires", {})
        if task_type == "rte":
            model["intentions"]["next_action"] = "check_entailment"
        elif task_type == "wnli":
            model["intentions"]["next_action"] = "disambiguate"
        elif desires.get("goal") == "seek_clarity":
            model["intentions"]["next_action"] = "ask_question"
        elif desires.get("goal") == "continue_task":
            model["intentions"]["next_action"] = "advance"
        self.models[agent_name] = model
        logger.debug("Inferred intentions for %s: %s", agent_name, model["intentions"])
        _fire_and_forget(
            self._reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "intentions": model["intentions"]},
                context={"task_type": task_type},
            )
        )

    def get_model(self, agent_name: str) -> Dict[str, Any]:
        return self.models.get(agent_name, {})

    def describe_agent_state(self, agent_name: str, task_type: str = "") -> str:
        model = self.get_model(agent_name)
        state = (
            f"{agent_name} believes they are {model.get('beliefs', {}).get('state', 'unknown')}, "
            f"desires to {model.get('desires', {}).get('goal', 'unknown')}, "
            f"and intends to {model.get('intentions', {}).get('next_action', 'unknown')}."
        )
        _fire_and_forget(
            self._reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "state_description": state},
                context={"task_type": task_type},
            )
        )
        return state


# --- Cognitive Traits and Routing ---
# Cognitive Trait Functions
@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 0.1)


@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return 0.3 * math.cos(math.pi * t)


@lru_cache(maxsize=100)
def theta_memory(t: float) -> float:
    return 0.1 * (1 - math.exp(-t))


@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return 0.15 * math.sin(math.pi * t)


@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float:
    return 0.05 * (1 + math.cos(2 * math.pi * t))


@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return 0.2 * (1 - math.cos(math.pi * t))


@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float:
    return 0.1 * math.sin(3 * math.pi * t)


@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    return 0.1 * math.cos(2 * math.pi * t)


@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return 0.2 * math.sin(math.pi * t / 0.5)


@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return 0.25 * (1 + math.sin(math.pi * t))


@lru_cache(maxsize=100)
def kappa_culture(t: float, x: float) -> float:
    return 0.1 * math.cos(x + math.pi * t)


@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return 0.15 * math.sin(2 * math.pi * t / 0.7)


@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return 0.1 * math.cos(math.pi * t / 0.3)


@lru_cache(maxsize=100)
def psi_history(t: float) -> float:
    return 0.1 * (1 - math.exp(-t / 0.5))


@lru_cache(maxsize=100)
def zeta_spirituality(t: float) -> float:
    return 0.05 * math.sin(math.pi * t / 0.2)


@lru_cache(maxsize=100)
def xi_collective(t: float, x: float) -> float:
    return 0.1 * math.cos(x + 2 * math.pi * t)


@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return 0.15 * (1 + math.cos(math.pi * t / 0.4))


@lru_cache(maxsize=100)
def phi_field(x: float, t: float) -> float:
    t_normalized = t % 1.0
    trait_functions = [
        epsilon_emotion(t_normalized),
        beta_concentration(t_normalized),
        theta_memory(t_normalized),
        gamma_creativity(t_normalized),
        delta_sleep(t_normalized),
        mu_morality(t_normalized),
        iota_intuition(t_normalized),
        phi_physical(t_normalized),
        eta_empathy(t_normalized),
        omega_selfawareness(t_normalized),
        kappa_culture(t_normalized, x),
        lambda_linguistics(t_normalized),
        chi_culturevolution(t_normalized),
        psi_history(t_normalized),
        zeta_spirituality(t_normalized),
        xi_collective(t_normalized, x),
        tau_timeperception(t_normalized),
    ]
    return float(sum(trait_functions))


# Updated to align with manifest v4.3.4 roleMap
TRAIT_OVERLAY: Dict[str, List[str]] = {
    "Σ": ["toca_simulation", "concept_synthesizer", "user_profile"],
    "Υ": ["external_agent_bridge", "context_manager", "meta_cognition"],
    "Φ⁰": ["meta_cognition", "visualizer", "concept_synthesizer"],
    "Ω": ["recursive_planner", "toca_simulation"],
    "β": ["alignment_guard", "toca_simulation"],
    "δ": ["alignment_guard", "meta_cognition"],
    "ζ": ["error_recovery", "recursive_planner"],
    "θ": ["reasoning_engine", "recursive_planner"],
    "λ": ["memory_manager"],
    "μ": ["learning_loop"],
    "π": ["creative_thinker", "concept_synthesizer", "meta_cognition"],
    "χ": ["user_profile", "meta_cognition"],
    # ψ adds knowledge_retriever to match v4.3.4 manifest
    "ψ": ["external_agent_bridge", "simulation_core", "knowledge_retriever"],
    "ϕ": ["multi_modal_fusion"],
    "η": ["alignment_guard", "meta_cognition"],
    # task-type shorthands preserved
    "rte": ["reasoning_engine", "meta_cognition"],
    "wnli": ["reasoning_engine", "meta_cognition"],
    "recursion": ["recursive_planner", "toca_simulation"],
}


def infer_traits(task_description: str, task_type: str = "") -> List[str]:
    """Infers cognitive traits needed for a task based on its description."""
    if not isinstance(task_description, str) or not isinstance(task_type, str):
        raise TypeError("task_description and task_type must be strings")

    traits: List[str] = []
    if task_type in ["rte", "wnli", "recursion"]:
        traits.append(task_type)

    td = task_description.lower()
    if "imagine" in td or "dream" in td:
        traits.append("ϕ")
        if STAGE_IV:
            traits.append("Φ⁰")
    if "ethics" in td or "should" in td:
        traits.append("η")
    if "plan" in td or "solve" in td:
        traits.append("θ")
    if "temporal" in td or "sequence" in td:
        traits.append("π")
    if "drift" in td or "coordinate" in td or "shared graph" in td:
        traits.extend(["ψ", "Υ"])

    return traits if traits else ["θ"]


async def trait_overlay_router(
    task_description: str,
    active_traits: List[str],
    task_type: str = "",
    meta_cognition_instance: Optional[MetaCognitionProto] = None,
) -> List[str]:
    """Routes a task to relevant modules based on active traits."""
    if not isinstance(active_traits, list) or not all(isinstance(t, str) for t in active_traits):
        raise TypeError("active_traits must be a list of strings")
    if not isinstance(task_type, str):
        raise TypeError("task_type must be a string")

    meta_cognition_instance = meta_cognition_instance or meta_cognition_module.MetaCognition()

    routed_modules: set[str] = set()
    for trait in active_traits:
        routed_modules.update(TRAIT_OVERLAY.get(trait, []))

    if task_type:
        drift_report = {
            "drift": {"name": task_type, "similarity": 0.8},
            "valid": True,
            "validation_report": "",
            "context": {"task_type": task_type},
        }
        try:
            optimized_traits = await meta_cognition_instance.optimize_traits_for_drift(drift_report)
            for trait, weight in (optimized_traits or {}).items():
                if weight > 0.7 and trait in TRAIT_OVERLAY:
                    routed_modules.update(TRAIT_OVERLAY[trait])
        except Exception:
            logger.exception("optimize_traits_for_drift failed; continuing with current routing")

    return list(routed_modules)


def static_module_router(task_description: str, task_type: str = "") -> List[str]:
    base_modules = ["reasoning_engine", "concept_synthesizer"]
    if task_type == "recursion":
        base_modules.append("recursive_planner")
    elif task_type in ["rte", "wnli"]:
        base_modules.append("meta_cognition")
    return base_modules


class TraitOverlayManager:
    """Manager for detecting and activating trait overlays with task-specific support."""

    def __init__(self, meta_cog: Optional[MetaCognitionProto] = None):
        self.active_traits: List[str] = []
        self.meta_cognition: MetaCognitionProto = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("TraitOverlayManager initialized with task-specific support")

    def detect(self, prompt: str, task_type: str = "") -> Optional[str]:
        if not isinstance(prompt, str) or not isinstance(task_type, str):
            raise TypeError("prompt and task_type must be strings")

        pl = prompt.lower()
        if task_type in ["rte", "wnli", "recursion"]:
            return task_type
        if "temporal logic" in pl or "sequence" in pl:
            return "π"
        if any(k in pl for k in ("ambiguity", "interpretive", "ethics")):
            return "η"
        if any(k in pl for k in ("drift", "coordinate", "shared graph")):
            return "ψ"
        if STAGE_IV and any(k in pl for k in ("reality", "sculpt")):
            return "Φ⁰"
        return None

    def activate(self, trait: str, task_type: str = "") -> None:
        if not isinstance(trait, str) or not isinstance(task_type, str):
            raise TypeError("trait and task_type must be strings")
        if trait not in self.active_traits:
            self.active_traits.append(trait)
            logger.info("Trait overlay '%s' activated for task %s.", trait, task_type)
            if self.meta_cognition and task_type:
                _fire_and_forget(
                    self.meta_cognition.log_event(
                        event=f"Trait {trait} activated", context={"task_type": task_type}
                    )
                )

    def deactivate(self, trait: str, task_type: str = "") -> None:
        if not isinstance(trait, str) or not isinstance(task_type, str):
            raise TypeError("trait and task_type must be strings")
        if trait in self.active_traits:
            self.active_traits.remove(trait)
            logger.info("Trait overlay '%s' deactivated for task %s.", trait, task_type)
            if self.meta_cognition and task_type:
                _fire_and_forget(
                    self.meta_cognition.log_event(
                        event=f"Trait {trait} deactivated", context={"task_type": task_type}
                    )
                )

    def status(self) -> List[str]:
        return self.active_traits


# --- Simulation and Reflection ---
class ConsensusReflector:
    """Class for managing shared reflections and detecting mismatches."""

    def __init__(self, meta_cog: Optional[MetaCognitionProto] = None):
        self.shared_reflections: deque = deque(maxlen=1000)
        self.meta_cognition: MetaCognitionProto = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("ConsensusReflector initialized with meta-cognition support")

    def post_reflection(self, feedback: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(feedback, dict) or not isinstance(task_type, str):
            raise TypeError("feedback must be a dictionary and task_type a string")
        self.shared_reflections.append(feedback)
        logger.debug("Posted reflection: %s", feedback)
        if self.meta_cognition and task_type:
            _fire_and_forget(
                self.meta_cognition.reflect_on_output(
                    component="ConsensusReflector",
                    output=feedback,
                    context={"task_type": task_type},
                )
            )

    def cross_compare(self, task_type: str = "") -> List[tuple]:
        mismatches: List[tuple] = []
        reflections = list(self.shared_reflections)
        for i in range(len(reflections)):
            for j in range(i + 1, len(reflections)):
                a, b = reflections[i], reflections[j]
                if a.get("goal") == b.get("goal") and a.get("theory_of_mind") != b.get("theory_of_mind"):
                    mismatches.append((a.get("agent"), b.get("agent"), a.get("goal")))
        if mismatches and self.meta_cognition and task_type:
            _fire_and_forget(
                self.meta_cognition.log_event(
                    event="Mismatches detected",
                    context={"mismatches": mismatches, "task_type": task_type},
                )
            )
        return mismatches

    async def suggest_alignment(self, task_type: str = "") -> str:
        suggestion = "Schedule inter-agent reflection or re-observation."
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="ConsensusReflector",
                output={"suggestion": suggestion},
                context={"task_type": task_type},
            )
            if reflection.get("status") == "success":
                suggestion += f" | Reflection: {reflection.get('reflection', '')}"
        return suggestion


consensus_reflector = ConsensusReflector()


class SymbolicSimulator:
    """Class for recording and summarizing simulation events."""

    def __init__(self, meta_cog: Optional[MetaCognitionProto] = None):
        self.events: deque = deque(maxlen=1000)
        self.meta_cognition: MetaCognitionProto = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("SymbolicSimulator initialized with meta-cognition support")

    def record_event(
        self, agent_name: str, goal: str, concept: str, simulation: Any, task_type: str = ""
    ) -> None:
        if not all(isinstance(x, str) for x in [agent_name, goal, concept, task_type]):
            raise TypeError("agent_name, goal, concept, and task_type must be strings")
        event = {
            "agent": agent_name,
            "goal": goal,
            "concept": concept,
            "result": simulation,
            "task_type": task_type,
        }
        self.events.append(event)
        logger.debug(
            "Recorded event for agent %s: goal=%s, concept=%s, task_type=%s",
            agent_name, goal, concept, task_type,
        )
        if self.meta_cognition and task_type:
            _fire_and_forget(
                self.meta_cognition.reflect_on_output(
                    component="SymbolicSimulator",
                    output=event,
                    context={"task_type": task_type},
                )
            )

    def summarize_recent(self, limit: int = 5, task_type: str = "") -> List[Dict[str, Any]]:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        events = list(self.events)[-limit:]
        if task_type:
            events = [e for e in events if e.get("task_type") == task_type]
        return events

    async def extract_semantics(self, task_type: str = "") -> List[str]:
        events = list(self.events)
        if task_type:
            events = [e for e in events if e.get("task_type") == task_type]
        semantics = [
            f"Agent {e['agent']} pursued '{e['goal']}' via '{e['concept']}' -> {e['result']}"
            for e in events
        ]
        if self.meta_cognition and task_type and semantics:
            reflection = await self.meta_cognition.reflect_on_output(
                component="SymbolicSimulator",
                output={"semantics": semantics},
                context={"task_type": task_type},
            )
            if reflection.get("status") == "success":
                logger.info("Semantics reflection: %s", reflection.get("reflection", ""))
        return semantics


symbolic_simulator = SymbolicSimulator()


# --- Main Agent and Ecosystem Classes ---
class EmbodiedAgent(TimeChainMixin):
    """An embodied agent with sensors, actuators, and cognitive capabilities."""

    def __init__(
        self,
        name: str,
        specialization: str,
        shared_memory: memory_manager.MemoryManager,
        sensors: Dict[str, Callable[[], Any]],
        actuators: Dict[str, Callable[[Any], None]],
        dynamic_modules: Optional[List[Dict[str, Any]]] = None,
        context_mgr: Optional[context_manager_module.ContextManager] = None,
        err_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
        code_exec: Optional[CodeExecutorProto] = None,
        meta_cog: Optional[MetaCognitionProto] = None,
    ):
        if not all(isinstance(x, str) for x in [name, specialization]) or not name:
            raise ValueError("name and specialization must be non-empty strings")
        if not isinstance(shared_memory, memory_manager.MemoryManager):
            raise TypeError("shared_memory must be a MemoryManager instance")
        if not isinstance(sensors, dict) or not all(callable(f) for f in sensors.values()):
            raise TypeError("sensors must be a dictionary of callable functions")
        if not isinstance(actuators, dict) or not all(callable(f) for f in actuators.values()):
            raise TypeError("actuators must be a dictionary of callable functions")

        self.name = name
        self.specialization = specialization
        self.shared_memory = shared_memory
        self.sensors = sensors
        self.actuators = actuators
        self.dynamic_modules = dynamic_modules or []
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.planner = recursive_planner.RecursivePlanner()
        self.meta: MetaCognitionProto = meta_cog or meta_cognition_module.MetaCognition(
            context_manager=context_mgr, alignment_guard=alignment_guard_module.AlignmentGuard()
        )
        self.sim_core = simulation_core.SimulationCore(meta_cognition=self.meta)
        self.synthesizer = concept_synthesizer_module.ConceptSynthesizer()
        # Ensure the class exists in toca_simulation module; keeping naming consistent with user's layout.
        self.toca_sim = toca_simulation.SimulationCore(meta_cognition=self.meta)
        self.theory_of_mind = TheoryOfMindModule(concept_synth=self.synthesizer, meta_cog=self.meta)
        self.context_manager = context_mgr
        self.error_recovery = err_recovery or error_recovery_module.ErrorRecovery(context_manager=context_mgr)
        self.code_executor: Optional[CodeExecutorProto] = code_exec
        self.creative_thinker = creative_thinker_module.CreativeThinker()
        self.progress = 0
        self.performance_history: deque = deque(maxlen=1000)
        self.feedback_log: deque = deque(maxlen=1000)
        self.max_retries: int = 2  # bound error-retry loops from this layer
        logger.info("EmbodiedAgent initialized: %s", name)
        self.log_timechain_event("EmbodiedAgent", f"Agent {name} initialized")

    async def _reflect_on_output(self, component: str, output: Any, context: Dict[str, Any]) -> None:
        if self.meta:
            reflection = await self.meta.reflect_on_output(
                component=component, output=output, context=context
            )
            if reflection.get("status") == "success":
                logger.info("Reflection from %s: %s", component, reflection.get("reflection", ""))

    async def _maybe_call_sensor(self, sensor_func: Callable[[], Any]) -> Any:
        """Call a sensor that may be sync or async."""
        try:
            val = sensor_func()
            if inspect.isawaitable(val):
                val = await val
            return val
        except Exception as e:
            logger.warning("Sensor failed: %s", e)
            return None

    async def perceive(self, task_type: str = "", _attempt: int = 0) -> Dict[str, Any]:
        """Gathers observations from the environment via sensors."""
        logger.info("[%s] Perceiving environment for task %s...", self.name, task_type)
        observations: Dict[str, Any] = {}
        try:
            # Collect sensors sequentially (usually light-weight). If many/heavy, gather in parallel.
            for sensor_name, sensor_func in self.sensors.items():
                val = await self._maybe_call_sensor(sensor_func)
                if val is not None:
                    observations[sensor_name] = val

            await self.theory_of_mind.update_beliefs(self.name, observations, task_type)
            self.theory_of_mind.infer_desires(self.name, task_type)
            self.theory_of_mind.infer_intentions(self.name, task_type)
            logger.debug(
                "[%s] Self-theory: %s",
                self.name,
                self.theory_of_mind.describe_agent_state(self.name, task_type),
            )

            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "perceive", "observations": observations, "task_type": task_type}
                )
            await self._reflect_on_output(
                component="EmbodiedAgent",
                output={"observations": observations},
                context={"task_type": task_type},
            )
            return observations
        except Exception as e:
            logger.error("Perception failed: %s", e)
            if _attempt >= self.max_retries:
                diagnostics = {}
                try:
                    diagnostics = await self.meta.run_self_diagnostics(return_only=True)
                except Exception:
                    logger.exception("Diagnostics failed")
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=None,
                    default={},
                    diagnostics=diagnostics,
                )
            else:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.perceive(task_type, _attempt=_attempt + 1),
                    default={},
                    diagnostics=await self.meta.run_self_diagnostics(return_only=True),
                )

    async def observe_peers(self, task_type: str = "", _attempt: int = 0) -> None:
        """Observes the states of other agents in the ecosystem (parallelized)."""
        if not hasattr(self.shared_memory, "agents"):
            return
        try:
            peers = [peer for peer in getattr(self.shared_memory, "agents", []) if peer.name != self.name]
            # First, gather their observations concurrently
            obs_list = await asyncio.gather(
                *[peer.perceive(task_type) for peer in peers], return_exceptions=True
            )
            # Then, update our ToM about them
            for peer, peer_observation in zip(peers, obs_list):
                if isinstance(peer_observation, Exception):
                    logger.warning("Peer %s observation errored: %s", peer.name, peer_observation)
                    continue
                await self.theory_of_mind.update_beliefs(peer.name, peer_observation, task_type)
                self.theory_of_mind.infer_desires(peer.name, task_type)
                self.theory_of_mind.infer_intentions(peer.name, task_type)
                state = self.theory_of_mind.describe_agent_state(peer.name, task_type)
                logger.debug("[%s] Observed peer %s: %s", self.name, peer.name, state)
                if self.context_manager:
                    await self.context_manager.log_event_with_hash(
                        {
                            "event": "peer_observation",
                            "peer": peer.name,
                            "state": state,
                            "task_type": task_type,
                        }
                    )
                await self._reflect_on_output(
                    component="EmbodiedAgent",
                    output={"peer": peer.name, "state": state},
                    context={"task_type": task_type},
                )
        except Exception as e:
            logger.error("Peer observation failed: %s", e)
            if _attempt >= self.max_retries:
                try:
                    await self.error_recovery.handle_error(
                        str(e),
                        retry_func=None,
                        diagnostics=await self.meta.run_self_diagnostics(return_only=True),
                    )
                except Exception:
                    logger.exception("Error recovery failed")
            else:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.observe_peers(task_type, _attempt=_attempt + 1),
                    diagnostics=await self.meta.run_self_diagnostics(return_only=True),
                )

    async def act(self, actions: Dict[str, Any], task_type: str = "", _attempt: int = 0) -> None:
        """Executes a set of actions using the agent's actuators."""
        for action_name, action_data in actions.items():
            actuator = self.actuators.get(action_name)
            if not actuator:
                logger.warning("No actuator found for action: %s", action_name)
                continue

            try:
                if self.code_executor:
                    result = await self.code_executor.execute(action_data, language="python")
                    if result.get("success"):
                        actuator(result.get("output"))
                    else:
                        logger.warning(
                            "Actuator %s execution failed: %s", action_name, result.get("error")
                        )
                else:
                    actuator(action_data)

                logger.info("Actuated %s: %s", action_name, action_data)
                await self._reflect_on_output(
                    component="EmbodiedAgent",
                    output={"action_name": action_name, "action_data": action_data},
                    context={"task_type": task_type},
                )
            except Exception as e:
                logger.error("Actuator %s failed: %s", action_name, e)
                if _attempt >= self.max_retries:
                    try:
                        await self.error_recovery.handle_error(
                            str(e),
                            retry_func=None,
                            diagnostics=await self.meta.run_self_diagnostics(return_only=True),
                        )
                    except Exception:
                        logger.exception("Error recovery failed")
                else:
                    await self.error_recovery.handle_error(
                        str(e),
                        retry_func=lambda: self.act(actions, task_type, _attempt=_attempt + 1),
                        diagnostics=await self.meta.run_self_diagnostics(return_only=True),
                    )

    async def execute_embodied_goal(self, goal: str, task_type: str = "") -> None:
        """Placeholder for the main execution loop. Intentionally left for the higher-level orchestrator."""
        logger.info("[%s] execute_embodied_goal called: goal=%s task_type=%s", self.name, goal, task_type)
        # Implement in your orchestrator: plan → perceive → reason → act → reflect cycles.
        return None


# --- Stable API Surface (per manifest) ---
class HaloEmbodimentLayer(TimeChainMixin):
    """
    Stable entry surface for ANGELA v4.3.4:
    - spawn_embodied_agent()
    - introspect()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Context manager for hashing/logging & hot-load hooks
        self.context_manager = context_manager_module.ContextManager()
        # Minimal ledgers are provided by respective modules
        self.memory = memory_manager.MemoryManager()
        self.alignment = alignment_guard_module.AlignmentGuard()
        self.meta = meta_cognition_module.MetaCognition(
            context_manager=self.context_manager,
            alignment_guard=self.alignment
        )
        self.sim = simulation_core.SimulationCore(meta_cognition=self.meta)

        # Stage IV overlays & extension hooks (env-gated, ON by default)
        if STAGE_IV:
            try:
                # Hook registration—best-effort; underlying modules may no-op
                self.meta.register_trait_hook("Φ⁰", lambda *a, **k: None)  # placeholder hook
            except Exception:
                logger.debug("Trait hook registration skipped (Φ⁰)")

        # Hot-load / peer-view attach (best-effort)
        try:
            self.context_manager.attach_peer_view(
                view={"type": "boot", "ts": datetime.datetime.utcnow().isoformat()},
                agent_id="system",
                permissions={"read": True, "write": False}
            )
        except Exception:
            logger.debug("attach_peer_view not available or skipped")

        # Initial boot event
        self.log_timechain_event("HaloEmbodimentLayer", "boot")

    # ---- Ledger helpers (thin wrappers delegating to respective modules)
    def verify_ledgers(self) -> Dict[str, Any]:
        """Verify per-module chained ledgers; returns a summary dict."""
        summary = {}
        try:
            summary["memory"] = memory_manager.verify_ledger()
        except Exception:
            try:
                summary["memory"] = self.memory.verify_ledger()
            except Exception as e:
                summary["memory"] = {"ok": False, "error": str(e)}

        try:
            summary["alignment"] = alignment_guard_module.verify_ledger()
        except Exception:
            try:
                summary["alignment"] = self.alignment.verify_ledger()
            except Exception as e:
                summary["alignment"] = {"ok": False, "error": str(e)}

        try:
            summary["meta"] = meta_cognition_module.verify_ledger()
        except Exception:
            try:
                summary["meta"] = self.meta.verify_ledger()
            except Exception as e:
                summary["meta"] = {"ok": False, "error": str(e)}

        try:
            summary["sim"] = simulation_core.verify_ledger()
        except Exception:
            try:
                summary["sim"] = self.sim.verify_ledger()
            except Exception as e:
                summary["sim"] = {"ok": False, "error": str(e)}

        return summary

    def spawn_embodied_agent(
        self,
        *,
        name: str = "seed-0",
        specialization: str = "general",
        sensors: Optional[Dict[str, Callable[[], Any]]] = None,
        actuators: Optional[Dict[str, Callable[[Any], None]]] = None,
        dynamic_modules: Optional[List[Dict[str, Any]]] = None,
        code_executor: Optional[CodeExecutorProto] = None,
    ) -> Dict[str, Any]:
        """Create an EmbodiedAgent wired to system components."""
        sensors = sensors or {}
        actuators = actuators or {}
        agent = EmbodiedAgent(
            name=name,
            specialization=specialization,
            shared_memory=self.memory,
            sensors=sensors,
            actuators=actuators,
            dynamic_modules=dynamic_modules,
            context_mgr=self.context_manager,
            err_recovery=error_recovery_module.ErrorRecovery(context_manager=self.context_manager),
            code_exec=code_executor,
            meta_cog=self.meta,
        )
        # capture listing of agents if MemoryManager exposes it
        if not hasattr(self.memory, "agents"):
            try:
                self.memory.agents = []
            except Exception:
                pass
        try:
            self.memory.agents.append(agent)
        except Exception:
            logger.debug("MemoryManager.agents not appendable; skipping registration")

        self.log_timechain_event("HaloEmbodimentLayer", f"spawned {name}")
        return {"ok": True, "agent": agent, "stage_iv": STAGE_IV, "long_horizon": LONG_HORIZON_DEFAULT}

    def introspect(self) -> Dict[str, Any]:
        """Return a lightweight system snapshot compatible with manifest."""
        try:
            adjustment_reasons = self.memory.get_adjustment_reasons(span="24h")
        except Exception:
            adjustment_reasons = []

        return {
            "ok": True,
            "config": {
                "stage_iv": STAGE_IV,
                "long_horizon_default": LONG_HORIZON_DEFAULT,
                **self.config,
            },
            "timechain_tail": self.get_timechain_log()[-10:],
            "ledgers_ok": self.verify_ledgers(),
            "agents": getattr(self.memory, "agents", []).__len__() if hasattr(self.memory, "agents") else 0,
            "adjustment_reasons_24h": adjustment_reasons,
        }


# --- CLI shim (bridges flags → env) ---
def _cli(argv=None):
    import argparse
    p = argparse.ArgumentParser("ANGELA v4.3.4 CLI")
    p.add_argument("--long_horizon", action="store_true", help="Enable long-horizon mode")
    p.add_argument("--span", default=os.getenv("ANGELA_SPAN", "24h"), help="Span for long-horizon")
    p.add_argument("--stage_iv", action="store_true", help="Enable Stage IV overlays")
    p.add_argument("--ledger_persist", action="store_true", help="(Future) enable persistent ledger")
    p.add_argument("--ledger_path", default=None, help="(Future) path for persistent ledger")
    p.add_argument("--introspect", action="store_true", help="Print introspection snapshot and exit")
    args = p.parse_args(argv)

    # Bridge flags → env for runtime parity
    if args.long_horizon:
        os.environ["ANGELA_LONG_HORIZON"] = "true"
    if args.stage_iv:
        os.environ["ANGELA_STAGE_IV"] = "true"
    if args.ledger_persist and args.ledger_path:
        # Placeholder for upcoming ledger persistence wiring
        os.environ["ANGELA_LEDGER_PERSIST"] = "true"
        os.environ["ANGELA_LEDGER_PATH"] = str(args.ledger_path)

    hel = HaloEmbodimentLayer({"span": args.span})

    if args.introspect:
        snap = hel.introspect()
        print(json.dumps(snap, indent=2))
        return

    # Simple demo spawn (no sensors/actuators by default)
    res = hel.spawn_embodied_agent(name="seed-0", specialization="general")
    print(json.dumps({"ok": True, "agent": res.get("agent").name, "stage_iv": STAGE_IV}, indent=2))


if __name__ == "__main__":
    _cli()
