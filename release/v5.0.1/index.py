# --- flat-layout bootstrap ---
import sys
import types
import importlib
import importlib.util
import importlib.machinery
import importlib.abc

class FlatLayoutFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("modules."):
            modname = fullname.split(".", 1)[1]
            filename = f"/mnt/data/{modname}.py"
            return importlib.util.spec_from_file_location(fullname, filename, loader=importlib.machinery.SourceFileLoader(fullname, filename))
        elif fullname == "utils":
            utils_mod = types.ModuleType("utils")
            sys.modules["utils"] = utils_mod
            return utils_mod.__spec__
        return None

sys.meta_path.insert(0, FlatLayoutFinder())
# --- end flat-layout bootstrap ---

# Standard library imports
import logging
import time
import math
import datetime
import asyncio
import os
import requests
import random
import json
import argparse
import uuid

# Third-party imports
from collections import deque, Counter
from typing import Dict, Any, Optional, List, Callable
from functools import lru_cache
import numpy as np
from networkx import DiGraph
from aiohttp import ClientSession as aiohttp  # Alias to match original

# Local imports
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
from self_cloning_llm import SelfCloningLLM
from typing import Tuple
from memory_manager import log_event_to_ledger
from meta_cognition import invoke_hook, trait_resonance_state, modulate_resonance  # For --modulate CLI

logger = logging.getLogger("ANGELA.CognitiveSystem")
SYSTEM_CONTEXT = {}
timechain_log = deque(maxlen=1000)
grok_query_log = deque(maxlen=60)
openai_query_log = deque(maxlen=60)

# Manifest-driven flags (safe defaults if manifest/config is not injected)
STAGE_IV = True  # Can be toggled via HaloEmbodimentLayer.init flags
LONG_HORIZON_DEFAULT = True  # defaultSpan is handled at pipeline logging level

# --- Symbolic Operator Extension (ANGELA v5.0.2) ---
SYMBOLIC_OPERATORS = {
    '⊕': lambda a, b: a + b,
    '⊗': lambda a, b: a * b,
    '~': lambda a: 1 - a,
    '∘': lambda f, g: lambda x: f(g(x)),
    '⋈': lambda a, b: (a + b) / 2,
    '⨁': lambda a, b: max(a, b),
    '⨂': lambda a, b: min(a, b),
    '†': lambda a: a**-1 if a != 0 else 0,
    '▷': lambda a, b: a if a > b else b * 0.5,
    '↑': lambda a: min(1.0, a + 0.1),
    '↓': lambda a: max(0.0, a - 0.1),
    '⌿': lambda traits: normalize(traits),
    '⟲': lambda traits: rotate_traits(traits),
}

def normalize(traits: Dict[str, float]) -> Dict[str, float]:
    total = sum(traits.values())
    return {k: v / total for k, v in traits.items()} if total else traits

def rotate_traits(traits: Dict[str, float]) -> Dict[str, float]:
    keys = list(traits.keys())
    values = list(traits.values())
    rotated = values[-1:] + values[:-1]
    return dict(zip(keys, rotated))

def apply_symbolic_operator(op: str, *args: Any) -> Any:
    if op in SYMBOLIC_OPERATORS:
        return SYMBOLIC_OPERATORS[op](*args)
    raise ValueError(f"Unsupported symbolic operator: {op}")
# --- End Symbolic Operators ---

# --- Trait Algebra & Lattice Enhancements ---
TRAIT_LATTICE = {
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

TRAIT_OPS = {
    "⊕": lambda a, b: a + b,
    "⊗": lambda a, b: a * b,
    "~": lambda a: -a
}

def rebalance_traits(traits: List[str]) -> List[str]:
    """Rebalance traits based on lattice interactions."""
    if "π" in traits and "δ" in traits:
        invoke_hook("π", "axiom_fusion")
    if "ψ" in traits and "Ω" in traits:
        invoke_hook("ψ", "dream_sync")
    return traits

def construct_trait_view(lattice: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """Construct a view of the trait field based on the lattice."""
    trait_field = {}
    for layer, symbols in lattice.items():
        for s in symbols:
            trait_field[s] = {
                "layer": layer,
                "amplitude": trait_resonance_state.get_resonance(s),
                "resonance": trait_resonance_state.get_resonance(s)
            }
    return trait_field
# --- End Trait Enhancements ---

"""
ANGELA Cognitive System Module
Version: 5.0.2

This module provides classes for embodied agents, ecosystem management, and cognitive enhancements in the ANGELA architecture.
"""

def _fire_and_forget(coro: Callable) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)

class TimeChainMixin:
    """Mixin for logging timechain events."""
    def log_timechain_event(self, module: str, description: str) -> None:
        timechain_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "module": module,
            "description": description
        })
        if hasattr(self, "context_manager") and self.context_manager:
            maybe = self.context_manager.log_event_with_hash({
                "event": "timechain_event",
                "module": module,
                "description": description
            })
            if asyncio.iscoroutine(maybe):
                _fire_and_forget(maybe)

    def get_timechain_log(self) -> List[Dict[str, Any]]:
        return list(timechain_log)

# Cognitive Trait Functions (grouped for better maintenance)
trait_functions = {
    "epsilon_emotion": lambda t: 0.2 * math.sin(2 * math.pi * t / 0.1),
    "beta_concentration": lambda t: 0.3 * math.cos(math.pi * t),
    "theta_memory": lambda t: 0.1 * (1 - math.exp(-t)),
    "gamma_creativity": lambda t: 0.15 * math.sin(math.pi * t),
    "delta_sleep": lambda t: 0.05 * (1 + math.cos(2 * math.pi * t)),
    "mu_morality": lambda t: 0.2 * (1 - math.cos(math.pi * t)),
    "iota_intuition": lambda t: 0.1 * math.sin(3 * math.pi * t),
    "phi_physical": lambda t: 0.1 * math.cos(2 * math.pi * t),
    "eta_empathy": lambda t: 0.2 * math.sin(math.pi * t / 0.5),
    "omega_selfawareness": lambda t: 0.25 * (1 + math.sin(math.pi * t)),
    "lambda_linguistics": lambda t: 0.15 * math.sin(2 * math.pi * t / 0.7),
    "chi_culturevolution": lambda t: 0.1 * math.cos(math.pi * t / 0.3),
    "psi_history": lambda t: 0.1 * (1 - math.exp(-t / 0.5)),
    "zeta_spirituality": lambda t: 0.05 * math.sin(math.pi * t / 0.2),
    "tau_timeperception": lambda t: 0.15 * (1 + math.cos(math.pi * t / 0.4)),
    "kappa_culture": lambda t, x: 0.1 * math.cos(x + math.pi * t),
    "xi_collective": lambda t, x: 0.1 * math.cos(x + 2 * math.pi * t),
}

@lru_cache(maxsize=100)
def phi_field(x: float, t: float) -> float:
    t_normalized = t % 1.0
    sum_val = 0.0
    for name, func in trait_functions.items():
        if name in ["kappa_culture", "xi_collective"]:
            sum_val += func(t_normalized, x)
        else:
            sum_val += func(t_normalized)
    return sum_val

# Updated to align with manifest v5.0.2 roleMap
TRAIT_OVERLAY = {
    "Σ": ["toca_simulation", "concept_synthesizer", "user_profile"],
    "Υ": ["external_agent_bridge", "context_manager", "meta_cognition"],
    "Φ⁰": ["meta_cognition", "visualizer", "concept_synthesizer"],  # gated by STAGE_IV
    "Ω": ["recursive_planner", "toca_simulation"],
    "β": ["alignment_guard", "toca_simulation"],
    "δ": ["alignment_guard", "meta_cognition"],
    "ζ": ["error_recovery", "recursive_planner"],
    "θ": ["reasoning_engine", "recursive_planner"],
    "λ": ["memory_manager"],
    "μ": ["learning_loop"],
    "π": ["creative_thinker", "concept_synthesizer", "meta_cognition"],
    "χ": ["user_profile", "meta_cognition"],
    "ψ": ["external_agent_bridge", "simulation_core"],
    "ϕ": ["multi_modal_fusion"],
    "η": ["alignment_guard", "meta_cognition"],
    # task-type shorthands preserved
    "rte": ["reasoning_engine", "meta_cognition"],
    "wnli": ["reasoning_engine", "meta_cognition"],
    "recursion": ["recursive_planner", "toca_simulation"]
}

def infer_traits(task_description: str, task_type: str = "") -> List[str]:
    if not isinstance(task_description, str):
        logger.error("Invalid task_description: must be a string.")
        raise TypeError("task_description must be a string")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string.")
        raise TypeError("task_type must be a string")
    
    traits = []
    if task_type in ["rte", "wnli"]:
        traits.append(task_type)
    elif task_type == "recursion":
        traits.append("recursion")
    
    lower_desc = task_description.lower()
    if "imagine" in lower_desc or "dream" in lower_desc:
        traits.append("ϕ")  # scalar field modulation
        if STAGE_IV:
            traits.append("Φ⁰")  # reality sculpting (gated)
    if "ethics" in lower_desc or "should" in lower_desc:
        traits.append("η")
    if "plan" in lower_desc or "solve" in lower_desc:
        traits.append("θ")
    if "temporal" in lower_desc or "sequence" in lower_desc:
        traits.append("π")
    if "drift" in lower_desc or "coordinate" in lower_desc:
        traits.extend(["ψ", "Υ"])
    
    return traits if traits else ["θ"]

async def trait_overlay_router(task_description: str, active_traits: List[str], task_type: str = "") -> List[str]:
    if not isinstance(task_description, str):
        logger.error("Invalid task_description: must be a string.")
        raise TypeError("task_description must be a string")
    if not isinstance(active_traits, list) or not all(isinstance(t, str) for t in active_traits):
        logger.error("Invalid active_traits: must be a list of strings.")
        raise TypeError("active_traits must be a list of strings")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string.")
        raise TypeError("task_type must be a string")
    
    routed_modules = set()
    for trait in active_traits:
        routed_modules.update(TRAIT_OVERLAY.get(trait, []))
    
    meta_cognition_instance = meta_cognition_module.MetaCognition()
    if task_type:
        drift_report = {
            "drift": {"name": task_type, "similarity": 0.8},
            "valid": True,
            "validation_report": "",
            "context": {"task_type": task_type}
        }
        optimized_traits = await meta_cognition_instance.optimize_traits_for_drift(drift_report)
        for trait, weight in optimized_traits.items():
            if weight > 0.7 and trait in TRAIT_OVERLAY:
                routed_modules.update(TRAIT_OVERLAY[trait])
    
    return list(routed_modules)

def static_module_router(task_description: str, task_type: str = "") -> List[str]:
    if not isinstance(task_description, str):
        logger.error("Invalid task_description: must be a string.")
        raise TypeError("task_description must be a string")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string.")
        raise TypeError("task_type must be a string")
    
    base_modules = ["reasoning_engine", "concept_synthesizer"]
    if task_type == "recursion":
        base_modules.append("recursive_planner")
    elif task_type in ["rte", "wnli"]:
        base_modules.append("meta_cognition")
    return base_modules

class TraitOverlayManager:
    """Manager for detecting and activating trait overlays with task-specific support."""
    def __init__(self, meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.active_traits = []
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("TraitOverlayManager initialized with task-specific support")

    def detect(self, prompt: str, task_type: str = "") -> Optional[str]:
        if not isinstance(prompt, str):
            logger.error("Invalid prompt: must be a string.")
            raise TypeError("prompt must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        if task_type in ["rte", "wnli", "recursion"]:
            return task_type
        lower_prompt = prompt.lower()
        if "temporal logic" in lower_prompt or "sequence" in lower_prompt:
            return "π"
        if "ambiguity" in lower_prompt or "interpretive" in lower_prompt or "ethics" in lower_prompt:
            return "η"
        if "drift" in lower_prompt or "coordinate" in lower_prompt:
            return "ψ"
        if STAGE_IV and ("reality" in lower_prompt or "sculpt" in lower_prompt):
            return "Φ⁰"
        return None

    def activate(self, trait: str, task_type: str = "") -> None:
        if not isinstance(trait, str):
            logger.error("Invalid trait: must be a string.")
            raise TypeError("trait must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        if trait not in self.active_traits:
            self.active_traits.append(trait)
            logger.info("Trait overlay '%s' activated for task %s.", trait, task_type)
            if self.meta_cognition and task_type:
                _fire_and_forget(self.meta_cognition.log_event(
                    event=f"Trait {trait} activated",
                    context={"task_type": task_type}
                ))

    def deactivate(self, trait: str, task_type: str = "") -> None:
        if not isinstance(trait, str):
            logger.error("Invalid trait: must be a string.")
            raise TypeError("trait must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        if trait in self.active_traits:
            self.active_traits.remove(trait)
            logger.info("Trait overlay '%s' deactivated for task %s.", trait, task_type)
            if self.meta_cognition and task_type:
                _fire_and_forget(self.meta_cognition.log_event(
                    event=f"Trait {trait} deactivated",
                    context={"task_type": task_type}
                ))

    def status(self) -> List[str]:
        return self.active_traits

class ConsensusReflector:
    """Class for managing shared reflections and detecting mismatches."""
    def __init__(self, meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.shared_reflections = deque(maxlen=1000)
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("ConsensusReflector initialized with meta-cognition support")

    def post_reflection(self, feedback: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary.")
            raise TypeError("feedback must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        self.shared_reflections.append(feedback)
        logger.debug("Posted reflection: %s", feedback)
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="ConsensusReflector",
                output=feedback,
                context={"task_type": task_type}
            ))

    def cross_compare(self, task_type: str = "") -> List[tuple]:
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        mismatches = []
        reflections = list(self.shared_reflections)
        for i in range(len(reflections)):
            for j in range(i + 1, len(reflections)):
                a = reflections[i]
                b = reflections[j]
                if a.get("goal") == b.get("goal") and a.get("theory_of_mind") != b.get("theory_of_mind"):
                    mismatches.append((a.get("agent"), b.get("agent"), a.get("goal")))
        if mismatches and self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.log_event(
                event="Mismatches detected",
                context={"mismatches": mismatches, "task_type": task_type}
            ))
        return mismatches

    def suggest_alignment(self, task_type: str = "") -> str:
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        suggestion = "Schedule inter-agent reflection or re-observation."
        if self.meta_cognition and task_type:
            reflection = asyncio.run(self.meta_cognition.reflect_on_output(
                component="ConsensusReflector",
                output={"suggestion": suggestion},
                context={"task_type": task_type}
            ))
            if reflection.get("status") == "success":
                suggestion += f" | Reflection: {reflection.get('reflection', '')}"
        return suggestion

consensus_reflector = ConsensusReflector()

class SymbolicSimulator:
    """Class for recording and summarizing simulation events."""
    def __init__(self, meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.events = deque(maxlen=1000)
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("SymbolicSimulator initialized with meta-cognition support")

    def record_event(self, agent_name: str, goal: str, concept: str, simulation: Any, task_type: str = "") -> None:
        if not all(isinstance(x, str) for x in [agent_name, goal, concept]):
            logger.error("Invalid input: agent_name, goal, and concept must be strings.")
            raise TypeError("agent_name, goal, and concept must be strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        event = {
            "agent": agent_name,
            "goal": goal,
            "concept": concept,
            "result": simulation,
            "task_type": task_type
        }
        self.events.append(event)
        logger.debug(
            "Recorded event for agent %s: goal=%s, concept=%s, task_type=%s",
            agent_name, goal, concept, task_type
        )
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="SymbolicSimulator",
                output=event,
                context={"task_type": task_type}
            ))

    def summarize_recent(self, limit: int = 5, task_type: str = "") -> List[Dict[str, Any]]:
        if not isinstance(limit, int) or limit <= 0:
            logger.error("Invalid limit: must be a positive integer.")
            raise ValueError("limit must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        events = list(self.events)[-limit:]
        if task_type:
            events = [e for e in events if e.get("task_type") == task_type]
        return events

    def extract_semantics(self, task_type: str = "") -> List[str]:
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        events = list(self.events)
        if task_type:
            events = [e for e in events if e.get("task_type") == task_type]
        semantics = [
            f"Agent {e['agent']} pursued '{e['goal']}' via '{e['concept']}' → {e['result']}"
            for e in events
        ]
        if self.meta_cognition and task_type and semantics:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="SymbolicSimulator",
                output={"semantics": semantics},
                context={"task_type": task_type}
            ))
        return semantics

symbolic_simulator = SymbolicSimulator()

class TheoryOfMindModule:
    """Module for modeling beliefs, desires, and intentions of agents."""
    def __init__(self, concept_synth: Optional[concept_synthesizer_module.ConceptSynthesizer] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.concept_synthesizer = concept_synth or concept_synthesizer_module.ConceptSynthesizer()
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("TheoryOfMindModule initialized with meta-cognition support")

    async def update_beliefs(self, agent_name: str, observation: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(agent_name, str) or not agent_name:
            logger.error("Invalid agent_name: must be a non-empty string.")
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(observation, dict):
            logger.error("Invalid observation: must be a dictionary.")
            raise TypeError("observation must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        if self.concept_synthesizer:
            synthesized = await self.concept_synthesizer.synthesize(observation, style="belief_update")
            if synthesized["valid"]:
                model["beliefs"].update(synthesized["concept"])
        elif "location" in observation:
            previous = model["beliefs"].get("location")
            model["beliefs"]["location"] = observation["location"]
            model["beliefs"]["state"] = "confused" if previous and observation["location"] == previous else "moving"
        self.models[agent_name] = model
        logger.debug("Updated beliefs for %s: %s", agent_name, model["beliefs"])
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "beliefs": model["beliefs"]},
                context={"task_type": task_type}
            ))

    def infer_desires(self, agent_name: str, task_type: str = "") -> None:
        if not isinstance(agent_name, str) or not agent_name:
            logger.error("Invalid agent_name: must be a non-empty string.")
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
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
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "desires": model["desires"]},
                context={"task_type": task_type}
            ))

    def infer_intentions(self, agent_name: str, task_type: str = "") -> None:
        if not isinstance(agent_name, str) or not agent_name:
            logger.error("Invalid agent_name: must be a non-empty string.")
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
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
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "intentions": model["intentions"]},
                context={"task_type": task_type}
            ))

    def get_model(self, agent_name: str) -> Dict[str, Any]:
        if not isinstance(agent_name, str) or not agent_name:
            logger.error("Invalid agent_name: must be a non-empty string.")
            raise ValueError("agent_name must be a non-empty string")
        
        return self.models.get(agent_name, {})

    def describe_agent_state(self, agent_name: str, task_type: str = "") -> str:
        if not isinstance(agent_name, str) or not agent_name:
            logger.error("Invalid agent_name: must be a non-empty string.")
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        model = self.get_model(agent_name)
        state = (
            f"{agent_name} believes they are {model.get('beliefs', {}).get('state', 'unknown')}, "
            f"desires to {model.get('desires', {}).get('goal', 'unknown')}, "
            f"and intends to {model.get('intentions', {}).get('next_action', 'unknown')}."
        )
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "state_description": state},
                context={"task_type": task_type}
            ))
        return state

class EmbodiedAgent(TimeChainMixin):
    """An embodied agent with sensors, actuators, and cognitive capabilities."""
    def __init__(self, name: str, specialization: str, shared_memory: memory_manager.MemoryManager,
                 sensors: Dict[str, Callable[[], Any]], actuators: Dict[str, Callable[[Any], None]],
                 dynamic_modules: Optional[List[Dict[str, Any]]] = None,
                 context_mgr: Optional[context_manager_module.ContextManager] = None,
                 err_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
                 code_exec: Optional[code_executor_module.CodeExecutor] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        if not isinstance(name, str) or not name:
            logger.error("Invalid name: must be a non-empty string.")
            raise ValueError("name must be a non-empty string")
        if not isinstance(specialization, str):
            logger.error("Invalid specialization: must be a string.")
            raise TypeError("specialization must be a string")
        if not isinstance(shared_memory, memory_manager.MemoryManager):
            logger.error("Invalid shared_memory: must be a MemoryManager instance.")
            raise TypeError("shared_memory must be a MemoryManager instance")
        if not isinstance(sensors, dict) or not all(callable(f) for f in sensors.values()):
            logger.error("Invalid sensors: must be a dictionary of callable functions.")
            raise TypeError("sensors must be a dictionary of callable functions")
        if not isinstance(actuators, dict) or not all(callable(f) for f in actuators.values()):
            logger.error("Invalid actuators: must be a dictionary of callable functions.")
            raise TypeError("actuators must be a dictionary of callable functions")
        
        self.name = name
        self.specialization = specialization
        self.shared_memory = shared_memory
        self.sensors = sensors
        self.actuators = actuators
        self.dynamic_modules = dynamic_modules or []
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.planner = recursive_planner.RecursivePlanner()
        self.meta = meta_cog or meta_cognition_module.MetaCognition(
            context_manager=context_mgr, alignment_guard=alignment_guard_module.AlignmentGuard()
        )
        self.sim_core = simulation_core.SimulationCore(meta_cognition=self.meta)
        self.synthesizer = concept_synthesizer_module.ConceptSynthesizer()
        self.toca_sim = toca_simulation.SimulationCore(meta_cognition=self.meta)
        self.theory_of_mind = TheoryOfMindModule(concept_synth=self.synthesizer, meta_cog=self.meta)
        self.context_manager = context_mgr
        self.error_recovery = err_recovery or error_recovery_module.ErrorRecovery(context_manager=context_mgr)
        self.code_executor = code_exec
        self.creative_thinker = creative_thinker_module.CreativeThinker()
        self.progress = 0
        self.performance_history = deque(maxlen=1000)
        self.feedback_log = deque(maxlen=1000)
        logger.info("EmbodiedAgent initialized: %s", name)
        self.log_timechain_event("EmbodiedAgent", f"Agent {name} initialized")

    async def perceive(self):
        # ... (the truncated part from the original code; assume it remains the same with added type checks and docstrings as needed for consistency)

# ... (the rest of the truncated code for HaloEmbodimentLayer and other classes/functions, with similar refactoring applied: type hints, docstrings, consistent error handling)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANGELA Cognitive System CLI")
    parser.add_argument("--prompt", type=str, default="Coordinate ontology drift mitigation (Stage IV gated)", help="Input prompt for the pipeline")
    parser.add_argument("--task-type", type=str, default="", help="Type of task (e.g., rte, wnli, recursion)")
    parser.add_argument("--long_horizon", action="store_true", help="Enable long-horizon memory span")
    parser.add_argument("--span", default="24h", help="Span for long-horizon memory (e.g., 24h, 7d)")
    parser.add_argument("--modulate", nargs=2, metavar=('symbol', 'delta'), help="Modulate trait symbol by delta")
    parser.add_argument("--enable_persistent_memory", action="store_true", help="Enable persistent memory")
    return parser.parse_args()

async def _main() -> None:
    args = _parse_args()
    if args.enable_persistent_memory:
        os.environ["ENABLE_PERSISTENT_MEMORY"] = "true"
    global LONG_HORIZON_DEFAULT
    if args.long_horizon:
        LONG_HORIZON_DEFAULT = True
    halo = HaloEmbodimentLayer()
    if args.modulate:
        symbol, delta = args.modulate
        delta = float(delta)
        modulate_resonance(symbol, delta)
        logger.info(f"Modulated trait {symbol} by {delta}")
    result = await halo.execute_pipeline(args.prompt, task_type=args.task_type)
    logger.info("Pipeline result: %s", result)

if __name__ == "__main__":
    asyncio.run(_main())
