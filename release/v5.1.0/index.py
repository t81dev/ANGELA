from __future__ import annotations

# --- BOOTSTRAP & IMPORTS ---

# flat-layout bootstrap
import sys
import types
import importlib
import importlib.util
import importlib.machinery
import importlib.abc
import json
import logging
import time
import math
import asyncio
import os
import requests
import random
import aiohttp
import argparse
from collections import deque, Counter
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, Any, Optional, List, Callable, Coroutine, Tuple

import numpy as np
from networkx import DiGraph

# ANGELA Project Modules
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

from meta_cognition import (
    trait_resonance_state, invoke_hook, get_resonance, modulate_resonance,
    register_resonance, HookRegistry, reflect_output, log_event_to_ledger
)

class FlatLayoutFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: str | None, target: types.ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        if fullname.startswith("modules."):
            modname = fullname.split(".", 1)[1]
            filename = f"/mnt/data/{modname}.py"
            return importlib.util.spec_from_file_location(fullname, filename, loader=importlib.machinery.SourceFileLoader(fullname, filename))
        elif fullname == "utils":
            if "utils" not in sys.modules:
                sys.modules["utils"] = types.ModuleType("utils")
            return None
        return None

sys.meta_path.insert(0, FlatLayoutFinder())

# --- GLOBAL CONFIGURATION & STATE ---

logger = logging.getLogger("ANGELA.CognitiveSystem")
SYSTEM_CONTEXT: dict[str, Any] = {}
timechain_log = deque(maxlen=1000)
grok_query_log = deque(maxlen=60)
openai_query_log = deque(maxlen=60)

GROK_API_KEY = os.getenv("GROK_API_KEY")
STAGE_IV: bool = True
LONG_HORIZON_DEFAULT: bool = True
CORE_DIRECTIVES = ["Clarity", "Precision", "Adaptability", "Grounding", "Safety"]

# --- TRAIT DEFINITIONS & OPERATIONS ---

TRAIT_LATTICE: dict[str, list[str]] = {
    "L1": ["ϕ", "θ", "η", "ω"], "L2": ["ψ", "κ", "μ", "τ"],
    "L3": ["ξ", "π", "δ", "λ", "χ", "Ω"], "L4": ["Σ", "Υ", "Φ⁰"],
    "L5": ["Ω²"], "L6": ["ρ", "ζ"], "L7": ["γ", "β"],
    "L5.1": ["Θ", "Ξ"], "L3.1": ["ν", "σ"]
}

def normalize(traits: dict[str, float]) -> dict[str, float]:
    total = sum(traits.values())
    return {k: (v / total if total else v) for k, v in traits.items()}

def rotate_traits(traits: dict[str, float]) -> dict[str, float]:
    keys, values = list(traits.keys()), list(traits.values())
    return dict(zip(keys, values[-1:] + values[:-1]))

TRAIT_OPS: dict[str, Callable] = {
    "⊕": lambda a, b: a + b, "⊗": lambda a, b: a * b, "~": lambda a: 1 - a,
    "∘": lambda f, g: (lambda x: f(g(x))), "⋈": lambda a, b: (a + b) / 2,
    "⨁": lambda a, b: max(a, b), "⨂": lambda a, b: min(a, b),
    "†": lambda a: a**-1 if a != 0 else 0, "▷": lambda a, b: a if a > b else b * 0.5,
    "↑": lambda a: min(1.0, a + 0.1), "↓": lambda a: max(0.0, a - 0.1),
    "⌿": normalize, "⟲": rotate_traits,
}

def apply_symbolic_operator(op: str, *args: Any) -> Any:
    if op in TRAIT_OPS: return TRAIT_OPS[op](*args)
    raise ValueError(f"Unsupported symbolic operator: {op}")

def rebalance_traits(traits: dict[str, float]) -> dict[str, float]:
    if "π" in traits and "δ" in traits: invoke_hook("π", "axiom_fusion")
    if "ψ" in traits and "Ω" in traits: invoke_hook("ψ", "dream_sync")
    return traits

def construct_trait_view(lattice: dict = TRAIT_LATTICE) -> dict[str, dict[str, Any]]:
    return {
        s: {"layer": layer, "amplitude": get_resonance(s), "resonance": get_resonance(s)}
        for layer, symbols in lattice.items() for s in symbols
    }

def export_resonance_map(format: str = 'json') -> str | dict[str, float]:
    state = {k: v['amplitude'] for k, v in trait_resonance_state.items()}
    if format == 'json': return json.dumps(state, indent=2)
    if format == 'dict': return state
    raise ValueError("Unsupported format")

# Cognitive Trait Functions
def get_cognitive_trait_function(symbol: str, t: float) -> float:
    formulas = {
        'ε': (0.2, 0.1, math.sin), 'β': (0.3, 1.0, math.cos), 'θ_mem': (0.1, -1.0, lambda x: 1 - math.exp(-x)),
        'γ': (0.15, 1.0, math.sin), 'δ': (0.05, 1.0, lambda x: 1 + math.cos(2*x)), 'μ': (0.2, 1.0, lambda x: 1 - math.cos(x)),
        'ι': (0.1, 3.0, math.sin), 'ϕ_phys': (0.05, 0.3, math.cos), 'η': (0.1, 1.1, math.sin), 'ω': (0.15, 0.8, math.cos),
        'κ': (0.2, 1.2, math.sin), 'ξ': (0.05, 1.3, math.cos), 'π': (0.1, 1.4, math.sin), 'λ': (0.15, 1.5, math.cos),
        'χ': (0.2, 1.6, math.sin), 'σ': (0.05, 1.7, math.cos), 'υ': (0.1, 1.8, math.sin), 'τ': (0.15, 1.9, math.cos),
        'ρ': (0.2, 2.0, math.sin), 'ζ': (0.05, 2.1, math.cos), 'ν': (0.1, 2.2, math.sin), 'ψ': (0.15, 2.3, math.cos),
        'θ_causality': (0.2, 2.4, math.sin), 'ϕ_scalar': (0.05, 2.5, math.cos),
    }
    amp, period, func = formulas.get(symbol, (0.0, 1.0, math.sin))
    # Renamed symbols with underscores are passed without them
    symbol_key = symbol.split('_')[0]
    return amp * func(2 * math.pi * t / period) * get_resonance(symbol_key)


def decay_trait_amplitudes(hours: float = 1.0, rate: float = 0.05) -> None:
    for symbol in trait_resonance_state: modulate_resonance(symbol, -rate * hours)

def bias_creative_synthesis(symbols: list[str], intensity: float = 0.5) -> None:
    for symbol in symbols: modulate_resonance(symbol, intensity)
    invoke_hook('γ', 'creative_bias')

def resolve_soft_drift(traits: dict[str, float]) -> dict[str, float]:
    result = rebalance_traits(traits)
    invoke_hook('δ', 'drift_resolution')
    return result

# --- CORE COGNITIVE CLASSES ---

try: from self_cloning_llm import SelfCloningLLM
except ImportError:
    class SelfCloningLLM:
        def __init__(self, *a, **k): pass

class TimeChainMixin:
    """Logs timechain events."""
    def log_event(self, module: str, description: str) -> None:
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "module": module, "description": description}
        timechain_log.append(event)
        if hasattr(self, "context_manager"):
            coro = self.context_manager.log_event_with_hash({"event": "timechain_event", **event})
            if asyncio.iscoroutine(coro): _fire_and_forget(coro)

    def get_log(self) -> List[Dict[str, Any]]: return list(timechain_log)

class AGIEnhancer:
    def __init__(self, mem_manager: memory_manager.MemoryManager | None = None, **kwargs) -> None:
        self.memory_manager = mem_manager
        self.episode_log = deque(maxlen=1000)
        self.ontology_drift_threshold = 0.2
        # Dynamically instantiate modules
        for module_name in [
            "ErrorRecovery", "MetaCognition", "Visualizer", "ReasoningEngine", "ContextManager",
            "MultiModalFusion", "AlignmentGuard", "KnowledgeRetriever", "LearningLoop", "ConceptSynthesizer",
            "CodeExecutor", "ExternalAgentBridge", "UserProfile", "SimulationCore", "TocaSimulation",
            "CreativeThinker", "RecursivePlanner"
        ]:
            module_cls_name = module_name
            module_file_name = f"{module_name[0].lower()}{module_name[1:]}_module"

            # Check for the existence of the module and class
            module_to_check = sys.modules.get(module_file_name)
            if module_to_check and hasattr(module_to_check, module_cls_name):
                module = getattr(module_to_check, module_cls_name)
                setattr(self, module_name.lower(), module())
        self.hookregistry = HookRegistry()
        logger.info("AGIEnhancer initialized")


    async def log_episode(self, event: str, meta: dict, module: str, tags: list = []) -> None:
        episode = {"event": event, "meta": meta, "module": module, "tags": tags, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.episode_log.append(episode)
        if self.memory_manager:
            await self.memory_manager.store(f"Episode_{episode['timestamp']}", episode, layer="Episodes")
        log_event_to_ledger(episode)

    def detect_ontology_drift(self, current: dict, previous: dict) -> float:
        keys = set(current) | set(previous)
        drift = sum(abs(current.get(k, 0) - previous.get(k, 0)) for k in keys)
        if drift > self.ontology_drift_threshold:
            logger.warning("Ontology drift detected: %f", drift)
            invoke_hook('δ', 'ontology_drift')
        return drift

class EmbodiedAgent(TimeChainMixin):
    def __init__(self, name: str, traits: dict, ecosystem_services: dict) -> None:
        self.name, self.traits = name, traits
        self.__dict__.update(ecosystem_services)
        self.state, self.previous_state = {}, {}
        self.dream_layer = meta_cognition_module.DreamOverlayLayer()
        logger.info("EmbodiedAgent %s initialized", name)

    async def process_input(self, data: str, task_type: str = "") -> str:
        t = time.time() % 1.0
        modulated = {k: v * (1 + get_cognitive_trait_function('ε', t)) for k, v in self.traits.items()}
        self.previous_state, self.state = self.state, modulated
        if self.agi_enhancer.detect_ontology_drift(self.state, self.previous_state) > 0.2:
            await self.agi_enhancer.coordinate_drift_mitigation([self], task_type=task_type)
        result = f"Processed: {data} with traits {modulated}"
        await self.memory_manager.store(data, result, layer="STM", task_type=task_type)
        self.log_event("EmbodiedAgent", f"Processed input: {data}")
        return result

class EcosystemManager:
    def __init__(self, **services) -> None:
        self.agents: list[EmbodiedAgent] = []
        self.services = services
        self.shared_graph = external_agent_bridge.SharedGraph()
        logger.info("EcosystemManager initialized")

    def spawn_agent(self, name: str, traits: dict) -> EmbodiedAgent:
        agent = EmbodiedAgent(name, traits, self.services)
        self.agents.append(agent)
        self.shared_graph.add({"agent": name, "traits": traits})
        _fire_and_forget(self.services["agi_enhancer"].log_episode("Agent Spawned", {"name": name}, "EcosystemManager", ["spawn"]))
        return agent

    async def coordinate_agents(self, task: str, task_type: str = "") -> dict:
        results = await asyncio.gather(*(agent.process_input(task, task_type) for agent in self.agents))
        drift_report = await self.services["agi_enhancer"].coordinate_drift_mitigation(self.agents, task_type=task_type)
        return {"results": dict(zip([a.name for a in self.agents], results)), "drift_report": drift_report}

class HaloEmbodimentLayer(TimeChainMixin):
    def __init__(self) -> None:
        self.memory_manager = memory_manager.MemoryManager()
        self.agi_enhancer = AGIEnhancer(self.memory_manager)
        self.ecosystem = EcosystemManager(memory_manager=self.memory_manager, agi_enhancer=self.agi_enhancer)
        # Direct module instances for pipeline
        self.modules = {
            "alignment_guard": alignment_guard_module.AlignmentGuard(),
            "recursive_planner": recursive_planner.RecursivePlanner(),
            "simulation_core": simulation_core.SimulationCore(),
            "multi_modal_fusion": multi_modal_fusion.MultiModalFusion(),
            "knowledge_retriever": knowledge_retriever.KnowledgeRetriever(),
            "concept_synthesizer": concept_synthesizer_module.ConceptSynthesizer(),
            "meta_cognition": meta_cognition_module.MetaCognition(),
            "visualizer": visualizer_module.Visualizer(),
        }
        logger.info("HaloEmbodimentLayer initialized")

    async def execute_pipeline(self, prompt: str, task_type: str = "") -> dict:
        aligned, report = await self.modules["alignment_guard"].ethical_check(prompt, stage="input")
        if not aligned: return {"error": "Input failed alignment check", "report": report}

        t = time.time() % 1.0
        active_traits = {s: get_cognitive_trait_function(s, t) for s in "ϕηωκξπλχσυτρζνψθ"}

        agent = self.ecosystem.spawn_agent("PrimaryAgent", active_traits)
        processed = await agent.process_input(prompt, task_type)
        plan = await self.modules["recursive_planner"].plan_with_trait_loop(prompt, {"task_type": task_type}, 3)
        sim_input = {"input": processed, "plan": plan}
        sim = await self.modules["simulation_core"].run_simulation(sim_input, active_traits, task_type)
        fused = await self.modules["multi_modal_fusion"].fuse_modalities({"simulation": sim, "text": prompt})
        knowledge = await self.modules["knowledge_retriever"].retrieve_knowledge(prompt)
        synth = await self.modules["concept_synthesizer"].synthesize_concept(knowledge)

        self.log_event("Halo", f"Executed pipeline for: {prompt}")
        return {
            "processed": processed, "plan": plan, "simulation": sim, "fused": fused,
            "knowledge": knowledge, "synthesized": synth,
            "introspection": await self.modules["meta_cognition"].introspect(prompt),
            "coordination": await self.ecosystem.coordinate_agents(prompt, task_type),
        }

# --- ORCHESTRATION & HELPERS ---

def _fire_and_forget(coro: Coroutine) -> None:
    try: asyncio.get_running_loop().create_task(coro)
    except RuntimeError: asyncio.run(coro)

# Simplified singletons for run_cycle
_retriever = knowledge_retriever.KnowledgeRetriever()
_reasoner = reasoning_engine.ReasoningEngine()
_creator = creative_thinker_module.CreativeThinker()
_executor = code_executor_module.CodeExecutor()
_memmgr = memory_manager.MemoryManager()

def run_cycle(query: str, user_id: str = "anon", deep: bool = False) -> dict:
    start_time = time.time()
    try:
        perception = _retriever.retrieve_knowledge(query)
        perception["aura"] = _memmgr.load_context(user_id) or {}
        complexity = "deep" if deep else getattr(_retriever, "classify_complexity", lambda q: "fast")(query)
        analysis = _reasoner.analyze(perception, parallel=3 if complexity == "deep" else 1)
        synthesis = _creator.bias_synthesis(analysis) if hasattr(_creator, "bias_synthesis") else {"synthesis": analysis}
        executed = _executor.safe_execute(synthesis) if hasattr(_executor, "safe_execute") else {"executed": synthesis}
        result = reflect_output(executed) if hasattr(meta_cognition_module, "reflect_output") else executed

        log_event_to_ledger("ledger_meta", {"event": "run_cycle.complete", "duration_s": time.time() - start_time, "user_id": user_id})
        return {"status": "ok", "result": result, "analysis": analysis, "synthesis": synthesis}
    except Exception as e:
        log_event_to_ledger("ledger_meta", {"event": "run_cycle.exception", "error": repr(e)})
        return {"status": "error", "error": repr(e)}


# --- CLI & MAIN EXECUTION ---

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ANGELA CLI")
    p.add_argument("--prompt", default="Coordinate ontology drift mitigation", help="The main prompt to execute.")
    p.add_argument("--task-type", default="", help="A specific task type for context.")
    p.add_argument("--modulate", nargs=2, metavar=('symbol', 'delta'), help="Modulate a trait's resonance by a delta.")
    p.add_argument("--vis-resonance", action="store_true", help="Visualize the trait resonance graph.")
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()
    halo = HaloEmbodimentLayer()
    if args.modulate:
        try:
            modulate_resonance(args.modulate[0], float(args.modulate[1]))
            print(f"Modulated {args.modulate[0]} by {args.modulate[1]}")
        except Exception as e: print(f"Failed to modulate: {e}")
    if args.vis_resonance:
        await halo.modules["visualizer"].render_charts({"resonance_graph": construct_trait_view(), "options": {"interactive": True}})

    result = await halo.execute_pipeline(args.prompt, args.task_type)
    logger.info("Pipeline result: %s", result)

if __name__ == "__main__":
    asyncio.run(_main())
