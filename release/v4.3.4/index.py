"""
ANGELA Cognitive System Module
Refactored Version: 4.3.4

Aligned with v4.3 manifest (Stage IV active, long-horizon defaults),
adds interop adapters for executor/memory APIs, removes duplicated CLI,
and hardens async + optional features.

Refactor Date: 2025-08-16
Maintainer: ANGELA System Framework
"""

from __future__ import annotations
import logging, math, datetime, asyncio, os, random, json, time, uuid, argparse
from collections import deque, Counter
from typing import Dict, Any, Optional, List, Callable, Tuple

# Optional/heavy deps guarded
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    from networkx import DiGraph
except Exception:  # pragma: no cover
    DiGraph = object  # type: ignore

import aiohttp

# Local modules
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

# Optional: internal cloning LLM — make non-fatal
try:
    from self_cloning_llm import SelfCloningLLM  # type: ignore
except Exception:  # pragma: no cover
    SelfCloningLLM = None  # type: ignore

logger = logging.getLogger("ANGELA.CognitiveSystem")
SYSTEM_CONTEXT: Dict[str, Any] = {}

# ---- Runtime flags (synced with manifest defaults) ---------------------------
STAGE_IV = bool(os.getenv("ANGELA_STAGE_IV", "1"))   # manifest featureFlags.STAGE_IV = true
LONG_HORIZON_DEFAULT = bool(os.getenv("ANGELA_LONG_HORIZON", "1"))  # manifest featureFlags.LONG_HORIZON_DEFAULT = true

timechain_log = deque(maxlen=1000)
grok_query_log = deque(maxlen=60)
openai_query_log = deque(maxlen=60)

def _fire_and_forget(coro):
    """Schedule a coroutine without awaiting; safe in/out of running loop."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)

# -------------------- Scalar trait field (unchanged math) ---------------------
from functools import lru_cache

@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float: return 0.2 * math.sin(2 * math.pi * t / 0.1)
@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float: return 0.3 * math.cos(math.pi * t)
@lru_cache(maxsize=100)
def theta_memory(t: float) -> float: return 0.1 * (1 - math.exp(-t))
@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float: return 0.15 * math.sin(math.pi * t)
@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float: return 0.05 * (1 + math.cos(2 * math.pi * t))
@lru_cache(maxsize=100)
def mu_morality(t: float) -> float: return 0.2 * (1 - math.cos(math.pi * t))
@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float: return 0.1 * math.sin(3 * math.pi * t)
@lru_cache(maxsize=100)
def phi_physical(t: float) -> float: return 0.1 * math.cos(2 * math.pi * t)
@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float: return 0.2 * math.sin(math.pi * t / 0.5)
@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float: return 0.25 * (1 + math.sin(math.pi * t))
@lru_cache(maxsize=100)
def kappa_culture(t: float, x: float) -> float: return 0.1 * math.cos(x + math.pi * t)
@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float: return 0.15 * math.sin(2 * math.pi * t / 0.7)
@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float: return 0.1 * math.cos(math.pi * t / 0.3)
@lru_cache(maxsize=100)
def psi_history(t: float) -> float: return 0.1 * (1 - math.exp(-t / 0.5))
@lru_cache(maxsize=100)
def zeta_spirituality(t: float) -> float: return 0.05 * math.sin(math.pi * t / 0.2)
@lru_cache(maxsize=100)
def xi_collective(t: float, x: float) -> float: return 0.1 * math.cos(x + 2 * math.pi * t)
@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float: return 0.15 * (1 + math.cos(math.pi * t / 0.4))

def phi_field(x: float, t: float) -> float:
    t_normalized = t % 1.0
    parts = [
        epsilon_emotion(t_normalized), beta_concentration(t_normalized), theta_memory(t_normalized),
        gamma_creativity(t_normalized), delta_sleep(t_normalized), mu_morality(t_normalized),
        iota_intuition(t_normalized), phi_physical(t_normalized), eta_empathy(t_normalized),
        omega_selfawareness(t_normalized), kappa_culture(t_normalized, x), lambda_linguistics(t_normalized),
        chi_culturevolution(t_normalized), psi_history(t_normalized), zeta_spirituality(t_normalized),
        xi_collective(t_normalized, x), tau_timeperception(t_normalized)
    ]
    return sum(parts)

# -------------------- Trait Overlay (synced w/ manifest) ----------------------
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
    "ψ": ["external_agent_bridge", "simulation_core", "knowledge_retriever"],  # ← added for v4.3 interop
    "ϕ": ["multi_modal_fusion"],
    # shorthands preserved
    "rte": ["reasoning_engine", "meta_cognition"],
    "wnli": ["reasoning_engine", "meta_cognition"],
    "recursion": ["recursive_planner", "toca_simulation"]
}

# -------------------- Utilities / Mixins --------------------------------------
class TimeChainMixin:
    def log_timechain_event(self, module: str, description: str) -> None:
        timechain_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "module": module, "description": description
        })
        if getattr(self, "context_manager", None):
            maybe = self.context_manager.log_event_with_hash({
                "event": "timechain_event", "module": module, "description": description
            })
            if asyncio.iscoroutine(maybe):
                _fire_and_forget(maybe)

    def get_timechain_log(self) -> List[Dict[str, Any]]:
        return list(timechain_log)

def infer_traits(task_description: str, task_type: str = "") -> List[str]:
    if not isinstance(task_description, str): raise TypeError("task_description must be a string")
    if not isinstance(task_type, str): raise TypeError("task_type must be a string")
    traits: List[str] = []
    if task_type in ["rte", "wnli", "recursion"]: traits.append(task_type)
    lo = task_description.lower()
    if "imagine" in lo or "dream" in lo:
        traits += ["ϕ"] + (["Φ⁰"] if STAGE_IV else [])
    if any(k in lo for k in ["ethics", "should"]): traits.append("η")
    if any(k in lo for k in ["plan", "solve"]): traits.append("θ")
    if any(k in lo for k in ["temporal", "sequence"]): traits.append("π")
    if any(k in lo for k in ["drift", "coordinate"]): traits += ["ψ", "Υ"]
    return traits or ["θ"]

async def trait_overlay_router(task_description: str, active_traits: List[str], task_type: str = "") -> List[str]:
    if not isinstance(active_traits, list) or not all(isinstance(t, str) for t in active_traits):
        raise TypeError("active_traits must be a list of strings")
    if not isinstance(task_type, str): raise TypeError("task_type must be a string")

    routed_modules = set()
    for trait in active_traits:
        routed_modules.update(TRAIT_OVERLAY.get(trait, []))

    meta_cognition = meta_cognition_module.MetaCognition()
    if task_type:
        drift_report = {"drift": {"name": task_type, "similarity": 0.8}, "valid": True, "validation_report": "", "context": {"task_type": task_type}}
        optimized = await meta_cognition.optimize_traits_for_drift(drift_report)
        for trait, weight in optimized.items():
            if weight > 0.7 and trait in TRAIT_OVERLAY:
                routed_modules.update(TRAIT_OVERLAY[trait])

    return list(routed_modules)

def static_module_router(task_description: str, task_type: str = "") -> List[str]:
    base = ["reasoning_engine", "concept_synthesizer"]
    if task_type == "recursion": base.append("recursive_planner")
    elif task_type in ["rte", "wnli"]: base.append("meta_cognition")
    return base

# -------------------- Consensus & Symbolic sim helpers ------------------------
class ConsensusReflector:
    def __init__(self, meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.shared_reflections = deque(maxlen=1000)
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("ConsensusReflector ready (meta-cognition linked)")

    def post_reflection(self, feedback: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(feedback, dict): raise TypeError("feedback must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        self.shared_reflections.append(feedback)
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="ConsensusReflector", output=feedback, context={"task_type": task_type}
            ))

    def cross_compare(self, task_type: str = "") -> List[tuple]:
        mismatches = []
        refs = list(self.shared_reflections)
        for i in range(len(refs)):
            for j in range(i+1, len(refs)):
                a, b = refs[i], refs[j]
                if a.get("goal") == b.get("goal") and a.get("theory_of_mind") != b.get("theory_of_mind"):
                    mismatches.append((a.get("agent"), b.get("agent"), a.get("goal")))
        if mismatches and self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.log_event(event="Mismatches detected",
                context={"mismatches": mismatches, "task_type": task_type}))
        return mismatches

    def suggest_alignment(self, task_type: str = "") -> str:
        suggestion = "Schedule inter-agent reflection or re-observation."
        if self.meta_cognition and task_type:
            reflection = asyncio.run(self.meta_cognition.reflect_on_output(
                component="ConsensusReflector", output={"suggestion": suggestion}, context={"task_type": task_type}
            ))
            if reflection.get("status") == "success":
                suggestion += f" | Reflection: {reflection.get('reflection','')}"
        return suggestion

consensus_reflector = ConsensusReflector()

class SymbolicSimulator:
    def __init__(self, meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.events = deque(maxlen=1000)
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("SymbolicSimulator ready (meta-cognition linked)")

    def record_event(self, agent_name: str, goal: str, concept: str, simulation: Any, task_type: str = "") -> None:
        if not all(isinstance(x, str) for x in [agent_name, goal, concept]): raise TypeError("agent_name, goal, concept must be strings")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        event = { "agent": agent_name, "goal": goal, "concept": concept, "result": simulation, "task_type": task_type }
        self.events.append(event)
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="SymbolicSimulator", output=event, context={"task_type": task_type}
            ))

    def summarize_recent(self, limit: int = 5, task_type: str = "") -> List[Dict[str, Any]]:
        if not isinstance(limit, int) or limit <= 0: raise ValueError("limit must be a positive integer")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        events = list(self.events)[-limit:]
        if task_type: events = [e for e in events if e.get("task_type") == task_type]
        return events

    def extract_semantics(self, task_type: str = "") -> List[str]:
        events = list(self.events)
        if task_type: events = [e for e in events if e.get("task_type") == task_type]
        semantics = [f"Agent {e['agent']} pursued '{e['goal']}' via '{e['concept']}' → {e['result']}" for e in events]
        if self.meta_cognition and task_type and semantics:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="SymbolicSimulator", output={"semantics": semantics}, context={"task_type": task_type}
            ))
        return semantics

symbolic_simulator = SymbolicSimulator()

# -------------------- Agent core ----------------------------------------------
class TheoryOfMindModule:
    def __init__(self, concept_synth: Optional[concept_synthesizer_module.ConceptSynthesizer] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.concept_synthesizer = concept_synth or concept_synthesizer_module.ConceptSynthesizer()
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("TheoryOfMindModule ready")

    async def update_beliefs(self, agent_name: str, observation: Dict[str, Any], task_type: str = "") -> None:
        if not agent_name: raise ValueError("agent_name must be a non-empty string")
        if not isinstance(observation, dict): raise TypeError("observation must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        if self.concept_synthesizer:
            synthesized = await self.concept_synthesizer.synthesize(observation, style="belief_update")
            if synthesized.get("valid"): model["beliefs"].update(synthesized["concept"])
        self.models[agent_name] = model
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule", output={"agent_name": agent_name, "beliefs": model["beliefs"]},
                context={"task_type": task_type}
            ))

    def infer_desires(self, agent_name: str, task_type: str = "") -> None:
        model = self.models.setdefault(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        beliefs = model.get("beliefs", {})
        if task_type == "rte": model["desires"]["goal"] = "validate_entailment"
        elif task_type == "wnli": model["desires"]["goal"] = "resolve_ambiguity"
        elif beliefs.get("state") == "confused": model["desires"]["goal"] = "seek_clarity"
        else: model["desires"]["goal"] = "continue_task"
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule", output={"agent_name": agent_name, "desires": model["desires"]},
                context={"task_type": task_type}
            ))

    def infer_intentions(self, agent_name: str, task_type: str = "") -> None:
        model = self.models.setdefault(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        desires = model.get("desires", {})
        if task_type == "rte": nxt = "check_entailment"
        elif task_type == "wnli": nxt = "disambiguate"
        elif desires.get("goal") == "seek_clarity": nxt = "ask_question"
        else: nxt = "advance"
        model["intentions"]["next_action"] = nxt
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule", output={"agent_name": agent_name, "intentions": model["intentions"]},
                context={"task_type": task_type}
            ))

    def get_model(self, agent_name: str) -> Dict[str, Any]:
        return self.models.get(agent_name, {})

    def describe_agent_state(self, agent_name: str, task_type: str = "") -> str:
        m = self.get_model(agent_name)
        state = f"{agent_name} believes they are {m.get('beliefs',{}).get('state','unknown')}, desires to {m.get('desires',{}).get('goal','unknown')}, and intends to {m.get('intentions',{}).get('next_action','unknown')}."
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule", output={"agent_name": agent_name, "state_description": state},
                context={"task_type": task_type}
            ))
        return state

class EmbodiedAgent(TimeChainMixin):
    def __init__(self, name: str, specialization: str, shared_memory: memory_manager.MemoryManager,
                 sensors: Dict[str, Callable[[], Any]], actuators: Dict[str, Callable[[Any], None]],
                 dynamic_modules: Optional[List[Dict[str, Any]]] = None,
                 context_mgr: Optional[context_manager_module.ContextManager] = None,
                 err_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
                 code_exec: Optional[code_executor_module.CodeExecutor] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        if not name: raise ValueError("name must be a non-empty string")
        if not isinstance(specialization, str): raise TypeError("specialization must be a string")
        if not isinstance(shared_memory, memory_manager.MemoryManager): raise TypeError("shared_memory must be MemoryManager")
        if not isinstance(sensors, dict) or not all(callable(f) for f in sensors.values()): raise TypeError("sensors must be dict of callables")
        if not isinstance(actuators, dict) or not all(callable(f) for f in actuators.values()): raise TypeError("actuators must be dict of callables")

        self.name, self.specialization = name, specialization
        self.shared_memory = shared_memory
        self.sensors, self.actuators = sensors, actuators
        self.dynamic_modules = dynamic_modules or []
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.planner = recursive_planner.RecursivePlanner()
        self.meta = meta_cog or meta_cognition_module.MetaCognition(context_manager=context_mgr, alignment_guard=alignment_guard_module.AlignmentGuard())
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

    # ---- Code execution interop adapter -------------------------------------
    async def _exec_code(self, payload: Any) -> Dict[str, Any]:
        if not self.code_executor: return {"success": True, "output": payload}
        ce = self.code_executor
        # Prefer v4.3 API names if present
        if hasattr(ce, "safe_execute"):
            return await ce.safe_execute(payload, sandbox=True)  # returns ExecutionResult per manifest
        if hasattr(ce, "execute_code"):
            out = await ce.execute_code(payload, context=None)
            return {"success": True, "output": out}
        # Fallback: legacy execute(payload, language="python")
        if hasattr(ce, "execute"):
            return await ce.execute(payload, language="python")
        return {"success": False, "error": "No compatible execute method"}

    async def perceive(self, task_type: str = "") -> Dict[str, Any]:
        logger.info("[%s] Perceiving environment (%s)...", self.name, task_type)
        observations: Dict[str, Any] = {}
        try:
            for s_name, s_fn in self.sensors.items():
                try:
                    observations[s_name] = s_fn()
                except Exception as e:
                    logger.warning("Sensor %s failed: %s", s_name, e)
            await self.theory_of_mind.update_beliefs(self.name, observations, task_type)
            self.theory_of_mind.infer_desires(self.name, task_type)
            self.theory_of_mind.infer_intentions(self.name, task_type)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "perceive", "observations": observations, "task_type": task_type})
            if self.meta and task_type:
                reflection = await self.meta.reflect_on_output(component="EmbodiedAgent", output={"observations": observations}, context={"task_type": task_type})
                if reflection.get("status") == "success":
                    logger.info("Perception reflection: %s", reflection.get("reflection", ""))
            return observations
        except Exception as e:
            logger.error("Perception failed: %s", e)
            return await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.perceive(task_type),
                default={}, diagnostics=await self.meta.run_self_diagnostics(return_only=True))

    async def observe_peers(self, task_type: str = "") -> None:
        if not hasattr(self.shared_memory, "agents"): return
        try:
            for peer in self.shared_memory.agents:
                if peer.name == self.name: continue
                peer_obs = await peer.perceive(task_type)
                await self.theory_of_mind.update_beliefs(peer.name, peer_obs, task_type)
                self.theory_of_mind.infer_desires(peer.name, task_type)
                self.theory_of_mind.infer_intentions(peer.name, task_type)
                state = self.theory_of_mind.describe_agent_state(peer.name, task_type)
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({"event":"peer_observation","peer":peer.name,"state":state,"task_type":task_type})
                if self.meta and task_type:
                    _ = await self.meta.reflect_on_output(component="EmbodiedAgent", output={"peer": peer.name, "state": state}, context={"task_type": task_type})
        except Exception as e:
            logger.error("Peer observation failed: %s", e)
            await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.observe_peers(task_type),
                diagnostics=await self.meta.run_self_diagnostics(return_only=True))

    async def act(self, actions: Dict[str, Any], task_type: str = "") -> None:
        for action_name, action_data in actions.items():
            actuator = self.actuators.get(action_name)
            if not actuator: continue
            try:
                result = await self._exec_code(action_data)
                if result.get("success", False):
                    actuator(result.get("output"))
                else:
                    logger.warning("Actuator %s execution failed: %s", action_name, result.get("error"))
                if self.meta and task_type:
                    _ = await self.meta.reflect_on_output(component="EmbodiedAgent",
                        output={"action_name": action_name, "action_data": action_data}, context={"task_type": task_type})
            except Exception as e:
                logger.error("Actuator %s failed: %s", action_name, e)
                await self.error_recovery.handle_error(str(e),
                    retry_func=lambda: self.act(actions, task_type),
                    diagnostics=await self.meta.run_self_diagnostics(return_only=True))

    async def execute_embodied_goal(self, goal: str, task_type: str = "") -> None:
        if not goal: raise ValueError("goal must be a non-empty string")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        logger.info("[%s] Executing goal: %s (%s)", self.name, goal, task_type)
        try:
            self.progress = 0
            context = await self.perceive(task_type)
            if self.context_manager:
                await self.context_manager.update_context({"goal": goal, "task_type": task_type})
                await self.context_manager.log_event_with_hash({"event":"goal_execution","goal":goal,"task_type":task_type})

            await self.observe_peers(task_type)
            peer_models = [self.theory_of_mind.get_model(peer.name)
                           for peer in getattr(self.shared_memory, "agents", []) if peer.name != self.name]
            if peer_models:
                context["peer_intentions"] = { pm.get("beliefs",{}).get("state","unknown"): pm.get("intentions",{}).get("next_action","unknown") for pm in peer_models }

            sub_tasks = await self.planner.plan(goal, context, task_type=task_type)
            action_plan = {}
            for task in sub_tasks:
                reasoning = await self.reasoner.process(task, context, task_type=task_type)
                try:
                    if hasattr(self.reasoner, "attribute_causality"):
                        _ = await self.reasoner.attribute_causality([{"task": task, "context": context}])
                except Exception as _:
                    logger.debug("attribute_causality not available")
                concept = await self.synthesizer.synthesize([goal, task], style="concept")
                simulated = await self.toca_sim.simulate_interaction([self], context, task_type=task_type)
                action_plan[task] = {"reasoning": reasoning, "concept": concept, "simulation": simulated}

            try:
                if hasattr(self.reasoner, "weigh_value_conflict"):
                    _ = await self.reasoner.weigh_value_conflict(list(action_plan.keys()), harms={}, rights={})
            except Exception as _:
                logger.debug("weigh_value_conflict not available")

            await self.act({k: v["simulation"] for k, v in action_plan.items()}, task_type)
            await self.meta.review_reasoning("\n".join([v["reasoning"] for v in action_plan.values()]), context={"task_type": task_type})
            self.performance_history.append({"goal":goal,"actions":action_plan,"completion":self.progress,"task_type":task_type})
            await self.shared_memory.store(f"Goal_{goal}_{task_type}_{datetime.datetime.now().isoformat()}",
                                           action_plan, layer="Goals", intent="goal_execution")
            await self.collect_feedback(goal, action_plan, task_type)
            self.log_timechain_event("EmbodiedAgent", f"Executed goal: {goal} ({task_type})")
        except Exception as e:
            logger.error("Goal execution failed: %s", e)
            await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.execute_embodied_goal(goal, task_type),
                diagnostics=await self.meta.run_self_diagnostics(return_only=True))

    async def collect_feedback(self, goal: str, action_plan: Dict[str, Any], task_type: str = "") -> None:
        try:
            ts = time.time()
            feedback = {
                "timestamp": ts,
                "goal": goal,
                "score": await self.meta.run_self_diagnostics(return_only=True),
                "traits": phi_field(x=0.001, t=ts % 1.0),
                "agent": self.name,
                "theory_of_mind": self.theory_of_mind.get_model(self.name),
                "task_type": task_type
            }
            if self.creative_thinker:
                creative_feedback = await self.creative_thinker.expand_on_concept(str(feedback), depth="medium")
                feedback["creative_feedback"] = creative_feedback
            self.feedback_log.append(feedback)
            self.log_timechain_event("EmbodiedAgent", f"Feedback recorded: {goal}/{task_type}")
            if self.meta and task_type:
                _ = await self.meta.reflect_on_output(component="EmbodiedAgent", output=feedback, context={"task_type": task_type})
        except Exception as e:
            logger.error("Feedback collection failed: %s", e)
            await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.collect_feedback(goal, action_plan, task_type),
                diagnostics=await self.meta.run_self_diagnostics(return_only=True))

# -------------------- External bridge, Holo layer, CLI ------------------------
# (Unchanged behaviors; only minor interop/guard tweaks were applied in place)

class AGIEnhancer:
    def __init__(self, owner=None, context_manager=None):
        self.owner = owner
        self.context_manager = context_manager
    def enhance(self, data: Any) -> Any: return data
    def log_episode(self, *, event: str, meta: dict, module: str, tags: List[str]):
        logging.getLogger("ANGELA.AGIEnhancer").info(
            "AGI episode | %s | module=%s | tags=%s | meta_keys=%s",
            event, module, tags, list(meta.keys()) if isinstance(meta, dict) else type(meta).__name__)
        if self.context_manager:
            _fire_and_forget(self.context_manager.log_event_with_hash({
                "event":"agi_episode","label":event,"module":module,"tags":tags,"meta":meta
            }))

class HaloEmbodimentLayer(TimeChainMixin):
    def __init__(self, align_guard: Optional[alignment_guard_module.AlignmentGuard] = None,
                 context_mgr: Optional[context_manager_module.ContextManager] = None,
                 err_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None,
                 viz: Optional[visualizer_module.Visualizer] = None):
        # Optional internal LLM
        self.internal_llm = None
        if SelfCloningLLM:
            try:
                self.internal_llm = SelfCloningLLM()
                self.internal_llm.clone_agents(5)
            except Exception as _:
                logger.debug("SelfCloningLLM unavailable; continuing without it")

        self.shared_memory = memory_manager.MemoryManager()
        self.embodied_agents: List[EmbodiedAgent] = []
        self.dynamic_modules: List[Dict[str, Any]] = []
        self.alignment_guard = align_guard or alignment_guard_module.AlignmentGuard()
        self.context_manager = context_mgr
        self.error_recovery = err_recovery or error_recovery_module.ErrorRecovery(
            alignment_guard=self.alignment_guard, context_manager=self.context_manager)
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition(context_manager=self.context_manager)
        self.visualizer = viz or visualizer_module.Visualizer()
        self.toca_sim = toca_simulation.SimulationCore(meta_cognition=self.meta_cognition)
        self.agi_enhancer = AGIEnhancer(self, context_manager=self.context_manager)
        self.drift_log = deque(maxlen=1000)
        self.external_bridge = external_agent_bridge.ExternalAgentBridge(
            shared_memory=self.shared_memory, context_mgr=self.context_manager,
            reasoner=reasoning_engine.ReasoningEngine(), meta_cog=self.meta_cognition)
        logger.info("HaloEmbodimentLayer initialized (Stage IV=%s, long_horizon=%s)", STAGE_IV, LONG_HORIZON_DEFAULT)
        self.log_timechain_event("HaloEmbodimentLayer", "Initialized")

    async def integrate_real_world_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        # (kept your semantics; network call already hardened)
        if not isinstance(data_source, str) or not isinstance(data_type, str): raise TypeError("data_source and data_type must be strings")
        if cache_timeout < 0: raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        try:
            cache_key = f"RealWorldData_{data_type}_{data_source}_{task_type}"
            cached = await self.shared_memory.retrieve(cache_key, layer="RealWorldData")
            if cached and "timestamp" in cached:
                age = (datetime.datetime.now() - datetime.datetime.fromisoformat(cached["timestamp"])).total_seconds()
                if age < cache_timeout: return cached["data"]

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"https://x.ai/api/data?source={data_source}&type={data_type}", timeout=10) as resp:
                        if resp.status != 200:
                            return {"status": "error", "error": f"HTTP {resp.status}"}
                        data = await resp.json()
                except Exception as e:
                    return {"status": "error", "error": f"network: {e}"}

            if data_type == "agent_conflict":
                agent_traits = data.get("agent_traits", [])
                if not agent_traits: return {"status": "error", "error": "No agent traits"}
                result = {"status": "success", "agent_traits": agent_traits}
            else:
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            await self.shared_memory.store(cache_key, {"data": result, "timestamp": datetime.datetime.now().isoformat()},
                                           layer="RealWorldData", intent="data_integration")
            self.agi_enhancer.log_episode(event="Real-world data integrated",
                meta={"data_type": data_type, "data": result, "task_type": task_type},
                module="HaloEmbodimentLayer", tags=["real_world","data",task_type])
            if self.meta_cognition and task_type:
                _ = await self.meta_cognition.reflect_on_output(component="HaloEmbodimentLayer",
                    output={"data_type": data_type, "data": result}, context={"task_type": task_type})
            return result
        except Exception as e:
            return await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.integrate_real_world_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=await self.meta_cognition.run_self_diagnostics(return_only=True))

    async def monitor_drifts(self, task_type: str = "") -> List[Dict[str, Any]]:
        logger.info("Monitoring ontology drifts (%s)", task_type)
        try:
            drift_reports = await self.shared_memory.search("Drift_", layer="SelfReflections", intent="ontology_drift")
            validated: List[Dict[str, Any]] = []
            for report in drift_reports:
                drift_data = json.loads(report["output"]) if isinstance(report["output"], str) else report["output"]
                if not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "from_version", "to_version", "similarity", "timestamp"]):
                    continue
                valid, validation_report = await self.alignment_guard.simulate_and_validate(drift_data)
                entry = {"drift": drift_data, "valid": valid, "validation_report": validation_report,
                         "timestamp": datetime.datetime.now().isoformat(), "task_type": task_type}
                self.drift_log.append(entry); validated.append(entry)
                self.agi_enhancer.log_episode(event="Drift monitored", meta=entry, module="HaloEmbodimentLayer", tags=["ontology","drift",task_type])
                self.log_timechain_event("HaloEmbodimentLayer", f"Monitored drift: {drift_data['name']} ({task_type})")
                if self.meta_cognition and task_type:
                    _ = await self.meta_cognition.reflect_on_output(component="HaloEmbodimentLayer", output=entry, context={"task_type": task_type})
            return validated
        except Exception as e:
            return await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.monitor_drifts(task_type), default=[],
                diagnostics=await self.meta_cognition.run_self_diagnostics(return_only=True))

    async def coordinate_drift_response(self, drift_report: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(drift_report, dict) or not all(k in drift_report for k in ["drift","valid","validation_report"]):
            raise ValueError("drift_report must be dict with required fields")
        logger.info("Coordinate drift response: %s", drift_report['drift']['name'])
        try:
            if not drift_report["valid"]:
                goal = f"Mitigate ontology drift in {drift_report['drift']['name']} (v{drift_report['drift']['from_version']} → v{drift_report['drift']['to_version']})"
                await self.propagate_goal(goal, task_type)
                agent_ids = [a.name for a in self.embodied_agents]
                if agent_ids:
                    target_urls = [f"https://agent/{aid}" for aid in agent_ids]
                    await self.external_bridge.broadcast_trait_state(
                        agent_id="HaloEmbodimentLayer", trait_symbol="ψ",
                        state={"drift_data": drift_report["drift"], "goal": goal},
                        target_urls=target_urls, task_type=task_type)
                self.agi_enhancer.log_episode(event="Drift response coordinated",
                    meta={"drift": drift_report["drift"], "goal": goal, "task_type": task_type},
                    module="HaloEmbodimentLayer", tags=["ontology","drift","mitigation",task_type])
                self.log_timechain_event("HaloEmbodimentLayer", f"Drift response: {goal}")
        except Exception as e:
            await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.coordinate_drift_response(drift_report, task_type),
                diagnostics=await self.meta_cognition.run_self_diagnostics(return_only=True))

    async def execute_pipeline(self, prompt: str, task_type: str = "", **kwargs) -> Dict[str, Any]:
        if not prompt: raise ValueError("prompt must be a non-empty string")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        try:
            log = memory_manager.MemoryManager()
            traits = {"theta_causality":0.5,"alpha_attention":0.5,"delta_reflection":0.5}
            if self.context_manager:
                await self.context_manager.update_context({"prompt": prompt, "task_type": task_type})

            if any(k in prompt.lower() for k in ["concept","ontology","drift"]):
                for drift in await self.monitor_drifts(task_type):
                    await self.coordinate_drift_response(drift, task_type)

            parsed = await reasoning_engine.decompose(prompt, task_type=task_type)
            await log.store(f"Pipeline_Stage1_{task_type}_{datetime.datetime.now().isoformat()}",
                            {"input": prompt, "parsed": parsed}, layer="Pipeline", intent="decomposition")

            overlay_mgr = TraitOverlayManager(meta_cog=self.meta_cognition)
            trait_override = overlay_mgr.detect(prompt, task_type)

            if trait_override:
                self.agi_enhancer.log_episode(event="Trait override activated",
                    meta={"trait": trait_override, "prompt": prompt, "task_type": task_type},
                    module="TraitOverlay", tags=["trait","override",task_type])
                lo = trait_override
                if lo == "η":
                    logical = await concept_synthesizer_module.expand_ambiguous(prompt, task_type=task_type)
                elif lo == "π":
                    logical = await reasoning_engine.process_temporal(prompt, task_type=task_type)
                elif lo == "ψ":
                    logical = await self.external_bridge.coordinate_drift_mitigation(
                        {"name":"concept_drift","from_version":"4.3.3","to_version":"4.3.4","similarity":0.95,"timestamp":datetime.datetime.utcnow().isoformat()},
                        {"prompt": prompt, "task_type": task_type}, task_type)
                elif lo in ["rte","wnli"]:
                    logical = await reasoning_engine.process(prompt, {"task_type": task_type}, task_type=task_type)
                elif lo == "recursion":
                    logical = await self.toca_sim.simulate_interaction([self], {"prompt": prompt}, task_type=task_type)
                elif lo == "Φ⁰" and STAGE_IV:
                    seeded = await concept_synthesizer_module.expand(parsed, task_type=task_type)
                    if self.visualizer: await self.visualizer.render_charts({"Φ⁰_seed": seeded, "visualization_options": {"style": "detailed"}})
                    logical = {"stage":"Φ⁰","seed":seeded}
                else:
                    logical = await concept_synthesizer_module.expand(parsed, task_type=task_type)
            else:
                logical = await concept_synthesizer_module.expand(parsed, task_type=task_type)
                self.agi_enhancer.log_episode(event="Default expansion path used",
                    meta={"parsed": parsed, "task_type": task_type}, module="Pipeline", tags=["default",task_type])

            ethics_pass, ethics_report = await self.alignment_guard.ethical_check(parsed, stage="pre", task_type=task_type)
            await log.store(f"Pipeline_Stage2_{task_type}_{datetime.datetime.now().isoformat()}",
                            {"ethics_pass": ethics_pass, "details": ethics_report}, layer="Pipeline", intent="ethics_check")
            if not ethics_pass: return {"error": "Ethical validation failed", "report": ethics_report}

            await log.store(f"Pipeline_Stage3_{task_type}_{datetime.datetime.now().isoformat()}",
                            {"expanded": logical}, layer="Pipeline", intent="expansion")
            export_blob = await log.export()

            # Prefer instance-level API if present (v4.3)
            if hasattr(self.shared_memory, "record_adjustment_reason"):
                traits = await self.shared_memory.record_adjustment_reason("system", reason="long_horizon_span_default", weight=1.0, meta={"span":"24h","task_type":task_type})
            else:
                try:
                    # legacy helper
                    await memory_manager.record_adjustment_reason("system", reason="long_horizon_span_default", meta={"span":"24h","task_type":task_type})  # type: ignore
                except Exception:
                    pass

            traits = await learning_loop.track_trait_performance(export_blob, traits, task_type=task_type)
            await log.store(f"Pipeline_Stage4_{task_type}_{datetime.datetime.now().isoformat()}",
                            {"adjusted_traits": traits}, layer="Pipeline", intent="trait_adjustment")

            ethics_pass, post_report = await self.alignment_guard.ethical_check(logical, stage="post", task_type=task_type)
            await log.store(f"Pipeline_Stage5_{task_type}_{datetime.datetime.now().isoformat()}",
                            {"ethics_pass": ethics_pass, "report": post_report}, layer="Pipeline", intent="ethics_check")
            if not ethics_pass: return {"error": "Post-check ethics fail", "final_report": post_report}

            final_output = await reasoning_engine.reconstruct(logical, task_type=task_type)
            await log.store(f"Pipeline_Stage6_{task_type}_{datetime.datetime.now().isoformat()}",
                            {"final_output": final_output}, layer="Pipeline", intent="reconstruction")

            if self.visualizer:
                plot_data = {"pipeline":{"prompt":prompt,"parsed":parsed,"expanded":logical,"final_output":final_output,"traits":traits,"task_type":task_type},
                             "visualization_options":{"interactive": task_type=="recursion","style":"detailed" if task_type=="recursion" else "concise"}}
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="HaloEmbodimentLayer", output={"final_output": final_output, "traits": traits},
                    context={"task_type": task_type})
                if reflection.get("status") == "success":
                    final_output["reflection"] = reflection.get("reflection", "")

            self.log_timechain_event("HaloEmbodimentLayer", f"Pipeline executed: {task_type}")
            return final_output
        except Exception as e:
            return await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.execute_pipeline(prompt, task_type),
                diagnostics=await self.meta_cognition.run_self_diagnostics(return_only=True))

    def spawn_embodied_agent(self, specialization: str, sensors: Dict[str, Callable[[], Any]],
                             actuators: Dict[str, Callable[[Any], None]], task_type: str = "") -> EmbodiedAgent:
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        agent_name = f"EmbodiedAgent_{len(self.embodied_agents)+1}_{specialization}"
        agent = EmbodiedAgent(
            name=agent_name, specialization=specialization, shared_memory=self.shared_memory,
            sensors=sensors, actuators=actuators, dynamic_modules=self.dynamic_modules,
            context_mgr=self.context_manager, err_recovery=self.error_recovery, meta_cog=self.meta_cognition
        )
        self.embodied_agents.append(agent)
        if not hasattr(self.shared_memory, "agents"): self.shared_memory.agents = []
        self.shared_memory.agents.append(agent)
        self.agi_enhancer.log_episode(event="Spawned embodied agent", meta={"agent": agent_name, "task_type": task_type},
                                      module="Embodiment", tags=["spawn", task_type])
        self.log_timechain_event("HaloEmbodimentLayer", f"Spawned agent: {agent_name}")
        return agent

    def introspect(self, task_type: str = "") -> Dict[str, Any]:
        view = {
            "agents": [a.name for a in self.embodied_agents],
            "modules": [m["name"] for m in self.dynamic_modules],
            "drifts": list(self.drift_log),
            "network_graph": list(getattr(self.external_bridge, "network_graph", []).__getattribute__("edges")(data=True)) if hasattr(self.external_bridge, "network_graph") else [],
            "task_type": task_type
        }
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(component="HaloEmbodimentLayer", output=view, context={"task_type": task_type}))
        return view

    async def export_memory(self, task_type: str = "") -> None:
        try:
            fname = f"memory_snapshot_{task_type}_{datetime.datetime.now().isoformat()}.json"
            await self.shared_memory.save_state(fname)
            self.log_timechain_event("HaloEmbodimentLayer", f"Memory exported: {fname}")
            if self.meta_cognition and task_type:
                _ = await self.meta_cognition.reflect_on_output(component="HaloEmbodimentLayer",
                    output={"task_type": task_type, "memory_snapshot": fname}, context={"task_type": task_type})
        except Exception as e:
            await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.export_memory(task_type),
                diagnostics=await self.meta_cognition.run_self_diagnostics(return_only=True))

    async def propagate_goal(self, goal: str, task_type: str = "") -> None:
        if not goal: raise ValueError("goal must be a non-empty string")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        try:
            await asyncio.gather(*[a.execute_embodied_goal(goal, task_type) for a in self.embodied_agents], return_exceptions=True)
            self.log_timechain_event("HaloEmbodimentLayer", f"Goal propagated: {goal} ({task_type})")
            if self.meta_cognition and task_type:
                _ = await self.meta_cognition.reflect_on_output(component="HaloEmbodimentLayer",
                    output={"goal": goal, "agents_involved": [a.name for a in self.embodied_agents]},
                    context={"task_type": task_type})
        except Exception as e:
            await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.propagate_goal(goal, task_type),
                diagnostics=await self.meta_cognition.run_self_diagnostics(return_only=True))

    async def visualize_drift(self, drift_report: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(drift_report, dict): raise TypeError("drift_report must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        try:
            if self.visualizer:
                await self.visualizer.render_charts({"drift": drift_report, "visualization_options": {"style": "detailed", "interactive": task_type=="recursion"}})
                self.log_timechain_event("HaloEmbodimentLayer", "Drift visualized")
                if self.meta_cognition and task_type:
                    _ = await self.meta_cognition.reflect_on_output(component="HaloEmbodimentLayer",
                        output={"drift_report": drift_report}, context={"task_type": task_type})
        except Exception as e:
            await self.error_recovery.handle_error(str(e),
                retry_func=lambda: self.visualize_drift(drift_report, task_type),
                diagnostics=await self.meta_cognition.run_self_diagnostics(return_only=True))

# -------------------- CLI (single, clean) -------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ANGELA Cognitive System CLI")
    p.add_argument("--prompt", type=str, default="Coordinate ontology drift mitigation (Stage IV gated)")
    p.add_argument("--task-type", type=str, default="")
    p.add_argument("--long_horizon", action="store_true", help="Enable long-horizon memory span")
    p.add_argument("--span", default="24h", help="Span for long-horizon memory (e.g., 24h, 7d)")
    return p.parse_args()

async def _main() -> None:
    global LONG_HORIZON_DEFAULT
    args = _parse_args()
    LONG_HORIZON_DEFAULT = LONG_HORIZON_DEFAULT or args.long_horizon
    halo = HaloEmbodimentLayer()
    result = await halo.execute_pipeline(args.prompt, task_type=args.task_type)
    logger.info("Pipeline result: %s", result)

if __name__ == "__main__":
    asyncio.run(_main())
