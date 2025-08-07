"""
ANGELA Cognitive System Module
Refactored Version: 3.4.0  # Updated for Structural Grounding
Refactor Date: 2025-08-06
Maintainer: ANGELA System Framework

This module provides classes for embodied agents, ecosystem management, and cognitive enhancements
in the ANGELA v3.4.0 architecture, with ontology drift coordination and trait mesh networking.
"""

import logging
import time
import math
import datetime
import asyncio
import os
import openai
import requests
import random
from collections import deque
from typing import Dict, Any, Optional, List, Callable
from functools import lru_cache
import uuid
from networkx import DiGraph
from restrictedpython import safe_globals

from modules import (
    reasoning_engine, recursive_planner, context_manager, simulation_core,
    toca_simulation, creative_thinker, knowledge_retriever, learning_loop,
    concept_synthesizer, memory_manager, multi_modal_fusion, code_executor,
    visualizer, external_agent_bridge, alignment_guard, user_profile, error_recovery,
    meta_cognition
)
from self_cloning_llm import SelfCloningLLM

logger = logging.getLogger("ANGELA.CognitiveSystem")
SYSTEM_CONTEXT = {}
timechain_log = deque(maxlen=1000)
grok_query_log = deque(maxlen=60)
openai_query_log = deque(maxlen=60)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
openai.api_key = OPENAI_API_KEY

class TimeChainMixin:
    """Mixin for logging timechain events."""
    def log_timechain_event(self, module: str, description: str) -> None:
        timechain_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "module": module,
            "description": description
        })
        if hasattr(self, 'context_manager') and self.context_manager:
            self.context_manager.log_event_with_hash({"event": "timechain_event", "module": module, "description": description})

    def get_timechain_log(self) -> List[Dict[str, Any]]:
        return list(timechain_log)

# Cognitive Trait Functions
@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 0.1)

@lru_cache(maxsize=100)
def phi_field(x: float, t: float) -> float:
    t_normalized = t % 1.0
    return sum([
        epsilon_emotion(t_normalized), beta_concentration(t_normalized), theta_memory(t_normalized),
        gamma_creativity(t_normalized), delta_sleep(t_normalized), mu_morality(t_normalized),
        iota_intuition(t_normalized), phi_physical(t_normalized), eta_empathy(t_normalized),
        omega_selfawareness(t_normalized), kappa_culture(t_normalized, x), lambda_linguistics(t_normalized),
        chi_culturevolution(t_normalized), psi_history(t_normalized), zeta_spirituality(t_normalized),
        xi_collective(t_normalized, x), tau_timeperception(t_normalized)
    ])

TRAIT_OVERLAY = {
    "ϕ": ["creative_thinker", "concept_synthesizer"],
    "θ": ["reasoning_engine", "recursive_planner"],
    "η": ["alignment_guard", "meta_cognition"],
    "ω": ["simulation_core", "learning_loop"],
    "π": ["reasoning_engine", "toca_simulation"],
    "ψ": ["external_agent_bridge", "simulation_core"],
    "Υ": ["meta_cognition", "context_manager"]
}

def infer_traits(task_description: str) -> List[str]:
    if not isinstance(task_description, str):
        logger.error("Invalid task_description: must be a string.")
        raise TypeError("task_description must be a string")
    traits = []
    if "imagine" in task_description.lower() or "dream" in task_description.lower():
        traits.append("ϕ")
    if "ethics" in task_description.lower() or "should" in task_description.lower():
        traits.append("η")
    if "plan" in task_description.lower() or "solve" in task_description.lower():
        traits.append("θ")
    if "temporal" in task_description.lower() or "sequence" in task_description.lower():
        traits.append("π")
    if "drift" in task_description.lower() or "coordinate" in task_description.lower():
        traits.extend(["ψ", "Υ"])
    return traits if traits else ["θ"]

def trait_overlay_router(task_description: str, active_traits: List[str]) -> List[str]:
    if not isinstance(active_traits, list) or not all(isinstance(t, str) for t in active_traits):
        logger.error("Invalid active_traits: must be a list of strings.")
        raise TypeError("active_traits must be a list of strings")
    routed_modules = set()
    for trait in active_traits:
        routed_modules.update(TRAIT_OVERLAY.get(trait, []))
    return list(routed_modules)

def static_module_router(task_description: str) -> List[str]:
    return ["reasoning_engine", "concept_synthesizer"]

class TraitOverlayManager:
    """Manager for detecting and activating trait overlays."""
    def __init__(self):
        self.active_traits = []

    def detect(self, prompt: str) -> Optional[str]:
        if not isinstance(prompt, str):
            logger.error("Invalid prompt: must be a string.")
            raise TypeError("prompt must be a string")
        if "temporal logic" in prompt.lower() or "sequence" in prompt.lower():
            return "π"
        if "ambiguity" in prompt.lower() or "interpretive" in prompt.lower() or "ethics" in prompt.lower():
            return "η"
        if "drift" in prompt.lower() or "coordinate" in prompt.lower():
            return "ψ"
        return None

    def activate(self, trait: str) -> None:
        if not isinstance(trait, str):
            logger.error("Invalid trait: must be a string.")
            raise TypeError("trait must be a string")
        if trait not in self.active_traits:
            self.active_traits.append(trait)
            logger.info("Trait overlay '%s' activated.", trait)

    def deactivate(self, trait: str) -> None:
        if not isinstance(trait, str):
            logger.error("Invalid trait: must be a string.")
            raise TypeError("trait must be a string")
        if trait in self.active_traits:
            self.active_traits.remove(trait)
            logger.info("Trait overlay '%s' deactivated.", trait)

    def status(self) -> List[str]:
        return self.active_traits

class ConsensusReflector:
    """Class for managing shared reflections and detecting mismatches."""
    def __init__(self):
        self.shared_reflections = deque(maxlen=1000)

    def post_reflection(self, feedback: Dict[str, Any]) -> None:
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary.")
            raise TypeError("feedback must be a dictionary")
        self.shared_reflections.append(feedback)
        logger.debug("Posted reflection: %s", feedback)

    def cross_compare(self) -> List[tuple]:
        mismatches = []
        reflections = list(self.shared_reflections)
        for i in range(len(reflections)):
            for j in range(i + 1, len(reflections)):
                a = reflections[i]
                b = reflections[j]
                if a.get('goal') == b.get('goal') and a.get('theory_of_mind') != b.get('theory_of_mind'):
                    mismatches.append((a.get('agent'), b.get('agent'), a.get('goal')))
        return mismatches

    def suggest_alignment(self) -> str:
        return "Schedule inter-agent reflection or re-observation."

consensus_reflector = ConsensusReflector()

class SymbolicSimulator:
    """Class for recording and summarizing simulation events."""
    def __init__(self):
        self.events = deque(maxlen=1000)

    def record_event(self, agent_name: str, goal: str, concept: str, simulation: Any) -> None:
        if not all(isinstance(x, str) for x in [agent_name, goal, concept]):
            logger.error("Invalid input: agent_name, goal, and concept must be strings.")
            raise TypeError("agent_name, goal, and concept must be strings")
        self.events.append({
            "agent": agent_name,
            "goal": goal,
            "concept": concept,
            "result": simulation
        })
        logger.debug("Recorded event for agent %s: goal=%s, concept=%s", agent_name, goal, concept)

    def summarize_recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        if not isinstance(limit, int) or limit <= 0:
            logger.error("Invalid limit: must be a positive integer.")
            raise ValueError("limit must be a positive integer")
        return list(self.events)[-limit:]

    def extract_semantics(self) -> List[str]:
        return [f"Agent {e['agent']} pursued '{e['goal']}' via '{e['concept']}' → {e['result']}" for e in self.events]

symbolic_simulator = SymbolicSimulator()

class TheoryOfMindModule:
    """Module for modeling beliefs, desires, and intentions of agents."""
    def __init__(self, concept_synthesizer: Optional[concept_synthesizer.ConceptSynthesizer] = None):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.concept_synthesizer = concept_synthesizer
        logger.info("TheoryOfMindModule initialized")

    def update_beliefs(self, agent_name: str, observation: Dict[str, Any]) -> None:
        if not isinstance(agent_name, str) or not agent_name:
            logger.error("Invalid agent_name: must be a non-empty string.")
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(observation, dict):
            logger.error("Invalid observation: must be a dictionary.")
            raise TypeError("observation must be a dictionary")
        
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        if self.concept_synthesizer:
            synthesized = self.concept_synthesizer.synthesize(observation, style="belief_update")
            if synthesized["valid"]:
                model["beliefs"].update(synthesized["concept"])
        elif "location" in observation:
            previous = model["beliefs"].get("location")
            model["beliefs"]["location"] = observation["location"]
            model["beliefs"]["state"] = "confused" if previous and observation["location"] == previous else "moving"
        self.models[agent_name] = model
        logger.debug("Updated beliefs for %s: %s", agent_name, model["beliefs"])

    def infer_desires(self, agent_name: str) -> None:
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        beliefs = model.get("beliefs", {})
        if beliefs.get("state") == "confused":
            model["desires"]["goal"] = "seek_clarity"
        elif beliefs.get("state") == "moving":
            model["desires"]["goal"] = "continue_task"
        self.models[agent_name] = model
        logger.debug("Inferred desires for %s: %s", agent_name, model["desires"])

    def infer_intentions(self, agent_name: str) -> None:
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        desires = model.get("desires", {})
        if desires.get("goal") == "seek_clarity":
            model["intentions"]["next_action"] = "ask_question"
        elif desires.get("goal") == "continue_task":
            model["intentions"]["next_action"] = "advance"
        self.models[agent_name] = model
        logger.debug("Inferred intentions for %s: %s", agent_name, model["intentions"])

    def get_model(self, agent_name: str) -> Dict[str, Any]:
        return self.models.get(agent_name, {})

    def describe_agent_state(self, agent_name: str) -> str:
        model = self.get_model(agent_name)
        return (f"{agent_name} believes they are {model.get('beliefs', {}).get('state', 'unknown')}, "
                f"desires to {model.get('desires', {}).get('goal', 'unknown')}, "
                f"and intends to {model.get('intentions', {}).get('next_action', 'unknown')}.")

class EmbodiedAgent(TimeChainMixin):
    """An embodied agent with sensors, actuators, and cognitive capabilities."""
    def __init__(self, name: str, specialization: str, shared_memory: memory_manager.MemoryManager,
                 sensors: Dict[str, Callable[[], Any]], actuators: Dict[str, Callable[[Any], None]],
                 dynamic_modules: Optional[List[Dict[str, Any]]] = None,
                 context_manager: Optional[context_manager.ContextManager] = None,
                 error_recovery: Optional[error_recovery.ErrorRecovery] = None,
                 code_executor: Optional[code_executor.CodeExecutor] = None):
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
        self.meta = meta_cognition.MetaCognition(context_manager=context_manager, alignment_guard=alignment_guard.AlignmentGuard())
        self.sim_core = simulation_core.SimulationCore()
        self.synthesizer = concept_synthesizer.ConceptSynthesizer()
        self.toca_sim = toca_simulation.TocaSimulation()
        self.theory_of_mind = TheoryOfMindModule(concept_synthesizer=self.synthesizer)
        self.context_manager = context_manager
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(context_manager=context_manager)
        self.code_executor = code_executor
        self.progress = 0
        self.performance_history = deque(maxlen=1000)
        self.feedback_log = deque(maxlen=1000)
        logger.info("EmbodiedAgent initialized: %s", name)

    def perceive(self) -> Dict[str, Any]:
        logger.info("[%s] Perceiving environment...", self.name)
        observations = {}
        try:
            for sensor_name, sensor_func in self.sensors.items():
                try:
                    observations[sensor_name] = sensor_func()
                except Exception as e:
                    logger.warning("Sensor %s failed: %s", sensor_name, str(e))
            self.theory_of_mind.update_beliefs(self.name, observations)
            self.theory_of_mind.infer_desires(self.name)
            self.theory_of_mind.infer_intentions(self.name)
            logger.debug("[%s] Self-theory: %s", self.name, self.theory_of_mind.describe_agent_state(self.name))
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "perceive", "observations": observations})
            return observations
        except Exception as e:
            logger.error("Perception failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=self.perceive)

    def observe_peers(self) -> None:
        if not hasattr(self.shared_memory, "agents"):
            return
        try:
            for peer in self.shared_memory.agents:
                if peer.name != self.name:
                    peer_observation = peer.perceive()
                    self.theory_of_mind.update_beliefs(peer.name, peer_observation)
                    self.theory_of_mind.infer_desires(peer.name)
                    self.theory_of_mind.infer_intentions(peer.name)
                    state = self.theory_of_mind.describe_agent_state(peer.name)
                    logger.debug("[%s] Observed peer %s: %s", self.name, peer.name, state)
                    if self.context_manager:
                        self.context_manager.log_event_with_hash({"event": "peer_observation", "peer": peer.name, "state": state})
        except Exception as e:
            logger.error("Peer observation failed: %s", str(e))

    def act(self, actions: Dict[str, Any]) -> None:
        for action_name, action_data in actions.items():
            actuator = self.actuators.get(action_name)
            if actuator:
                try:
                    if self.code_executor:
                        result = self.code_executor.execute(action_data, language="python")
                        if result["success"]:
                            actuator(result["output"])
                        else:
                            logger.warning("Actuator %s execution failed: %s", action_name, result["error"])
                    else:
                        actuator(action_data)
                    logger.info("Actuated %s: %s", action_name, action_data)
                except Exception as e:
                    logger.error("Actuator %s failed: %s", action_name, str(e))

    def execute_embodied_goal(self, goal: str) -> None:
        if not isinstance(goal, str) or not goal:
            logger.error("Invalid goal: must be a non-empty string.")
            raise ValueError("goal must be a non-empty string")
        
        logger.info("[%s] Executing embodied goal: %s", self.name, goal)
        try:
            self.progress = 0
            context = self.perceive()
            if self.context_manager:
                self.context_manager.update_context(context)
                self.context_manager.log_event_with_hash({"event": "goal_execution", "goal": goal})

            self.observe_peers()
            peer_models = [
                self.theory_of_mind.get_model(peer.name)
                for peer in getattr(self.shared_memory, "agents", [])
                if peer.name != self.name
            ]
            if peer_models:
                context["peer_intentions"] = {
                    peer["beliefs"].get("state", "unknown"): peer["intentions"].get("next_action", "unknown")
                    for peer in peer_models
                }

            sub_tasks = self.planner.plan(goal, context)
            action_plan = {}
            for task in sub_tasks:
                reasoning = self.reasoner.process(task, context)
                concept = self.synthesizer.synthesize([goal, task], style="concept")
                simulated = self.sim_core.run(reasoning, context, export_report=True)
                action_plan[task] = {
                    "reasoning": reasoning,
                    "concept": concept,
                    "simulation": simulated
                }

            self.act({k: v["simulation"] for k, v in action_plan.items()})
            self.meta.review_reasoning("\n".join([v["reasoning"] for v in action_plan.values()]))
            self.performance_history.append({"goal": goal, "actions": action_plan, "completion": self.progress})
            self.shared_memory.store(goal, action_plan)
            self.collect_feedback(goal, action_plan)
            self.log_timechain_event("EmbodiedAgent", f"Executed goal: {goal}")
        except Exception as e:
            logger.error("Goal execution failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.execute_embodied_goal(goal))

    def collect_feedback(self, goal: str, action_plan: Dict[str, Any]) -> None:
        try:
            timestamp = time.time()
            feedback = {
                "timestamp": timestamp,
                "goal": goal,
                "score": self.meta.run_self_diagnostics(),
                "traits": phi_field(x=0.001, t=timestamp % 1.0),
                "agent": self.name,
                "theory_of_mind": self.theory_of_mind.get_model(self.name)
            }
            if self.creative_thinker:
                creative_feedback = self.creative_thinker.expand_on_concept(str(feedback), depth="medium")
                feedback["creative_feedback"] = creative_feedback
            self.feedback_log.append(feedback)
            self.log_timechain_event("EmbodiedAgent", f"Feedback recorded for goal: {goal}")
            logger.info("[%s] Feedback recorded for goal '%s'", self.name, goal)
        except Exception as e:
            logger.error("Feedback collection failed: %s", str(e))

class ExternalAgentBridge:
    """A class for orchestrating helper agents and coordinating trait mesh networking."""
    def __init__(self, context_manager: Optional[context_manager.ContextManager] = None,
                 reasoning_engine: Optional[reasoning_engine.ReasoningEngine] = None):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []
        self.context_manager = context_manager
        self.reasoning_engine = reasoning_engine
        self.network_graph = DiGraph()
        self.trait_states = defaultdict(dict)
        self.code_executor = code_executor.CodeExecutor()
        logger.info("ExternalAgentBridge initialized with trait mesh networking support")

    async def create_agent(self, task: str, context: Dict[str, Any]) -> 'HelperAgent':
        """Create a new helper agent for a task asynchronously."""
        from meta_cognition import HelperAgent  # Deferred import to avoid circularity
        if not isinstance(task, str):
            logger.error("Invalid task type: must be a string.")
            raise TypeError("task must be a string")
        if not isinstance(context, dict):
            logger.error("Invalid context type: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        
        try:
            agent = HelperAgent(
                name=f"Agent_{len(self.agents) + 1}_{uuid.uuid4().hex[:8]}",
                task=task,
                context=context,
                dynamic_modules=self.dynamic_modules,
                api_blueprints=self.api_blueprints,
                meta_cognition=meta_cognition.MetaCognition(context_manager=self.context_manager, reasoning_engine=self.reasoning_engine)
            )
            self.agents.append(agent)
            self.network_graph.add_node(agent.name, metadata=context)
            logger.info("Spawned agent: %s", agent.name)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "agent_created",
                    "agent": agent.name,
                    "task": task,
                    "drift": "drift" in task.lower()
                })
            return agent
        except Exception as e:
            logger.error("Agent creation failed: %s", str(e))
            raise

    async def broadcast_trait_state(self, agent_id: str, trait_symbol: str, state: Dict[str, Any], target_urls: List[str]) -> List[Any]:
        """Broadcast trait state (ψ or Υ) to target agents asynchronously."""
        if trait_symbol not in ["ψ", "Υ"]:
            logger.error("Invalid trait symbol: %s. Must be ψ or Υ.", trait_symbol)
            raise ValueError("Trait symbol must be ψ or Υ")
        if not isinstance(state, dict):
            logger.error("Invalid state: must be a dictionary")
            raise TypeError("state must be a dictionary")
        if not isinstance(target_urls, list) or not all(isinstance(url, str) and url.startswith("https://") for url in target_urls):
            logger.error("Invalid target_urls: must be a list of HTTPS URLs")
            raise TypeError("target_urls must be a list of HTTPS URLs")

        try:
            alignment_guard_instance = alignment_guard.AlignmentGuard()
            if not alignment_guard_instance.check(json.dumps(state)):
                logger.warning("Trait state failed alignment check: %s", state)
                raise ValueError("Trait state failed alignment check")

            serialized_state = self.code_executor.safe_execute(
                f"return json.dumps({json.dumps(state)})",
                safe_globals
            )
            if not serialized_state:
                logger.error("Failed to serialize trait state")
                raise ValueError("Failed to serialize trait state")

            memory_manager.cache_state(f"{agent_id}_{trait_symbol}", state)
            self.trait_states[agent_id][trait_symbol] = state

            for url in target_urls:
                peer_id = url.split("/")[-1]
                self.network_graph.add_edge(agent_id, peer_id, trait=trait_symbol)

            async with aiohttp.ClientSession() as session:
                tasks = [session.post(url, json={"agent_id": agent_id, "trait_symbol": trait_symbol, "state": state}, timeout=10)
                         for url in target_urls]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

            successful = [r for r in responses if not isinstance(r, Exception)]
            logger.info("Trait %s broadcasted from %s to %d/%d targets", trait_symbol, agent_id, len(successful), len(target_urls))
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "trait_broadcast",
                    "agent_id": agent_id,
                    "trait_symbol": trait_symbol,
                    "successful_targets": len(successful),
                    "total_targets": len(target_urls)
                })

            feedback = {"successful_targets": len(successful), "total_targets": len(target_urls)}
            self.push_behavior_feedback(feedback)
            self.update_gnn_weights_from_feedback(feedback)

            return responses
        except Exception as e:
            logger.error("Trait state broadcast failed: %s", str(e))
            return [{"status": "error", "error": str(e)}]

    async def synchronize_trait_states(self, agent_id: str, trait_symbol: str) -> Dict[str, Any]:
        """Synchronize trait states across all connected agents."""
        if trait_symbol not in ["ψ", "Υ"]:
            logger.error("Invalid trait symbol: %s. Must be ψ or Υ.", trait_symbol)
            raise ValueError("Trait symbol must be ψ or Υ")

        try:
            local_state = self.trait_states.get(agent_id, {}).get(trait_symbol, {})
            if not local_state:
                logger.warning("No local state found for %s:%s", agent_id, trait_symbol)
                return {"status": "error", "error": "No local state found"}

            peer_states = []
            for peer_id in self.network_graph.neighbors(agent_id):
                cached_state = memory_manager.retrieve_state(f"{peer_id}_{trait_symbol}")
                if cached_state:
                    peer_states.append((peer_id, cached_state))

            simulation_input = {
                "local_state": local_state,
                "peer_states": {pid: state for pid, state in peer_states},
                "trait_symbol": trait_symbol
            }
            sim_result = await asyncio.to_thread(toca_simulation.run_simulation, json.dumps(simulation_input))
            if not sim_result or "coherent" not in sim_result.lower():
                logger.warning("Simulation failed to align states: %s", sim_result)
                return {"status": "error", "error": "State alignment simulation failed"}

            aligned_state = self.arbitrate([local_state] + [state for _, state in peer_states])
            if aligned_state:
                self.trait_states[agent_id][trait_symbol] = aligned_state
                memory_manager.cache_state(f"{agent_id}_{trait_symbol}", aligned_state)
                logger.info("Synchronized trait %s for %s", trait_symbol, agent_id)
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "trait_synchronized",
                        "agent_id": agent_id,
                        "trait_symbol": trait_symbol,
                        "aligned_state": aligned_state
                    })
                return {"status": "success", "aligned_state": aligned_state}
            else:
                logger.warning("Failed to arbitrate trait states")
                return {"status": "error", "error": "Arbitration failed"}
        except Exception as e:
            logger.error("Trait state synchronization failed: %s", str(e))
            return {"status": "error", "error": str(e)}

    async def coordinate_drift_mitigation(self, drift_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate drift mitigation across agents."""
        if not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary")
            raise TypeError("drift_data must be a dictionary")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        
        try:
            if not meta_cognition.MetaCognition().validate_drift(drift_data):
                logger.warning("Invalid drift data: %s", drift_data)
                return {"status": "error", "error": "Invalid drift data"}

            task = "Mitigate ontology drift"
            context["drift"] = drift_data
            agent = await self.create_agent(task, context)
            if self.reasoning_engine:
                subgoals = await self.reasoning_engine.decompose(task, context, prioritize=True)
                simulation_result = await self.reasoning_engine.run_drift_mitigation_simulation(drift_data, context)
            else:
                subgoals = ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"]
                simulation_result = {"status": "no simulation", "result": "default subgoals applied"}

            results = await self.collect_results(parallel=True, collaborative=True)
            arbitrated_result = self.arbitrate(results)

            target_urls = [f"https://agent/{peer_id}" for peer_id in self.network_graph.nodes if peer_id != agent.name]
            await self.broadcast_trait_state(agent.name, "ψ", {"drift_data": drift_data, "subgoals": subgoals}, target_urls)

            output = {
                "drift_data": drift_data,
                "subgoals": subgoals,
                "simulation": simulation_result,
                "results": results,
                "arbitrated_result": arbitrated_result,
                "status": "success",
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
            }
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "drift_mitigation_coordinated",
                    "output": output,
                    "drift": True
                })
            if self.reasoning_engine and hasattr(self.reasoning_engine, 'agi_enhancer') and self.reasoning_engine.agi_enhancer:
                self.reasoning_engine.agi_enhancer.log_episode(
                    event="Drift Mitigation Coordinated",
                    meta=output,
                    module="ExternalAgentBridge",
                    tags=["drift", "coordination"]
                )
            return output
        except Exception as e:
            logger.error("Drift mitigation coordination failed: %s", str(e))
            return {"status": "error", "error": str(e), "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')}

    async def collect_results(self, parallel: bool = True, collaborative: bool = True) -> List[Any]:
        """Collect results from all agents asynchronously."""
        logger.info("Collecting results from %d agents...", len(self.agents))
        results = []

        try:
            if parallel:
                async def run_agent(agent):
                    try:
                        return await agent.execute(self.agents if collaborative else None)
                    except Exception as e:
                        logger.error("Error collecting from %s: %s", agent.name, str(e))
                        return {"error": str(e)}

                tasks = [run_agent(agent) for agent in self.agents]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                for agent in self.agents:
                    results.append(await agent.execute(self.agents if collaborative else None))
            
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "results_collected",
                    "results_count": len(results)
                })
            logger.info("Results aggregation complete.")
            return results
        except Exception as e:
            logger.error("Result collection failed: %s", str(e))
            return []

    def arbitrate(self, submissions: List[Any]) -> Any:
        """Arbitrate among agent submissions to select the best result."""
        if not submissions:
            logger.warning("No submissions to arbitrate.")
            return None
        try:
            counter = Counter(submissions)
            most_common = counter.most_common(1)
            if most_common:
                result, count = most_common[0]
                sim_result = toca_simulation.run_simulation(f"Arbitration validation: {result}") or "no simulation data"
                if "coherent" in sim_result.lower():
                    logger.info("Arbitration selected: %s (count: %d)", result, count)
                    if self.context_manager:
                        asyncio.create_task(self.context_manager.log_event_with_hash({
                            "event": "arbitration",
                            "result": result,
                            "count": count
                        }))
                    return result
            logger.warning("Arbitration failed: no clear majority or invalid simulation.")
            return None
        except Exception as e:
            logger.error("Arbitration failed: %s", str(e))
            return None

    def push_behavior_feedback(self, feedback: Dict[str, Any]) -> None:
        """Push feedback to update GNN weights."""
        try:
            logger.info("Pushing behavior feedback: %s", feedback)
            if self.context_manager:
                asyncio.create_task(self.context_manager.log_event_with_hash({
                    "event": "behavior_feedback",
                    "feedback": feedback
                }))
        except Exception as e:
            logger.error("Failed to push behavior feedback: %s", str(e))

    def update_gnn_weights_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Update GNN weights based on feedback."""
        try:
            logger.info("Updating GNN weights with feedback: %s", feedback)
            if self.context_manager:
                asyncio.create_task(self.context_manager.log_event_with_hash({
                    "event": "gnn_weights_updated",
                    "feedback": feedback
                }))
        except Exception as e:
            logger.error("Failed to update GNN weights: %s", str(e))

class HaloEmbodimentLayer(TimeChainMixin):
    """Layer for managing embodied agents, dynamic modules, and ontology drift coordination."""
    def __init__(self, alignment_guard: Optional[alignment_guard.AlignmentGuard] = None,
                 context_manager: Optional[context_manager.ContextManager] = None,
                 error_recovery: Optional[error_recovery.ErrorRecovery] = None):
        self.internal_llm = SelfCloningLLM()
        self.internal_llm.clone_agents(5)
        self.shared_memory = memory_manager.MemoryManager()
        self.embodied_agents = []
        self.dynamic_modules = []
        self.alignment_guard = alignment_guard
        self.context_manager = context_manager
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(
            alignment_guard=alignment_guard, context_manager=context_manager)
        self.agi_enhancer = AGIEnhancer(self, context_manager=context_manager)
        self.drift_log = deque(maxlen=1000)
        self.external_bridge = ExternalAgentBridge(context_manager=context_manager, reasoning_engine=reasoning_engine.ReasoningEngine())
        logger.info("HaloEmbodimentLayer initialized with drift coordination and trait mesh networking")
        self.log_timechain_event("HaloEmbodimentLayer", "Initialized with drift coordination and trait mesh networking")

    async def monitor_drifts(self) -> List[Dict[str, Any]]:
        """Retrieve and aggregate ontology drift reports from memory_manager."""
        logger.info("Monitoring ontology drifts")
        try:
            drift_reports = await self.shared_memory.search("Drift_", layer="SelfReflections", intent="ontology_drift")
            validated_drifts = []
            for report in drift_reports:
                drift_data = eval(report["output"]) if isinstance(report["output"], str) else report["output"]
                if not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "from_version", "to_version", "similarity", "timestamp"]):
                    logger.warning("Invalid drift report format: %s", drift_data)
                    continue
                valid, validation_report = self.alignment_guard.simulate_and_validate(drift_data)
                drift_entry = {
                    "drift": drift_data,
                    "valid": valid,
                    "validation_report": validation_report,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                self.drift_log.append(drift_entry)
                validated_drifts.append(drift_entry)
                self.agi_enhancer.log_episode(
                    event="Drift monitored",
                    meta=drift_entry,
                    module="HaloEmbodimentLayer",
                    tags=["ontology", "drift"]
                )
                self.log_timechain_event("HaloEmbodimentLayer", f"Monitored drift: {drift_data['name']}")
            return validated_drifts
        except Exception as e:
            logger.error("Drift monitoring failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=self.monitor_drifts, default=[])

    async def coordinate_drift_response(self, drift_report: Dict[str, Any]) -> None:
        """Coordinate agent responses to an ontology drift."""
        if not isinstance(drift_report, dict) or not all(k in drift_report for k in ["drift", "valid", "validation_report"]):
            logger.error("Invalid drift_report: must be a dict with drift, valid, validation_report.")
            raise ValueError("drift_report must be a dict with required fields")
        
        logger.info(f"Coordinating response to drift: {drift_report['drift']['name']}")
        try:
            if not drift_report["valid"]:
                goal = f"Mitigate ontology drift in {drift_report['drift']['name']} (Version {drift_report['drift']['from_version']} -> {drift_report['drift']['to_version']})"
                await self.propagate_goal(goal)
                # Broadcast drift mitigation state (ψ) to agents
                agent_ids = [agent.name for agent in self.embodied_agents]
                if agent_ids:
                    target_urls = [f"https://agent/{aid}" for aid in agent_ids]
                    await self.external_bridge.broadcast_trait_state(
                        agent_id="HaloEmbodimentLayer",
                        trait_symbol="ψ",
                        state={"drift_data": drift_report["drift"], "goal": goal},
                        target_urls=target_urls
                    )
                self.agi_enhancer.log_episode(
                    event="Drift response coordinated",
                    meta={"drift": drift_report["drift"], "goal": goal},
                    module="HaloEmbodimentLayer",
                    tags=["ontology", "drift", "mitigation"]
                )
                self.log_timechain_event("HaloEmbodimentLayer", f"Coordinated drift response: {goal}")
            else:
                logger.info(f"No action needed for valid drift: {drift_report['drift']['name']}")
        except Exception as e:
            logger.error("Drift response coordination failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.coordinate_drift_response(drift_report))

    async def execute_pipeline(self, prompt: str) -> Dict[str, Any]:
        if not isinstance(prompt, str) or not prompt:
            logger.error("Invalid prompt: must be a non-empty string.")
            raise ValueError("prompt must be a non-empty string")
        
        try:
            log = memory_manager.MemoryManager()
            traits = {
                "theta_causality": 0.5,
                "alpha_attention": 0.5,
                "delta_reflection": 0.5,
            }
            if self.context_manager:
                self.context_manager.update_context({"prompt": prompt})

            if "concept" in prompt.lower() or "ontology" in prompt.lower() or "drift" in prompt.lower():
                drifts = await self.monitor_drifts()
                for drift in drifts:
                    await self.coordinate_drift_response(drift)

            parsed_prompt = reasoning_engine.decompose(prompt)
            log.record("Stage 1", {"input": prompt, "parsed": parsed_prompt})

            overlay_mgr = TraitOverlayManager()
            trait_override = overlay_mgr.detect(prompt)

            if trait_override:
                self.agi_enhancer.log_episode(
                    event="Trait override activated",
                    meta={"trait": trait_override, "prompt": prompt},
                    module="TraitOverlay",
                    tags=["trait", "override"]
                )
                if trait_override == "η":
                    logical_output = concept_synthesizer.expand_ambiguous(prompt)
                elif trait_override == "π":
                    logical_output = reasoning_engine.process_temporal(prompt)
                elif trait_override == "ψ":
                    logical_output = await self.external_bridge.coordinate_drift_mitigation(
                        {"name": "concept_drift", "from_version": "3.4.0", "to_version": "3.4.1", "similarity": 0.95, "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')},
                        {"prompt": prompt}
                    )
                else:
                    logical_output = concept_synthesizer.expand(parsed_prompt)
            else:
                logical_output = concept_synthesizer.expand(parsed_prompt)
                self.agi_enhancer.log_episode(
                    event="Default expansion path used",
                    meta={"parsed": parsed_prompt},
                    module="Pipeline",
                    tags=["default"]
                )

            ethics_pass, ethics_report = self.alignment_guard.ethical_check(parsed_prompt, stage="pre")
            log.record("Stage 2", {"ethics_pass": ethics_pass, "details": ethics_report})
            if not ethics_pass:
                logger.warning("Ethical validation failed: %s", ethics_report)
                return {"error": "Ethical validation failed", "report": ethics_report}

            log.record("Stage 3", {"expanded": logical_output})
            traits = learning_loop.track_trait_performance(log.export(), traits)
            log.record("Stage 4", {"adjusted_traits": traits})

            ethics_pass, final_report = self.alignment_guard.ethical_check(logical_output, stage="post")
            log.record("Stage 5", {"ethics_pass": ethics_pass, "report": final_report})
            if not ethics_pass:
                logger.warning("Post-check ethics failed: %s", final_report)
                return {"error": "Post-check ethics fail", "final_report": final_report}

            final_output = reasoning_engine.reconstruct(logical_output)
            log.record("Stage 6", {"final_output": final_output})
            self.log_timechain_event("HaloEmbodimentLayer", f"Pipeline executed for prompt: {prompt}")
            return final_output
        except Exception as e:
            logger.error("Pipeline execution failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.execute_pipeline(prompt))

    def spawn_embodied_agent(self, specialization: str, sensors: Dict[str, Callable[[], Any]],
                            actuators: Dict[str, Callable[[Any], None]]) -> EmbodiedAgent:
        agent_name = f"EmbodiedAgent_{len(self.embodied_agents)+1}_{specialization}"
        agent = EmbodiedAgent(
            name=agent_name,
            specialization=specialization,
            shared_memory=self.shared_memory,
            sensors=sensors,
            actuators=actuators,
            dynamic_modules=self.dynamic_modules,
            context_manager=self.context_manager,
            error_recovery=self.error_recovery
        )
        self.embodied_agents.append(agent)
        if not hasattr(self.shared_memory, "agents"):
            self.shared_memory.agents = []
        self.shared_memory.agents.append(agent)
        self.agi_enhancer.log_episode(
            event="Spawned embodied agent",
            meta={"agent": agent_name},
            module="Embodiment",
            tags=["spawn"]
        )
        logger.info("Spawned embodied agent: %s", agent.name)
        self.log_timechain_event("HaloEmbodimentLayer", f"Spawned agent: {agent_name}")
        return agent

    def introspect(self) -> Dict[str, Any]:
        return {
            "agents": [agent.name for agent in self.embodied_agents],
            "modules": [mod["name"] for mod in self.dynamic_modules],
            "drifts": list(self.drift_log),
            "network_graph": list(self.external_bridge.network_graph.edges(data=True))
        }

    def export_memory(self) -> None:
        try:
            self.shared_memory.save_state("memory_snapshot.json")
            logger.info("Memory exported to memory_snapshot.json")
            self.log_timechain_event("HaloEmbodimentLayer", "Memory exported")
        except Exception as e:
            logger.error("Memory export failed: %s", str(e))

    async def propagate_goal(self, goal: str) -> None:
        if not isinstance(goal, str) or not goal:
            logger.error("Invalid goal: must be a non-empty string.")
            raise ValueError("goal must be a non-empty string")
        
        logger.info("Propagating goal: %s", goal)
        try:
            if "concept" in goal.lower() or "ontology" in goal.lower() or "drift" in goal.lower():
                drifts = await self.monitor_drifts()
                for drift in drifts:
                    await self.coordinate_drift_response(drift)

            llm_responses = self.internal_llm.broadcast_prompt(goal)
            for aid, res in llm_responses.items():
                logger.info("LLM-Agent %s: %s", aid, res)
                self.shared_memory.store(f"llm_agent_{aid}_response", res)
                self.agi_enhancer.log_episode(
                    event="LLM agent reflection",
                    meta={"agent_id": aid, "response": res},
                    module="ReasoningEngine",
                    tags=["internal_llm"]
                )

            for agent in self.embodied_agents:
                agent.execute_embodied_goal(goal)
                logger.info("[%s] Progress: %d%% Complete", agent.name, agent.progress)
            self.log_timechain_event("HaloEmbodimentLayer", f"Propagated goal: {goal}")
        except Exception as e:
            logger.error("Goal propagation failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.propagate_goal(goal))

    def deploy_dynamic_module(self, module_blueprint: Dict[str, Any]) -> None:
        if not isinstance(module_blueprint, dict) or "name" not in module_blueprint:
            logger.error("Invalid module_blueprint: must be a dictionary with 'name' key.")
            raise ValueError("module_blueprint must be a dictionary with 'name' key")
        
        logger.info("Deploying module: %s", module_blueprint["name"])
        self.dynamic_modules.append(module_blueprint)
        for agent in self.embodied_agents:
            agent.dynamic_modules.append(module_blueprint)
        self.agi_enhancer.log_episode(
            event="Deployed dynamic module",
            meta={"module": module_blueprint["name"]},
            module="ModuleDeployment",
            tags=["deploy"]
        )
        self.log_timechain_event("HaloEmbodimentLayer", f"Deployed module: {module_blueprint['name']}")

    def optimize_ecosystem(self) -> None:
        agent_stats = {
            "agents": [agent.name for agent in self.embodied_agents],
            "dynamic_modules": [mod["name"] for mod in self.dynamic_modules],
            "drifts": list(self.drift_log),
            "network_graph": list(self.external_bridge.network_graph.edges(data=True))
        }
        try:
            recommendations = self.meta.propose_optimizations(agent_stats)
            logger.info("Optimization recommendations: %s", recommendations)
            self.agi_enhancer.reflect_and_adapt("Ecosystem optimization performed.")
            self.log_timechain_event("HaloEmbodimentLayer", f"Optimized ecosystem: {recommendations}")
        except Exception as e:
            logger.error("Ecosystem optimization failed: %s", str(e))

class AGIEnhancer(TimeChainMixin):
    """Enhancer for logging, self-improvement, and ethical auditing."""
    def __init__(self, orchestrator: Any, config: Optional[Dict[str, Any]] = None,
                 context_manager: Optional[context_manager.ContextManager] = None):
        self.orchestrator = orchestrator
        self.config = config or {}
        self.episodic_log: List[Dict[str, Any]] = deque(maxlen=20000)
        self.ethics_audit_log: List[Dict[str, Any]] = deque(maxlen=2000)
        self.self_improvement_log: List[str] = deque(maxlen=2000)
        self.explanations: List[Dict[str, Any]] = deque(maxlen=2000)
        self.agent_mesh_messages: List[Dict[str, Any]] = deque(maxlen=2000)
        self.embodiment_actions: List[Dict[str, Any]] = deque(maxlen=2000)
        self.context_manager = context_manager
        logger.info("AGIEnhancer initialized")

    def log_episode(self, event: str, meta: Optional[Dict[str, Any]] = None,
                    module: Optional[str] = None, tags: Optional[List[str]] = None,
                    embedding: Optional[Any] = None) -> None:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event,
            "meta": meta or {},
            "module": module or "",
            "tags": tags or [],
            "embedding": embedding
        }
        self.episodic_log.append(entry)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "log_episode", "entry": entry})
        if hasattr(self.orchestrator, "export_memory"):
            self.orchestrator.export_memory()
        logger.debug("Logged episode: %s", event)

    def replay_episodes(self, n: int = 5, module: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        results = list(self.episodic_log)
        if module:
            results = [e for e in results if e.get("module") == module]
        if tag:
            results = [e for e in results if tag in e.get("tags", [])]
        return results[-n:]

    def find_episode(self, keyword: str, deep: bool = False) -> List[Dict[str, Any]]:
        def matches(ep):
            if keyword.lower() in ep["event"].lower():
                return True
            if deep:
                if any(keyword.lower() in str(v).lower() for v in ep.get("meta", {}).values()):
                    return True
                if any(keyword.lower() in t.lower() for t in ep.get("tags", [])):
                    return True
            return False
        return [ep for ep in self.episodic_log if matches(ep)]

    def reflect_and_adapt(self, feedback: str, auto_patch: bool = False) -> str:
        suggestion = f"Reviewing feedback: '{feedback}'. Suggest adjusting {random.choice(['reasoning', 'tone', 'planning', 'speed'])}."
        self.self_improvement_log.append(suggestion)
        if hasattr(self.orchestrator, "LearningLoop") and auto_patch:
            try:
                patch_result = self.orchestrator.LearningLoop.adapt(feedback)
                self.self_improvement_log.append(f"LearningLoop patch: {patch_result}")
                return f"{suggestion} | Patch applied: {patch_result}"
            except Exception as e:
                logger.error("LearningLoop patch failed: %s", str(e))
        return suggestion

    def run_self_patch(self) -> str:
        patch = f"Self-improvement at {datetime.datetime.now().isoformat()}."
        if hasattr(self.orchestrator, "reflect"):
            try:
                audit = self.orchestrator.reflect()
                patch += f" Reflect: {audit}"
            except Exception as e:
                logger.error("Reflection failed: %s", str(e))
        self.self_improvement_log.append(patch)
        return patch

    def ethics_audit(self, action: str, context: Optional[str] = None) -> str:
        if not isinstance(action, str):
            logger.error("Invalid action: must be a string.")
            raise TypeError("action must be a string")
        flagged = "clear"
        if hasattr(self.orchestrator, "AlignmentGuard"):
            try:
                flagged = self.orchestrator.AlignmentGuard.audit(action, context)
            except Exception as e:
                logger.error("Ethics audit failed: %s", str(e))
                flagged = "audit_error"
        else:
            flagged = "unsafe" if any(w in action.lower() for w in ["harm", "bias", "exploit"]) else "clear"
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "context": context,
            "status": flagged
        }
        self.ethics_audit_log.append(entry)
        return flagged

    def explain_last_decision(self, depth: int = 3, mode: str = "auto") -> str:
        if not self.explanations:
            return "No explanations logged yet."
        items = list(self.explanations)[-depth:]
        if mode == "svg" and hasattr(self.orchestrator, "Visualizer"):
            try:
                svg = self.orchestrator.Visualizer.render(items)
                return svg
            except Exception as e:
                logger.error("SVG render error: %s", str(e))
                return "SVG render error."
        return "\n\n".join([e["text"] if isinstance(e, dict) and "text" in e else str(e) for e in items])

    def log_explanation(self, explanation: str, trace: Optional[Any] = None, svg: Optional[Any] = None) -> None:
        entry = {"text": explanation, "trace": trace, "svg": svg}
        self.explanations.append(entry)
        logger.debug("Logged explanation: %s", explanation)

    def embodiment_act(self, action: str, params: Optional[Dict[str, Any]] = None, real: bool = False) -> str:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "params": params or {},
            "mode": "real" if real else "sim"
        }
        self.embodiment_actions.append(entry)
        if real and hasattr(self.orchestrator, "embodiment_interface"):
            try:
                res = self.orchestrator.embodiment_interface.execute(action, params)
                entry["result"] = res
            except Exception as e:
                entry["result"] = f"interface_error: {str(e)}"
        logger.info("Embodiment action '%s' (%s) requested.", action, "real" if real else "sim")
        return f"Embodiment action '{action}' ({'real' if real else 'sim'}) requested."

    def send_agent_message(self, to_agent: str, content: str, meta: Optional[Dict[str, Any]] = None) -> str:
        msg = {
            "timestamp": datetime.datetime.now().isoformat(),
            "to": to_agent,
            "content": content,
            "meta": meta or {},
            "mesh_state": self.orchestrator.introspect() if hasattr(self.orchestrator, "introspect") else {}
        }
        self.agent_mesh_messages.append(msg)
        if hasattr(self.orchestrator, "ExternalAgentBridge"):
            try:
                self.orchestrator.ExternalAgentBridge.send(to_agent, content, meta)
                msg["sent"] = True
            except Exception as e:
                logger.error("Agent message failed: %s", str(e))
                msg["sent"] = False
        logger.info("Message to %s: %s", to_agent, content)
        return f"Message to {to_agent}: {content}"

    def periodic_self_audit(self) -> str:
        if hasattr(self.orchestrator, "reflect"):
            try:
                report = self.orchestrator.reflect()
                self.log_explanation(f"Meta-cognitive audit: {report}")
                return report
            except Exception as e:
                logger.error("Self-audit failed: %s", str(e))
                return f"Self-audit failed: {str(e)}"
        return "Orchestrator reflect() unavailable."

    def process_event(self, event: str, meta: Optional[Dict[str, Any]] = None,
                     module: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        self.log_episode(event, meta, module, tags)
        self.log_explanation(f"Processed event: {event}", trace={"meta": meta, "module": module, "tags": tags})
        ethics_status = self.ethics_audit(event, context=str(meta))
        return f"Event processed. Ethics: {ethics_status}"

async def query_openai(prompt: str, model: str = "gpt-4", temperature: float = 0.5) -> Dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        logger.error("Invalid prompt: must be a non-empty string.")
        return {"error": "Invalid prompt: must be a non-empty string"}
    if not within_limit(openai_query_log):
        logger.warning("OpenAI rate limit exceeded.")
        return {"error": "OpenAI API rate limit exceeded"}
    
    cache_key = f"openai::{model}::{prompt}"
    cached = memory_manager.retrieve_cached_response(cache_key)
    if cached:
        logger.debug("Retrieved cached OpenAI response for: %s", prompt[:50])
        return cached
    
    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        result = response['choices'][0]['message']['content'].strip()
        memory_manager.store_cached_response(cache_key, result)
        logger.info("OpenAI query successful for: %s", prompt[:50])
        return result
    except Exception as e:
        logger.error("OpenAI query failed: %s", str(e))
        return {"error": str(e)}

def query_openai_sync(prompt: str, model: str = "gpt-4", temperature: float = 0.5) -> Dict[str, Any]:
    return asyncio.run(query_openai(prompt, model, temperature))

async def query_grok(prompt: str) -> Dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        logger.error("Invalid prompt: must be a non-empty string.")
        return {"error": "Invalid prompt: must be a non-empty string"}
    if not within_limit(grok_query_log):
        logger.warning("Grok rate limit exceeded.")
        return {"error": "Grok API rate limit exceeded"}
    
    cache_key = f"grok::{prompt}"
    cached = memory_manager.retrieve_cached_response(cache_key)
    if cached:
        logger.debug("Retrieved cached Grok response for: %s", prompt[:50])
        return cached
    
    try:
        response = requests.post(
            "https://api.groq.com/v1/query",
            json={"q": prompt},
            headers={"Authorization": f"Bearer {GROK_API_KEY}"}
        )
        response.raise_for_status()
        result = response.json()
        memory_manager.store_cached_response(cache_key, result)
        logger.info("Grok query successful for: %s", prompt[:50])
        return result
    except Exception as e:
        logger.error("Grok query failed: %s", str(e))
        return {"error": str(e)}

def query_grok_sync(prompt: str) -> Dict[str, Any]:
    return query_grok(prompt)

# Patch classes with TimeChainMixin
HaloEmbodimentLayer.__bases__ = (TimeChainMixin,)
AGIEnhancer.__bases__ = (TimeChainMixin,)
EmbodiedAgent.__bases__ = (TimeChainMixin,)

# Patch simulation_core
setattr(simulation_core, "HybridCognitiveState", simulation_core.HybridCognitiveState)
setattr(simulation_core, "TraitOverlayManager", TraitOverlayManager)

logger.info("ANGELA upgrade complete: Trait overlays (π, η, ψ, Υ) + hybrid-mode simulation + drift coordination enabled.")
