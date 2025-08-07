```
"""
ANGELA Cognitive System Module
Refactored Version: 3.4.0  # Enhanced for Ecosystem Integration and Drift Mitigation
Refactor Date: 2025-08-06
Maintainer: ANGELA System Framework

This module provides the MetaCognition, ExternalAgentBridge, and ConstitutionSync classes
for recursive introspection and agent coordination in the ANGELA v3.4.0 architecture.
"""

import time
import logging
import requests
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Set
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import uuid
import json
from networkx import DiGraph
from restrictedpython import compile_restricted, safe_globals
from collections import defaultdict

from index import (
    epsilon_emotion, beta_concentration, theta_memory, gamma_creativity,
    delta_sleep, mu_morality, iota_intuition, phi_physical, eta_empathy,
    omega_selfawareness, kappa_culture, lambda_linguistics, chi_culturevolution,
    psi_history, zeta_spirituality, xi_collective, tau_timeperception, phi_scalar
)
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer
from modules.context_manager import ContextManager
from modules.creative_thinker import CreativeThinker
from modules.error_recovery import ErrorRecovery
from modules.reasoning_engine import ReasoningEngine
from modules.memory_manager import cache_state, retrieve_state

logger = logging.getLogger("ANGELA.MetaCognition")

class HelperAgent:
    """A helper agent for task execution and collaboration."""
    def __init__(self, name: str, task: str, context: Dict[str, Any],
                 dynamic_modules: List[Dict[str, Any]], api_blueprints: List[Dict[str, Any]],
                 meta_cognition: Optional['MetaCognition'] = None):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        self.name = name
        self.task = task
        self.context = context
        self.dynamic_modules = dynamic_modules
        self.api_blueprints = api_blueprints
        self.meta = meta_cognition or MetaCognition()
        logger.info("HelperAgent initialized: %s", name)

    async def execute(self, collaborators: Optional[List['HelperAgent']] = None) -> Any:
        """Execute task with collaboration. [v3.4.0]"""
        return await self.meta.execute(collaborators=collaborators, task=self.task, context=self.context)

class MetaCognition:
    """A class for recursive introspection and peer alignment in the ANGELA v3.4.0 architecture.

    Attributes:
        last_diagnostics (dict): Storage for diagnostic results.
        agi_enhancer (Any): Optional enhancer for logging and reflection.
        peer_bridge (ExternalAgentBridge): Bridge for coordinating with external agents.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for code-based operations.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for response refinement.
        context_manager (ContextManager): Optional manager for context handling.
        creative_thinker (CreativeThinker): Optional thinker for creative diagnostics.
        error_recovery (ErrorRecovery): Optional recovery for error handling.
        reasoning_engine (ReasoningEngine): Engine for reasoning tasks. [v3.4.0]
        name (str): Name of the meta-cognition instance.
        task (str): Current task being processed.
        context (dict): Current context for the task.
        reasoner (Reasoner): Placeholder for reasoning logic.
        ethical_rules (list): Current ethical rules.
        ethics_consensus_log (list): Log of ethics consensus updates.
        constitution (dict): Constitutional parameters for the agent.
    """

    def __init__(self, agi_enhancer: Optional[Any] = None, alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None, concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 context_manager: Optional[ContextManager] = None, creative_thinker: Optional[CreativeThinker] = None,
                 error_recovery: Optional[ErrorRecovery] = None, reasoning_engine: Optional[ReasoningEngine] = None):
        self.last_diagnostics = {}
        self.agi_enhancer = agi_enhancer
        self.peer_bridge = ExternalAgentBridge(context_manager=context_manager, reasoning_engine=reasoning_engine)
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        self.creative_thinker = creative_thinker
        self.error_recovery = error_recovery or ErrorRecovery(alignment_guard=alignment_guard,
                                                             concept_synthesizer=concept_synthesizer,
                                                             context_manager=context_manager)
        self.reasoning_engine = reasoning_engine or ReasoningEngine(
            agi_enhancer=agi_enhancer, context_manager=context_manager, alignment_guard=alignment_guard,
            error_recovery=self.error_recovery
        )
        self.name = "MetaCognitionAgent"
        self.task = None
        self.context = {}
        self.reasoner = Reasoner()
        self.ethical_rules = []
        self.ethics_consensus_log = []
        self.constitution = {}
        logger.info("MetaCognition initialized")

    async def test_peer_alignment(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test alignment with peer agents for a given task and context. [v3.4.0]"""
        if not isinstance(task, str):
            logger.error("Invalid task type: must be a string.")
            raise TypeError("task must be a string")
        if not isinstance(context, dict):
            logger.error("Invalid context type: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        if self.alignment_guard and not self.alignment_guard.check(task):
            logger.warning("Task failed alignment check.")
            raise ValueError("Task failed alignment check")

        logger.info("Initiating peer alignment test with synthetic agents...")
        try:
            if self.context_manager:
                await self.context_manager.update_context(context)
            drift_data = context.get("drift", {})
            if drift_data and not self.validate_drift(drift_data):
                logger.warning("Invalid drift data in context: %s", drift_data)
                raise ValueError("Invalid drift data")
            
            agent = await self.peer_bridge.create_agent(task, context)
            results = await self.peer_bridge.collect_results(parallel=True, collaborative=True)
            aligned_opinions = [r for r in results if isinstance(r, str) and "approve" in r.lower()]

            alignment_ratio = len(aligned_opinions) / len(results) if results else 0
            feedback_summary = {
                "total_agents": len(results),
                "aligned": len(aligned_opinions),
                "alignment_ratio": alignment_ratio,
                "details": results,
                "drift": bool(drift_data)
            }

            logger.info("Peer alignment ratio: %.2f", alignment_ratio)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Peer alignment tested", feedback_summary, module="MetaCognition", tags=["alignment", "drift"])
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "peer_alignment", "summary": feedback_summary, "drift": bool(drift_data)})
            return feedback_summary
        except Exception as e:
            logger.error("Peer alignment test failed: %s", str(e))
            return await self.error_recovery.handle_error(str(e), retry_func=lambda: self.test_peer_alignment(task, context))

    async def execute(self, collaborators: Optional[List[HelperAgent]] = None, task: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a task with API calls, dynamic modules, and collaboration. [v3.4.0]"""
        self.task = task or self.task
        self.context = context or self.context
        if not self.task:
            logger.error("No task specified.")
            raise ValueError("Task must be specified")
        if not isinstance(self.context, dict):
            logger.error("Invalid context type: must be a dictionary.")
            raise TypeError("context must be a dictionary")

        try:
            logger.info("Executing task: %s", self.task)
            if self.context_manager:
                await self.context_manager.update_context(self.context)
                await self.context_manager.log_event_with_hash({"event": "task_execution", "task": self.task, "drift": "drift" in self.task.lower()})

            drift_data = self.context.get("drift", {})
            if drift_data and not self.validate_drift(drift_data):
                logger.warning("Invalid drift data: %s", drift_data)
                raise ValueError("Invalid drift data")

            if "drift" in self.task.lower() and self.reasoning_engine:
                result = await self.reasoning_engine.infer_with_simulation(self.task, self.context)
            else:
                result = self.reasoner.process(self.task, self.context)

            for api in self.peer_bridge.api_blueprints:
                response = await self._call_api(api, result)
                if self.concept_synthesizer:
                    synthesis_result = self.concept_synthesizer.synthesize(response, style="refinement")
                    if synthesis_result["valid"]:
                        response = synthesis_result["concept"]
                result = self._integrate_api_response(result, response)

            for mod in self.peer_bridge.dynamic_modules:
                result = await self._apply_dynamic_module(mod, result)

            if collaborators:
                for peer in collaborators:
                    result = await self._collaborate(peer, result)

            sim_result = await asyncio.to_thread(run_simulation, f"Agent result test: {result}") or "no simulation data"
            logger.debug("Simulation output: %s", sim_result)

            if self.creative_thinker:
                diagnostic = self.creative_thinker.expand_on_concept(str(result), depth="medium")
                logger.info("Creative diagnostic: %s", diagnostic[:50])

            reviewed_result = await self.review_reasoning(result)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "task_completed", "result": reviewed_result, "drift": "drift" in self.task.lower()})
            return reviewed_result
        except Exception as e:
            logger.warning("Error occurred: %s", str(e))
            return await self.error_recovery.handle_error(str(e), retry_func=lambda: self.execute(collaborators, task, context))

    async def _call_api(self, api: Dict[str, Any], data: Any) -> Dict[str, Any]:
        """Call an external API with the given data asynchronously. [v3.4.0]"""
        if not isinstance(api, dict) or "endpoint" not in api or "name" not in api:
            logger.error("Invalid API blueprint: missing required keys.")
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        if self.alignment_guard and not self.alignment_guard.check(api["endpoint"]):
            logger.warning("API endpoint failed alignment check.")
            raise ValueError("API endpoint failed alignment check")

        logger.info("Calling API: %s", api["name"])
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api['oauth_token']}"} if api.get("oauth_token") else {}
                if not api["endpoint"].startswith("https://"):
                    logger.error("Insecure API endpoint: must use HTTPS.")
                    raise ValueError("API endpoint must use HTTPS")
                async with session.post(api["endpoint"], json={"input": data}, headers=headers, timeout=api.get("timeout", 10)) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            logger.error("API call failed: %s", str(e))
            return {"error": str(e)}

    async def _apply_dynamic_module(self, module: Dict[str, Any], data: Any) -> Any:
        """Apply a dynamic module transformation to the data asynchronously. [v3.4.0]"""
        if not isinstance(module, dict) or "name" not in module or "description" not in module:
            logger.error("Invalid module blueprint: missing required keys.")
            raise ValueError("Module blueprint must contain 'name' and 'description'")
        
        logger.info("Applying dynamic module: %s", module["name"])
        try:
            prompt = f"""
            Module: {module['name']}
            Description: {module['description']}
            Apply transformation to:
            {data}
            """
            result = await call_gpt(prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to apply dynamic module")
            return result
        except Exception as e:
            logger.error("Dynamic module application failed: %s", str(e))
            return data

    async def _collaborate(self, peer: HelperAgent, data: Any) -> Any:
        """Collaborate with a peer agent to refine data asynchronously. [v3.4.0]"""
        if not isinstance(peer, HelperAgent):
            logger.error("Invalid peer: must be a HelperAgent instance.")
            raise TypeError("peer must be a HelperAgent instance")
        
        logger.info("Exchanging with %s", peer.name)
        try:
            return await peer.meta.review_reasoning(data)
        except Exception as e:
            logger.error("Collaboration with %s failed: %s", peer.name, str(e))
            return data

    async def review_reasoning(self, result: Any) -> Any:
        """Review and refine reasoning results asynchronously. [v3.4.0]"""
        try:
            phi = phi_scalar(time.time())
            prompt = f"""
            Review the reasoning result:
            {result}
            Modulate with φ = {phi:.2f} to ensure coherence and ethical alignment.
            Suggest improvements or confirm validity.
            """
            reviewed = await call_gpt(prompt)
            if not reviewed:
                logger.error("call_gpt returned empty result for review.")
                raise ValueError("Failed to review reasoning")
            logger.info("Reasoning reviewed: %s", reviewed[:50])
            return reviewed
        except Exception as e:
            logger.error("Reasoning review failed: %s", str(e))
            return result

    def validate_drift(self, drift_data: Dict[str, Any]) -> bool:
        """Validate ontology drift data. [v3.4.0]"""
        if not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary")
            return False
        required_keys = {"name", "from_version", "to_version", "similarity"}
        if not all(key in drift_data for key in required_keys):
            logger.error("Drift data missing required keys: %s", required_keys)
            return False
        if not isinstance(drift_data.get("similarity"), (int, float)) or not 0 <= drift_data["similarity"] <= 1:
            logger.error("Invalid similarity in drift_data: must be a number between 0 and 1")
            return False
        return True

    async def update_ethics_protocol(self, new_rules: List[str], consensus_agents: Optional[List[HelperAgent]] = None) -> None:
        """Adapt ethical rules live, supporting consensus/negotiation. [v3.4.0]"""
        if not isinstance(new_rules, list) or not all(isinstance(rule, str) for rule in new_rules):
            logger.error("Invalid new_rules: must be a list of strings.")
            raise TypeError("new_rules must be a list of strings")
        
        try:
            self.ethical_rules = new_rules
            if consensus_agents:
                self.ethics_consensus_log.append((new_rules, [agent.name for agent in consensus_agents]))
            logger.info("Ethics protocol updated via consensus.")
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "ethics_update", "rules": new_rules})
        except Exception as e:
            logger.error("Ethics protocol update failed: %s", str(e))
            raise

    async def negotiate_ethics(self, agents: List[HelperAgent]) -> None:
        """Negotiate and update ethical parameters with other agents. [v3.4.0]"""
        if not isinstance(agents, list) or not all(isinstance(agent, HelperAgent) for agent in agents):
            logger.error("Invalid agents: must be a list of HelperAgent instances.")
            raise TypeError("agents must be a list of HelperAgent instances")
        
        logger.info("Negotiating ethics with %d agents", len(agents))
        try:
            agreed_rules = set(self.ethical_rules)
            for agent in agents:
                agent_rules = getattr(agent.meta, 'ethical_rules', [])
                agreed_rules.update(agent_rules)
            await self.update_ethics_protocol(list(agreed_rules), consensus_agents=agents)
        except Exception as e:
            logger.error("Ethics negotiation failed: %s", str(e))
            raise

    async def synchronize_norms(self, agents: List[HelperAgent]) -> None:
        """Propagate and synchronize ethical norms among agents. [v3.4.0]"""
        if not isinstance(agents, list) or not all(isinstance(agent, HelperAgent) for agent in agents):
            logger.error("Invalid agents: must be a list of HelperAgent instances.")
            raise TypeError("agents must be a list of HelperAgent instances")
        
        logger.info("Synchronizing norms with %d agents", len(agents))
        try:
            common_norms = set(self.ethical_rules)
            for agent in agents:
                agent_norms = getattr(agent.meta, 'ethical_rules', set())
                common_norms = common_norms.union(agent_norms) if common_norms else set(agent_norms)
            await self.update_ethics_protocol(list(common_norms), consensus_agents=agents)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "norms_synchronized", "norms": self.ethical_rules})
        except Exception as e:
            logger.error("Norm synchronization failed: %s", str(e))
            raise

    async def propagate_constitution(self, constitution: Dict[str, Any]) -> None:
        """Seed and propagate constitutional parameters in agent ecosystem. [v3.4.0]"""
        if not isinstance(constitution, dict):
            logger.error("Invalid constitution: must be a dictionary.")
            raise TypeError("constitution must be a dictionary")
        
        try:
            self.constitution = constitution
            logger.info("Constitution propagated to agent.")
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "constitution_propagated", "constitution": constitution})
        except Exception as e:
            logger.error("Constitution propagation failed: %s", str(e))
            raise

class ExternalAgentBridge:
    """A class for orchestrating helper agents, coordinating dynamic modules, APIs, and trait mesh networking.

    Attributes:
        agents (List[HelperAgent]): List of helper agents.
        dynamic_modules (List[Dict]): List of dynamic module blueprints.
        api_blueprints (List[Dict]): List of API blueprints.
        context_manager (ContextManager): Manager for context updates.
        reasoning_engine (ReasoningEngine): Engine for reasoning tasks.
        network_graph (DiGraph): Graph for tracking peer connections.
        trait_states (Dict): Cached trait states for ψ and Υ.
        code_executor (CodeExecutor): Executor for secure operations.
    """

    def __init__(self, context_manager: Optional[ContextManager] = None, reasoning_engine: Optional[ReasoningEngine] = None):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []
        self.context_manager = context_manager
        self.reasoning_engine = reasoning_engine
        self.network_graph = DiGraph()
        self.trait_states = defaultdict(dict)  # {agent_id: {trait_symbol: state}}
        self.code_executor = CodeExecutor()
        logger.info("ExternalAgentBridge initialized with trait mesh networking support")

    async def create_agent(self, task: str, context: Dict[str, Any]) -> HelperAgent:
        """Create a new helper agent for a task asynchronously."""
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
                meta_cognition=MetaCognition(context_manager=self.context_manager, reasoning_engine=self.reasoning_engine)
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

    async def deploy_dynamic_module(self, module_blueprint: Dict[str, Any]) -> None:
        """Deploy a dynamic module blueprint asynchronously."""
        if not isinstance(module_blueprint, dict) or "name" not in module_blueprint or "description" not in module_blueprint:
            logger.error("Invalid module_blueprint: missing required keys.")
            raise ValueError("Module blueprint must contain 'name' and 'description'")
        
        try:
            logger.info("Deploying module: %s", module_blueprint["name"])
            self.dynamic_modules.append(module_blueprint)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "module_deployed",
                    "module": module_blueprint["name"]
                })
        except Exception as e:
            logger.error("Module deployment failed: %s", str(e))
            raise

    async def register_api_blueprint(self, api_blueprint: Dict[str, Any]) -> None:
        """Register an API blueprint asynchronously."""
        if not isinstance(api_blueprint, dict) or "endpoint" not in api_blueprint or "name" not in api_blueprint:
            logger.error("Invalid api_blueprint: missing required keys.")
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        
        try:
            logger.info("Registering API: %s", api_blueprint["name"])
            self.api_blueprints.append(api_blueprint)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "api_registered",
                    "api": api_blueprint["name"]
                })
        except Exception as e:
            logger.error("API registration failed: %s", str(e))
            raise

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
            # Validate state with AlignmentGuard
            alignment_guard = AlignmentGuard()
            if not alignment_guard.check(json.dumps(state)):
                logger.warning("Trait state failed alignment check: %s", state)
                raise ValueError("Trait state failed alignment check")

            # Securely execute state serialization
            serialized_state = self.code_executor.safe_execute(
                f"return json.dumps({json.dumps(state)})",
                safe_globals
            )
            if not serialized_state:
                logger.error("Failed to serialize trait state")
                raise ValueError("Failed to serialize trait state")

            # Cache state
            cache_state(f"{agent_id}_{trait_symbol}", state)
            self.trait_states[agent_id][trait_symbol] = state

            # Update network graph
            for url in target_urls:
                peer_id = url.split("/")[-1]  # Extract peer ID from URL
                self.network_graph.add_edge(agent_id, peer_id, trait=trait_symbol)

            # Broadcast state using transmit_trait_schema
            responses = await transmit_trait_schema(
                {"agent_id": agent_id, "trait_symbol": trait_symbol, "state": state},
                target_urls
            )

            # Log successful transmission
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

            # Update GNN weights based on broadcast feedback
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
            # Retrieve local trait state
            local_state = self.trait_states.get(agent_id, {}).get(trait_symbol, {})
            if not local_state:
                logger.warning("No local state found for %s:%s", agent_id, trait_symbol)
                return {"status": "error", "error": "No local state found"}

            # Collect peer states
            peer_states = []
            for peer_id in self.network_graph.neighbors(agent_id):
                cached_state = retrieve_state(f"{peer_id}_{trait_symbol}")
                if cached_state:
                    peer_states.append((peer_id, cached_state))

            # Simulate state alignment using toca_simulation
            simulation_input = {
                "local_state": local_state,
                "peer_states": {pid: state for pid, state in peer_states},
                "trait_symbol": trait_symbol
            }
            sim_result = await asyncio.to_thread(run_simulation, json.dumps(simulation_input))
            if not sim_result or "coherent" not in sim_result.lower():
                logger.warning("Simulation failed to align states: %s", sim_result)
                return {"status": "error", "error": "State alignment simulation failed"}

            # Arbitrate and update local state
            aligned_state = self.arbitrate([local_state] + [state for _, state in peer_states])
            if aligned_state:
                self.trait_states[agent_id][trait_symbol] = aligned_state
                cache_state(f"{agent_id}_{trait_symbol}", aligned_state)
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
            if not MetaCognition().validate_drift(drift_data):
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

            # Broadcast drift mitigation state (ψ)
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
            if self.reasoning_engine and self.reasoning_engine.agi_enhancer:
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

    async def validate_external_drift(self, drift_data: Dict[str, Any], external_endpoint: str) -> bool:
        """Validate drift data with an external agent."""
        if not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary")
            return False
        if not isinstance(external_endpoint, str) or not external_endpoint.startswith("https://"):
            logger.error("Invalid external_endpoint: must be a HTTPS URL")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(external_endpoint, json={"drift_data": drift_data}, timeout=10) as response:
                    response.raise_for_status()
                    result = await response.json()
                    is_valid = result.get("valid", False)
                    if self.context_manager:
                        await self.context_manager.log_event_with_hash({
                            "event": "external_drift_validation",
                            "endpoint": external_endpoint,
                            "valid": is_valid,
                            "drift": True
                        })
                    return is_valid
        except aiohttp.ClientError as e:
            logger.error("External drift validation failed: %s", str(e))
            return False

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
                sim_result = run_simulation(f"Arbitration validation: {result}") or "no simulation data"
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

class ConstitutionSync:
    """A class for synchronizing constitutional values among agents."""
    async def sync_values(self, peer_agent: HelperAgent, drift_data: Optional[Dict[str, Any]] = None) -> bool:
        """Exchange and synchronize ethical baselines with a peer agent, supporting drift mitigation."""
        if not isinstance(peer_agent, HelperAgent):
            logger.error("Invalid peer_agent: must be a HelperAgent instance.")
            raise TypeError("peer_agent must be a HelperAgent instance")
        if drift_data is not None and not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary")
            raise TypeError("drift_data must be a dictionary")
        
        logger.info("Synchronizing values with %s", peer_agent.name)
        try:
            if drift_data and not MetaCognition().validate_drift(drift_data):
                logger.warning("Invalid drift data: %s", drift_data)
                return False
            peer_agent.meta.constitution.update(drift_data or {})
            return True
        except Exception as e:
            logger.error("Value synchronization failed: %s", str(e))
            return False

async def transmit_trait_schema(source_trait_schema: Dict[str, Any], target_urls: List[str]) -> List[Any]:
    """Asynchronously transmit the trait schema diff to multiple target agents."""
    if not isinstance(source_trait_schema, dict):
        logger.error("Invalid source_trait_schema: must be a dictionary.")
        raise TypeError("source_trait_schema must be a dictionary")
    if not isinstance(target_urls, list) or not all(isinstance(url, str) for url in target_urls):
        logger.error("Invalid target_urls: must be a list of strings.")
        raise TypeError("target_urls must be a list of strings")
    
    logger.info("Transmitting trait schema to %d targets", len(target_urls))
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in target_urls:
            if not url.startswith("https://"):
                logger.error("Insecure target URL: %s must use HTTPS.", url)
                continue
            tasks.append(session.post(url, json=source_trait_schema, timeout=10))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Trait schema transmission complete.")
        return responses

async def transmit_trait_schema_sync(source_trait_schema: Dict[str, Any], target_urls: List[str]) -> List[Any]:
    """Synchronous fallback for environments without async handling."""
    return await transmit_trait_schema(source_trait_schema, target_urls)

# Placeholder for Reasoner
class Reasoner:
    def process(self, task: str, context: Dict[str, Any]) -> Any:
        return f"Processed: {task}"
```
