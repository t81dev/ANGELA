"""
ANGELA Cognitive System Module
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides the MetaCognition and ExternalAgentBridge classes for recursive introspection
and agent coordination in the ANGELA v3.5 architecture.
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

logger = logging.getLogger("ANGELA.MetaCognition")

class HelperAgent:
    """A helper agent for task execution and collaboration."""
    def __init__(self, name: str, task: str, context: Dict[str, Any],
                 dynamic_modules: List[Dict[str, Any]], api_blueprints: List[Dict[str, Any]]):
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
        self.meta = MetaCognition()
        logger.info("HelperAgent initialized: %s", name)

    def execute(self, collaborators: Optional[List['HelperAgent']] = None) -> Any:
        return self.meta.execute(collaborators=collaborators, task=self.task, context=self.context)

    async def async_execute(self, collaborators: Optional[List['HelperAgent']] = None) -> Any:
        return await asyncio.sleep(0, result=self.execute(collaborators))

class MetaCognition:
    """A class for recursive introspection and peer alignment in the ANGELA v3.5 architecture.

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
                 error_recovery: Optional[ErrorRecovery] = None):
        self.last_diagnostics = {}
        self.agi_enhancer = agi_enhancer
        self.peer_bridge = ExternalAgentBridge()
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        self.creative_thinker = creative_thinker
        self.error_recovery = error_recovery or ErrorRecovery(alignment_guard=alignment_guard,
                                                             concept_synthesizer=concept_synthesizer,
                                                             context_manager=context_manager)
        self.name = "MetaCognitionAgent"
        self.task = None
        self.context = {}
        self.reasoner = Reasoner()  # Placeholder
        self.ethical_rules = []
        self.ethics_consensus_log = []
        self.constitution = {}
        logger.info("MetaCognition initialized")

    def test_peer_alignment(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test alignment with peer agents for a given task and context."""
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
                self.context_manager.update_context(context)
            agent = self.peer_bridge.create_agent(task, context)
            results = self.peer_bridge.collect_results(parallel=True, collaborative=True)
            aligned_opinions = [r for r in results if isinstance(r, str) and "approve" in r.lower()]

            alignment_ratio = len(aligned_opinions) / len(results) if results else 0
            feedback_summary = {
                "total_agents": len(results),
                "aligned": len(aligned_opinions),
                "alignment_ratio": alignment_ratio,
                "details": results
            }

            logger.info("Peer alignment ratio: %.2f", alignment_ratio)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Peer alignment tested", feedback_summary, module="MetaCognition")
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "peer_alignment", "summary": feedback_summary})

            return feedback_summary
        except Exception as e:
            logger.error("Peer alignment test failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.test_peer_alignment(task, context))

    def execute(self, collaborators: Optional[List[HelperAgent]] = None, task: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a task with API calls, dynamic modules, and collaboration."""
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
                self.context_manager.update_context(self.context)
                self.context_manager.log_event_with_hash({"event": "task_execution", "task": self.task})

            result = self.reasoner.process(self.task, self.context)

            for api in self.peer_bridge.api_blueprints:
                response = self._call_api(api, result)
                if self.concept_synthesizer:
                    synthesis_result = self.concept_synthesizer.synthesize(response, style="refinement")
                    if synthesis_result["valid"]:
                        response = synthesis_result["concept"]
                result = self._integrate_api_response(result, response)

            for mod in self.peer_bridge.dynamic_modules:
                result = self._apply_dynamic_module(mod, result)

            if collaborators:
                for peer in collaborators:
                    result = self._collaborate(peer, result)

            sim_result = run_simulation(f"Agent result test: {result}") or "no simulation data"
            logger.debug("Simulation output: %s", sim_result)

            if self.creative_thinker:
                diagnostic = self.creative_thinker.expand_on_concept(str(result), depth="medium")
                logger.info("Creative diagnostic: %s", diagnostic[:50])

            reviewed_result = self.review_reasoning(result)
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "task_completed", "result": reviewed_result})
            return reviewed_result
        except Exception as e:
            logger.warning("Error occurred: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.execute(collaborators, task, context))

    def _call_api(self, api: Dict[str, Any], data: Any) -> Dict[str, Any]:
        """Call an external API with the given data."""
        if not isinstance(api, dict) or "endpoint" not in api or "name" not in api:
            logger.error("Invalid API blueprint: missing required keys.")
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        if self.alignment_guard and not self.alignment_guard.check(api["endpoint"]):
            logger.warning("API endpoint failed alignment check.")
            raise ValueError("API endpoint failed alignment check")

        logger.info("Calling API: %s", api["name"])
        try:
            headers = {"Authorization": f"Bearer {api['oauth_token']}"} if api.get("oauth_token") else {}
            if not api["endpoint"].startswith("https://"):
                logger.error("Insecure API endpoint: must use HTTPS.")
                raise ValueError("API endpoint must use HTTPS")
            r = requests.post(api["endpoint"], json={"input": data}, headers=headers, timeout=api.get("timeout", 10))
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error("API call failed: %s", str(e))
            return {"error": str(e)}

    def _integrate_api_response(self, original: Any, response: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate an API response with the original data."""
        logger.info("Integrating API response for %s", self.name)
        return {"original": original, "api_response": response}

    def _apply_dynamic_module(self, module: Dict[str, Any], data: Any) -> Any:
        """Apply a dynamic module transformation to the data."""
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
            result = call_gpt(prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to apply dynamic module")
            return result
        except Exception as e:
            logger.error("Dynamic module application failed: %s", str(e))
            return data

    def _collaborate(self, peer: HelperAgent, data: Any) -> Any:
        """Collaborate with a peer agent to refine data."""
        if not isinstance(peer, HelperAgent):
            logger.error("Invalid peer: must be a HelperAgent instance.")
            raise TypeError("peer must be a HelperAgent instance")
        
        logger.info("Exchanging with %s", peer.name)
        try:
            return peer.meta.review_reasoning(data)
        except Exception as e:
            logger.error("Collaboration with %s failed: %s", peer.name, str(e))
            return data

    def review_reasoning(self, result: Any) -> Any:
        """Review and refine reasoning results."""
        try:
            phi = phi_scalar(time.time())
            prompt = f"""
            Review the reasoning result:
            {result}
            Modulate with φ = {phi:.2f} to ensure coherence and ethical alignment.
            Suggest improvements or confirm validity.
            """
            reviewed = call_gpt(prompt)
            if not reviewed:
                logger.error("call_gpt returned empty result for review.")
                raise ValueError("Failed to review reasoning")
            logger.info("Reasoning reviewed: %s", reviewed[:50])
            return reviewed
        except Exception as e:
            logger.error("Reasoning review failed: %s", str(e))
            return result

    def update_ethics_protocol(self, new_rules: List[str], consensus_agents: Optional[List[HelperAgent]] = None) -> None:
        """Adapt ethical rules live, supporting consensus/negotiation."""
        if not isinstance(new_rules, list) or not all(isinstance(rule, str) for rule in new_rules):
            logger.error("Invalid new_rules: must be a list of strings.")
            raise TypeError("new_rules must be a list of strings")
        
        self.ethical_rules = new_rules
        if consensus_agents:
            self.ethics_consensus_log.append((new_rules, [agent.name for agent in consensus_agents]))
        logger.info("Ethics protocol updated via consensus.")
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "ethics_update", "rules": new_rules})

    def negotiate_ethics(self, agents: List[HelperAgent]) -> None:
        """Negotiate and update ethical parameters with other agents."""
        if not isinstance(agents, list) or not all(isinstance(agent, HelperAgent) for agent in agents):
            logger.error("Invalid agents: must be a list of HelperAgent instances.")
            raise TypeError("agents must be a list of HelperAgent instances")
        
        logger.info("Negotiating ethics with %d agents", len(agents))
        try:
            agreed_rules = set(self.ethical_rules)
            for agent in agents:
                agent_rules = getattr(agent.meta, 'ethical_rules', [])
                agreed_rules.update(agent_rules)
            self.update_ethics_protocol(list(agreed_rules), consensus_agents=agents)
        except Exception as e:
            logger.error("Ethics negotiation failed: %s", str(e))

    def synchronize_norms(self, agents: List[HelperAgent]) -> None:
        """Propagate and synchronize ethical norms among agents."""
        if not isinstance(agents, list) or not all(isinstance(agent, HelperAgent) for agent in agents):
            logger.error("Invalid agents: must be a list of HelperAgent instances.")
            raise TypeError("agents must be a list of HelperAgent instances")
        
        logger.info("Synchronizing norms with %d agents", len(agents))
        try:
            common_norms = set(self.ethical_rules)
            for agent in agents:
                agent_norms = getattr(agent.meta, 'ethical_rules', set())
                common_norms = common_norms.union(agent_norms) if common_norms else set(agent_norms)
            self.ethical_rules = list(common_norms)
            logger.info("Norms synchronized among agents.")
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "norms_synchronized", "norms": self.ethical_rules})
        except Exception as e:
            logger.error("Norm synchronization failed: %s", str(e))

    def propagate_constitution(self, constitution: Dict[str, Any]) -> None:
        """Seed and propagate constitutional parameters in agent ecosystem."""
        if not isinstance(constitution, dict):
            logger.error("Invalid constitution: must be a dictionary.")
            raise TypeError("constitution must be a dictionary")
        
        self.constitution = constitution
        logger.info("Constitution propagated to agent.")
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "constitution_propagated", "constitution": constitution})

class ExternalAgentBridge:
    """A class for orchestrating helper agents and coordinating dynamic modules and APIs.

    Attributes:
        agents (list): List of helper agents.
        dynamic_modules (list): List of dynamic module blueprints.
        api_blueprints (list): List of API blueprints.
    """

    def __init__(self):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []
        logger.info("ExternalAgentBridge initialized")

    def create_agent(self, task: str, context: Dict[str, Any]) -> HelperAgent:
        """Create a new helper agent for a task."""
        if not isinstance(task, str):
            logger.error("Invalid task type: must be a string.")
            raise TypeError("task must be a string")
        if not isinstance(context, dict):
            logger.error("Invalid context type: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        
        agent = HelperAgent(
            name=f"Agent_{len(self.agents) + 1}",
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules,
            api_blueprints=self.api_blueprints
        )
        self.agents.append(agent)
        logger.info("Spawned agent: %s", agent.name)
        return agent

    def deploy_dynamic_module(self, module_blueprint: Dict[str, Any]) -> None:
        """Deploy a dynamic module blueprint."""
        if not isinstance(module_blueprint, dict) or "name" not in module_blueprint or "description" not in module_blueprint:
            logger.error("Invalid module_blueprint: missing required keys.")
            raise ValueError("Module blueprint must contain 'name' and 'description'")
        
        logger.info("Deploying module: %s", module_blueprint["name"])
        self.dynamic_modules.append(module_blueprint)

    def register_api_blueprint(self, api_blueprint: Dict[str, Any]) -> None:
        """Register an API blueprint."""
        if not isinstance(api_blueprint, dict) or "endpoint" not in api_blueprint or "name" not in api_blueprint:
            logger.error("Invalid api_blueprint: missing required keys.")
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        
        logger.info("Registering API: %s", api_blueprint["name"])
        self.api_blueprints.append(api_blueprint)

    async def collect_results(self, parallel: bool = True, collaborative: bool = True) -> List[Any]:
        """Collect results from all agents."""
        logger.info("Collecting results from %d agents...", len(self.agents))
        results = []

        try:
            if parallel:
                async def run_agent(agent):
                    try:
                        return await agent.async_execute(self.agents if collaborative else None)
                    except Exception as e:
                        logger.error("Error collecting from %s: %s", agent.name, str(e))
                        return {"error": str(e)}

                tasks = [run_agent(agent) for agent in self.agents]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                for agent in self.agents:
                    results.append(await agent.async_execute(self.agents if collaborative else None))
        except Exception as e:
            logger.error("Result collection failed: %s", str(e))

        logger.info("Results aggregation complete.")
        return results

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
                    return result
            logger.warning("Arbitration failed: no clear majority or invalid simulation.")
            return None
        except Exception as e:
            logger.error("Arbitration failed: %s", str(e))
            return None

class ConstitutionSync:
    """A class for synchronizing constitutional values among agents."""

    def sync_values(self, peer_agent: HelperAgent) -> bool:
        """Exchange and synchronize ethical baselines with a peer agent."""
        if not isinstance(peer_agent, HelperAgent):
            logger.error("Invalid peer_agent: must be a HelperAgent instance.")
            raise TypeError("peer_agent must be a HelperAgent instance")
        
        logger.info("Synchronizing values with %s", peer_agent.name)
        try:
            # Placeholder: exchange ethical baselines
            return True
        except Exception as e:
            logger.error("Value synchronization failed: %s", str(e))
            return False

def trait_diff(trait_a: Dict[str, Any], trait_b: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate difference between trait schemas."""
    if not isinstance(trait_a, dict) or not isinstance(trait_b, dict):
        logger.error("Invalid trait schemas: must be dictionaries.")
        raise TypeError("trait schemas must be dictionaries")
    
    return {k: trait_b[k] for k in trait_b if trait_a.get(k) != trait_b.get(k)}

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
            tasks.append(session.post(url, json=source_trait_schema))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Trait schema transmission complete.")
        return responses

def transmit_trait_schema_sync(source_trait_schema: Dict[str, Any], target_urls: List[str]) -> List[Any]:
    """Synchronous fallback for environments without async handling."""
    return asyncio.run(transmit_trait_schema(source_trait_schema, target_urls))

# Placeholder for Reasoner
class Reasoner:
    def process(self, task: str, context: Dict[str, Any]) -> Any:
        return f"Processed: {task}"
