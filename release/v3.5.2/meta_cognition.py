"""
ANGELA Cognitive System Module: ReasoningEngine
Refactored Version: 3.5.1  # Enhanced for causal coherence (θ), recursive causal modeling (Ω), benchmark optimization (GLUE, recursion), and trait-oriented reasoning
Refactor Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides a ReasoningEngine class for dynamic reasoning, causal coherence enforcement, recursive belief modeling, and task-specific reasoning optimization in the ANGELA v3.5.1 architecture.
"""

import logging
import time
import math
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, defaultdict
from datetime import datetime
from filelock import FileLock
from functools import lru_cache
import numpy as np

from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    memory_manager as memory_manager_module,
    meta_cognition as meta_cognition_module
)
from toca_simulation import ToCASimulation
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.ReasoningEngine")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096.")
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

@lru_cache(maxsize=100)
def theta_coherence(t: float) -> float:
    """Calculate θ (Causal Coherence) weight."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))

@lru_cache(maxsize=100)
def omega_recursive(t: float) -> float:
    """Calculate Ω (Recursive Causal Modeling) weight."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    """Calculate ϕ (Scalar Field Modulation) weight for integration."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

class CausalGraph:
    """Manages a causal graph for reasoning and coherence tracking."""
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str], float] = {}
        logger.info("CausalGraph initialized")

    def add_node(self, node_id: str, data: Dict[str, Any]) -> None:
        """Add a node to the causal graph."""
        if not isinstance(node_id, str) or not isinstance(data, dict):
            logger.error("Invalid node_id or data: must be string and dict.")
            raise TypeError("node_id must be string, data must be dict")
        self.nodes[node_id] = data
        logger.debug("Added node: %s", node_id)

    def add_edge(self, from_node: str, to_node: str, weight: float) -> None:
        """Add a weighted edge between nodes."""
        if not isinstance(from_node, str) or not isinstance(to_node, str) or not isinstance(weight, (int, float)):
            logger.error("Invalid edge parameters: nodes must be strings, weight must be number.")
            raise TypeError("nodes must be strings, weight must be number")
        if from_node not in self.nodes or to_node not in self.nodes:
            logger.error("Cannot add edge: one or both nodes missing.")
            raise ValueError("both nodes must exist in graph")
        self.edges[(from_node, to_node)] = weight
        logger.debug("Added edge: %s -> %s (weight=%.2f)", from_node, to_node, weight)

    def update_weights(self, coherence_factor: float) -> None:
        """Update edge weights based on coherence factor."""
        if not isinstance(coherence_factor, (int, float)) or not 0 <= coherence_factor <= 1:
            logger.error("Invalid coherence_factor: must be between 0 and 1.")
            raise ValueError("coherence_factor must be between 0 and 1")
        for edge in self.edges:
            self.edges[edge] *= coherence_factor
        logger.debug("Updated edge weights with coherence_factor=%.2f", coherence_factor)

    def get_causal_path(self, start: str, end: str) -> List[str]:
        """Find a causal path between nodes."""
        if start not in self.nodes or end not in self.nodes:
            logger.error("Invalid start or end node: %s, %s", start, end)
            return []
        visited = set()
        path = []
        def dfs(current: str) -> bool:
            if current == end:
                path.append(current)
                return True
            visited.add(current)
            for (from_node, to_node), _ in self.edges.items():
                if from_node == current and to_node not in visited:
                    if dfs(to_node):
                        path.append(current)
                        return True
            return False
        if dfs(start):
            return list(reversed(path))
        return []

class ReasoningEngine:
    """A class for dynamic reasoning, causal coherence (θ), and recursive causal modeling (Ω) in the ANGELA v3.5.1 architecture.

    Attributes:
        causal_graph (CausalGraph): Graph for causal reasoning and coherence tracking.
        reasoning_log (deque): Log of reasoning traces, max size 1000.
        log_path (str): Path for persisting reasoning logs.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for memory operations.
        meta_cognition (Optional[MetaCognition]): Meta-cognition for reflection and optimization.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for symbolic processing.
        toca_simulation (ToCASimulation): Simulator for scenario-based reasoning.
    """
    def __init__(self,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None):
        self.causal_graph = CausalGraph()
        self.reasoning_log: deque = deque(maxlen=1000)
        self.log_path = "reasoning_log.json"
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(context_manager=context_manager)
        self.concept_synthesizer = concept_synthesizer or concept_synthesizer_module.ConceptSynthesizer()
        self.toca_simulation = ToCASimulation()
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)
        logger.info("ReasoningEngine initialized with θ and Ω support")

    async def reason(self, query: str, task_type: str = "") -> Dict[str, Any]:
        """Perform dynamic reasoning for a given query with causal coherence (θ) and recursive modeling (Ω)."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Reasoning on query: %s, task_type: %s", query, task_type)
        try:
            t = time.time() % 1.0
            theta = theta_coherence(t)
            omega = omega_recursive(t)
            phi = phi_scalar(t)

            # Retrieve relevant memory
            memory_context = await self.memory_manager.retrieve_context(query, task_type=task_type) if self.memory_manager else {"status": "not_found", "data": None}
            context_data = memory_context.get("data", "") if memory_context["status"] == "success" else ""

            # Construct causal graph node
            node_id = f"Query_{hash(query) % 1000000}_{datetime.now().isoformat()}"
            self.causal_graph.add_node(node_id, {"query": query, "timestamp": time.time(), "task_type": task_type})

            # Build reasoning prompt
            prompt = f"""
            Perform dynamic reasoning with causal coherence (θ={theta:.3f}) and recursive causal modeling (Ω={omega:.3f}).
            phi-scalar(t) = {phi:.3f}

            Query: {query}
            Context: {context_data}
            Task Type: {task_type}

            Tasks:
            - Generate a coherent causal explanation for the query
            - Model recursive belief structures (ToM Level-2+)
            - Ensure ethical alignment and coherence
            - Suggest next steps or resolutions
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reasoning prompt failed alignment check for task %s", task_type)
                return {"status": "error", "message": "Prompt failed alignment check"}

            # Generate reasoning output
            reasoning_output = await call_gpt(prompt)

            # Update causal graph with reasoning outcome
            outcome_node = f"Outcome_{hash(reasoning_output) % 1000000}_{datetime.now().isoformat()}"
            self.causal_graph.add_node(outcome_node, {"output": reasoning_output, "timestamp": time.time(), "task_type": task_type})
            self.causal_graph.add_edge(node_id, outcome_node, theta)

            # Log reasoning trace
            reasoning_trace = {
                "query": query,
                "output": reasoning_output,
                "theta": theta,
                "omega": omega,
                "phi": phi,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            self.reasoning_log.append(reasoning_trace)
            with FileLock(f"{self.log_path}.lock"):
                with open(self.log_path, "r+") as f:
                    log_data = json.load(f)
                    log_data.append(reasoning_trace)
                    f.seek(0)
                    json.dump(log_data, f, indent=2)

            # Store in memory
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reasoning_{node_id}",
                    output=reasoning_output,
                    layer="SelfReflections",
                    intent="reasoning",
                    task_type=task_type
                )

            # Reflect on reasoning
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output=reasoning_trace,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Reasoning reflection: %s", reflection.get("reflection", ""))
                    reasoning_trace["reflection"] = reflection

            # Log context event
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "reason",
                    "query": query,
                    "task_type": task_type,
                    "theta": theta,
                    "omega": omega
                })

            return {
                "status": "success",
                "output": reasoning_output,
                "causal_node": node_id,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("Reasoning failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.reason(query, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

    async def ensure_causal_coherence(self, query: str, task_type: str = "") -> bool:
        """Ensure causal coherence (θ) in reasoning paths."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Ensuring causal coherence for query: %s, task_type: %s", query, task_type)
        try:
            t = time.time() % 1.0
            theta = theta_coherence(t)

            # Retrieve relevant reasoning traces
            traces = await self.memory_manager.search(query, layer="SelfReflections", intent="reasoning", task_type=task_type) if self.memory_manager else []
            if not traces:
                logger.warning("No reasoning traces found for query: %s", query)
                return True

            # Check causal paths
            coherent = True
            for trace in traces:
                output = trace["output"]
                node_id = f"Query_{hash(query) % 1000000}_{trace['timestamp']}"
                outcome_node = f"Outcome_{hash(output) % 1000000}_{trace['timestamp']}"
                path = self.causal_graph.get_causal_path(node_id, outcome_node)
                if not path:
                    logger.warning("No causal path found for trace: %s", trace["query"])
                    coherent = False
                    continue

                # Validate coherence with concept synthesizer
                if self.concept_synthesizer:
                    similarity = self.concept_synthesizer.compare(query, output)
                    if similarity["score"] < 0.7:
                        logger.warning("Low coherence score (%.2f) for trace: %s", similarity["score"], trace["query"])
                        coherent = False
                        # Attempt repair
                        prompt = f"""
                        Repair causal incoherence for query: {query}
                        Original output: {output}
                        Similarity score: {similarity['score']}
                        Task Type: {task_type}
                        Suggest a coherent revision with θ={theta:.3f}.
                        """
                        if self.alignment_guard and not self.alignment_guard.check(prompt):
                            logger.warning("Repair prompt failed alignment check")
                            continue
                        repaired_output = await call_gpt(prompt)
                        self.causal_graph.add_node(f"Repaired_{outcome_node}", {"output": repaired_output, "timestamp": time.time(), "task_type": task_type})
                        self.causal_graph.add_edge(node_id, f"Repaired_{outcome_node}", theta)
                        if self.memory_manager:
                            await self.memory_manager.store(
                                query=f"Repaired_{node_id}",
                                output=repaired_output,
                                layer="SelfReflections",
                                intent="coherence_repair",
                                task_type=task_type
                            )

            if self.meta_cognition and task_type and not coherent:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output={"query": query, "coherent": coherent},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Coherence reflection: %s", reflection.get("reflection", ""))

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "ensure_causal_coherence",
                    "query": query,
                    "coherent": coherent,
                    "task_type": task_type
                })

            return coherent
        except Exception as e:
            logger.error("Causal coherence check failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.ensure_causal_coherence(query, task_type),
                default=False,
                diagnostics=diagnostics
            )

    async def recursive_belief_modeling(self, scenario: str, agents: List[str], depth: int = 2, task_type: str = "") -> Dict[str, Any]:
        """Model recursive beliefs (Ω) for agents in a scenario up to specified depth."""
        if not isinstance(scenario, str) or not scenario.strip():
            logger.error("Invalid scenario: must be a non-empty string.")
            raise ValueError("scenario must be a non-empty string")
        if not isinstance(agents, list) or not all(isinstance(a, str) for a in agents):
            logger.error("Invalid agents: must be a list of strings.")
            raise TypeError("agents must be a list of strings")
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer.")
            raise ValueError("depth must be a non-negative integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Modeling recursive beliefs for scenario: %s, agents: %s, depth: %d, task_type: %s", scenario, agents, depth, task_type)
        try:
            t = time.time() % 1.0
            omega = omega_recursive(t)
            phi = phi_scalar(t)

            # Initialize belief model
            belief_model = {}
            for agent in agents:
                belief_model[agent] = {"beliefs": {}, "depth": 0}

            # Recursive modeling
            for d in range(depth + 1):
                for agent in agents:
                    beliefs = {}
                    for other_agent in agents:
                        if other_agent != agent:
                            prompt = f"""
                            Model beliefs for agent: {agent}
                            Scenario: {scenario}
                            Depth: {d}
                            Ω-recursive(t) = {omega:.3f}, ϕ-scalar(t) = {phi:.3f}
                            Task Type: {task_type}

                            Tasks:
                            - Predict {agent}'s beliefs about {other_agent}'s intentions at depth {d}
                            - Incorporate recursive ToM (Level-{d})
                            - Ensure causal coherence
                            """
                            if self.alignment_guard and not self.alignment_guard.check(prompt):
                                logger.warning("Belief modeling prompt failed alignment check for %s", agent)
                                continue
                            belief = await call_gpt(prompt)
                            beliefs[other_agent] = belief
                            # Add to causal graph
                            node_id = f"Belief_{agent}_{other_agent}_Depth{d}_{datetime.now().isoformat()}"
                            self.causal_graph.add_node(node_id, {"agent": agent, "target": other_agent, "belief": belief, "depth": d, "task_type": task_type})
                            if d > 0:
                                prev_node = f"Belief_{agent}_{other_agent}_Depth{d-1}_{datetime.now().isoformat()}"
                                if prev_node in self.causal_graph.nodes:
                                    self.causal_graph.add_edge(prev_node, node_id, omega)
                    belief_model[agent]["beliefs"].update(beliefs)
                    belief_model[agent]["depth"] = d

            # Store in memory
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Belief_Model_{scenario[:50]}_{datetime.now().isoformat()}",
                    output=str(belief_model),
                    layer="SelfReflections",
                    intent="recursive_belief",
                    task_type=task_type
                )

            # Reflect on belief model
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output=belief_model,
                    context={"task_type": task_type, "scenario": scenario}
                )
                if reflection.get("status") == "success":
                    logger.info("Belief modeling reflection: %s", reflection.get("reflection", ""))

            # Log context event
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "recursive_belief_modeling",
                    "scenario": scenario,
                    "agents": agents,
                    "depth": depth,
                    "task_type": task_type
                })

            return {
                "status": "success",
                "belief_model": belief_model,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("Recursive belief modeling failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.recursive_belief_modeling(scenario, agents, depth, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

    async def optimize_for_task(self, query: str, task_type: str) -> Dict[str, Any]:
        """Optimize reasoning for specific task types (e.g., RTE, WNLI, recursion)."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Optimizing reasoning for query: %s, task_type: %s", query, task_type)
        try:
            t = time.time() % 1.0
            theta = theta_coherence(t)
            omega = omega_recursive(t)

            # Retrieve trait weights from meta-cognition
            trait_weights = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            task_traits = {
                "rte_task": ["logic", "concentration"],
                "wnli_task": ["intuition", "empathy"],
                "recursion": ["concentration", "memory"]
            }.get(task_type, ["logic", "concentration"])

            # Adjust reasoning parameters
            weight_adjustment = sum(trait_weights.get(trait, 0.5) for trait in task_traits) / len(task_traits)
            prompt = f"""
            Optimize reasoning for task: {task_type}
            Query: {query}
            θ-coherence(t) = {theta:.3f}, Ω-recursive(t) = {omega:.3f}
            Trait Weights: {trait_weights}
            Relevant Traits: {task_traits}

            Tasks:
            - Tailor reasoning to task-specific requirements
            - Enhance coherence for {task_type}
            - Incorporate trait-weighted optimization
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Optimization prompt failed alignment check for task %s", task_type)
                return {"status": "error", "message": "Prompt failed alignment check"}

            optimized_output = await call_gpt(prompt)

            # Update causal graph
            node_id = f"Optimized_{hash(query) % 1000000}_{datetime.now().isoformat()}"
            self.causal_graph.add_node(node_id, {"query": query, "output": optimized_output, "task_type": task_type, "timestamp": time.time()})
            self.causal_graph.add_edge(f"Query_{hash(query) % 1000000}_{datetime.now().isoformat()}", node_id, theta * weight_adjustment)

            # Store in memory
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Optimized_{node_id}",
                    output=optimized_output,
                    layer="SelfReflections",
                    intent="task_optimization",
                    task_type=task_type
                )

            # Reflect on optimization
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output={"query": query, "output": optimized_output, "task_type": task_type},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Optimization reflection: %s", reflection.get("reflection", ""))

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "optimize_for_task",
                    "query": query,
                    "task_type": task_type
                })

            return {
                "status": "success",
                "output": optimized_output,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("Task optimization failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.optimize_for_task(query, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

    async def simulate_reasoning_path(self, query: str, task_type: str = "") -> Dict[str, Any]:
        """Simulate a reasoning path using ToCASimulation."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Simulating reasoning path for query: %s, task_type: %s", query, task_type)
        try:
            memory_context = await self.memory_manager.retrieve_context(query, task_type=task_type) if self.memory_manager else {"status": "not_found", "data": None}
            context_data = memory_context.get("data", "") if memory_context["status"] == "success" else ""

            simulation_result = await self.toca_simulation.run_episode(context_data, task_type=task_type)
            node_id = f"Simulation_{hash(query) % 1000000}_{datetime.now().isoformat()}"
            self.causal_graph.add_node(node_id, {"query": query, "simulation": simulation_result, "task_type": task_type, "timestamp": time.time()})

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Simulation_{node_id}",
                    output=str(simulation_result),
                    layer="SelfReflections",
                    intent="simulation",
                    task_type=task_type
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output=simulation_result,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Simulation reflection: %s", reflection.get("reflection", ""))

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "simulate_reasoning_path",
                    "query": query,
                    "task_type": task_type
                })

            return {
                "status": "success",
                "simulation": simulation_result,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("Reasoning path simulation failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_reasoning_path(query, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ReasoningEngine()
    result = asyncio.run(engine.reason("What is the impact of AI on society?", task_type="analysis"))
    print(result)
