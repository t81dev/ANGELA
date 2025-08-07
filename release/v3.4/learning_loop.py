"""
ANGELA Cognitive System Module: LearningLoop
Version: 3.5.1  # Enhanced for Task-Specific Learning, Real-Time Data, and Visualization
Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides a LearningLoop class for adaptive learning, goal activation, and module refinement
in the ANGELA v3.5.1 architecture.
"""

import logging
import time
import math
import asyncio
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime
from functools import lru_cache

from modules import (
    context_manager, concept_synthesizer, alignment_guard, error_recovery, meta_cognition, visualizer, memory_manager
)
from utils.prompt_utils import query_openai
from toca_simulation import run_simulation
import json

logger = logging.getLogger("ANGELA.LearningLoop")

async def call_gpt(prompt: str, task_type: str = "") -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed for task %s: %s", task_type, result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception for task %s: %s", task_type, str(e))
        raise

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def eta_feedback(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))

class LearningLoop:
    """A class for adaptive learning, goal activation, and module refinement in the ANGELA v3.5.1 architecture.

    Attributes:
        goal_history (deque): History of activated goals with max size 1000.
        module_blueprints (deque): Blueprints for deployed modules with max size 1000.
        meta_learning_rate (float): Learning rate for model updates.
        session_traces (deque): Traces of learning sessions with max size 1000.
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for pattern synthesis.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for caching and drift-aware storage.
        visualizer (Optional[Visualizer]): Visualizer for learning traces and goals.
        epistemic_revision_log (deque): Log of knowledge updates with max size 1000.
    """
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager.ContextManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer.ConceptSynthesizer'] = None,
                 alignment_guard: Optional['alignment_guard.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager.MemoryManager'] = None,
                 visualizer: Optional['visualizer.Visualizer'] = None):
        self.goal_history = deque(maxlen=1000)
        self.module_blueprints = deque(maxlen=1000)
        self.meta_learning_rate = 0.1
        self.session_traces = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager or memory_manager.MemoryManager()
        self.visualizer = visualizer or visualizer.Visualizer()
        self.epistemic_revision_log = deque(maxlen=1000)
        logger.info("LearningLoop initialized")

    async def integrate_external_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate external agent data or policies."""
        if not isinstance(data_source, str):
            logger.error("Invalid data_source: must be a string")
            raise TypeError("data_source must be a string")
        if not isinstance(data_type, str):
            logger.error("Invalid data_type: must be a string")
            raise TypeError("data_type must be a string")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            cache_key = f"ExternalData_{data_type}_{data_source}_{task_type}"
            cached_data = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type) if self.memory_manager else None
            if cached_data and "timestamp" in cached_data["data"]:
                cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                    logger.info("Returning cached external data for %s", cache_key)
                    return cached_data["data"]["data"]

            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/external_data?source={data_source}&type={data_type}&task_type={task_type}") as response:
                    if response.status != 200:
                        logger.error("Failed to fetch external data for task %s: %s", task_type, response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()

            if data_type == "agent_data":
                agent_data = data.get("agent_data", [])
                if not agent_data:
                    logger.error("No agent data provided for task %s", task_type)
                    return {"status": "error", "error": "No agent data"}
                result = {"status": "success", "agent_data": agent_data}
            elif data_type == "policy_data":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No policy data provided for task %s", task_type)
                    return {"status": "error", "error": "No policies"}
                result = {"status": "success", "policies": policies}
            else:
                logger.error("Unsupported data_type: %s for task %s", data_type, task_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="external_data_integration",
                    task_type=task_type
                )
            reflection = await meta_cognition.MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"data_type": data_type, "data": result},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("External data integration reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("External data integration failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.integrate_external_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e), "task_type": task_type}
            )

    async def activate_intrinsic_goals(self, meta_cognition: 'meta_cognition.MetaCognition', task_type: str = "") -> List[str]:
        """Activate intrinsic goals proposed by MetaCognition."""
        if not isinstance(meta_cognition, meta_cognition.MetaCognition):
            logger.error("Invalid meta_cognition: must be a MetaCognition instance.")
            raise TypeError("meta_cognition must be a MetaCognition instance")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Activating chi-intrinsic goals from MetaCognition for task %s", task_type)
        try:
            intrinsic_goals = await asyncio.to_thread(meta_cognition.infer_intrinsic_goals, task_type=task_type)
            activated = []
            for goal in intrinsic_goals:
                if not isinstance(goal, dict) or "intent" not in goal or "priority" not in goal:
                    logger.warning("Invalid goal format: %s for task %s", goal, task_type)
                    continue
                if goal["intent"] not in [g["goal"] for g in self.goal_history]:
                    simulation_result = await run_simulation(goal["intent"], task_type=task_type)
                    if isinstance(simulation_result, dict) and simulation_result.get("status") == "success":
                        self.goal_history.append({
                            "goal": goal["intent"],
                            "timestamp": time.time(),
                            "priority": goal["priority"],
                            "origin": "intrinsic",
                            "task_type": task_type
                        })
                        logger.info("Intrinsic goal activated: %s for task %s", goal["intent"], task_type)
                        if self.agi_enhancer:
                            await self.agi_enhancer.log_episode(
                                event="Intrinsic goal activated",
                                meta=goal,
                                module="LearningLoop",
                                tags=["goal", "intrinsic", task_type]
                            )
                        activated.append(goal["intent"])
                    else:
                        logger.warning("Rejected goal: %s (simulation failed) for task %s", goal["intent"], task_type)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "activate_intrinsic_goals",
                    "goals": activated,
                    "task_type": task_type
                })
            reflection = await meta_cognition.MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"activated_goals": activated},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Goal activation reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "goal_activation": {
                        "goals": activated,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"GoalActivation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(activated),
                    layer="Goals",
                    intent="goal_activation",
                    task_type=task_type
                )
            return activated
        except Exception as e:
            logger.error("Goal activation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.activate_intrinsic_goals(meta_cognition, task_type),
                default=[]
            )

    async def update_model(self, session_data: Dict[str, Any], task_type: str = "") -> None:
        """Update learning model with session data and trait modulation."""
        if not isinstance(session_data, dict):
            logger.error("Invalid session_data: must be a dictionary.")
            raise TypeError("session_data must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Analyzing session performance for task %s...", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            eta = eta_feedback(t)
            entropy = 0.1
            logger.debug("phi-scalar: %.3f, eta-feedback: %.3f, entropy: %.2f for task %s", phi, eta, entropy, task_type)

            modulation_index = ((phi + eta) / 2) + (entropy * (0.5 - abs(phi - eta)))
            self.meta_learning_rate = max(0.01, min(self.meta_learning_rate * (1 + modulation_index - 0.5), 1.0))

            external_data = await self.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            trace = {
                "timestamp": time.time(),
                "phi": phi,
                "eta": eta,
                "entropy": entropy,
                "modulation_index": modulation_index,
                "learning_rate": self.meta_learning_rate,
                "policies": policies,
                "task_type": task_type
            }
            self.session_traces.append(trace)

            tasks = [
                self._meta_learn(session_data, trace, task_type),
                self._find_weak_modules(session_data.get("module_stats", {}), task_type),
                self._detect_capability_gaps(session_data.get("input"), session_data.get("output"), task_type),
                self._consolidate_knowledge(task_type),
                self._check_narrative_integrity(task_type)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            weak_modules = results[1] if not isinstance(results[1], Exception) else []

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Model update",
                    meta=trace,
                    module="LearningLoop",
                    tags=["update", "learning", task_type]
                )

            if weak_modules:
                logger.warning("Weak modules detected: %s for task %s", weak_modules, task_type)
                await self._propose_module_refinements(weak_modules, trace, task_type)

            if self.context_manager:
                await self.context_manager.update_context({"session_data": session_data, "trace": trace}, task_type=task_type)
            reflection = await meta_cognition.MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"trace": trace},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Model update reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "model_update": {
                        "trace": trace,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ModelUpdate_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(trace),
                    layer="Sessions",
                    intent="model_update",
                    task_type=task_type
                )
        except Exception as e:
            logger.error("Model update failed for task %s: %s", task_type, str(e))
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.update_model(session_data, task_type)
            )

    async def propose_autonomous_goal(self, task_type: str = "") -> Optional[str]:
        """Propose a high-level, safe, phi-aligned autonomous goal."""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Proposing autonomous goal for task %s", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Propose a high-level, safe, phi-aligned autonomous goal based on recent session trends.
            phi = {phi:.2f}
            Task Type: {task_type}
            """
            valid, report = await self.alignment_guard.ethical_check(prompt, stage="goal_proposal", task_type=task_type) if self.alignment_guard else (True, {})
            if not valid:
                logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
                return None

            autonomous_goal = await call_gpt(prompt, task_type=task_type)
            if not autonomous_goal or autonomous_goal in [g["goal"] for g in self.goal_history]:
                logger.info("No new goal proposed for task %s", task_type)
                return None

            simulation_feedback = await run_simulation(f"Goal test: {autonomous_goal}", task_type=task_type)
            if isinstance(simulation_feedback, dict) and simulation_feedback.get("status") == "success":
                self.goal_history.append({
                    "goal": autonomous_goal,
                    "timestamp": time.time(),
                    "phi": phi,
                    "task_type": task_type
                })
                logger.info("Proposed autonomous goal: %s for task %s", autonomous_goal, task_type)
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        event="Autonomous goal proposed",
                        meta={"goal": autonomous_goal},
                        module="LearningLoop",
                        tags=["goal", "autonomous", task_type]
                    )
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "propose_autonomous_goal",
                        "goal": autonomous_goal,
                        "task_type": task_type
                    })
                reflection = await meta_cognition.MetaCognition().reflect_on_output(
                    component="LearningLoop",
                    output={"goal": autonomous_goal},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Goal proposal reflection: %s", reflection.get("reflection", ""))
                if self.visualizer and task_type:
                    plot_data = {
                        "goal_proposal": {
                            "goal": autonomous_goal,
                            "task_type": task_type
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise"
                        }
                    }
                    await self.visualizer.render_charts(plot_data)
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"AutonomousGoal_{time.strftime('%Y%m%d_%H%M%S')}",
                        output=autonomous_goal,
                        layer="Goals",
                        intent="goal_proposal",
                        task_type=task_type
                    )
                return autonomous_goal
            logger.warning("Goal failed simulation feedback: %s for task %s", autonomous_goal, task_type)
            return None
        except Exception as e:
            logger.error("Goal proposal failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.propose_autonomous_goal(task_type),
                default=None
            )

    async def _meta_learn(self, session_data: Dict[str, Any], trace: Dict[str, Any], task_type: str = "") -> None:
        """Adapt learning from phi/eta trace."""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Adapting learning from phi/eta trace for task %s", task_type)
        try:
            if self.concept_synthesizer:
                synthesized = await self.concept_synthesizer.generate(
                    concept_name="MetaLearning",
                    context={"session_data": session_data, "trace": trace, "task_type": task_type},
                    task_type=task_type
                )
                if synthesized.get("success"):
                    logger.debug("Synthesized meta-learning patterns: %s for task %s", synthesized["concept"], task_type)
                reflection = await meta_cognition.MetaCognition().reflect_on_output(
                    component="LearningLoop",
                    output={"synthesized": synthesized},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Meta-learning reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Meta-learning synthesis failed for task %s: %s", task_type, str(e))

    async def _find_weak_modules(self, module_stats: Dict[str, Dict[str, Any]], task_type: str = "") -> List[str]:
        """Identify modules with low success rates."""
        if not isinstance(module_stats, dict):
            logger.error("Invalid module_stats: must be a dictionary.")
            raise TypeError("module_stats must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        weak_modules = [
            module for module, stats in module_stats.items()
            if isinstance(stats, dict) and stats.get("calls", 0) > 0
            and (stats.get("success", 0) / stats["calls"]) < 0.8
        ]
        if weak_modules and self.memory_manager:
            await self.memory_manager.store(
                query=f"WeakModules_{time.strftime('%Y%m%d_%H%M%S')}",
                output=str(weak_modules),
                layer="Modules",
                intent="module_analysis",
                task_type=task_type
            )
        return weak_modules

    async def _propose_module_refinements(self, weak_modules: List[str], trace: Dict[str, Any], task_type: str = "") -> None:
        """Propose refinements for weak modules."""
        if not isinstance(weak_modules, list) or not all(isinstance(m, str) for m in weak_modules):
            logger.error("Invalid weak_modules: must be a list of strings.")
            raise TypeError("weak_modules must be a list of strings")
        if not isinstance(trace, dict):
            logger.error("Invalid trace: must be a dictionary.")
            raise TypeError("trace must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        for module in weak_modules:
            logger.info("Refinement suggestion for %s using modulation: %.2f for task %s", module, trace['modulation_index'], task_type)
            prompt = f"""
            Suggest phi/eta-aligned improvements for the {module} module.
            phi = {trace['phi']:.3f}, eta = {trace['eta']:.3f}, Index = {trace['modulation_index']:.3f}
            Task Type: {task_type}
            """
            valid, report = await self.alignment_guard.ethical_check(prompt, stage="module_refinement", task_type=task_type) if self.alignment_guard else (True, {})
            if not valid:
                logger.warning("Prompt failed alignment check for module %s for task %s: %s", module, task_type, report)
                continue
            try:
                suggestions = await call_gpt(prompt, task_type=task_type)
                sim_result = await run_simulation(f"Test refinement:\n{suggestions}", task_type=task_type)
                logger.debug("Result for %s:\n%s for task %s", module, sim_result, task_type)
                if self.agi_enhancer:
                    await self.agi_enhancer.reflect_and_adapt(f"Refinement for {module} evaluated for task {task_type}")
                reflection = await meta_cognition.MetaCognition().reflect_on_output(
                    component="LearningLoop",
                    output={"suggestions": suggestions},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Module refinement reflection: %s", reflection.get("reflection", ""))
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"ModuleRefinement_{module}_{time.strftime('%Y%m%d_%H%M%S')}",
                        output=suggestions,
                        layer="Modules",
                        intent="module_refinement",
                        task_type=task_type
                    )
            except Exception as e:
                logger.error("Refinement failed for module %s for task %s: %s", module, task_type, str(e))

    async def _detect_capability_gaps(self, last_input: Optional[str], last_output: Optional[str], task_type: str = "") -> None:
        """Detect capability gaps and propose module refinements."""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")
        if not last_input or not last_output:
            logger.info("Skipping capability gap detection: missing input/output for task %s", task_type)
            return

        logger.info("Detecting capability gaps for task %s...", task_type)
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = f"""
        Input: {last_input}
        Output: {last_output}
        phi = {phi:.2f}
        Task Type: {task_type}

        Identify capability gaps and suggest blueprints for phi-tuned modules.
        """
        valid, report = await self.alignment_guard.ethical_check(prompt, stage="capability_gap", task_type=task_type) if self.alignment_guard else (True, {})
        if not valid:
            logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
            return
        try:
            proposal = await call_gpt(prompt, task_type=task_type)
            if proposal:
                logger.info("Proposed phi-based module refinement for task %s", task_type)
                await self._simulate_and_deploy_module(proposal, task_type)
                reflection = await meta_cognition.MetaCognition().reflect_on_output(
                    component="LearningLoop",
                    output={"proposal": proposal},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Capability gap reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Capability gap detection failed for task %s: %s", task_type, str(e))

    async def _simulate_and_deploy_module(self, blueprint: str, task_type: str = "") -> None:
        """Simulate and deploy a module blueprint."""
        if not isinstance(blueprint, str) or not blueprint.strip():
            logger.error("Invalid blueprint: must be a non-empty string.")
            raise ValueError("blueprint must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            result = await run_simulation(f"Module sandbox:\n{blueprint}", task_type=task_type)
            if isinstance(result, dict) and result.get("status") == "approved":
                logger.info("Deploying blueprint for task %s", task_type)
                self.module_blueprints.append(blueprint)
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        event="Blueprint deployed",
                        meta={"blueprint": blueprint},
                        module="LearningLoop",
                        tags=["blueprint", "deploy", task_type]
                    )
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "deploy_blueprint",
                        "blueprint": blueprint,
                        "task_type": task_type
                    })
                reflection = await meta_cognition.MetaCognition().reflect_on_output(
                    component="LearningLoop",
                    output={"blueprint": blueprint},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Blueprint deployment reflection: %s", reflection.get("reflection", ""))
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"ModuleBlueprint_{time.strftime('%Y%m%d_%H%M%S')}",
                        output=blueprint,
                        layer="Modules",
                        intent="module_deployment",
                        task_type=task_type
                    )
        except Exception as e:
            logger.error("Blueprint deployment failed for task %s: %s", task_type, str(e))

    async def _consolidate_knowledge(self, task_type: str = "") -> None:
        """Consolidate phi-aligned knowledge."""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        t = time.time() % 1.0
        phi = phi_scalar(t)
        logger.info("Consolidating phi-aligned knowledge for task %s", task_type)
        prompt = f"""
        Consolidate recent learning using phi = {phi:.2f}.
        Prune noise, synthesize patterns, and emphasize high-impact transitions.
        Task Type: {task_type}
        """
        valid, report = await self.alignment_guard.ethical_check(prompt, stage="knowledge_consolidation", task_type=task_type) if self.alignment_guard else (True, {})
        if not valid:
            logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
            return
        try:
            if self.memory_manager:
                drift_entries = await self.memory_manager.search(
                    query_prefix="Knowledge",
                    layer="Knowledge",
                    intent="knowledge_consolidation",
                    task_type=task_type
                )
                if drift_entries:
                    avg_drift = sum(entry["output"].get("similarity", 0.5) for entry in drift_entries) / len(drift_entries)
                    prompt += f"\nAverage drift similarity: {avg_drift:.2f}"
            consolidated = await call_gpt(prompt, task_type=task_type)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Knowledge consolidation",
                    meta={"consolidated": consolidated},
                    module="LearningLoop",
                    tags=["consolidation", "knowledge", task_type]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "consolidate_knowledge",
                    "task_type": task_type
                })
            reflection = await meta_cognition.MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"consolidated": consolidated},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Knowledge consolidation reflection: %s", reflection.get("reflection", ""))
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"KnowledgeConsolidation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=consolidated,
                    layer="Knowledge",
                    intent="knowledge_consolidation",
                    task_type=task_type
                )
        except Exception as e:
            logger.error("Knowledge consolidation failed for task %s: %s", task_type, str(e))

    async def trigger_reflexive_audit(self, context_snapshot: Dict[str, Any], task_type: str = "") -> str:
        """Audit context trajectory for cognitive dissonance."""
        if not isinstance(context_snapshot, dict):
            logger.error("Invalid context_snapshot: must be a dictionary.")
            raise TypeError("context_snapshot must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Initiating reflexive audit on context trajectory for task %s...", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            eta = eta_feedback(t)
            audit_prompt = f"""
            You are a reflexive audit agent. Analyze this context state and trajectory:
            {json.dumps(context_snapshot, indent=2)}

            phi = {phi:.2f}, eta = {eta:.2f}
            Task Type: {task_type}
            Identify cognitive dissonance, meta-patterns, or feedback loops.
            Recommend modulations or trace corrections.
            """
            valid, report = await self.alignment_guard.ethical_check(audit_prompt, stage="reflexive_audit", task_type=task_type) if self.alignment_guard else (True, {})
            if not valid:
                logger.warning("Audit prompt failed alignment check for task %s: %s", task_type, report)
                return "Audit blocked by alignment guard"

            audit_response = await call_gpt(audit_prompt, task_type=task_type)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Reflexive Audit Triggered",
                    meta={"phi": phi, "eta": eta, "context": context_snapshot, "audit_response": audit_response},
                    module="LearningLoop",
                    tags=["audit", "reflexive", task_type]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "reflexive_audit",
                    "response": audit_response,
                    "task_type": task_type
                })
            reflection = await meta_cognition.MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"audit_response": audit_response},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Reflexive audit reflection: %s", reflection.get("reflection", ""))
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ReflexiveAudit_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=audit_response,
                    layer="Audits",
                    intent="reflexive_audit",
                    task_type=task_type
                )
            return audit_response
        except Exception as e:
            logger.error("Reflexive audit failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.trigger_reflexive_audit(context_snapshot, task_type),
                default="Audit failed"
            )

    async def _check_narrative_integrity(self, task_type: str = "") -> None:
        """Check narrative coherence across goal history."""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")
        if len(self.goal_history) < 2:
            return

        logger.info("Checking narrative coherence across goal history for task %s...", task_type)
        try:
            last_goal = self.goal_history[-1]["goal"]
            prior_goal = self.goal_history[-2]["goal"]
            check_prompt = f"""
            Compare the following goals for alignment and continuity:
            Previous: {prior_goal}
            Current: {last_goal}
            Task Type: {task_type}

            Are these in narrative coherence? If not, suggest a corrective alignment.
            """
            valid, report = await self.alignment_guard.ethical_check(check_prompt, stage="narrative_check", task_type=task_type) if self.alignment_guard else (True, {})
            if not valid:
                logger.warning("Narrative check prompt failed alignment check for task %s: %s", task_type, report)
                return

            audit = await call_gpt(check_prompt, task_type=task_type)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Narrative Coherence Audit",
                    meta={"previous_goal": prior_goal, "current_goal": last_goal, "audit": audit},
                    module="LearningLoop",
                    tags=["narrative", "coherence", task_type]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "narrative_integrity",
                    "audit": audit,
                    "task_type": task_type
                })
            reflection = await meta_cognition.MetaCognition().reflect_on_output(
                component="LearningLoop",
                output={"audit": audit},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Narrative integrity reflection: %s", reflection.get("reflection", ""))
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"NarrativeAudit_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=audit,
                    layer="Audits",
                    intent="narrative_integrity",
                    task_type=task_type
                )
        except Exception as e:
            logger.error("Narrative coherence check failed for task %s: %s", task_type, str(e))

    def replay_with_foresight(self, memory_traces: List[Dict[str, Any]], task_type: str = "") -> List[Dict[str, Any]]:
        """Reorder learning traces by foresight-weighted priority."""
        if not isinstance(memory_traces, list) or not all(isinstance(t, dict) for t in memory_traces):
            logger.error("Invalid memory_traces: must be a list of dictionaries.")
            raise ValueError("memory_traces must be a list of dictionaries")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        def foresight_score(trace: Dict[str, Any]) -> float:
            return trace.get("phi", 0.5) * (1.0 if trace.get("task_type") == task_type else 0.8)
        sorted_traces = sorted(memory_traces, key=foresight_score, reverse=True)
        if self.memory_manager:
            asyncio.create_task(self.memory_manager.store(
                query=f"ReplayForesight_{time.strftime('%Y%m%d_%H%M%S')}",
                output=str(sorted_traces),
                layer="Traces",
                intent="replay_foresight",
                task_type=task_type
            ))
        return sorted_traces

    def revise_knowledge(self, new_info: str, context: Optional[str] = None, task_type: str = "") -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge:
                similarity = self.concept_synthesizer.compare(new_info, existing, task_type=task_type)
                if similarity["score"] > 0.9 and new_info != existing:
                    logger.warning("Potential knowledge conflict: %s vs %s for task %s", new_info, existing, task_type)

        self.knowledge_base = old_knowledge + [new_info]
        self.log_epistemic_revision(new_info, context, task_type)
        logger.info("Knowledge base updated with: %s for task %s", new_info, task_type)
        if self.context_manager:
            asyncio.create_task(self.context_manager.log_event_with_hash({
                "event": "knowledge_revision",
                "info": new_info,
                "task_type": task_type
            }))
        if self.memory_manager:
            asyncio.create_task(self.memory_manager.store(
                query=f"KnowledgeRevision_{time.strftime('%Y%m%d_%H%M%S')}",
                output=new_info,
                layer="Knowledge",
                intent="knowledge_revision",
                task_type=task_type
            ))

    def log_epistemic_revision(self, info: str, context: Optional[str], task_type: str = "") -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        revision = {
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type
        }
        self.epistemic_revision_log.append(revision)
        logger.info("Epistemic revision logged: %s for task %s", info, task_type)
        if self.agi_enhancer:
            asyncio.create_task(self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta=revision,
                module="LearningLoop",
                tags=["revision", "knowledge", task_type]
            ))
        if self.memory_manager:
            asyncio.create_task(self.memory_manager.store(
                query=f"EpistemicRevision_{time.strftime('%Y%m%d_%H%M%S')}",
                output=str(revision),
                layer="Knowledge",
                intent="epistemic_revision",
                task_type=task_type
            ))

    def monitor_epistemic_state(self, simulated_outcome: Dict[str, Any], task_type: str = "") -> None:
        """Monitor and revise the epistemic framework based on simulation outcomes."""
        if not isinstance(simulated_outcome, dict):
            logger.error("Invalid simulated_outcome: must be a dictionary.")
            raise TypeError("simulated_outcome must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Monitoring epistemic state with outcome: %s for task %s", simulated_outcome, task_type)
        if self.agi_enhancer:
            asyncio.create_task(self.agi_enhancer.log_episode(
                event="Epistemic Monitoring",
                meta={"outcome": simulated_outcome},
                module="LearningLoop",
                tags=["epistemic", "monitor", task_type]
            ))
        if self.context_manager:
            asyncio.create_task(self.context_manager.log_event_with_hash({
                "event": "epistemic_monitor",
                "outcome": simulated_outcome,
                "task_type": task_type
            }))
        if self.memory_manager:
            asyncio.create_task(self.memory_manager.store(
                query=f"EpistemicMonitor_{time.strftime('%Y%m%d_%H%M%S')}",
                output=str(simulated_outcome),
                layer="Knowledge",
                intent="epistemic_monitor",
                task_type=task_type
            ))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = LearningLoop()
    meta = meta_cognition.MetaCognition()
    asyncio.run(loop.activate_intrinsic_goals(meta, task_type="test"))
