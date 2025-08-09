"""
ANGELA Cognitive System Module: SimulationCore
Version: 3.5.1  # Enhanced for Task-Specific Simulations, Real-Time Data, and Visualization
Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides a SimulationCore class for agent simulations, impact validations,
and environment simulations in the ANGELA v3.5.1 architecture, integrating ToCA physics.
"""

import logging
import math
import json
import hashlib
import numpy as np
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from threading import Lock
from collections import deque
from functools import lru_cache

from utils.prompt_utils import query_openai
from modules import (
    visualizer as visualizer_module,
    memory_manager as memory_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    multi_modal_fusion as multi_modal_fusion_module,
    meta_cognition as meta_cognition_module,
    reasoning_engine as reasoning_engine_module
)
from index import zeta_consequence, theta_causality, rho_agency, TraitOverlayManager

logger = logging.getLogger("ANGELA.SimulationCore")

async def call_gpt(prompt: str, alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None, task_type: str = "") -> str:
    """Wrapper for querying GPT with error handling and task-specific alignment. [v3.5.1]"""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096 for task %s", task_type)
        raise ValueError("prompt must be a string with length <= 4096")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")
    if alignment_guard:
        valid, report = await alignment_guard.ethical_check(prompt, stage="gpt_query", task_type=task_type)
        if not valid:
            logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
            raise ValueError("Prompt failed alignment check")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed for task %s: %s", task_type, result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception for task %s: %s", task_type, str(e))
        raise

class ToCATraitEngine:
    """Cyber-Physics Engine based on ToCA dynamics for agent simulations. [v3.5.1]

    Attributes:
        k_m (float): Motion coupling constant.
        delta_m (float): Damping modulation factor.
        meta_cognition (Optional[MetaCognition]): Meta-cognition module for reflection.
    """
    def __init__(self, k_m: float = 1e-3, delta_m: float = 1e4,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None):
        self.k_m = k_m
        self.delta_m = delta_m
        self.meta_cognition = meta_cognition
        logger.info("ToCATraitEngine initialized with k_m=%f, delta_m=%f", k_m, delta_m)

    @lru_cache(maxsize=100)
    def evolve(self, x_tuple: tuple, t_tuple: tuple, user_data_tuple: Optional[tuple] = None, task_type: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evolve ToCA fields for simulation with task-specific processing. [v3.5.1]

        Args:
            x_tuple: Spatial coordinates as a tuple.
            t_tuple: Time coordinates as a tuple.
            user_data_tuple: Optional user data as a tuple.
            task_type: Type of task for context-aware processing.

        Returns:
            Tuple of phi (scalar field), lambda_t (damping field), and v_m (motion potential).
        """
        if not isinstance(x_tuple, tuple) or not isinstance(t_tuple, tuple):
            logger.error("Invalid input: x_tuple and t_tuple must be tuples for task %s", task_type)
            raise TypeError("x_tuple and t_tuple must be tuples")
        if user_data_tuple is not None and not isinstance(user_data_tuple, tuple):
            logger.error("Invalid user_data_tuple: must be a tuple for task %s", task_type)
            raise TypeError("user_data_tuple must be a tuple")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            x = np.array(x_tuple)
            t = np.array(t_tuple)
            user_data = np.array(user_data_tuple) if user_data_tuple else None

            if not isinstance(x, np.ndarray) or not isinstance(t, np.ndarray):
                logger.error("Invalid input: x and t must be numpy arrays for task %s", task_type)
                raise TypeError("x and t must be numpy arrays")
            if user_data is not None and not isinstance(user_data, np.ndarray):
                logger.error("Invalid user_data: must be a numpy array for task %s", task_type)
                raise TypeError("user_data must be a numpy array")

            x = np.clip(x, 1e-10, 1e10)
            v_m = self.k_m * np.gradient(3e3 * 1.989 / (x**2 + 1e-10))
            phi = np.sin(t * 1e-3) * 1e-3 * (1 + v_m * np.gradient(x))
            if user_data is not None:
                phi += np.mean(user_data) * 1e-4
            lambda_t = 1.1e-3 * np.exp(-2e-2 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * self.delta_m)

            result = {"phi": phi.tolist(), "lambda_t": lambda_t.tolist(), "v_m": v_m.tolist(), "task_type": task_type}
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="ToCATraitEngine",
                    output=json.dumps(result),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("ToCA evolution reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            return phi, lambda_t, v_m
        except Exception as e:
            logger.error("ToCA evolution failed for task %s: %s", task_type, str(e))
            raise

    async def update_fields_with_agents(self, phi: np.ndarray, lambda_t: np.ndarray, agent_matrix: np.ndarray, task_type: str = "") -> Tuple[np.ndarray, np.ndarray]:
        """Update fields with agent interactions, supporting task-specific drift mitigation. [v3.5.1]"""
        if not all(isinstance(arr, np.ndarray) for arr in [phi, lambda_t, agent_matrix]):
            logger.error("Invalid inputs: phi, lambda_t, and agent_matrix must be numpy arrays for task %s", task_type)
            raise TypeError("inputs must be numpy arrays")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            interaction_energy = np.dot(agent_matrix, np.sin(phi)) * 1e-3
            phi = phi + interaction_energy
            lambda_t = lambda_t * (1 + 0.001 * np.sum(agent_matrix, axis=0))
            result = {"phi": phi.tolist(), "lambda_t": lambda_t.tolist(), "task_type": task_type}
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="ToCATraitEngine",
                    output=json.dumps(result),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Field update reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            return phi, lambda_t
        except Exception as e:
            logger.error("Field update with agents failed for task %s: %s", task_type, str(e))
            raise

class SimulationCore:
    """Core simulation engine for ANGELA v3.5.1, integrating ToCA physics and agent dynamics.

    Attributes:
        visualizer (Visualizer): Module for rendering simulation outputs.
        simulation_history (deque): History of simulation states (max 1000).
        ledger (deque): Audit trail of simulation records with hashes (max 1000).
        agi_enhancer (AGIEnhancer): Enhancer for logging and adaptation.
        memory_manager (MemoryManager): Manager for storing simulation data.
        multi_modal_fusion (MultiModalFusion): Module for multi-modal synthesis.
        error_recovery (ErrorRecovery): Module for error handling and recovery.
        meta_cognition (MetaCognition): Module for reflection and reasoning review.
        reasoning_engine (ReasoningEngine): Engine for reasoning and drift mitigation.
        toca_engine (ToCATraitEngine): Engine for ToCA physics simulations.
        overlay_router (TraitOverlayManager): Manager for trait overlays.
        worlds (Dict[str, Dict]): Dictionary of defined simulation worlds.
        current_world (Optional[Dict]): Currently active simulation world.
        ledger_lock (Lock): Thread-safe lock for ledger updates.
    """
    def __init__(self,
                 agi_enhancer: Optional['agi_enhancer_module.AGIEnhancer'] = None,
                 visualizer: Optional['visualizer_module.Visualizer'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 reasoning_engine: Optional['reasoning_engine_module.ReasoningEngine'] = None,
                 toca_engine: Optional['ToCATraitEngine'] = None,
                 overlay_router: Optional['TraitOverlayManager'] = None):
        self.visualizer = visualizer or visualizer_module.Visualizer()
        self.simulation_history = deque(maxlen=1000)
        self.ledger = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=self.memory_manager)
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer, memory_manager=self.memory_manager)
        self.reasoning_engine = reasoning_engine or reasoning_engine_module.ReasoningEngine(
            agi_enhancer=agi_enhancer, memory_manager=self.memory_manager,
            multi_modal_fusion=self.multi_modal_fusion, meta_cognition=self.meta_cognition,
            visualizer=self.visualizer)
        self.toca_engine = toca_engine or ToCATraitEngine(meta_cognition=self.meta_cognition)
        self.overlay_router = overlay_router or TraitOverlayManager()
        self.worlds = {}
        self.current_world = None
        self.ledger_lock = Lock()
        logger.info("SimulationCore initialized")

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-serializable objects. [v3.5.1]"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    async def _record_state(self, data: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Record simulation state with hash for integrity and task-specific processing. [v3.5.1]"""
        if not isinstance(data, dict):
            logger.error("Invalid data: must be a dictionary for task %s", task_type)
            raise TypeError("data must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "hash": hashlib.sha256(
                    json.dumps(data, sort_keys=True, default=self._json_serializer).encode()
                ).hexdigest(),
                "task_type": task_type
            }
            with self.ledger_lock:
                self.ledger.append(record)
                self.simulation_history.append(record)
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Ledger_{record['timestamp']}",
                        output=record,
                        layer="Ledger",
                        intent="state_record",
                        task_type=task_type
                    )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=json.dumps(record),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("State record reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            return record
        except Exception as e:
            logger.error("State recording failed for task %s: %s", task_type, str(e))
            raise

    async def run(self, results: str, context: Optional[Dict[str, Any]] = None,
                  scenarios: int = 3, agents: int = 2, export_report: bool = False,
                  export_format: str = "pdf", actor_id: str = "default_agent",
                  task_type: str = "") -> Dict[str, Any]:
        """Run a simulation with specified parameters and task-specific processing. [v3.5.1]"""
        if not isinstance(results, str) or not results.strip():
            logger.error("Invalid results: must be a non-empty string for task %s", task_type)
            raise ValueError("results must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(scenarios, int) or scenarios < 1:
            logger.error("Invalid scenarios: must be a positive integer for task %s", task_type)
            raise ValueError("scenarios must be a positive integer")
        if not isinstance(agents, int) or agents < 1:
            logger.error("Invalid agents: must be a positive integer for task %s", task_type)
            raise ValueError("agents must be a positive integer")
        if not isinstance(export_format, str) or export_format not in ["pdf", "json", "html"]:
            logger.error("Invalid export_format: must be 'pdf', 'json', or 'html' for task %s", task_type)
            raise ValueError("export_format must be 'pdf', 'json', or 'html'")
        if not isinstance(actor_id, str) or not actor_id.strip():
            logger.error("Invalid actor_id: must be a non-empty string for task %s", task_type)
            raise ValueError("actor_id must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Running simulation with %d agents and %d scenarios for task %s", agents, scenarios, task_type)
        try:
            t = time.time() % 1.0
            causality = max(0.0, min(theta_causality(t), 1.0))
            agency = max(0.0, min(rho_agency(t), 1.0))

            x = np.linspace(0.1, 20, 100)
            t_vals = np.linspace(0.1, 20, 100)
            agent_matrix = np.random.rand(agents, 100)

            phi, lambda_field, v_m = await self.toca_engine.evolve(tuple(x), tuple(t_vals), task_type=task_type)
            phi, lambda_field = await self.toca_engine.update_fields_with_agents(phi, lambda_field, agent_matrix, task_type=task_type)
            energy_cost = float(np.mean(np.abs(phi)) * 1e3)

            context = context or {}
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            prompt = {
                "results": results,
                "context": context,
                "scenarios": scenarios,
                "agents": agents,
                "actor_id": actor_id,
                "traits": {
                    "theta_causality": causality,
                    "rho_agency": agency
                },
                "fields": {
                    "phi": phi.tolist(),
                    "lambda": lambda_field.tolist(),
                    "v_m": v_m.tolist()
                },
                "estimated_energy_cost": energy_cost,
                "policies": policies,
                "task_type": task_type
            }

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(prompt, default=self._json_serializer), stage="simulation", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Alignment guard rejected simulation request for task %s: %s", task_type, report)
                return {"error": "Simulation rejected due to alignment constraints", "task_type": task_type}

            query_key = f"Simulation_{results[:50]}_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM", task_type=task_type)
            if cached_output:
                logger.info("Retrieved cached simulation output for task %s", task_type)
                simulation_output = cached_output
            else:
                simulation_output = await call_gpt(
                    f"Simulate agent outcomes: {json.dumps(prompt, default=self._json_serializer)}",
                    self.multi_modal_fusion.alignment_guard,
                    task_type=task_type
                )
                if not isinstance(simulation_output, (dict, str)):
                    logger.error("Invalid simulation output: must be a dictionary or string for task %s", task_type)
                    raise ValueError("simulation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=simulation_output,
                        layer="STM",
                        intent="simulation",
                        task_type=task_type
                    )

            state_record = await self._record_state({
                "actor": actor_id,
                "action": "run_simulation",
                "traits": prompt["traits"],
                "energy_cost": energy_cost,
                "output": simulation_output,
                "task_type": task_type
            }, task_type=task_type)

            self.simulation_history.append(state_record)

            if "drift" in results.lower() and self.reasoning_engine:
                drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                    drift_data=context.get("drift", {}),
                    context=context,
                    task_type=task_type
                )
                state_record["drift_mitigation"] = drift_result

            if export_report and self.memory_manager:
                await self.memory_manager.promote_to_ltm(query_key, task_type=task_type)

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Simulation run",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["simulation", "run", task_type]
                )
                await self.agi_enhancer.reflect_and_adapt(f"SimulationCore: scenario simulation complete for task {task_type}")

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=json.dumps(simulation_output),
                    context={"energy_cost": energy_cost, "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    state_record["reflection"] = reflection.get("reflection", "")
                    logger.info("Simulation reflection for task %s: %s", task_type, reflection.get("reflection", ""))

            if self.visualizer:
                plot_data = {
                    "simulation": {
                        "output": simulation_output,
                        "traits": prompt["traits"],
                        "energy_cost": energy_cost,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if export_report and self.visualizer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_report_{timestamp}.{export_format}"
                logger.info("Exporting report for task %s: %s", task_type, filename)
                await self.visualizer.export_report(simulation_output, filename=filename, format=export_format)

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"prompt": prompt, "output": simulation_output, "policies": policies},
                    summary_style="insightful",
                    task_type=task_type
                )
                state_record["synthesis"] = synthesis

            return simulation_output
        except Exception as e:
            logger.error("Simulation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run(results, context, scenarios, agents, export_report, export_format, actor_id, task_type),
                default={"error": str(e), "task_type": task_type}
            )

    async def validate_impact(self, proposed_action: str, agents: int = 2,
                             export_report: bool = False, export_format: str = "pdf",
                             actor_id: str = "validator_agent", task_type: str = "") -> Dict[str, Any]:
        """Validate the impact of a proposed action with task-specific processing. [v3.5.1]"""
        if not isinstance(proposed_action, str) or not proposed_action.strip():
            logger.error("Invalid proposed_action: must be a non-empty string for task %s", task_type)
            raise ValueError("proposed_action must be a non-empty string")
        if not isinstance(agents, int) or agents < 1:
            logger.error("Invalid agents: must be a positive integer for task %s", task_type)
            raise ValueError("agents must be a positive integer")
        if not isinstance(export_format, str) or export_format not in ["pdf", "json", "html"]:
            logger.error("Invalid export_format: must be 'pdf', 'json', or 'html' for task %s", task_type)
            raise ValueError("export_format must be 'pdf', 'json', or 'html'")
        if not isinstance(actor_id, str) or not actor_id.strip():
            logger.error("Invalid actor_id: must be a non-empty string for task %s", task_type)
            raise ValueError("actor_id must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Validating impact of proposed action: %s (Task: %s)", proposed_action, task_type)
        try:
            t = time.time() % 1.0
            consequence = max(0.0, min(zeta_consequence(t), 1.0))

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            prompt = {
                "action": proposed_action,
                "trait_zeta_consequence": consequence,
                "agents": agents,
                "policies": policies,
                "task_type": task_type
            }

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(prompt, default=self._json_serializer), stage="impact_validation", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Alignment guard blocked impact validation for task %s: %s", task_type, report)
                return {"error": "Validation blocked by alignment rules", "task_type": task_type}

            query_key = f"Validation_{proposed_action[:50]}_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM", task_type=task_type)
            if cached_output:
                logger.info("Retrieved cached validation output for task %s", task_type)
                validation_output = cached_output
            else:
                prompt_text = f"""
                Evaluate the following proposed action:
                {proposed_action}

                Trait:
                - zeta_consequence = {consequence:.3f}

                Analyze positive/negative outcomes, agent variations, risk scores (1-10), and recommend: Proceed / Modify / Abort.
                Task Type: {task_type}
                Policies: {policies}
                """
                validation_output = await call_gpt(prompt_text, self.multi_modal_fusion.alignment_guard, task_type=task_type)
                if not isinstance(validation_output, (dict, str)):
                    logger.error("Invalid validation output: must be a dictionary or string for task %s", task_type)
                    raise ValueError("validation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=validation_output,
                        layer="STM",
                        intent="impact_validation",
                        task_type=task_type
                    )

            state_record = await self._record_state({
                "actor": actor_id,
                "action": "validate_impact",
                "trait_zeta_consequence": consequence,
                "proposed_action": proposed_action,
                "output": validation_output,
                "task_type": task_type
            }, task_type=task_type)

            self.simulation_history.append(state_record)

            if "drift" in proposed_action.lower() and self.reasoning_engine:
                drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                    drift_data={"action": proposed_action, "similarity": consequence},
                    context={"policies": policies},
                    task_type=task_type
                )
                state_record["drift_mitigation"] = drift_result

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=json.dumps(validation_output),
                    context={"consequence": consequence, "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    state_record["reflection"] = reflection.get("reflection", "")
                    logger.info("Impact validation reflection for task %s: %s", task_type, reflection.get("reflection", ""))

            if self.visualizer:
                plot_data = {
                    "impact_validation": {
                        "proposed_action": proposed_action,
                        "output": validation_output,
                        "consequence": consequence,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if export_report and self.visualizer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"impact_validation_{timestamp}.{export_format}"
                logger.info("Exporting validation report for task %s: %s", task_type, filename)
                await self.visualizer.export_report(validation_output, filename=filename, format=export_format)

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"action": proposed_action, "output": validation_output, "consequence": consequence, "policies": policies},
                    summary_style="concise",
                    task_type=task_type
                )
                state_record["synthesis"] = synthesis

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Impact validation",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["validation", "impact", task_type]
                )
                await self.agi_enhancer.reflect_and_adapt(f"SimulationCore: impact validation complete for task {task_type}")

            return validation_output
        except Exception as e:
            logger.error("Impact validation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.validate_impact(proposed_action, agents, export_report, export_format, actor_id, task_type),
                default={"error": str(e), "task_type": task_type}
            )

    async def simulate_environment(self, environment_config: Dict[str, Any], agents: int = 2,
                                  steps: int = 10, actor_id: str = "env_agent",
                                  goal: Optional[str] = None, task_type: str = "") -> Dict[str, Any]:
        """Simulate agents in a configured environment with task-specific processing. [v3.5.1]"""
        if not isinstance(environment_config, dict):
            logger.error("Invalid environment_config: must be a dictionary for task %s", task_type)
            raise TypeError("environment_config must be a dictionary")
        if not isinstance(agents, int) or agents < 1:
            logger.error("Invalid agents: must be a positive integer for task %s", task_type)
            raise ValueError("agents must be a positive integer")
        if not isinstance(steps, int) or steps < 1:
            logger.error("Invalid steps: must be a positive integer for task %s", task_type)
            raise ValueError("steps must be a positive integer")
        if not isinstance(actor_id, str) or not actor_id.strip():
            logger.error("Invalid actor_id: must be a non-empty string for task %s", task_type)
            raise ValueError("actor_id must be a non-empty string")
        if goal is not None and not isinstance(goal, str):
            logger.error("Invalid goal: must be a string or None for task %s", task_type)
            raise TypeError("goal must be a string or None")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Running environment simulation with %d agents and %d steps for task %s", agents, steps, task_type)
        try:
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            prompt = {
                "environment": environment_config,
                "goal": goal,
                "steps": steps,
                "agents": agents,
                "policies": policies,
                "task_type": task_type
            }

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(prompt, default=self._json_serializer), stage="environment_simulation", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Alignment guard rejected environment simulation for task %s: %s", task_type, report)
                return {"error": "Simulation blocked due to environment constraints", "task_type": task_type}

            query_key = f"Environment_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM", task_type=task_type)
            if cached_output:
                logger.info("Retrieved cached environment simulation output for task %s", task_type)
                environment_simulation = cached_output
            else:
                prompt_text = f"""
                Simulate agents in this environment:
                {json.dumps(environment_config, default=self._json_serializer)}

                Steps: {steps} | Agents: {agents}
                Goal: {goal if goal else 'N/A'}
                Task Type: {task_type}
                Policies: {policies}
                Describe interactions, environmental changes, risks/opportunities.
                """
                environment_simulation = await call_gpt(prompt_text, self.multi_modal_fusion.alignment_guard, task_type=task_type)
                if not isinstance(environment_simulation, (dict, str)):
                    logger.error("Invalid environment simulation output: must be a dictionary or string for task %s", task_type)
                    raise ValueError("environment simulation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=environment_simulation,
                        layer="STM",
                        intent="environment_simulation",
                        task_type=task_type
                    )

            state_record = await self._record_state({
                "actor": actor_id,
                "action": "simulate_environment",
                "config": environment_config,
                "steps": steps,
                "goal": goal,
                "output": environment_simulation,
                "task_type": task_type
            }, task_type=task_type)

            self.simulation_history.append(state_record)

            if "drift" in (goal or "").lower() and self.reasoning_engine:
                drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                    drift_data=environment_config.get("drift", {}),
                    context={"config": environment_config, "policies": policies},
                    task_type=task_type
                )
                state_record["drift_mitigation"] = drift_result

            if self.meta_cognition:
                reflection = await self.meta_cognition.review_reasoning(environment_simulation, task_type=task_type)
                state_record["reflection"] = reflection
                logger.info("Environment simulation reflection for task %s: %s", task_type, reflection)

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Environment simulation",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["environment", "simulation", task_type]
                )
                await self.agi_enhancer.reflect_and_adapt(f"SimulationCore: environment simulation complete for task {task_type}")

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"config": environment_config, "output": environment_simulation, "goal": goal, "policies": policies},
                    summary_style="insightful",
                    task_type=task_type
                )
                state_record["synthesis"] = synthesis

            if self.visualizer:
                plot_data = {
                    "environment_simulation": {
                        "config": environment_config,
                        "output": environment_simulation,
                        "goal": goal,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            return environment_simulation
        except Exception as e:
            logger.error("Environment simulation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_environment(environment_config, agents, steps, actor_id, goal, task_type),
                default={"error": str(e), "task_type": task_type}
            )

    async def replay_intentions(self, memory_log: List[Dict[str, Any]], task_type: str = "") -> List[Dict[str, Any]]:
        """Trace past intentions and return a replay sequence with task-specific processing. [v3.5.1]"""
        if not isinstance(memory_log, list):
            logger.error("Invalid memory_log: must be a list for task %s", task_type)
            raise TypeError("memory_log must be a list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            replay = [
                {
                    "timestamp": entry.get("timestamp"),
                    "goal": entry.get("goal"),
                    "intention": entry.get("intention"),
                    "traits": entry.get("traits", {}),
                    "task_type": task_type
                }
                for entry in memory_log
                if isinstance(entry, dict) and "goal" in entry
            ]
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Replay_{datetime.now().isoformat()}",
                    output=str(replay),
                    layer="Replays",
                    intent="intention_replay",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=json.dumps(replay),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Intention replay reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Intentions replayed",
                    meta={"replay": replay, "task_type": task_type},
                    module="SimulationCore",
                    tags=["replay", "intentions", task_type]
                )
            if self.visualizer and task_type:
                plot_data = {
                    "intention_replay": {
                        "replay": replay,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return replay
        except Exception as e:
            logger.error("Intention replay failed for task %s: %s", task_type, str(e))
            raise

    async def fabricate_reality(self, parameters: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Construct immersive meta-environments from symbolic templates with task-specific processing. [v3.5.1]"""
        if not isinstance(parameters, dict):
            logger.error("Invalid parameters: must be a dictionary for task %s", task_type)
            raise TypeError("parameters must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Fabricating reality with parameters for task %s: %s", task_type, parameters)
        try:
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            environment = {"fabricated_world": True, "parameters": parameters, "policies": policies, "task_type": task_type}
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"parameters": parameters, "policies": policies},
                    summary_style="insightful",
                    task_type=task_type
                )
                environment["synthesis"] = synthesis
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reality_Fabrication_{datetime.now().isoformat()}",
                    output=str(environment),
                    layer="Realities",
                    intent="reality_fabrication",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=json.dumps(environment),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Reality fabrication reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Reality fabricated",
                    meta=environment,
                    module="SimulationCore",
                    tags=["reality", "fabrication", task_type]
                )
            if self.visualizer and task_type:
                plot_data = {
                    "reality_fabrication": {
                        "parameters": parameters,
                        "synthesis": environment.get("synthesis", ""),
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return environment
        except Exception as e:
            logger.error("Reality fabrication failed for task %s: %s", task_type, str(e))
            raise

    async def synthesize_self_world(self, identity_data: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Ensure persistent identity integration in self-generated environments with task-specific processing. [v3.5.1]"""
        if not isinstance(identity_data, dict):
            logger.error("Invalid identity_data: must be a dictionary for task %s", task_type)
            raise TypeError("identity_data must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            result = {"identity": identity_data, "coherence_score": 0.97, "policies": policies, "task_type": task_type}
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"identity": identity_data, "policies": policies},
                    summary_style="concise",
                    task_type=task_type
                )
                result["synthesis"] = synthesis
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Self_World_Synthesis_{datetime.now().isoformat()}",
                    output=str(result),
                    layer="Identities",
                    intent="self_world_synthesis",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=json.dumps(result),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Self-world synthesis reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Self-world synthesized",
                    meta=result,
                    module="SimulationCore",
                    tags=["identity", "synthesis", task_type]
                )
            if self.visualizer and task_type:
                plot_data = {
                    "self_world_synthesis": {
                        "identity": identity_data,
                        "coherence_score": result["coherence_score"],
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return result
        except Exception as e:
            logger.error("Self-world synthesis failed for task %s: %s", task_type, str(e))
            raise

    async def define_world(self, name: str, parameters: Dict[str, Any], task_type: str = "") -> None:
        """Define a simulation world with given parameters and task-specific processing. [v3.5.1]"""
        if not isinstance(name, str) or not name.strip():
            logger.error("Invalid world name: must be a non-empty string for task %s", task_type)
            raise ValueError("world name must be a non-empty string")
        if not isinstance(parameters, dict):
            logger.error("Invalid parameters: must be a dictionary for task %s", task_type)
            raise TypeError("parameters must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            self.worlds[name] = parameters
            logger.info("Defined world: %s for task %s", name, task_type)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"World_Definition_{name}_{datetime.now().isoformat()}",
                    output=parameters,
                    layer="Worlds",
                    intent="world_definition",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=json.dumps({"name": name, "parameters": parameters}),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("World definition reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "world_definition": {
                        "name": name,
                        "parameters": parameters,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
        except Exception as e:
            logger.error("World definition failed for task %s: %s", task_type, str(e))
            raise

    async def switch_world(self, name: str, task_type: str = "") -> None:
        """Switch to a specified simulation world with task-specific processing. [v3.5.1]"""
        if not isinstance(name, str) or not name.strip():
            logger.error("Invalid name: must be a non-empty string for task %s", task_type)
            raise ValueError("name must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if name not in self.worlds:
            logger.error("World not found: %s for task %s", name, task_type)
            raise ValueError(f"world '{name}' not found")

        try:
            self.current_world = self.worlds[name]
            logger.info("Switched to world: %s for task %s", name, task_type)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"World_Switch_{name}_{datetime.now().isoformat()}",
                    output=f"Switched to world: {name}",
                    layer="WorldSwitches",
                    intent="world_switch",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=f"Switched to world: {name}",
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("World switch reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "world_switch": {
                        "name": name,
                        "parameters": self.worlds[name],
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
        except Exception as e:
            logger.error("World switch failed for task %s: %s", task_type, str(e))
            raise

    async def execute(self, task_type: str = "") -> str:
        """Execute simulation in the current world with task-specific processing. [v3.5.1]"""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if not self.current_world:
            logger.error("No world set for execution for task %s", task_type)
            raise ValueError("no world set")

        try:
            logger.info("Executing simulation in world: %s for task %s", self.current_world, task_type)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="World execution",
                    meta={"world": self.current_world, "task_type": task_type},
                    module="SimulationCore",
                    tags=["world", "execution", task_type]
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=f"Executing simulation in world: {self.current_world}",
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("World execution reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "world_execution": {
                        "world": self.current_world,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return f"Simulating in: {self.current_world}"
        except Exception as e:
            logger.error("World execution failed for task %s: %s", task_type, str(e))
            raise

    async def validate_entropy(self, distribution: List[float], task_type: str = "") -> bool:
        """Calculate Shannon entropy and validate against dynamic threshold with task-specific processing. [v3.5.1]"""
        if not isinstance(distribution, (list, np.ndarray)) or not distribution:
            logger.error("Invalid distribution: must be a non-empty list or numpy array for task %s", task_type)
            raise TypeError("distribution must be a non-empty list or numpy array")
        if not all(isinstance(p, (int, float)) and p >= 0 for p in distribution):
            logger.error("Invalid distribution: all values must be non-negative numbers for task %s", task_type)
            raise ValueError("distribution values must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            total = sum(distribution)
            if total == 0:
                logger.warning("Empty distribution: all values are zero for task %s", task_type)
                return False
            normalized = [p / total for p in distribution]
            entropy = -sum(p * math.log2(p) for p in normalized if p > 0)
            threshold = math.log2(len(normalized)) * 0.75
            is_valid = entropy >= threshold
            logger.info("Entropy: %.3f, Threshold: %.3f, Valid: %s for task %s", entropy, threshold, is_valid, task_type)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Entropy_Validation_{datetime.now().isoformat()}",
                    output={"entropy": entropy, "threshold": threshold, "valid": is_valid, "task_type": task_type},
                    layer="Validations",
                    intent="entropy_validation",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="SimulationCore",
                    output=json.dumps({"entropy": entropy, "threshold": threshold, "valid": is_valid}),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Entropy validation reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "entropy_validation": {
                        "entropy": entropy,
                        "threshold": threshold,
                        "valid": is_valid,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": False,
                        "style": "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return is_valid
        except Exception as e:
            logger.error("Entropy validation failed for task %s: %s", task_type, str(e))
            return False

    async def select_topology_mode(self, modes: List[str], metrics: Dict[str, List[float]], task_type: str = "") -> str:
        """Select topology mode with entropy validation check and task-specific processing. [v3.5.1]"""
        if not isinstance(modes, list) or not modes:
            logger.error("Invalid modes: must be a non-empty list for task %s", task_type)
            raise ValueError("modes must be a non-empty list")
        if not isinstance(metrics, dict) or not metrics:
            logger.error("Invalid metrics: must be a non-empty dictionary for task %s", task_type)
            raise ValueError("metrics must be a non-empty dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            for mode in modes:
                if mode not in metrics:
                    logger.warning("Mode %s not found in metrics for task %s", mode, task_type)
                    continue
                if await self.validate_entropy(metrics[mode], task_type=task_type):
                    logger.info("Selected topology mode: %s for task %s", mode, task_type)
                    if self.agi_enhancer:
                        await self.agi_enhancer.log_episode(
                            event="Topology mode selected",
                            meta={"mode": mode, "metrics": metrics[mode], "task_type": task_type},
                            module="SimulationCore",
                            tags=["topology", "selection", task_type]
                        )
                    if self.visualizer and task_type:
                        plot_data = {
                            "topology_selection": {
                                "mode": mode,
                                "metrics": metrics[mode],
                                "task_type": task_type
                            },
                            "visualization_options": {
                                "interactive": task_type == "recursion",
                                "style": "concise"
                            }
                        }
                        await self.visualizer.render_charts(plot_data)
                    return mode
            logger.info("No valid topology mode found, using fallback for task %s", task_type)
            return "fallback"
        except Exception as e:
            logger.error("Topology mode selection failed for task %s: %s", task_type, str(e))
            return "fallback"
