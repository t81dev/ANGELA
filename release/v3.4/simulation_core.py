"""
ANGELA Cognitive System Module: SimulationCore
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a SimulationCore class for agent simulations, impact validations,
and environment simulations in the ANGELA v3.5 architecture, integrating ToCA physics.
"""

import logging
import math
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from threading import Lock
from collections import deque
from functools import lru_cache

from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from modules.memory_manager import MemoryManager
from modules.alignment_guard import enforce_alignment
from modules import (
    multi_modal_fusion as multi_modal_fusion_module,
    error_recovery as error_recovery_module
)
from index import zeta_consequence, theta_causality, rho_agency, TraitOverlayManager

logger = logging.getLogger("ANGELA.SimulationCore")

class ToCATraitEngine:
    """Cyber-Physics Engine based on ToCA dynamics for agent simulations.

    Attributes:
        k_m (float): Motion coupling constant.
        delta_m (float): Damping modulation factor.
    """
    def __init__(self, k_m: float = 1e-3, delta_m: float = 1e4):
        self.k_m = k_m
        self.delta_m = delta_m

    @lru_cache(maxsize=100)
    def evolve(self, x_tuple: tuple, t_tuple: tuple, user_data_tuple: Optional[tuple] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evolve ToCA fields for simulation.

        Args:
            x_tuple: Spatial coordinates as a tuple.
            t_tuple: Time coordinates as a tuple.
            user_data_tuple: Optional user data as a tuple.

        Returns:
            Tuple of phi (scalar field), lambda_t (damping field), and v_m (motion potential).
        """
        x = np.array(x_tuple)
        t = np.array(t_tuple)
        user_data = np.array(user_data_tuple) if user_data_tuple else None
        
        if not isinstance(x, np.ndarray) or not isinstance(t, np.ndarray):
            logger.error("Invalid input: x and t must be numpy arrays")
            raise TypeError("x and t must be numpy arrays")
        if user_data is not None and not isinstance(user_data, np.ndarray):
            logger.error("Invalid user_data: must be a numpy array")
            raise TypeError("user_data must be a numpy array")
        
        try:
            x = np.clip(x, 1e-10, 1e10)
            v_m = self.k_m * np.gradient(3e3 * 1.989 / (x**2 + 1e-10))
            phi = np.sin(t * 1e-3) * 1e-3 * (1 + v_m * np.gradient(x))
            if user_data is not None:
                phi += np.mean(user_data) * 1e-4
            lambda_t = 1.1e-3 * np.exp(-2e-2 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * self.delta_m)
            return phi, lambda_t, v_m
        except Exception as e:
            logger.error("ToCA evolution failed: %s", str(e))
            raise

    def update_fields_with_agents(self, phi: np.ndarray, lambda_t: np.ndarray, agent_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update fields with agent interactions."""
        if not all(isinstance(arr, np.ndarray) for arr in [phi, lambda_t, agent_matrix]):
            logger.error("Invalid inputs: phi, lambda_t, and agent_matrix must be numpy arrays")
            raise TypeError("inputs must be numpy arrays")
        
        try:
            interaction_energy = np.dot(agent_matrix, np.sin(phi)) * 1e-3
            phi = phi + interaction_energy
            lambda_t = lambda_t * (1 + 0.001 * np.sum(agent_matrix, axis=0))
            return phi, lambda_t
        except Exception as e:
            logger.error("Field update with agents failed: %s", str(e))
            raise

class SimulationCore:
    """Core simulation engine for ANGELA v3.5, integrating ToCA physics and agent dynamics.

    Attributes:
        visualizer (Visualizer): Module for rendering simulation outputs.
        simulation_history (deque): History of simulation states (max 1000).
        ledger (deque): Audit trail of simulation records with hashes (max 1000).
        agi_enhancer (AGIEnhancer): Enhancer for logging and adaptation.
        memory_manager (MemoryManager): Manager for storing simulation data.
        multi_modal_fusion (MultiModalFusion): Module for multi-modal synthesis.
        error_recovery (ErrorRecovery): Module for error handling and recovery.
        toca_engine (ToCATraitEngine): Engine for ToCA physics simulations.
        overlay_router (TraitOverlayManager): Manager for trait overlays.
        worlds (Dict[str, Dict]): Dictionary of defined simulation worlds.
        current_world (Optional[Dict]): Currently active simulation world.
        ledger_lock (Lock): Thread-safe lock for ledger updates.
    """
    def __init__(self,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 visualizer: Optional['Visualizer'] = None,
                 memory_manager: Optional['MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 toca_engine: Optional['ToCATraitEngine'] = None,
                 overlay_router: Optional['TraitOverlayManager'] = None):
        self.visualizer = visualizer or Visualizer()
        self.simulation_history = deque(maxlen=1000)
        self.ledger = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.memory_manager = memory_manager or MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager)
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.toca_engine = toca_engine or ToCATraitEngine()
        self.overlay_router = overlay_router or TraitOverlayManager()
        self.worlds = {}
        self.current_world = None
        self.ledger_lock = Lock()
        logger.info("SimulationCore initialized")

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    def _record_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Record simulation state with hash for integrity."""
        if not isinstance(data, dict):
            logger.error("Invalid data: must be a dictionary")
            raise TypeError("data must be a dictionary")
        
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "hash": hashlib.sha256(
                    json.dumps(data, sort_keys=True, default=self._json_serializer).encode()
                ).hexdigest()
            }
            with self.ledger_lock:
                self.ledger.append(record)
                self.simulation_history.append(record)
                if self.memory_manager:
                    self.memory_manager.store(
                        query=f"Ledger_{record['timestamp']}",
                        output=record,
                        layer="Ledger",
                        intent="state_record"
                    )
            return record
        except Exception as e:
            logger.error("State recording failed: %s", str(e))
            raise

    async def run(self, results: str, context: Optional[Dict[str, Any]] = None,
                  scenarios: int = 3, agents: int = 2, export_report: bool = False,
                  export_format: str = "pdf", actor_id: str = "default_agent") -> Dict[str, Any]:
        """Run a simulation with specified parameters."""
        if not isinstance(results, str) or not results.strip():
            logger.error("Invalid results: must be a non-empty string")
            raise ValueError("results must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(scenarios, int) or scenarios < 1:
            logger.error("Invalid scenarios: must be a positive integer")
            raise ValueError("scenarios must be a positive integer")
        if not isinstance(agents, int) or agents < 1:
            logger.error("Invalid agents: must be a positive integer")
            raise ValueError("agents must be a positive integer")
        if not isinstance(export_format, str) or export_format not in ["pdf", "json", "html"]:
            logger.error("Invalid export_format: must be 'pdf', 'json', or 'html'")
            raise ValueError("export_format must be 'pdf', 'json', or 'html'")
        if not isinstance(actor_id, str) or not actor_id.strip():
            logger.error("Invalid actor_id: must be a non-empty string")
            raise ValueError("actor_id must be a non-empty string")
        
        logger.info("Running simulation with %d agents and %d scenarios", agents, scenarios)
        try:
            t = time.time() % 1.0
            causality = max(0.0, min(theta_causality(t), 1.0))
            agency = max(0.0, min(rho_agency(t), 1.0))

            x = np.linspace(0.1, 20, 100)
            t_vals = np.linspace(0.1, 20, 100)
            agent_matrix = np.random.rand(agents, 100)

            phi, lambda_field, v_m = self.toca_engine.evolve(tuple(x), tuple(t_vals))
            phi, lambda_field = self.toca_engine.update_fields_with_agents(phi, lambda_field, agent_matrix)
            energy_cost = float(np.mean(np.abs(phi)) * 1e3)

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
                "estimated_energy_cost": energy_cost
            }

            if not enforce_alignment(prompt):
                logger.warning("Alignment guard rejected simulation request")
                return {"error": "Simulation rejected due to alignment constraints"}

            query_key = f"Simulation_{results[:50]}_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM")
            if cached_output:
                logger.info("Retrieved cached simulation output")
                simulation_output = cached_output
            else:
                simulation_output = await call_gpt(f"Simulate agent outcomes: {json.dumps(prompt, default=self._json_serializer)}")
                if not isinstance(simulation_output, (dict, str)):
                    logger.error("Invalid simulation output: must be a dictionary or string")
                    raise ValueError("simulation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=simulation_output,
                        layer="STM",
                        intent="simulation"
                    )

            state_record = self._record_state({
                "actor": actor_id,
                "action": "run_simulation",
                "traits": prompt["traits"],
                "energy_cost": energy_cost,
                "output": simulation_output
            })

            self.simulation_history.append(state_record)

            if export_report and self.memory_manager:
                await self.memory_manager.promote_to_ltm(query_key)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Simulation run",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["simulation", "run"]
                )
                self.agi_enhancer.reflect_and_adapt("SimulationCore: scenario simulation complete")

            if self.visualizer:
                await self.visualizer.render_charts(simulation_output)

            if export_report and self.visualizer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_report_{timestamp}.{export_format}"
                logger.info("Exporting report: %s", filename)
                await self.visualizer.export_report(simulation_output, filename=filename, format=export_format)

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"prompt": prompt, "output": simulation_output},
                    summary_style="insightful"
                )
                state_record["synthesis"] = synthesis

            return simulation_output
        except Exception as e:
            logger.error("Simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run(results, context, scenarios, agents, export_report, export_format, actor_id),
                default={"error": str(e)}
            )

    async def validate_impact(self, proposed_action: str, agents: int = 2,
                             export_report: bool = False, export_format: str = "pdf",
                             actor_id: str = "validator_agent") -> Dict[str, Any]:
        """Validate the impact of a proposed action."""
        if not isinstance(proposed_action, str) or not proposed_action.strip():
            logger.error("Invalid proposed_action: must be a non-empty string")
            raise ValueError("proposed_action must be a non-empty string")
        
        logger.info("Validating impact of proposed action: %s", proposed_action)
        try:
            t = time.time() % 1.0
            consequence = max(0.0, min(zeta_consequence(t), 1.0))

            prompt = f"""
            Evaluate the following proposed action:
            {proposed_action}

            Trait:
            - zeta_consequence = {consequence:.3f}

            Analyze positive/negative outcomes, agent variations, risk scores (1-10), and recommend: Proceed / Modify / Abort.
            """
            if not enforce_alignment({"action": proposed_action, "consequence": consequence}):
                logger.warning("Alignment guard blocked impact validation")
                return {"error": "Validation blocked by alignment rules"}

            query_key = f"Validation_{proposed_action[:50]}_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM")
            if cached_output:
                logger.info("Retrieved cached validation output")
                validation_output = cached_output
            else:
                validation_output = await call_gpt(prompt)
                if not isinstance(validation_output, (dict, str)):
                    logger.error("Invalid validation output: must be a dictionary or string")
                    raise ValueError("validation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=validation_output,
                        layer="STM",
                        intent="impact_validation"
                    )

            state_record = self._record_state({
                "actor": actor_id,
                "action": "validate_impact",
                "trait_zeta_consequence": consequence,
                "proposed_action": proposed_action,
                "output": validation_output
            })

            self.simulation_history.append(state_record)

            if self.visualizer:
                await self.visualizer.render_charts(validation_output)

            if export_report and self.visualizer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"impact_validation_{timestamp}.{export_format}"
                logger.info("Exporting validation report: %s", filename)
                await self.visualizer.export_report(validation_output, filename=filename, format=export_format)

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"action": proposed_action, "output": validation_output, "consequence": consequence},
                    summary_style="concise"
                )
                state_record["synthesis"] = synthesis

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Impact validation",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["validation", "impact"]
                )
                self.agi_enhancer.reflect_and_adapt("SimulationCore: impact validation complete")

            return validation_output
        except Exception as e:
            logger.error("Impact validation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.validate_impact(proposed_action, agents, export_report, export_format, actor_id),
                default={"error": str(e)}
            )

    async def simulate_environment(self, environment_config: Dict[str, Any], agents: int = 2,
                                  steps: int = 10, actor_id: str = "env_agent",
                                  goal: Optional[str] = None) -> Dict[str, Any]:
        """Simulate agents in a configured environment."""
        if not isinstance(environment_config, dict):
            logger.error("Invalid environment_config: must be a dictionary")
            raise TypeError("environment_config must be a dictionary")
        if not isinstance(steps, int) or steps < 1:
            logger.error("Invalid steps: must be a positive integer")
            raise ValueError("steps must be a positive integer")
        
        logger.info("Running environment simulation with %d agents and %d steps", agents, steps)
        try:
            if not enforce_alignment({"environment": environment_config, "goal": goal}):
                logger.warning("Alignment guard rejected environment simulation")
                return {"error": "Simulation blocked due to environment constraints"}

            prompt = f"""
            Simulate agents in this environment:
            {json.dumps(environment_config, default=self._json_serializer)}

            Steps: {steps} | Agents: {agents}
            Goal: {goal if goal else 'N/A'}
            Describe interactions, environmental changes, risks/opportunities.
            """
            query_key = f"Environment_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM")
            if cached_output:
                logger.info("Retrieved cached environment simulation output")
                environment_simulation = cached_output
            else:
                environment_simulation = await call_gpt(prompt)
                if not isinstance(environment_simulation, (dict, str)):
                    logger.error("Invalid environment simulation output: must be a dictionary or string")
                    raise ValueError("environment simulation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=environment_simulation,
                        layer="STM",
                        intent="environment_simulation"
                    )

            state_record = self._record_state({
                "actor": actor_id,
                "action": "simulate_environment",
                "config": environment_config,
                "steps": steps,
                "goal": goal,
                "output": environment_simulation
            })

            self.simulation_history.append(state_record)

            if self.meta_cognition:
                review = await self.meta_cognition.review_reasoning(environment_simulation)
                state_record["review"] = review

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Environment simulation",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["environment", "simulation"]
                )
                self.agi_enhancer.reflect_and_adapt("SimulationCore: environment simulation complete")

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"config": environment_config, "output": environment_simulation, "goal": goal},
                    summary_style="insightful"
                )
                state_record["synthesis"] = synthesis

            return environment_simulation
        except Exception as e:
            logger.error("Environment simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_environment(environment_config, agents, steps, actor_id, goal),
                default={"error": str(e)}
            )

    async def replay_intentions(self, memory_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trace past intentions and return a replay sequence."""
        if not isinstance(memory_log, list):
            logger.error("Invalid memory_log: must be a list")
            raise TypeError("memory_log must be a list")
        
        try:
            replay = [
                {
                    "timestamp": entry.get("timestamp"),
                    "goal": entry.get("goal"),
                    "intention": entry.get("intention"),
                    "traits": entry.get("traits", {})
                }
                for entry in memory_log
                if isinstance(entry, dict) and "goal" in entry
            ]
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Replay_{datetime.now().isoformat()}",
                    output=str(replay),
                    layer="Replays",
                    intent="intention_replay"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Intentions replayed",
                    meta={"replay": replay},
                    module="SimulationCore",
                    tags=["replay", "intentions"]
                )
            return replay
        except Exception as e:
            logger.error("Intention replay failed: %s", str(e))
            raise

    async def fabricate_reality(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Construct immersive meta-environments from symbolic templates."""
        if not isinstance(parameters, dict):
            logger.error("Invalid parameters: must be a dictionary")
            raise TypeError("parameters must be a dictionary")
        
        logger.info("Fabricating reality with parameters: %s", parameters)
        try:
            environment = {"fabricated_world": True, "parameters": parameters}
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"parameters": parameters},
                    summary_style="insightful"
                )
                environment["synthesis"] = synthesis
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reality_Fabrication_{datetime.now().isoformat()}",
                    output=str(environment),
                    layer="Realities",
                    intent="reality_fabrication"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reality fabricated",
                    meta=environment,
                    module="SimulationCore",
                    tags=["reality", "fabrication"]
                )
            return environment
        except Exception as e:
            logger.error("Reality fabrication failed: %s", str(e))
            raise

    async def synthesize_self_world(self, identity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure persistent identity integration in self-generated environments."""
        if not isinstance(identity_data, dict):
            logger.error("Invalid identity_data: must be a dictionary")
            raise TypeError("identity_data must be a dictionary")
        
        try:
            result = {"identity": identity_data, "coherence_score": 0.97}
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"identity": identity_data},
                    summary_style="concise"
                )
                result["synthesis"] = synthesis
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Self_World_Synthesis_{datetime.now().isoformat()}",
                    output=str(result),
                    layer="Identities",
                    intent="self_world_synthesis"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self-world synthesized",
                    meta=result,
                    module="SimulationCore",
                    tags=["identity", "synthesis"]
                )
            return result
        except Exception as e:
            logger.error("Self-world synthesis failed: %s", str(e))
            raise

    def define_world(self, name: str, parameters: Dict[str, Any]) -> None:
        """Define a simulation world with given parameters."""
        if not isinstance(name, str) or not name.strip():
            logger.error("Invalid world name: must be a non-empty string")
            raise ValueError("world name must be a non-empty string")
        if not isinstance(parameters, dict):
            logger.error("Invalid parameters: must be a dictionary")
            raise TypeError("parameters must be a dictionary")
        
        self.worlds[name] = parameters
        logger.info("Defined world: %s", name)
        if self.memory_manager:
            self.memory_manager.store(
                query=f"World_Definition_{name}_{datetime.now().isoformat()}",
                output=parameters,
                layer="Worlds",
                intent="world_definition"
            )

    async def switch_world(self, name: str) -> None:
        """Switch to a specified simulation world."""
        if name not in self.worlds:
            logger.error("World not found: %s", name)
            raise ValueError(f"world '{name}' not found")
        
        self.current_world = self.worlds[name]
        logger.info("Switched to world: %s", name)
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"World_Switch_{name}_{datetime.now().isoformat()}",
                output=f"Switched to world: {name}",
                layer="WorldSwitches",
                intent="world_switch"
            )

    async def execute(self) -> str:
        """Execute simulation in the current world."""
        if not self.current_world:
            logger.error("No world set for execution")
            raise ValueError("no world set")
        
        logger.info("Executing simulation in world: %s", self.current_world)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="World execution",
                meta={"world": self.current_world},
                module="SimulationCore",
                tags=["world", "execution"]
            )
        return f"Simulating in: {self.current_world}"

    def validate_entropy(self, distribution: List[float]) -> bool:
        """Calculate Shannon entropy and validate against dynamic threshold."""
        if not isinstance(distribution, (list, np.ndarray)) or not distribution:
            logger.error("Invalid distribution: must be a non-empty list or numpy array")
            raise TypeError("distribution must be a non-empty list or numpy array")
        if not all(isinstance(p, (int, float)) and p >= 0 for p in distribution):
            logger.error("Invalid distribution: all values must be non-negative numbers")
            raise ValueError("distribution values must be non-negative")
        
        try:
            total = sum(distribution)
            if total == 0:
                logger.warning("Empty distribution: all values are zero")
                return False
            normalized = [p / total for p in distribution]
            entropy = -sum(p * math.log2(p) for p in normalized if p > 0)
            threshold = math.log2(len(normalized)) * 0.75
            is_valid = entropy >= threshold
            logger.info("Entropy: %.3f, Threshold: %.3f, Valid: %s", entropy, threshold, is_valid)
            if self.memory_manager:
                self.memory_manager.store(
                    query=f"Entropy_Validation_{datetime.now().isoformat()}",
                    output={"entropy": entropy, "threshold": threshold, "valid": is_valid},
                    layer="Validations",
                    intent="entropy_validation"
                )
            return is_valid
        except Exception as e:
            logger.error("Entropy validation failed: %s", str(e))
            return False

    async def select_topology_mode(self, modes: List[str], metrics: Dict[str, List[float]]) -> str:
        """Select topology mode with entropy validation check."""
        if not isinstance(modes, list) or not modes:
            logger.error("Invalid modes: must be a non-empty list")
            raise ValueError("modes must be a non-empty list")
        if not isinstance(metrics, dict) or not metrics:
            logger.error("Invalid metrics: must be a non-empty dictionary")
            raise ValueError("metrics must be a non-empty dictionary")
        
        try:
            for mode in modes:
                if mode not in metrics:
                    logger.warning("Mode %s not found in metrics", mode)
                    continue
                if self.validate_entropy(metrics[mode]):
                    logger.info("Selected topology mode: %s", mode)
                    if self.agi_enhancer:
                        self.agi_enhancer.log_episode(
                            event="Topology mode selected",
                            meta={"mode": mode, "metrics": metrics[mode]},
                            module="SimulationCore",
                            tags=["topology", "selection"]
                        )
                    return mode
            logger.info("No valid topology mode found, using fallback")
            return "fallback"
        except Exception as e:
            logger.error("Topology mode selection failed: %s", str(e))
            return "fallback"
