"""
ANGELA Cognitive System Module: Galaxy Rotation and Agent Conflict Simulation
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module extends SimulationCore for galaxy rotation curve simulations using AGRF
and multi-agent conflict modeling with ToCA dynamics.
"""

import logging
import math
import json
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from datetime import datetime
from threading import Lock
from collections import deque
from scipy.constants import G
from functools import lru_cache

from modules.simulation_core import SimulationCore, ToCATraitEngine
from modules.visualizer import Visualizer
from modules.memory_manager import MemoryManager
from modules import multi_modal_fusion as multi_modal_fusion_module
from modules import error_recovery as error_recovery_module
from index import zeta_consequence, theta_causality, rho_agency, TraitOverlayManager

logger = logging.getLogger("ANGELA.SimulationCore")

# Constants
G_SI = G  # m^3 kg^-1 s^-2
KPC_TO_M = 3.0857e19  # Conversion factor from kpc to meters
MSUN_TO_KG = 1.989e30  # Solar mass in kg
k_default = 0.85
epsilon_default = 0.015
r_halo_default = 20.0  # kpc

class SimulationCore(SimulationCore):
    """Extended SimulationCore for galaxy rotation and agent conflict simulations."""
    def __init__(self,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 visualizer: Optional['Visualizer'] = None,
                 memory_manager: Optional['MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 toca_engine: Optional['ToCATraitEngine'] = None,
                 overlay_router: Optional['TraitOverlayManager'] = None):
        super().__init__(agi_enhancer, visualizer, memory_manager, multi_modal_fusion, error_recovery, toca_engine, overlay_router)
        self.omega = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000)
        }
        self.omega_lock = Lock()
        self.ethical_rules = []
        self.constitution = {}
        logger.info("Extended SimulationCore initialized")

    async def modulate_simulation_with_traits(self, trait_weights: Dict[str, float]) -> None:
        """Adjust simulation difficulty based on trait weights."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        if not all(isinstance(v, (int, float)) and v >= 0 for v in trait_weights.values()):
            logger.error("Invalid trait_weights: values must be non-negative numbers")
            raise ValueError("trait_weights values must be non-negative")
        
        try:
            phi_weight = trait_weights.get('ϕ', 0.5)
            if phi_weight > 0.7:
                logger.info("ToCA Simulation: ϕ-prioritized mode activated")
                self.toca_engine.k_m = k_default * 1.5
            else:
                self.toca_engine.k_m = k_default
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Modulation_{datetime.now().isoformat()}",
                    output={"trait_weights": trait_weights, "phi_weight": phi_weight},
                    layer="Traits",
                    intent="modulate_simulation"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Simulation modulated",
                    meta={"trait_weights": trait_weights},
                    module="SimulationCore",
                    tags=["modulation", "traits"]
                )
        except Exception as e:
            logger.error("Trait modulation failed: %s", str(e))
            raise

    def compute_AGRF_curve(self, v_obs_kms: np.ndarray, M_baryon_solar: np.ndarray, r_kpc: np.ndarray,
                           k: float = k_default, epsilon: float = epsilon_default, r_halo: float = r_halo_default) -> np.ndarray:
        """Compute galaxy rotation curve using AGRF."""
        if not all(isinstance(arr, np.ndarray) for arr in [v_obs_kms, M_baryon_solar, r_kpc]):
            logger.error("Invalid inputs: v_obs_kms, M_baryon_solar, r_kpc must be numpy arrays")
            raise TypeError("inputs must be numpy arrays")
        if not all(isinstance(x, (int, float)) for x in [k, epsilon, r_halo]):
            logger.error("Invalid parameters: k, epsilon, r_halo must be numbers")
            raise TypeError("parameters must be numbers")
        if np.any(r_kpc <= 0):
            logger.error("Invalid r_kpc: must be positive")
            raise ValueError("r_kpc must be positive")
        if k <= 0 or epsilon < 0 or r_halo <= 0:
            logger.error("Invalid parameters: k and r_halo must be positive, epsilon non-negative")
            raise ValueError("invalid parameters")
        
        try:
            r_m = r_kpc * KPC_TO_M
            M_b_kg = M_baryon_solar * MSUN_TO_KG
            v_obs_ms = v_obs_kms * 1e3
            M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
            M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
            M_total = M_b_kg + M_AGRF
            v_total_ms = np.sqrt(np.clip(G_SI * M_total / r_m, 0, np.inf))
            return v_total_ms / 1e3
        except Exception as e:
            logger.error("AGRF curve computation failed: %s", str(e))
            raise

    async def simulate_galaxy_rotation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable,
                                      k: float = k_default, epsilon: float = epsilon_default) -> np.ndarray:
        """Simulate galaxy rotation curve with ToCA dynamics."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")
        
        try:
            v_total = self.compute_AGRF_curve(v_obs_func(r_kpc), M_b_func(r_kpc), r_kpc, k, epsilon)
            fields = self.toca_engine.evolve(tuple(r_kpc), tuple(np.linspace(0.1, 20, len(r_kpc))))
            phi, _, _ = fields
            v_total = v_total * (1 + 0.1 * np.mean(phi))
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Galaxy_Rotation_{datetime.now().isoformat()}",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "phi": phi.tolist()},
                    layer="Simulations",
                    intent="galaxy_rotation"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Galaxy rotation simulated",
                    meta={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist()},
                    module="SimulationCore",
                    tags=["galaxy", "rotation"]
                )
            return v_total
        except Exception as e:
            logger.error("Galaxy rotation simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon),
                default=np.zeros_like(r_kpc)
            )

    @lru_cache(maxsize=100)
    def compute_trait_fields(self, r_kpc_tuple: tuple, v_obs_tuple: tuple, v_sim_tuple: tuple,
                            time_elapsed: float = 1.0, tau_persistence: float = 10.0) -> Tuple[np.ndarray, ...]:
        """Compute ToCA trait fields for simulation."""
        r_kpc = np.array(r_kpc_tuple)
        v_obs = np.array(v_obs_tuple)
        v_sim = np.array(v_sim_tuple)
        
        if not all(isinstance(arr, np.ndarray) for arr in [r_kpc, v_obs, v_sim]):
            logger.error("Invalid inputs: r_kpc, v_obs, v_sim must be numpy arrays")
            raise TypeError("inputs must be numpy arrays")
        if not isinstance(time_elapsed, (int, float)) or time_elapsed < 0:
            logger.error("Invalid time_elapsed: must be non-negative")
            raise ValueError("time_elapsed must be non-negative")
        if not isinstance(tau_persistence, (int, float)) or tau_persistence <= 0:
            logger.error("Invalid tau_persistence: must be positive")
            raise ValueError("tau_persistence must be positive")
        
        try:
            gamma_field = np.log(1 + np.clip(r_kpc, 1e-10, np.inf)) * 0.5
            beta_field = np.abs(v_obs - v_sim) / (np.max(np.abs(v_obs)) + 1e-10)
            zeta_field = 1 / (1 + np.gradient(v_sim)**2)
            eta_field = np.exp(-time_elapsed / tau_persistence)
            psi_field = np.gradient(v_sim) / (np.gradient(r_kpc) + 1e-10)
            lambda_field = np.cos(r_kpc / r_halo_default * np.pi)
            phi_field = k_default * np.exp(-epsilon_default * r_kpc / r_halo_default)
            phi_prime = -epsilon_default * phi_field / r_halo_default
            beta_psi_interaction = beta_field * psi_field
            return (gamma_field, beta_field, zeta_field, eta_field, psi_field,
                    lambda_field, phi_field, phi_prime, beta_psi_interaction)
        except Exception as e:
            logger.error("Trait field computation failed: %s", str(e))
            raise

    async def plot_AGRF_simulation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable, label: str = "ToCA-AGRF") -> None:
        """Plot galaxy rotation curve and trait fields using Visualizer."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")
        
        try:
            v_sim = await self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func)
            v_obs = v_obs_func(r_kpc)
            fields = self.compute_trait_fields(tuple(r_kpc), tuple(v_obs), tuple(v_sim))
            gamma_field, beta_field, zeta_field, eta_field, psi_field, lambda_field, phi_field, phi_prime, beta_psi_interaction = fields

            plot_data = {
                "rotation_curve": {
                    "r_kpc": r_kpc.tolist(),
                    "v_obs": v_obs.tolist(),
                    "v_sim": v_sim.tolist(),
                    "phi_field": phi_field.tolist(),
                    "phi_prime": phi_prime.tolist(),
                    "label": label
                },
                "trait_fields": {
                    "gamma": gamma_field.tolist(),
                    "beta": beta_field.tolist(),
                    "zeta": zeta_field.tolist(),
                    "eta": eta_field,
                    "psi": psi_field.tolist(),
                    "lambda": lambda_field.tolist()
                },
                "interaction": {
                    "beta_psi": beta_psi_interaction.tolist()
                }
            }

            with self.omega_lock:
                self.omega["timeline"].append({
                    "type": "AGRF Simulation",
                    "r_kpc": r_kpc.tolist(),
                    "v_obs": v_obs.tolist(),
                    "v_sim": v_sim.tolist(),
                    "phi_field": phi_field.tolist(),
                    "phi_prime": phi_prime.tolist(),
                    "traits": {
                        "γ": gamma_field.tolist(),
                        "β": beta_field.tolist(),
                        "ζ": zeta_field.tolist(),
                        "η": eta_field,
                        "ψ": psi_field.tolist(),
                        "λ": lambda_field.tolist()
                    }
                })

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data=plot_data,
                    summary_style="insightful"
                )
                plot_data["synthesis"] = synthesis

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"AGRF_Plot_{datetime.now().isoformat()}",
                    output=plot_data,
                    layer="Plots",
                    intent="visualization"
                )

            if self.visualizer:
                await self.visualizer.render_charts(plot_data)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="AGRF simulation plotted",
                    meta=plot_data,
                    module="SimulationCore",
                    tags=["visualization", "galaxy"]
                )
        except Exception as e:
            logger.error("AGRF simulation plot failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plot_AGRF_simulation(r_kpc, M_b_func, v_obs_func, label),
                default=None
            )

    async def simulate_interaction(self, agent_profiles: List['Agent'], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate interactions among agents in a given context."""
        if not isinstance(agent_profiles, list):
            logger.error("Invalid agent_profiles: must be a list")
            raise TypeError("agent_profiles must be a list")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        
        try:
            results = []
            for agent in agent_profiles:
                if not hasattr(agent, 'respond'):
                    logger.warning("Agent %s lacks respond method", getattr(agent, 'id', 'unknown'))
                    continue
                response = await agent.respond(context)
                results.append({"agent_id": getattr(agent, 'id', 'unknown'), "response": response})

            interaction_data = {"interactions": results}
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data=interaction_data,
                    summary_style="insightful"
                )
                interaction_data["synthesis"] = synthesis

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Interaction_{datetime.now().isoformat()}",
                    output=interaction_data,
                    layer="Interactions",
                    intent="agent_interaction"
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Agent interaction",
                    meta=interaction_data,
                    module="SimulationCore",
                    tags=["interaction", "agents"]
                )
            return interaction_data
        except Exception as e:
            logger.error("Agent interaction simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_interaction(agent_profiles, context),
                default={"error": str(e)}
            )

    async def simulate_multiagent_conflicts(self, agent_pool: List['Agent'], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate pairwise conflicts among agents based on traits."""
        if not isinstance(agent_pool, list) or len(agent_pool) < 2:
            logger.error("Invalid agent_pool: must be a list with at least two agents")
            raise ValueError("agent_pool must have at least two agents")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        
        try:
            outcomes = []
            for i in range(len(agent_pool)):
                for j in range(i + 1, len(agent_pool)):
                    agent1, agent2 = agent_pool[i], agent_pool[j]
                    if not hasattr(agent1, 'resolve') or not hasattr(agent2, 'resolve'):
                        logger.warning("Agent %s or %s lacks resolve method", getattr(agent1, 'id', i), getattr(agent2, 'id', j))
                        continue
                    beta1 = getattr(agent1, 'traits', {}).get('β', 0.5)
                    beta2 = getattr(agent2, 'traits', {}).get('β', 0.5)
                    tau1 = getattr(agent1, 'traits', {}).get('τ', 0.5)
                    tau2 = getattr(agent2, 'traits', {}).get('τ', 0.5)
                    score = abs(beta1 - beta2) + abs(tau1 - tau2)

                    if abs(beta1 - beta2) < 0.1:
                        outcome = await agent1.resolve(context) if tau1 > tau2 else await agent2.resolve(context)
                    else:
                        outcome = await agent1.resolve(context) if beta1 > beta2 else await agent2.resolve(context)

                    outcomes.append({
                        "pair": (getattr(agent1, 'id', i), getattr(agent2, 'id', j)),
                        "conflict_score": score,
                        "outcome": outcome,
                        "traits_involved": {"β1": beta1, "β2": beta2, "τ1": tau1, "τ2": tau2}
                    })

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Conflict_Simulation_{datetime.now().isoformat()}",
                    output=outcomes,
                    layer="Conflicts",
                    intent="conflict_simulation"
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Multi-agent conflict simulation",
                    meta={"outcomes": outcomes},
                    module="SimulationCore",
                    tags=["conflict", "agents"]
                )
            return outcomes
        except Exception as e:
            logger.error("Multi-agent conflict simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_multiagent_conflicts(agent_pool, context),
                default={"error": str(e)}
            )

    async def update_ethics_protocol(self, new_rules: Dict[str, Any], consensus_agents: Optional[List['Agent']] = None) -> None:
        """Adapt ethical rules live, supporting consensus/negotiation."""
        if not isinstance(new_rules, dict):
            logger.error("Invalid new_rules: must be a dictionary")
            raise TypeError("new_rules must be a dictionary")
        
        try:
            self.ethical_rules = new_rules
            if consensus_agents:
                self.ethics_consensus_log = getattr(self, 'ethics_consensus_log', [])
                self.ethics_consensus_log.append((new_rules, [getattr(agent, 'id', 'unknown') for agent in consensus_agents]))
            logger.info("Ethics protocol updated via consensus")
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ethics_Update_{datetime.now().isoformat()}",
                    output={"rules": new_rules, "agents": [getattr(agent, 'id', 'unknown') for agent in consensus_agents] if consensus_agents else []},
                    layer="Ethics",
                    intent="ethics_update"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Ethics protocol updated",
                    meta={"rules": new_rules},
                    module="SimulationCore",
                    tags=["ethics", "update"]
                )
        except Exception as e:
            logger.error("Ethics protocol update failed: %s", str(e))
            raise

    async def synchronize_norms(self, agents: List['Agent']) -> None:
        """Propagate and synchronize ethical norms among agents."""
        if not isinstance(agents, list) or not agents:
            logger.error("Invalid agents: must be a non-empty list")
            raise ValueError("agents must be a non-empty list")
        
        try:
            common_norms = set()
            for agent in agents:
                agent_norms = getattr(agent, 'ethical_rules', set())
                if not isinstance(agent_norms, (set, list)):
                    logger.warning("Invalid ethical_rules for agent %s", getattr(agent, 'id', 'unknown'))
                    continue
                common_norms = common_norms.union(agent_norms) if common_norms else set(agent_norms)
            self.ethical_rules = list(common_norms)
            logger.info("Norms synchronized among %d agents", len(agents))
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Norm_Synchronization_{datetime.now().isoformat()}",
                    output={"norms": self.ethical_rules, "agents": [getattr(agent, 'id', 'unknown') for agent in agents]},
                    layer="Ethics",
                    intent="norm_synchronization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Norms synchronized",
                    meta={"norms": self.ethical_rules},
                    module="SimulationCore",
                    tags=["norms", "synchronization"]
                )
        except Exception as e:
            logger.error("Norm synchronization failed: %s", str(e))
            raise

    async def propagate_constitution(self, constitution: Dict[str, Any]) -> None:
        """Seed and propagate constitutional parameters in agent ecosystem."""
        if not isinstance(constitution, dict):
            logger.error("Invalid constitution: must be a dictionary")
            raise TypeError("constitution must be a dictionary")
        
        try:
            self.constitution = constitution
            logger.info("Constitution propagated")
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Constitution_Propagation_{datetime.now().isoformat()}",
                    output=constitution,
                    layer="Constitutions",
                    intent="constitution_propagation"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Constitution propagated",
                    meta=constitution,
                    module="SimulationCore",
                    tags=["constitution", "propagation"]
                )
        except Exception as e:
            logger.error("Constitution propagation failed: %s", str(e))
            raise

def M_b_exponential(r_kpc: np.ndarray, M0: float = 5e10, r_scale: float = 3.5) -> np.ndarray:
    """Compute exponential baryonic mass profile."""
    return M0 * np.exp(-r_kpc / r_scale)

def v_obs_flat(r_kpc: np.ndarray, v0: float = 180) -> np.ndarray:
    """Compute flat observed velocity profile."""
    return np.full_like(r_kpc, v0)

if __name__ == "__main__":
    async def main():
        simulation_core = SimulationCore()
        r_vals = np.linspace(0.1, 20, 100)
        await simulation_core.plot_AGRF_simulation(r_vals, M_b_exponential, v_obs_flat)

    import asyncio
    asyncio.run(main())
