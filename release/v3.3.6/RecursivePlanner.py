"""
ANGELA Cognitive System Module: RecursivePlanner
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a RecursivePlanner class for recursive goal planning in the ANGELA v3.5 architecture.
"""

import logging
import random
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple, Protocol
from datetime import datetime
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from toca_simulation import run_AGRF_with_traits
from modules import (
    reasoning_engine as reasoning_engine_module,
    meta_cognition as meta_cognition_module,
    alignment_guard as alignment_guard_module,
    simulation_core as simulation_core_module,
    memory_manager as memory_manager_module,
    multi_modal_fusion as multi_modal_fusion_module,
    error_recovery as error_recovery_module
)

logger = logging.getLogger("ANGELA.RecursivePlanner")

class AgentProtocol(Protocol):
    name: str

    def process_subgoal(self, subgoal: str) -> Any:
        ...

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    """Calculate concentration trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    """Calculate self-awareness trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.7), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    """Calculate morality trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.9), 1.0))

@lru_cache(maxsize=100)
def eta_reflexivity(t: float) -> float:
    """Calculate reflexivity trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.1), 1.0))

@lru_cache(maxsize=100)
def lambda_narrative(t: float) -> float:
    """Calculate narrative trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.3), 1.0))

@lru_cache(maxsize=100)
def delta_moral_drift(t: float) -> float:
    """Calculate moral drift trait value."""
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    """Calculate coherence trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

class RecursivePlanner:
    """A class for recursive goal planning in the ANGELA v3.5 architecture.

    Supports trait-weighted subgoal decomposition, agent collaboration, ToCA physics
    simulations, and meta-cognitive alignment checks with concurrent execution.

    Attributes:
        reasoning_engine (ReasoningEngine): Engine for goal decomposition.
        meta_cognition (MetaCognition): Module for reasoning review and goal rewriting.
        alignment_guard (AlignmentGuard): Guard for ethical checks.
        simulation_core (SimulationCore): Core for running simulations.
        memory_manager (MemoryManager): Manager for storing events and traces.
        multi_modal_fusion (MultiModalFusion): Module for multi-modal synthesis.
        error_recovery (ErrorRecovery): Module for error handling and recovery.
        agi_enhancer (AGIEnhancer): Enhancer for logging and auditing.
        max_workers (int): Maximum number of concurrent workers for subgoal processing.
        omega (Dict[str, Any]): Global narrative state with timeline, traits, and symbolic log.
        omega_lock (Lock): Thread-safe lock for omega updates.
    """
    def __init__(self, max_workers: int = 4,
                 reasoning_engine: Optional['reasoning_engine_module.ReasoningEngine'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 simulation_core: Optional['simulation_core_module.SimulationCore'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 agi_enhancer: Optional['AGIEnhancer'] = None):
        self.reasoning_engine = reasoning_engine or reasoning_engine_module.ReasoningEngine(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=meta_cognition,
            error_recovery=error_recovery)
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(agi_enhancer=agi_enhancer)
        self.alignment_guard = alignment_guard or alignment_guard_module.AlignmentGuard()
        self.simulation_core = simulation_core or simulation_core_module.SimulationCore()
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=meta_cognition,
            error_recovery=error_recovery)
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.agi_enhancer = agi_enhancer
        self.max_workers = max(1, min(max_workers, 8))
        self.omega = {"timeline": [], "traits": {}, "symbolic_log": []}
        self.omega_lock = Lock()
        logger.info("RecursivePlanner initialized")

    def adjust_plan_depth(self, trait_weights: Dict[str, float]) -> int:
        """Adjust planning depth based on trait weights."""
        omega = trait_weights.get("omega", 0.0)
        if not isinstance(omega, (int, float)):
            logger.error("Invalid omega: must be a number")
            raise ValueError("omega must be a number")
        if omega > 0.7:
            logger.info("Expanding recursion depth due to high omega: %.2f", omega)
            return 2
        return 1

    async def plan(self, goal: str, context: Optional[Dict[str, Any]] = None,
                   depth: int = 0, max_depth: int = 5,
                   collaborating_agents: Optional[List['AgentProtocol']] = None) -> List[str]:
        """Recursively decompose and plan a goal with trait-based depth adjustment."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer")
            raise ValueError("depth must be a non-negative integer")
        if not isinstance(max_depth, int) or max_depth < 1:
            logger.error("Invalid max_depth: must be a positive integer")
            raise ValueError("max_depth must be a positive integer")
        if collaborating_agents is not None and not isinstance(collaborating_agents, list):
            logger.error("Invalid collaborating_agents: must be a list")
            raise TypeError("collaborating_agents must be a list")
        
        logger.info("Planning for goal: '%s'", goal)
        try:
            if not self.alignment_guard.is_goal_safe(goal):
                logger.error("Goal '%s' violates alignment constraints", goal)
                raise ValueError("Unsafe goal detected")
            
            t = time.time() % 1.0
            concentration = beta_concentration(t)
            awareness = omega_selfawareness(t)
            moral_weight = mu_morality(t)
            reflexivity = eta_reflexivity(t)
            narrative = lambda_narrative(t)
            moral_drift = delta_moral_drift(t)
            
            with self.omega_lock:
                self.omega["traits"].update({
                    "beta": concentration, "omega": awareness, "mu": moral_weight,
                    "eta": reflexivity, "lambda": narrative, "delta": moral_drift,
                    "phi": phi_scalar(t)
                })
            
            trait_mod = concentration * 0.4 + reflexivity * 0.2 + narrative * 0.2 - moral_drift * 0.2
            dynamic_depth_limit = max_depth + int(trait_mod * 10) + self.adjust_plan_depth(self.omega["traits"])
            
            if depth > dynamic_depth_limit:
                logger.warning("Trait-based dynamic max recursion depth reached: depth=%d, limit=%d", depth, dynamic_depth_limit)
                return [goal]
            
            subgoals = await self.reasoning_engine.decompose(goal, context, prioritize=True)
            if not subgoals:
                logger.info("No subgoals found. Returning atomic goal: '%s'", goal)
                return [goal]
            
            if collaborating_agents:
                logger.info("Collaborating with agents: %s", [agent.name for agent in collaborating_agents])
                subgoals = await self._distribute_subgoals(subgoals, collaborating_agents)
            
            validated_plan = []
            tasks = [self._plan_subgoal(subgoal, context, depth + 1, dynamic_depth_limit) for subgoal in subgoals]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for subgoal, result in zip(subgoals, results):
                if isinstance(result, Exception):
                    logger.error("Error planning subgoal '%s': %s", subgoal, str(result))
                    recovery_plan = await self.meta_cognition.review_reasoning(str(result))
                    validated_plan.extend(recovery_plan)
                    await self._update_omega(subgoal, recovery_plan, error=True)
                else:
                    validated_plan.extend(result)
                    await self._update_omega(subgoal, result)
            
            logger.info("Final validated plan for goal '%s': %s", goal, validated_plan)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Plan_{goal[:50]}_{datetime.now().isoformat()}",
                    output=str(validated_plan),
                    layer="Plans",
                    intent="goal_planning"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Plan generated",
                    meta={"goal": goal, "plan": validated_plan},
                    module="RecursivePlanner",
                    tags=["planning", "recursive"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "plan", "plan": validated_plan})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"goal": goal, "plan": validated_plan, "context": context or {}},
                    summary_style="insightful"
                )
                logger.info("Plan synthesis: %s", synthesis)
            return validated_plan
        except Exception as e:
            logger.error("Planning failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan(goal, context, depth, max_depth, collaborating_agents), default=[goal]
            )

    async def _update_omega(self, subgoal: str, result: List[str], error: bool = False) -> None:
        """Update the global narrative state with subgoal results."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        if not isinstance(result, list):
            logger.error("Invalid result: must be a list")
            raise TypeError("result must be a list")
        
        event = {
            "subgoal": subgoal,
            "result": result,
            "timestamp": time.time(),
            "error": error
        }
        symbolic_tag = await self.meta_cognition.extract_symbolic_signature(subgoal) if self.meta_cognition else "unknown"
        with self.omega_lock:
            self.omega["timeline"].append(event)
            self.omega["symbolic_log"].append(symbolic_tag)
            if len(self.omega["timeline"]) > 1000:
                self.omega["timeline"] = self.omega["timeline"][-500:]
                self.omega["symbolic_log"] = self.omega["symbolic_log"][-500:]
                logger.info("Trimmed omega state to maintain size limit")
        if self.memory_manager:
            await self.memory_manager.store_symbolic_event(event, symbolic_tag)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Omega state updated",
                meta=event,
                module="RecursivePlanner",
                tags=["omega", "update"]
            )

    async def plan_from_intrinsic_goal(self, generated_goal: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Plan from an intrinsic goal."""
        if not isinstance(generated_goal, str) or not generated_goal.strip():
            logger.error("Invalid generated_goal: must be a non-empty string")
            raise ValueError("generated_goal must be a non-empty string")
        
        logger.info("Initiating plan from intrinsic goal: '%s'", generated_goal)
        try:
            if self.meta_cognition:
                validated_goal = await self.meta_cognition.rewrite_goal(generated_goal)
            else:
                validated_goal = generated_goal
            plan = await self.plan(validated_goal, context)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Intrinsic_Plan_{validated_goal[:50]}_{datetime.now().isoformat()}",
                    output=str(plan),
                    layer="IntrinsicPlans",
                    intent="intrinsic_goal_planning"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Intrinsic goal plan generated",
                    meta={"goal": validated_goal, "plan": plan},
                    module="RecursivePlanner",
                    tags=["intrinsic", "planning"]
                )
            return plan
        except Exception as e:
            logger.error("Intrinsic goal planning failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_from_intrinsic_goal(generated_goal, context), default=[]
            )

    async def _plan_subgoal(self, subgoal: str, context: Optional[Dict[str, Any]],
                            depth: int, max_depth: int) -> List[str]:
        """Plan a single subgoal with simulation and alignment checks."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        
        logger.info("Evaluating subgoal: '%s'", subgoal)
        try:
            if not self.alignment_guard.is_goal_safe(subgoal):
                logger.warning("Subgoal '%s' failed alignment check", subgoal)
                return []
            
            if "gravity" in subgoal.lower() or "scalar" in subgoal.lower():
                sim_traits = run_AGRF_with_traits(context or {})
                with self.omega_lock:
                    self.omega["traits"].update(sim_traits.get("fields", {}))
                    self.omega["timeline"].append({
                        "subgoal": subgoal,
                        "traits": sim_traits.get("fields", {}),
                        "timestamp": time.time()
                    })
                if self.multi_modal_fusion:
                    synthesis = await self.multi_modal_fusion.analyze(
                        data={"subgoal": subgoal, "simulation_traits": sim_traits},
                        summary_style="concise"
                    )
                    logger.info("Simulation synthesis: %s", synthesis)
            
            simulation_feedback = await self.simulation_core.run(subgoal, context=context, scenarios=2, agents=1)
            approved, _ = await self.meta_cognition.pre_action_alignment_check(subgoal)
            if not approved:
                logger.warning("Subgoal '%s' denied by meta-cognitive alignment check", subgoal)
                return []
            
            sub_plan = await self.plan(subgoal, context, depth + 1, max_depth)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Subgoal_Plan_{subgoal[:50]}_{datetime.now().isoformat()}",
                    output=str(sub_plan),
                    layer="SubgoalPlans",
                    intent="subgoal_planning"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Subgoal plan generated",
                    meta={"subgoal": subgoal, "sub_plan": sub_plan},
                    module="RecursivePlanner",
                    tags=["subgoal", "planning"]
                )
            return sub_plan
        except Exception as e:
            logger.error("Subgoal '%s' planning failed: %s", subgoal, str(e))
            return []

    async def _distribute_subgoals(self, subgoals: List[str], agents: List['AgentProtocol']) -> List[str]:
        """Distribute subgoals among collaborating agents with conflict resolution."""
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list")
            raise TypeError("subgoals must be a list")
        if not isinstance(agents, list) or not agents:
            logger.error("Invalid agents: must be a non-empty list")
            raise ValueError("agents must be a non-empty list")
        
        logger.info("Distributing subgoals among agents")
        distributed = []
        for i, subgoal in enumerate(subgoals):
            agent = agents[i % len(agents)]
            logger.info("Assigning subgoal '%s' to agent '%s'", subgoal, agent.name)
            if await self._resolve_conflicts(subgoal, agent):
                distributed.append(subgoal)
            else:
                logger.warning("Conflict detected for subgoal '%s'. Skipping assignment", subgoal)
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Subgoal_Distribution_{datetime.now().isoformat()}",
                output=str(distributed),
                layer="Distributions",
                intent="subgoal_distribution"
            )
        return distributed

    async def _resolve_conflicts(self, subgoal: str, agent: 'AgentProtocol') -> bool:
        """Resolve conflicts for subgoal assignment to an agent."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        if not hasattr(agent, 'name') or not hasattr(agent, 'process_subgoal'):
            logger.error("Invalid agent: must have name and process_subgoal attributes")
            raise ValueError("agent must have name and process_subgoal attributes")
        
        logger.info("Resolving conflicts for subgoal '%s' and agent '%s'", subgoal, agent.name)
        try:
            if self.meta_cognition:
                alignment = await self.meta_cognition.pre_action_alignment_check(subgoal)
                if not alignment[0]:
                    logger.warning("Subgoal '%s' failed meta-cognitive alignment for agent '%s'", subgoal, agent.name)
                    return False
            capability_check = agent.process_subgoal(subgoal)
            if isinstance(capability_check, (int, float)) and capability_check < 0.5:
                logger.warning("Agent '%s' lacks capability for subgoal '%s' (score: %.2f)", agent.name, subgoal, capability_check)
                return False
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Conflict_Resolution_{subgoal[:50]}_{agent.name}_{datetime.now().isoformat()}",
                    output=f"Resolved: {subgoal} assigned to {agent.name}",
                    layer="ConflictResolutions",
                    intent="conflict_resolution"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Conflict resolved",
                    meta={"subgoal": subgoal, "agent": agent.name},
                    module="RecursivePlanner",
                    tags=["conflict", "resolution"]
                )
            return True
        except Exception as e:
            logger.error("Conflict resolution failed: %s", str(e))
            return False

    async def plan_with_trait_loop(self, initial_goal: str, context: Optional[Dict[str, Any]] = None,
                                   iterations: int = 3) -> List[Tuple[str, List[str]]]:
        """Iteratively plan with trait-based goal rewriting."""
        if not isinstance(initial_goal, str) or not initial_goal.strip():
            logger.error("Invalid initial_goal: must be a non-empty string")
            raise ValueError("initial_goal must be a non-empty string")
        if not isinstance(iterations, int) or iterations < 1:
            logger.error("Invalid iterations: must be a positive integer")
            raise ValueError("iterations must be a positive integer")
        
        current_goal = initial_goal
        all_plans = []
        previous_goals = set()
        try:
            for i in range(iterations):
                if current_goal in previous_goals:
                    logger.info("Goal convergence detected: '%s'", current_goal)
                    break
                previous_goals.add(current_goal)
                logger.info("Loop iteration %d: Planning goal '%s'", i + 1, current_goal)
                plan = await self.plan(current_goal, context)
                all_plans.append((current_goal, plan))
                
                with self.omega_lock:
                    traits = self.omega.get("traits", {})
                phi = traits.get("phi", phi_scalar(time.time() % 1.0))
                psi = traits.get("psi_foresight", 0.5)
                if phi > 0.7 or psi > 0.6:
                    current_goal = f"Expand on {current_goal} using scalar field insights"
                elif traits.get("beta", 1.0) < 0.3:
                    logger.info("Convergence detected: beta conflict low")
                    break
                else:
                    current_goal = await self.meta_cognition.rewrite_goal(current_goal)
                
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Trait_Loop_{current_goal[:50]}_{datetime.now().isoformat()}",
                        output=str((current_goal, plan)),
                        layer="TraitLoopPlans",
                        intent="trait_loop_planning"
                    )
            
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait loop planning completed",
                    meta={"initial_goal": initial_goal, "all_plans": all_plans},
                    module="RecursivePlanner",
                    tags=["trait_loop", "planning"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "plan_with_trait_loop", "all_plans": all_plans})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"initial_goal": initial_goal, "all_plans": all_plans},
                    summary_style="insightful"
                )
                logger.info("Trait loop synthesis: %s", synthesis)
            return all_plans
        except Exception as e:
            logger.error("Trait loop planning failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_with_trait_loop(initial_goal, context, iterations), default=[]
            )

    async def plan_with_traits(self, goal: str, context: Dict[str, Any], traits: Dict[str, float]) -> Dict[str, Any]:
        """Generate a plan with trait-adjusted depth and bias."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary")
            raise TypeError("traits must be a dictionary")
        
        try:
            depth = int(3 + traits.get("phi", 0.5) * 4 - traits.get("eta", 0.5) * 2)
            depth = max(1, min(depth, 7))
            plan = [f"Step {i+1}: process {goal}" for i in range(depth)]
            bias = "cautious" if traits.get("omega", 0.0) > 0.6 else "direct"
            result = {
                "plan": plan,
                "planning_depth": depth,
                "bias": bias,
                "traits_applied": traits,
                "timestamp": datetime.now().isoformat()
            }
            if self.meta_cognition:
                review = await self.meta_cognition.review_reasoning(str(result))
                result["review"] = review
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Plan_With_Traits_{goal[:50]}_{result['timestamp']}",
                    output=str(result),
                    layer="Plans",
                    intent="trait_based_planning"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Plan with traits generated",
                    meta=result,
                    module="RecursivePlanner",
                    tags=["planning", "traits"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "plan_with_traits", "result": result})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"goal": goal, "plan": result, "context": context},
                    summary_style="concise"
                )
                result["synthesis"] = synthesis
            return result
        except Exception as e:
            logger.error("Plan with traits failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_with_traits(goal, context, traits)
            )
