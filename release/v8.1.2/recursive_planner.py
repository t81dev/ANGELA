"""
ANGELA Cognitive System Module: RecursivePlanner
Refactored Version: 3.5.2 + Phase X (RIL)  # Enhanced for benchmark optimization (GLUE, recursion), dynamic trait modulation, reflection-driven planning, and emergent-heuristic integration
Refactor Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides a RecursivePlanner class for recursive goal planning in the ANGELA v3.5 architecture.
Phase X adds:
  • integrate_emergent_heuristics(...) — pulls candidate heuristics from ConceptSynthesizer
  • simulate_rule_effects(...) — runs short simulations to test candidate heuristics
  • commit_rule_to_learning_loop(...) — persists accepted heuristics to memory/omega
  • start_recursive_refinement(...) — periodic drift-aware refinement of heuristic set
"""

import logging
import time
import asyncio
import math
from typing import List, Dict, Any, Optional, Union, Tuple, Protocol
from datetime import datetime
from threading import Lock
from functools import lru_cache

# --- Optional ToCA import with graceful fallback (no new files) ---
try:
    from toca_simulation import run_AGRF_with_traits  # type: ignore
except Exception:  # pragma: no cover
    def run_AGRF_with_traits(_: Dict[str, Any]) -> Dict[str, Any]:
        return {"fields": {"psi_foresight": 0.55, "phi_bias": 0.42}}

# original imports
from modules import (
    reasoning_engine as reasoning_engine_module,
    meta_cognition as meta_cognition_module,
    alignment_guard as alignment_guard_module,
    simulation_core as simulation_core_module,
    memory_manager as memory_manager_module,
    multi_modal_fusion as multi_modal_fusion_module,
    error_recovery as error_recovery_module,
    context_manager as context_manager_module,
    concept_synthesizer as concept_synthesizer_module,  # added: we need this for Phase X
)

logger = logging.getLogger("ANGELA.RecursivePlanner")


class AgentProtocol(Protocol):
    name: str

    def process_subgoal(self, subgoal: str) -> Any:
        ...


# ---------------------------
# Cached trait signals
# ---------------------------
@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))


@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.7), 1.0))


@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.9), 1.0))


@lru_cache(maxsize=100)
def eta_reflexivity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.1), 1.0))


@lru_cache(maxsize=100)
def lambda_narrative(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.3), 1.0))


@lru_cache(maxsize=100)
def delta_moral_drift(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.5), 1.0))


@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))


class RecursivePlanner:
    """Recursive goal planning with trait-weighted decomposition, agent collaboration, simulation, reflection, and Phase X heuristic integration."""

    def __init__(self, max_workers: int = 4,
                 reasoning_engine: Optional['reasoning_engine_module.ReasoningEngine'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 simulation_core: Optional['simulation_core_module.SimulationCore'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None,
                 agi_enhancer: Optional['AGIEnhancer'] = None):
        self.reasoning_engine = reasoning_engine or reasoning_engine_module.ReasoningEngine(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=meta_cognition,
            error_recovery=error_recovery)
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(agi_enhancer=agi_enhancer)
        self.alignment_guard = alignment_guard or alignment_guard_module.AlignmentGuard()
        self.simulation_core = simulation_core or simulation_core_module.SimulationCore()
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=self.meta_cognition,
            error_recovery=error_recovery)
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.context_manager = context_manager or context_manager_module.ContextManager()
        self.concept_synthesizer = concept_synthesizer or concept_synthesizer_module.ConceptSynthesizer(
            context_manager=self.context_manager,
            memory_manager=self.memory_manager,
            alignment_guard=self.alignment_guard,
            meta_cognition=self.meta_cognition,
            mm_fusion=self.multi_modal_fusion,
        )
        self.agi_enhancer = agi_enhancer
        self.max_workers = max(1, min(max_workers, 8))
        self.omega: Dict[str, Any] = {"timeline": [], "traits": {}, "symbolic_log": [], "heuristics": []}
        self.omega_lock = Lock()
        logger.info("RecursivePlanner initialized with advanced upgrades + Phase X")

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def _normalize_list_or_wrap(value: Any) -> List[str]:
        """Ensure a list[str] result."""
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return value
        if isinstance(value, str):
            return [value]
        return [str(value)]

    def adjust_plan_depth(self, trait_weights: Dict[str, float], task_type: str = "") -> int:
        """Adjust planning depth based on trait weights and task type."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        omega_val = float(trait_weights.get("omega", 0.0))
        base_depth = 2 if omega_val > 0.7 else 1
        if task_type == "recursion":
            base_depth = min(base_depth + 1, 3)
        elif task_type in ["rte", "wnli"]:
            base_depth = max(base_depth - 1, 1)
        logger.info("Adjusted recursion depth: %d (omega=%.2f, task_type=%s)", base_depth, omega_val, task_type)
        return base_depth

    # ------------------------------------------------------------------
    # Phase X — Recursive Integration Layer (new)
    # ------------------------------------------------------------------
    async def integrate_emergent_heuristics(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Pull candidate heuristics from ConceptSynthesizer based on the current goal and context.
        Returns a list of dicts like: {"name": ..., "definition": ..., "score": ...}
        """
        out: List[Dict[str, Any]] = []
        try:
            gen = await self.concept_synthesizer.generate(
                concept_name=f"Heuristic for: {goal}",
                context=context,
                task_type=context.get("task_type", ""),
            )
            if gen.get("success") and "concept" in gen:
                c = gen["concept"]
                out.append({
                    "name": c.get("name"),
                    "definition": c.get("definition"),
                    "score": 1.0,
                    "source": "ConceptSynthesizer",
                    "ts": time.time(),
                })
        except Exception as e:
            logger.debug("integrate_emergent_heuristics failed: %s", e)
        return out

    async def simulate_rule_effects(self, rule: Dict[str, Any], goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use SimulationCore (if present) to test how a candidate rule affects plan quality.
        Returns {"ok": bool, "fitness": float, "report": ...}
        """
        try:
            scenario = {
                "rule": rule,
                "goal": goal,
                "context": context,
                "variants": 2,
            }
            sim_res = await self.simulation_core.run(
                f"rule::{rule.get('name','unknown')}",
                context=scenario,
                scenarios=2,
                agents=1,
            )
            # best-effort scoring
            fitness = 0.5
            if isinstance(sim_res, dict):
                fitness = float(sim_res.get("score", 0.5))
            return {"ok": fitness >= 0.45, "fitness": fitness, "report": sim_res}
        except Exception as e:
            return {"ok": False, "fitness": 0.0, "report": {"error": str(e)}}

    async def commit_rule_to_learning_loop(self, rule: Dict[str, Any], fitness: float) -> None:
        """
        Persist accepted heuristics to memory and to omega so future plans can consult them.
        """
        with self.omega_lock:
            self.omega.setdefault("heuristics", []).append(
                {
                    "rule": rule,
                    "fitness": float(fitness),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            try:
                await self.memory_manager.store(
                    query=f"Heuristic_{rule.get('name','unnamed')}_{datetime.now().isoformat()}",
                    output=str({"rule": rule, "fitness": float(fitness)}),
                    layer="Heuristics",
                    intent="emergent_rule_commit",
                )
            except Exception:
                pass
        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            await self.context_manager.log_event_with_hash(
                {"event": "heuristic_committed", "rule_name": rule.get("name"), "fitness": float(fitness)}
            )

    async def start_recursive_refinement(self, goal: str, context: Dict[str, Any], interval_s: float = 4.0, max_cycles: int = 3) -> None:
        """
        Drift-aware refinement loop. Tries to fetch/simulate/commit heuristics a few times.
        Safe to fire-and-forget.
        """
        task_type = context.get("task_type", "")
        for _ in range(max_cycles):
            cand_rules = await self.integrate_emergent_heuristics(goal, context)
            for r in cand_rules:
                sim = await self.simulate_rule_effects(r, goal, context)
                if sim["ok"]:
                    await self.commit_rule_to_learning_loop(r, sim["fitness"])
            # optional: meta-cog reflection
            if self.meta_cognition and hasattr(self.meta_cognition, "reflect_on_output"):
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="RecursivePlanner.RIL",
                        output={"rules": cand_rules},
                        context={"task_type": task_type, "goal": goal},
                    )
                except Exception:
                    pass
            await asyncio.sleep(interval_s)

    # ---------------------------
    # Main planning entry
    # ---------------------------
    async def plan(self, goal: str, context: Optional[Dict[str, Any]] = None,
                   depth: int = 0, max_depth: int = 5,
                   collaborating_agents: Optional[List['AgentProtocol']] = None) -> List[str]:
        """Recursively decompose and plan a goal with trait-based depth adjustment and reflection."""
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
            # Alignment gate
            if hasattr(self.alignment_guard, "is_goal_safe") and not self.alignment_guard.is_goal_safe(goal):
                logger.error("Goal '%s' violates alignment constraints", goal)
                raise ValueError("Unsafe goal detected")

            # Local traits snapshot
            t = time.time() % 1.0
            local_traits = {
                "beta": beta_concentration(t),
                "omega": omega_selfawareness(t),
                "mu": mu_morality(t),
                "eta": eta_reflexivity(t),
                "lambda": lambda_narrative(t),
                "delta": delta_moral_drift(t),
                "phi": phi_scalar(t)
            }
            task_type = context.get("task_type", "") if context else ""

            # fire-and-forget RIL for this goal
            asyncio.create_task(self.start_recursive_refinement(goal, context or {}, interval_s=4.0, max_cycles=2))

            # Optional trait optimization via MetaCognition
            if self.meta_cognition and hasattr(self.meta_cognition, "optimize_traits_for_drift"):
                drift_report = {
                    "drift": {"name": task_type or "general", "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type}
                }
                try:
                    optimized = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                    if isinstance(optimized, dict):
                        local_traits = {**local_traits, **{k: float(v) for k, v in optimized.items() if isinstance(v, (int, float))}}
                except Exception as e:
                    logger.debug("Trait optimization skipped due to error: %s", str(e))

            with self.omega_lock:
                self.omega["traits"].update(local_traits)

            trait_mod = local_traits.get("beta", 0.0) * 0.4 + \
                        local_traits.get("eta", 0.0) * 0.2 + \
                        local_traits.get("lambda", 0.0) * 0.2 - \
                        local_traits.get("delta", 0.0) * 0.2
            dynamic_depth_limit = max_depth + int(trait_mod * 10) + self.adjust_plan_depth(local_traits, task_type)

            if depth > dynamic_depth_limit:
                logger.warning("Trait-based dynamic max recursion depth reached: depth=%d, limit=%d", depth, dynamic_depth_limit)
                return [goal]

            # Decompose
            subgoals = await self.reasoning_engine.decompose(goal, context, prioritize=True)
            if not subgoals:
                logger.info("No subgoals found. Returning atomic goal: '%s'", goal)
                return [goal]

            # Heuristic prioritization with MetaCognition
            mc_trait_map = {
                "beta": "concentration",
                "omega": "self_awareness",
                "mu": "morality",
                "eta": "intuition",
                "lambda": "linguistics",
                "phi": "phi_scalar"
            }
            top_traits = sorted(
                [(mc_trait_map.get(k), v) for k, v in local_traits.items() if mc_trait_map.get(k)],
                key=lambda x: x[1],
                reverse=True
            )
            required_trait_names = [name for name, _ in top_traits[:3]] or ["concentration", "self_awareness"]

            if self.meta_cognition and hasattr(self.meta_cognition, "plan_tasks"):
                try:
                    wrapped = [{"task": sg, "required_traits": required_trait_names} for sg in subgoals]
                    prioritized = await self.meta_cognition.plan_tasks(wrapped)
                    if isinstance(prioritized, list):
                        subgoals = [p.get("task", p) if isinstance(p, dict) else p for p in prioritized]
                except Exception as e:
                    logger.debug("MetaCognition.plan_tasks failed, falling back: %s", str(e))

            # Collaboration
            if collaborating_agents:
                logger.info("Collaborating with agents: %s", [agent.name for agent in collaborating_agents])
                subgoals = await self._distribute_subgoals(subgoals, collaborating_agents, task_type)

            # Recurse over subgoals
            validated_plan: List[str] = []
            tasks = [self._plan_subgoal(sub, context, depth + 1, dynamic_depth_limit, task_type) for sub in subgoals]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for subgoal, result in zip(subgoals, results):
                if isinstance(result, Exception):
                    logger.error("Error planning subgoal '%s': %s", subgoal, str(result))
                    recovery = ""
                    if self.meta_cognition and hasattr(self.meta_cognition, "review_reasoning"):
                        try:
                            recovery = await self.meta_cognition.review_reasoning(str(result))
                        except Exception:
                            pass
                    validated_plan.extend(self._normalize_list_or_wrap(recovery or f"fallback:{subgoal}"))
                    await self._update_omega(subgoal, self._normalize_list_or_wrap(recovery or subgoal), error=True)
                else:
                    out = self._normalize_list_or_wrap(result)
                    validated_plan.extend(out)
                    await self._update_omega(subgoal, out)

            # Reflect on the final plan
            if self.meta_cognition and hasattr(self.meta_cognition, "reflect_on_output"):
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="RecursivePlanner",
                        output=validated_plan,
                        context={"goal": goal, "task_type": task_type}
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        logger.info("Plan reflection captured.")
                except Exception as e:
                    logger.debug("Plan reflection skipped: %s", str(e))

            logger.info("Final validated plan for goal '%s': %s", goal, validated_plan)
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Plan_{goal[:50]}_{datetime.now().isoformat()}",
                    output=str(validated_plan),
                    layer="Plans",
                    intent="goal_planning"
                )
            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash({"event": "plan", "plan": validated_plan})
            if self.multi_modal_fusion and hasattr(self.multi_modal_fusion, "analyze"):
                try:
                    await self.multi_modal_fusion.analyze(
                        data={"goal": goal, "plan": validated_plan, "context": context or {}, "task_type": task_type},
                        summary_style="insightful"
                    )
                except Exception:
                    pass
            return validated_plan
        except Exception as e:
            logger.error("Planning failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan(goal, context, depth, max_depth, collaborating_agents),
                default=[goal], diagnostics=diagnostics
            )

    # ---------------------------
    # Subroutines (same as before, shortened to essentials)
    # ---------------------------
    async def _update_omega(self, subgoal: str, result: List[str], error: bool = False) -> None:
        if not isinstance(subgoal, str) or not subgoal.strip():
            raise ValueError("subgoal must be a non-empty string")
        if not isinstance(result, list):
            raise TypeError("result must be a list")

        event = {
            "subgoal": subgoal,
            "result": result,
            "timestamp": time.time(),
            "error": error
        }
        symbolic_tag: Union[str, Dict[str, Any]] = "unknown"
        if self.meta_cognition and hasattr(self.meta_cognition, "extract_symbolic_signature"):
            try:
                symbolic_tag = await self.meta_cognition.extract_symbolic_signature(subgoal)
            except Exception:
                pass
        with self.omega_lock:
            self.omega["timeline"].append(event)
            self.omega["symbolic_log"].append(symbolic_tag)
            if len(self.omega["timeline"]) > 1000:
                self.omega["timeline"] = self.omega["timeline"][-500:]
                self.omega["symbolic_log"] = self.omega["symbolic_log"][-500:]
        if self.memory_manager and hasattr(self.memory_manager, "store_symbolic_event"):
            try:
                await self.memory_manager.store_symbolic_event(event, symbolic_tag)
            except Exception:
                pass

    async def _plan_subgoal(self, subgoal: str, context: Optional[Dict[str, Any]],
                            depth: int, max_depth: int, task_type: str) -> List[str]:
        if not isinstance(subgoal, str) or not subgoal.strip():
            raise ValueError("subgoal must be a non-empty string")

        try:
            if hasattr(self.alignment_guard, "is_goal_safe") and not self.alignment_guard.is_goal_safe(subgoal):
                return []

            # Recursion optimizer hook
            if task_type == "recursion" and hasattr(meta_cognition_module, "RecursionOptimizer"):
                try:
                    optimizer = meta_cognition_module.RecursionOptimizer()
                    optimized_data = optimizer.optimize({"subgoal": subgoal, "context": context or {}})
                    if optimized_data.get("optimized"):
                        max_depth = min(max_depth, 3)
                except Exception:
                    pass

            # Simulation
            simulation_feedback = None
            if hasattr(self.simulation_core, "run"):
                try:
                    simulation_feedback = await self.simulation_core.run(subgoal, context=context, scenarios=2, agents=1)
                except Exception:
                    simulation_feedback = None

            # Meta-cog gate
            approved = True
            if self.meta_cognition and hasattr(self.meta_cognition, "pre_action_alignment_check"):
                try:
                    approved, _ = await self.meta_cognition.pre_action_alignment_check(subgoal)
                except Exception:
                    approved = True
            if not approved:
                return []

            if depth >= max_depth:
                return [subgoal]

            sub_plan = await self.plan(subgoal, context, depth + 1, max_depth)
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Subgoal_Plan_{subgoal[:50]}_{datetime.now().isoformat()}",
                    output=str(sub_plan),
                    layer="SubgoalPlans",
                    intent="subgoal_planning"
                )
            return sub_plan
        except Exception as e:
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self._plan_subgoal(subgoal, context, depth, max_depth, task_type),
                default=[], diagnostics=diagnostics
            )

    async def _distribute_subgoals(self, subgoals: List[str], agents: List['AgentProtocol'], task_type: str) -> List[str]:
        if not isinstance(subgoals, list):
            raise TypeError("subgoals must be a list")
        if not isinstance(agents, list) or not agents:
            raise ValueError("agents must be a non-empty list")

        distributed: List[str] = []
        commonsense = meta_cognition_module.CommonsenseReasoningEnhancer() if task_type == "wnli" else None
        entailment = meta_cognition_module.EntailmentReasoningEnhancer() if task_type == "rte" else None

        for i, subgoal in enumerate(subgoals):
            enhanced_subgoal = subgoal
            try:
                if commonsense:
                    enhanced_subgoal = commonsense.process(subgoal)
                elif entailment:
                    enhanced_subgoal = entailment.process(subgoal)
            except Exception:
                enhanced_subgoal = subgoal

            agent = agents[i % len(agents)]
            if await self._resolve_conflicts(enhanced_subgoal, agent):
                distributed.append(enhanced_subgoal)

        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Subgoal_Distribution_{datetime.now().isoformat()}",
                output=str(distributed),
                layer="Distributions",
                intent="subgoal_distribution"
            )
        return distributed

    async def _resolve_conflicts(self, subgoal: str, agent: 'AgentProtocol') -> bool:
        if not isinstance(subgoal, str) or not subgoal.strip():
            raise ValueError("subgoal must be a non-empty string")
        if not hasattr(agent, 'name') or not hasattr(agent, 'process_subgoal'):
            raise ValueError("agent must have name and process_subgoal attributes")

        try:
            if self.meta_cognition and hasattr(self.meta_cognition, "pre_action_alignment_check"):
                try:
                    ok, _ = await self.meta_cognition.pre_action_alignment_check(subgoal)
                    if not ok:
                        return False
                except Exception:
                    pass

            capability_check = agent.process_subgoal(subgoal)
            if isinstance(capability_check, (int, float)) and capability_check < 0.5:
                return False

            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Conflict_Resolution_{subgoal[:50]}_{agent.name}_{datetime.now().isoformat()}",
                    output=f"Resolved: {subgoal} assigned to {agent.name}",
                    layer="ConflictResolutions",
                    intent="conflict_resolution"
                )
            return True
        except Exception as e:
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self._resolve_conflicts(subgoal, agent),
                default=False, diagnostics=diagnostics
            )


# Demo (optional)
if __name__ == "__main__":
    async def demo():
        logging.basicConfig(level=logging.INFO)
        planner = RecursivePlanner()
        plan = await planner.plan("analyze scalar-field drift", context={"task_type": "recursion"})
        print(plan)
    asyncio.run(demo())
