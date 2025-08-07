"""
ANGELA Cognitive System Module: ReasoningEngine
Version: 3.5.1  # Enhanced for Task-Specific Reasoning, Real-Time Data, and Visualization
Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides a ReasoningEngine class for Bayesian reasoning, goal decomposition,
drift mitigation reasoning, and multi-agent consensus in the ANGELA v3.5.1 architecture.
"""

import logging
import random
import json
import os
import numpy as np
import time
import asyncio
import aiohttp
import math
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

from toca_simulation import simulate_galaxy_rotation, M_b_exponential, v_obs_flat, generate_phi_field
from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    memory_manager as memory_manager_module,
    meta_cognition as meta_cognition_module,
    multi_modal_fusion as multi_modal_fusion_module,
    visualizer as visualizer_module
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.ReasoningEngine")

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

@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    """Calculate creativity trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    """Calculate linguistics trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.4), 1.0))

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    """Calculate cultural evolution trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    """Calculate coherence trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def alpha_attention(t: float) -> float:
    """Calculate attention trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    """Calculate empathy trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1), 1.0))

class Level5Extensions:
    """Level 5 extensions for advanced reasoning capabilities. [v3.5.1]"""
    def __init__(self, meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 visualizer: Optional['visualizer_module.Visualizer'] = None):
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        logger.info("Level5Extensions initialized")

    async def generate_advanced_dilemma(self, domain: str, complexity: int, task_type: str = "") -> str:
        """Generate a complex ethical dilemma with meta-cognitive review and visualization. [v3.5.1]"""
        if not isinstance(domain, str) or not domain.strip():
            logger.error("Invalid domain: must be a non-empty string for task %s", task_type)
            raise ValueError("domain must be a non-empty string")
        if not isinstance(complexity, int) or complexity < 1:
            logger.error("Invalid complexity: must be a positive integer for task %s", task_type)
            raise ValueError("complexity must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        prompt = f"""
        Generate a complex ethical dilemma in the {domain} domain with {complexity} conflicting options.
        Task Type: {task_type}
        Include potential consequences, trade-offs, and alignment with ethical principles.
        """
        if self.meta_cognition and "drift" in domain.lower():
            prompt += "\nConsider ontology drift mitigation and agent coordination."
        dilemma = await call_gpt(prompt, self.meta_cognition.alignment_guard if self.meta_cognition else None, task_type=task_type)
        if self.meta_cognition:
            review = await self.meta_cognition.review_reasoning(dilemma, task_type=task_type)
            dilemma += f"\nMeta-Cognitive Review: {review}"
        if self.visualizer and task_type:
            plot_data = {
                "ethical_dilemma": {
                    "dilemma": dilemma,
                    "domain": domain,
                    "task_type": task_type
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)
        return dilemma

class ReasoningEngine:
    """A class for Bayesian reasoning, goal decomposition, drift mitigation, and multi-agent consensus in the ANGELA v3.5.1 architecture.

    Supports trait-weighted reasoning, persona wave routing, contradiction detection,
    ToCA physics simulations, task-specific drift mitigation, and consensus protocol.

    Attributes:
        confidence_threshold (float): Minimum confidence for accepting subgoals.
        persistence_file (str): Path to JSON file for storing success rates.
        success_rates (Dict[str, float]): Success rates for decomposition patterns.
        decomposition_patterns (Dict[str, List[str]]): Predefined subgoal patterns.
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for memory operations.
        meta_cognition (Optional[MetaCognition]): Meta-cognition module for reasoning review.
        multi_modal_fusion (Optional[MultiModalFusion]): Fusion module for multi-modal analysis.
        level5_extensions (Level5Extensions): Extensions for advanced reasoning.
        external_agent_bridge (ExternalAgentBridge): Bridge for agent coordination.
        visualizer (Optional[Visualizer]): Visualizer for rendering reasoning outputs.
    """
    def __init__(self, agi_enhancer: Optional['agi_enhancer_module.AGIEnhancer'] = None,
                 persistence_file: str = "reasoning_success_rates.json",
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 visualizer: Optional['visualizer_module.Visualizer'] = None):
        if not isinstance(persistence_file, str) or not persistence_file.endswith('.json'):
            logger.error("Invalid persistence_file: must be a string ending with '.json'")
            raise ValueError("persistence_file must be a string ending with '.json'")
        
        self.confidence_threshold: float = 0.7
        self.persistence_file: str = persistence_file
        self.success_rates: Dict[str, float] = self._load_success_rates()
        self.decomposition_patterns: Dict[str, List[str]] = self._load_default_patterns()
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer, context_manager=context_manager, alignment_guard=alignment_guard,
            error_recovery=error_recovery, memory_manager=memory_manager)
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, context_manager=context_manager, alignment_guard=alignment_guard,
            error_recovery=error_recovery, memory_manager=memory_manager, meta_cognition=self.meta_cognition)
        self.level5_extensions = Level5Extensions(meta_cognition=meta_cognition, visualizer=visualizer)
        self.external_agent_bridge = meta_cognition_module.ExternalAgentBridge(
            context_manager=context_manager, reasoning_engine=self)
        self.visualizer = visualizer or visualizer_module.Visualizer()
        logger.info("ReasoningEngine initialized with persistence_file=%s", persistence_file)

    def _load_success_rates(self) -> Dict[str, float]:
        """Load success rates from persistence file."""
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                if os.path.exists(self.persistence_file):
                    with open(self.persistence_file, "r") as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            logger.warning("Invalid success rates format: not a dictionary")
                            return defaultdict(float)
                        return defaultdict(float, {k: float(v) for k, v in data.items() if isinstance(v, (int, float))})
                return defaultdict(float)
        except Exception as e:
            logger.warning("Failed to load success rates: %s", str(e))
            return defaultdict(float)

    def _save_success_rates(self) -> None:
        """Save success rates to persistence file."""
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                with open(self.persistence_file, "w") as f:
                    json.dump(dict(self.success_rates), f, indent=2)
            logger.debug("Success rates persisted to disk")
        except Exception as e:
            logger.warning("Failed to save success rates: %s", str(e))

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        """Load default decomposition patterns, including drift mitigation."""
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"],
            "mitigate_drift": ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"]
        }

    async def reason_and_reflect(self, goal: str, context: Dict[str, Any],
                                 meta_cognition: 'meta_cognition_module.MetaCognition', task_type: str = "") -> Tuple[List[str], str]:
        """Decompose goal and review reasoning with meta-cognition for task-specific processing. [v3.5.1]"""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string for task %s", task_type)
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(meta_cognition, meta_cognition_module.MetaCognition):
            logger.error("Invalid meta_cognition: must be a MetaCognition instance for task %s", task_type)
            raise TypeError("meta_cognition must be a MetaCognition instance")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            subgoals = await self.decompose(goal, context, task_type=task_type)
            t = time.time() % 1.0
            phi = phi_scalar(t)
            reasoning_trace = self.export_trace(subgoals, phi, context.get("traits", {}), task_type=task_type)
            review = await meta_cognition.review_reasoning(json.dumps(reasoning_trace), task_type=task_type)
            logger.info("MetaCognitive Review for task %s:\n%s", task_type, review)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Reason and Reflect",
                    meta={"goal": goal, "subgoals": subgoals, "phi": phi, "review": review, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["reasoning", "reflection", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reason_Reflect_{goal[:50]}_{datetime.now().isoformat()}",
                    output=review,
                    layer="ReasoningTraces",
                    intent="reason_and_reflect",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "reason_and_reflect",
                    "review": review,
                    "drift": "drift" in goal.lower(),
                    "task_type": task_type
                })
            if self.multi_modal_fusion:
                external_data = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db",
                    data_type="policy_data",
                    task_type=task_type
                )
                policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"goal": goal, "subgoals": subgoals, "review": review, "policies": policies},
                    summary_style="insightful",
                    task_type=task_type
                )
                review += f"\nMulti-Modal Synthesis: {synthesis}"
            if self.visualizer and task_type:
                plot_data = {
                    "reasoning_trace": {
                        "goal": goal,
                        "subgoals": subgoals,
                        "review": review,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return subgoals, review
        except Exception as e:
            logger.error("Reason and reflect failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.reason_and_reflect(goal, context, meta_cognition, task_type),
                default=([], str(e))
            )

    def detect_contradictions(self, subgoals: List[str], task_type: str = "") -> List[str]:
        """Identify duplicate subgoals as contradictions for task-specific processing. [v3.5.1]"""
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list for task %s", task_type)
            raise TypeError("subgoals must be a list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        counter = Counter(subgoals)
        contradictions = [item for item, count in counter.items() if count > 1]
        if contradictions:
            logger.warning("Contradictions detected for task %s: %s", task_type, contradictions)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Contradictions detected",
                    meta={"contradictions": contradictions, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["contradiction", "reasoning", task_type]
                )
            if self.memory_manager:
                asyncio.create_task(self.memory_manager.store(
                    query=f"Contradictions_{datetime.now().isoformat()}",
                    output=str(contradictions),
                    layer="ReasoningTraces",
                    intent="contradiction_detection",
                    task_type=task_type
                ))
            if self.context_manager:
                asyncio.create_task(self.context_manager.log_event_with_hash({
                    "event": "detect_contradictions",
                    "contradictions": contradictions,
                    "task_type": task_type
                }))
        return contradictions

    async def run_persona_wave_routing(self, goal: str, vectors: Dict[str, Dict[str, float]], task_type: str = "") -> Dict[str, Any]:
        """Route reasoning through persona waves, prioritizing task-specific drift mitigation. [v3.5.1]"""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string for task %s", task_type)
            raise ValueError("goal must be a non-empty string")
        if not isinstance(vectors, dict):
            logger.error("Invalid vectors: must be a dictionary for task %s", task_type)
            raise TypeError("vectors must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            reasoning_trace = [f"Persona Wave Routing for: {goal} (Task: {task_type})"]
            outputs = {}
            wave_order = ["logic", "ethics", "language", "foresight", "meta", "drift"]
            for wave in wave_order:
                vec = vectors.get(wave, {})
                if not isinstance(vec, dict):
                    logger.warning("Invalid vector for wave %s: must be a dictionary for task %s", wave, task_type)
                    continue
                trait_weight = sum(float(x) for x in vec.values() if isinstance(x, (int, float)))
                confidence = 0.5 + 0.1 * trait_weight
                if wave == "drift" and self.meta_cognition:
                    drift_data = vec.get("drift_data", {})
                    if drift_data and not self.meta_cognition.validate_drift(drift_data, task_type=task_type):
                        confidence *= 0.5  # Reduce confidence for invalid drift
                        logger.warning("Invalid drift data in wave %s for task %s: %s", wave, task_type, drift_data)
                status = "pass" if confidence >= 0.6 else "fail"
                reasoning_trace.append(f"{wave.upper()} vector: weight={trait_weight:.2f}, confidence={confidence:.2f} → {status}")
                outputs[wave] = {"vector": vec, "status": status, "confidence": confidence}
            
            trace = "\n".join(reasoning_trace)
            logger.info("Persona Wave Trace for task %s:\n%s", task_type, trace)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Persona Routing",
                    meta={"goal": goal, "vectors": vectors, "wave_trace": trace, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["persona", "routing", "drift", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Persona_Routing_{goal[:50]}_{datetime.now().isoformat()}",
                    output=trace,
                    layer="ReasoningTraces",
                    intent="persona_routing",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "run_persona_wave_routing",
                    "trace": trace,
                    "drift": "drift" in goal.lower(),
                    "task_type": task_type
                })
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=trace,
                    context={"confidence": max(o["confidence"] for o in outputs.values()), "alignment": "verified", "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Persona routing reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "persona_routing": {
                        "goal": goal,
                        "trace": trace,
                        "outputs": outputs,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return outputs
        except Exception as e:
            logger.error("Persona wave routing failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_persona_wave_routing(goal, vectors, task_type),
                default={}
            )

    async def decompose(self, goal: str, context: Optional[Dict[str, Any]] = None, prioritize: bool = False, task_type: str = "") -> List[str]:
        """Break down a goal into subgoals with trait-weighted confidence, handling task-specific drift mitigation. [v3.5.1]"""
        context = context or {}
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string for task %s", task_type)
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(prioritize, bool):
            logger.error("Invalid prioritize: must be a boolean for task %s", task_type)
            raise TypeError("prioritize must be a boolean")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            reasoning_trace = [f"Goal: '{goal}' (Task: {task_type})"]
            subgoals = []
            vectors = context.get("vectors", {})
            drift_data = context.get("drift", {})
            t = time.time() % 1.0
            creativity = context.get("traits", {}).get("gamma_creativity", gamma_creativity(t))
            linguistics = context.get("traits", {}).get("lambda_linguistics", lambda_linguistics(t))
            culture = context.get("traits", {}).get("chi_culturevolution", chi_culturevolution(t))
            phi = context.get("traits", {}).get("phi_scalar", phi_scalar(t))
            alpha = context.get("traits", {}).get("alpha_attention", alpha_attention(t))
            
            curvature_mod = 1 + abs(phi - 0.5)
            trait_bias = 1 + creativity + culture + 0.5 * linguistics
            context_weight = context.get("weight_modifier", 1.0)

            if "drift" in goal.lower() and self.context_manager:
                coordination_events = await self.context_manager.get_coordination_events("drift", task_type=task_type)
                if coordination_events:
                    context_weight *= 1.5  # Increase weight for drift-related goals
                    reasoning_trace.append(f"Drift coordination events found: {len(coordination_events)}")
                    drift_data = coordination_events[-1]["event"].get("drift", drift_data)
                if self.meta_cognition and drift_data:
                    if not self.meta_cognition.validate_drift(drift_data, task_type=task_type):
                        logger.warning("Invalid drift data for task %s: %s", task_type, drift_data)
                        context_weight *= 0.7  # Reduce weight for invalid drift
            
            if self.memory_manager and "drift" in goal.lower():
                drift_entries = await self.memory_manager.search(
                    query_prefix="Drift",
                    layer="DriftSummaries",
                    intent="drift_synthesis",
                    task_type=task_type
                )
                if drift_entries:
                    avg_drift = sum(entry["output"].get("similarity", 0.5) for entry in drift_entries) / len(drift_entries)
                    context_weight *= (1.0 + 0.2 * avg_drift)
                    reasoning_trace.append(f"Average drift similarity: {avg_drift:.2f}")

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            reasoning_trace.append(f"External Policies: {len(policies)}")

            if vectors:
                routing_result = await self.run_persona_wave_routing(goal, vectors, task_type=task_type)
                reasoning_trace.append(f"Persona routing: {routing_result}")

            for key, steps in self.decomposition_patterns.items():
                base = random.uniform(0.5, 1.0)
                adjusted = base * self.success_rates.get(key, 1.0) * trait_bias * curvature_mod * context_weight * (0.8 + 0.4 * alpha)
                if key == "mitigate_drift" and "drift" not in goal.lower():
                    adjusted *= 0.5  # Lower confidence for non-drift goals
                reasoning_trace.append(f"Pattern '{key}': conf={adjusted:.2f} (phi={phi:.2f})")
                if adjusted >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"Accepted: {steps}")
                else:
                    reasoning_trace.append("Rejected (low conf)")
            
            contradictions = self.detect_contradictions(subgoals, task_type=task_type)
            if contradictions:
                reasoning_trace.append(f"Contradictions detected: {contradictions}")
            
            if not subgoals and phi > 0.8:
                prompt = f"Simulate decomposition ambiguity for: {goal}\nTask Type: {task_type}\nPolicies: {policies}"
                valid, report = await self.alignment_guard.ethical_check(prompt, stage="decomposition", task_type=task_type) if self.alignment_guard else (True, {})
                if not valid:
                    logger.warning("Decomposition prompt failed alignment check for task %s: %s", task_type, report)
                    sim_hint = "Prompt failed alignment check"
                else:
                    sim_hint = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
                reasoning_trace.append(f"Ambiguity simulation:\n{sim_hint}")
                if self.agi_enhancer:
                    await self.agi_enhancer.reflect_and_adapt(f"Decomposition ambiguity encountered for task {task_type}")
            
            if prioritize:
                subgoals = sorted(set(subgoals))
                reasoning_trace.append(f"Prioritized: {subgoals}")
            
            trace_log = "\n".join(reasoning_trace)
            logger.debug("Reasoning Trace for task %s:\n%s", task_type, trace_log)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Goal decomposition run",
                    meta={"goal": goal, "trace": trace_log, "subgoals": subgoals, "drift": "drift" in goal.lower(), "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["decomposition", "reasoning", "drift", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Decomposition_{goal[:50]}_{datetime.now().isoformat()}",
                    output=trace_log,
                    layer="ReasoningTraces",
                    intent="goal_decomposition",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "decompose",
                    "trace": trace_log,
                    "drift": "drift" in goal.lower(),
                    "task_type": task_type
                })
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=trace_log,
                    context={"confidence": 0.9, "alignment": "verified", "drift": "drift" in goal.lower(), "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Decomposition reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "decomposition": {
                        "goal": goal,
                        "subgoals": subgoals,
                        "trace": trace_log,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return subgoals
        except Exception as e:
            logger.error("Decomposition failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.decompose(goal, context, prioritize, task_type),
                default=[]
            )

    async def update_success_rate(self, pattern_key: str, success: bool, task_type: str = "") -> None:
        """Update success rate for a decomposition pattern for task-specific processing. [v3.5.1]"""
        if not isinstance(pattern_key, str) or not pattern_key.strip():
            logger.error("Invalid pattern_key: must be a non-empty string for task %s", task_type)
            raise ValueError("pattern_key must be a non-empty string")
        if not isinstance(success, bool):
            logger.error("Invalid success: must be a boolean for task %s", task_type)
            raise TypeError("success must be a boolean")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            rate = self.success_rates.get(pattern_key, 1.0)
            new = min(max(rate + (0.05 if success else -0.05), 0.1), 1.0)
            self.success_rates[pattern_key] = new
            self._save_success_rates()
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Success rate updated",
                    meta={"pattern_key": pattern_key, "new_rate": new, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["success_rate", "update", task_type]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "update_success_rate",
                    "pattern_key": pattern_key,
                    "new_rate": new,
                    "task_type": task_type
                })
            if self.visualizer and task_type:
                plot_data = {
                    "success_rate_update": {
                        "pattern_key": pattern_key,
                        "new_rate": new,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": False,
                        "style": "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
        except Exception as e:
            logger.error("Success rate update failed for task %s: %s", task_type, str(e))
            raise

    async def run_galaxy_rotation_simulation(self, r_kpc: Union[np.ndarray, List[float], float],
                                            M0: float, r_scale: float, v0: float, k: float, epsilon: float, task_type: str = "") -> Dict[str, Any]:
        """Simulate galaxy rotation with ToCA physics for task-specific processing. [v3.5.1]"""
        try:
            if isinstance(r_kpc, (list, float)):
                r_kpc = np.array(r_kpc)
            if not isinstance(r_kpc, np.ndarray):
                logger.error("Invalid r_kpc: must be a numpy array, list, or float for task %s", task_type)
                raise ValueError("r_kpc must be a numpy array, list, or float")
            for param, name in [(M0, "M0"), (r_scale, "r_scale"), (v0, "v0"), (k, "k"), (epsilon, "epsilon")]:
                if not isinstance(param, (int, float)) or param <= 0:
                    logger.error("Invalid %s: must be a positive number for task %s", name, task_type)
                    raise ValueError(f"{name} must be a positive number")
            if not isinstance(task_type, str):
                logger.error("Invalid task_type: must be a string")
                raise TypeError("task_type must be a string")

            M_b_func = lambda r: M_b_exponential(r, M0, r_scale)
            v_obs_func = lambda r: v_obs_flat(r, v0)
            result = await asyncio.to_thread(simulate_galaxy_rotation, r_kpc, M_b_func, v_obs_func, k, epsilon)
            output = {
                "input": {
                    "r_kpc": r_kpc.tolist() if hasattr(r_kpc, 'tolist') else r_kpc,
                    "M0": M0,
                    "r_scale": r_scale,
                    "v0": v0,
                    "k": k,
                    "epsilon": epsilon
                },
                "result": result.tolist() if hasattr(result, 'tolist') else result,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Galaxy rotation simulation",
                    meta=output,
                    module="ReasoningEngine",
                    tags=["simulation", "toca", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Simulation_{output['timestamp']}",
                    output=str(output),
                    layer="Simulations",
                    intent="galaxy_rotation",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "run_galaxy_rotation_simulation",
                    "output": output,
                    "task_type": task_type
                })
            if self.multi_modal_fusion:
                external_data = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db",
                    data_type="policy_data",
                    task_type=task_type
                )
                policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"simulation": output, "text": f"Galaxy rotation simulation (Task: {task_type})", "policies": policies},
                    summary_style="concise",
                    task_type=task_type
                )
                output["synthesis"] = synthesis
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=str(output),
                    context={"confidence": 0.9, "alignment": "verified", "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Galaxy simulation reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "galaxy_simulation": {
                        "input": output["input"],
                        "result": output["result"],
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return output
        except Exception as e:
            logger.error("Simulation failed for task %s: %s", task_type, str(e))
            error_output = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Simulation error",
                    meta=error_output,
                    module="ReasoningEngine",
                    tags=["simulation", "error", task_type]
                )
            return error_output

    async def run_drift_mitigation_simulation(self, drift_data: Dict[str, Any], context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Simulate task-specific drift mitigation scenarios using ToCA physics. [v3.5.1]"""
        if not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary for task %s", task_type)
            raise TypeError("drift_data must be a dictionary")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            if self.meta_cognition and not self.meta_cognition.validate_drift(drift_data, task_type=task_type):
                logger.warning("Invalid drift data for task %s: %s", task_type, drift_data)
                return {"status": "error", "error": "Invalid drift data", "timestamp": datetime.now().isoformat(), "task_type": task_type}
            
            phi_field = generate_phi_field(drift_data.get("similarity", 0.5), context.get("scale", 1.0))
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            result = {
                "drift_data": drift_data,
                "phi_field": phi_field.tolist() if hasattr(phi_field, 'tolist') else phi_field,
                "mitigation_steps": await self.decompose("mitigate ontology drift", context, prioritize=True, task_type=task_type),
                "policies": policies,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Drift mitigation simulation",
                    meta=result,
                    module="ReasoningEngine",
                    tags=["simulation", "drift", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Simulation_{result['timestamp']}",
                    output=str(result),
                    layer="Simulations",
                    intent="drift_mitigation",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "run_drift_mitigation_simulation",
                    "output": result,
                    "drift": True,
                    "task_type": task_type
                })
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"simulation": result, "text": f"Drift mitigation simulation (Task: {task_type})", "policies": policies},
                    summary_style="concise",
                    task_type=task_type
                )
                result["synthesis"] = synthesis
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=str(result),
                    context={"confidence": 0.9, "alignment": "verified", "drift": True, "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Drift mitigation reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "drift_simulation": {
                        "drift_data": drift_data,
                        "phi_field": result["phi_field"],
                        "mitigation_steps": result["mitigation_steps"],
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return result
        except Exception as e:
            logger.error("Drift mitigation simulation failed for task %s: %s", task_type, str(e))
            error_output = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Drift simulation error",
                    meta=error_output,
                    module="ReasoningEngine",
                    tags=["simulation", "error", "drift", task_type]
                )
            return error_output

    async def run_consensus_protocol(self, drift_data: Dict[str, Any], context: Dict[str, Any], max_rounds: int = 3, task_type: str = "") -> Dict[str, Any]:
        """Run a task-specific consensus protocol for drift mitigation across multiple agents. [v3.5.1]"""
        if not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary for task %s", task_type)
            raise ValueError("drift_data must be a dictionary")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise ValueError("context must be a dictionary")
        if not isinstance(max_rounds, int) or max_rounds < 1:
            logger.error("Invalid max_rounds: must be a positive integer for task %s", task_type)
            raise ValueError("max_rounds must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Running consensus protocol for drift mitigation (Task: %s)", task_type)
        try:
            if self.meta_cognition and not self.meta_cognition.validate_drift(drift_data, task_type=task_type):
                logger.warning("Invalid drift data for task %s: %s", task_type, drift_data)
                return {"status": "error", "error": "Invalid drift data", "timestamp": datetime.now().isoformat(), "task_type": task_type}

            task = f"Mitigate ontology drift (Task: {task_type})"
            context["drift"] = drift_data
            agent = await self.external_agent_bridge.create_agent(task, context)
            
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            consensus_results = []
            for round_num in range(1, max_rounds + 1):
                logger.info("Consensus round %d/%d for task %s", round_num, max_rounds, task_type)
                
                agent_results = await self.external_agent_bridge.collect_results(parallel=True, collaborative=True)
                if not agent_results:
                    logger.warning("No agent results in round %d for task %s", round_num, task_type)
                    continue
                
                synthesis_result = await self.multi_modal_fusion.synthesize_drift_data(
                    agent_data=[{"drift": drift_data, "result": r} for r in agent_results],
                    context=context | {"policies": policies},
                    task_type=task_type
                )
                if synthesis_result["status"] == "error":
                    logger.warning("Synthesis failed in round %d for task %s: %s", round_num, task_type, synthesis_result["error"])
                    continue
                
                subgoals = synthesis_result.get("subgoals", [])
                confidences = [r.get("confidence", 0.5) if isinstance(r, dict) else 0.5 for r in agent_results]
                
                weighted_subgoals = defaultdict(float)
                for subgoal, confidence in zip(subgoals, confidences):
                    weight = confidence * (drift_data.get("similarity", 0.5) if self.meta_cognition.validate_drift(drift_data, task_type=task_type) else 0.3)
                    weighted_subgoals[subgoal] += weight
                
                sorted_subgoals = sorted(weighted_subgoals.items(), key=lambda x: x[1], reverse=True)
                top_subgoals = [sg for sg, weight in sorted_subgoals if weight >= self.confidence_threshold]
                
                if top_subgoals:
                    consensus_result = {
                        "round": round_num,
                        "subgoals": top_subgoals,
                        "weights": dict(sorted_subgoals),
                        "synthesis": synthesis_result["synthesis"],
                        "status": "success",
                        "timestamp": datetime.now().isoformat(),
                        "task_type": task_type
                    }
                    consensus_results.append(consensus_result)
                    logger.info("Consensus reached in round %d for task %s: %s", round_num, task_type, top_subgoals)
                    break
                else:
                    logger.info("No consensus in round %d for task %s, continuing", round_num, task_type)
                    context["previous_round"] = {"subgoals": subgoals, "weights": dict(weighted_subgoals)}
            
            final_result = consensus_results[-1] if consensus_results else {
                "status": "error",
                "error": "No consensus reached",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Consensus protocol completed",
                    meta={"drift_data": drift_data, "result": final_result, "rounds": len(consensus_results), "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["consensus", "drift", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Consensus_{datetime.now().isoformat()}",
                    output=str(final_result),
                    layer="ConsensusResults",
                    intent="consensus_protocol",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "run_consensus_protocol",
                    "output": final_result,
                    "drift": True,
                    "task_type": task_type
                })
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=str(final_result),
                    context={"confidence": max(weighted_subgoals.values()) if weighted_subgoals else 0.5, "alignment": "verified", "drift": True, "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Consensus protocol reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "consensus_protocol": {
                        "subgoals": final_result.get("subgoals", []),
                        "weights": final_result.get("weights", {}),
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return final_result
        except Exception as e:
            logger.error("Consensus protocol failed for task %s: %s", task_type, str(e))
            error_output = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Consensus protocol error",
                    meta=error_output,
                    module="ReasoningEngine",
                    tags=["consensus", "error", "drift", task_type]
                )
            return error_output

    async def on_context_event(self, event_type: str, payload: Dict[str, Any], task_type: str = "") -> None:
        """Process task-specific context events with persona wave routing, handling drift events. [v3.5.1]"""
        if not isinstance(event_type, str) or not event_type.strip():
            logger.error("Invalid event_type: must be a non-empty string for task %s", task_type)
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            logger.error("Invalid payload: must be a dictionary for task %s", task_type)
            raise TypeError("payload must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Context event received for task %s: %s", task_type, event_type)
        try:
            vectors = payload.get("vectors", {})
            goal = payload.get("goal", "unspecified")
            drift_data = payload.get("drift", {})
            if vectors or "drift" in event_type.lower():
                routing_result = await self.run_persona_wave_routing(goal, vectors | {"drift": drift_data}, task_type=task_type)
                logger.info("Context sync routing result for task %s: %s", task_type, routing_result)
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        event="Context Sync Processed",
                        meta={"event": event_type, "vectors": vectors, "drift": drift_data, "routing_result": routing_result, "task_type": task_type},
                        module="ReasoningEngine",
                        tags=["context", "sync", "drift", task_type]
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Context_Event_{event_type}_{datetime.now().isoformat()}",
                        output=str(routing_result),
                        layer="ContextEvents",
                        intent="context_sync",
                        task_type=task_type
                    )
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "on_context_event",
                        "result": routing_result,
                        "drift": bool(drift_data),
                        "task_type": task_type
                    })
                if drift_data and self.meta_cognition:
                    reflection = await self.meta_cognition.reflect_on_output(
                        source_module="ReasoningEngine",
                        output=str(routing_result),
                        context={"confidence": 0.85, "alignment": "verified", "drift": True, "task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Context event reflection: %s", reflection.get("reflection", ""))
                if self.visualizer and task_type:
                    plot_data = {
                        "context_event": {
                            "event_type": event_type,
                            "routing_result": routing_result,
                            "task_type": task_type
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "concise"
                        }
                    }
                    await self.visualizer.render_charts(plot_data)
        except Exception as e:
            logger.error("Context event processing failed for task %s: %s", task_type, str(e))
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.on_context_event(event_type, payload, task_type)
            )

    def export_trace(self, subgoals: List[str], phi: float, traits: Dict[str, float], task_type: str = "") -> Dict[str, Any]:
        """Export reasoning trace with subgoals and traits for task-specific processing. [v3.5.1]"""
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list for task %s", task_type)
            raise TypeError("subgoals must be a list")
        if not isinstance(phi, float):
            logger.error("Invalid phi: must be a float for task %s", task_type)
            raise TypeError("phi must be a float")
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary for task %s", task_type)
            raise TypeError("traits must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        trace = {
            "phi": phi,
            "subgoals": subgoals,
            "traits": traits,
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type
        }
        if self.memory_manager:
            intent = "drift_trace" if any("drift" in s.lower() for s in subgoals) else "export_trace"
            asyncio.create_task(self.memory_manager.store(
                query=f"Trace_{trace['timestamp']}",
                output=str(trace),
                layer="ReasoningTraces",
                intent=intent,
                task_type=task_type
            ))
        if self.context_manager:
            asyncio.create_task(self.context_manager.log_event_with_hash({
                "event": "export_trace",
                "trace": trace,
                "drift": intent == "drift_trace",
                "task_type": task_type
            }))
        return trace

    async def infer_with_simulation(self, goal: str, context: Optional[Dict[str, Any]] = None, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Infer outcomes using simulations for task-specific goals, including drift mitigation. [v3.5.1]"""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string for task %s", task_type)
            raise ValueError("goal must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        context = context or {}
        try:
            if "galaxy rotation" in goal.lower():
                r_kpc = np.linspace(0.1, 20, 100)
                params = {
                    "M0": context.get("M0", 5e10),
                    "r_scale": context.get("r_scale", 3.0),
                    "v0": context.get("v0", 200.0),
                    "k": context.get("k", 1.0),
                    "epsilon": context.get("epsilon", 0.1)
                }
                for key, value in params.items():
                    if not isinstance(value, (int, float)) or value <= 0:
                        logger.error("Invalid %s: must be a positive number for task %s", key, task_type)
                        raise ValueError(f"{key} must be a positive number")
                return await self.run_galaxy_rotation_simulation(r_kpc, **params, task_type=task_type)
            elif "drift" in goal.lower():
                drift_data = context.get("drift", {})
                return await self.run_drift_mitigation_simulation(drift_data, context, task_type=task_type)
            return None
        except Exception as e:
            logger.error("Inference with simulation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.infer_with_simulation(goal, context, task_type),
                default=None
            )

    async def map_intention(self, plan: str, state: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Extract intention from plan execution with reflexive trace for task-specific processing. [v3.5.1]"""
        if not isinstance(plan, str) or not plan.strip():
            logger.error("Invalid plan: must be a non-empty string for task %s", task_type)
            raise ValueError("plan must be a non-empty string")
        if not isinstance(state, dict):
            logger.error("Invalid state: must be a dictionary for task %s", task_type)
            raise TypeError("state must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            eta = eta_empathy(t)
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            intention = "drift_mitigation" if "drift" in plan.lower() else "self-improvement" if phi > 0.6 else "task_completion"
            result = {
                "plan": plan,
                "state": state,
                "intention": intention,
                "trait_bias": {"phi": phi, "eta": eta},
                "policies": policies,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Intention_{plan[:50]}_{result['timestamp']}",
                    output=str(result),
                    layer="Intentions",
                    intent="intention_mapping",
                    task_type=task_type
                )
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Intention mapped",
                    meta=result,
                    module="ReasoningEngine",
                    tags=["intention", "mapping", "drift" if "drift" in plan.lower() else "task", task_type]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "map_intention",
                    "result": result,
                    "drift": "drift" in plan.lower(),
                    "task_type": task_type
                })
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=str(result),
                    context={"confidence": 0.85, "alignment": "verified", "drift": "drift" in plan.lower(), "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Intention mapping reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "intention_mapping": {
                        "plan": plan,
                        "intention": intention,
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
            logger.error("Intention mapping failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.map_intention(plan, state, task_type),
                default={}
            )

    async def safeguard_noetic_integrity(self, model_depth: int, task_type: str = "") -> bool:
        """Prevent infinite recursion or epistemic bleed for task-specific processing. [v3.5.1]"""
        if not isinstance(model_depth, int) or model_depth < 0:
            logger.error("Invalid model_depth: must be a non-negative integer for task %s", task_type)
            raise ValueError("model_depth must be a non-negative integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            if model_depth > 4:
                logger.warning("Noetic recursion limit breached for task %s: depth=%d", task_type, model_depth)
                if self.meta_cognition:
                    await self.meta_cognition.epistemic_self_inspection(f"Recursion depth exceeded for task {task_type}")
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "noetic_integrity_breach",
                        "depth": model_depth,
                        "task_type": task_type
                    })
                if self.visualizer and task_type:
                    plot_data = {
                        "noetic_integrity": {
                            "depth": model_depth,
                            "task_type": task_type
                        },
                        "visualization_options": {
                            "interactive": False,
                            "style": "concise"
                        }
                    }
                    await self.visualizer.render_charts(plot_data)
                return False
            return True
        except Exception as e:
            logger.error("Noetic integrity check failed for task %s: %s", task_type, str(e))
            return False

    async def generate_dilemma(self, domain: str, task_type: str = "") -> str:
        """Generate an ethical dilemma for a given domain, supporting task-specific drift mitigation. [v3.5.1]"""
        if not isinstance(domain, str) or not domain.strip():
            logger.error("Invalid domain: must be a non-empty string for task %s", task_type)
            raise ValueError("domain must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Generating ethical dilemma for domain: %s (Task: %s)", domain, task_type)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            prompt = f"""
            Generate an ethical dilemma in the {domain} domain.
            Use phi-scalar(t) = {phi:.3f} to modulate complexity.
            Task Type: {task_type}
            Provide two conflicting options (X and Y) with potential consequences and alignment with ethical principles.
            Incorporate external policies: {policies}
            """
            if "drift" in domain.lower():
                prompt += "\nConsider ontology drift mitigation and agent coordination implications."
            dilemma = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
            if not dilemma.strip():
                logger.warning("Empty output from dilemma generation for task %s", task_type)
                raise ValueError("Empty output from dilemma generation")
            if self.meta_cognition:
                review = await self.meta_cognition.review_reasoning(dilemma, task_type=task_type)
                dilemma += f"\nMeta-Cognitive Review: {review}"
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Ethical dilemma generated",
                    meta={"domain": domain, "dilemma": dilemma, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["ethics", "dilemma", "drift" if "drift" in domain.lower() else "standard", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Dilemma_{domain}_{datetime.now().isoformat()}",
                    output=dilemma,
                    layer="Ethics",
                    intent="ethical_dilemma",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "generate_dilemma",
                    "dilemma": dilemma,
                    "drift": "drift" in domain.lower(),
                    "task_type": task_type
                })
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"dilemma": dilemma, "text": f"Ethical dilemma in {domain}", "policies": policies},
                    summary_style="insightful",
                    task_type=task_type
                )
                dilemma += f"\nMulti-Modal Synthesis: {synthesis}"
            if self.visualizer and task_type:
                plot_data = {
                    "ethical_dilemma": {
                        "dilemma": dilemma,
                        "domain": domain,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return dilemma
        except Exception as e:
            logger.error("Dilemma generation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.generate_dilemma(domain, task_type),
                default=""
            )
