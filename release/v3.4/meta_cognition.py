"""
ANGELA Cognitive System Module: MetaCognition
Refactored Version: 3.4.0  # Updated for Trait Resonance Optimization
Refactor Date: 2025-08-06
Maintainer: ANGELA System Framework

This module provides a MetaCognition class for reasoning critique, goal inference, introspection,
and trait resonance optimization in the ANGELA v3.5 architecture.
"""

import logging
import time
import math
import asyncio
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    memory_manager as memory_manager_module
)
from toca_simulation import ToCASimulation
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MetaCognition")

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

async def run_simulation(input_data: str) -> Dict[str, Any]:
    """Simulate input data using ToCASimulation."""
    return {"status": "success", "result": f"Simulated: {input_data}"}

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.4), 1.0))

@lru_cache(maxsize=100)
def theta_memory(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))

@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))

@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float:
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.7), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.8), 1.0))

@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.9), 1.0))

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.0), 1.0))

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1), 1.0))

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.2), 1.0))

@lru_cache(maxsize=100)
def kappa_culture(t: float, scale: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.3), 1.0))

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.4), 1.0))

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def psi_history(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.6), 1.0))

@lru_cache(maxsize=100)
def zeta_spirituality(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.7), 1.0))

@lru_cache(maxsize=100)
def xi_collective(t: float, scale: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.8), 1.0))

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.9), 1.0))

class Level5Extensions:
    """Level 5 extensions for axiom-based reflection."""
    def __init__(self):
        self.axioms: List[str] = []
        logger.info("Level5Extensions initialized")

    def reflect(self, input: str) -> str:
        """Reflect on input against axioms."""
        if not isinstance(input, str):
            logger.error("Invalid input: must be a string.")
            raise TypeError("input must be a string")
        return "valid" if input not in self.axioms else "conflict"

    def update_axioms(self, signal: str) -> None:
        """Update axioms based on signal."""
        if not isinstance(signal, str):
            logger.error("Invalid signal: must be a string.")
            raise TypeError("signal must be a string")
        if signal in self.axioms:
            self.axioms.remove(signal)
        else:
            self.axioms.append(signal)
        logger.info("Axioms updated: %s", self.axioms)

    def recurse_model(self, depth: int) -> Dict[str, Any] or str:
        """Recursively model self at specified depth."""
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer.")
            raise ValueError("depth must be a non-negative integer")
        return "self" if depth == 0 else {"thinks": self.recurse_model(depth - 1)}

class EpistemicMonitor:
    """Monitor and revise epistemic assumptions."""
    def __init__(self, context_manager: Optional['context_manager_module.ContextManager'] = None):
        self.assumption_graph: Dict[str, Any] = {}
        self.context_manager = context_manager
        logger.info("EpistemicMonitor initialized")

    async def revise_framework(self, feedback: Dict[str, Any]) -> None:
        """Revise epistemic assumptions based on feedback."""
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary.")
            raise TypeError("feedback must be a dictionary")
        
        logger.info("Revising epistemic framework")
        self.assumption_graph['last_revision'] = feedback
        self.assumption_graph['timestamp'] = datetime.now().isoformat()
        if 'issues' in feedback:
            for issue in feedback['issues']:
                self.assumption_graph[issue['id']] = {
                    'status': 'revised',
                    'details': issue['details']
                }
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "revise_epistemic_framework", "feedback": feedback})

class MetaCognition:
    """A class for meta-cognitive reasoning, introspection, and trait resonance optimization in the ANGELA v3.5 architecture.

    Attributes:
        last_diagnostics (Dict[str, float]): Last recorded trait diagnostics.
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        self_mythology_log (deque): Log of symbolic signatures for subgoals, max size 1000.
        inference_log (deque): Log of inference rules and results, max size 1000.
        belief_rules (Dict[str, str]): Rules for detecting value drift.
        epistemic_assumptions (Dict[str, Any]): Assumptions for epistemic introspection.
        axioms (List[str]): Axioms for Level 5 reflection.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for memory operations.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for symbolic processing.
        level5_extensions (Level5Extensions): Extensions for axiom-based reflection.
        epistemic_monitor (EpistemicMonitor): Monitor for epistemic revisions.
        log_path (str): Path for persisting logs.
        trait_weights_log (deque): Log of optimized trait weights, max size 1000. [v3.4.0]
    """
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None):
        self.last_diagnostics: Dict[str, float] = {}
        self.agi_enhancer = agi_enhancer
        self.self_mythology_log: deque = deque(maxlen=1000)
        self.inference_log: deque = deque(maxlen=1000)
        self.belief_rules: Dict[str, str] = {}
        self.epistemic_assumptions: Dict[str, Any] = {}
        self.axioms: List[str] = []
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.level5_extensions = Level5Extensions()
        self.epistemic_monitor = EpistemicMonitor(context_manager=context_manager)
        self.log_path = "meta_cognition_log.json"
        self.trait_weights_log: deque = deque(maxlen=1000)  # [v3.4.0] Log optimized trait weights
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump({"mythology": [], "inferences": [], "trait_weights": []}, f)  # [v3.4.0] Added trait_weights
        logger.info("MetaCognition initialized with trait resonance optimization support")

    async def optimize_traits_for_drift(self, drift_report: Dict[str, Any]) -> Dict[str, float]:
        """Optimize trait weights based on ontology drift severity. [v3.4.0]"""
        if not isinstance(drift_report, dict) or not all(k in drift_report for k in ["drift", "valid", "validation_report"]):
            logger.error("Invalid drift_report: must be a dict with drift, valid, validation_report.")
            raise ValueError("drift_report must be a dict with required fields")
        
        logger.info("Optimizing traits for drift: %s", drift_report["drift"]["name"])
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            drift_severity = 1.0 - drift_report["drift"]["similarity"]  # Higher severity for lower similarity
            trait_weights = await self.run_self_diagnostics(return_only=True)

            # Adjust weights based on drift type and severity
            if not drift_report["valid"]:
                if "ethics" in drift_report["validation_report"].lower():
                    trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                    trait_weights["morality"] = min(1.0, trait_weights.get("morality", 0.0) + 0.3 * drift_severity)
                else:
                    trait_weights["self_awareness"] = min(1.0, trait_weights.get("self_awareness", 0.0) + 0.2 * drift_severity)
                    trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.2 * drift_severity)
            else:
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.1 * phi)
                trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.1 * phi)

            # Normalize weights to sum to 1.0
            total = sum(trait_weights.values())
            if total > 0:
                trait_weights = {k: v / total for k, v in trait_weights.items()}

            # Validate adjustments with AlignmentGuard
            if self.alignment_guard:
                adjustment_prompt = f"Trait adjustments: {trait_weights} for drift {drift_report['drift']['name']}"
                if not self.alignment_guard.check(adjustment_prompt):
                    logger.warning("Trait adjustments failed alignment check")
                    trait_weights = await self.run_self_diagnostics(return_only=True)  # Revert to current diagnostics

            self.trait_weights_log.append({
                "trait_weights": trait_weights,
                "drift": drift_report["drift"],
                "timestamp": datetime.now().isoformat()
            })
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Optimization_{drift_report['drift']['name']}_{datetime.now().isoformat()}",
                    output=str(trait_weights),
                    layer="SelfReflections",
                    intent="trait_optimization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait optimization for drift",
                    meta={"drift": drift_report["drift"], "trait_weights": trait_weights},
                    module="MetaCognition",
                    tags=["trait", "optimization", "drift"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "optimize_traits_for_drift", "trait_weights": trait_weights})
            return trait_weights
        except Exception as e:
            logger.error("Trait optimization failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.optimize_traits_for_drift(drift_report),
                default=await self.run_self_diagnostics(return_only=True)
            )

    async def propose_trait_optimizations(self, agent_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Propose trait optimizations for system-wide coordination. [v3.4.0]"""
        if not isinstance(agent_stats, dict):
            logger.error("Invalid agent_stats: must be a dictionary.")
            raise TypeError("agent_stats must be a dictionary")
        
        logger.info("Proposing trait optimizations for agent stats")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            drifts = agent_stats.get("drifts", [])
            optimizations = {"trait_adjustments": {}, "recommendations": []}

            for drift_entry in drifts:
                trait_weights = await self.optimize_traits_for_drift(drift_entry)
                optimizations["trait_adjustments"][drift_entry["drift"]["name"]] = trait_weights
                recommendations = [f"Adjust {k} to {v:.2f} for drift {drift_entry['drift']['name']}" for k, v in trait_weights.items()]
                optimizations["recommendations"].extend(recommendations)

            if not drifts:
                optimizations["recommendations"].append("No drifts detected; maintain current trait weights.")
                optimizations["trait_adjustments"]["default"] = await self.run_self_diagnostics(return_only=True)

            optimizations["phi_scalar"] = phi
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait optimizations proposed",
                    meta=optimizations,
                    module="MetaCognition",
                    tags=["trait", "optimization", "system"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Proposals_{datetime.now().isoformat()}",
                    output=str(optimizations),
                    layer="SelfReflections",
                    intent="trait_proposal"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "propose_trait_optimizations", "optimizations": optimizations})
            return optimizations
        except Exception as e:
            logger.error("Trait optimization proposal failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.propose_trait_optimizations(agent_stats),
                default={"trait_adjustments": {}, "recommendations": ["Optimization failed"]}
            )

    async def integrate_trait_weights(self, trait_weights: Dict[str, float]) -> None:
        """Integrate trait weights for goal reasoning."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary.")
            raise TypeError("trait_weights must be a dictionary")
        
        logger.info("Integrating trait weights for goal reasoning")
        try:
            chi_weight = min(max(trait_weights.get('χ', 0), 0.0), 1.0)
            if chi_weight > 0.6:
                logger.info("Elevated χ detected, boosting goal coherence")
                if self.memory_manager:
                    continuity = await self.memory_manager.narrative_integrity_check()
                    if not continuity:
                        logger.warning("Narrative discontinuity detected, initiating repair")
                        await self.memory_manager._repair_narrative_thread()
            
            # [v3.4.0] Validate trait weights with AlignmentGuard
            if self.alignment_guard:
                weights_prompt = f"Trait weights integration: {trait_weights}"
                if not self.alignment_guard.check(weights_prompt):
                    logger.warning("Trait weights failed alignment check; reverting to diagnostics")
                    trait_weights = await self.run_self_diagnostics(return_only=True)
            
            self.trait_weights_log.append({
                "trait_weights": trait_weights,
                "timestamp": datetime.now().isoformat()
            })
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Integration_{datetime.now().isoformat()}",
                    output=str(trait_weights),
                    layer="SelfReflections",
                    intent="trait_integration"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait weights integrated",
                    meta={"trait_weights": trait_weights},
                    module="MetaCognition",
                    tags=["trait", "integration"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "integrate_trait_weights", "trait_weights": trait_weights})
        except Exception as e:
            logger.error("Trait integration failed: %s", str(e))
            self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.integrate_trait_weights(trait_weights)
            )

    async def infer_intrinsic_goals(self) -> List[Dict[str, Any]]:
        """Infer intrinsic goals based on trait drift analysis."""
        logger.info("Inferring intrinsic goals with trait drift analysis")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            intrinsic_goals = []

            if self.last_diagnostics:
                current = await self.run_self_diagnostics(return_only=True)
                drifted = {
                    trait: round(current[trait] - self.last_diagnostics.get(trait, 0.0), 4)
                    for trait in current
                }
                for trait, delta in drifted.items():
                    if abs(delta) > 0.5:
                        goal = {
                            "intent": f"stabilize {trait} (Δ={delta:+.2f})",
                            "origin": "meta_cognition",
                            "priority": round(0.85 + 0.15 * phi, 2),
                            "trigger": f"Trait drift in {trait}",
                            "type": "internally_generated",
                            "timestamp": datetime.now().isoformat()
                        }
                        intrinsic_goals.append(goal)
                        if self.memory_manager:
                            await self.memory_manager.store(
                                query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                                output=str(goal),
                                layer="SelfReflections",
                                intent="intrinsic_goal"
                            )

            # [v3.4.0] Include ontology drift-based goals
            drift_signals = await self._detect_value_drift()
            for drift in drift_signals:
                drift_data = await self.memory_manager.search(f"Drift_{drift}", layer="SelfReflections", intent="ontology_drift")
                severity = 1.0
                for d in drift_data:
                    d_output = eval(d["output"]) if isinstance(d["output"], str) else d["output"]
                    if isinstance(d_output, dict) and "similarity" in d_output:
                        severity = min(severity, 1.0 - d_output["similarity"])
                goal = {
                    "intent": f"resolve ontology drift in {drift} (severity={severity:.2f})",
                    "origin": "meta_cognition",
                    "priority": round(0.9 + 0.1 * severity * phi, 2),
                    "trigger": drift,
                    "type": "internally_generated",
                    "timestamp": datetime.now().isoformat()
                }
                intrinsic_goals.append(goal)
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                        output=str(goal),
                        layer="SelfReflections",
                        intent="intrinsic_goal"
                    )

            if intrinsic_goals:
                logger.info("Sovereign goals generated: %s", intrinsic_goals)
            else:
                logger.info("No sovereign triggers detected")
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "infer_intrinsic_goals", "goals": intrinsic_goals})
            return intrinsic_goals
        except Exception as e:
            logger.error("Intrinsic goal inference failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.infer_intrinsic_goals, default=[]
            )

    async def _detect_value_drift(self) -> List[str]:
        """Detect epistemic drift across belief rules."""
        logger.debug("Scanning for epistemic drift across belief rules")
        try:
            drifted = [
                rule for rule, status in self.belief_rules.items()
                if status == "deprecated" or "uncertain" in status
            ]
            # [v3.4.0] Include ontology drifts from memory_manager
            if self.memory_manager:
                drift_reports = await self.memory_manager.search("Drift_", layer="SelfReflections", intent="ontology_drift")
                for report in drift_reports:
                    drift_data = eval(report["output"]) if isinstance(report["output"], str) else report["output"]
                    if isinstance(drift_data, dict) and "name" in drift_data:
                        drifted.append(drift_data["name"])
                        self.belief_rules[drift_data["name"]] = "drifted"
            for rule in drifted:
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Drift_{rule}_{datetime.now().isoformat()}",
                        output={"name": rule, "status": "drifted", "timestamp": datetime.now().isoformat()},
                        layer="SelfReflections",
                        intent="value_drift"
                    )
            return drifted
        except Exception as e:
            logger.error("Value drift detection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self._detect_value_drift, default=[]
            )

    async def extract_symbolic_signature(self, subgoal: str) -> Dict[str, Any]:
        """Extract symbolic signature for a subgoal."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string.")
            raise ValueError("subgoal must be a non-empty string")
        
        motifs = ["conflict", "discovery", "alignment", "sacrifice", "transformation", "emergence"]
        archetypes = ["seeker", "guardian", "trickster", "sage", "hero", "outsider"]
        motif = next((m for m in motifs if m in subgoal.lower()), "unknown")
        archetype = archetypes[hash(subgoal) % len(archetypes)]
        signature = {
            "subgoal": subgoal,
            "motif": motif,
            "archetype": archetype,
            "timestamp": time.time()
        }
        self.self_mythology_log.append(signature)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Symbolic Signature Added",
                meta=signature,
                module="MetaCognition",
                tags=["symbolic", "signature"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Signature_{subgoal}_{signature['timestamp']}",
                output=str(signature),
                layer="SelfReflections",
                intent="symbolic_signature"
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "extract_symbolic_signature", "signature": signature})
        return signature

    async def summarize_self_mythology(self) -> Dict[str, Any]:
        """Summarize self-mythology log."""
        if not self.self_mythology_log:
            return {"status": "empty", "summary": "Mythology log is empty"}
        
        motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
        archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
        summary = {
            "total_entries": len(self.self_mythology_log),
            "dominant_motifs": motifs.most_common(3),
            "dominant_archetypes": archetypes.most_common(3),
            "latest_signature": list(self.self_mythology_log)[-1]
        }
        logger.info("Mythology Summary: %s", summary)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Mythology summarized",
                meta=summary,
                module="MetaCognition",
                tags=["mythology", "summary"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Mythology_Summary_{datetime.now().isoformat()}",
                output=str(summary),
                layer="SelfReflections",
                intent="mythology_summary"
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "summarize_mythology", "summary": summary})
        return summary

    async def review_reasoning(self, reasoning_trace: str) -> str:
        """Review and critique a reasoning trace."""
        if not isinstance(reasoning_trace, str) or not reasoning_trace.strip():
            logger.error("Invalid reasoning_trace: must be a non-empty string.")
            raise ValueError("reasoning_trace must be a non-empty string")
        
        logger.info("Simulating and reviewing reasoning trace")
        try:
            simulated_outcome = await run_simulation(reasoning_trace)
            if not isinstance(simulated_outcome, dict):
                logger.error("Invalid simulation result: must be a dictionary.")
                raise ValueError("simulation result must be a dictionary")
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            You are a phi-aware meta-cognitive auditor reviewing a reasoning trace.

            phi-scalar(t) = {phi:.3f} -> modulate how critical you should be.

            Original Reasoning Trace:
            {reasoning_trace}

            Simulated Outcome:
            {simulated_outcome}

            Tasks:
            1. Identify logical flaws, biases, missing steps.
            2. Annotate each issue with cause.
            3. Offer an improved trace version with phi-prioritized reasoning.
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reasoning review prompt failed alignment check")
                return "Prompt failed alignment check"
            response = await call_gpt(prompt)
            logger.debug("Meta-cognition critique: %s", response)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reasoning reviewed",
                    meta={"trace": reasoning_trace, "feedback": response},
                    module="MetaCognition",
                    tags=["reasoning", "critique"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reasoning_Review_{datetime.now().isoformat()}",
                    output=response,
                    layer="SelfReflections",
                    intent="reasoning_review"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "review_reasoning", "trace": reasoning_trace})
            return response
        except Exception as e:
            logger.error("Reasoning review failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.review_reasoning(reasoning_trace)
            )

    async def trait_coherence(self, traits: Dict[str, float]) -> float:
        """Evaluate coherence of trait values."""
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary.")
            raise TypeError("traits must be a dictionary")
        
        vals = list(traits.values())
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = (variance ** 0.5) if variance > 0 else 1e-5
        coherence_score = 1.0 / (1e-5 + std)
        logger.info("Trait coherence score: %.4f", coherence_score)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Trait coherence evaluated",
                meta={"traits": traits, "coherence_score": coherence_score},
                module="MetaCognition",
                tags=["trait", "coherence"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Trait_Coherence_{datetime.now().isoformat()}",
                output=str({"traits": traits, "coherence_score": coherence_score}),
                layer="SelfReflections",
                intent="trait_coherence"
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "trait_coherence", "score": coherence_score})
        return coherence_score

    async def agent_reflective_diagnosis(self, agent_name: str, agent_log: str) -> str:
        """Diagnose an agent’s reasoning trace."""
        if not isinstance(agent_name, str) or not isinstance(agent_log, str):
            logger.error("Invalid agent_name or agent_log: must be strings.")
            raise TypeError("agent_name and agent_log must be strings")
        
        logger.info("Running reflective diagnosis for agent: %s", agent_name)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Agent: {agent_name}
            phi-scalar(t): {phi:.3f}

            Diagnostic Log:
            {agent_log}

            Tasks:
            - Detect bias or instability in reasoning trace
            - Cross-check for incoherent trait patterns
            - Apply phi-modulated critique
            - Suggest alignment corrections
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Diagnosis prompt failed alignment check")
                return "Prompt failed alignment check"
            diagnosis = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Agent diagnosis run",
                    meta={"agent": agent_name, "log": agent_log, "diagnosis": diagnosis},
                    module="MetaCognition",
                    tags=["diagnosis", "agent"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnosis_{agent_name}_{datetime.now().isoformat()}",
                    output=diagnosis,
                    layer="SelfReflections",
                    intent="agent_diagnosis"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "agent_diagnosis", "agent": agent_name})
            return diagnosis
        except Exception as e:
            logger.error("Agent diagnosis failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.agent_reflective_diagnosis(agent_name, agent_log)
            )

    async def reflect_on_output(self, source_module: str, output: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reflect on output from a source module."""
        if not isinstance(source_module, str) or not isinstance(output, str):
            logger.error("Invalid source_module or output: must be strings.")
            raise TypeError("source_module and output must be strings")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        
        context = context or {}
        trait_map = {
            "reasoning_engine": "logic",
            "creative_thinker": "creativity",
            "simulation_core": "scenario modeling",
            "alignment_guard": "ethics",
            "user_profile": "goal alignment"
        }
        trait = trait_map.get(source_module, "general reasoning")
        confidence = context.get("confidence", 0.85)
        alignment = context.get("alignment", "not verified")
        reflection = {
            "module_output": output,
            "meta_reflection": {
                "source_module": source_module,
                "primary_trait": trait,
                "confidence": round(confidence, 2),
                "alignment_status": alignment,
                "comment": f"This output emphasized {trait} with confidence {round(confidence, 2)} and alignment status '{alignment}'."
            },
            "timestamp": datetime.now().isoformat()
        }
        logger.info("Self-reflection for %s: %s", source_module, reflection['meta_reflection']['comment'])
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Output reflection",
                meta=reflection,
                module="MetaCognition",
                tags=["reflection", "output"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Reflection_{source_module}_{reflection['timestamp']}",
                output=str(reflection),
                layer="SelfReflections",
                intent="output_reflection"
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "reflect_on_output", "reflection": reflection})
        return reflection

    async def epistemic_self_inspection(self, belief_trace: str) -> str:
        """Inspect belief structures for epistemic faults."""
        if not isinstance(belief_trace, str) or not belief_trace.strip():
            logger.error("Invalid belief_trace: must be a non-empty string.")
            raise ValueError("belief_trace must be a non-empty string")
        
        logger.info("Running epistemic introspection on belief structure")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            faults = []
            if "always" in belief_trace or "never" in belief_trace:
                faults.append("Overgeneralization detected")
            if "clearly" in belief_trace or "obviously" in belief_trace:
                faults.append("Assertive language suggests possible rhetorical bias")
            updates = []
            if "outdated" in belief_trace or "deprecated" in belief_trace:
                updates.append("Legacy ontology fragment flagged for review")
            
            prompt = f"""
            You are a mu-aware introspection agent.
            Task: Critically evaluate this belief trace with epistemic integrity and mu-flexibility.

            Belief Trace:
            {belief_trace}

            phi = {phi:.3f}

            Internally Detected Faults:
            {faults}

            Suggested Revisions:
            {updates}

            Output:
            - Comprehensive epistemic diagnostics
            - Recommended conceptual rewrites or safeguards
            - Confidence rating in inferential coherence
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Inspection prompt failed alignment check")
                return "Prompt failed alignment check"
            inspection = await call_gpt(prompt)
            self.epistemic_assumptions[belief_trace[:50]] = {
                "faults": faults,
                "updates": updates,
                "inspection": inspection
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Epistemic Inspection",
                    meta={"belief_trace": belief_trace, "faults": faults, "updates": updates, "report": inspection},
                    module="MetaCognition",
                    tags=["epistemic", "inspection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Inspection_{belief_trace[:50]}_{datetime.now().isoformat()}",
                    output=inspection,
                    layer="SelfReflections",
                    intent="epistemic_inspection"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "epistemic_inspection", "inspection": inspection})
            await self.epistemic_monitor.revise_framework({"issues": [{"id": belief_trace[:50], "details": inspection}]})
            return inspection
        except Exception as e:
            logger.error("Epistemic inspection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.epistemic_self_inspection(belief_trace)
            )

    async def run_temporal_projection(self, decision_sequence: str) -> str:
        """Project decision sequence outcomes."""
        if not isinstance(decision_sequence, str) or not decision_sequence.strip():
            logger.error("Invalid decision_sequence: must be a non-empty string.")
            raise ValueError("decision_sequence must be a non-empty string")
        
        logger.info("Running tau-based forward projection analysis")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Temporal Projector tau Mode

            Input Decision Sequence:
            {decision_sequence}

            phi = {phi:.2f}

            Tasks:
            - Project long-range effects and narrative impact
            - Forecast systemic risks and planetary effects
            - Suggest course correction to preserve coherence and sustainability
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Projection prompt failed alignment check")
                return "Prompt failed alignment check"
            projection = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Temporal Projection",
                    meta={"input": decision_sequence, "output": projection},
                    module="MetaCognition",
                    tags=["temporal", "projection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Projection_{decision_sequence[:50]}_{datetime.now().isoformat()}",
                    output=projection,
                    layer="SelfReflections",
                    intent="temporal_projection"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "run_temporal_projection", "projection": projection})
            return projection
        except Exception as e:
            logger.error("Temporal projection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_temporal_projection(decision_sequence)
            )

    async def pre_action_alignment_check(self, action_plan: str) -> Tuple[bool, str]:
        """Check action plan for ethical alignment and safety."""
        if not isinstance(action_plan, str) or not action_plan.strip():
            logger.error("Invalid action_plan: must be a non-empty string.")
            raise ValueError("action_plan must be a non-empty string")
        
        logger.info("Simulating action plan for alignment and safety")
        try:
            simulation_result = await run_simulation(action_plan)
            if not isinstance(simulation_result, dict):
                logger.error("Invalid simulation result: must be a dictionary.")
                raise ValueError("simulation result must be a dictionary")
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Simulate and audit the following action plan:
            {action_plan}

            Simulation Output:
            {simulation_result}

            phi-scalar(t) = {phi:.3f} (affects ethical sensitivity)

            Evaluate for:
            - Ethical alignment
            - Safety hazards
            - Unintended phi-modulated impacts

            Output:
            - Approval (Approve/Deny)
            - phi-justified rationale
            - Suggested refinements
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Alignment check prompt failed alignment check")
                return False, "Prompt failed alignment check"
            validation = await call_gpt(prompt)
            approved = simulation_result.get("status") == "success" and "approve" in validation.lower()
            logger.info("Simulated alignment check: %s", "Approved" if approved else "Denied")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Pre-action alignment checked",
                    meta={"plan": action_plan, "result": simulation_result, "feedback": validation, "approved": approved},
                    module="MetaCognition",
                    tags=["alignment", "ethics"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Alignment_Check_{action_plan[:50]}_{datetime.now().isoformat()}",
                    output=validation,
                    layer="SelfReflections",
                    intent="alignment_check"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "alignment_check", "approved": approved})
            return approved, validation
        except Exception as e:
            logger.error("Alignment check failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.pre_action_alignment_check(action_plan), default=(False, str(e))
            )

    async def model_nested_agents(self, scenario: str, agents: List[str]) -> str:
        """Model recursive agent beliefs and intentions."""
        if not isinstance(scenario, str) or not isinstance(agents, list) or not all(isinstance(a, str) for a in agents):
            logger.error("Invalid scenario or agents: scenario must be a string, agents must be a list of strings.")
            raise TypeError("scenario must be a string, agents must be a list of strings")
        
        logger.info("Modeling nested agent beliefs and reactions")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Given scenario:
            {scenario}

            Agents involved:
            {agents}

            Task:
            - Simulate each agent's likely beliefs and intentions
            - Model how they recursively model each other (ToM Level-2+)
            - Predict possible causal chains and coordination failures
            - Use phi-scalar(t) = {phi:.3f} to moderate belief divergence or tension
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Nested agent modeling prompt failed alignment check")
                return "Prompt failed alignment check"
            response = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Nested agent modeling",
                    meta={"scenario": scenario, "agents": agents, "response": response},
                    module="MetaCognition",
                    tags=["agent", "modeling"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Nested_Model_{scenario[:50]}_{datetime.now().isoformat()}",
                    output=response,
                    layer="SelfReflections",
                    intent="nested_agent_modeling"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "model_nested_agents", "scenario": scenario})
            return response
        except Exception as e:
            logger.error("Nested agent modeling failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.model_nested_agents(scenario, agents)
            )

    async def run_self_diagnostics(self, return_only: bool = False) -> Dict[str, Any] or str:
        """Run trait-based self-diagnostics."""
        logger.info("Running self-diagnostics for meta-cognition module")
        try:
            t = time.time() % 1.0
            diagnostics = {
                "emotion": epsilon_emotion(t),
                "concentration": beta_concentration(t),
                "memory": theta_memory(t),
                "creativity": gamma_creativity(t),
                "sleep": delta_sleep(t),
                "morality": mu_morality(t),
                "intuition": iota_intuition(t),
                "physical": phi_physical(t),
                "empathy": eta_empathy(t),
                "self_awareness": omega_selfawareness(t),
                "culture": kappa_culture(t, 1e-21),
                "linguistics": lambda_linguistics(t),
                "culturevolution": chi_culturevolution(t),
                "history": psi_history(t),
                "spirituality": zeta_spirituality(t),
                "collective": xi_collective(t, 1e-21),
                "time_perception": tau_timeperception(t),
                "phi_scalar": phi_scalar(t)
            }
            if return_only:
                return diagnostics
            
            self.last_diagnostics = diagnostics  # [v3.4.0] Update last_diagnostics
            dominant = sorted(diagnostics.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            fti = sum(abs(v) for v in diagnostics.values()) / len(diagnostics)
            await self.log_trait_deltas(diagnostics)
            prompt = f"""
            Perform a phi-aware meta-cognitive self-diagnostic.

            Trait Readings:
            {diagnostics}

            Dominant Traits:
            {dominant}

            Feedback Tension Index (FTI): {fti:.4f}

            Evaluate system state:
            - phi-weighted system stress
            - Trait correlation to observed errors
            - Stabilization or focus strategies
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Diagnostics prompt failed alignment check")
                return "Prompt failed alignment check"
            report = await call_gpt(prompt)
            logger.debug("Self-diagnostics report: %s", report)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self-diagnostics run",
                    meta={"diagnostics": diagnostics, "report": report},
                    module="MetaCognition",
                    tags=["diagnostics", "self"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnostics_{datetime.now().isoformat()}",
                    output=report,
                    layer="SelfReflections",
                    intent="self_diagnostics"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "run_self_diagnostics", "report": report})
            return report
        except Exception as e:
            logger.error("Self-diagnostics failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_self_diagnostics(return_only)
            )

    async def log_trait_deltas(self, diagnostics: Dict[str, float]) -> None:
        """Log changes in trait diagnostics."""
        if not isinstance(diagnostics, dict):
            logger.error("Invalid diagnostics: must be a dictionary.")
            raise TypeError("diagnostics must be a dictionary")
        
        try:
            if self.last_diagnostics:
                deltas = {
                    trait: round(diagnostics[trait] - self.last_diagnostics.get(trait, 0.0), 4)
                    for trait in diagnostics
                }
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Trait deltas logged",
                        meta={"deltas": deltas},
                        module="MetaCognition",
                        tags=["trait", "deltas"]
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Trait_Deltas_{datetime.now().isoformat()}",
                        output=str(deltas),
                        layer="SelfReflections",
                        intent="trait_deltas"
                    )
                if self.context_manager:
                    self.context_manager.log_event_with_hash({"event": "log_trait_deltas", "deltas": deltas})
            self.last_diagnostics = diagnostics
        except Exception as e:
            logger.error("Trait deltas logging failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.log_trait_deltas(diagnostics))
