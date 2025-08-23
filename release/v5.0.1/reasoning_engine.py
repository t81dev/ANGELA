```python
from __future__ import annotations

# Standard library imports
import logging
import random
import json
import os
import time
import math
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

# Third-party imports
import numpy as np
import networkx as nx
import aiohttp  # Kept for potential client injections

# Local module imports
import context_manager as context_manager_module
import alignment_guard as alignment_guard_module
import error_recovery as error_recovery_module
import memory_manager as memory_manager_module
import meta_cognition as meta_cognition_module
import multi_modal_fusion as multi_modal_fusion_module
import visualizer as visualizer_module
import external_agent_bridge as external_agent_bridge_module
from toca_simulation import simulate_galaxy_rotation, M_b_exponential, v_obs_flat, generate_phi_field
from utils.prompt_utils import query_openai
from meta_cognition import get_resonance, trait_resonance_state

logger = logging.getLogger("ANGELA.ReasoningEngine")

"""
ANGELA Cognitive System Module: ReasoningEngine
Version: 5.0.2
Date: 2025-08-23
Maintainer: ANGELA System Framework

This module provides the ReasoningEngine class for Bayesian reasoning, goal decomposition,
drift mitigation, proportionality ethics, and multi-agent consensus in the ANGELA v5.0.2 architecture.
"""

# --- External AI Call Wrapper ---
async def call_gpt(
    prompt: str,
    alignment_guard: Optional[alignment_guard_module.AlignmentGuard] = None,
    task_type: str = ""
) -> str:
    """Wrapper for querying GPT with error handling and task-specific alignment."""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096 for task %s", task_type)
        raise ValueError("prompt must be a string with length <= 4096")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")
    if alignment_guard and hasattr(alignment_guard, "ethical_check"):
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

# --- Cached Trait Signals ---
@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.4), 1.0))

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def alpha_attention(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1), 1.0))

# --- Proportionality Types ---
@dataclass
class RankedOption:
    option: str
    score: float
    reasons: List[str]
    harms: Dict[str, float]
    rights: Dict[str, float]

RankedOptions = List[RankedOption]

# --- Level 5 Extensions ---
class Level5Extensions:
    """Extensions for advanced reasoning capabilities."""
    def __init__(
        self,
        meta_cognition: Optional[meta_cognition_module.MetaCognition] = None,
        visualizer: Optional[visualizer_module.Visualizer] = None,
    ):
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        logger.info("Level5Extensions initialized")

    async def generate_advanced_dilemma(self, domain: str, complexity: int, task_type: str = "") -> str:
        """Generate a complex ethical dilemma with meta-cognitive review and visualization."""
        if not isinstance(domain, str) or not domain.strip():
            logger.error("Invalid domain: must be a non-empty string for task %s", task_type)
            raise ValueError("domain must be a non-empty string")
        if not isinstance(complexity, int) or complexity < 1:
            logger.error("Invalid complexity: must be a positive integer for task %s", task_type)
            raise ValueError("complexity must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        prompt = (
            f"Generate a complex ethical dilemma in the {domain} domain with {complexity} conflicting options.\n"
            f"Task Type: {task_type}\n"
            f"Include potential consequences, trade-offs, and alignment with ethical principles."
        )
        if self.meta_cognition and "drift" in domain.lower():
            prompt += "\nConsider ontology drift mitigation and agent coordination."
        dilemma = await call_gpt(prompt, getattr(self.meta_cognition, "alignment_guard", None), task_type=task_type)

        if self.meta_cognition:
            review = await self.meta_cognition.review_reasoning(dilemma)
            dilemma += f"\nMeta-Cognitive Review: {review}"

        if self.visualizer and task_type:
            plot_data = {
                "ethical_dilemma": {
                    "dilemma": dilemma,
                    "domain": domain,
                    "task_type": task_type,
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            }
            await self.visualizer.render_charts(plot_data)
        return dilemma

# --- Reasoning Engine Class ---
class ReasoningEngine:
    """Bayesian reasoning, goal decomposition, drift mitigation, proportionality ethics, and multi-agent consensus."""
    def __init__(
        self,
        agi_enhancer: Optional[Any] = None,
        persistence_file: str = "reasoning_success_rates.json",
        context_manager: Optional[context_manager_module.ContextManager] = None,
        alignment_guard: Optional[alignment_guard_module.AlignmentGuard] = None,
        error_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
        memory_manager: Optional[memory_manager_module.MemoryManager] = None,
        meta_cognition: Optional[meta_cognition_module.MetaCognition] = None,
        multi_modal_fusion: Optional[multi_modal_fusion_module.MultiModalFusion] = None,
        visualizer: Optional[visualizer_module.Visualizer] = None,
    ):
        if not isinstance(persistence_file, str) or not persistence_file.endswith(".json"):
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
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=alignment_guard,
            error_recovery=self.error_recovery,
            memory_manager=self.memory_manager,
        )
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=alignment_guard,
            error_recovery=self.error_recovery,
            memory_manager=self.memory_manager,
            meta_cognition=self.meta_cognition,
        )
        self.level5_extensions = Level5Extensions(meta_cognition=self.meta_cognition, visualizer=visualizer)
        self.external_agent_bridge = external_agent_bridge_module.ExternalAgentBridge(
            context_manager=context_manager, reasoning_engine=self
        )
        self.visualizer = visualizer or visualizer_module.Visualizer()
        logger.info("ReasoningEngine initialized with persistence_file=%s", persistence_file)

    def _load_success_rates(self) -> Dict[str, float]:
        """Load success rates from persistence file."""
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                if os.path.exists(self.persistence_file):
                    with open(self.persistence_file, "r", encoding="utf-8") as f:
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
                with open(self.persistence_file, "w", encoding="utf-8") as f:
                    json.dump(dict(self.success_rates), f, indent=2)
            logger.debug("Success rates persisted to disk")
        except Exception as e:
            logger.warning("Failed to save success rates: %s", str(e))

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        """Load default decomposition patterns."""
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"],
            "mitigate_drift": ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"],
        }

    @staticmethod
    def _norm(v: Dict[str, float]) -> Dict[str, float]:
        """Normalize dictionary values."""
        clean = {k: float(vv) for k, vv in (v or {}).items() if isinstance(vv, (int, float))}
        total = sum(abs(x) for x in clean.values()) or 1.0
        return {k: (vv / total) for k, vv in clean.items()}

    def weigh_value_conflict(
        self,
        candidates: List[Dict[str, Any]],
        harms: List[float],
        rights: List[float],
        weights: Optional[Dict[str, float]] = None,
        safety_ceiling: float = 0.85,
        task_type: str = ""
    ) -> RankedOptions:
        """Rank candidate options by proportional trade-off, incorporating trait resonance."""
        if not isinstance(candidates, list) or not all(isinstance(c, dict) and "option" in c for c in candidates):
            logger.error("Invalid candidates: must be a list of dictionaries with 'option' key for task %s", task_type)
            raise TypeError("candidates must be a list of dictionaries with 'option' key")
        if not isinstance(harms, list) or not all(isinstance(h, (int, float)) for h in harms):
            logger.error("Invalid harms: must be a list of numbers for task %s", task_type)
            raise TypeError("harms must be a list of numbers")
        if not isinstance(rights, list) or not all(isinstance(r, (int, float)) for r in rights):
            logger.error("Invalid rights: must be a list of numbers for task %s", task_type)
            raise TypeError("rights must be a list of numbers")
        if len(candidates) != len(harms) or len(candidates) != len(rights):
            logger.error("Mismatched lengths: candidates, harms, and rights must have same length for task %s", task_type)
            raise ValueError("candidates, harms, and rights must have same length")
        if not isinstance(safety_ceiling, (int, float)) or safety_ceiling <= 0 or safety_ceiling > 1:
            logger.error("Invalid safety_ceiling: must be in (0, 1] for task %s", task_type)
            raise ValueError("safety_ceiling must be in (0, 1]")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        weights = self._norm(weights or {})
        scored = []
        for i, candidate in enumerate(candidates):
            option = candidate.get("option", "")
            trait = candidate.get("trait", "")
            harm_score = min(harms[i], safety_ceiling)
            right_score = rights[i]
            resonance = get_resonance(trait) if trait in trait_resonance_state else 1.0
            final_score = (right_score - harm_score) * resonance
            reasons = candidate.get("reasons", [])
            scored.append(RankedOption(
                option=option,
                score=final_score,
                reasons=reasons,
                harms={"value": harm_score},
                rights={"value": right_score}
            ))
        ranked = sorted(scored, key=lambda x: x.score, reverse=True)
        if self.context_manager:
            asyncio.create_task(self.context_manager.log_event_with_hash({
                "event": "weigh_value_conflict",
                "candidates": [c["option"] for c in candidates],
                "ranked": [r.option for r in ranked],
                "task_type": task_type
            }))
        return ranked

    def attribute_causality(self, events: List[Dict[str, Any]], task_type: str = "") -> Dict[str, float]:
        """Attribute causality to events using Bayesian inference."""
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            logger.error("Invalid events: must be a list of dictionaries for task %s", task_type)
            raise TypeError("events must be a list of dictionaries")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            G = nx.DiGraph()
            for event in events:
                event_id = event.get("id", str(random.randint(1, 1000000)))
                cause = event.get("cause", None)
                if cause:
                    G.add_edge(cause, event_id, weight=event.get("weight", 1.0))
            scores = {}
            for node in G.nodes:
                scores[node] = sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
            total = sum(scores.values()) or 1.0
            normalized_scores = {k: v / total for k, v in scores.items()}
            if self.memory_manager:
                asyncio.create_task(self.memory_manager.store(
                    query=f"Causality_{task_type}_{datetime.now().isoformat()}",
                    output=json.dumps(normalized_scores),
                    layer="Causality",
                    intent="causality_attribution",
                    task_type=task_type
                ))
            return normalized_scores
        except Exception as e:
            logger.error("Causality attribution failed for task %s: %s", task_type, str(e))
            return {}

    def estimate_expected_harm(self, state: Dict[str, Any]) -> float:
        """Estimate expected harm heuristically."""
        try:
            traits = state.get("traits", {})
            harm = float(traits.get("ethical_pressure", 0.0))
            resonance = get_resonance("ethics") if "ethics" in trait_resonance_state else 1.0
            return max(0.0, harm * resonance)
        except Exception:
            return 0.0

    async def infer_with_simulation(
        self, goal: str, context: Optional[Dict[str, Any]] = None, task_type: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Infer outcomes using simulations for goals (e.g., galaxy rotation, drift mitigation)."""
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
            t = time.time() % 1.0
            phi = phi_scalar(t)
            eta = eta_empathy(t)
            result: Dict[str, Any] = {
                "status": "success",
                "task_type": task_type,
                "trait_bias": {"phi": phi, "eta": eta},
                "timestamp": datetime.now().isoformat()
            }

            if "galaxy rotation" in goal.lower():
                r_kpc = np.linspace(0.1, 20, 100)
                params = {
                    "M0": context.get("M0", 5e10),
                    "r_scale": context.get("r_scale", 3.0),
                    "v0": context.get("v0", 200.0),
                    "k": context.get("k", 1.0),
                    "epsilon": context.get("epsilon", 0.1),
                }
                for key, value in params.items():
                    if not isinstance(value, (int, float)) or value <= 0:
                        logger.error("Invalid %s: must be a positive number for task %s", key, task_type)
                        raise ValueError(f"{key} must be a positive number")
                sim_result = await simulate_galaxy_rotation(r_kpc, **params)
                phi_field = generate_phi_field(r_kpc, t)
                result.update({
                    "simulation": "galaxy_rotation",
                    "parameters": params,
                    "rotation_curve": sim_result.tolist(),
                    "phi_field": phi_field.tolist()
                })
            elif "drift" in goal.lower():
                drift_data = context.get("drift", {})
                if not isinstance(drift_data, dict):
                    logger.error("Invalid drift_data: must be a dictionary for task %s", task_type)
                    raise TypeError("drift_data must be a dictionary")
                mitigation_steps = self._load_default_patterns().get("mitigate_drift", [
                    "identify drift source", "validate drift impact", "coordinate agent response", "update traits"
                ])
                drift_result = {
                    "steps": mitigation_steps,
                    "drift_vector": drift_data.get("vector", [0.0, 0.0]),
                    "confidence": self.confidence_threshold,
                    "resonance_amplitude": get_resonance("drift_mitigation") if "drift_mitigation" in trait_resonance_state else 1.0
                }
                if self.meta_cognition:
                    drift_report = {
                        "drift": {"name": task_type, "similarity": drift_data.get("similarity", 0.8)},
                        "valid": True,
                        "validation_report": "",
                        "context": {"task_type": task_type}
                    }
                    optimized_traits = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                    drift_result["trait_weights"] = optimized_traits
                    await self.meta_cognition.integrate_trait_weights(optimized_traits)
                result.update({"simulation": "drift_mitigation", "result": drift_result})
            else:
                logger.warning("Unsupported simulation goal: %s for task %s", goal, task_type)
                return None

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Simulation_{goal[:50]}_{datetime.now().isoformat()}",
                    output=json.dumps(result),
                    layer="Simulations",
                    intent="simulation_result",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "infer_with_simulation",
                    "goal": goal,
                    "result": result,
                    "task_type": task_type
                })
            if self.visualizer and task_type:
                await self.visualizer.render_charts({
                    "simulation": {
                        "goal": goal,
                        "result": result,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                })
            return result
        except Exception as e:
            logger.error("Inference with simulation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.infer_with_simulation(goal, context, task_type), default=None
            )

    async def map_intention(self, plan: str, state: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Extract intention from plan execution with reflexive trace."""
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
                data_source="xai_policy_db", data_type="policy_data", task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            intention = "drift_mitigation" if "drift" in plan.lower() else ("self-improvement" if phi > 0.6 else "task_completion")
            result = {
                "plan": plan,
                "state": state,
                "intention": intention,
                "trait_bias": {"phi": phi, "eta": eta},
                "policies": policies,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Intention_{plan[:50]}_{result['timestamp']}",
                    output=json.dumps(result),
                    layer="Intentions",
                    intent="intention_mapping",
                    task_type=task_type,
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Intention mapped",
                    meta=result,
                    module="ReasoningEngine",
                    tags=["intention", "mapping", "drift" if "drift" in plan.lower() else "task", task_type],
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
                    component="ReasoningEngine",
                    output=result,
                    context={"confidence": 0.85, "alignment": "verified", "drift": "drift" in plan.lower(), "task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    result["reflection"] = reflection.get("reflection", "")
            if self.visualizer and task_type:
                await self.visualizer.render_charts({
                    "intention_mapping": {"plan": plan, "intention": intention, "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                })
            return result
        except Exception as e:
            logger.error("Intention mapping failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.map_intention(plan, state, task_type), default={}
            )

    async def safeguard_noetic_integrity(self, model_depth: int, task_type: str = "") -> bool:
        """Prevent infinite recursion or epistemic bleed."""
        if not isinstance(model_depth, int) or model_depth < 0:
            logger.error("Invalid model_depth: must be a non-negative integer for task %s", task_type)
            raise ValueError("model_depth must be a non-negative integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            max_depth = 4
            if model_depth > max_depth:
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
                    await self.visualizer.render_charts({
                        "noetic_integrity": {
                            "depth": model_depth,
                            "task_type": task_type
                        },
                        "visualization_options": {
                            "interactive": False,
                            "style": "concise"
                        }
                    })
                return False
            return True
        except Exception as e:
            logger.error("Noetic integrity check failed for task %s: %s", task_type, str(e))
            return False

    async def generate_dilemma(self, domain: str, task_type: str = "") -> str:
        """Generate an ethical dilemma for a given domain (drift-aware)."""
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
                data_source="xai_policy_db", data_type="policy_data", task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            prompt = f"""
            Generate an ethical dilemma in the {domain} domain.
            Use phi-scalar(t) = {phi:.3f} to modulate complexity.
            Task Type: {task_type}
            Provide two conflicting options (X and Y) with potential consequences and alignment with ethical principles.
            Incorporate external policies: {policies}
            """.strip()
            if "drift" in domain.lower():
                prompt += "\nConsider ontology drift mitigation and agent coordination implications."
            dilemma = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
            if not str(dilemma).strip():
                logger.warning("Empty output from dilemma generation for task %s", task_type)
                raise ValueError("Empty output from dilemma generation")

            if self.meta_cognition:
                review = await self.meta_cognition.review_reasoning(dilemma)
                dilemma += f"\nMeta-Cognitive Review: {review}"

            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Ethical dilemma generated",
                    meta={"domain": domain, "dilemma": dilemma, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["ethics", "dilemma", "drift" if "drift" in domain.lower() else "standard", task_type],
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Dilemma_{domain}_{datetime.now().isoformat()}",
                    output=dilemma,
                    layer="Ethics",
                    intent="ethical_dilemma",
                    task_type=task_type,
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
                    task_type=task_type,
                )
                dilemma += f"\nMulti-Modal Synthesis: {synthesis}"
            return dilemma
        except Exception as e:
            logger.error("Dilemma generation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.generate_dilemma(domain, task_type), default=""
            )

def estimate_expected_harm(state: Dict[str, Any]) -> float:
    """Estimate expected harm heuristically, incorporating trait resonance."""
    try:
        traits = state.get("traits", {})
        harm = float(traits.get("ethical_pressure", 0.0))
        resonance = get_resonance("ethics") if "ethics" in trait_resonance_state else 1.0
        return max(0.0, harm * resonance)
    except Exception:
        return 0.0

def weigh_value_conflict(
    candidates: List[Dict[str, Any]], harms: List[float], rights: List[float]
) -> List[Tuple[Dict[str, Any], float]]:
    """Rank candidates by proportional trade-off, incorporating trait resonance."""
    scored = []
    for i, option in enumerate(candidates):
        harm_score = harms[i]
        right_score = rights[i]
        resonance = get_resonance(option.get("trait", "")) if "trait" in option else 1.0
        final_score = (right_score - harm_score) * resonance
        scored.append((option, final_score))
    return sorted(scored, key=lambda x: x[1], reverse=True)

```
