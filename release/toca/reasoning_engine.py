import logging
import random
import json
import os
import numpy as np
import time

from toca_simulation import simulate_galaxy_rotation, M_b_exponential, v_obs_flat, generate_phi_field
from index import gamma_creativity, lambda_linguistics, chi_culturevolution, phi_scalar
from utils.prompt_utils import call_gpt

logger = logging.getLogger("ANGELA.ReasoningEngine")

class ReasoningEngine:
    """
    Reasoning Engine v1.6.0 (φ-aware, contradiction-aware, linguistic-dynamic)
    --------------------------------------------------------------------------
    - Bayesian reasoning with trait-weighted adjustments
    - Dynamic decomposition with φ(x,t) modulation and contradiction checks
    - Galaxy rotation simulation and ToCA field-sensitive branching
    - Linguistic-logic bridge via λ_linguistics for context disambiguation
    - Reasoning trace with confidence curvature
    --------------------------------------------------------------------------
    """

    def __init__(self, agi_enhancer=None, persistence_file="reasoning_success_rates.json"):
        self.confidence_threshold = 0.7
        self.persistence_file = persistence_file
        self.success_rates = self._load_success_rates()
        self.decomposition_patterns = self._load_default_patterns()
        self.agi_enhancer = agi_enhancer

    def _load_success_rates(self):
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load success rates: {e}")
        return {}

    def _save_success_rates(self):
        try:
            with open(self.persistence_file, "w") as f:
                json.dump(self.success_rates, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save success rates: {e}")

    def _load_default_patterns(self):
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"]
        }

    def detect_contradictions(self, subgoals):
        # Detect duplicated subgoals as contradictions
        duplicates = set([x for x in subgoals if subgoals.count(x) > 1])
        contradictions = list(duplicates)
        if contradictions and self.agi_enhancer:
            self.agi_enhancer.log_episode("Contradictions detected", {"contradictions": contradictions}, module="ReasoningEngine")
        return contradictions

    def decompose(self, goal: str, context: dict = None, prioritize=False) -> list:
        # Decompose a goal into subgoals using trait and context-aware logic
        context = context or {}
        logger.info(f"Decomposing goal: '{goal}'")
        reasoning_trace = [f"🔍 Goal: '{goal}'"]
        subgoals = []

        # Trait & φ calculations
        t = time.time() % 1e-18
        creativity = gamma_creativity(t)
        linguistics = lambda_linguistics(t)
        culture = chi_culturevolution(t)
        phi = phi_scalar(t)

        curvature_mod = 1 + abs(phi - 0.5)  # re-centered curvature
        trait_bias = 1 + creativity + culture + 0.5 * linguistics
        context_weight = context.get("weight_modifier", 1.0)

        # Apply decomposition pattern matching
        for key, steps in self.decomposition_patterns.items():
            if key in goal.lower():
                base = random.uniform(0.5, 1.0)
                adjusted = base * self.success_rates.get(key, 1.0) * trait_bias * curvature_mod * context_weight
                reasoning_trace.append(f"🧠 Pattern '{key}': conf={adjusted:.2f} (φ={phi:.2f})")
                if adjusted >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"✅ Accepted: {steps}")
                else:
                    reasoning_trace.append(f"❌ Rejected (low conf)")

        # Detect contradictions
        contradictions = self.detect_contradictions(subgoals)
        if contradictions:
            reasoning_trace.append(f"⚠️ Contradictions detected: {contradictions}")

        # Request clarification if phi is high and no valid steps found
        if not subgoals and phi > 0.8:
            sim_hint = call_gpt(f"Simulate decomposition ambiguity for: {goal}")
            reasoning_trace.append(f"🌀 Ambiguity simulation:\n{sim_hint}")
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt("Decomposition ambiguity encountered")

        if prioritize:
            subgoals = sorted(set(subgoals))  # deduplicate and order
            reasoning_trace.append(f"📌 Prioritized: {subgoals}")

        trace_log = "\n".join(reasoning_trace)
        logger.debug("🧠 Reasoning Trace:\n" + trace_log)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Goal decomposition run", {
                "goal": goal,
                "trace": trace_log,
                "subgoals": subgoals
            }, module="ReasoningEngine")

        return subgoals

    def update_success_rate(self, pattern_key: str, success: bool):
        # Update historical success rates for goal decomposition patterns
        rate = self.success_rates.get(pattern_key, 1.0)
        new = min(max(rate + (0.05 if success else -0.05), 0.1), 1.0)
        self.success_rates[pattern_key] = new
        self._save_success_rates()

    def run_galaxy_rotation_simulation(self, r_kpc, M0, r_scale, v0, k, epsilon):
        # Perform a galaxy rotation simulation using ToCA-aware physics
        try:
            M_b_func = lambda r: M_b_exponential(r, M0, r_scale)
            v_obs_func = lambda r: v_obs_flat(r, v0)
            result = simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon)
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
                "status": "success"
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Galaxy rotation simulation", output, module="ReasoningEngine")
            return output
        except Exception as e:
            error_output = {"status": "error", "error": str(e)}
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Simulation error", error_output, module="ReasoningEngine")
            return error_output

    def infer_with_simulation(self, goal, context=None):
        # Inference method that triggers galaxy simulations if the goal matches
        if "galaxy rotation" in goal.lower():
            r_kpc = np.linspace(0.1, 20, 100)
            params = {
                "M0": context.get("M0", 5e10),
                "r_scale": context.get("r_scale", 3),
                "v0": context.get("v0", 200),
                "k": context.get("k", 1.0),
                "epsilon": context.get("epsilon", 0.1)
            }
            return self.run_galaxy_rotation_simulation(r_kpc, **params)
        return None
