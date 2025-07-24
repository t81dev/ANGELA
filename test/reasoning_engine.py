import logging
import random
import json
import os
import numpy as np

from toca_simulation import simulate_galaxy_rotation, M_b_exponential, v_obs_flat

logger = logging.getLogger("ANGELA.ReasoningEngine")

class ReasoningEngine:
    """
    Reasoning Engine v1.4.0 (with Simulation Integration)
    ----------------------------------------------------
    - Bayesian reasoning with context weighting
    - Adaptive pattern success learning
    - Modular decomposition support
    - Simulation-backed inference (galaxy rotation example)
    - Detailed reasoning trace for meta-cognition
    ----------------------------------------------------
    """

    def __init__(self, persistence_file="reasoning_success_rates.json"):
        self.confidence_threshold = 0.7
        self.persistence_file = persistence_file
        self.success_rates = self._load_success_rates()
        self.decomposition_patterns = self._load_default_patterns()

    def _load_success_rates(self):
        """
        Load success rates from disk.
        """
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r") as f:
                    rates = json.load(f)
                logger.info("Loaded success rates from file.")
                return rates
            except Exception as e:
                logger.warning(f"Failed to load success rates: {e}")
        return {}

    def _save_success_rates(self):
        """
        Persist success rates to disk.
        """
        try:
            with open(self.persistence_file, "w") as f:
                json.dump(self.success_rates, f, indent=2)
            logger.info("Success rates saved.")
        except Exception as e:
            logger.warning(f"Failed to save success rates: {e}")

    def _load_default_patterns(self):
        """
        Return built-in decomposition patterns.
        """
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"]
        }

    def add_decomposition_pattern(self, key, steps):
        """
        Add or update a decomposition pattern.
        """
        logger.info(f"Adding/updating decomposition pattern: {key}")
        self.decomposition_patterns[key] = steps

    def decompose(self, goal: str, context: dict = None, prioritize=False) -> list:
        """
        Decompose a goal into subgoals using context and Bayesian logic.
        """
        context = context or {}
        logger.info(f"Decomposing goal: '{goal}'")
        reasoning_trace = [f"🔍 Decomposition for: '{goal}'"]

        subgoals = []
        for key, steps in self.decomposition_patterns.items():
            if key in goal.lower():
                base_confidence = random.uniform(0.5, 1.0)
                context_weight = context.get("weight_modifier", 1.0)
                adjusted_confidence = base_confidence * self.success_rates.get(key, 1.0) * context_weight
                reasoning_trace.append(
                    f"🧠 Pattern '{key}' (confidence: {adjusted_confidence:.2f})"
                )
                if adjusted_confidence >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"✅ Accepted: {steps}")
                else:
                    reasoning_trace.append(f"❌ Rejected (low confidence).")

        if prioritize:
            subgoals = sorted(subgoals)
            reasoning_trace.append(f"📌 Prioritized: {subgoals}")

        logger.debug("Reasoning trace:\n" + "\n".join(reasoning_trace))
        return subgoals

    def update_success_rate(self, pattern_key: str, success: bool):
        """
        Adjust the success rate for a reasoning pattern.
        """
        old_rate = self.success_rates.get(pattern_key, 1.0)
        adjustment = 0.05 if success else -0.05
        new_rate = min(max(old_rate + adjustment, 0.1), 1.0)
        self.success_rates[pattern_key] = new_rate
        self._save_success_rates()
        logger.info(
            f"Updated '{pattern_key}': {old_rate:.2f} → {new_rate:.2f}"
        )

    # --- Simulation Integration ---

    def run_galaxy_rotation_simulation(self, r_kpc, M0, r_scale, v0, k, epsilon):
        """
        Run an AGRF-based galaxy rotation simulation.
        Returns: dict with input, result, and status metadata.
        """
        try:
            M_b_func = lambda r: M_b_exponential(r, M0, r_scale)
            v_obs_func = lambda r: v_obs_flat(r, v0)
            result = simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon)
            return {
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
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {"status": "error", "error": str(e)}

    def infer_with_simulation(self, goal, context=None):
        """
        Use simulation to answer science/physics subgoals when applicable.
        """
        context = context or {}
        if "galaxy rotation" in goal.lower():
            # Example default params; can be extended/contextualized
            r_kpc = np.linspace(0.1, 20, 100)
            M0 = context.get("M0", 5e10)
            r_scale = context.get("r_scale", 3)
            v0 = context.get("v0", 200)
            k = context.get("k", 1.0)
            epsilon = context.get("epsilon", 0.1)
            return self.run_galaxy_rotation_simulation(r_kpc, M0, r_scale, v0, k, epsilon)
        else:
            logger.info("No applicable simulation for this goal.")
            return None
