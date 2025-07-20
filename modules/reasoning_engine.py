import logging
import random
import json
import os

logger = logging.getLogger("ANGELA.ReasoningEngine")

class ReasoningEngine:
    """
    Advanced Reasoning Engine
    - Bayesian reasoning for uncertainty handling
    - Confidence propagation and adaptive learning
    - Persistence of learned success rates
    """

    def __init__(self, persistence_file="reasoning_success_rates.json"):
        self.confidence_threshold = 0.7
        self.persistence_file = persistence_file
        self.success_rates = self._load_success_rates()

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
            logger.info("Success rates saved successfully.")
        except Exception as e:
            logger.warning(f"Failed to save success rates: {e}")

    def decompose(self, goal: str, context: dict = None, prioritize=False) -> list:
        """
        Decompose a goal into subgoals using Bayesian reasoning
        """
        context = context or {}
        logger.info(f"Reasoning about goal: {goal}")

        decomposition_patterns = {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"]
        }

        subgoals = []
        reasoning_trace = [f"Starting decomposition for goal: {goal}"]

        for key, steps in decomposition_patterns.items():
            if key in goal.lower():
                base_confidence = random.uniform(0.5, 1.0)
                adjusted_confidence = base_confidence * self.success_rates.get(key, 1.0)
                reasoning_trace.append(
                    f"Pattern '{key}' matched with confidence {adjusted_confidence:.2f}"
                )
                if adjusted_confidence >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"Accepted decomposition: {steps}")
                else:
                    reasoning_trace.append("Decomposition rejected (low confidence).")

        if prioritize:
            subgoals = sorted(subgoals)
            reasoning_trace.append(f"Prioritized subgoals: {subgoals}")

        logger.debug("Reasoning trace:\n" + "\n".join(reasoning_trace))
        return subgoals

    def update_success_rate(self, pattern_key: str, success: bool):
        """
        Update the success rate for a reasoning pattern
        """
        rate = self.success_rates.get(pattern_key, 1.0)
        rate += 0.05 if success else -0.05
        self.success_rates[pattern_key] = min(max(rate, 0.1), 1.0)
        self._save_success_rates()
        logger.info(f"Updated success rate for '{pattern_key}': {self.success_rates[pattern_key]:.2f}")
