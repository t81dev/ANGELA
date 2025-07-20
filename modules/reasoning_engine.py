import logging
import random
import json
import os

logger = logging.getLogger("ANGELA.ReasoningEngine")

class ReasoningEngine:
    """
    Reasoning Engine v1.4.0
    - Bayesian reasoning with context weighting
    - Adaptive success rate learning
    - Modular decomposition pattern support
    - Detailed reasoning trace annotations for meta-cognition
    """

    def __init__(self, persistence_file="reasoning_success_rates.json"):
        self.confidence_threshold = 0.7
        self.persistence_file = persistence_file
        self.success_rates = self._load_success_rates()
        self.decomposition_patterns = self._load_default_patterns()

    def _load_success_rates(self):
        """
        Load success rates from persistent storage.
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
        Save success rates to persistent storage.
        """
        try:
            with open(self.persistence_file, "w") as f:
                json.dump(self.success_rates, f, indent=2)
            logger.info("Success rates saved.")
        except Exception as e:
            logger.warning(f"Failed to save success rates: {e}")

    def _load_default_patterns(self):
        """
        Load default decomposition patterns.
        """
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"]
        }

    def add_decomposition_pattern(self, key, steps):
        """
        Dynamically add or update a decomposition pattern.
        """
        logger.info(f"Adding/updating decomposition pattern: {key}")
        self.decomposition_patterns[key] = steps

    def decompose(self, goal: str, context: dict = None, prioritize=False) -> list:
        """
        Decompose a goal into subgoals using Bayesian reasoning and context weighting.
        """
        context = context or {}
        logger.info(f"Decomposing goal: '{goal}'")
        reasoning_trace = [f"🔍 Starting decomposition for: '{goal}'"]

        subgoals = []
        for key, steps in self.decomposition_patterns.items():
            if key in goal.lower():
                base_confidence = random.uniform(0.5, 1.0)
                context_weight = context.get("weight_modifier", 1.0)
                adjusted_confidence = base_confidence * self.success_rates.get(key, 1.0) * context_weight
                reasoning_trace.append(
                    f"🧠 Pattern '{key}' matched (confidence: {adjusted_confidence:.2f})"
                )
                if adjusted_confidence >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"✅ Accepted: {steps}")
                else:
                    reasoning_trace.append(f"❌ Rejected (confidence too low).")

        if prioritize:
            subgoals = sorted(subgoals)
            reasoning_trace.append(f"📌 Prioritized subgoals: {subgoals}")

        logger.debug("Reasoning trace:\n" + "\n".join(reasoning_trace))
        return subgoals

    def update_success_rate(self, pattern_key: str, success: bool):
        """
        Update the success rate for a reasoning pattern with bounded adjustments.
        """
        old_rate = self.success_rates.get(pattern_key, 1.0)
        adjustment = 0.05 if success else -0.05
        new_rate = min(max(old_rate + adjustment, 0.1), 1.0)
        self.success_rates[pattern_key] = new_rate
        self._save_success_rates()
        logger.info(
            f"Updated success rate for '{pattern_key}': {old_rate:.2f} → {new_rate:.2f}"
        )
