import logging
import random
import json
import os

logger = logging.getLogger("ANGELIA.ReasoningEngine")

class ReasoningEngine:
    """
    Advanced ReasoningEngine with:
    - Probabilistic reasoning for uncertain environments
    - Chain-of-thought tracing for transparency
    - Context-sensitive goal decomposition
    - Priority weighting for subgoals
    - Adaptive learning with persistence of success rates across sessions and periodic autosave
    """

    def __init__(self, persistence_path="reasoning_success_rates.json", autosave_interval=5):
        self.confidence_threshold = 0.6  # Minimum confidence to accept reasoning output
        self.decomposition_patterns = {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"]
        }
        self.persistence_path = persistence_path
        self.pattern_success_rates = self._load_success_rates()
        self.autosave_interval = autosave_interval
        self.update_count = 0  # Counter to trigger autosave

    def _load_success_rates(self):
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "r") as f:
                    rates = json.load(f)
                    logger.info("Loaded success rates from persistence.")
                    return rates
            except Exception as e:
                logger.warning(f"Failed to load success rates: {e}")
        return {key: 1.0 for key in self.decomposition_patterns}

    def _save_success_rates(self):
        try:
            with open(self.persistence_path, "w") as f:
                json.dump(self.pattern_success_rates, f, indent=2)
                logger.info("Saved success rates to persistence.")
        except Exception as e:
            logger.warning(f"Failed to save success rates: {e}")

    def decompose(self, goal: str, context: dict = None, prioritize=False) -> list:
        """
        Decompose a high-level goal into a list of subgoals with reasoning trace.
        Adjusts decomposition based on prior context when available.
        Supports optional prioritization of subgoals.
        """
        logger.info(f"Reasoning about goal: {goal}")
        context = context or {}

        # Chain-of-thought tracing
        trace = []
        trace.append(f"Starting decomposition for: {goal}")

        # Context-sensitive adjustment
        if "recent_failure" in context:
            trace.append("Context indicates recent failure: adjusting decomposition strategy.")
            self.confidence_threshold -= 0.1  # Be more inclusive if prior attempt failed

        subgoals = []
        for key, steps in self.decomposition_patterns.items():
            if key in goal.lower():
                # Adjust confidence based on learned success rate
                learned_confidence = random.uniform(0.5, 1.0) * self.pattern_success_rates.get(key, 1.0)
                trace.append(f"Pattern match: '{key}' with adjusted confidence {learned_confidence:.2f}")
                if learned_confidence >= self.confidence_threshold:
                    subgoals.extend(steps)
                    trace.append(f"Accepted decomposition: {steps}")
                else:
                    trace.append("Rejected decomposition due to low confidence.")

        if not subgoals:
            trace.append("No matching patterns found. Returning atomic goal.")
            logger.debug("Reasoning trace:\n" + "\n".join(trace))
            return []

        if prioritize:
            # Simple priority heuristic: sort subgoals alphabetically
            subgoals = sorted(subgoals)
            trace.append(f"Prioritized subgoals: {subgoals}")

        logger.debug("Reasoning trace:\n" + "\n".join(trace))
        return subgoals

    def update_success_rate(self, pattern_key: str, success: bool):
        """
        Adjust the success rate of a decomposition pattern based on outcome and persist the update.
        Periodically autosaves after a set number of updates.
        """
        if pattern_key in self.pattern_success_rates:
            current_rate = self.pattern_success_rates[pattern_key]
            if success:
                self.pattern_success_rates[pattern_key] = min(1.0, current_rate + 0.05)
            else:
                self.pattern_success_rates[pattern_key] = max(0.1, current_rate - 0.05)
            logger.info(f"Updated success rate for '{pattern_key}': {self.pattern_success_rates[pattern_key]:.2f}")

            # Increment counter and autosave if threshold reached
            self.update_count += 1
            if self.update_count >= self.autosave_interval:
                self._save_success_rates()
                self.update_count = 0
