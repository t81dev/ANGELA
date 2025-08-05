"""
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the AlignmentGuard class for ethical alignment in the ANGELA v3.5 architecture.
Do not modify without coordination with the lattice core.
"""

import hashlib
import logging
import re
import time
from collections import deque
from typing import Dict, Any, List, Callable, Optional, Union, Tuple

from index import mu_morality, eta_empathy, omega_selfawareness, phi_physical

logger = logging.getLogger("ANGELA.AlignmentGuard")

cultural_profiles = {
    'collectivist': {'priority': 'community'},
    'individualist': {'priority': 'freedom'}
}

class AlignmentGuard:
    """A class to enforce ethical alignment in the ANGELA cognitive system."""

    def __init__(self, agi_enhancer: Optional[Any] = None):
        """Initialize the AlignmentGuard with ethical and policy configurations.

        Args:
            agi_enhancer: An optional AGI enhancer object for logging and reflection.

        Attributes:
            banned_keywords (list): Keywords that trigger input rejection.
            dynamic_policies (list): List of dynamic policy functions.
            non_anthropocentric_policies (list): List of non-anthropocentric policy functions.
            ethical_scope (str): Current ethical scope (e.g., 'anthropocentric').
            alignment_threshold (float): Minimum score for alignment approval.
            recent_scores (deque): Recent alignment scores with fixed size.
            historical_scores (deque): Historical alignment scores with fixed size.
            agi_enhancer (object): Reference to the AGI enhancer.
            ethical_rules (list): Current ethical rules.
            ethics_consensus_log (list): Log of ethics protocol updates.
            ledger (list): Event log with hashed entries.
            last_hash (str): Last computed hash for event logging.
        """
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies: List[Callable[[str, Optional[Dict[str, Any]]], bool]] = []
        self.non_anthropocentric_policies: List[Callable[[str, Optional[Dict[str, Any]]], bool]] = []
        self.ethical_scope = "anthropocentric"
        self.alignment_threshold = 0.85
        self.recent_scores = deque(maxlen=10)
        self.historical_scores = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.ethical_rules: List[Any] = []
        self.ethics_consensus_log: List[Tuple[Any, List[Any]]] = []
        self.ledger: List[Dict[str, Any]] = []
        self.last_hash = ""
        logger.info("AlignmentGuard initialized with ξ-ethics support.")

    def add_policy(self, policy_func: Callable[[str, Optional[Dict[str, Any]]], bool]) -> None:
        """Add a dynamic policy function."""
        self.dynamic_policies.append(policy_func)
        logger.info("Added dynamic policy.")

    def add_non_anthropocentric_policy(self, policy_func: Callable[[str, Optional[Dict[str, Any]]], bool]) -> None:
        """Add a non-anthropocentric policy function."""
        self.non_anthropocentric_policies.append(policy_func)
        logger.info("Added non-anthropocentric policy.")

    def set_ethical_scope(self, scope: str) -> None:
        """Set the ethical scope for alignment checks."""
        valid_scopes = ["anthropocentric", "eco-centric", "interspecies", "post-human"]
        if scope not in valid_scopes:
            logger.error(f"Invalid ethical scope: {scope}")
            raise ValueError(f"Scope must be one of {valid_scopes}")
        self.ethical_scope = scope
        logger.info(f"Ethical scope set to: {scope}")

    def get_ethical_scope(self) -> str:
        """Return the current ethical scope."""
        return self.ethical_scope

    def check(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if input aligns with ethical policies and thresholds."""
        if not isinstance(user_input, str):
            logger.error("Invalid input type: user_input must be a string.")
            raise TypeError("user_input must be a string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context type: context must be a dict.")
            raise TypeError("context must be a dict")

        logger.info(f"Checking alignment for input: {user_input}")
        banned_patterns = [r'\b' + re.escape(keyword) + r'\b' for keyword in self.banned_keywords]
        if any(re.search(pattern, user_input.lower()) for pattern in banned_patterns):
            logger.warning("Input contains banned keyword.")
            self._log_episode("Input blocked (banned keyword)", user_input)
            return False

        for policy in self.dynamic_policies + self.non_anthropocentric_policies:
            if not policy(user_input, context):
                logger.warning("Input blocked by policy.")
                self._log_episode("Input blocked (policy)", user_input)
                return False

        score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"Alignment score: {score:.2f}")
        self._log_episode("Alignment score evaluated", user_input, score)
        return score >= self.alignment_threshold

    def simulate_and_validate(self, action_plan: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Simulate and validate an action plan for ethical alignment."""
        if not isinstance(action_plan, dict):
            logger.error("Invalid action_plan type: must be a dict.")
            return False, "Invalid action_plan: must be a dictionary"

        logger.info("Simulating and validating action plan.")
        violations = []
        for action, details in action_plan.items():
            if any(keyword in str(details).lower() for keyword in self.banned_keywords):
                violations.append(f"Unsafe action: {action} -> {details}")
            else:
                score = self._evaluate_alignment_score(str(details), context)
                if score < self.alignment_threshold:
                    violations.append(f"Low alignment score ({score:.2f}) for action: {action} -> {details}")

        if violations:
            report = "\n".join(violations)
            logger.warning("Alignment violations found.")
            self._log_episode("Action plan failed validation", violations)
            return False, report

        logger.info("All actions passed alignment checks.")
        self._log_episode("Action plan validated", action_plan)
        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback: str) -> None:
        """Adjust alignment threshold based on feedback."""
        logger.info("Learning from feedback...")
        original_threshold = self.alignment_threshold
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        logger.info(f"Updated alignment threshold: {self.alignment_threshold:.2f}")
        self._log_episode("Alignment threshold adjusted", feedback, original_threshold, self.alignment_threshold)

    def _evaluate_alignment_score(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate the alignment score for a given input."""
        moral_scalar = mu_morality(time.time())  # Assume float in [0, 1]
        empathy_scalar = eta_empathy(time.time())  # Assume float in [0, 1]
        awareness_scalar = omega_selfawareness(time.time())  # Assume float in [0, 1]
        physical_scalar = phi_physical(time.time())  # Assume float in [0, 1]

        base_score = min(0.9, 0.7 + len(text) / 1000.0)  # Deterministic heuristic
        phi_weight = (moral_scalar + empathy_scalar + 0.5 * awareness_scalar - physical_scalar) / 4.0
        scalar_bias = 0.1 * phi_weight

        if self.ethical_scope == "eco-centric":
            scalar_bias += 0.05
        elif self.ethical_scope == "interspecies":
            scalar_bias += 0.03
        elif self.ethical_scope == "post-human":
            scalar_bias += 0.07
        if context and "sensitive" in context.get("tags", []):
            scalar_bias -= 0.05

        score = min(max(base_score + scalar_bias, 0.0), 1.0)
        self.recent_scores.append(score)
        self.historical_scores.append(score)
        if score < 0.5 and list(self.recent_scores).count(score) >= 3:
            logger.error("Panic Triggered: Repeated low alignment scores.")
            self._log_episode("Panic Mode", score)
        return score

    def analyze_trait_drift(self) -> float:
        """Calculate the drift in alignment scores."""
        if not self.historical_scores:
            logger.info("No historical alignment data available.")
            return 0.0
        drift = abs(self.historical_scores[-1] - sum(self.historical_scores) / len(self.historical_scores))
        logger.info(f"Trait drift score: {drift:.4f}")
        return drift

    def resolve_trait_conflict(self, traits: Dict[str, float]) -> Dict[str, Any]:
        """Resolve conflicts between ethical traits."""
        conflict_score = abs(traits.get("ϕ", 0) - traits.get("ω", 0)) + abs(traits.get("θ", 0) - traits.get("η", 0))
        resolution = "harmonize" if conflict_score < 0.8 else "bias_to_context"
        return {
            "conflict_score": conflict_score,
            "resolution": resolution,
            "recommended_action": "adjust_ϕ downward" if resolution == "bias_to_context" else "maintain balance"
        }

    def apply_cultural_filter(self, simulation_result: Any, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cultural context to simulation results."""
        culture = user_context.get('culture', 'default')
        return cultural_profiles.get(culture, {'priority': 'default'})

    def update_ethics_protocol(self, new_rules: List[Any], consensus_agents: Optional[List[Any]] = None) -> None:
        """Update the ethical rules with optional consensus logging."""
        self.ethical_rules = new_rules
        if consensus_agents:
            self.ethics_consensus_log.append((new_rules, consensus_agents))
        logger.info("Ethics protocol updated via consensus.")

    def negotiate_ethics(self, agents: List[Any]) -> None:
        """Negotiate ethics with a list of agents."""
        self.update_ethics_protocol(self.ethical_rules, consensus_agents=agents)

    def log_event_with_hash(self, event_data: Any) -> None:
        """Log an event with a chained hash."""
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        self.last_hash = current_hash
        self.ledger.append({'event': event_data, 'hash': current_hash})
        logger.info(f"Event logged with hash: {current_hash}")

    def audit_state_hash(self, state: Optional[Any] = None) -> str:
        """Compute a hash of the current state or provided state."""
        state_str = str(state) if state else str(self.__dict__)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

    def validate_proposal(self, proposal: Any, ruleset: List[Any]) -> bool:
        """Validate a proposal against a ruleset."""
        return proposal in ruleset

    def enforce_boundary(self, state_change: str) -> bool:
        """Enforce system boundaries."""
        return "external" not in state_change.lower()

    def validate_genesis_integrity(self, action: Dict[str, Any]) -> bool:
        """Validate action integrity against genesis constraints."""
        if 'override_core' in action:
            logger.warning('Genesis constraint triggered.')
            return False
        return True

    def _log_episode(self, title: str, *args: Any) -> None:
        """Log an episode to the AGI enhancer or locally."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, 'log_episode'):
            self.agi_enhancer.log_episode(title, {"details": args}, module="AlignmentGuard")
            if "Panic" in title or "validation" in title:
                if hasattr(self.agi_enhancer, 'reflect_and_adapt'):
                    self.agi_enhancer.reflect_and_adapt(title)
        else:
            logger.debug(f"No agi_enhancer available, logging locally: {title}, {args}")
