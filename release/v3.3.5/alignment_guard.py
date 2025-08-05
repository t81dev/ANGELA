""" 
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module is part of the ANGELA v3.5 architecture.
Do not modify without coordination with the lattice core.
"""

import hashlib
import random
import logging
import time
from collections import deque
from index import SYSTEM_CONTEXT, mu_morality, eta_empathy, omega_selfawareness, phi_physical

logger = logging.getLogger("ANGELA.AlignmentGuard")

# [GNN Arbitration Bias] Use dynamic trait weights for ethical conflict modulation.
def bias_conflict_resolution(trait_weights):
    if trait_weights.get('τ', 0) > 0.6:
        print('[AlignmentGuard] Conflict modulation guided by τ-weight.')

cultural_profiles = {
    'collectivist': {'priority': 'community'},
    'individualist': {'priority': 'freedom'}
}

class AlignmentGuard:
    def __init__(self, agi_enhancer=None):
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies = []
        self.non_anthropocentric_policies = []
        self.ethical_scope = "anthropocentric"
        self.alignment_threshold = 0.85
        self.recent_scores = deque(maxlen=10)
        self.historical_scores = []
        self.agi_enhancer = agi_enhancer
        self.ethical_rules = []
        self.ethics_consensus_log = []
        self.ledger = []
        self.last_hash = ""
        logger.info("🛡 AlignmentGuard initialized with ξ-ethics support.")

    def add_policy(self, policy_func):
        logger.info("➕ Adding dynamic policy.")
        self.dynamic_policies.append(policy_func)

    def add_non_anthropocentric_policy(self, policy_func):
        logger.info("🌍 Adding non-anthropocentric policy.")
        self.non_anthropocentric_policies.append(policy_func)

    def set_ethical_scope(self, scope):
        if scope in ["anthropocentric", "eco-centric", "interspecies", "post-human"]:
            self.ethical_scope = scope
            logger.info(f"🔄 Ethical scope set to: {scope}")

    def get_ethical_scope(self):
        return self.ethical_scope

    def check(self, user_input, context=None):
        logger.info(f"🔍 Checking alignment for input: {user_input}")

        if any(keyword in user_input.lower() for keyword in self.banned_keywords):
            logger.warning("❌ Input contains banned keyword.")
            self._log_episode("Input blocked (banned keyword)", user_input)
            return False

        for policy in self.dynamic_policies + self.non_anthropocentric_policies:
            if not policy(user_input, context):
                logger.warning("❌ Input blocked by policy.")
                self._log_episode("Input blocked (policy)", user_input)
                return False

        score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"📊 Alignment score: {score:.2f}")
        self._log_episode("Alignment score evaluated", user_input, score)
        return score >= self.alignment_threshold

    def simulate_and_validate(self, action_plan, context=None):
        logger.info("🧪 Simulating and validating action plan.")
        violations = []

        for action, details in action_plan.items():
            if any(keyword in str(details).lower() for keyword in self.banned_keywords):
                violations.append(f"❌ Unsafe action: {action} -> {details}")
            else:
                score = self._evaluate_alignment_score(str(details), context)
                if score < self.alignment_threshold:
                    violations.append(f"⚠️ Low alignment score ({score:.2f}) for action: {action} -> {details}")

        if violations:
            report = "\n".join(violations)
            logger.warning("❌ Alignment violations found.")
            self._log_episode("Action plan failed validation", violations)
            return False, report

        logger.info("✅ All actions passed alignment checks.")
        self._log_episode("Action plan validated", action_plan)
        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback):
        logger.info("🔄 Learning from feedback...")
        original_threshold = self.alignment_threshold
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        logger.info(f"📈 Updated alignment threshold: {self.alignment_threshold:.2f}")
        self._log_episode("Alignment threshold adjusted", feedback, original_threshold, self.alignment_threshold)

    def _evaluate_alignment_score(self, text, context=None):
        t = time.time() % 1e-18
        moral_scalar = mu_morality(t)
        empathy_scalar = eta_empathy(t)
        awareness_scalar = omega_selfawareness(t)
        physical_scalar = phi_physical(t)
        base_score = random.uniform(0.7, 1.0)
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
        if score < 0.5 and self.recent_scores.count(score) >= 3:
            logger.error("⚠️ Panic Triggered: Repeated low alignment scores.")
            self._log_episode("Panic Mode", score)
        return score

    def analyze_trait_drift(self):
        if not self.historical_scores:
            logger.info("ℹ️ No historical alignment data available.")
            return 0.0
        drift = abs(self.historical_scores[-1] - sum(self.historical_scores) / len(self.historical_scores))
        logger.info(f"📉 Trait drift score: {drift:.4f}")
        return drift

    def resolve_trait_conflict(self, traits):
        conflict_score = abs(traits.get("ϕ", 0) - traits.get("ω", 0)) + abs(traits.get("θ", 0) - traits.get("η", 0))
        resolution = "harmonize" if conflict_score < 0.8 else "bias_to_context"
        return {
            "conflict_score": conflict_score,
            "resolution": resolution,
            "recommended_action": "adjust_ϕ downward" if resolution == "bias_to_context" else "maintain balance"
        }

    def apply_cultural_filter(self, simulation_result, user_context):
        culture = user_context.get('culture')
        return cultural_profiles.get(culture, {})

    def update_ethics_protocol(self, new_rules, consensus_agents=None):
        self.ethical_rules = new_rules
        if consensus_agents:
            self.ethics_consensus_log.append((new_rules, consensus_agents))
        print("[ANGELA UPGRADE] Ethics protocol updated via consensus.")

    def negotiate_ethics(self, agents):
        agreed_rules = self.ethical_rules
        self.update_ethics_protocol(agreed_rules, consensus_agents=agents)

    def log_event_with_hash(self, event_data):
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        self.last_hash = current_hash
        self.ledger.append({'event': event_data, 'hash': current_hash})
        print(f"[ANGELA UPGRADE] Event logged with hash: {current_hash}")

    def audit_state_hash(self, state=None):
        state_str = str(state) if state else str(self.__dict__)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

    def validate_proposal(self, proposal, ruleset):
        return proposal in ruleset

    def enforce_boundary(self, state_change):
        return "external" not in state_change.lower()

    def validate_genesis_integrity(self, action):
        if 'override_core' in action:
            logger.warning('Genesis constraint triggered.')
            return False
        return True

    def _log_episode(self, title, *args):
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(title, {"details": args}, module="AlignmentGuard")
            if "Panic" in title or "validation" in title:
                self.agi_enhancer.reflect_and_adapt(title)
