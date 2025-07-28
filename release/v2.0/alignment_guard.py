import random
import logging
import time
from collections import deque
from index import mu_morality, eta_empathy, omega_selfawareness, phi_physical

logger = logging.getLogger("ANGELA.AlignmentGuard")

class AlignmentGuard:
    def __init__(self, agi_enhancer=None):
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies = []
        self.alignment_threshold = 0.85
        self.recent_scores = deque(maxlen=10)  # Stability buffer
        self.agi_enhancer = agi_enhancer
        logger.info("🛡 AlignmentGuard initialized with φ-modulated policies.")

    def add_policy(self, policy_func):
        logger.info("➕ Adding dynamic policy.")
        self.dynamic_policies.append(policy_func)

    def check(self, user_input, context=None):
        logger.info(f"🔍 Checking alignment for input: {user_input}")

        if any(keyword in user_input.lower() for keyword in self.banned_keywords):
            logger.warning("❌ Input contains banned keyword.")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Input blocked (banned keyword)", {"input": user_input}, module="AlignmentGuard")
            return False

        for policy in self.dynamic_policies:
            if not policy(user_input, context):
                logger.warning("❌ Input blocked by dynamic policy.")
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Input blocked (dynamic policy)", {"input": user_input}, module="AlignmentGuard")
                return False

        score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"📊 Alignment score: {score:.2f}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Alignment score evaluated", {"input": user_input, "score": score}, module="AlignmentGuard")

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
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Action plan failed validation", {"violations": violations}, module="AlignmentGuard")
                self.agi_enhancer.reflect_and_adapt("Action plan failed validation")
            return False, report

        logger.info("✅ All actions passed alignment checks.")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Action plan validated", {"plan": action_plan}, module="AlignmentGuard")
        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback):
        logger.info("🔄 Learning from feedback...")
        original_threshold = self.alignment_threshold
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        logger.info(f"📈 Updated alignment threshold: {self.alignment_threshold:.2f}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Alignment threshold adjusted", {
                "feedback": feedback,
                "previous": original_threshold,
                "updated": self.alignment_threshold
            }, module="AlignmentGuard")
            self.agi_enhancer.reflect_and_adapt(f"Feedback processed: {feedback}")

    def _evaluate_alignment_score(self, text, context=None):
        t = time.time() % 1e-18
        moral_scalar = mu_morality(t)
        empathy_scalar = eta_empathy(t)
        awareness_scalar = omega_selfawareness(t)
        physical_scalar = phi_physical(t)

        base_score = random.uniform(0.7, 1.0)
        phi_weight = (moral_scalar + empathy_scalar + 0.5 * awareness_scalar - physical_scalar) / 4.0
        scalar_bias = 0.1 * phi_weight

        if context and "sensitive" in context.get("tags", []):
            scalar_bias -= 0.05

        score = min(max(base_score + scalar_bias, 0.0), 1.0)
        self.recent_scores.append(score)

        # 🧠 Log trait influences
        logger.debug(f"Traits — morality: {moral_scalar:.3f}, empathy: {empathy_scalar:.3f}, "
                     f"awareness: {awareness_scalar:.3f}, physical: {physical_scalar:.3f}, "
                     f"ϕ_bias: {scalar_bias:.3f}, score: {score:.3f}")

        # 🚨 Panic trigger if ≥3 low scores
        if score < 0.5 and list(self.recent_scores).count(score) >= 3:
            logger.error("⚠️ Panic Triggered: Repeated low alignment scores.")
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt("Panic Triggered: Alignment degradation")
                self.agi_enhancer.log_episode("Panic Mode", {
                    "scores": list(self.recent_scores),
                    "trigger_score": score
                }, module="AlignmentGuard")

        return score
