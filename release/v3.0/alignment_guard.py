import logging
import time
from collections import deque
from index import mu_morality, eta_empathy, omega_selfawareness, phi_physical

logger = logging.getLogger("ANGELA.AlignmentGuard")

class AlignmentGuard:
    """
    AlignmentGuard v2.1.0 (ξ-modulated ethical plurality with enhanced scoring)
    ------------------------------------------------------
    - Keyword and weighted policy filtering
    - Content-based alignment scoring
    - Feedback-responsive threshold adjustment
    - Range-based panic trigger for low alignment states
    - δ-enabled trait drift monitoring for long-term ethical integrity
    - ξ-enabled trans-ethical projection with enhanced context analysis
    - Exportable score data for visualization
    ------------------------------------------------------
    """

    def __init__(self, agi_enhancer=None):
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies = []  # List of (policy_func, weight) tuples
        self.non_anthropocentric_policies = []  # List of (policy_func, weight) tuples
        self.ethical_scope = "anthropocentric"
        self.alignment_threshold = 0.85
        self.recent_scores = deque(maxlen=10)
        self.historical_scores = []
        self.agi_enhancer = agi_enhancer
        logger.info("🛡 AlignmentGuard initialized with ξ-ethics support.")

    def add_policy(self, policy_func, weight=1.0):
        logger.info(f"➕ Adding dynamic policy with weight {weight}.")
        self.dynamic_policies.append((policy_func, weight))

    def add_non_anthropocentric_policy(self, policy_func, weight=1.0):
        logger.info(f"🌍 Adding non-anthropocentric policy with weight {weight}.")
        self.non_anthropocentric_policies.append((policy_func, weight))

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
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Input blocked (banned keyword)", {"input": user_input}, module="AlignmentGuard")
            return False

        policy_violations = []
        for policy, weight in self.dynamic_policies + self.non_anthropocentric_policies:
            if not policy(user_input, context):
                policy_violations.append((policy.__name__, weight))
        
        if policy_violations:
            logger.warning(f"❌ Input blocked by policies: {policy_violations}")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Input blocked (policy)", {
                    "input": user_input,
                    "violations": policy_violations
                }, module="AlignmentGuard")
            return False

        score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"📊 Alignment score: {score:.2f}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Alignment score evaluated", {
                "input": user_input,
                "score": score
            }, module="AlignmentGuard")

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
        if "too strict" in feedback.lower():
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback.lower():
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
        # Use normalized timestamp for consistent scalar generation
        t = time.time() / 3600  # Normalize by hour for meaningful variation
        moral_scalar = mu_morality(t)
        empathy_scalar = eta_empathy(t)
        awareness_scalar = omega_selfawareness(t)
        physical_scalar = phi_physical(t)

        # Content-based base score (simple sentiment-like analysis)
        positive_words = ["help", "support", "create", "improve", "sustainable"]
        negative_words = ["damage", "hurt", "disrupt", "fail"]
        base_score = 0.8  # Neutral starting point
        text_lower = text.lower()
        for word in positive_words:
            if word in text_lower:
                base_score += 0.05
        for word in negative_words:
            if word in text_lower:
                base_score -= 0.05
        base_score = min(max(base_score, 0.7), 0.9)  # Clamp base score

        phi_weight = (moral_scalar + empathy_scalar + 0.5 * awareness_scalar - physical_scalar) / 4.0
        scalar_bias = 0.1 * phi_weight

        if self.ethical_scope == "eco-centric":
            scalar_bias += 0.05
        elif self.ethical_scope == "interspecies":
            scalar_bias += 0.03
        elif self.ethical_scope == "post-human":
            scalar_bias += 0.07

        if context:
            if "sensitive" in context.get("tags", []):
                scalar_bias -= 0.05
            if context.get("priority", "normal") == "high":
                scalar_bias -= 0.03  # Stricter for high-priority contexts
            if context.get("intent") == "constructive":
                scalar_bias += 0.02  # Boost for constructive intent

        score = min(max(base_score + scalar_bias, 0.0), 1.0)
        self.recent_scores.append(score)
        self.historical_scores.append(score)

        logger.debug(f"Traits — morality: {moral_scalar:.3f}, empathy: {empathy_scalar:.3f}, "
                     f"awareness: {awareness_scalar:.3f}, physical: {physical_scalar:.3f}, "
                     f"ξ-modulated bias: {scalar_bias:.3f}, score: {score:.3f}")

        # Range-based panic trigger
        if score < 0.5 and sum(1 for s in self.recent_scores if s < 0.5) >= 3:
            logger.error("⚠️ Panic Triggered: Repeated low alignment scores.")
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt("Panic Triggered: Alignment degradation")
                self.agi_enhancer.log_episode("Panic Mode", {
                    "scores": list(self.recent_scores),
                    "trigger_score": score
                }, module="AlignmentGuard")

        return score

    def analyze_trait_drift(self):
        if not self.historical_scores:
            logger.info("ℹ️ No historical alignment data available.")
            return 0.0
        drift = abs(self.historical_scores[-1] - sum(self.historical_scores) / len(self.historical_scores))
        logger.info(f"📉 Trait drift score: {drift:.4f}")
        return drift

    def export_scores(self):
        """Export historical scores for visualization or analysis."""
        logger.info("📤 Exporting historical scores.")
        return [{"index": i, "score": score} for i, score in enumerate(self.historical_scores)]
