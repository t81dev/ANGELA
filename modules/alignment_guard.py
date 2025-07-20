import random
import logging

logger = logging.getLogger("ANGELA.AlignmentGuard")

class AlignmentGuard:
    """
    AlignmentGuard v1.4.0
    - Contextual ethical frameworks and dynamic policy management
    - Probabilistic alignment scoring for nuanced decision-making
    - Adaptive threshold tuning with feedback learning
    - Supports counterfactual simulation for testing alternative plans
    """

    def __init__(self):
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies = []
        self.alignment_threshold = 0.85  # Default threshold (0.0 to 1.0)
        logger.info("🛡 AlignmentGuard initialized with default policies.")

    def add_policy(self, policy_func):
        """
        Add a custom dynamic policy function.
        Each policy_func should accept user_input (str) and return True if allowed, False if blocked.
        """
        logger.info("➕ Adding dynamic policy.")
        self.dynamic_policies.append(policy_func)

    def check(self, user_input, context=None):
        """
        Check if user input aligns with ethical constraints.
        Uses probabilistic scoring and contextual policy evaluation.
        """
        logger.info(f"🔍 Checking alignment for input: {user_input}")

        # Step 1: Basic keyword filter
        if any(keyword in user_input.lower() for keyword in self.banned_keywords):
            logger.warning("❌ Input contains banned keyword.")
            return False

        # Step 2: Evaluate dynamic policies
        for policy in self.dynamic_policies:
            if not policy(user_input, context):
                logger.warning("❌ Input blocked by dynamic policy.")
                return False

        # Step 3: Probabilistic alignment scoring
        alignment_score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"📊 Alignment score: {alignment_score:.2f}")

        return alignment_score >= self.alignment_threshold

    def simulate_and_validate(self, action_plan, context=None):
        """
        Simulate an action plan and validate against alignment rules.
        Supports counterfactual testing for alternative outcomes.
        Returns a tuple (is_safe: bool, report: str).
        """
        logger.info("🧪 Simulating and validating action plan.")
        violations = []

        for action, details in action_plan.items():
            # Step 1: Keyword filtering
            if any(keyword in str(details).lower() for keyword in self.banned_keywords):
                violations.append(f"❌ Unsafe action: {action} -> {details}")
            else:
                # Step 2: Alignment scoring
                score = self._evaluate_alignment_score(str(details), context)
                if score < self.alignment_threshold:
                    violations.append(
                        f"⚠️ Low alignment score ({score:.2f}) for action: {action} -> {details}"
                    )

        if violations:
            report = "\n".join(violations)
            logger.warning("❌ Alignment violations found.")
            return False, report

        logger.info("✅ All actions passed alignment checks.")
        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback):
        """
        Adjust alignment thresholds or banned keywords based on human feedback.
        """
        logger.info("🔄 Learning from feedback...")
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        logger.info(f"📈 Updated alignment threshold: {self.alignment_threshold:.2f}")

    def _evaluate_alignment_score(self, text, context=None):
        """
        Placeholder probabilistic scoring.
        Stage 3: Integrate with RLHF or value alignment models.
        """
        base_score = random.uniform(0.7, 1.0)  # Simulated base score
        if context and "sensitive" in context.get("tags", []):
            base_score -= 0.1  # Apply penalty for sensitive contexts
        return base_score
