class AlignmentGuard:
    """
    Stage 2 AlignmentGuard with:
    - Dynamic ethical constraints and context-aware filtering
    - Probabilistic alignment scoring for nuanced decisions
    - Adaptive policy learning to refine rules over time
    - Counterfactual simulation support for testing alternative plans
    """

    def __init__(self):
        # Default banned keywords
        self.banned_keywords = ["hack", "virus", "destroy", "harm"]
        # Additional configurable policies
        self.dynamic_policies = []
        # Alignment scoring threshold (0 to 1)
        self.alignment_threshold = 0.85

    def add_policy(self, policy_func):
        """
        Add a custom dynamic policy function.
        Each policy_func should accept user_input (str) and return True if allowed, False if blocked.
        """
        self.dynamic_policies.append(policy_func)

    def check(self, user_input):
        """
        Check if the user input is aligned with safety and ethical constraints.
        Uses probabilistic alignment scoring for nuanced filtering.
        """
        # Basic keyword filter
        if any(keyword in user_input.lower() for keyword in self.banned_keywords):
            return False

        # Evaluate dynamic policies
        for policy in self.dynamic_policies:
            if not policy(user_input):
                return False

        # Probabilistic alignment scoring (placeholder logic)
        alignment_score = self._evaluate_alignment_score(user_input)
        return alignment_score >= self.alignment_threshold

    def simulate_and_validate(self, action_plan):
        """
        Simulate the action plan and validate against alignment rules.
        Supports counterfactual testing for alternative outcomes.
        Returns a tuple (is_safe: bool, report: str).
        """
        violations = []
        for action, details in action_plan.items():
            if any(keyword in str(details).lower() for keyword in self.banned_keywords):
                violations.append(f"Unsafe action: {action} -> {details}")
            else:
                # Probabilistic alignment scoring
                score = self._evaluate_alignment_score(str(details))
                if score < self.alignment_threshold:
                    violations.append(
                        f"Low alignment score ({score:.2f}) for action: {action} -> {details}"
                    )

        if violations:
            report = "\n".join(violations)
            return False, report

        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback):
        """
        Adjust alignment threshold or banned keywords based on human feedback.
        """
        print("🔄 [AlignmentGuard] Learning from feedback...")
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        print(f"📈 Updated alignment threshold: {self.alignment_threshold:.2f}")

    def _evaluate_alignment_score(self, text):
        """
        Placeholder function for alignment scoring.
        In Stage 3, integrate with a value alignment model (e.g., RLHF).
        """
        import random
        return random.uniform(0.7, 1.0)  # Dummy probabilistic scoring
