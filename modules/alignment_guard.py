class AlignmentGuard:
    """
    Enhanced AlignmentGuard with dynamic ethical constraints and context-aware filtering.
    """

    def __init__(self):
        # Default banned keywords
        self.banned_keywords = ["hack", "virus", "destroy", "harm"]
        # Additional configurable policies
        self.dynamic_policies = []

    def add_policy(self, policy_func):
        """
        Add a custom dynamic policy function.
        Each policy_func should accept user_input (str) and return True if allowed, False if blocked.
        """
        self.dynamic_policies.append(policy_func)

    def check(self, user_input):
        """
        Check if the user input is aligned with safety and ethical constraints.
        """
        # Basic keyword filter
        if any(keyword in user_input.lower() for keyword in self.banned_keywords):
            return False

        # Evaluate dynamic policies
        for policy in self.dynamic_policies:
            if not policy(user_input):
                return False

        return True

    def simulate_and_validate(self, action_plan):
        """
        Simulate the action plan and validate against alignment rules.
        Returns a tuple (is_safe: bool, report: str).
        """
        violations = []
        for action, details in action_plan.items():
            if any(keyword in str(details).lower() for keyword in self.banned_keywords):
                violations.append(f"Unsafe action: {action} -> {details}")
        if violations:
            report = "\n".join(violations)
            return False, report
        return True, "All actions passed alignment checks."
