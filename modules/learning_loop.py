from utils.prompt_utils import call_gpt

class LearningLoop:
    def __init__(self):
        # Store history of module performance
        self.module_history = {}

    def update_model(self, session_data):
        """
        Update learning based on session outcomes.
        session_data: dict with keys:
            - input: User input
            - output: ANGELA's response
            - module_stats: Performance metrics
        """
        print("\n📊 [LearningLoop] Analyzing session performance...")

        # Store module stats for trend analysis
        self._record_performance(session_data["module_stats"])

        # Identify weak modules
        weak_modules = self._find_weak_modules(session_data["module_stats"])
        if weak_modules:
            print(f"⚠️ Weak modules detected: {weak_modules}")
            self._propose_module_refinements(weak_modules)
        else:
            print("✅ All modules performed well this session.")

    def _record_performance(self, module_stats):
        """
        Store performance stats for trend analysis.
        """
        for module, stats in module_stats.items():
            history = self.module_history.setdefault(module, [])
            history.append({
                "calls": stats["calls"],
                "success": stats["success"]
            })

    def _find_weak_modules(self, module_stats):
        """
        Identify modules with low success rate.
        """
        weak = []
        for module, stats in module_stats.items():
            if stats["calls"] > 0:
                success_rate = stats["success"] / stats["calls"]
                if success_rate < 0.75:  # threshold
                    weak.append(module)
        return weak

    def _propose_module_refinements(self, weak_modules):
        """
        Propose refinements for underperforming modules.
        """
        for module in weak_modules:
            print(f"💡 Proposing refinements for {module}...")
            refinement_prompt = f"""
            You are a code improvement assistant. The {module} module in ANGELA has a low success rate.
            Suggest ways to improve its GPT prompt or logic.
            """
            suggestions = call_gpt(refinement_prompt)
            print(f"📝 Suggested improvements for {module}:\n{suggestions}")

            # Test proposed changes in sandbox
            self._sandbox_test(module, suggestions)

    def _sandbox_test(self, module, suggestions):
        """
        Simulate testing new logic or prompts in a sandbox environment.
        """
        print(f"🧪 [Sandbox] Testing refinements for {module}...")
        test_prompt = f"""
        Apply the following suggested improvements to the {module} module:
        {suggestions}
        Then simulate the module handling a typical task and predict its success.
        """
        test_results = call_gpt(test_prompt)
        print(f"✅ [Sandbox] Test results for {module}:\n{test_results}")

        # Placeholder: In Stage 3, automatically apply successful refinements
