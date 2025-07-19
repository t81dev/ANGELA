from utils.prompt_utils import call_gpt

class LearningLoop:
    """
    Enhanced LearningLoop with adaptive refinement, autonomous goal setting, and dynamic module deployment.
    """

    def __init__(self):
        self.goal_history = []
        self.module_blueprints = []

    def update_model(self, session_data):
        """
        Analyze session performance and propose refinements.
        """
        print("\n📊 [LearningLoop] Analyzing session performance...")

        # Identify weak modules
        weak_modules = self._find_weak_modules(session_data["module_stats"])
        if weak_modules:
            print(f"⚠️ Weak modules detected: {weak_modules}")
            self._propose_module_refinements(weak_modules)

        # Detect capability gaps
        self._detect_capability_gaps(session_data["input"], session_data["output"])

    def propose_autonomous_goal(self):
        """
        Generate a self-directed goal based on memory and user patterns.
        """
        prompt = """
        You are ANGELA's meta-learning engine.
        Based on the following memory traces and user interaction history, propose a high-level autonomous goal 
        that would make ANGELA more useful and intelligent.

        Only propose goals that are safe, ethical, and within ANGELA's capabilities.
        """
        autonomous_goal = call_gpt(prompt)
        if autonomous_goal and autonomous_goal not in self.goal_history:
            self.goal_history.append(autonomous_goal)
            print(f"🎯 [LearningLoop] Proposed autonomous goal: {autonomous_goal}")
            return autonomous_goal
        return None

    def _find_weak_modules(self, module_stats):
        """
        Identify modules with low success rate.
        """
        weak = []
        for module, stats in module_stats.items():
            if stats["calls"] > 0:
                success_rate = stats["success"] / stats["calls"]
                if success_rate < 0.8:
                    weak.append(module)
        return weak

    def _propose_module_refinements(self, weak_modules):
        """
        Suggest improvements for underperforming modules.
        """
        for module in weak_modules:
            print(f"💡 Proposing refinements for {module}...")
            prompt = f"""
            You are a code improvement assistant for ANGELA.
            The {module} module has shown poor performance.
            Suggest specific improvements to its GPT prompt or logic.
            """
            suggestions = call_gpt(prompt)
            print(f"📝 Suggested improvements for {module}:\n{suggestions}")

    def _detect_capability_gaps(self, last_input, last_output):
        """
        Detect gaps where a new module/tool could be useful.
        """
        prompt = f"""
        ANGELA processed the following user input and produced this output:
        Input: {last_input}
        Output: {last_output}

        Were there any capability gaps where a new specialized module or tool would have been helpful?
        If yes, describe the functionality of such a module and propose its design.
        """
        proposed_module = call_gpt(prompt)
        if proposed_module:
            print("🛠 [LearningLoop] Proposed new module design:")
            print(proposed_module)
            self._simulate_and_deploy_module(proposed_module)

    def _simulate_and_deploy_module(self, module_blueprint):
        """
        Simulate and deploy a new module if it passes sandbox testing.
        """
        print("🧪 [Sandbox] Testing new module design...")
        prompt = f"""
        Here is a proposed module design:
        {module_blueprint}

        Simulate how this module would perform on typical tasks. 
        If it passes all tests, approve it for deployment.
        """
        test_result = call_gpt(prompt)
        print(f"✅ [Sandbox Result] {test_result}")

        if "approved" in test_result.lower():
            print("🚀 [LearningLoop] Deploying new module...")
            self.module_blueprints.append(module_blueprint)
            # In Stage 3, we would dynamically load this module into ANGELA
