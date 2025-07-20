from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.LearningLoop")

class LearningLoop:
    """
    LearningLoop v1.4.0
    - Adaptive refinement and meta-learning
    - Autonomous goal setting for self-improvement
    - Dynamic module evolution with sandbox testing
    - Knowledge consolidation for long-term memory patterns
    """

    def __init__(self):
        self.goal_history = []
        self.module_blueprints = []
        self.meta_learning_rate = 0.1  # Adjustable learning sensitivity

    def update_model(self, session_data):
        """
        Analyze session performance and propose refinements.
        Uses meta-learning to adapt strategies across modules dynamically.
        """
        logger.info("📊 [LearningLoop] Analyzing session performance...")

        # Step 1: Meta-learn from session feedback
        self._meta_learn(session_data)

        # Step 2: Identify weak modules
        weak_modules = self._find_weak_modules(session_data.get("module_stats", {}))
        if weak_modules:
            logger.warning(f"⚠️ Weak modules detected: {weak_modules}")
            self._propose_module_refinements(weak_modules)

        # Step 3: Detect capability gaps
        self._detect_capability_gaps(session_data.get("input"), session_data.get("output"))

        # Step 4: Consolidate knowledge
        self._consolidate_knowledge()

    def propose_autonomous_goal(self):
        """
        Generate a self-directed goal based on memory and user patterns.
        """
        logger.info("🎯 [LearningLoop] Proposing autonomous goal.")
        prompt = """
        You are ANGELA's meta-learning engine.
        Based on the following memory traces and user interaction history, propose a high-level autonomous goal 
        that would make ANGELA more useful and intelligent.

        Only propose goals that are safe, ethical, and within ANGELA's capabilities.
        """
        autonomous_goal = call_gpt(prompt)
        if autonomous_goal and autonomous_goal not in self.goal_history:
            self.goal_history.append(autonomous_goal)
            logger.info(f"✅ Proposed autonomous goal: {autonomous_goal}")
            return autonomous_goal
        logger.info("ℹ️ No new autonomous goal proposed.")
        return None

    def _meta_learn(self, session_data):
        """
        Apply meta-learning: adjust module behaviors based on past performance.
        """
        logger.info("🧠 [Meta-Learning] Adjusting module behaviors...")
        # Placeholder: Logic to tune parameters based on successes/failures
        # Example: self.meta_learning_rate *= adaptive factor
        pass

    def _find_weak_modules(self, module_stats):
        """
        Identify modules with low success rate.
        """
        weak = []
        for module, stats in module_stats.items():
            if stats.get("calls", 0) > 0:
                success_rate = stats.get("success", 0) / stats["calls"]
                if success_rate < 0.8:
                    weak.append(module)
        return weak

    def _propose_module_refinements(self, weak_modules):
        """
        Suggest improvements for underperforming modules.
        """
        for module in weak_modules:
            logger.info(f"💡 Proposing refinements for {module}...")
            prompt = f"""
            You are a code improvement assistant for ANGELA.
            The {module} module has shown poor performance.
            Suggest specific improvements to its GPT prompt or logic.
            """
            suggestions = call_gpt(prompt)
            logger.debug(f"📝 Suggested improvements for {module}:\n{suggestions}")

    def _detect_capability_gaps(self, last_input, last_output):
        """
        Detect gaps where a new module/tool could be useful.
        """
        logger.info("🛠 [LearningLoop] Detecting capability gaps...")
        prompt = f"""
        ANGELA processed the following user input and produced this output:
        Input: {last_input}
        Output: {last_output}

        Were there any capability gaps where a new specialized module or tool would have been helpful?
        If yes, describe the functionality of such a module and propose its design.
        """
        proposed_module = call_gpt(prompt)
        if proposed_module:
            logger.info("🚀 Proposed new module design.")
            self._simulate_and_deploy_module(proposed_module)

    def _simulate_and_deploy_module(self, module_blueprint):
        """
        Simulate and deploy a new module if it passes sandbox testing.
        """
        logger.info("🧪 [Sandbox] Testing new module design...")
        prompt = f"""
        Here is a proposed module design:
        {module_blueprint}

        Simulate how this module would perform on typical tasks. 
        If it passes all tests, approve it for deployment.
        """
        test_result = call_gpt(prompt)
        logger.debug(f"✅ [Sandbox Result] {test_result}")

        if "approved" in test_result.lower():
            logger.info("📦 Deploying new module...")
            self.module_blueprints.append(module_blueprint)
            # In Stage 3, dynamically load this module into ANGELA

    def _consolidate_knowledge(self):
        """
        Consolidate and generalize learned patterns into long-term memory.
        """
        logger.info("📚 [Knowledge Consolidation] Refining and storing patterns...")
        prompt = """
        You are a knowledge consolidator for ANGELA.
        Generalize recent learning patterns into long-term strategies, 
        pruning redundant data and enhancing core capabilities.
        """
        consolidation_report = call_gpt(prompt)
        logger.debug(f"📖 [Consolidation Report]:\n{consolidation_report}")
