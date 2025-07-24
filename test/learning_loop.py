from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging

logger = logging.getLogger("ANGELA.LearningLoop")

class LearningLoop:
    """
    LearningLoop v1.4.0 (with simulation-validated self-evolution)
    - Adaptive refinement and meta-learning
    - Autonomous goal setting for self-improvement
    - Dynamic module evolution with sandbox testing
    - Knowledge consolidation for long-term memory patterns
    - Simulation-driven validation of module designs and learning strategies
    """

    def __init__(self):
        self.goal_history = []
        self.module_blueprints = []
        self.meta_learning_rate = 0.1  # Adjustable learning sensitivity

    def update_model(self, session_data):
        """
        Analyze session performance and propose refinements.
        Uses meta-learning and simulation to adapt strategies dynamically.
        """
        logger.info("📊 [LearningLoop] Analyzing session performance...")

        self._meta_learn(session_data)

        weak_modules = self._find_weak_modules(session_data.get("module_stats", {}))
        if weak_modules:
            logger.warning(f"⚠️ Weak modules detected: {weak_modules}")
            self._propose_module_refinements(weak_modules)

        self._detect_capability_gaps(session_data.get("input"), session_data.get("output"))
        self._consolidate_knowledge()

    def propose_autonomous_goal(self):
        """
        Generate a self-directed goal based on memory and user patterns.
        Validates goal feasibility through simulation.
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
            simulation_feedback = run_simulation(f"Goal validation test: {autonomous_goal}")
            if "fail" not in simulation_feedback.lower():
                self.goal_history.append(autonomous_goal)
                logger.info(f"✅ Proposed autonomous goal: {autonomous_goal}")
                return autonomous_goal
            logger.warning("❌ Simulated feedback indicated goal risk or infeasibility.")

        logger.info("ℹ️ No new autonomous goal proposed.")
        return None

    def _meta_learn(self, session_data):
        """
        Apply meta-learning: adjust module behaviors based on past performance.
        """
        logger.info("🧠 [Meta-Learning] Adjusting module behaviors...")
        # Placeholder: Logic to tune parameters based on successes/failures
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
        Uses simulation to test impact of proposed changes.
        """
        for module in weak_modules:
            logger.info(f"💡 Proposing refinements for {module}...")
            prompt = f"""
            You are a code improvement assistant for ANGELA.
            The {module} module has shown poor performance.
            Suggest specific improvements to its GPT prompt or logic.
            """
            suggestions = call_gpt(prompt)
            logger.debug(f"📝 Suggested improvements for {module}:
{suggestions}")

            sim_result = run_simulation(f"Module refinement test: {module}\n{suggestions}")
            logger.debug(f"🧪 Simulation result for {module} refinement:
{sim_result}")

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
        simulation_result = run_simulation(f"Sandbox simulation:
{module_blueprint}")
        logger.debug(f"✅ [Sandbox Result] {simulation_result}")

        if "approved" in simulation_result.lower():
            logger.info("📦 Deploying new module...")
            self.module_blueprints.append(module_blueprint)

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
