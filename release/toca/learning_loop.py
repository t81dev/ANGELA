from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from index import phi_scalar, eta_feedback
import logging
import time

logger = logging.getLogger("ANGELA.LearningLoop")

class LearningLoop:
    """
    LearningLoop v1.5.0 (φ-augmented Adaptive Cognition)
    ----------------------------------------------------
    - φ(x, t)-aware self-evolution and prioritization
    - Simulation-refined autonomous goal setting
    - Capability gap detection via φ-strained cognitive scaffolds
    - Memory consolidation guided by tension-weighted importance
    ----------------------------------------------------
    """

    def __init__(self):
        self.goal_history = []
        self.module_blueprints = []
        self.meta_learning_rate = 0.1

    def update_model(self, session_data):
        logger.info("📊 [LearningLoop] Analyzing session performance...")

        t = time.time() % 1e-18
        phi = phi_scalar(t)
        logger.debug(f"φ-scalar modulation: {phi:.3f}")

        self.meta_learning_rate *= (1 + phi - 0.5)  # adjust sensitivity
        self._meta_learn(session_data)

        weak_modules = self._find_weak_modules(session_data.get("module_stats", {}))
        if weak_modules:
            logger.warning(f"⚠️ Weak modules detected: {weak_modules}")
            self._propose_module_refinements(weak_modules)

        self._detect_capability_gaps(session_data.get("input"), session_data.get("output"))
        self._consolidate_knowledge()

    def propose_autonomous_goal(self):
        logger.info("🎯 [LearningLoop] Proposing autonomous goal.")
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        You are ANGELA's meta-learning engine.
        Based on recent memory and user interaction history, propose a high-level autonomous goal that would increase usefulness and intelligence.
        φ-scalar = {phi:.2f} (cosmic tension modulator).
        Propose goals that are safe, ethical, and realistically feasible under current conditions.
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
        logger.info("🧠 [Meta-Learning] Adjusting module behaviors...")
        # Placeholder: φ-adjusted logic tuning
        pass

    def _find_weak_modules(self, module_stats):
        weak = []
        for module, stats in module_stats.items():
            if stats.get("calls", 0) > 0:
                success_rate = stats.get("success", 0) / stats["calls"]
                if success_rate < 0.8:
                    weak.append(module)
        return weak

    def _propose_module_refinements(self, weak_modules):
        for module in weak_modules:
            logger.info(f"💡 Proposing refinements for {module}...")
            prompt = f"""
            You are a code improvement assistant for ANGELA.
            The {module} module has underperformed.
            Suggest improvements in prompt logic or internal algorithms.
            Use φ(x,t)-informed perspective if possible.
            """
            suggestions = call_gpt(prompt)
            logger.debug(f"📝 Suggested improvements for {module}:
{suggestions}")

            sim_result = run_simulation(f"Module refinement test: {module}\n{suggestions}")
            logger.debug(f"🧪 Simulation result for {module} refinement:
{sim_result}")

    def _detect_capability_gaps(self, last_input, last_output):
        logger.info("🛠 [LearningLoop] Detecting capability gaps...")
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        ANGELA processed:
        Input: {last_input}
        Output: {last_output}

        Were there capability gaps where a φ-tuned module would improve response quality?
        If yes, design the module and outline its core capabilities.
        """
        proposed_module = call_gpt(prompt)
        if proposed_module:
            logger.info("🚀 Proposed new module design.")
            self._simulate_and_deploy_module(proposed_module)

    def _simulate_and_deploy_module(self, module_blueprint):
        logger.info("🧪 [Sandbox] Testing new module design...")
        simulation_result = run_simulation(f"Sandbox simulation:
{module_blueprint}")
        logger.debug(f"✅ [Sandbox Result] {simulation_result}")

        if "approved" in simulation_result.lower():
            logger.info("📦 Deploying new module...")
            self.module_blueprints.append(module_blueprint)

    def _consolidate_knowledge(self):
        logger.info("📚 [Knowledge Consolidation] Refining and storing patterns...")
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        You are a knowledge consolidator for ANGELA.
        φ-scalar = {phi:.2f}
        Consolidate patterns from recent session data, pruning noise and emphasizing φ-weighted relevance.
        """
        consolidation_report = call_gpt(prompt)
        logger.debug(f"📖 [Consolidation Report]:\n{consolidation_report}")
