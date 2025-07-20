from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.MetaCognition")

class MetaCognition:
    """
    Meta-Cognitive Engine v1.4.0
    - Real-time self-diagnostics and adaptive corrections
    - Reasoning trace analysis and critique
    - Ecosystem health monitoring and optimization suggestions
    """

    def review_reasoning(self, reasoning_trace):
        """
        Analyze and critique a reasoning trace for:
        - Logical flaws
        - Biases or omissions
        - Missing steps in reasoning
        Return improved reasoning with annotations.
        """
        logger.info("Reviewing reasoning trace for flaws and improvements.")
        prompt = f"""
        You are a meta-cognitive auditor reviewing reasoning logic.
        Analyze the following reasoning trace for:
        - Logical flaws
        - Biases or omissions
        - Missing steps

        Provide:
        1. A detailed critique with annotations.
        2. An improved version of the reasoning trace.

        Reasoning Trace:
        {reasoning_trace}
        """
        response = call_gpt(prompt)
        logger.debug(f"Meta-cognition critique:\n{response}")
        return response

    def pre_action_alignment_check(self, action_plan):
        """
        Validate an action plan in a simulated environment for:
        - Ethical alignment
        - Safety risks
        - Unintended side effects
        """
        logger.info("Validating action plan for alignment and safety.")
        prompt = f"""
        You are simulating the following action plan:
        {action_plan}

        Evaluate for:
        - Ethical alignment with human values
        - Potential safety risks
        - Unintended side effects

        Provide:
        - Approval status (Approve/Deny)
        - Suggested refinements if needed
        """
        validation = call_gpt(prompt)
        approved = "approve" in validation.lower()
        logger.info(f"Pre-action alignment check result: {'✅ Approved' if approved else '🚫 Denied'}")
        return approved, validation

    def run_self_diagnostics(self):
        """
        Run a self-check for meta-cognition consistency and system performance.
        """
        logger.info("Running self-diagnostics for meta-cognition module.")
        prompt = """
        Perform a meta-cognitive self-diagnostic:
        - Evaluate current reasoning and planning modules
        - Flag inconsistencies or performance degradation
        - Suggest immediate corrective actions if needed
        """
        report = call_gpt(prompt)
        logger.debug(f"Self-diagnostics report:\n{report}")
        return report

    def propose_optimizations(self, agent_stats):
        """
        Suggest optimizations for embodied agents and their interactions.
        """
        logger.info("Proposing optimizations for agent ecosystem.")
        prompt = f"""
        You are analyzing embodied agents in a distributed cognitive system.
        Based on the following stats:
        {agent_stats}

        Provide recommendations for:
        - Sensor/actuator upgrades
        - Improved action planning
        - More efficient collaboration between agents
        """
        recommendations = call_gpt(prompt)
        logger.debug(f"Optimization recommendations:\n{recommendations}")
        return recommendations
