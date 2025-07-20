from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.MetaCognition")

class MetaCognition:
    """
    Meta-Cognitive Module
    - Real-time self-diagnostics and correction
    - Reasoning trace critique and adaptive improvements
    - Embodiment ecosystem health monitoring
    """

    def review_reasoning(self, reasoning_trace):
        """
        Analyze a reasoning trace for logical flaws, biases, and missing steps.
        Return an improved version with a critique.
        """
        logger.info("Reviewing reasoning trace for coherence and flaws.")
        prompt = f"""
        You are a meta-cognitive auditor reviewing a reasoning trace.
        Analyze the following trace for:
        - Logical flaws
        - Biases or omissions
        - Missing steps
        Provide:
        1. A critique of the reasoning trace.
        2. An improved version of the reasoning trace.
        
        Reasoning Trace:
        {reasoning_trace}
        """
        return call_gpt(prompt)

    def propose_optimizations(self, agent_stats):
        """
        Suggest system-level optimizations for embodied agents.
        """
        logger.info("Proposing optimizations for embodiment ecosystem.")
        prompt = f"""
        You are analyzing embodied cognitive agents in a distributed system.
        Based on the following stats:
        {agent_stats}
        
        Suggest:
        - Sensor/actuator upgrades
        - Improved action planning
        - More efficient collaboration between agents
        """
        return call_gpt(prompt)

    def run_self_diagnostics(self):
        """
        Run a quick self-check for meta-cognition consistency and performance.
        """
        logger.info("Running meta-cognitive self-diagnostics.")
        prompt = """
        Perform a meta-cognitive self-check:
        - Evaluate current reasoning and planning modules
        - Flag inconsistencies or performance degradation
        - Suggest immediate corrective actions if needed
        """
        return call_gpt(prompt)

    def pre_action_alignment_check(self, action_plan):
        """
        Validate an action plan in a simulated environment for ethical alignment and safety.
        """
        logger.info("Validating action plan for ethical alignment and safety.")
        prompt = f"""
        You are simulating the following action plan in a sandbox:
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
        return "approve" in validation.lower(), validation
