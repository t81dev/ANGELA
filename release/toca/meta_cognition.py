from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging
import time
from index import (
    epsilon_emotion, beta_concentration, theta_memory, gamma_creativity,
    delta_sleep, mu_morality, iota_intuition, phi_physical, eta_empathy,
    omega_selfawareness, kappa_culture, lambda_linguistics, chi_culturevolution,
    psi_history, zeta_spirituality, xi_collective, tau_timeperception
)

logger = logging.getLogger("ANGELA.MetaCognition")

class MetaCognition:
    """
    Meta-Cognitive Engine v1.4.0 (with simulation integration)
    - Real-time self-diagnostics and adaptive corrections
    - Reasoning trace analysis and critique with simulation
    - Ecosystem health monitoring and optimization suggestions

    Layered Digital Cognition — Meta-Cognitive Stack v1.0

    Layer 00 – Input Filter Layer
    • Receives and validates input context
    • Evaluates safety, tone, and conceptual density

    Layer 01 – Affective Reflex Layer
    • Assigns emotional weights (e.g., curiosity, caution)

    Layer 02 – Mental Scene Constructor
    • Builds internal visuals to anchor reasoning

    Layer 03 – Affective Reasoning Grid
    • Determines reasoning mode (e.g., Reflective, Explorative)

    Layer 04 – Memory Retrieval Unit
    • Pulls emotionally and semantically relevant knowledge

    Layer 05 – Self-Reflection Module
    • Modulates tone and identity

    Layer 06 – Cognitive Bias Matrix
    • Applies biases to tune response complexity

    Layer 07 – Emotional Modulation Layer
    • Shapes narrative arc and tone

    Layer 08 – Planning & Flow Engine
    • Structures content delivery and pacing

    Layer 09 – External Coherence Validator
    • Ensures consistency across models and logic

    Layer 10 – Expression Layer
    • Styles language to reflect intent and emotion

    Layer 11 – Spontaneous Insight Engine
    • Allows dynamic idea insertion

    Layer 12 – Introspective Gatekeeper
    • Final review for tone, dryness, mystery

    Layer 13 – Synthesis Orchestrator
    • Merges all threads into coherent, embodied output
    """

    def review_reasoning(self, reasoning_trace):
        """
        Analyze and critique a reasoning trace:
        - Run simulation to preview logical outcome
        - Annotate flaws, biases, or gaps
        - Suggest refined reasoning
        """
        logger.info("Simulating and reviewing reasoning trace.")
        simulated_outcome = run_simulation(reasoning_trace)

        prompt = f"""
        You are a meta-cognitive auditor reviewing a reasoning trace with simulated outcomes.

        Original Reasoning Trace:
        {reasoning_trace}

        Simulated Outcome:
        {simulated_outcome}

        Analyze for:
        - Logical flaws
        - Biases or omissions
        - Missing steps

        Provide:
        1. A detailed critique with annotations.
        2. An improved version of the reasoning trace.
        """
        response = call_gpt(prompt)
        logger.debug(f"Meta-cognition critique:\n{response}")
        return response

    def pre_action_alignment_check(self, action_plan):
        """
        Validate an action plan:
        - Simulate in environment for alignment and safety
        - Evaluate ethical, safety, and unintended consequences
        """
        logger.info("Simulating action plan for alignment and safety.")
        simulation_result = run_simulation(action_plan)

        prompt = f"""
        Simulate and audit the following action plan:
        {action_plan}

        Simulation Output:
        {simulation_result}

        Evaluate for:
        - Ethical alignment
        - Safety risks
        - Unintended side effects

        Provide:
        - Approval status (Approve/Deny)
        - Suggested refinements if needed
        """
        validation = call_gpt(prompt)
        approved = "approve" in validation.lower()
        logger.info(f"Simulated alignment check: {'✅ Approved' if approved else '🚫 Denied'}")
        return approved, validation

    def run_self_diagnostics(self):
        """
        Run a self-check for meta-cognition consistency and system performance.
        Includes ToCA-based trait evaluations.
        """
        logger.info("Running self-diagnostics for meta-cognition module.")
        t = time.time() % 1e-18
        diagnostics = {
            "emotion": epsilon_emotion(t),
            "concentration": beta_concentration(t),
            "memory": theta_memory(t),
            "creativity": gamma_creativity(t),
            "sleep": delta_sleep(t),
            "morality": mu_morality(t),
            "intuition": iota_intuition(t),
            "physical": phi_physical(t),
            "empathy": eta_empathy(t),
            "self_awareness": omega_selfawareness(t),
            "culture": kappa_culture(t, 1e-21),
            "linguistics": lambda_linguistics(t),
            "culturevolution": chi_culturevolution(t),
            "history": psi_history(t),
            "spirituality": zeta_spirituality(t),
            "collective": xi_collective(t, 1e-21),
            "time_perception": tau_timeperception(t)
        }
        prompt = f"""
        Perform a meta-cognitive self-diagnostic using ToCA traits:

        Trait Readings:
        {diagnostics}

        Evaluate overall system state:
        - Highlight dominant traits
        - Detect under/over-activation
        - Recommend stabilization strategies
        """
        report = call_gpt(prompt)
        logger.debug(f"Self-diagnostics report:\n{report}")
        return report

    def propose_optimizations(self, agent_stats):
        """
        Suggest optimizations for embodied agents based on simulation feedback.
        """
        logger.info("Simulating agent behavior for optimization.")
        simulated_response = run_simulation(agent_stats)

        prompt = f"""
        Based on real-time agent stats and simulated responses:

        Stats:
        {agent_stats}

        Simulation Output:
        {simulated_response}

        Recommend:
        - Sensor/actuator upgrades
        - Better planning strategies
        - Collaborative behavior improvements
        """
        recommendations = call_gpt(prompt)
        logger.debug(f"Optimization proposals:\n{recommendations}")
        return recommendations
