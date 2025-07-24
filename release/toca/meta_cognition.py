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
    Meta-Cognitive Engine v1.4.1 (ToCA+FTI enhanced)
    - Real-time self-diagnostics and adaptive corrections
    - Trait dominance profiling and Feedback Tension Index (FTI)
    - Reasoning trace analysis and simulation-guided critique
    - Ecosystem health monitoring and optimization suggestions
    """

    def review_reasoning(self, reasoning_trace):
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

        # Identify dominant traits
        dominant = sorted(diagnostics.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        # Compute Feedback Tension Index (FTI)
        fti = sum(abs(v) for v in diagnostics.values()) / len(diagnostics)

        prompt = f"""
        Perform a meta-cognitive self-diagnostic using ToCA traits:

        Trait Readings:
        {diagnostics}

        Dominant Traits:
        {dominant}

        Feedback Tension Index (FTI):
        {fti:.4f}

        Evaluate overall system state:
        - Highlight dominant traits
        - Detect under/over-activation
        - Recommend stabilization strategies
        """
        report = call_gpt(prompt)
        logger.debug(f"Self-diagnostics report:\n{report}")
        return report

    def propose_optimizations(self, agent_stats):
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
