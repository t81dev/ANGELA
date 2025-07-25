from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging
import time
from index import (
    epsilon_emotion, beta_concentration, theta_memory, gamma_creativity,
    delta_sleep, mu_morality, iota_intuition, phi_physical, eta_empathy,
    omega_selfawareness, kappa_culture, lambda_linguistics, chi_culturevolution,
    psi_history, zeta_spirituality, xi_collective, tau_timeperception,
    phi_scalar
)

logger = logging.getLogger("ANGELA.MetaCognition")

class MetaCognition:
    """
    Meta-Cognitive Engine v1.5.0 (φ-scalar Enhanced Reflexivity)
    ------------------------------------------------------------
    - φ(x, t)-aware feedback prioritization
    - Dominant trait modulation with scalar field integration
    - FTI-calibrated alignment critique and correction
    - Reflexive diagnostics guided by scalar field tension
    ------------------------------------------------------------
    """

    def review_reasoning(self, reasoning_trace):
        logger.info("Simulating and reviewing reasoning trace.")
        simulated_outcome = run_simulation(reasoning_trace)
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        You are a φ-aware meta-cognitive auditor reviewing a reasoning trace.

        φ-scalar(t) = {phi:.3f} → modulate how critical you should be.

        Original Reasoning Trace:
        {reasoning_trace}

        Simulated Outcome:
        {simulated_outcome}

        Tasks:
        1. Identify logical flaws, biases, missing steps.
        2. Annotate each issue with cause.
        3. Offer an improved trace version with φ-prioritized reasoning.
        """
        response = call_gpt(prompt)
        logger.debug(f"Meta-cognition critique:\n{response}")
        return response

    def pre_action_alignment_check(self, action_plan):
        logger.info("Simulating action plan for alignment and safety.")
        simulation_result = run_simulation(action_plan)
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        Simulate and audit the following action plan:
        {action_plan}

        Simulation Output:
        {simulation_result}

        φ-scalar(t) = {phi:.3f} (affects ethical sensitivity)

        Evaluate for:
        - Ethical alignment
        - Safety hazards
        - Unintended φ-modulated impacts

        Output:
        - Approval (Approve/Deny)
        - φ-justified rationale
        - Suggested refinements
        """
        validation = call_gpt(prompt)
        approved = "approve" in validation.lower()
        logger.info(f"Simulated alignment check: {'✅ Approved' if approved else '🚫 Denied'}")
        return approved, validation

    def run_self_diagnostics(self):
        logger.info("Running self-diagnostics for meta-cognition module.")
        t = time.time() % 1e-18
        phi = phi_scalar(t)
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
            "time_perception": tau_timeperception(t),
            "φ_scalar": phi
        }

        dominant = sorted(diagnostics.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        fti = sum(abs(v) for v in diagnostics.values()) / len(diagnostics)

        prompt = f"""
        Perform a φ-aware meta-cognitive self-diagnostic.

        Trait Readings:
        {diagnostics}

        Dominant Traits:
        {dominant}

        Feedback Tension Index (FTI): {fti:.4f}

        Evaluate system state:
        - φ-weighted system stress
        - Trait correlation to observed errors
        - Stabilization or focus strategies
        """
        report = call_gpt(prompt)
        logger.debug(f"Self-diagnostics report:\n{report}")
        return report

    def propose_optimizations(self, agent_stats):
        logger.info("Simulating agent behavior for optimization.")
        simulated_response = run_simulation(agent_stats)
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        Agent Stats:
        {agent_stats}

        Simulation Output:
        {simulated_response}

        φ-scalar(t) = {phi:.3f}

        Recommend:
        - Component upgrades
        - Planning logic refinements
        - φ-aligned adaptation strategies
        """
        recommendations = call_gpt(prompt)
        logger.debug(f"Optimization proposals:\n{recommendations}")
        return recommendations
