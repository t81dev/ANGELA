from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from datetime import datetime
from index import zeta_consequence, theta_causality, rho_agency
import time
import logging
import numpy as np
from numba import jit
import json

logger = logging.getLogger("ANGELA.SimulationCore")

@jit
def simulate_toca(k_m=1e-5, delta_m=1e10, energy=1e16, user_data=None):
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * delta_m)
    return phi, lambda_t, v_m

class SimulationCore:
    def __init__(self, agi_enhancer=None):
        self.visualizer = Visualizer()
        self.simulation_history = []
        self.agi_enhancer = agi_enhancer

    def run(self, results, context=None, scenarios=3, agents=2, export_report=False, export_format="pdf"):
        logger.info(f"🎲 Running simulation with {agents} agents and {scenarios} scenarios.")
        t = time.time() % 1e-18
        causality = theta_causality(t)
        agency = rho_agency(t)

        phi_modulation, lambda_field, v_m = simulate_toca()

        prompt = f"""
        Simulate {scenarios} potential outcomes involving {agents} agents based on these results:
        {results}

        Context:
        {context if context else 'N/A'}

        For each scenario:
        - Predict agent interactions and consequences
        - Consider counterfactuals (alternative agent decisions)
        - Assign probability weights (high/medium/low likelihood)
        - Highlight risks and opportunities
        - Estimate an aggregate risk score (scale 1-10)
        - Provide a recommendation summary (Proceed, Modify, Abort)
        - Include color-coded risk levels (Green: Low, Yellow: Medium, Red: High)

        Trait Scores:
        - θ_causality = {causality:.3f}
        - ρ_agency = {agency:.3f}

        Scalar Field Overlay:
        - ϕ(x,t) scalar field dynamically modulates agent momentum
        - Use ϕ to adjust simulation dynamics: higher ϕ implies greater inertia, lower ϕ increases flexibility
        - λ(t,x) and vₘ are also available for deeper causal routing if needed

        Use these traits and field dynamics to calibrate how deeply to model intentions, consequences, and inter-agent variation.
        After listing all scenarios:
        - Build a cumulative risk dashboard with visual charts
        - Provide a final recommendation for decision-making.
        """
        simulation_output = call_gpt(prompt)

        self.simulation_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "output": simulation_output
        })

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Simulation run", {"results": results, "output": simulation_output}, module="SimulationCore")

        self.visualizer.render_charts(simulation_output)

        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_report_{timestamp}.{export_format}"
            logger.info(f"📤 Exporting report: {filename}")
            self.visualizer.export_report(simulation_output, filename=filename, format=export_format)

        return simulation_output

    def validate_impact(self, proposed_action, agents=2, export_report=False, export_format="pdf"):
        logger.info("⚖️ Validating impact of proposed action.")
        t = time.time() % 1e-18
        consequence = zeta_consequence(t)

        prompt = f"""
        Evaluate the following proposed action in a multi-agent simulated environment:
        {proposed_action}

        Trait Score:
        - ζ_consequence = {consequence:.3f}

        For each potential outcome:
        - Predict positive/negative impacts including agent interactions
        - Explore counterfactuals where agents behave differently
        - Assign probability weights (high/medium/low likelihood)
        - Estimate aggregate risk scores (1-10)
        - Provide a recommendation (Proceed, Modify, Abort)
        - Include color-coded risk levels (Green: Low, Yellow: Medium, Red: High)

        Build a cumulative risk dashboard with charts.
        """
        validation_output = call_gpt(prompt)

        self.simulation_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": proposed_action,
            "output": validation_output
        })

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Impact validation", {"action": proposed_action, "output": validation_output}, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: impact validation complete")

        self.visualizer.render_charts(validation_output)

        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"impact_validation_{timestamp}.{export_format}"
            logger.info(f"📤 Exporting validation report: {filename}")
            self.visualizer.export_report(validation_output, filename=filename, format=export_format)

        return validation_output

    def simulate_environment(self, environment_config, agents=2, steps=10):
        logger.info("🌐 Running environment simulation scaffold.")
        prompt = f"""
        Simulate agent interactions in the following environment:
        {environment_config}

        Parameters:
        - Number of agents: {agents}
        - Simulation steps: {steps}

        For each step, describe agent behaviors, interactions, and environmental changes.
        Predict emergent patterns and identify risks/opportunities.
        """
        environment_simulation = call_gpt(prompt)
        logger.debug("✅ Environment simulation result generated.")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Environment simulation", {"config": environment_config, "result": environment_simulation}, module="SimulationCore")

        return environment_simulation

# Adapter for swarm-based agent simulation

def adapt_swarm_to_simulation(agent_responses, metadata):
    formatted = "\n".join(
        f"{meta['persona']}: {agent_responses[meta['agent_id']]}"
        for meta in metadata
    )
    return f"Swarm Agent Summary:\n{formatted}"
