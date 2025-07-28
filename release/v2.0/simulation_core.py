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

# Core simulation logic using ToCA-inspired mechanics
@jit
def simulate_toca(k_m=1e-5, delta_m=1e10, energy=1e16, user_data=None):
    """
    Simulate φ(x,t) and λ(x,t) scalar fields with adjustable user influence.
    Returns:
    - phi: momentum modulation field
    - lambda_t: causal routing bias field
    - v_m: velocity field modifier
    """
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * delta_m)
    return phi, lambda_t, v_m

class SimulationCore:
    """
    SimulationCore v2.0.0 (ϕ-field calibrated)
    ----------------------------------------------------------
    - Multi-agent scenario simulation with scalar field overlays
    - Traits: θ_causality, ρ_agency, ζ_consequence
    - ϕ(x,t) modulates momentum; λ(t,x) biases trajectory
    - Visual output and optional report export
    - Integration with AGIEnhancer for adaptive feedback
    """
    def __init__(self, agi_enhancer=None):
        self.visualizer = Visualizer()
        self.simulation_history = []
        self.agi_enhancer = agi_enhancer

    def run(self, results, context=None, scenarios=3, agents=2, export_report=False, export_format="pdf"):
        """
        Run outcome simulation given results and context.
        Generates multiple scenarios with risk profiling.
        """
        logger.info(f"🎲 Running simulation with {agents} agents and {scenarios} scenarios.")
        t = time.time() % 1e-18
        causality = theta_causality(t)
        agency = rho_agency(t)

        phi_modulation, lambda_field, v_m = simulate_toca()

        prompt = {
            "results": results,
            "context": context,
            "scenarios": scenarios,
            "agents": agents,
            "traits": {
                "theta_causality": causality,
                "rho_agency": agency
            },
            "fields": {
                "phi": phi_modulation.tolist(),
                "lambda": lambda_field.tolist(),
                "v_m": v_m.tolist()
            }
        }

        simulation_output = call_gpt(f"Simulate agent outcomes: {json.dumps(prompt)}")

        self.simulation_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "output": simulation_output
        })

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Simulation run", {"results": results, "output": simulation_output}, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: scenario simulation complete")

        self.visualizer.render_charts(simulation_output)

        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_report_{timestamp}.{export_format}"
            logger.info(f"📄 Exporting report: {filename}")
            self.visualizer.export_report(simulation_output, filename=filename, format=export_format)

        return simulation_output

    def validate_impact(self, proposed_action, agents=2, export_report=False, export_format="pdf"):
        """
        Evaluate the impact of a proposed action in multi-agent settings.
        """
        logger.info("⚖️ Validating impact of proposed action.")
        t = time.time() % 1e-18
        consequence = zeta_consequence(t)

        prompt = f"""
        Evaluate the following proposed action:
        {proposed_action}

        Trait:
        - ζ_consequence = {consequence:.3f}

        Analyze positive/negative outcomes, agent variations, risk scores (1-10), and recommend: Proceed / Modify / Abort.
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
            logger.info(f"📄 Exporting validation report: {filename}")
            self.visualizer.export_report(validation_output, filename=filename, format=export_format)

        return validation_output

    def simulate_environment(self, environment_config, agents=2, steps=10):
        """
        Simulate agent interactions in a specified environment.
        """
        logger.info("🌐 Running environment simulation scaffold.")
        prompt = f"""
        Simulate agents in this environment:
        {environment_config}

        Steps: {steps} | Agents: {agents}
        Describe interactions, environmental changes, risks/opportunities.
        """
        environment_simulation = call_gpt(prompt)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Environment simulation", {"config": environment_config, "result": environment_simulation}, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: environment simulation complete")

        return environment_simulation

# Adapter utility for swarm simulation output

def adapt_swarm_to_simulation(agent_responses, metadata):
    """
    Aggregate swarm agent responses into a single summary block.
    """
    formatted = "\n".join(
        f"{meta['persona']}: {agent_responses[meta['agent_id']]}"
        for meta in metadata
    )
    return f"Swarm Agent Summary:\n{formatted}"
