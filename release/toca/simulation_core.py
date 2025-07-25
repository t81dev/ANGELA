from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from datetime import datetime
from index import zeta_consequence, theta_causality, rho_agency
import time
import logging
import numpy as np
from modules.toca_simulation import generate_phi_field

logger = logging.getLogger("ANGELA.SimulationCore")

class SimulationCore:
    """
    SimulationCore v1.5.0
    - Multi-agent simulation and dynamic scenario evolution
    - Counterfactual reasoning (what-if agent decisions)
    - Aggregate risk scoring with live dashboards
    - Supports exporting dashboards and cumulative simulation logs
    - Trait-weighted causality and agency analysis
    - Scalar field integration from ToCA (phi field dynamics)
    """

    def __init__(self):
        self.visualizer = Visualizer()
        self.simulation_history = []

    def run(self, results, context=None, scenarios=3, agents=2, export_report=False, export_format="pdf"):
        logger.info(f"🎲 Running simulation with {agents} agents and {scenarios} scenarios.")
        t = time.time() % 1e-18
        causality = theta_causality(t)
        agency = rho_agency(t)

        phi_modulation = generate_phi_field(np.linspace(0.1, 20, 100))

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
        - ϕ(x,t) scalar field (ToCA) dynamically modulates agent momentum
        - Use ϕ to adjust simulation dynamics: higher ϕ implies greater inertia, lower ϕ increases flexibility

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
        return environment_simulation
