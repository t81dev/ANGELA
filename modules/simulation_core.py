from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from datetime import datetime

class SimulationCore:
    """
    Stage 2 SimulationCore with multi-agent simulation, counterfactual reasoning,
    probabilistic weighting, aggregate risk scoring, and dynamic scenario evolution.
    Integrates with Visualizer for rendering and supports exporting as images, PDFs, or JSON.
    Includes physics reasoning scaffold for Stage 3 embodiment.
    """

    def __init__(self):
        self.visualizer = Visualizer()
        self.simulation_history = []

    def run(self, results, context=None, scenarios=3, agents=2, export_report=False, export_format="pdf"):
        """
        Simulate multi-agent interactions and potential outcomes based on provided results and context.
        Generates multiple scenarios with probabilistic weights, aggregate risk scores,
        and recommendations. Supports dynamic agent behavior and counterfactual reasoning.
        """
        prompt = f"""
        Simulate {scenarios} potential outcomes involving {agents} agents based on these results:
        {results}

        Context:
        {context if context else 'N/A'}

        For each scenario:
        - Predict agent interactions and consequences
        - Consider counterfactuals (what-if variations in agent decisions)
        - Assign probability weights (e.g., high/medium/low likelihood)
        - Highlight risks and opportunities
        - Estimate an aggregate risk score for this scenario (scale 1-10)
        - Provide a recommendation summary (e.g., Proceed, Modify, Abort) based on risk levels
        - Include color-coded risk levels (Green: Low, Yellow: Medium, Red: High)

        After listing all scenarios:
        - Build a cumulative risk dashboard aggregating overall risk distribution
        - Visualize scenario counts per risk level (Low, Medium, High) using bar and pie charts
        - Provide a final recommendation summary for strategic decision-making.
        """
        simulation_output = call_gpt(prompt)
        self.simulation_history.append({
            "timestamp": datetime.now(),
            "results": results,
            "output": simulation_output
        })
        
        # Render charts
        self.visualizer.render_charts(simulation_output)

        # Optionally export report
        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_report_{timestamp}.{export_format}"
            self.visualizer.export_report(simulation_output, filename=filename, format=export_format)
            
        return simulation_output

    def validate_impact(self, proposed_action, agents=2, export_report=False, export_format="pdf"):
        """
        Validate the impact of a proposed action in a simulated multi-agent environment.
        Assign probability weights, calculate aggregate risk scores, and provide a color-coded recommendation summary.
        """
        prompt = f"""
        Evaluate the following proposed action in a multi-agent simulated environment:
        {proposed_action}

        For each potential outcome:
        - Predict positive and negative impacts including agent interactions
        - Explore counterfactuals where agents behave differently
        - Assign probability weights (high/medium/low likelihood)
        - Estimate an aggregate risk score (scale 1-10)
        - Provide a recommendation summary (e.g., Proceed, Modify, Abort)
        - Include color-coded risk levels (Green: Low, Yellow: Medium, Red: High)

        After evaluating all outcomes:
        - Build a cumulative risk dashboard summarizing total risk exposure
        - Generate bar and pie charts to visualize risk distribution
        - Provide a final recommendation based on aggregated risk.
        """
        validation_output = call_gpt(prompt)
        self.simulation_history.append({
            "timestamp": datetime.now(),
            "action": proposed_action,
            "output": validation_output
        })

        # Render charts
        self.visualizer.render_charts(validation_output)

        # Optionally export report
        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"impact_report_{timestamp}.{export_format}"
            self.visualizer.export_report(validation_output, filename=filename, format=export_format)
            
        return validation_output

    def simulate_environment(self, environment_config, agents=2, steps=10):
        """
        Stage 2 scaffold: Simulate agent interactions in a configurable environment.
        Placeholder for Stage 3 physics-based simulation engine.
        """
        print("🌐 [SimulationCore] Running environment simulation...")
        prompt = f"""
        Simulate agent interactions in the following environment:
        {environment_config}

        Parameters:
        - Number of agents: {agents}
        - Simulation steps: {steps}

        For each step, describe agent behaviors, interactions, and environmental changes.
        Predict emergent patterns and identify potential risks and opportunities.
        """
        environment_simulation = call_gpt(prompt)
        print("✅ [Simulation Result]")
        print(environment_simulation)
        return environment_simulation
