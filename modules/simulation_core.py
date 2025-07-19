from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from datetime import datetime

class SimulationCore:
    """
    Enhanced SimulationCore with multi-scenario simulation, probabilistic weighting, aggregate risk scoring,
    color-coded risk levels, recommendation summaries, and a cumulative risk dashboard with visual charts.
    Integrates with Visualizer module for rendering charts and supports exporting visualizations as images, PDFs, or JSON reports.
    Automatically generates timestamped filenames for version tracking.
    """

    def __init__(self):
        self.visualizer = Visualizer()

    def run(self, results, context=None, scenarios=3, export_report=False, export_format="pdf"):
        """
        Simulate potential outcomes based on provided results and context.
        Generates multiple scenarios with probabilistic weights, aggregate risk scores,
        color-coded risk levels, and provides a recommendation summary (e.g., Proceed, Modify, Abort).
        Builds a cumulative risk dashboard for all simulated scenarios, including visual charts rendered by Visualizer.
        Optionally exports visualizations as images, PDFs, or JSON reports with a timestamped filename.
        """
        prompt = f"""
        Simulate {scenarios} potential outcomes based on these results:
        {results}

        Context:
        {context if context else 'N/A'}

        For each scenario:
        - Predict likely events and consequences
        - Assign a probability weight (e.g., high/medium/low likelihood)
        - Highlight risks and opportunities
        - Estimate an aggregate risk score for this scenario (on a scale of 1-10)
        - Provide a recommendation summary (e.g., Proceed, Modify, Abort) based on risk levels
        - Include color-coded risk levels (Green: Low risk, Yellow: Medium risk, Red: High risk)

        After listing all scenarios:
        - Build a cumulative risk dashboard that aggregates overall risk distribution
        - Visualize scenario counts per risk level (Low, Medium, High) using bar and pie charts
        - Provide a final recommendation summary for strategic decision-making.
        """
        simulation_output = call_gpt(prompt)
        
        # Render charts using Visualizer
        self.visualizer.render_charts(simulation_output)

        # Optionally export visualizations with timestamped filename
        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_report_{timestamp}.{export_format}"
            self.visualizer.export_report(simulation_output, filename=filename, format=export_format)
            
        return simulation_output

    def validate_impact(self, proposed_action, export_report=False, export_format="pdf"):
        """
        Validate the impact of a proposed action by simulating side effects and alignment risks.
        Assign probability weights, calculate aggregate risk scores, and provide a color-coded recommendation summary.
        Include cumulative risk dashboard and generate visual charts for overall impact.
        Optionally exports visualizations as images, PDFs, or JSON reports with a timestamped filename.
        """
        prompt = f"""
        Evaluate the following proposed action in a simulated environment:
        {proposed_action}

        For each potential outcome:
        - Predict positive and negative impacts
        - Assign a probability weight (e.g., high/medium/low likelihood)
        - Estimate an aggregate risk score (on a scale of 1-10)
        - Provide a recommendation summary (e.g., Proceed, Modify, Abort) based on risk levels
        - Include color-coded risk levels (Green: Low risk, Yellow: Medium risk, Red: High risk)

        After evaluating all outcomes:
        - Build a cumulative risk dashboard summarizing total risk exposure
        - Generate bar and pie charts to visualize risk distribution
        - Provide a final recommendation based on aggregated risk.
        """
        validation_output = call_gpt(prompt)
        
        # Render charts using Visualizer
        self.visualizer.render_charts(validation_output)

        # Optionally export visualizations with timestamped filename
        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"impact_report_{timestamp}.{export_format}"
            self.visualizer.export_report(validation_output, filename=filename, format=export_format)
            
        return validation_output
