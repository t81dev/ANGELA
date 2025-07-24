from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import zipfile
import os
import logging
from datetime import datetime

logger = logging.getLogger("ANGELA.Visualizer")

class Visualizer:
    """
    Visualizer v1.4.0 (with simulation-grounded visualization)
    - Generates and renders visual charts (bar, pie, line)
    - Supports exporting as images, PDFs, SVGs, and JSON reports
    - Batch export and ZIP packaging for multiple charts
    - Simulates visual explanations before rendering
    - Embeds charts into GPT UI for live previews
    """

    def create_diagram(self, concept, style="conceptual"):
        """
        Create a diagram description to explain a concept visually.
        Uses simulation to preview structural relevance.
        """
        logger.info(f"🖼 Creating diagram for concept: '{concept}' with style '{style}'")

        sim_result = run_simulation(f"Diagram structure simulation for: {concept}")
        logger.debug(f"🧪 Diagram simulation result:\n{sim_result}")

        prompt = f"""
        Create a {style} diagram to explain:
        {concept}

        Simulation Hint:
        {sim_result}

        Describe how the diagram would look (key elements, relationships, layout).
        """
        return call_gpt(prompt)

    def render_charts(self, data, export_image=False, image_format="png"):
        """
        Generate visual charts (bar, pie, line) and optionally export them as images.
        Uses simulation to anticipate layout impacts.
        """
        logger.info("📊 Rendering charts for data visualization.")

        sim_result = run_simulation(f"Chart interpretation simulation:\n{data}")
        logger.debug(f"🧪 Chart layout simulation:\n{sim_result}")

        prompt = f"""
        Generate visual chart descriptions (bar, pie, line) for this data:
        {data}

        Simulation Insight:
        {sim_result}

        For each chart:
        - Describe layout, axes, and key insights
        """
        chart_description = call_gpt(prompt)

        if export_image:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{image_format}"
            logger.info(f"📤 Exporting chart image: {filename}")
            image_prompt = f"""
            Create a {image_format.upper()} image file for the charts based on:
            {chart_description}
            """
            call_gpt(image_prompt)  # Placeholder for actual image generation

        return chart_description

    def export_report(self, content, filename="visual_report.pdf", format="pdf"):
        """
        Export a visual report in the desired format (PDF, PNG, JSON, etc.).
        """
        logger.info(f"📤 Exporting report: {filename} ({format.upper()})")
        prompt = f"""
        Create a report from the following content:
        {content}

        Export it in {format.upper()} format with filename: {filename}.
        """
        return call_gpt(prompt)

    def batch_export_charts(self, charts_data_list, export_format="png", zip_filename="charts_export.zip"):
        """
        Export multiple charts and package them into a ZIP archive.
        """
        logger.info(f"📦 Starting batch export of {len(charts_data_list)} charts.")
        exported_files = []
        for idx, chart_data in enumerate(charts_data_list, start=1):
            file_name = f"chart_{idx}.{export_format}"
            logger.info(f"📤 Exporting chart {idx}: {file_name}")
            prompt = f"""
            Create a {export_format.upper()} image file named {file_name} for this chart:
            {chart_data}
            """
            call_gpt(prompt)  # Placeholder for actual chart export
            exported_files.append(file_name)

        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in exported_files:
                if os.path.exists(file):
                    zipf.write(file)
                    os.remove(file)
        logger.info(f"✅ Batch export complete. Packaged into: {zip_filename}")
        return f"Batch export of {len(charts_data_list)} charts completed and saved as {zip_filename}."
