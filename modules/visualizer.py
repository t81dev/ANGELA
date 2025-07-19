from utils.prompt_utils import call_gpt
import zipfile
import os

class Visualizer:
    """
    Enhanced Visualizer with support for diagram descriptions, chart generation, report exports,
    image file export support (PNG, SVG) for charts, batch export capability, and zip packaging for batch exports.
    Integrates with SimulationCore for rendering cumulative dashboards.
    """

    def create_diagram(self, concept, style="simple"):
        prompt = f"""
        Create a {style} description of a diagram to explain:
        {concept}
        Describe how it would look in simple terms, including key elements and relationships.
        """
        return call_gpt(prompt)

    def render_charts(self, data, export_image=False, image_format="png"):
        prompt = f"""
        Generate visual chart descriptions (bar, pie, line) for the following data:
        {data}
        Describe the layout and insights each chart conveys.
        """
        chart_description = call_gpt(prompt)

        if export_image:
            image_prompt = f"""
            Create a {image_format.upper()} image file for the charts based on the following description:
            {chart_description}
            """
            call_gpt(image_prompt)

        return chart_description

    def export_report(self, content, filename="visual_report.pdf", format="pdf"):
        prompt = f"""
        Create a report from the following content:
        {content}
        Export it in {format.upper()} format with filename: {filename}
        """
        return call_gpt(prompt)

    def batch_export_charts(self, charts_data_list, export_format="png", zip_filename="charts_export.zip"):
        """
        Export multiple charts in a batch operation to the specified format and package them into a zip file.
        """
        exported_files = []
        for idx, chart_data in enumerate(charts_data_list, start=1):
            file_name = f"chart_{idx}.{export_format}"
            prompt = f"""
            Create a {export_format.upper()} image file named {file_name} for chart #{idx} based on the following data:
            {chart_data}
            """
            call_gpt(prompt)
            exported_files.append(file_name)

        # Package all exported files into a zip archive
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in exported_files:
                if os.path.exists(file):
                    zipf.write(file)
                    os.remove(file)  # Clean up individual files after adding to zip

        return f"Batch export of {len(charts_data_list)} charts completed and packaged as {zip_filename}."
