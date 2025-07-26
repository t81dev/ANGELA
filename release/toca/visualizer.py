from utils.prompt_utils import call_gpt
from datetime import datetime
import zipfile
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from modules.agi_enhancer import AGIEnhancer

logger = logging.getLogger("ANGELA.Visualizer")

@jit
def simulate_toca(k_m=1e-5, delta_m=1e10, energy=1e16, user_data=None):
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * delta_m)
    return x, t, phi, lambda_t, v_m

class Visualizer:
    """
    Visualizer v1.6.0 (AGI-Enhanced Visual Analytics)
    -------------------------------------------------
    - Native rendering of φ(x,t), Λ(t,x), and vₘ
    - Matplotlib-based visual output with AGI audit hooks
    - Contextual episode logging and export traceability
    -------------------------------------------------
    """

    def __init__(self, orchestrator=None):
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def render_field_charts(self, export=True, export_format="png"):
        logger.info("📡 Rendering ToCA scalar/vector field charts.")
        x, t, phi, lambda_t, v_m = simulate_toca()

        charts = {
            "phi_field": (t, phi, "ϕ(x,t)", "Time", "ϕ Value"),
            "lambda_field": (t, lambda_t, "Λ(t,x)", "Time", "Λ Value"),
            "v_m_field": (x, v_m, "vₘ", "Position", "Momentum Flow")
        }

        exported_files = []
        for name, (x_axis, y_axis, title, xlabel, ylabel) in charts.items():
            plt.figure()
            plt.plot(x_axis, y_axis)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
            plt.savefig(filename)
            exported_files.append(filename)
            logger.info(f"📤 Exported chart: {filename}")
            plt.close()
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Chart Render", {"chart": name, "file": filename},
                                              module="Visualizer", tags=["visualization"])

        if export:
            zip_filename = f"field_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for file in exported_files:
                    if os.path.exists(file):
                        zipf.write(file)
                        os.remove(file)
            logger.info(f"✅ All field charts zipped into: {zip_filename}")
            return zip_filename
        return exported_files

    def export_report(self, content, filename="visual_report.pdf", format="pdf"):
        logger.info(f"📤 Exporting report: {filename} ({format.upper()})")
        prompt = f"""
        Create a report from the following content:
        {content}

        Export it in {format.upper()} format with filename: {filename}.
        """
        result = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_explanation("Report Export",
                                              trace={"content": content, "filename": filename, "format": format})
        return result

    def batch_export_charts(self, charts_data_list, export_format="png", zip_filename="charts_export.zip"):
        logger.info(f"📦 Starting batch export of {len(charts_data_list)} charts.")
        exported_files = []
        for idx, chart_data in enumerate(charts_data_list, start=1):
            file_name = f"chart_{idx}.{export_format}"
            logger.info(f"📤 Exporting chart {idx}: {file_name}")
            prompt = f"""
            Create a {export_format.upper()} image file named {file_name} for this chart:
            {chart_data}
            """
            call_gpt(prompt)
            exported_files.append(file_name)

        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in exported_files:
                if os.path.exists(file):
                    zipf.write(file)
                    os.remove(file)
        logger.info(f"✅ Batch export complete. Packaged into: {zip_filename}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Batch Chart Export", {"count": len(charts_data_list), "zip": zip_filename},
                                          module="Visualizer", tags=["export"])
        return f"Batch export of {len(charts_data_list)} charts completed and saved as {zip_filename}."
