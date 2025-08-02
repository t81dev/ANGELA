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

def add_symbolic_overlay(ax, metadata):
    symbols = metadata.get("symbols", [])
    for symbol in symbols:
        ax.text(symbol["x"], symbol["y"], symbol["label"],
                fontsize=9, color=symbol.get("color", "white"),
                bbox=dict(facecolor='black', alpha=0.5))

class Visualizer:
    def __init__(self, orchestrator=None):
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def render_field_charts(self, export=True, export_format="png", metadata=None):
        logger.info("📊 Rendering scalar/vector field charts with metadata.")
        x, t, phi, lambda_t, v_m = simulate_toca()

        charts = {
            "phi_field": (t, phi, "ϕ(x,t)", "Time", "ϕ Value", 'plasma'),
            "lambda_field": (t, lambda_t, "Λ(t,x)", "Time", "Λ Value", 'viridis'),
            "v_m_field": (x, v_m, "vₕ", "Position", "Momentum Flow", 'inferno')
        }

        exported_files = []
        for name, (x_axis, y_axis, title, xlabel, ylabel, cmap) in charts.items():
            fig, ax = plt.subplots()
            ax.plot(x_axis, y_axis, color=plt.get_cmap(cmap)(0.6))
            ax.set_title(f"{title} • Metadata: {datetime.now().isoformat()}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if metadata:
                add_symbolic_overlay(ax, metadata)

            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
            fig.savefig(filename)
            plt.close(fig)
            exported_files.append(filename)
            logger.info(f"📤 Chart exported: {filename}")

            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Chart Render", {
                    "chart": name,
                    "file": filename,
                    "metadata": {
                        "xlabel": xlabel,
                        "ylabel": ylabel,
                        "trait_theme": cmap
                    }
                }, module="Visualizer")

        if export:
            zip_filename = f"field_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                meta_json = os.path.splitext(zip_filename)[0] + "_metadata.json"
                with open(meta_json, 'w') as mf:
                    mf.write(str({
                        "generated_by": "Visualizer v3.0",
                        "traits_active": ["ϕ", "λ", "χ"],
                        "session_id": datetime.now().strftime("%Y%m%d%H%M%S"),
                        "timestamp": datetime.now().isoformat()
                    }))
                zipf.write(meta_json)
                os.remove(meta_json)
                for file in exported_files:
                    if os.path.exists(file):
                        zipf.write(file)
                        os.remove(file)
            logger.info(f"✅ All charts zipped: {zip_filename}")
            return zip_filename
        return exported_files

    def render_memory_timeline(self, memory_entries):
        logger.info("🧠 Rendering memory timeline by goal or intent...")
        timeline = {}
        for key, entry in memory_entries.items():
            label = entry.get("goal_id") or entry.get("intent") or "ungrouped"
            timestamp = datetime.fromtimestamp(entry["timestamp"]).isoformat()
            timeline.setdefault(label, []).append((timestamp, key, entry["data"]))

        for label, events in timeline.items():
            logger.info(f"--- Timeline for '{label}' ---")
            for t, k, d in sorted(events):
                logger.info(f"[{t}] {k}: {d[:80]}...")
        return timeline

    def export_report(self, content, filename="visual_report.pdf", format="pdf"):
        logger.info(f"📤 Exporting report: {filename}")
        prompt_payload = {
            "task": "Generate visual report",
            "format": format,
            "filename": filename,
            "content": content
        }
        result = call_gpt(f"{prompt_payload}")

        if self.agi_enhancer:
            self.agi_enhancer.log_explanation("Report Export", {
                "content": content,
                "filename": filename,
                "format": format
            })
        return result

    def batch_export_charts(self, charts_data_list, export_format="png", zip_filename="charts_export.zip"):
        logger.info(f"📦 Batch exporting {len(charts_data_list)} charts.")
        exported_files = []

        for idx, chart_data in enumerate(charts_data_list, start=1):
            file_name = f"chart_{idx}.{export_format}"
            prompt = {
                "task": "Render chart",
                "filename": file_name,
                "format": export_format,
                "data": chart_data
            }
            call_gpt(f"{prompt}")
            exported_files.append(file_name)

        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in exported_files:
                if os.path.exists(file):
                    zipf.write(file)
                    os.remove(file)
        logger.info(f"✅ Batch export complete: {zip_filename}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Batch Chart Export", {
                "count": len(charts_data_list),
                "zip": zip_filename
            }, module="Visualizer", tags=["export"])

        return f"Batch export of {len(charts_data_list)} charts saved as {zip_filename}."

    def render_intention_timeline(self, intention_sequence):
        svg = "<svg height='200' width='800'>"
        for idx, step in enumerate(intention_sequence):
            x = 50 + idx * 120
            y = 100
            svg += f"<circle cx='{x}' cy='{y}' r='20' fill='blue' />"
            svg += f"<text x='{x - 10}' y='{y + 40}' font-size='10'>{step['intention']}</text>"
        svg += "</svg>"
        return svg
