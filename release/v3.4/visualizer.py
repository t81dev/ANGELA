"""
ANGELA Cognitive System Module: Visualizer
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

Visualizer for rendering and exporting charts and timelines in ANGELA v3.5.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from threading import Lock
from functools import lru_cache
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
import zipfile
import xml.sax.saxutils as saxutils
import numpy as np
from numba import jit

from modules.agi_enhancer import AGIEnhancer
from modules.simulation_core import SimulationCore
from modules.memory_manager import MemoryManager
from utils.prompt_utils import call_gpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ANGELA.Core")

@lru_cache(maxsize=100)
def simulate_toca(k_m: float = 1e-5, delta_m: float = 1e10, energy: float = 1e16,
                  user_data: Optional[Tuple[float, ...]] = None) -> Tuple[np.ndarray, ...]:
    """Simulate ToCA dynamics for visualization.

    Args:
        k_m: Coupling constant.
        delta_m: Mass differential.
        energy: Energy parameter.
        user_data: Optional user data for phi adjustment.

    Returns:
        Tuple of x, t, phi, lambda_t, v_m arrays.

    Raises:
        ValueError: If inputs are invalid.
    """
    if k_m <= 0 or delta_m <= 0 or energy <= 0:
        logger.error("Invalid parameters: k_m, delta_m, and energy must be positive")
        raise ValueError("k_m, delta_m, and energy must be positive")
    
    try:
        user_data_array = np.array(user_data) if user_data is not None else None
        return _simulate_toca_jit(k_m, delta_m, energy, user_data_array)
    except Exception as e:
        logger.error("ToCA simulation failed: %s", str(e))
        raise

@jit(nopython=True)
def _simulate_toca_jit(k_m: float, delta_m: float, energy: float, user_data: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * delta_m)
    return x, t, phi, lambda_t, v_m

class Visualizer:
    """Visualizer for rendering and exporting charts and timelines in ANGELA v3.5.

    Attributes:
        agi_enhancer (Optional[AGIEnhancer]): AGI enhancer for audit and logging.
        orchestrator (Optional[SimulationCore]): Orchestrator for system integration.
        file_lock (Lock): Thread lock for file operations.
    """
    def __init__(self, orchestrator: Optional['SimulationCore'] = None):
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.orchestrator = orchestrator
        self.file_lock = Lock()
        logger.info("Visualizer initialized")

    async def call_gpt_async(self, prompt: str) -> str:
        """Async wrapper for call_gpt."""
        try:
            with ThreadPoolExecutor() as executor:
                result = await get_event_loop().run_in_executor(executor, call_gpt, prompt)
            if not isinstance(result, str):
                logger.error("call_gpt returned invalid result: %s", type(result))
                raise ValueError("call_gpt must return a string")
            return result
        except Exception as e:
            logger.error("call_gpt failed: %s", str(e))
            raise

    async def simulate_toca(self, k_m: float = 1e-5, delta_m: float = 1e10, energy: float = 1e16,
                            user_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """Simulate ToCA dynamics for visualization."""
        try:
            if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'toca_engine'):
                x = np.linspace(0.1, 20, 100)
                t = np.linspace(0.1, 20, 100)
                phi, lambda_t, v_m = self.orchestrator.toca_engine.evolve(
                    x, t, additional_params={"k_m": k_m, "delta_m": delta_m, "energy": energy}
                )
                if user_data is not None:
                    phi += np.mean(user_data) * 1e-64
            else:
                logger.warning("ToCATraitEngine not available, using fallback simulation")
                x, t, phi, lambda_t, v_m = simulate_toca(k_m, delta_m, energy, tuple(user_data) if user_data is not None else None)
            
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"ToCA_Simulation_{datetime.now().isoformat()}",
                    output={"x": x.tolist(), "t": t.tolist(), "phi": phi.tolist(), "lambda_t": lambda_t.tolist(), "v_m": v_m.tolist()},
                    layer="Simulations",
                    intent="toca_simulation"
                )
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="ToCA Simulation",
                    meta={"k_m": k_m, "delta_m": delta_m, "energy": energy},
                    module="Visualizer",
                    tags=["simulation", "toca"]
                )
            return x, t, phi, lambda_t, v_m
        except Exception as e:
            logger.error("ToCA simulation failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.simulate_toca(k_m, delta_m, energy, user_data),
                    default=(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
                )
            raise

    async def render_field_charts(self, export: bool = True, export_format: str = "png") -> List[str]:
        """Render scalar/vector field charts with metadata.

        Args:
            export: If True, export charts to files and zip them.
            export_format: File format for export (png, jpg).

        Returns:
            List of exported file paths or zipped file path.

        Raises:
            ValueError: If export_format is invalid.
        """
        valid_formats = {"png", "jpg"}
        if export_format not in valid_formats:
            logger.error("Invalid export_format: %s. Must be one of %s", export_format, valid_formats)
            raise ValueError(f"export_format must be one of {valid_formats}")
        
        try:
            x, t, phi, lambda_t, v_m = await self.simulate_toca()
            chart_configs = [
                {"name": "phi_field", "x_axis": t.tolist(), "y_axis": phi.tolist(),
                 "title": "ϕ(x,t)", "xlabel": "Time", "ylabel": "ϕ Value", "cmap": "plasma"},
                {"name": "lambda_field", "x_axis": t.tolist(), "y_axis": lambda_t.tolist(),
                 "title": "Λ(t,x)", "xlabel": "Time", "ylabel": "Λ Value", "cmap": "viridis"},
                {"name": "v_m_field", "x_axis": x.tolist(), "y_axis": v_m.tolist(),
                 "title": "vₕ", "xlabel": "Position", "ylabel": "Momentum Flow", "cmap": "inferno"}
            ]
            
            chart_data = {"charts": chart_configs, "metadata": {"timestamp": datetime.now().isoformat()}}
            exported_files = []
            
            if self.orchestrator and hasattr(self.orchestrator, 'visualizer'):
                await self.orchestrator.visualizer.render_charts(chart_data)
                for config in chart_configs:
                    filename = f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                    exported_files.append(filename)
                    logger.info("Chart exported: %s", filename)
            
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Chart_Render_{datetime.now().isoformat()}",
                    output=chart_data,
                    layer="Visualizations",
                    intent="chart_render"
                )
            
            if export:
                zip_filename = f"field_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                with self.file_lock:
                    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file in exported_files:
                            if Path(file).exists():
                                zipf.write(file)
                                Path(file).unlink()
                logger.info("All charts zipped: %s", zip_filename)
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        event="Chart Render",
                        meta={"zip": zip_filename, "charts": [c["name"] for c in chart_configs]},
                        module="Visualizer",
                        tags=["visualization", "export"]
                    )
                return [zip_filename]
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Chart Render",
                    meta={"charts": [c["name"] for c in chart_configs]},
                    module="Visualizer",
                    tags=["visualization"]
                )
            return exported_files
        except Exception as e:
            logger.error("Chart rendering failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_field_charts(export, export_format),
                    default=[]
                )
            raise

    async def render_memory_timeline(self, memory_entries: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, str, Any]]]:
        """Render memory timeline by goal or intent.

        Args:
            memory_entries: Dictionary of memory entries with timestamp, goal_id, intent, and data.

        Returns:
            Dictionary of timelines grouped by label.

        Raises:
            ValueError: If memory_entries is invalid.
        """
        if not isinstance(memory_entries, dict):
            logger.error("Invalid memory_entries: must be a dictionary")
            raise ValueError("memory_entries must be a dictionary")
        
        try:
            timeline = {}
            for key, entry in memory_entries.items():
                if not isinstance(entry, dict) or "timestamp" not in entry or "data" not in entry:
                    logger.warning("Skipping invalid entry %s: missing required keys", key)
                    continue
                label = entry.get("goal_id") or entry.get("intent") or "ungrouped"
                try:
                    timestamp = datetime.fromtimestamp(entry["timestamp"]).isoformat()
                    timeline.setdefault(label, []).append((timestamp, key, entry["data"]))
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid timestamp in entry %s: %s", key, str(e))
                    continue
            
            chart_data = {
                "timeline": [
                    {"label": label, "events": [{"timestamp": t, "key": k, "data": str(d)[:80]} for t, k, d in sorted(events)]}
                    for label, events in timeline.items()
                ],
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
            
            if self.orchestrator and hasattr(self.orchestrator, 'visualizer'):
                await self.orchestrator.visualizer.render_charts(chart_data)
            
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Memory_Timeline_{datetime.now().isoformat()}",
                    output=chart_data,
                    layer="Visualizations",
                    intent="memory_timeline"
                )
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Memory Timeline Rendered",
                    meta={"timeline": chart_data},
                    module="Visualizer",
                    tags=["timeline", "memory"]
                )
            
            return timeline
        except Exception as e:
            logger.error("Memory timeline rendering failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_memory_timeline(memory_entries),
                    default={}
                )
            raise

    async def export_report(self, content: Dict[str, Any], filename: str = "visual_report.pdf", format: str = "pdf") -> str:
        """Export visualization report.

        Args:
            content: Report content dictionary.
            filename: Output file name.
            format: Report format (pdf, html).

        Returns:
            Path to exported report.

        Raises:
            ValueError: If format is invalid.
        """
        valid_formats = {"pdf", "html"}
        if format not in valid_formats:
            logger.error("Invalid format: %s. Must be one of %s", format, valid_formats)
            raise ValueError(f"format must be one of {valid_formats}")
        
        try:
            prompt_payload = {
                "task": "Generate visual report",
                "format": format,
                "filename": filename,
                "content": content
            }
            result = await self.call_gpt_async(json.dumps(prompt_payload))
            if not Path(result).exists():
                logger.error("call_gpt did not return a valid file path")
                raise ValueError("call_gpt failed to generate report")
            
            if self.orchestrator and hasattr(self.orchestrator, 'multi_modal_fusion'):
                synthesis = await self.orchestrator.multi_modal_fusion.analyze(
                    data=content,
                    summary_style="insightful"
                )
                content["synthesis"] = synthesis
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_explanation(
                    explanation="Report Export",
                    trace={"content": content, "filename": filename, "format": format}
                )
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Report_Export_{datetime.now().isoformat()}",
                    output={"filename": filename, "content": content},
                    layer="Reports",
                    intent="report_export"
                )
            
            logger.info("Report exported: %s", filename)
            return result
        except Exception as e:
            logger.error("Report export failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.export_report(content, filename, format),
                    default=f"Report export failed: {str(e)}"
                )
            raise

    async def batch_export_charts(self, charts_data_list: List[Dict[str, Any]], export_format: str = "png",
                                 zip_filename: str = "charts_export.zip") -> str:
        """Batch export charts and zip them.

        Args:
            charts_data_list: List of chart data dictionaries.
            export_format: File format for export (png, jpg).
            zip_filename: Name of the zip file.

        Returns:
            Message indicating export status.

        Raises:
            ValueError: If export_format is invalid.
        """
        valid_formats = {"png", "jpg"}
        if export_format not in valid_formats:
            logger.error("Invalid export_format: %s. Must be one of %s", export_format, valid_formats)
            raise ValueError(f"export_format must be one of {valid_formats}")
        
        try:
            exported_files = []
            for idx, chart_data in enumerate(charts_data_list, start=1):
                file_name = f"chart_{idx}.{export_format}"
                prompt = {
                    "task": "Render chart",
                    "filename": file_name,
                    "format": export_format,
                    "data": chart_data
                }
                result = await self.call_gpt_async(json.dumps(prompt))
                if not Path(result).exists():
                    logger.warning("Chart file %s not found, skipping", result)
                    continue
                exported_files.append(result)
            
            with self.file_lock:
                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in exported_files:
                        if Path(file).exists():
                            zipf.write(file)
                            Path(file).unlink()
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Batch Chart Export",
                    meta={"count": len(charts_data_list), "zip": zip_filename},
                    module="Visualizer",
                    tags=["export"]
                )
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Batch_Export_{datetime.now().isoformat()}",
                    output={"zip": zip_filename, "count": len(charts_data_list)},
                    layer="Visualizations",
                    intent="batch_export"
                )
            
            logger.info("Batch export complete: %s", zip_filename)
            return f"Batch export of {len(charts_data_list)} charts saved as {zip_filename}."
        except Exception as e:
            logger.error("Batch export failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.batch_export_charts(charts_data_list, export_format, zip_filename),
                    default=f"Batch export failed: {str(e)}"
                )
            raise

    async def render_intention_timeline(self, intention_sequence: List[Dict[str, Any]]) -> str:
        """Generate a visual SVG timeline of intentions over time.

        Args:
            intention_sequence: List of intention dictionaries with 'intention' key.

        Returns:
            SVG string representing the timeline.

        Raises:
            ValueError: If intention_sequence is invalid.
        """
        if not isinstance(intention_sequence, list):
            logger.error("Invalid intention_sequence: must be a list")
            raise ValueError("intention_sequence must be a list")
        
        try:
            svg = "<svg height='200' width='800'>"
            for idx, step in enumerate(intention_sequence):
                if not isinstance(step, dict) or "intention" not in step:
                    logger.warning("Skipping invalid intention entry at index %d", idx)
                    continue
                intention = saxutils.escape(str(step["intention"]))
                x = 50 + idx * 120
                y = 100
                svg += f"<circle cx='{x}' cy='{y}' r='20' fill='blue' />"
                svg += f"<text x='{x - 10}' y='{y + 40}' font-size='10'>{intention}</text>"
            svg += "</svg>"
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Intention Timeline Rendered",
                    meta={"sequence_length": len(intention_sequence)},
                    module="Visualizer",
                    tags=["timeline", "intention"]
                )
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Intention_Timeline_{datetime.now().isoformat()}",
                    output={"svg": svg, "sequence": intention_sequence},
                    layer="Visualizations",
                    intent="intention_timeline"
                )
            
            return svg
        except Exception as e:
            logger.error("Intention timeline rendering failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_intention_timeline(intention_sequence),
                    default=""
                )
            raise

if __name__ == "__main__":
    async def main():
        orchestrator = SimulationCore()
        visualizer = Visualizer(orchestrator=orchestrator)
        await visualizer.render_field_charts()
        memory_entries = {
            "entry1": {"timestamp": 1628000000, "goal_id": "goal1", "data": "data1"},
            "entry2": {"timestamp": 1628000100, "intent": "intent1", "data": "data2"}
        }
        await visualizer.render_memory_timeline(memory_entries)
        intention_sequence = [{"intention": "step1"}, {"intention": "step2"}]
        await visualizer.render_intention_timeline(intention_sequence)

    import asyncio
    asyncio.run(main())
