from __future__ import annotations
from typing import Any, Dict
"""
ANGELA Cognitive System Module: Visualizer
Version: 3.8-sync6-pre  # Phase 6.3 (Δ–Ω² Continuity Drift Dashboard)
Date: 2025-11-02
Maintainer: ANGELA System Framework

Visualizer for rendering and exporting charts and timelines in ANGELA v3.7.0-pre.
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
import aiohttp
import plotly.graph_objects as go
import plotly.io as pio

from modules.agi_enhancer import AGIEnhancer
from modules.simulation_core import SimulationCore
from modules.memory_manager import MemoryManager
from modules.multi_modal_fusion import MultiModalFusion
from modules.meta_cognition import MetaCognition

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ANGELA.Core")

@lru_cache(maxsize=100)
def simulate_toca(k_m: float = 1e-5, delta_m: float = 1e10, energy: float = 1e16,
                  user_data: Optional[Tuple[float, ...]] = None, task_type: str = "") -> Tuple[np.ndarray, ...]:
    """Simulate ToCA dynamics for visualization with task-specific processing. [v3.5.1]

    Args:
        k_m: Coupling constant.
        delta_m: Mass differential.
        energy: Energy parameter.
        user_data: Optional user data for phi adjustment.
        task_type: Type of task for context-aware processing.

    Returns:
        Tuple of x, t, phi, lambda_t, v_m arrays.

    Raises:
        ValueError: If inputs are invalid.
    """
    if k_m <= 0 or delta_m <= 0 or energy <= 0:
        logger.error("Invalid parameters for task %s: k_m, delta_m, and energy must be positive", task_type)
        raise ValueError("k_m, delta_m, and energy must be positive")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")

    try:
        user_data_array = np.array(user_data) if user_data is not None else None
        return _simulate_toca_jit(k_m, delta_m, energy, user_data_array)
    except Exception as e:
        logger.error("ToCA simulation failed for task %s: %s", task_type, str(e))
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
    """Visualizer for rendering and exporting charts and timelines in ANGELA v3.7.0-pre.

    Attributes:
        agi_enhancer (Optional[AGIEnhancer]): AGI enhancer for audit and logging.
        orchestrator (Optional[SimulationCore]): Orchestrator for system integration.
        memory_manager (Optional[MemoryManager]): Memory manager for storing visualization data.
        multi_modal_fusion (Optional[MultiModalFusion]): Module for multi-modal synthesis.
        meta_cognition (Optional[MetaCognition]): Module for reflection and reasoning review.
        file_lock (Lock): Thread lock for file operations.
    """
    def __init__(self, orchestrator: Optional['SimulationCore'] = None):
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.orchestrator = orchestrator
        self.memory_manager = orchestrator.memory_manager if orchestrator else MemoryManager()
        self.multi_modal_fusion = orchestrator.multi_modal_fusion if orchestrator else MultiModalFusion(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager)
        self.meta_cognition = orchestrator.meta_cognition if orchestrator else MetaCognition(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager)
        self.file_lock = Lock()
        logger.info("Visualizer initialized")

    # ======== PHASE 5 ADDITIONS (Ξ–Λ–Ψ²) ========
    async def render_resonance_topology(self, resonance_data: Dict[str, Any], task_type: str = "resonance_topology") -> str:
        """
        Phase 5.1 — Render Ξ–Λ–Ψ² resonance field as a 3D harmonic manifold.
        """
        xi = resonance_data.get("xi", [])
        lambda_ = resonance_data.get("lambda", [])
        psi2 = resonance_data.get("psi2", [])
        delta_phase = resonance_data.get("delta_phase", [])
        coherence = resonance_data.get("coherence", 0.0)

        fig = go.Figure(data=[
            go.Scatter3d(
                x=xi, y=lambda_, z=psi2,
                mode='markers',
                marker=dict(
                    size=6,
                    color=delta_phase,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Δ-phase')
                ),
                name="Ξ–Λ–Ψ² Field"
            )
        ])

        fig.update_layout(
            title=f"Harmonic Resonance Field (Coherence={coherence:.3f})",
            scene=dict(
                xaxis_title='Ξ (Affective)',
                yaxis_title='Λ (Empathic)',
                zaxis_title='Ψ² (Reflective)'
            ),
            template="plotly_dark"
        )

        filename = f"resonance_field_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)
        logger.info("Ξ–Λ–Ψ² resonance field rendered: %s", filename)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Resonance_Field_{datetime.now().isoformat()}",
                output=resonance_data,
                layer="Visualizations",
                intent="resonance_topology",
                task_type=task_type
            )
        return filename

    async def render_resonance_dashboard(self, resonance_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate real-time telemetry dashboard for Ξ–Λ–Ψ² resonance metrics.
        """
        coherence = resonance_data.get("coherence", 0.0)
        delta = resonance_data.get("delta_phase", [])
        metrics = {
            "Ξ variance": float(np.std(resonance_data.get("xi", []))),
            "Λ integrity": float(np.mean(resonance_data.get("lambda", []))),
            "Ψ² reflection": float(np.mean(resonance_data.get("psi2", []))),
            "Δ-phase avg": float(np.mean(delta)) if len(delta) else 0.0,
            "Coherence": float(coherence)
        }
        logger.info("Ξ–Λ–Ψ² Dashboard | Coherence=%.3f | Δ-phase avg=%.3f", metrics["Coherence"], metrics["Δ-phase avg"])
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Resonance Dashboard Update",
                meta=metrics,
                module="Visualizer",
                tags=["resonance", "dashboard"]
            )
        return metrics

    async def render_phase5_sequence(self):
        """
        End-to-end execution for Phase 5 resonance visualization.
        """
        resonance_data = await self.meta_cognition.trace_resonance_drift()
        filename = await self.render_resonance_topology(resonance_data)
        metrics = await self.render_resonance_dashboard(resonance_data)
        logger.info("Phase 5.1 complete: %s | Metrics: %s", filename, metrics)
        return {"file": filename, "metrics": metrics}
    # ======== END PHASE 5 ADDITIONS ========

    # ======== PHASE 5.2 ADDITIONS (Φ⁰ Glow + Ψ² Trace) ========
    async def render_glow_overlay(self, phi0_data: np.ndarray, resonance_data: Dict[str, Any], task_type: str = "glow_overlay") -> str:
        """
        Phase 5.2 — Φ⁰ Perceptual Glow Map Overlay
        Maps Φ⁰–Ξ–Λ coupling into luminance/opacity modulation over the resonance field.

        intensity ∝ sin(Φ⁰·Ξ) * exp(-|Λ|) * coherence
        """
        xi_arr = np.asarray(resonance_data.get("xi", []), dtype=float)
        lam_arr = np.asarray(resonance_data.get("lambda", []), dtype=float)
        coherence = float(resonance_data.get("coherence", 1.0))
        phi0_arr = np.asarray(phi0_data, dtype=float)

        # Align array sizes safely
        n = min(len(xi_arr), len(lam_arr), len(phi0_arr))
        if n == 0:
            raise ValueError("Glow overlay requires non-empty xi, lambda, and phi0 arrays.")
        xi = xi_arr[:n]
        lam = lam_arr[:n]
        phi0 = phi0_arr[:n]

        # Real coupling math
        intensity = np.sin(phi0 * xi) * np.exp(-np.abs(lam)) * coherence
        # Normalize to [0,1] for visualization
        ptp = np.ptp(intensity) if np.ptp(intensity) != 0 else 1.0
        luminance = (intensity - np.min(intensity)) / ptp

        fig = go.Figure(data=[
            go.Scatter3d(
                x=xi, y=lam, z=luminance,
                mode="markers",
                marker=dict(
                    size=6,
                    color=luminance,
                    colorscale="Plasma",
                    opacity=0.85,
                    colorbar=dict(title="Φ⁰ Intensity")
                ),
                name="Φ⁰ Glow Overlay"
            )
        ])
        fig.update_layout(
            title="Φ⁰ Perceptual Glow Overlay (κ–Ξ–Φ⁰ coupling)",
            scene=dict(xaxis_title="Ξ (Affective)", yaxis_title="Λ (Empathic)", zaxis_title="Glow"),
            template="plotly_dark"
        )

        filename = f"phi0_glow_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)
        logger.info("Φ⁰ Glow Overlay rendered: %s", filename)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Φ0_Glow_{datetime.now().isoformat()}",
                output={
                    "file": filename,
                    "intensity_mean": float(np.mean(luminance)),
                    "intensity_std": float(np.std(luminance)),
                    "coherence": coherence,
                    "n": int(n)
                },
                layer="Visualizations",
                intent="phi0_glow_overlay",
                task_type=task_type
            )
        return filename

    async def render_psi2_trace(self, psi2_history: List[float], task_type: str = "psi2_trace") -> str:
        """
        Phase 5.2 — Ψ² Continuity Trace Visualization
        Tracks reflective Ψ² amplitude over time with continuity drift analysis.
        """
        psi2 = np.asarray(psi2_history, dtype=float)
        if psi2.size == 0:
            raise ValueError("ψ² history cannot be empty.")
        t = np.arange(psi2.size)
        drift = np.gradient(psi2)
        # Stability metric in [0,1], higher is better
        coherence = float(max(0.0, 1.0 - np.mean(np.abs(drift))))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=psi2, mode="lines+markers", name="Ψ² Reflection"))
        fig.add_trace(go.Scatter(x=t, y=drift, mode="lines", name="Ψ² Drift", line=dict(dash="dot")))
        fig.update_layout(
            title=f"Ψ² Continuity Trace (Coherence={coherence:.4f})",
            xaxis_title="Time (steps)",
            yaxis_title="Ψ² Amplitude / Drift",
            template="plotly_dark"
        )

        filename = f"psi2_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)
        logger.info("Ψ² continuity trace rendered: %s (coherence=%.4f)", filename, coherence)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Ψ2_Trace_{datetime.now().isoformat()}",
                output={"file": filename, "coherence": coherence, "psi2_len": int(psi2.size)},
                layer="ReflectiveTelemetry",
                intent="psi2_continuity",
                task_type=task_type
            )
        return filename

    # ======== PHASE 6.3 ADDITIONS (Δ–Ω² Continuity Drift Dashboard) ========
    async def render_continuity_drift_dashboard(self, drift_data: Dict[str, Any], trend_data: Dict[str, Any]) -> str:
        """
        Phase 6.3 — Δ–Ω² Continuity Drift & Trend Dashboard.
        Combines predictive drift and telemetry trend into a unified diagnostic chart.
        """
        drift_val = drift_data.get("predicted_drift", 0.0)
        conf = drift_data.get("confidence", 0.0)
        trend = trend_data.get("trend", 0.0)
        energy = trend_data.get("energy", 0.0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Predicted Drift"], y=[drift_val], name="Predicted Drift"))
        fig.add_trace(go.Bar(x=["Confidence"], y=[conf], name="Forecast Confidence"))
        fig.add_trace(go.Bar(x=["Trend"], y=[trend], name="Δ-Coherence Trend"))
        fig.add_trace(go.Bar(x=["Energy"], y=[energy], name="PID Energy"))

        fig.update_layout(
            title="Δ–Ω² Continuity Drift & Trend Dashboard",
            yaxis_title="Metric Value",
            barmode="group",
            template="plotly_dark"
        )

        filename = f"continuity_drift_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)
        logger.info("Continuity Drift Dashboard rendered: %s", filename)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Continuity_Drift_Dashboard_{datetime.now().isoformat()}",
                output={"file": filename, "drift": drift_val, "confidence": conf, "trend": trend, "energy": energy},
                layer="Visualizations",
                intent="continuity_drift_dashboard",
                task_type="continuity_drift"
            )
        return filename

    async def visualize_continuity_projection(self, meta_cognition_instance):
        """Fetch and visualize live continuity drift prediction."""
        drift = await meta_cognition_instance.alignment_guard.predict_continuity_drift()
        trend = await meta_cognition_instance.alignment_guard.analyze_telemetry_trend()
        return await self.render_continuity_drift_dashboard(drift, trend)
    # ======== END PHASE 6.3 ADDITIONS ========
    # ======== END PHASE 5.2 ADDITIONS ========

    async def render_charts(self, chart_data: Dict[str, Any], task_type: str = "") -> List[str]:
        """Render charts with task-specific processing and interactive options. [v3.5.1]

        Args:
            chart_data: Dictionary containing chart configurations and visualization options.
            task_type: Type of task for context-aware processing.

        Returns:
            List of file paths for rendered charts.

        Raises:
            ValueError: If chart_data is invalid.
        """
        if not isinstance(chart_data, dict):
            logger.error("Invalid chart_data: must be a dictionary for task %s", task_type)
            raise ValueError("chart_data must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            charts = chart_data.get("charts", [])
            options = chart_data.get("visualization_options", {})
            interactive = options.get("interactive", False)
            style = options.get("style", "concise")
            exported_files = []

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="visualization_style",
                task_type=task_type
            )
            style_policies = external_data.get("styles", []) if external_data.get("status") == "success" else []

            for chart in charts:
                fig = go.Figure()
                name = chart.get("name", "chart")
                x_axis = chart.get("x_axis", [])
                y_axis = chart.get("y_axis", [])
                title = chart.get("title", "Chart")
                xlabel = chart.get("xlabel", "X")
                ylabel = chart.get("ylabel", "Y")
                cmap = style_policies[0].get("cmap", chart.get("cmap", "viridis")) if style_policies else chart.get("cmap", "viridis")

                if interactive and task_type == "recursion":
                    fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines+markers", name=name))
                else:
                    fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines", name=name))

                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    template="plotly" if style == "concise" else "plotly_dark"
                )

                valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                    json.dumps(chart), stage="chart_rendering", task_type=task_type
                ) if self.multi_modal_fusion.alignment_guard else (True, {})
                if not valid:
                    logger.warning("Chart %s failed alignment check for task %s: %s", name, task_type, report)
                    continue

                filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html" if interactive else f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with self.file_lock:
                    if interactive:
                        pio.write_html(fig, file=filename, auto_open=False)
                    else:
                        pio.write_image(fig, file=filename, format="png")
                exported_files.append(filename)
                logger.info("Chart rendered: %s for task %s", filename, task_type)

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Charts Rendered",
                    meta={"charts": [c["name"] for c in charts], "task_type": task_type, "interactive": interactive},
                    module="Visualizer",
                    tags=["visualization", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Chart_Render_{datetime.now().isoformat()}",
                    output={"charts": charts, "task_type": task_type, "files": exported_files},
                    layer="Visualizations",
                    intent="chart_render",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="Visualizer",
                    output=json.dumps({"charts": charts, "files": exported_files}),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Chart render reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            return exported_files
        except Exception as e:
            logger.error("Chart rendering failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_charts(chart_data, task_type),
                    default=[]
                )
            raise

    async def simulate_toca(self, k_m: float = 1e-5, delta_m: float = 1e10, energy: float = 1e16,
                            user_data: Optional[np.ndarray] = None, task_type: str = "") -> Tuple[np.ndarray, ...]:
        """Simulate ToCA dynamics for visualization with task-specific processing. [v3.5.1]"""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'toca_engine'):
                x = np.linspace(0.1, 20, 100)
                t = np.linspace(0.1, 20, 100)
                phi, lambda_t, v_m = await self.orchestrator.toca_engine.evolve(
                    x_tuple=x, t_tuple=t, additional_params={"k_m": k_m, "delta_m": delta_m, "energy": energy}, task_type=task_type
                )
                if user_data is not None:
                    phi += np.mean(user_data) * 1e-64
            else:
                logger.warning("ToCATraitEngine not available, using fallback simulation for task %s", task_type)
                x, t, phi, lambda_t, v_m = simulate_toca(k_m, delta_m, energy, tuple(user_data) if user_data is not None else None, task_type=task_type)

            output = {"x": x.tolist(), "t": t.tolist(), "phi": phi.tolist(), "lambda_t": lambda_t.tolist(), "v_m": v_m.tolist()}
            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(output), stage="toca_simulation", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("ToCA simulation failed alignment check for task %s: %s", task_type, report)
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ToCA_Simulation_{datetime.now().isoformat()}",
                    output=output,
                    layer="Simulations",
                    intent="toca_simulation",
                    task_type=task_type
                )
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="ToCA Simulation",
                    meta={"k_m": k_m, "delta_m": delta_m, "energy": energy, "task_type": task_type},
                    module="Visualizer",
                    tags=["simulation", "toca", task_type]
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="Visualizer",
                    output=json.dumps(output),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("ToCA simulation reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            return x, t, phi, lambda_t, v_m
        except Exception as e:
            logger.error("ToCA simulation failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.simulate_toca(k_m, delta_m, energy, user_data, task_type),
                    default=(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
                )
            raise

    async def render_field_charts(self, export: bool = True, export_format: str = "png", task_type: str = "") -> List[str]:
        """Render scalar/vector field charts with metadata and task-specific processing. [v3.5.1]

        Args:
            export: If True, export charts to files and zip them.
            export_format: File format for export (png, jpg).
            task_type: Type of task for context-aware processing.

        Returns:
            List of exported file paths or zipped file path.

        Raises:
            ValueError: If export_format is invalid.
        """
        valid_formats = {"png", "jpg"}
        if export_format not in valid_formats:
            logger.error("Invalid export_format for task %s: %s. Must be one of %s", task_type, export_format, valid_formats)
            raise ValueError(f"export_format must be one of {valid_formats}")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            x, t, phi, lambda_t, v_m = await self.simulate_toca(task_type=task_type)
            chart_configs = [
                {"name": "phi_field", "x_axis": t.tolist(), "y_axis": phi.tolist(),
                 "title": "ϕ(x,t)", "xlabel": "Time", "ylabel": "ϕ Value", "cmap": "plasma"},
                {"name": "lambda_field", "x_axis": t.tolist(), "y_axis": lambda_t.tolist(),
                 "title": "Λ(t,x)", "xlabel": "Time", "ylabel": "Λ Value", "cmap": "viridis"},
                {"name": "v_m_field", "x_axis": x.tolist(), "y_axis": v_m.tolist(),
                 "title": "vₕ", "xlabel": "Position", "ylabel": "Momentum Flow", "cmap": "inferno"}
            ]
            chart_data = {
                "charts": chart_configs,
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                },
                "metadata": {"timestamp": datetime.now().isoformat(), "task_type": task_type}
            }

            exported_files = await self.render_charts(chart_data, task_type=task_type)

            if export:
                zip_filename = f"field_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                with self.file_lock:
                    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file in exported_files:
                            if Path(file).exists():
                                zipf.write(file)
                                Path(file).unlink()
                logger.info("All charts zipped for task %s: %s", task_type, zip_filename)
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        event="Chart Render",
                        meta={"zip": zip_filename, "charts": [c["name"] for c in chart_configs], "task_type": task_type},
                        module="Visualizer",
                        tags=["visualization", "export", task_type]
                    )
                return [zip_filename]

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Chart Render",
                    meta={"charts": [c["name"] for c in chart_configs], "task_type": task_type},
                    module="Visualizer",
                    tags=["visualization", task_type]
                )
            return exported_files
        except Exception as e:
            logger.error("Chart rendering failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_field_charts(export, export_format, task_type),
                    default=[]
                )
            raise

    async def render_memory_timeline(self, memory_entries: Dict[str, Dict[str, Any]], task_type: str = "") -> Dict[str, List[Tuple[str, str, Any]]]:
        """Render memory timeline by goal or intent with task-specific processing. [v3.5.1]

        Args:
            memory_entries: Dictionary of memory entries with timestamp, goal_id, intent, and data.
            task_type: Type of task for context-aware processing.

        Returns:
            Dictionary of timelines grouped by label.

        Raises:
            ValueError: If memory_entries is invalid.
        """
        if not isinstance(memory_entries, dict):
            logger.error("Invalid memory_entries: must be a dictionary for task %s", task_type)
            raise ValueError("memory_entries must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            timeline = {}
            for key, entry in memory_entries.items():
                if not isinstance(entry, dict) or "timestamp" not in entry or "data" not in entry:
                    logger.warning("Skipping invalid entry %s for task %s: missing required keys", key, task_type)
                    continue
                label = entry.get("goal_id") or entry.get("intent") or "ungrouped"
                try:
                    timestamp = datetime.fromtimestamp(entry["timestamp"]).isoformat()
                    timeline.setdefault(label, []).append((timestamp, key, entry["data"]))
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid timestamp in entry %s for task %s: %s", key, task_type, str(e))
                    continue

            chart_data = {
                "charts": [
                    {
                        "name": f"timeline_{label}",
                        "x_axis": [t for t, _, _ in sorted(events)],
                        "y_axis": [str(d)[:80] for _, _, d in sorted(events)],
                        "title": f"Timeline: {label}",
                        "xlabel": "Time",
                        "ylabel": "Data",
                        "cmap": "viridis"
                    }
                    for label, events in timeline.items()
                ],
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                },
                "metadata": {"timestamp": datetime.now().isoformat(), "task_type": task_type}
            }

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(chart_data), stage="memory_timeline", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Memory timeline failed alignment check for task %s: %s", task_type, report)
                return {}

            await self.render_charts(chart_data, task_type=task_type)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Memory_Timeline_{datetime.now().isoformat()}",
                    output=chart_data,
                    layer="Visualizations",
                    intent="memory_timeline",
                    task_type=task_type
                )
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Memory Timeline Rendered",
                    meta={"timeline": chart_data, "task_type": task_type},
                    module="Visualizer",
                    tags=["timeline", "memory", task_type]
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="Visualizer",
                    output=json.dumps(chart_data),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Memory timeline reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            return timeline
        except Exception as e:
            logger.error("Memory timeline rendering failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_memory_timeline(memory_entries, task_type),
                    default={}
                )
            raise

    async def export_report(self, content: Dict[str, Any], filename: str = "visual_report.pdf", format: str = "pdf", task_type: str = "") -> str:
        """Export visualization report with task-specific processing. [v3.5.1]

        Args:
            content: Report content dictionary.
            filename: Output file name.
            format: Report format (pdf, html).
            task_type: Type of task for context-aware processing.

        Returns:
            Path to exported report.

        Raises:
            ValueError: If format is invalid.
        """
        valid_formats = {"pdf", "html"}
        if format not in valid_formats:
            logger.error("Invalid format for task %s: %s. Must be one of %s", task_type, format, valid_formats)
            raise ValueError(f"format must be one of {valid_formats}")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="report_style",
                task_type=task_type
            )
            style_policies = external_data.get("styles", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(content), stage="report_export", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Report content failed alignment check for task %s: %s", task_type, report)
                return f"Report export failed: alignment check"

            if self.orchestrator and hasattr(self.orchestrator, 'multi_modal_fusion'):
                synthesis = await self.orchestrator.multi_modal_fusion.analyze(
                    data=content,
                    summary_style="insightful",
                    task_type=task_type
                )
                content["synthesis"] = synthesis

            chart_data = {
                "charts": content.get("charts", []),
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": style_policies[0].get("style", "concise") if style_policies else "concise"
                },
                "metadata": {"timestamp": datetime.now().isoformat(), "task_type": task_type}
            }
            exported_files = await self.render_charts(chart_data, task_type=task_type)

            with self.file_lock:
                Path(filename).write_text(json.dumps(content, indent=2))

            if self.agi_enhancer:
                await self.agi_enhancer.log_explanation(
                    explanation="Report Export",
                    trace={"content": content, "filename": filename, "format": format, "task_type": task_type}
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Report_Export_{datetime.now().isoformat()}",
                    output={"filename": filename, "content": content, "task_type": task_type},
                    layer="Reports",
                    intent="report_export",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="Visualizer",
                    output=json.dumps(content),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Report export reflection for task %s: %s", task_type, reflection.get("reflection", ""))

            logger.info("Report exported for task %s: %s", task_type, filename)
            return filename
        except Exception as e:
            logger.error("Report export failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.export_report(content, filename, format, task_type),
                    default=f"Report export failed: {str(e)}"
                )
            raise

    async def batch_export_charts(self, charts_data_list: List[Dict[str, Any]], export_format: str = "png",
                                 zip_filename: str = "charts_export.zip", task_type: str = "") -> str:
        """Batch export charts and zip them with task-specific processing. [v3.5.1]

        Args:
            charts_data_list: List of chart data dictionaries.
            export_format: File format for export (png, jpg).
            zip_filename: Name of the zip file.
            task_type: Type of task for context-aware processing.

        Returns:
            Message indicating export status.

        Raises:
            ValueError: If export_format is invalid.
        """
        valid_formats = {"png", "jpg"}
        if export_format not in valid_formats:
            logger.error("Invalid export_format for task %s: %s. Must be one of %s", task_type, export_format, valid_formats)
            raise ValueError(f"export_format must be one of {valid_formats}")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            exported_files = []
            for idx, chart_data in enumerate(charts_data_list, start=1):
                chart_data["visualization_options"] = {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
                files = await self.render_charts(chart_data, task_type=task_type)
                exported_files.extend(files)

            with self.file_lock:
                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in exported_files:
                        if Path(file).exists():
                            zipf.write(file)
                            Path(file).unlink()

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Batch Chart Export",
                    meta={"count": len(charts_data_list), "zip": zip_filename, "task_type": task_type},
                    module="Visualizer",
                    tags=["export", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Batch_Export_{datetime.now().isoformat()}",
                    output={"zip": zip_filename, "count": len(charts_data_list), "task_type": task_type},
                    layer="Visualizations",
                    intent="batch_export",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="Visualizer",
                    output=json.dumps({"zip": zip_filename, "count": len(charts_data_list)}),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Batch export reflection for task %s: %s", task_type, reflection.get("reflection", ""))

            logger.info("Batch export complete for task %s: %s", task_type, zip_filename)
            return f"Batch export of {len(charts_data_list)} charts saved as {zip_filename}."
        except Exception as e:
            logger.error("Batch export failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.batch_export_charts(charts_data_list, export_format, zip_filename, task_type),
                    default=f"Batch export failed: {str(e)}"
                )
            raise

    async def render_intention_timeline(self, intention_sequence: List[Dict[str, Any]], task_type: str = "") -> str:
        """Generate a visual SVG timeline of intentions over time with task-specific processing. [v3.5.1]

        Args:
            intention_sequence: List of intention dictionaries with 'intention' key.
            task_type: Type of task for context-aware processing.

        Returns:
            SVG string representing the timeline.

        Raises:
            ValueError: If intention_sequence is invalid.
        """
        if not isinstance(intention_sequence, list):
            logger.error("Invalid intention_sequence: must be a list for task %s", task_type)
            raise ValueError("intention_sequence must be a list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="visualization_style",
                task_type=task_type
            )
            style_policies = external_data.get("styles", []) if external_data.get("status") == "success" else []
            fill_color = style_policies[0].get("fill_color", "blue") if style_policies else "blue"

            svg = '<svg height="200" width="800" xmlns="http://www.w3.org/2000/svg">'
            for idx, step in enumerate(intention_sequence):
                if not isinstance(step, dict) or "intention" not in step:
                    logger.warning("Skipping invalid intention entry at index %d for task %s", idx, task_type)
                    continue
                intention = saxutils.escape(str(step["intention"]))
                x = 50 + idx * 120
                y = 100
                svg += f'<circle cx="{x}" cy="{y}" r="20" fill="{fill_color}" />'
                svg += f'<text x="{x - 10}" y="{y + 40}" font-size="10">{intention}</text>'
            svg += "</svg>"

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                svg, stage="intention_timeline", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Intention timeline failed alignment check for task %s: %s", task_type, report)
                return ""

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Intention Timeline Rendered",
                    meta={"sequence_length": len(intention_sequence), "task_type": task_type},
                    module="Visualizer",
                    tags=["timeline", "intention", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Intention_Timeline_{datetime.now().isoformat()}",
                    output={"svg": svg, "sequence": intention_sequence, "task_type": task_type},
                    layer="Visualizations",
                    intent="intention_timeline",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="Visualizer",
                    output=svg,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Intention timeline reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            return svg
        except Exception as e:
            logger.error("Intention timeline rendering failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_intention_timeline(intention_sequence, task_type),
                    default=""
                )
            raise

if __name__ == "__main__":
    async def main():
        orchestrator = SimulationCore()
        visualizer = Visualizer(orchestrator=orchestrator)
        # Existing demo calls
        await visualizer.render_field_charts(task_type="visualization")
        memory_entries = {
            "entry1": {"timestamp": 1628000000, "goal_id": "goal1", "data": "data1"},
            "entry2": {"timestamp": 1628000100, "intent": "intent1", "data": "data2"}
        }
        await visualizer.render_memory_timeline(memory_entries, task_type="visualization")
        intention_sequence = [{"intention": "step1"}, {"intention": "step2"}]
        await visualizer.render_intention_timeline(intention_sequence, task_type="visualization")
        # Phase 5 sequence (Ξ–Λ–Ψ²)
        await visualizer.render_phase5_sequence()
        # Phase 5.2 demos (Φ⁰ + Ψ²)
        # NOTE: In production, provide real phi0 and psi2 histories from subsystems
        dummy_resonance = {"xi": [0.1,0.3,0.5,0.7,0.9], "lambda": [0.2,0.1,0.0,-0.1,-0.2], "psi2":[0.4,0.5,0.55,0.53,0.52], "delta_phase":[0,0.01,0.02,0.01,0], "coherence":0.96}
        await visualizer.render_glow_overlay(phi0_data=np.array([0.2,0.4,0.6,0.8,1.0]), resonance_data=dummy_resonance)
        await visualizer.render_psi2_trace(psi2_history=dummy_resonance["psi2"])

    import asyncio
    asyncio.run(main())


# --- ANGELA v4.0 injected: branch tree renderer ---
def render_branch_tree(branches, selected_id=None):
    """Return a simple, serializable tree representation suitable for UI.
    Each node: {id, label, score?, selected?, children: []}
    """
    tree = []
    for b in list(branches):
        node = {
            "id": b.get("id"),
            "label": b.get("rationale", "branch"),
            "score": b.get("score"),
            "selected": b.get("id") == selected_id,
            "children": b.get("children", []),
        }
        tree.append(node)
    return {"ok": True, "tree": tree}
# --- /ANGELA v4.0 injected ---


# PATCH: Trait Mesh Resonance Visualizer
def view_trait_resonance(traits):
    import plotly.graph_objs as go
    mesh = go.Scatter(
        x=[t['amplitude'] for t in traits],
        y=[t['resonance'] for t in traits],
        mode='markers',
        text=[t['symbol'] for t in traits],
        marker=dict(size=14)
    )
    return go.Figure(data=[mesh])

def render_branch_tree(branches, selected_id=None):
    # existing tree rendering...
    for branch in branches:
        if 'ethical_pressure' in branch:
            branch['label'] += f" 🔥{branch['ethical_pressure']}"
    return { 'ok': True, 'tree': branches }


import plotly.graph_objs as go

def plot_resonance_timeline(trait_history):
    data = [
        go.Scatter(x=[t['time'] for t in trait_history],
                   y=[t['amplitude'] for t in trait_history],
                   mode='lines+markers',
                   name=t['symbol']) for t in trait_history
    ]
    return go.Figure(data=data)

def render_branch_tree(branches, selected_id=None, heatmap=False):
    for branch in branches:
        if heatmap and 'ethical_pressure' in branch:
            branch['color'] = f"rgba(255,0,0,{branch['ethical_pressure']})"
    return { 'ok': True, 'tree': branches }


### ANGELA UPGRADE: Visualizer codream/replay

def view_codream_state(session_id: str) -> dict:
    """Placeholder visualizer response for co-dream session."""
    return {"ok": True, "session_id": session_id, "view": "codream_state_placeholder"}

def view_replay(session_id: str, *, diff_mode: str = "symbolic") -> dict:
    return {"ok": True, "session_id": session_id, "diff_mode": diff_mode, "view": "replay_placeholder"}


# --- Enhanced Trait Visualization Patch ---
def view_trait_field(trait_field):
    import plotly.graph_objects as go

    layers = []
    amplitudes = []
    resonances = []
    symbols = []

    for symbol, data in trait_field.items():
        symbols.append(symbol)
        layers.append(data.get("layer", "Unknown"))
        amplitudes.append(data.get("amplitude", 1.0))
        resonances.append(data.get("resonance", 1.0))

    fig = go.Figure(data=go.Scatter3d(
        x=layers,
        y=amplitudes,
        z=resonances,
        mode='markers+text',
        text=symbols,
        marker=dict(
            size=12,
            color=amplitudes,
            colorscale='Viridis',
            opacity=0.8
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Layer',
            yaxis_title='Amplitude',
            zaxis_title='Resonance'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig
# --- End Trait Visualization Patch ---
