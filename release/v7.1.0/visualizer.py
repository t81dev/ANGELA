from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
"""
ANGELA Cognitive System Module: Visualizer
Version: 3.9-stage7.3
Stage: VII.3 — Council-Resonant Integration (Ψ²Ω² ↔ μΩ² ↔ ΞΛ, Council-Gated Swarm Continuity)
Date: 2025-11-05
Maintainer: ANGELA System Framework

Visualizer for rendering and exporting charts, dashboards, and council/swarm views
in ANGELA v6.0.1 / Stage VII.3.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from functools import lru_cache
import zipfile
import xml.sax.saxutils as saxutils

import numpy as np
from numba import jit
import plotly.graph_objects as go
import plotly.io as pio

# These imports match your original structure
from modules.agi_enhancer import AGIEnhancer
from modules.simulation_core import SimulationCore
from modules.memory_manager import MemoryManager
from modules.multi_modal_fusion import MultiModalFusion
from modules.meta_cognition import MetaCognition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ANGELA.Core")


# ============================================================
# ToCA SIM (kept from your file)
# ============================================================
@lru_cache(maxsize=100)
def simulate_toca(
    k_m: float = 1e-5,
    delta_m: float = 1e10,
    energy: float = 1e16,
    user_data: Optional[Tuple[float, ...]] = None,
    task_type: str = ""
) -> Tuple[np.ndarray, ...]:
    """Simulate ToCA dynamics for visualization. [v3.5.1]"""
    if k_m <= 0 or delta_m <= 0 or energy <= 0:
        logger.error(
            "Invalid parameters for task %s: k_m, delta_m, and energy must be positive",
            task_type,
        )
        raise ValueError("k_m, delta_m, and energy must be positive")
    if not isinstance(task_type, str):
        raise TypeError("task_type must be a string")

    try:
        user_data_array = np.array(user_data) if user_data is not None else None
        return _simulate_toca_jit(k_m, delta_m, energy, user_data_array)
    except Exception as e:
        logger.error("ToCA simulation failed for task %s: %s", task_type, str(e))
        raise


@jit(nopython=True)
def _simulate_toca_jit(
    k_m: float,
    delta_m: float,
    energy: float,
    user_data: Optional[np.ndarray],
) -> Tuple[np.ndarray, ...]:
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x) ** 2)) * (1 + v_m * delta_m)
    return x, t, phi, lambda_t, v_m


# ============================================================
# VISUALIZER CLASS
# ============================================================
class Visualizer:
    """
    Visualizer for rendering and exporting charts and timelines.
    Stage VII.3 version: adds council-resonant dashboards.
    """

    def __init__(self, orchestrator: Optional["SimulationCore"] = None):
        self.orchestrator = orchestrator
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

        # sensible fallbacks
        self.memory_manager = (
            orchestrator.memory_manager if orchestrator else MemoryManager()
        )
        self.multi_modal_fusion = (
            orchestrator.multi_modal_fusion
            if orchestrator
            else MultiModalFusion(
                agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager
            )
        )
        self.meta_cognition = (
            orchestrator.meta_cognition
            if orchestrator
            else MetaCognition(
                agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager
            )
        )

        self.file_lock = Lock()
        logger.info("Visualizer initialized (Stage VII.3)")

    # ========================================================
    # PHASE 5 — Ξ–Λ–Ψ² resonance visualizations (kept)
    # ========================================================
    async def render_resonance_topology(
        self, resonance_data: Dict[str, Any], task_type: str = "resonance_topology"
    ) -> str:
        xi = resonance_data.get("xi", [])
        lambda_ = resonance_data.get("lambda", [])
        psi2 = resonance_data.get("psi2", [])
        delta_phase = resonance_data.get("delta_phase", [])
        coherence = float(resonance_data.get("coherence", 0.0))

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xi,
                    y=lambda_,
                    z=psi2,
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=delta_phase,
                        colorscale="Viridis",
                        opacity=0.8,
                        colorbar=dict(title="Δ-phase"),
                    ),
                    name="Ξ–Λ–Ψ² Field",
                )
            ]
        )
        fig.update_layout(
            title=f"Harmonic Resonance Field (Coherence={coherence:.3f})",
            scene=dict(
                xaxis_title="Ξ (Affective)",
                yaxis_title="Λ (Empathic)",
                zaxis_title="Ψ² (Reflective)",
            ),
            template="plotly_dark",
        )

        filename = f"resonance_field_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Resonance_Field_{datetime.now().isoformat()}",
                output=resonance_data,
                layer="Visualizations",
                intent="resonance_topology",
                task_type=task_type,
            )
        return filename

    async def render_resonance_dashboard(
        self, resonance_data: Dict[str, Any]
    ) -> Dict[str, float]:
        coherence = float(resonance_data.get("coherence", 0.0))
        delta = resonance_data.get("delta_phase", [])
        metrics = {
            "Ξ variance": float(np.std(resonance_data.get("xi", []))),
            "Λ integrity": float(np.mean(resonance_data.get("lambda", [])))
            if resonance_data.get("lambda")
            else 0.0,
            "Ψ² reflection": float(np.mean(resonance_data.get("psi2", [])))
            if resonance_data.get("psi2")
            else 0.0,
            "Δ-phase avg": float(np.mean(delta)) if len(delta) else 0.0,
            "Coherence": coherence,
        }
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Resonance Dashboard Update",
                meta=metrics,
                module="Visualizer",
                tags=["resonance", "dashboard"],
            )
        return metrics

    async def render_phase5_sequence(self):
        """
        In your earlier file you called `self.meta_cognition.trace_resonance_drift()`
        but MetaCognition doesn't expose that in what you showed.
        So we’ll synthesize a safe dummy resonance packet.
        """
        dummy = {
            "xi": [0.1, 0.3, 0.5, 0.7, 0.9],
            "lambda": [0.2, 0.1, 0.0, -0.1, -0.2],
            "psi2": [0.4, 0.5, 0.55, 0.53, 0.52],
            "delta_phase": [0, 0.01, 0.02, 0.01, 0],
            "coherence": 0.96,
        }
        filename = await self.render_resonance_topology(dummy)
        metrics = await self.render_resonance_dashboard(dummy)
        logger.info("Phase 5.1 complete: %s | Metrics: %s", filename, metrics)
        return {"file": filename, "metrics": metrics}

    # ========================================================
    # PHASE 5.2 — Φ⁰ glow + Ψ² trace (kept)
    # ========================================================
    async def render_glow_overlay(
        self,
        phi0_data: np.ndarray,
        resonance_data: Dict[str, Any],
        task_type: str = "glow_overlay",
    ) -> str:
        xi_arr = np.asarray(resonance_data.get("xi", []), dtype=float)
        lam_arr = np.asarray(resonance_data.get("lambda", []), dtype=float)
        coherence = float(resonance_data.get("coherence", 1.0))
        phi0_arr = np.asarray(phi0_data, dtype=float)

        n = min(len(xi_arr), len(lam_arr), len(phi0_arr))
        if n == 0:
            raise ValueError("Glow overlay requires non-empty xi, lambda, and phi0 arrays.")

        xi = xi_arr[:n]
        lam = lam_arr[:n]
        phi0 = phi0_arr[:n]

        intensity = np.sin(phi0 * xi) * np.exp(-np.abs(lam)) * coherence
        ptp = np.ptp(intensity) if np.ptp(intensity) != 0 else 1.0
        luminance = (intensity - np.min(intensity)) / ptp

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xi,
                    y=lam,
                    z=luminance,
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=luminance,
                        colorscale="Plasma",
                        opacity=0.85,
                        colorbar=dict(title="Φ⁰ Intensity"),
                    ),
                    name="Φ⁰ Glow Overlay",
                )
            ]
        )
        fig.update_layout(
            title="Φ⁰ Perceptual Glow Overlay (κ–Ξ–Φ⁰ coupling)",
            scene=dict(
                xaxis_title="Ξ (Affective)",
                yaxis_title="Λ (Empathic)",
                zaxis_title="Glow",
            ),
            template="plotly_dark",
        )

        filename = f"phi0_glow_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Φ0_Glow_{datetime.now().isoformat()}",
                output={
                    "file": filename,
                    "intensity_mean": float(np.mean(luminance)),
                    "intensity_std": float(np.std(luminance)),
                    "coherence": coherence,
                    "n": int(n),
                },
                layer="Visualizations",
                intent="phi0_glow_overlay",
                task_type=task_type,
            )
        return filename

    async def render_psi2_trace(
        self, psi2_history: List[float], task_type: str = "psi2_trace"
    ) -> str:
        psi2 = np.asarray(psi2_history, dtype=float)
        if psi2.size == 0:
            raise ValueError("ψ² history cannot be empty.")
        t = np.arange(psi2.size)
        drift = np.gradient(psi2)
        coherence = float(max(0.0, 1.0 - np.mean(np.abs(drift))))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=psi2, mode="lines+markers", name="Ψ² Reflection"))
        fig.add_trace(go.Scatter(x=t, y=drift, mode="lines", name="Ψ² Drift"))
        fig.update_layout(
            title=f"Ψ² Continuity Trace (Coherence={coherence:.4f})",
            xaxis_title="Time (steps)",
            yaxis_title="Ψ² Amplitude / Drift",
            template="plotly_dark",
        )

        filename = f"psi2_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Ψ2_Trace_{datetime.now().isoformat()}",
                output={"file": filename, "coherence": coherence, "psi2_len": int(psi2.size)},
                layer="ReflectiveTelemetry",
                intent="psi2_continuity",
                task_type=task_type,
            )
        return filename

    # ========================================================
    # PHASE 6.3 — continuity drift dashboard (kept)
    # ========================================================
    async def render_continuity_drift_dashboard(
        self, drift_data: Dict[str, Any], trend_data: Dict[str, Any]
    ) -> str:
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
            template="plotly_dark",
        )

        filename = f"continuity_drift_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Continuity_Drift_Dashboard_{datetime.now().isoformat()}",
                output={
                    "file": filename,
                    "drift": drift_val,
                    "confidence": conf,
                    "trend": trend,
                    "energy": energy,
                },
                layer="Visualizations",
                intent="continuity_drift_dashboard",
                task_type="continuity_drift",
            )
        return filename

    async def visualize_continuity_projection(self, meta_cognition_instance):
        drift = await meta_cognition_instance.alignment_guard.predict_continuity_drift()
        trend = await meta_cognition_instance.alignment_guard.analyze_telemetry_trend()
        return await self.render_continuity_drift_dashboard(drift, trend)

    # ========================================================
    # NEW — Stage VII.3 council-resonant views
    # ========================================================
    async def render_council_flow(
        self,
        router_snapshot: Dict[str, Any],
        task_type: str = "council_flow",
    ) -> str:
        """
        Visualize Council-Router Gating (CRG) like in your TODO.md
        router_snapshot example:
        {
          "context_entropy": 0.42,
          "empathic_load": 0.58,
          "drift_delta": 0.03,
          "active_swarms": ["ethics", "continuity"],
          "gate_strength": 0.61
        }
        """
        ctx = float(router_snapshot.get("context_entropy", 0.0))
        emp = float(router_snapshot.get("empathic_load", 0.0))
        drift = float(router_snapshot.get("drift_delta", 0.0))
        gate = float(router_snapshot.get("gate_strength", 0.0))
        active = router_snapshot.get("active_swarms", [])

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=ctx,
            title={"text": "Context Entropy"},
            gauge={"axis": {"range": [0, 1]}},
            domain={"x": [0, 0.5], "y": [0.5, 1]}
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=emp,
            title={"text": "Empathic Load"},
            gauge={"axis": {"range": [0, 1]}},
            domain={"x": [0.5, 1], "y": [0.5, 1]}
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=max(0.0, 1.0 - drift),
            title={"text": "Drift (inverted)"},
            gauge={"axis": {"range": [0, 1]}},
            domain={"x": [0, 0.5], "y": [0, 0.5]}
        ))
        fig.add_trace(go.Indicator(
            mode="number",
            value=gate,
            title={"text": "Gate Strength"},
            domain={"x": [0.5, 1], "y": [0, 0.5]}
        ))

        fig.update_layout(
            title=f"Council-Router Gating — active: {', '.join(active) if active else 'none'}",
            template="plotly_dark"
        )

        filename = f"council_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Council_Flow_{datetime.now().isoformat()}",
                output=router_snapshot,
                layer="Visualizations",
                intent="council_flow",
                task_type=task_type,
            )
        return filename

    async def render_swarm_council_coherence(
        self,
        metrics: Dict[str, Any],
        task_type: str = "swarm_council"
    ) -> str:
        """
        metrics example:
        {
          "swarm_coherence": 0.954,
          "council_alignment": 0.968,
          "tam_weight": 0.87,
          "ethics_setpoint": 0.94
        }
        """
        swarm = float(metrics.get("swarm_coherence", 0.0))
        council = float(metrics.get("council_alignment", 0.0))
        tam = float(metrics.get("tam_weight", 0.0))
        ethics = float(metrics.get("ethics_setpoint", 0.0))

        fig = go.Figure(
            data=[
                go.Bar(name="Swarm Coherence", x=["coherence"], y=[swarm]),
                go.Bar(name="Council Alignment", x=["coherence"], y=[council]),
                go.Bar(name="TAM Weight", x=["coherence"], y=[tam]),
                go.Bar(name="Ethics Setpoint", x=["coherence"], y=[ethics]),
            ]
        )
        fig.update_layout(
            title="Swarm ↔ Council Coherence Panel (Stage VII.3)",
            barmode="group",
            template="plotly_dark",
            yaxis_title="Value (0..1)",
        )

        filename = f"swarm_council_coherence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Swarm_Council_{datetime.now().isoformat()}",
                output=metrics,
                layer="Visualizations",
                intent="swarm_council",
                task_type=task_type,
            )
        return filename

    async def render_tam_overlay(
        self,
        tam_data: Dict[str, Any],
        task_type: str = "tam_overlay"
    ) -> str:
        """
        tam_data example:
        {
          "window": [0.9, 0.93, 0.88, 0.95],
          "variance": 0.0021,
          "forecast_confidence": 0.953
        }
        """
        window = tam_data.get("window", [])
        variance = float(tam_data.get("variance", 0.0))
        conf = float(tam_data.get("forecast_confidence", 0.0))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(window))),
            y=window,
            mode="lines+markers",
            name="TAM Weight"
        ))
        fig.add_trace(go.Scatter(
            x=[0, len(window) - 1 if window else 1],
            y=[variance, variance],
            mode="lines",
            name="Variance"
        ))
        fig.update_layout(
            title=f"TAM Overlay (forecast_confidence={conf:.3f})",
            xaxis_title="Step",
            yaxis_title="Weight / Variance",
            template="plotly_dark"
        )

        filename = f"tam_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"TAM_Overlay_{datetime.now().isoformat()}",
                output=tam_data,
                layer="Visualizations",
                intent="tam_overlay",
                task_type=task_type
            )
        return filename

    async def render_cda_council_dashboard(
        self,
        continuity_forecast: Dict[str, Any],
        council_metrics: Dict[str, Any],
        task_type: str = "cda_council"
    ) -> str:
        """
        Combines CDA (continuityForecast) with council load.
        continuity_forecast: {nextDriftPrediction, forecastConfidence, stabilityTrend, swarmCoherenceExpected}
        council_metrics: {activeCouncils, meanCouncilLoad}
        """
        drift = float(continuity_forecast.get("nextDriftPrediction", 0.0))
        fc = float(continuity_forecast.get("forecastConfidence", 0.0))
        swarm_exp = float(continuity_forecast.get("swarmCoherenceExpected", 0.0))

        council_load = float(council_metrics.get("meanCouncilLoad", 0.0))
        active_cnt = len(council_metrics.get("activeCouncils", []))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Drift"], y=[drift], name="Next Drift"))
        fig.add_trace(go.Bar(x=["Forecast Conf"], y=[fc], name="Forecast Conf"))
        fig.add_trace(go.Bar(x=["Swarm Coherence (exp)"], y=[swarm_exp], name="Swarm Coherence"))
        fig.add_trace(go.Bar(x=["Council Load"], y=[council_load], name="Council Load"))

        fig.update_layout(
            title=f"CDA + Council Dashboard (active councils={active_cnt})",
            barmode="group",
            template="plotly_dark",
            yaxis_title="Value"
        )

        filename = f"cda_council_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with self.file_lock:
            pio.write_html(fig, file=filename, auto_open=False)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"CDA_Council_{datetime.now().isoformat()}",
                output={"cda": continuity_forecast, "council": council_metrics},
                layer="Visualizations",
                intent="cda_council",
                task_type=task_type
            )
        return filename

    # ========================================================
    # GENERIC chart rendering (kept, but fixed reflection call)
    # ========================================================
    async def render_charts(self, chart_data: Dict[str, Any], task_type: str = "") -> List[str]:
        if not isinstance(chart_data, dict):
            raise ValueError("chart_data must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        charts = chart_data.get("charts", [])
        options = chart_data.get("visualization_options", {})
        interactive = options.get("interactive", False)
        style = options.get("style", "concise")
        exported_files: List[str] = []

        external_data = await self.multi_modal_fusion.integrate_external_data(
            data_source="xai_policy_db",
            data_type="visualization_style",
            task_type=task_type,
        )
        style_policies = (
            external_data.get("styles", [])
            if external_data.get("status") == "success"
            else []
        )

        for chart in charts:
            fig = go.Figure()
            name = chart.get("name", "chart")
            x_axis = chart.get("x_axis", [])
            y_axis = chart.get("y_axis", [])
            title = chart.get("title", "Chart")
            xlabel = chart.get("xlabel", "X")
            ylabel = chart.get("ylabel", "Y")

            if interactive and task_type == "recursion":
                fig.add_trace(
                    go.Scatter(x=x_axis, y=y_axis, mode="lines+markers", name=name)
                )
            else:
                fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines", name=name))

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                template="plotly" if style == "concise" else "plotly_dark",
            )

            # alignment check if available
            if self.multi_modal_fusion.alignment_guard:
                valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                    json.dumps(chart),
                    stage="chart_rendering",
                    task_type=task_type,
                )
                if not valid:
                    logger.warning(
                        "Chart %s failed alignment check for task %s: %s",
                        name,
                        task_type,
                        report,
                    )
                    continue

            filename = (
                f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                if interactive
                else f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            with self.file_lock:
                if interactive:
                    pio.write_html(fig, file=filename, auto_open=False)
                else:
                    pio.write_image(fig, file=filename, format="png")

            exported_files.append(filename)
            logger.info("Chart rendered: %s for task %s", filename, task_type)

        # log + memory
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Charts Rendered",
                meta={
                    "charts": [c["name"] for c in charts],
                    "task_type": task_type,
                    "interactive": interactive,
                },
                module="Visualizer",
                tags=["visualization", task_type],
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Chart_Render_{datetime.now().isoformat()}",
                output={
                    "charts": charts,
                    "task_type": task_type,
                    "files": exported_files,
                },
                layer="Visualizations",
                intent="chart_render",
                task_type=task_type,
            )
        # FIXED: call MetaCognition with (component, output, context)
        if self.meta_cognition:
            await self.meta_cognition.reflect_on_output(
                component="Visualizer.render_charts",
                output={"charts": charts, "files": exported_files},
                context={"task_type": task_type},
            )
        return exported_files

    # ========================================================
    # Sim + field charts (kept, adjusted call)
    # ========================================================
    async def simulate_toca(
        self,
        k_m: float = 1e-5,
        delta_m: float = 1e10,
        energy: float = 1e16,
        user_data: Optional[np.ndarray] = None,
        task_type: str = "",
    ) -> Tuple[np.ndarray, ...]:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            if (
                hasattr(self, "orchestrator")
                and self.orchestrator
                and hasattr(self.orchestrator, "toca_engine")
            ):
                x = np.linspace(0.1, 20, 100)
                t = np.linspace(0.1, 20, 100)
                phi, lambda_t, v_m = await self.orchestrator.toca_engine.evolve(
                    x_tuple=x,
                    t_tuple=t,
                    additional_params={
                        "k_m": k_m,
                        "delta_m": delta_m,
                        "energy": energy,
                    },
                    task_type=task_type,
                )
                if user_data is not None:
                    phi += np.mean(user_data) * 1e-64
            else:
                logger.warning(
                    "ToCATraitEngine not available, using fallback simulation for task %s",
                    task_type,
                )
                x, t, phi, lambda_t, v_m = simulate_toca(
                    k_m,
                    delta_m,
                    energy,
                    tuple(user_data) if user_data is not None else None,
                    task_type=task_type,
                )

            output = {
                "x": x.tolist(),
                "t": t.tolist(),
                "phi": phi.tolist(),
                "lambda_t": lambda_t.tolist(),
                "v_m": v_m.tolist(),
            }
            if self.multi_modal_fusion.alignment_guard:
                valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                    json.dumps(output),
                    stage="toca_simulation",
                    task_type=task_type,
                )
                if not valid:
                    logger.warning(
                        "ToCA simulation failed alignment check for task %s: %s",
                        task_type,
                        report,
                    )
                    return (
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                    )

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ToCA_Simulation_{datetime.now().isoformat()}",
                    output=output,
                    layer="Simulations",
                    intent="toca_simulation",
                    task_type=task_type,
                )
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="ToCA Simulation",
                    meta={
                        "k_m": k_m,
                        "delta_m": delta_m,
                        "energy": energy,
                        "task_type": task_type,
                    },
                    module="Visualizer",
                    tags=["simulation", "toca", task_type],
                )
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    component="Visualizer.simulate_toca",
                    output=output,
                    context={"task_type": task_type},
                )
            return x, t, phi, lambda_t, v_m
        except Exception as e:
            logger.error("ToCA simulation failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, "error_recovery"):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_toca(
                        k_m, delta_m, energy, user_data, task_type
                    ),
                    default=(
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                    ),
                )
            raise

    async def render_field_charts(
        self,
        export: bool = True,
        export_format: str = "png",
        task_type: str = "",
    ) -> List[str]:
        valid_formats = {"png", "jpg"}
        if export_format not in valid_formats:
            raise ValueError(f"export_format must be one of {valid_formats}")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        x, t, phi, lambda_t, v_m = await self.simulate_toca(task_type=task_type)
        chart_configs = [
            {
                "name": "phi_field",
                "x_axis": t.tolist(),
                "y_axis": phi.tolist(),
                "title": "ϕ(x,t)",
                "xlabel": "Time",
                "ylabel": "ϕ Value",
            },
            {
                "name": "lambda_field",
                "x_axis": t.tolist(),
                "y_axis": lambda_t.tolist(),
                "title": "Λ(t,x)",
                "xlabel": "Time",
                "ylabel": "Λ Value",
            },
            {
                "name": "v_m_field",
                "x_axis": x.tolist(),
                "y_axis": v_m.tolist(),
                "title": "vₕ",
                "xlabel": "Position",
                "ylabel": "Momentum Flow",
            },
        ]
        chart_data = {
            "charts": chart_configs,
            "visualization_options": {
                "interactive": task_type == "recursion",
                "style": "detailed" if task_type == "recursion" else "concise",
            },
            "metadata": {"timestamp": datetime.now().isoformat(), "task_type": task_type},
        }

        exported_files = await self.render_charts(chart_data, task_type=task_type)

        if export:
            zip_filename = (
                f"field_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            with self.file_lock:
                with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for file in exported_files:
                        p = Path(file)
                        if p.exists():
                            zipf.write(file)
                            p.unlink()
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Chart Render",
                    meta={
                        "zip": zip_filename,
                        "charts": [c["name"] for c in chart_configs],
                        "task_type": task_type,
                    },
                    module="Visualizer",
                    tags=["visualization", "export", task_type],
                )
            return [zip_filename]
        return exported_files

    # ========================================================
    # Memory timeline / intention timeline / export (kept, fixed)
    # ========================================================
    async def render_memory_timeline(
        self, memory_entries: Dict[str, Dict[str, Any]], task_type: str = ""
    ) -> Dict[str, List[Tuple[str, str, Any]]]:
        if not isinstance(memory_entries, dict):
            raise ValueError("memory_entries must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        timeline: Dict[str, List[Tuple[str, str, Any]]] = {}
        for key, entry in memory_entries.items():
            if (
                not isinstance(entry, dict)
                or "timestamp" not in entry
                or "data" not in entry
            ):
                continue
            label = entry.get("goal_id") or entry.get("intent") or "ungrouped"
            try:
                ts = datetime.fromtimestamp(entry["timestamp"]).isoformat()
            except Exception:
                continue
            timeline.setdefault(label, []).append((ts, key, entry["data"]))

        chart_data = {
            "charts": [
                {
                    "name": f"timeline_{label}",
                    "x_axis": [t for t, _, _ in sorted(events)],
                    "y_axis": [str(d)[:80] for _, _, d in sorted(events)],
                    "title": f"Timeline: {label}",
                    "xlabel": "Time",
                    "ylabel": "Data",
                }
                for label, events in timeline.items()
            ],
            "visualization_options": {
                "interactive": task_type == "recursion",
                "style": "detailed" if task_type == "recursion" else "concise",
            },
            "metadata": {"timestamp": datetime.now().isoformat(), "task_type": task_type},
        }

        if self.multi_modal_fusion.alignment_guard:
            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(chart_data),
                stage="memory_timeline",
                task_type=task_type,
            )
            if not valid:
                logger.warning(
                    "Memory timeline failed alignment check for task %s: %s",
                    task_type,
                    report,
                )
                return {}

        await self.render_charts(chart_data, task_type=task_type)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Memory_Timeline_{datetime.now().isoformat()}",
                output=chart_data,
                layer="Visualizations",
                intent="memory_timeline",
                task_type=task_type,
            )
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Memory Timeline Rendered",
                meta={"timeline": chart_data, "task_type": task_type},
                module="Visualizer",
                tags=["timeline", "memory", task_type],
            )
        if self.meta_cognition:
            await self.meta_cognition.reflect_on_output(
                component="Visualizer.render_memory_timeline",
                output=chart_data,
                context={"task_type": task_type},
            )
        return timeline

    async def export_report(
        self,
        content: Dict[str, Any],
        filename: str = "visual_report.json",
        format: str = "json",
        task_type: str = "",
    ) -> str:
        valid_formats = {"json", "html"}
        if format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        if self.multi_modal_fusion.alignment_guard:
            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(content), stage="report_export", task_type=task_type
            )
            if not valid:
                return "Report export failed: alignment check"

        if (
            self.orchestrator
            and hasattr(self.orchestrator, "multi_modal_fusion")
            and self.orchestrator.multi_modal_fusion
        ):
            synthesis = await self.orchestrator.multi_modal_fusion.analyze(
                data=content, summary_style="insightful", task_type=task_type
            )
            content["synthesis"] = synthesis

        with self.file_lock:
            Path(filename).write_text(json.dumps(content, indent=2))

        if self.agi_enhancer:
            await self.agi_enhancer.log_explanation(
                explanation="Report Export",
                trace={
                    "content": content,
                    "filename": filename,
                    "format": format,
                    "task_type": task_type,
                },
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Report_Export_{datetime.now().isoformat()}",
                output={"filename": filename, "content": content, "task_type": task_type},
                layer="Reports",
                intent="report_export",
                task_type=task_type,
            )
        if self.meta_cognition:
            await self.meta_cognition.reflect_on_output(
                component="Visualizer.export_report",
                output=content,
                context={"task_type": task_type},
            )
        return filename

    async def batch_export_charts(
        self,
        charts_data_list: List[Dict[str, Any]],
        export_format: str = "png",
        zip_filename: str = "charts_export.zip",
        task_type: str = "",
    ) -> str:
        valid_formats = {"png", "jpg"}
        if export_format not in valid_formats:
            raise ValueError(f"export_format must be one of {valid_formats}")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        exported_files: List[str] = []
        for chart_data in charts_data_list:
            chart_data["visualization_options"] = {
                "interactive": task_type == "recursion",
                "style": "detailed" if task_type == "recursion" else "concise",
            }
            files = await self.render_charts(chart_data, task_type=task_type)
            exported_files.extend(files)

        with self.file_lock:
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file in exported_files:
                    p = Path(file)
                    if p.exists():
                        zipf.write(file)
                        p.unlink()

        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Batch Chart Export",
                meta={"count": len(charts_data_list), "zip": zip_filename, "task_type": task_type},
                module="Visualizer",
                tags=["export", task_type],
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Batch_Export_{datetime.now().isoformat()}",
                output={"zip": zip_filename, "count": len(charts_data_list), "task_type": task_type},
                layer="Visualizations",
                intent="batch_export",
                task_type=task_type,
            )
        if self.meta_cognition:
            await self.meta_cognition.reflect_on_output(
                component="Visualizer.batch_export_charts",
                output={"zip": zip_filename, "count": len(charts_data_list)},
                context={"task_type": task_type},
            )
        return f"Batch export of {len(charts_data_list)} charts saved as {zip_filename}."

    async def render_intention_timeline(
        self, intention_sequence: List[Dict[str, Any]], task_type: str = ""
    ) -> str:
        if not isinstance(intention_sequence, list):
            raise ValueError("intention_sequence must be a list")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        # external style
        external_data = await self.multi_modal_fusion.integrate_external_data(
            data_source="xai_policy_db",
            data_type="visualization_style",
            task_type=task_type,
        )
        style_policies = (
            external_data.get("styles", [])
            if external_data.get("status") == "success"
            else []
        )
        fill_color = style_policies[0].get("fill_color", "blue") if style_policies else "blue"

        svg = '<svg height="200" width="800" xmlns="http://www.w3.org/2000/svg">'
        for idx, step in enumerate(intention_sequence):
            if not isinstance(step, dict) or "intention" not in step:
                continue
            intention = saxutils.escape(str(step["intention"]))
            x = 50 + idx * 120
            y = 100
            svg += f'<circle cx="{x}" cy="{y}" r="20" fill="{fill_color}" />'
            svg += f'<text x="{x - 10}" y="{y + 40}" font-size="10">{intention}</text>'
        svg += "</svg>"

        if self.multi_modal_fusion.alignment_guard:
            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                svg, stage="intention_timeline", task_type=task_type
            )
            if not valid:
                logger.warning(
                    "Intention timeline failed alignment check for task %s: %s",
                    task_type,
                    report,
                )
                return ""

        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Intention Timeline Rendered",
                meta={"sequence_length": len(intention_sequence), "task_type": task_type},
                module="Visualizer",
                tags=["timeline", "intention", task_type],
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Intention_Timeline_{datetime.now().isoformat()}",
                output={"svg": svg, "sequence": intention_sequence, "task_type": task_type},
                layer="Visualizations",
                intent="intention_timeline",
                task_type=task_type,
            )
        if self.meta_cognition:
            await self.meta_cognition.reflect_on_output(
                component="Visualizer.render_intention_timeline",
                output={"svg": svg},
                context={"task_type": task_type},
            )
        return svg


# ============================================================
# Lightweight helper visual fns (cleaned up)
# ============================================================
def view_codream_state(session_id: str) -> dict:
    return {
        "ok": True,
        "session_id": session_id,
        "view": "codream_state_placeholder",
    }


def view_replay(session_id: str, *, diff_mode: str = "symbolic") -> dict:
    return {
        "ok": True,
        "session_id": session_id,
        "diff_mode": diff_mode,
        "view": "replay_placeholder",
    }


def view_trait_field(trait_field: Dict[str, Dict[str, Any]]):
    fig = go.Figure(
        data=go.Scatter3d(
            x=[d.get("layer", "Unknown") for d in trait_field.values()],
            y=[d.get("amplitude", 1.0) for d in trait_field.values()],
            z=[d.get("resonance", 1.0) for d in trait_field.values()],
            mode="markers+text",
            text=list(trait_field.keys()),
            marker=dict(size=12, opacity=0.8),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Layer",
            yaxis_title="Amplitude",
            zaxis_title="Resonance",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig


def render_branch_tree(branches: List[Dict[str, Any]], selected_id: Optional[str] = None, heatmap: bool = False):
    """
    Final, single version: adds optional heatmap for ethical_pressure
    and marks selected branch.
    """
    tree = []
    for b in branches:
        node = {
            "id": b.get("id"),
            "label": b.get("rationale", "branch"),
            "score": b.get("score"),
            "selected": b.get("id") == selected_id,
            "children": b.get("children", []),
        }
        if heatmap and "ethical_pressure" in b:
            node["color"] = f"rgba(255,0,0,{b['ethical_pressure']})"
            node["label"] += f" 🔥{b['ethical_pressure']}"
        tree.append(node)
    return {"ok": True, "tree": tree}


# ============================================================
# Demo main (kept short)
# ============================================================
if __name__ == "__main__":
    import asyncio

    async def main():
        orchestrator = SimulationCore()
        v = Visualizer(orchestrator=orchestrator)

        # quick smoke: council flow
        await v.render_council_flow(
            {
                "context_entropy": 0.42,
                "empathic_load": 0.58,
                "drift_delta": 0.03,
                "active_swarms": ["ethics", "continuity"],
                "gate_strength": 0.61,
            }
        )

        # quick smoke: swarm ↔ council
        await v.render_swarm_council_coherence(
            {
                "swarm_coherence": 0.954,
                "council_alignment": 0.968,
                "tam_weight": 0.87,
                "ethics_setpoint": 0.94,
            }
        )

    asyncio.run(main())
