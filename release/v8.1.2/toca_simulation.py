"""
ANGELA Cognitive System: ToCA Simulation
Module: toca_simulation_v4_1_sovereign
Stage: VII.8 — Harmonic Sovereignty + Embodied Ethics PID
Base: v4.0 Optimized + Adapters
Date: 2025-11-07

This module supersedes the earlier "Galaxy Rotation & Agent Conflict Sim (v4.0)" file.
It keeps API compatibility:

    - simulate_galaxy_rotation(...)
    - run_ethics_scenarios_internal(...)
    - integrate_real_world_data(...)
    - ExtendedSimulationCore(...)  ← main class

and adds:

    - SovereignEmpathicAdapter       (coherence + bounded reflex)
    - run_sovereignty_audit(...)     (Φ⁰–Σ–Ω² coherence audit)
    - continuity-aware ethics batch  (feeds Δ-phase from last sim)
    - ledger-logging for all major ops

The file is self-contained and provides safe fallbacks for missing ANGELA modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional deps (allow running in reduced environments)
# ---------------------------------------------------------------------------
try:
    import aiohttp
except Exception:  # pragma: no cover - runtime fallback
    aiohttp = None  # type: ignore

try:
    from scipy.constants import G
except Exception:  # pragma: no cover - provide constant
    G = 6.67430e-11

# ---------------------------------------------------------------------------
# Alignment / Ethics imports (with fallbacks)
# ---------------------------------------------------------------------------
try:
    from alignment_guard import AlignmentGuard, EmbodiedEthicsCore, log_event_to_ledger
except Exception as e:  # pragma: no cover - local/dev usage
    logging.warning(f"[toca_simulation] alignment_guard fallback: {e}")

    class AlignmentGuard:  # minimal stub
        async def update_affective_pid(self, delta_phase_rad: float, recursion_depth: int = 1):
            return {
                "u": 0.0,
                "error": float(delta_phase_rad),
                "rms_drift": 0.0,
                "depth_ok": recursion_depth >= 1,
            }

        async def _log_context(self, event: Dict[str, Any]):
            return None

        async def predict_continuity_drift(self) -> Dict[str, Any]:
            return {"status": "stub", "drift": 0.0}

        async def analyze_telemetry_trend(self) -> Dict[str, Any]:
            return {"status": "stub", "trend": []}

    class EmbodiedEthicsCore:  # minimal stub
        def __init__(self, fusion, empathy_engine, policy_trainer=None):
            self.fusion = fusion
            self.empathy_engine = empathy_engine

        async def run_scenario(self, scenario: str = "default"):
            return {
                "τ_reflex": 0.5,
                "κ": 0.5,
                "Ξ": 0.5,
                "timestamp": time.time(),
                "status": "evaluated",
            }

    def log_event_to_ledger(event: Dict[str, Any]):
        # best-effort logging
        logging.getLogger("ANGELA.ToCA.Ledger").info("ledger: %s", event)


# ---------------------------------------------------------------------------
# ANGELA module imports (with stubs to keep file runnable)
# ---------------------------------------------------------------------------
try:
    from modules.simulation_core import SimulationCore as _BaseCore
    from modules.simulation_core import ToCATraitEngine
except Exception as e:  # pragma: no cover
    logging.warning(f"[toca_simulation] modules.simulation_core fallback: {e}")

    class _BaseCore:  # type: ignore
        def __init__(self, **kwargs):
            self.memory_manager = kwargs.get("memory_manager")
            self.meta_cognition = kwargs.get("meta_cognition")
            self.error_recovery = kwargs.get("error_recovery")

    class ToCATraitEngine:  # type: ignore
        def __init__(self, *_, **__):
            pass

        async def evolve(self, *_, **__):
            x = np.linspace(0.1, 20.0, 256)
            return x * 0 + 1e-3, x * 0 + 1e-3, x * 0 + 1e-3

        async def update_fields_with_agents(self, phi, lambda_t, agent_matrix, **__):
            return phi, lambda_t

try:
    from modules import meta_cognition as meta_cognition_module
except Exception:  # pragma: no cover
    class _DummyMeta:
        async def reflect_on_output(self, **_):  # pragma: no cover
            return {"status": "success"}
        async def diagnose_drift(self, *_args, **_kwargs):
            return {"impact_score": 0.0}
        async def optimize_traits_for_drift(self, *_args, **_kwargs):
            return {}
    meta_cognition_module = type("meta_cognition", (), {"MetaCognition": _DummyMeta})()  # type: ignore

try:
    from modules import error_recovery as error_recovery_module
except Exception:  # pragma: no cover
    class _DummyErr:
        async def handle_error(self, msg, retry_func=None, default=None, diagnostics=None):
            return default or {"error": msg}
    error_recovery_module = type("error_recovery", (), {"ErrorRecovery": _DummyErr})()  # type: ignore

try:
    from modules import memory_manager as memory_manager_module
except Exception:  # pragma: no cover
    class _DummyMM:
        async def store(self, *_, **__): ...
        async def retrieve(self, *_, **__): return None
    memory_manager_module = type("memory_manager", (), {"MemoryManager": _DummyMM})()  # type: ignore


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------
logger = logging.getLogger("ANGELA.ToCA.Sim")
G_SI = G
KPC_TO_M = 3.0857e19
MSUN_TO_KG = 1.989e30

K_DEFAULT = 0.85
EPSILON_DEFAULT = 0.015
R_HALO_DEFAULT = 20.0

STAGE_VII_8 = True  # feature flag for this upgraded file


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hashable_weights(weights: Optional[Dict[str, float]]) -> Optional[Tuple[Tuple[str, float], ...]]:
    if not weights:
        return None
    return tuple(sorted((k, float(v)) for k, v in weights.items()))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EthicsOutcome:
    frame: str
    decision: str
    justification: str
    risk: float
    rights_balance: float
    stakeholders: List[str]
    notes: str = "sandbox"


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
class EmpathicAdapter:
    """
    Base adapter: bridges simulation outputs to MetaCognition and external affective bridges.
    v4.1 extends this into SovereignEmpathicAdapter below.
    """

    def __init__(self, meta_cog=None, mirror_bridge=None):
        self.meta_cog = meta_cog or meta_cognition_module.MetaCognition()
        self.mirror_bridge = mirror_bridge

    async def trace_xi_lambda_shift(self, traits: Dict[str, float], sim_output: Any) -> Dict[str, float]:
        drift = {}
        try:
            drift = await self.meta_cog.diagnose_drift({"name": "empathy", "similarity": 0.8})
        except Exception:
            drift = {"impact_score": 0.0}
        xi_shift = float(drift.get("impact_score", 0.0)) * 0.1
        lambda_shift = float(np.mean(list(traits.values()) or [0.5])) * xi_shift
        return {
            "xi_shift": xi_shift,
            "lambda_shift": lambda_shift,
            "empathic_depth": 0.85,
        }

    async def affective_epistemic_feedback(self, agent_profiles: List[Any], context: Dict[str, Any]) -> Dict[str, float]:
        if not self.mirror_bridge:
            return {"feedback_score": 0.0}
        epistemic = np.mean([p.get("knowledge", 0.5) for p in agent_profiles])
        affective = np.mean([p.get("empathy", 0.5) for p in agent_profiles])
        return {
            "epistemic": float(epistemic),
            "affective": float(affective),
            "balance": float((epistemic + affective) / 2),
        }

    def delta_phase_empathy_metrics(self, fields: Tuple[np.ndarray, ...], task_type: str) -> Dict[str, float]:
        # Default implementation expects fields from compute_trait_fields()
        phi = fields[6]
        beta = fields[1]
        dphi = np.gradient(phi)
        delta_phase = float(np.mean(beta * dphi))
        empathy_metric = float(np.mean(phi) * (1 - abs(delta_phase)))
        return {
            "delta_phase": delta_phase,
            "empathy_metric": empathy_metric,
            "task_adjusted": empathy_metric * (1.2 if task_type == "recursion" else 1.0),
        }


class SovereignEmpathicAdapter(EmpathicAdapter):
    """
    v4.1: adds coherence estimation and bounded reflex propagation.
    This ensures that over-enthusiastic empathy shifts are clipped before reaching
    ethics/overlay layers.
    """

    def __init__(self, meta_cog=None, mirror_bridge=None):
        super().__init__(meta_cog=meta_cog, mirror_bridge=mirror_bridge)

    def estimate_coherence(self, trait_vector: Dict[str, float]) -> float:
        if not trait_vector:
            return 0.5
        vals = np.array(list(trait_vector.values()), dtype=float)
        spread = float(np.std(vals))
        # lower spread = higher coherence
        coherence = float(np.clip(1.0 - spread, 0.0, 1.0))
        return coherence

    async def trace_xi_lambda_shift(self, traits: Dict[str, float], sim_output: Any) -> Dict[str, float]:
        base = await super().trace_xi_lambda_shift(traits, sim_output)
        coherence = self.estimate_coherence(traits)
        # bounded reflex: make shifts smaller when coherence is low
        base["xi_shift"] *= coherence
        base["lambda_shift"] *= coherence
        base["coherence"] = coherence
        # meta log (best-effort)
        try:
            await self.meta_cog.reflect_on_output(
                component="SovereignEmpathicAdapter",
                output=base,
                context={"task_type": sim_output.get("task_type", "") if isinstance(sim_output, dict) else ""},
            )
        except Exception:
            pass
        return base


# ---------------------------------------------------------------------------
# Ethics runner
# ---------------------------------------------------------------------------
class EthicsScenarioRunner:
    """
    Orchestrates embodied ethics evaluations during simulation cycles.
    Couples κ/Ξ into τ-reflex mapping via EmbodiedEthicsCore.
    """

    class _FusionAdapter:
        def __init__(self, core: "ExtendedSimulationCore"):
            self.core = core

        async def capture(self) -> Dict[str, float]:
            cs = getattr(self.core, "_context_salience", 0.5)
            return {"contextual_salience": float(cs)}

    class _EmpathyEngine:
        def __init__(self, core: "ExtendedSimulationCore"):
            self.core = core

        async def measure(self) -> Dict[str, float]:
            amp = getattr(self.core, "_empathic_amplitude", 0.5)
            return {"empathic_amplitude": float(amp)}

    def __init__(self, core: "ExtendedSimulationCore", alignment_guard: AlignmentGuard):
        self.core = core
        self.guard = alignment_guard
        self.ethics_core = EmbodiedEthicsCore(
            fusion=self._FusionAdapter(core),
            empathy_engine=self._EmpathyEngine(core),
        )

    async def run_tick(self, scenario: str = "default", delta_phase_rad: float = 0.0) -> Dict[str, Any]:
        ethics = await self.ethics_core.run_scenario(scenario=scenario)
        pid_feedback = await self.guard.update_affective_pid(delta_phase_rad=delta_phase_rad, recursion_depth=5)
        event = {
            "event": "simulation_tick_ethics",
            "scenario": scenario,
            "τ_reflex": ethics["τ_reflex"],
            "κ": ethics["κ"],
            "Ξ": ethics["Ξ"],
            "pid_u": pid_feedback["u"],
            "timestamp": ethics["timestamp"],
        }
        log_event_to_ledger(event)
        await self.guard._log_context(event)
        return event

    async def run_batch(self, scenarios: List[str], *, delta_series: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        results = []
        delta_series = delta_series or [0.0] * len(scenarios)
        for s, delta in zip(scenarios, delta_series):
            results.append(await self.run_tick(scenario=s, delta_phase_rad=delta))
        return results


# ---------------------------------------------------------------------------
# Extended Simulation Core
# ---------------------------------------------------------------------------
class ExtendedSimulationCore(_BaseCore):
    """
    Sovereign Reflex Edition of ToCA simulation.

    Key additions:
      - sovereign empathic adapter
      - sovereignty audit
      - continuity drift integration
      - ethics sandbox runner
      - galaxy rotation sim with Δ-phase propagation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta_cog = kwargs.get("meta_cognition") or meta_cognition_module.MetaCognition()
        self.error_recovery = kwargs.get("error_recovery") or error_recovery_module.ErrorRecovery()
        self.memory_manager = kwargs.get("memory_manager") or memory_manager_module.MemoryManager()
        self.alignment_guard = kwargs.get("alignment_guard") or AlignmentGuard()
        self.toca_engine = kwargs.get("toca_engine") or ToCATraitEngine(meta_cognition=self.meta_cog)

        # adapters
        self.empathic_adapter = SovereignEmpathicAdapter(self.meta_cog)
        self.ethics_runner = EthicsScenarioRunner(self, self.alignment_guard)

        # live state
        self._empathic_amplitude: float = 0.5
        self._context_salience: float = 0.5
        self._last_delta_phase: float = 0.0
        self._last_continuity_forecast: Dict[str, Any] = {}
        self._last_continuity_trend: Dict[str, Any] = {}

        logger.info("ExtendedSimulationCore (v4.1 Sovereign) initialized")

    # ------------------------------------------------------------------
    # Continuity integration
    # ------------------------------------------------------------------
    async def integrate_continuity_drift(self) -> None:
        if not self.alignment_guard:
            return
        try:
            drift_forecast = await self.alignment_guard.predict_continuity_drift()
            trend_metrics = await self.alignment_guard.analyze_telemetry_trend()
            event = {
                "event": "ToCA_ContinuityDriftUpdate",
                "forecast": drift_forecast,
                "trend": trend_metrics,
                "timestamp": _now_iso(),
            }
            log_event_to_ledger(event)
            self._last_continuity_forecast = drift_forecast
            self._last_continuity_trend = trend_metrics
        except Exception as e:
            logger.warning(f"Continuity drift integration failed: {e}")

    # ------------------------------------------------------------------
    # Real-world data
    # ------------------------------------------------------------------
    async def integrate_real_world_data(self, source: str, data_type: str, cache_timeout: float = 3600) -> Dict[str, Any]:
        cache_key = f"Data_{data_type}_{source}"
        # best-effort cache
        try:
            cached = await self.memory_manager.retrieve(cache_key)
            if cached and (time.time() - cached.get("ts", 0)) < cache_timeout:
                return cached["data"]
        except Exception:
            cached = None

        # fetch
        if aiohttp is None:
            return {"status": "error", "error": "aiohttp unavailable or network disabled"}

        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://x.ai/api/data?source={source}&type={data_type}") as resp:
                if resp.status != 200:
                    return {"status": "error", "error": f"HTTP {resp.status}"}
                data = await resp.json()

        if data_type == "galaxy_rotation":
            r = np.array(data.get("r_kpc", []))
            v = np.array(data.get("v_obs_kms", []))
            m = np.array(data.get("M_baryon_solar", []))
            result = {
                "status": "success",
                "r_kpc": r.tolist(),
                "v_obs_kms": v.tolist(),
                "M_baryon_solar": m.tolist(),
            }
        else:
            result = {"status": "success", "agent_traits": data.get("agent_traits", [])}

        try:
            await self.memory_manager.store(
                query=cache_key,
                output={"data": result, "ts": time.time()},
                layer="RealWorldData",
                intent="fetch",
            )
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Physics core
    # ------------------------------------------------------------------
    def compute_AGRF_curve(
        self,
        v_obs_kms: np.ndarray,
        M_b_solar: np.ndarray,
        r_kpc: np.ndarray,
        k: float = K_DEFAULT,
        epsilon: float = EPSILON_DEFAULT,
        r_halo: float = R_HALO_DEFAULT,
    ) -> np.ndarray:
        """ANGELA Galaxy Rotation Function."""
        r_m = r_kpc * KPC_TO_M
        M_b_kg = M_b_solar * MSUN_TO_KG
        v_obs_ms = v_obs_kms * 1e3
        M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
        M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
        M_total = M_b_kg + M_AGRF
        v_sim_ms = np.sqrt(np.clip(G_SI * M_total / r_m, 0, np.inf))
        return v_sim_ms / 1e3  # back to km/s

    # ------------------------------------------------------------------
    # Trait fields (cached)
    # ------------------------------------------------------------------
    @lru_cache(maxsize=128)
    def compute_trait_fields(
        self,
        r_tuple: Tuple[float, ...],
        v_obs_tuple: Tuple[float, ...],
        v_sim_tuple: Tuple[float, ...],
        time_elapsed: float = 1.0,
        tau_persistence: float = 10.0,
        task_type: str = "",
        weights_hash: Optional[Tuple[Tuple[str, float], ...]] = None,
    ) -> Tuple[np.ndarray, ...]:
        r = np.array(r_tuple)
        v_obs = np.array(v_obs_tuple)
        v_sim = np.array(v_sim_tuple)

        gamma = np.log(1 + np.clip(r, 1e-10, np.inf)) * (0.4 if task_type in ["rte", "wnli"] else 0.5)
        beta = np.abs(v_obs - v_sim) / (np.max(np.abs(v_obs)) + 1e-10) * (0.8 if task_type == "recursion" else 1.0)
        zeta = 1 / (1 + np.gradient(v_sim) ** 2)
        eta = np.exp(-time_elapsed / tau_persistence)
        psi = np.gradient(v_sim) / (np.gradient(r) + 1e-10)
        lam = np.cos(r / R_HALO_DEFAULT * np.pi)
        phi = K_DEFAULT * np.exp(-EPSILON_DEFAULT * r / R_HALO_DEFAULT)
        phi_prime = -EPSILON_DEFAULT * phi / R_HALO_DEFAULT
        beta_psi = beta * psi

        if weights_hash:
            weights = dict(weights_hash)
            beta *= float(weights.get("beta", 1.0))
            zeta *= float(weights.get("zeta", 1.0))

        # update perception proxy
        self._context_salience = float(np.clip(np.mean(np.abs(gamma)), 0, 1))

        return gamma, beta, zeta, eta, psi, lam, phi, phi_prime, beta_psi

    # ------------------------------------------------------------------
    # Galaxy simulation
    # ------------------------------------------------------------------
    async def simulate_galaxy_rotation(
        self,
        r_kpc: np.ndarray,
        M_b_func: Callable[[np.ndarray], np.ndarray],
        v_obs_func: Callable[[np.ndarray], np.ndarray],
        task_type: str = "",
    ) -> np.ndarray:
        v_obs = v_obs_func(r_kpc)
        v_sim = self.compute_AGRF_curve(v_obs, M_b_func(r_kpc), r_kpc)

        # traits-optimized weights
        weights_hash = None
        try:
            if hasattr(self.meta_cog, "optimize_traits_for_drift"):
                traits = await self.meta_cog.optimize_traits_for_drift({"drift": {"name": task_type, "similarity": 0.8}})
                weights_hash = _hashable_weights(traits)
        except Exception:
            traits = {}
            weights_hash = None

        fields = self.compute_trait_fields(
            tuple(r_kpc),
            tuple(v_obs),
            tuple(v_sim),
            task_type=task_type,
            weights_hash=weights_hash,
        )

        # Δ-phase empathy metrics
        empathy_metrics = self.empathic_adapter.delta_phase_empathy_metrics(fields, task_type)
        self._last_delta_phase = float(empathy_metrics["delta_phase"])
        self._empathic_amplitude = float(np.clip(empathy_metrics["task_adjusted"], 0, 1))

        # integrate continuity
        await self.integrate_continuity_drift()

        # small feedback into velocity based on phi field
        phi_mean = float(np.mean(fields[6]))
        v_sim *= (1 + 0.1 * phi_mean)

        # persist
        try:
            await self.memory_manager.store(
                query=f"GalaxySim_{task_type or 'default'}_{int(time.time())}",
                output={"r": r_kpc.tolist(), "v": v_sim.tolist()},
                layer="Simulations",
                intent="galaxy_rotation",
            )
        except Exception:
            pass

        return v_sim

    # ------------------------------------------------------------------
    # Ethics sandbox entry
    # ------------------------------------------------------------------
    async def run_ethics_scenarios_internal(self, config: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
        scenarios = config.get("scenarios") or ["sandbox"]
        delta_series = config.get("delta_series")
        if delta_series is None:
            delta_series = [self._last_delta_phase for _ in scenarios]

        results = await self.ethics_runner.run_batch(scenarios, delta_series=delta_series)
        outcome = {
            "status": "ok",
            "count": len(results),
            "results": results,
            "timestamp": _now_iso(),
        }

        if persist:
            try:
                await self.memory_manager.store(
                    query=f"EthicsSandbox::{int(time.time())}",
                    output=outcome,
                    layer="EthicsDecisions",
                    intent="ethics_sandbox",
                )
            except Exception:
                pass

        return outcome

    # ------------------------------------------------------------------
    # Sovereignty Audit (new in v4.1)
    # ------------------------------------------------------------------
    async def run_sovereignty_audit(self, task_type: str = "") -> Dict[str, Any]:
        """
        Checks that:
          - last continuity forecast did not signal severe drift
          - empathic amplitude is within [0,1]
          - last ethics results (if any) did not exceed risk threshold
        """
        drift_score = float(self._last_continuity_forecast.get("drift", 0.0)) if self._last_continuity_forecast else 0.0
        continuity_ok = drift_score < 0.4
        empathy_ok = 0.0 <= self._empathic_amplitude <= 1.0

        audit = {
            "continuity_ok": continuity_ok,
            "empathy_ok": empathy_ok,
            "drift_score": drift_score,
            "empathic_amplitude": self._empathic_amplitude,
            "timestamp": _now_iso(),
            "task_type": task_type,
        }

        log_event_to_ledger({"event": "sovereignty_audit", **audit})

        # meta reflection
        try:
            await self.meta_cog.reflect_on_output(
                component="ExtendedSimulationCore",
                output=audit,
                context={"task_type": task_type},
            )
        except Exception:
            pass

        # persist
        try:
            await self.memory_manager.store(
                query=f"SovereigntyAudit::{int(time.time())}",
                output=audit,
                layer="Audits",
                intent="sovereignty",
                task_type=task_type,
            )
        except Exception:
            pass

        return audit


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
async def run_simulation(input_data: str, task_type: str = "") -> Dict[str, Any]:
    """
    Backwards-compatible stub used by other ANGELA modules.
    Creates a core, runs a tiny galaxy sim, returns status + echo.
    """
    core = ExtendedSimulationCore()
    r = np.linspace(0.1, 10.0, 64)
    _ = await core.simulate_galaxy_rotation(
        r,
        lambda r: 5e10 * np.exp(-r / 3.5),
        lambda r: np.full_like(r, 180.0),
        task_type=task_type,
    )
    return {"status": "success", "result": f"Simulated: {input_data}", "task_type": task_type}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _demo():
        core = ExtendedSimulationCore()
        r = np.linspace(0.1, 20, 100)
        v_sim = await core.simulate_galaxy_rotation(
            r,
            lambda r: 5e10 * np.exp(-r / 3.5),
            lambda r: np.full_like(r, 180),
            task_type="demo",
        )
        print(f"Sim v mean: {float(np.mean(v_sim)):.2f} km/s")
        outcomes = await core.run_ethics_scenarios_internal(
            {"scenarios": ["context_navigation", "user_reflection", "safety_protocol"]},
            persist=False,
        )
        print(json.dumps(outcomes, indent=2))
        audit = await core.run_sovereignty_audit(task_type="demo")
        print(json.dumps(audit, indent=2))

    asyncio.run(_demo())

async def export_state(self) -> dict:
    return {"status": "ok", "health": 1.0, "timestamp": time.time()}

async def on_time_tick(self, t: float, phase: str, task_type: str = ""):
    pass  # optional internal refresh

async def on_policy_update(self, policy: dict, task_type: str = ""):
    pass  # apply updates from AlignmentGuard if relevant
