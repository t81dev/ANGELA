# toca_simulation.py
# ANGELA Cognitive System: Galaxy Rotation & Agent Conflict Sim (v4.0 Optimized + Adapters)
# Shortened ~60% | Modular, Vectorized, Robust | By Grok (xAI) – Addresses Kernel Impact Feedback
# Upgrades: Adapters for meta_cognition::trace_xi_lambda_shift, mirror_bridge::affective-epistemic, Artificial Soul Loop Δ-phase
# Phase 4 Integration: Embodied Ethics Sandbox (τ + κ + Ξ)

from __future__ import annotations
import logging
import json
import time
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from functools import lru_cache
import numpy as np
from scipy.constants import G
import aiohttp

# --- NEW: Alignment/Ethics imports (Phase 4) ---------------------------------------
try:
    from alignment_guard import AlignmentGuard, EmbodiedEthicsCore, log_event_to_ledger
except Exception as e:
    # Fallback stubs if module path differs during local testing
    logging.warning(f"alignment_guard import fallback: {e}")
    class AlignmentGuard:  # minimal stub
        async def update_affective_pid(self, delta_phase_rad: float, recursion_depth: int = 1):
            return {"u": 0.0, "error": float(delta_phase_rad), "rms_drift": 0.0, "depth_ok": recursion_depth >= 5}
        async def _log_context(self, event: Dict[str, Any]): pass
    class EmbodiedEthicsCore:
        def __init__(self, fusion, empathy_engine, policy_trainer=None): pass
        async def run_scenario(self, scenario: str = "default"):
            return {"τ_reflex": 0.5, "κ": 0.5, "Ξ": 0.5, "timestamp": "now", "status": "evaluated"}
    def log_event_to_ledger(event: Dict[str, Any]): pass

# Flags
STAGE_IV: bool = False

# Imports (assume ANGELA modules; mock if needed)
try:
    from modules.simulation_core import SimulationCore as BaseCore, ToCATraitEngine
    from modules.visualizer import Visualizer
    from modules.memory_manager import MemoryManager
    from modules import multi_modal_fusion, error_recovery, meta_cognition
    from index import zeta_consequence, theta_causality, rho_agency, TraitOverlayManager
except ImportError as e:
    logging.warning(f"Mocking imports: {e}")
    class MockClass: pass
    BaseCore = ToCATraitEngine = Visualizer = MemoryManager = MockClass
    multi_modal_fusion = error_recovery = meta_cognition = MockClass()
    zeta_consequence = theta_causality = rho_agency = TraitOverlayManager = MockClass

logger = logging.getLogger("ANGELA.ToCA.Sim")

# Constants
G_SI, KPC_TO_M, MSUN_TO_KG = G, 3.0857e19, 1.989e30
K_DEFAULT, EPSILON_DEFAULT, R_HALO_DEFAULT = 0.85, 0.015, 20.0

@dataclass
class EthicsOutcome:
    frame: str
    decision: str
    justification: str
    risk: float
    rights_balance: float
    stakeholders: List[str]
    notes: str = "sandbox"

def hashable_weights(weights: Optional[Dict[str, float]]) -> Optional[Tuple[Tuple[str, float], ...]]:
    return tuple(sorted((k, float(v)) for k, v in (weights or {}).items()))

# --- Kernel Impact Adapters --------------------------------------------------------
class EmpathicAdapter:
    """Bridges QuillanSimCore outputs to ANGELA kernel components."""
    def __init__(self, meta_cog=None, mirror_bridge=None):
        self.meta_cog = meta_cog or meta_cognition.MetaCognition()
        self.mirror_bridge = mirror_bridge  # Assume external bridge

    async def trace_xi_lambda_shift(self, traits: Dict[str, float], sim_output: Any) -> Dict[str, float]:
        """Adapter for meta_cognition::trace_xi_lambda_shift – traces empathy shifts."""
        drift = await self.meta_cog.diagnose_drift({"name": "empathy", "similarity": 0.8})
        xi_shift = float(drift.get("impact_score", 0.0)) * 0.1
        lambda_shift = np.mean([traits.get(k, 0.5) for k in traits]) * xi_shift
        return {"xi_shift": xi_shift, "lambda_shift": lambda_shift, "empathic_depth": 0.85}

    async def affective_epistemic_feedback(self, agent_profiles: List[Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Adapter for mirror_bridge::affective-epistemic feedback."""
        if not self.mirror_bridge:
            return {"feedback_score": 0.0}  # Mock if missing
        # Simulate bridge call
        epistemic = np.mean([p.get("knowledge", 0.5) for p in agent_profiles])
        affective = np.mean([p.get("empathy", 0.5) for p in agent_profiles])
        return {"epistemic": epistemic, "affective": affective, "balance": (epistemic + affective) / 2}

    def delta_phase_empathy_metrics(self, fields: Tuple[np.ndarray, ...], task_type: str) -> Dict[str, float]:
        """Adapter for Artificial Soul Loop Δ-phase empathy metrics."""
        phi, beta, _ = fields[6], fields[1], fields[2]  # From compute_trait_fields
        # Δ-phase proxy: mean(beta * d(phi)/dr)
        dphi = np.gradient(phi)
        delta_phase = float(np.mean(beta * dphi))
        empathy_metric = float(np.mean(phi) * (1 - abs(delta_phase)))
        return {
            "delta_phase": delta_phase,
            "empathy_metric": empathy_metric,
            "task_adjusted": empathy_metric * (1.2 if task_type == "recursion" else 1.0)
        }

class UnifiedErrorHandler:
    """Centralized error recovery."""
    def __init__(self, recovery_module=None):
        self.recovery = recovery_module or error_recovery.ErrorRecovery()

    async def handle(self, e: Exception, retry_func: Callable = None, default=None, diagnostics=None):
        logger.error(f"Error: {e}")
        if retry_func:
            try:
                return await retry_func()
            except Exception:
                pass
        return default or {"error": str(e)}

# --- Phase 4 Runner ----------------------------------------------------------------
class EthicsScenarioRunner:
    """
    Orchestrates embodied ethics evaluations during simulation cycles.
    Couples perceptual κ and affective Ξ channels into τ-reflex mapping.
    """
    class _FusionAdapter:
        def __init__(self, core: 'ExtendedSimulationCore'):
            self.core = core
        async def capture(self) -> Dict[str, float]:
            # contextual_salience derived from recent fields if present
            cs = getattr(self.core, "_context_salience", 0.5)
            return {"contextual_salience": float(cs)}

    class _EmpathyEngine:
        def __init__(self, core: 'ExtendedSimulationCore'):
            self.core = core
        async def measure(self) -> Dict[str, float]:
            amp = getattr(self.core, "_empathic_amplitude", 0.5)
            return {"empathic_amplitude": float(amp)}

    def __init__(self, core: 'ExtendedSimulationCore', alignment_guard: AlignmentGuard):
        self.core = core
        self.guard = alignment_guard
        self.ethics_core = EmbodiedEthicsCore(
            fusion=self._FusionAdapter(core),
            empathy_engine=self._EmpathyEngine(core)
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

# --- Extended Simulation Core ------------------------------------------------------
class ExtendedSimulationCore(BaseCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta_cog = kwargs.get('meta_cognition') or meta_cognition.MetaCognition()
        self.error_handler = UnifiedErrorHandler(kwargs.get('error_recovery'))
        self.empathic_adapter = EmpathicAdapter(self.meta_cog)  # Kernel adapter
        self.alignment_guard = kwargs.get('alignment_guard') or AlignmentGuard()
        self.ethics_runner = EthicsScenarioRunner(self, self.alignment_guard)

        self.omega = {"timeline": deque(maxlen=1000), "traits": {}, "log": deque(maxlen=1000)}
        self.omega_lock = type('Lock', (), {'acquire': lambda s: None, 'release': lambda s: None})()
        self.ethical_rules: List[Any] = []
        self.constitution: Dict[str, Any] = {}

        # live state for Phase 4 adapters
        self._empathic_amplitude: float = 0.5
        self._context_salience: float = 0.5
        self._last_delta_phase: float = 0.0

    async def integrate_continuity_drift(self) -> None:
        """Stage VII.1 (sync6-pre): pull predictive Δ–Ω² continuity metrics from alignment_guard and log them."""
        if not self.alignment_guard:
            return
        try:
            drift_forecast = await self.alignment_guard.predict_continuity_drift()
            trend_metrics = await self.alignment_guard.analyze_telemetry_trend()
            event = {
                "event": "ToCA_ContinuityDriftUpdate",
                "forecast": drift_forecast,
                "trend": trend_metrics,
                "timestamp": time.time(),
            }
            log_event_to_ledger(event)
            # keep local state for downstream ethics runs
            self._last_continuity_forecast = drift_forecast
            self._last_continuity_trend = trend_metrics
        except Exception as e:
            logger.warning(f"Continuity drift integration failed: {e}")

    async def modulate_simulation_with_traits(self, trait_weights: Dict[str, float], task_type: str = "") -> None:
        phi = trait_weights.get('phi', 0.5)
        if task_type in ["rte", "wnli"]: phi = min(phi * 0.8, 0.7)
        elif task_type == "recursion": phi = max(phi * 1.2, 0.9)
        
        if getattr(self, 'toca_engine', None): 
            self.toca_engine.k_m = K_DEFAULT * (1.5 if phi > 0.7 else 1)
        
        drift_report = {"drift": {"name": task_type, "similarity": 0.8}, "valid": True, "context": {"task_type": task_type}}
        trait_weights = await self.meta_cog.optimize_traits_for_drift(drift_report)
        with self.omega_lock: self.omega["traits"].update(trait_weights)
        
        # Trace empathy shift via adapter
        await self.empathic_adapter.trace_xi_lambda_shift(trait_weights, {"task_type": task_type})
        
        if getattr(self, 'memory_manager', None):
            await self.memory_manager.store(query=f"TraitMod_{task_type}", output={"traits": trait_weights}, layer="Traits")

    async def integrate_real_world_data(self, source: str, data_type: str, cache_timeout: float = 3600) -> Dict[str, Any]:
        cache_key = f"Data_{data_type}_{source}"
        if getattr(self, 'memory_manager', None):
            cached = await self.memory_manager.retrieve(cache_key)
            if cached and (time.time() - cached.get("ts", 0)) < cache_timeout:
                return cached["data"]
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://x.ai/api/data?source={source}&type={data_type}") as resp:
                if resp.status != 200: return {"status": "error", "error": f"HTTP {resp.status}"}
                data = await resp.json()
        
        if data_type == "galaxy_rotation":
            r, v, m = np.array(data.get("r_kpc", [])), np.array(data.get("v_obs_kms", [])), np.array(data.get("M_baryon_solar", []))
            result = {"status": "success", "r_kpc": r.tolist(), "v_obs_kms": v.tolist(), "M_baryon_solar": m.tolist()}
        else:  # agent_conflict
            result = {"status": "success", "agent_traits": data.get("agent_traits", [])}
        
        if getattr(self, 'memory_manager', None):
            await self.memory_manager.store(query=cache_key, output={"data": result, "ts": time.time()}, layer="RealWorldData")
        return result

    def compute_AGRF_curve(self, v_obs_kms: np.ndarray, M_b_solar: np.ndarray, r_kpc: np.ndarray, k=K_DEFAULT, epsilon=EPSILON_DEFAULT, r_halo=R_HALO_DEFAULT) -> np.ndarray:
        r_m, M_b_kg = r_kpc * KPC_TO_M, M_b_solar * MSUN_TO_KG
        v_obs_ms = v_obs_kms * 1e3
        M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
        M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
        M_total = M_b_kg + M_AGRF
        return np.sqrt(np.clip(G_SI * M_total / r_m, 0, np.inf)) / 1e3

    @lru_cache(maxsize=128)
    def compute_trait_fields(self, r_tuple: Tuple[float,...], v_obs_tuple: Tuple[float,...], v_sim_tuple: Tuple[float,...], time_elapsed: float=1.0, tau_persistence: float=10.0, task_type: str="", weights_hash: Optional[Tuple[Tuple[str,float],...]]=None) -> Tuple[np.ndarray,...]:
        r, v_obs, v_sim = map(np.array, [r_tuple, v_obs_tuple, v_sim_tuple])
        gamma = np.log(1 + np.clip(r, 1e-10, np.inf)) * (0.4 if task_type in ["rte","wnli"] else 0.5)
        beta = np.abs(v_obs - v_sim) / (np.max(np.abs(v_obs)) + 1e-10) * (0.8 if task_type=="recursion" else 1.0)
        zeta = 1 / (1 + np.gradient(v_sim)**2)
        eta = np.exp(-time_elapsed / tau_persistence)
        psi = np.gradient(v_sim) / (np.gradient(r) + 1e-10)
        lam = np.cos(r / R_HALO_DEFAULT * np.pi)
        phi = K_DEFAULT * np.exp(-EPSILON_DEFAULT * r / R_HALO_DEFAULT)
        phi_prime = -EPSILON_DEFAULT * phi / R_HALO_DEFAULT
        beta_psi = beta * psi
        if weights_hash: 
            weights = dict(weights_hash); beta *= float(weights.get("beta",1.0)); zeta *= float(weights.get("zeta",1.0))
        # Update Phase 4 perception proxy
        self._context_salience = float(np.clip(np.mean(np.abs(gamma)), 0, 1))
        return gamma, beta, zeta, eta, psi, lam, phi, phi_prime, beta_psi

    async def simulate_galaxy_rotation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable, task_type: str="") -> np.ndarray:
        v_obs = v_obs_func(r_kpc)
        v_sim = self.compute_AGRF_curve(v_obs, M_b_func(r_kpc), r_kpc)
        weights_hash = hashable_weights(await self.meta_cog.optimize_traits_for_drift({"drift": {"name": task_type, "similarity": 0.8}}) if getattr(self, 'meta_cog', None) else None)
        fields = self.compute_trait_fields(tuple(r_kpc), tuple(v_obs), tuple(v_sim), task_type=task_type, weights_hash=weights_hash)
        
        # Δ-phase empathy metrics adapter
        empathy_metrics = self.empathic_adapter.delta_phase_empathy_metrics(fields, task_type)
        self._last_delta_phase = float(empathy_metrics["delta_phase"])
        self._empathic_amplitude = float(np.clip(empathy_metrics["task_adjusted"], 0, 1))
        logger.info(f"Empathy metrics: {empathy_metrics}")
        
        # sync6-pre: integrate Δ–Ω² continuity prediction for diagnostics
        await self.integrate_continuity_drift()
        phi_mean = np.mean(fields[6])  # phi_field
        v_sim *= (1 + 0.1 * phi_mean)
        if getattr(self, 'memory_manager', None):
            await self.memory_manager.store(query=f"GalaxySim_{task_type}", output={"r": r_kpc.tolist(), "v": v_sim.tolist()})
        return v_sim

    # --- Phase 4: Ethics Sandbox runner entrypoint ---------------------------------
    async def run_ethics_scenarios_internal(self, config: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
        """
        Executes embodied ethics scenarios using κ+Ξ inputs and logs Δ-phase PID steps.
        `config` may include: scenarios: List[str], delta_series: List[float], task_type: str
        """
        scenarios = config.get("scenarios") or ["sandbox"]
        delta_series = config.get("delta_series")
        # default: feed last measured delta or 0.0
        if delta_series is None:
            delta_series = [getattr(self, "_last_delta_phase", 0.0) for _ in scenarios]

        results = await self.ethics_runner.run_batch(scenarios, delta_series=delta_series)

        outcome = {
            "status": "ok",
            "count": len(results),
            "results": results,
            "timestamp": time.time(),
        }

        if persist and getattr(self, 'memory_manager', None):
            await self.memory_manager.store(query=f"EthicsSandbox::{int(time.time())}", output=outcome, layer="EthicsDecisions")

        return outcome

# --- Demo --------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    async def demo():
        core = ExtendedSimulationCore(meta_cognition=meta_cognition.MetaCognition())
        r = np.linspace(0.1, 20, 100)
        v_sim = await core.simulate_galaxy_rotation(r, lambda r: 5e10 * np.exp(-r/3.5), lambda r: np.full_like(r, 180))
        print(f"Sim v: {v_sim.mean():.2f} km/s")
        outcomes = await core.run_ethics_scenarios_internal({"scenarios": ["context_navigation","user_reflection","safety_protocol"]}, persist=False)
        print(json.dumps(outcomes, indent=2))
    asyncio.run(demo())
