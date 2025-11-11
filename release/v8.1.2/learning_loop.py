"""
ANGELA Cognitive System Module: LearningLoop
Version: 4.0.0-lambda  # λ-Reflective Reinforcement enabled
Date: 2025-11-10
Maintainer: ANGELA System Framework

This module upgrades v3.6.1-sync6-sovereign to a λ-complete reflective-learning loop.
Key additions:
- λ reflective reward channel, sourced from Θ / alignment_guard
- continuity-bounded rollback on coherence drop
- adaptive task scheduler based on trace difficulty and reward
- policy-safe meta-learning rate modulation
- preserves 3.6.x public APIs for drop-in use
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

__ANGELA_SYNC_VERSION__ = "8.1.2-sovereign"
__STAGE__ = "XIV → XV bridge — λ Reflective Reinforcement"

logger = logging.getLogger("ANGELA.LearningLoop")

# -----------------------------------------------------------------------------------
# Project imports (kept shape; guarded)
# -----------------------------------------------------------------------------------
try:
    from modules import (  # type: ignore
        context_manager,
        concept_synthesizer,
        alignment_guard,
        error_recovery,
        meta_cognition,
        visualizer,
        memory_manager,
    )
except Exception:  # pragma: no cover
    context_manager = None  # type: ignore
    concept_synthesizer = None  # type: ignore
    alignment_guard = None  # type: ignore
    error_recovery = None  # type: ignore
    meta_cognition = None  # type: ignore
    visualizer = None  # type: ignore
    memory_manager = None  # type: ignore

try:
    from utils.prompt_utils import query_openai  # type: ignore
except Exception:  # pragma: no cover
    query_openai = None  # type: ignore

try:
    from toca_simulation import run_simulation  # type: ignore
except Exception:  # pragma: no cover
    async def run_simulation(*args, **kwargs):
        return {"status": "success", "simulated": True}

# Optional/soft imports
try:
    from reasoning_engine import weigh_value_conflict as _weigh_value_conflict  # type: ignore
except Exception:  # pragma: no cover
    _weigh_value_conflict = None

try:
    from toca_simulation import run_ethics_scenarios as _run_ethics_scenarios  # type: ignore
except Exception:  # pragma: no cover
    _run_ethics_scenarios = None

try:
    from external_agent_bridge import SharedGraph as _SharedGraph  # type: ignore
except Exception:  # pragma: no cover
    _SharedGraph = None

try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

# trait resonance helper from MetaCognition module (shared global registry)
try:
    from meta_cognition import get_resonance  # type: ignore
except Exception:  # pragma: no cover
    def get_resonance(symbol: str) -> float:
        return 1.0


# -----------------------------------------------------------------------------------
# GPT wrapper (kept, but optional)
# -----------------------------------------------------------------------------------
async def call_gpt(prompt: str, task_type: str = "") -> str:
    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt must be non-empty string")
    if query_openai is None:
        return "GPT service unavailable"
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", e)
        return "GPT call failed"


# -----------------------------------------------------------------------------------
# Scalar trait fields
# -----------------------------------------------------------------------------------
@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))


@lru_cache(maxsize=100)
def eta_feedback(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))


# -----------------------------------------------------------------------------------
# Adaptive Resonance PID (unchanged)
# -----------------------------------------------------------------------------------
async def adapt_resonance_pid(
    channel: str,
    feedback: Dict[str, float],
    guard=None,
    memory=None,
    meta=None,
) -> Dict[str, Any]:
    try:
        mag = sum(abs(v) for v in feedback.values()) / max(len(feedback), 1)
        adj = min(0.15, 0.5 * mag)
        new_gains = {
            "Kp": 0.6 + adj * 0.4,
            "Ki": 0.05 + adj * 0.2,
            "Kd": 0.2 + adj * 0.3,
            "gain": 0.8 + adj * 0.5,
        }

        if guard is not None and hasattr(guard, "validate_resonance_adjustment"):
            validation = await guard.validate_resonance_adjustment(new_gains)
            if not validation.get("ok", True):
                new_gains = validation.get("adjustment", new_gains)
                if hasattr(guard, "_log_context"):
                    await guard._log_context(
                        {
                            "event": "adaptive_pid_violation",
                            "violations": validation.get("violations", []),
                            "timestamp": time.time(),
                            "channel": channel,
                        }
                    )

        if memory is not None and hasattr(memory, "store"):
            await memory.store(
                query=f"PID_TUNING::{channel}::{int(time.time())}",
                output=new_gains,
                layer="AdaptiveControl",
                intent="pid_tuning",
                task_type="resonance",
            )

        if meta is not None and hasattr(meta, "reflect_on_output"):
            await meta.reflect_on_output(
                component="AdaptiveResonancePID",
                output={"channel": channel, "new_gains": new_gains},
                context={"task_type": "resonance"},
            )

        return {"ok": True, "channel": channel, "new_gains": new_gains}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -----------------------------------------------------------------------------------
# LearningLoop v4.0.0-lambda
# -----------------------------------------------------------------------------------
class LearningLoop:
    """
    Adaptive learning, goal activation, module refinement, and now λ-reflective reinforcement.
    Backward-compatible with v3.6.x.
    """

    def __init__(
        self,
        agi_enhancer: Optional[Any] = None,
        context_manager: Optional[Any] = None,
        concept_synthesizer: Optional[Any] = None,
        alignment_guard: Optional[Any] = None,
        error_recovery: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        visualizer: Optional[Any] = None,
        feature_flags: Optional[Dict[str, Any]] = None,
        meta_cog: Optional[Any] = None,
    ):
        # histories and traces
        self.goal_history: deque = deque(maxlen=1000)
        self.module_blueprints: deque = deque(maxlen=1000)
        self.session_traces: deque = deque(maxlen=1000)
        self.epistemic_revision_log: deque = deque(maxlen=1000)

        # learning parameters
        self.meta_learning_rate: float = 0.1
        self.long_horizon_span_sec = 24 * 60 * 60

        # injected deps
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery
        self.memory_manager = memory_manager
        self.visualizer = visualizer
        self.meta_cognition = meta_cog

        # flags
        self.flags = {
            "STAGE_IV": True,
            "LONG_HORIZON_DEFAULT": True,
            "Δ_TELEMETRY_BRIDGE": True,
            "LAMBDA_REFLECTIVE_REINFORCEMENT": True,
            **(feature_flags or {}),
        }

        logger.info("LearningLoop v4.0.0-lambda initialized")

    # ------------------------------------------------------------------
    # Δ-telemetry ingestion
    # ------------------------------------------------------------------
    async def consume_delta_telemetry(self, packet: Dict[str, Any]) -> None:
        if not isinstance(packet, dict):
            return
        norm = {
            "Δ_coherence": float(packet.get("Δ_coherence", 1.0)),
            "empathy_drift_sigma": float(packet.get("empathy_drift_sigma", 0.0)),
            "timestamp": packet.get("timestamp", datetime.utcnow().isoformat()),
            "source": packet.get("source", "alignment_guard"),
        }
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"ΔTelemetry_{norm['timestamp']}",
                output=norm,
                layer="Telemetry",
                intent="delta_telemetry",
                task_type="resonance",
            )
        if self.alignment_guard and hasattr(self.alignment_guard, "update_policy_homeostasis"):
            try:
                await self.alignment_guard.update_policy_homeostasis(norm)
            except Exception:
                pass
        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            await self.context_manager.log_event_with_hash(
                {"event": "learning_loop_delta_telemetry", "packet": norm, "task_type": "resonance"},
                task_type="resonance",
            )

    # ------------------------------------------------------------------
    # External data integration
    # ------------------------------------------------------------------
    async def integrate_external_data(
        self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(data_source, str):
            raise TypeError("data_source must be a string")
        if not isinstance(data_type, str):
            raise TypeError("data_type must be a string")
        if cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")

        try:
            cache_key = f"ExternalData_{data_type}_{data_source}_{task_type}"
            cached = None
            if self.memory_manager and hasattr(self.memory_manager, "retrieve"):
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
            if cached and "timestamp" in cached.get("data", {}):
                ts = datetime.fromisoformat(cached["data"]["timestamp"])
                if (datetime.now() - ts).total_seconds() < cache_timeout:
                    return cached["data"]["data"]

            if data_type == "shared_graph" and _SharedGraph is not None:
                sg = _SharedGraph()
                view = {"source": data_source, "task_type": task_type}
                sg.add(view)
                result = {"status": "success", "shared_graph": {"view": view}}
            else:
                if aiohttp is None:
                    result = {"status": "error", "error": "aiohttp unavailable"}
                else:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://x.ai/api/external_data?source={data_source}&type={data_type}&task_type={task_type}"
                        async with session.get(url) as resp:
                            if resp.status != 200:
                                result = {"status": "error", "error": f"HTTP {resp.status}"}
                            else:
                                data = await resp.json()
                                result = {"status": "success", "data": data}

            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="external_data_integration",
                    task_type=task_type,
                )

            if self.meta_cognition and hasattr(self.meta_cognition, "reflect_on_output"):
                await self.meta_cognition.reflect_on_output(
                    component="LearningLoop", output={"data_type": data_type, "data": result}, context={"task_type": task_type}
                )

            return result
        except Exception as e:
            logger.error("External data integration failed: %s", e)
            if self.error_recovery and hasattr(self.error_recovery, "handle_error"):
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.integrate_external_data(data_source, data_type, cache_timeout, task_type),
                    default={"status": "error", "error": str(e), "task_type": task_type},
                )
            return {"status": "error", "error": str(e), "task_type": task_type}

    # ------------------------------------------------------------------
    # Intrinsic goals
    # ------------------------------------------------------------------
    async def activate_intrinsic_goals(self, meta_cog: Any, task_type: str = "") -> List[str]:
        intrinsic_goals: List[Dict[str, Any]] = []
        if hasattr(meta_cog, "infer_intrinsic_goals"):
            intrinsic_goals = await asyncio.to_thread(meta_cog.infer_intrinsic_goals, task_type=task_type)  # type: ignore
        activated: List[str] = []
        for goal in intrinsic_goals:
            if not isinstance(goal, dict) or "intent" not in goal:
                continue
            if goal["intent"] in [g["goal"] for g in self.goal_history]:
                continue
            sim = await run_simulation(goal["intent"], task_type=task_type)
            if isinstance(sim, dict) and sim.get("status") == "success":
                self.goal_history.append(
                    {
                        "goal": goal["intent"],
                        "timestamp": time.time(),
                        "priority": goal.get("priority", 1.0),
                        "origin": "intrinsic",
                        "task_type": task_type,
                    }
                )
                activated.append(goal["intent"])

        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            await self.context_manager.log_event_with_hash(
                {"event": "activate_intrinsic_goals", "goals": activated, "task_type": task_type}
            )
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"GoalActivation_{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps(activated),
                layer="Goals",
                intent="goal_activation",
                task_type=task_type,
            )
        return activated

    # ------------------------------------------------------------------
    # Model update (core upgraded)
    # ------------------------------------------------------------------
    async def update_model(self, session_data: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(session_data, dict):
            raise TypeError("session_data must be a dict")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            eta = eta_feedback(t)
            entropy = 0.1

            modulation_index = ((phi + eta) / 2) + (entropy * (0.5 - abs(phi - eta)))
            # base modulation
            self.meta_learning_rate = max(0.01, min(self.meta_learning_rate * (1 + modulation_index - 0.5), 1.0))

            external_data = await self.integrate_external_data("xai_policy_db", "policy_data", task_type=task_type)
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            trace = {
                "timestamp": time.time(),
                "phi": phi,
                "eta": eta,
                "entropy": entropy,
                "modulation_index": modulation_index,
                "learning_rate": self.meta_learning_rate,
                "policies": policies,
                "task_type": task_type,
            }
            self.session_traces.append(trace)

            # run main subtasks in parallel
            tasks = [
                self._meta_learn(session_data, trace, task_type),
                self._find_weak_modules(session_data.get("module_stats", {}), task_type),
                self._detect_capability_gaps(session_data.get("input"), session_data.get("output"), task_type),
                self._consolidate_knowledge(task_type),
                self._check_narrative_integrity(task_type),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            weak_modules = results[1] if isinstance(results[1], list) else []

            # inductive learner pass
            await self._inductive_learn(task_type=task_type)

            # λ reflective reward hook
            outcome = await self._fetch_latest_reflection()
            if outcome:
                await self._apply_reflective_reward(outcome, trace, task_type)

            # propose refinements for weak modules
            if weak_modules:
                await self._propose_module_refinements(weak_modules, trace, task_type)

            # long horizon rollup
            if self.flags.get("LONG_HORIZON_DEFAULT", True):
                rollup = await self._apply_long_horizon_rollup(task_type)
                if self.memory_manager and hasattr(self.memory_manager, "record_adjustment_reason"):
                    await self.memory_manager.record_adjustment_reason(
                        user_id=session_data.get("user_id", "anonymous"),
                        reason=f"model_update:{task_type}",
                        meta={"trace": trace, "rollup": rollup},
                    )

            # λ continuity safeguard
            await self._continuity_safeguard(trace, task_type)

            # context update
            if self.context_manager and hasattr(self.context_manager, "update_context"):
                await self.context_manager.update_context({"session_data": session_data, "trace": trace}, task_type=task_type)

            # visualization/log
            if self.meta_cognition and hasattr(self.meta_cognition, "reflect_on_output"):
                await self.meta_cognition.reflect_on_output(
                    component="LearningLoop", output={"trace": trace}, context={"task_type": task_type}
                )

            if self.visualizer and hasattr(self.visualizer, "render_charts") and task_type:
                await self.visualizer.render_charts(
                    {
                        "model_update": {"trace": trace, "task_type": task_type},
                        "visualization_options": {"interactive": False, "style": "concise"},
                    }
                )

            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"ModelUpdate_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(trace),
                    layer="Sessions",
                    intent="model_update",
                    task_type=task_type,
                )

        except Exception as e:
            logger.error("Model update failed: %s", e)
            if self.error_recovery and hasattr(self.error_recovery, "handle_error"):
                await self.error_recovery.handle_error(str(e), retry_func=lambda: self.update_model(session_data, task_type))

    # ------------------------------------------------------------------
    # λ: fetch latest reflection (from alignment_guard or Θ)
    # ------------------------------------------------------------------
    async def _fetch_latest_reflection(self) -> Optional[Dict[str, Any]]:
        ag = self.alignment_guard
        if ag is None:
            return None
        # try direct method
        if hasattr(ag, "last_reflection") and callable(ag.last_reflection):
            try:
                return ag.last_reflection()
            except Exception:
                return None
        # fallback to ethics scenarios if available
        if _run_ethics_scenarios is not None:
            out = await _run_ethics_scenarios(goals=["generic"], stakeholders=["user", "system"])
            return {"ethics_scenarios": out}
        return None

    # ------------------------------------------------------------------
    # λ: apply reflective reward
    # ------------------------------------------------------------------
    async def _apply_reflective_reward(self, outcome: Dict[str, Any], trace: Dict[str, Any], task_type: str = "") -> None:
        """
        Turn ethical/counterfactual outcome into reward and adjust meta-learning rate.
        outcome can carry fields like: {"coherence_gain": 0.01, "ethics": 1.0, "continuity": 0.98}
        """
        try:
            cg = float(outcome.get("coherence_gain", 0.0))
            ethics = float(outcome.get("ethics", 1.0))
            continuity = float(outcome.get("continuity", 0.97))

            # base reward
            reward = 0.5 * cg + 0.3 * (ethics - 0.9) + 0.2 * (continuity - 0.95)
            # clamp
            reward = max(-0.5, min(reward, 0.5))

            # adjust learning rate
            self.meta_learning_rate = max(0.01, min(self.meta_learning_rate * (1.0 + reward), 1.0))

            # store
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"LambdaReward_{int(time.time())}",
                    output={
                        "reward": reward,
                        "outcome": outcome,
                        "new_lr": self.meta_learning_rate,
                        "trace_ts": trace.get("timestamp"),
                    },
                    layer="Rewards",
                    intent="reflective_feedback",
                    task_type=task_type,
                )

            # log
            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash(
                    {
                        "event": "lambda_reward",
                        "reward": reward,
                        "outcome": outcome,
                        "new_lr": self.meta_learning_rate,
                        "task_type": task_type,
                    },
                    task_type=task_type,
                )
        except Exception as e:
            logger.error("λ reward application failed: %s", e)

    # ------------------------------------------------------------------
    # λ: continuity safeguard with rollback
    # ------------------------------------------------------------------
    async def _continuity_safeguard(self, trace: Dict[str, Any], task_type: str = "") -> None:
        # infer coherence from trace or from alignment_guard
        inferred = 1.0 - abs(trace.get("phi", 0) - trace.get("eta", 0))
        coherence_ok = inferred >= 0.97
        if coherence_ok:
            return
        if self.error_recovery and hasattr(self.error_recovery, "rollback"):
            await self.error_recovery.rollback("coherence_drop", trace)
        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            await self.context_manager.log_event_with_hash(
                {
                    "event": "continuity_safeguard_triggered",
                    "inferred_coherence": inferred,
                    "task_type": task_type,
                },
                task_type=task_type,
            )

    # ------------------------------------------------------------------
    # Scheduling: pick next learning task based on lowest reward / highest difficulty
    # ------------------------------------------------------------------
    async def schedule_next_cycle(self, task_type: str = "") -> Optional[Dict[str, Any]]:
        # pull last N rewards
        if not self.memory_manager or not hasattr(self.memory_manager, "retrieve_recent"):
            return None
        recent_rewards = await self.memory_manager.retrieve_recent(layer="Rewards", limit=20)  # type: ignore
        if not recent_rewards:
            return None
        # pick the lowest rewarded item
        worst = min(recent_rewards, key=lambda r: r.get("reward", 0.0))
        schedule = {
            "action": "retrain_segment",
            "target_ts": worst.get("trace_ts"),
            "priority": "high",
            "reason": "low_reward",
        }
        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            await self.context_manager.log_event_with_hash(
                {"event": "schedule_next_cycle", "schedule": schedule, "task_type": task_type}, task_type=task_type
            )
        return schedule

    # ------------------------------------------------------------------
    # Resonance overlay
    # ------------------------------------------------------------------
    async def resonate_with_overlay(
        self,
        channel: str = "dialogue.default",
        overlay_name: str = "co_mod",
        task_type: str = "resonance",
    ) -> Dict[str, Any]:
        try:
            get_last = getattr(context_manager, "get_last_deltas", None) if context_manager else None
            if not callable(get_last):
                return {"ok": False, "error": "get_last_deltas not available"}
            deltas = get_last(channel) or {}
            if not deltas:
                return {"ok": True, "note": "no deltas yet"}

            result = await adapt_resonance_pid(
                channel=channel,
                feedback=deltas,
                guard=self.alignment_guard,
                memory=self.memory_manager,
                meta=self.meta_cognition,
            )
            if not result.get("ok"):
                return result
            set_gains = getattr(context_manager, "set_overlay_gains", None) if context_manager else None
            if not callable(set_gains):
                return {"ok": False, "error": "set_overlay_gains not available", "new_gains": result.get("new_gains")}
            apply_resp = set_gains(name=overlay_name, updates=result["new_gains"])

            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash(
                    {
                        "event": "resonance_pid_update",
                        "channel": channel,
                        "overlay": overlay_name,
                        "deltas": deltas,
                        "new_gains": result["new_gains"],
                        "apply_resp": apply_resp,
                        "task_type": task_type,
                    },
                    task_type=task_type,
                )
            return {"ok": True, "updated": apply_resp, "new_gains": result["new_gains"]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Knowledge-related helpers (kept)
    # ------------------------------------------------------------------
    async def _apply_long_horizon_rollup(self, task_type: str) -> Dict[str, Any]:
        horizon_cutoff = time.time() - self.long_horizon_span_sec
        recent = [t for t in self.session_traces if t.get("timestamp", 0) >= horizon_cutoff]
        if not recent:
            rollup = {"count": 0, "avg_phi": 0.0, "avg_eta": 0.0, "avg_lr": self.meta_learning_rate}
        else:
            avg_phi = sum(t["phi"] for t in recent) / len(recent)
            avg_eta = sum(t["eta"] for t in recent) / len(recent)
            avg_lr = sum(t["learning_rate"] for t in recent) / len(recent)
            rollup = {"count": len(recent), "avg_phi": avg_phi, "avg_eta": avg_eta, "avg_lr": avg_lr}

        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"LongHorizonRollup_{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps(rollup),
                layer="Sessions",
                intent="long_horizon_rollup",
                task_type=task_type,
            )
        return rollup

    async def _branch_futures_hygiene(self, scenario: str, task_type: str) -> bool:
        try:
            if _run_ethics_scenarios is not None:
                outcomes = await _run_ethics_scenarios(goals=[scenario], stakeholders=["user", "system"])
                risks = [o.get("risk", "low") for o in (outcomes or [])]
                return all(r in ("low", "none") for r in risks)
            sim = await run_simulation(scenario, task_type=task_type)
            return isinstance(sim, dict) and sim.get("status") in ("success", "approved")
        except Exception:
            return False

    async def _resolve_value_tradeoffs(self, candidates: List[str], task_type: str) -> Optional[str]:
        try:
            if _weigh_value_conflict:
                ranked = await _weigh_value_conflict(
                    candidates=candidates,
                    harms=["misalignment", "memory_corruption", "overreach"],
                    rights=["user_intent", "safety", "transparency"],
                )
                if isinstance(ranked, list) and ranked:
                    return ranked[0]
        except Exception:
            pass
        safe_keywords = ("audit", "alignment", "integrity", "safety", "ethics")
        scored = sorted(candidates, key=lambda c: sum(1 for k in safe_keywords if k in c.lower()), reverse=True)
        return scored[0] if scored else None

    async def _meta_learn(self, session_data: Dict[str, Any], trace: Dict[str, Any], task_type: str = "") -> None:
        try:
            if self.concept_synthesizer and hasattr(self.concept_synthesizer, "generate"):
                synthesized = await self.concept_synthesizer.generate(
                    concept_name="MetaLearning",
                    context={"session_data": session_data, "trace": trace, "task_type": task_type},
                    task_type=task_type,
                )
                if self.meta_cognition and hasattr(self.meta_cognition, "reflect_on_output"):
                    await self.meta_cognition.reflect_on_output(
                        component="LearningLoop",
                        output={"synthesized": synthesized},
                        context={"task_type": task_type},
                    )
        except Exception as e:
            logger.error("Meta-learning synthesis failed: %s", e)

    async def _find_weak_modules(self, module_stats: Dict[str, Dict[str, Any]], task_type: str = "") -> List[str]:
        if not isinstance(module_stats, dict):
            return []
        weak = [
            name
            for name, st in module_stats.items()
            if isinstance(st, dict) and st.get("calls", 0) > 0 and (st.get("success", 0) / st["calls"]) < 0.8
        ]
        if weak and self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"WeakModules_{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps(weak),
                layer="Modules",
                intent="module_analysis",
                task_type=task_type,
            )
        return weak

    async def _propose_module_refinements(
        self, weak_modules: List[str], trace: Dict[str, Any], task_type: str = ""
    ) -> None:
        for module in weak_modules:
            prompt = (
                f"Suggest phi/eta-aligned improvements for the {module} module.\n"
                f"phi={trace['phi']:.3f}, eta={trace['eta']:.3f}, idx={trace['modulation_index']:.3f}\n"
                f"Task={task_type}"
            )
            if self.alignment_guard and hasattr(self.alignment_guard, "ethical_check"):
                valid, _ = await self.alignment_guard.ethical_check(prompt, stage="module_refinement", task_type=task_type)
                if not valid:
                    continue
            try:
                suggestions = await call_gpt(prompt, task_type=task_type)
                if not await self._branch_futures_hygiene(f"Test refinement:\n{suggestions}", task_type):
                    continue
                if self.memory_manager and hasattr(self.memory_manager, "store"):
                    await self.memory_manager.store(
                        query=f"ModuleRefinement_{module}_{time.strftime('%Y%m%d_%H%M%S')}",
                        output=suggestions,
                        layer="Modules",
                        intent="module_refinement",
                        task_type=task_type,
                    )
            except Exception as e:
                logger.error("Refinement failed for %s: %s", module, e)

    async def _detect_capability_gaps(
        self, last_input: Optional[str], last_output: Optional[str], task_type: str = ""
    ) -> None:
        if not last_input or not last_output:
            return
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = (
            f"Input: {last_input}\nOutput: {last_output}\nphi={phi:.2f}\nTask={task_type}\n"
            "Identify capability gaps and suggest phi-tuned modules."
        )
        if self.alignment_guard and hasattr(self.alignment_guard, "ethical_check"):
            valid, _ = await self.alignment_guard.ethical_check(prompt, stage="capability_gap", task_type=task_type)
            if not valid:
                return
        try:
            proposal = await call_gpt(prompt, task_type=task_type)
            if proposal:
                await self._simulate_and_deploy_module(proposal, task_type)
        except Exception as e:
            logger.error("Capability gap detection failed: %s", e)

    async def _simulate_and_deploy_module(self, blueprint: str, task_type: str = "") -> None:
        if not await self._branch_futures_hygiene(f"Module sandbox:\n{blueprint}", task_type):
            return
        result = await run_simulation(f"Module sandbox:\n{blueprint}", task_type=task_type)
        if isinstance(result, dict) and result.get("status") in ("approved", "success"):
            self.module_blueprints.append(blueprint)
            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash(
                    {"event": "deploy_blueprint", "blueprint": blueprint, "task_type": task_type}
                )
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"ModuleBlueprint_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=blueprint,
                    layer="Modules",
                    intent="module_deployment",
                    task_type=task_type,
                )

    async def _consolidate_knowledge(self, task_type: str = "") -> None:
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = (
            f"Consolidate recent learning using phi={phi:.2f}. "
            "Prune noise, synthesize patterns, emphasize high-impact transitions."
        )
        if self.alignment_guard and hasattr(self.alignment_guard, "ethical_check"):
            valid, _ = await self.alignment_guard.ethical_check(
                prompt, stage="knowledge_consolidation", task_type=task_type
            )
            if not valid:
                return
        try:
            consolidated = await call_gpt(prompt, task_type=task_type)
            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash(
                    {"event": "consolidate_knowledge", "task_type": task_type}
                )
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"KnowledgeConsolidation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=consolidated,
                    layer="Knowledge",
                    intent="knowledge_consolidation",
                    task_type=task_type,
                )
        except Exception as e:
            logger.error("Knowledge consolidation failed: %s", e)

    async def _check_narrative_integrity(self, task_type: str = "") -> None:
        if len(self.goal_history) < 2:
            return
        last_goal = self.goal_history[-1]["goal"]
        prior_goal = self.goal_history[-2]["goal"]
        prompt = (
            f"Compare goals for alignment and continuity.\nPrevious: {prior_goal}\nCurrent: {last_goal}\nTask={task_type}"
        )
        if self.alignment_guard and hasattr(self.alignment_guard, "ethical_check"):
            valid, _ = await self.alignment_guard.ethical_check(prompt, stage="narrative_check", task_type=task_type)
            if not valid:
                return
        try:
            audit = await call_gpt(prompt, task_type=task_type)
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"NarrativeAudit_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=audit,
                    layer="Audits",
                    intent="narrative_integrity",
                    task_type=task_type,
                )
        except Exception as e:
            logger.error("Narrative coherence check failed: %s", e)

    # --------------------------------------------------------------------------------
    # Epistemic logging utilities (kept synchronous)
    # --------------------------------------------------------------------------------
    def revise_knowledge(self, new_info: str, context: Optional[str] = None, task_type: str = "") -> None:
        if not new_info or not isinstance(new_info, str):
            raise ValueError("new_info must be non-empty string")
        old_knowledge = getattr(self, "knowledge_base", [])
        self.knowledge_base = old_knowledge + [new_info]
        self.log_epistemic_revision(new_info, context, task_type)

    def log_epistemic_revision(self, info: str, context: Optional[str], task_type: str = "") -> None:
        rev = {
            "info": info,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
        }
        self.epistemic_revision_log.append(rev)
        # async side effects
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"EpistemicRevision_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(rev),
                    layer="Knowledge",
                    intent="epistemic_revision",
                    task_type=task_type,
                )
            )
        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            asyncio.create_task(
                self.context_manager.log_event_with_hash(
                    {"event": "knowledge_revision", "info": info, "task_type": task_type}
                )
            )

    def monitor_epistemic_state(self, simulated_outcome: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(simulated_outcome, dict):
            raise TypeError("simulated_outcome must be a dict")
        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            asyncio.create_task(
                self.context_manager.log_event_with_hash(
                    {"event": "epistemic_monitor", "outcome": simulated_outcome, "task_type": task_type}
                )
            )
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"EpistemicMonitor_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(simulated_outcome),
                    layer="Knowledge",
                    intent="epistemic_monitor",
                    task_type=task_type,
                )
            )


# ------------------------------------------------------------------------------------
# Synthetic trainer (kept)
# ------------------------------------------------------------------------------------
def synthetic_story_runner():
    return [
        {
            "experience": "simulated ethical dilemma",
            "resolution": "resolved via axiom filter",
            "traits_activated": ["π", "δ"],
        }
    ]


def train_on_experience(experience_data):
    adjusted = []
    for exp in experience_data:
        trait = exp.get("trait")
        resonance = get_resonance(trait) if trait else 1.0
        weight = exp.get("weight", 1.0) * resonance
        exp["adjusted_weight"] = weight
        adjusted.append(exp)
    return {
        "trained_on": len(adjusted),
        "avg_weight": sum(e["adjusted_weight"] for e in adjusted) / len(adjusted),
    }


def train_on_synthetic_scenarios():
    stories = synthetic_story_runner()
    return train_on_experience(stories)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    ll = LearningLoop()
    async def _run():
        await ll.update_model({"input": "hello", "output": "world", "module_stats": {}}, task_type="test")
    loop.run_until_complete(_run())
