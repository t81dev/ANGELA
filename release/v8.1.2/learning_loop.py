"""
ANGELA Cognitive System Module: LearningLoop
Version: 3.6.1-sync6-sovereign  # keeps 3.6.0 APIs, fixes MetaCognition calls, inlines resonance, safer optional deps
Date: 2025-11-07
Maintainer: ANGELA System Framework

This module provides a LearningLoop class for adaptive learning, goal activation, and module refinement
in the ANGELA v3.6.x architecture.

Changes vs 3.6.0:
- keep all public APIs and behaviors
- stop creating fresh MetaCognition() instances; reuse injected self.meta_cognition when available
- harden Δ-telemetry ingestion path and policy-homeostasis calls
- fold resonate_with_overlay(...) into LearningLoop as an instance method
- guard imports and optional runtime calls more tightly
- preserve “branch futures hygiene” and shared-graph v2 ingestion
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

# Stage markers (aligned with other modules)
__ANGELA_SYNC_VERSION__ = "6.2.0-sovereign"
__STAGE__ = "VII.8 — Harmonic Sovereignty Layer (LearningLoop)"

logger = logging.getLogger("ANGELA.LearningLoop")

# -------------------------------------------------------------------------------------------------
# Project imports (kept same shape for drop-in compatibility)
# -------------------------------------------------------------------------------------------------
from modules import (  # type: ignore
    context_manager,
    concept_synthesizer,
    alignment_guard,
    error_recovery,
    meta_cognition,
    visualizer,
    memory_manager,
)
from utils.prompt_utils import query_openai  # type: ignore
from toca_simulation import run_simulation  # type: ignore

# Optional/soft imports ---------------------------------------------------------------
try:
    # upcoming API
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
    import aiohttp  # may be absent or network-blocked at runtime
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

# trait resonance helper from MetaCognition module (shared global registry)
try:
    from meta_cognition import get_resonance  # type: ignore
except Exception:  # pragma: no cover
    def get_resonance(symbol: str) -> float:
        return 1.0


# -------------------------------------------------------------------------------------------------
# GPT wrapper (unchanged API, guarded)
# -------------------------------------------------------------------------------------------------
async def call_gpt(prompt: str, task_type: str = "") -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(task_type, str):
        raise TypeError("task_type must be a string")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception for task %s: %s", task_type, e)
        raise


# -------------------------------------------------------------------------------------------------
# Scalar trait fields
# -------------------------------------------------------------------------------------------------
@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))


@lru_cache(maxsize=100)
def eta_feedback(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))


# -------------------------------------------------------------------------------------------------
# Adaptive Resonance PID (shared helper)
# -------------------------------------------------------------------------------------------------
async def adapt_resonance_pid(
    channel: str,
    feedback: Dict[str, float],
    guard=None,
    memory=None,
    meta=None,
) -> Dict[str, Any]:
    """
    Adaptive tuning of Co-Mod PID coefficients based on recent resonance feedback.
    Returned gains are already validated/clamped when guard is present.
    """
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

        if memory is not None:
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


# -------------------------------------------------------------------------------------------------
# LearningLoop v3.6.1
# -------------------------------------------------------------------------------------------------
class LearningLoop:
    """Adaptive learning, goal activation, and module refinement (v3.6.1).

    Backward-compatible with 3.5.3/3.6.0, with stronger sync6 alignment.
    """

    def __init__(
        self,
        agi_enhancer: Optional["AGIEnhancer"] = None,
        context_manager: Optional["context_manager.ContextManager"] = None,
        concept_synthesizer: Optional["concept_synthesizer.ConceptSynthesizer"] = None,
        alignment_guard: Optional["alignment_guard.AlignmentGuard"] = None,
        error_recovery: Optional["error_recovery.ErrorRecovery"] = None,
        memory_manager: Optional["memory_manager.MemoryManager"] = None,
        visualizer: Optional["visualizer.Visualizer"] = None,
        feature_flags: Optional[Dict[str, Any]] = None,
        meta_cog: Optional["meta_cognition.MetaCognition"] = None,
    ):
        self.goal_history: deque = deque(maxlen=1000)
        self.module_blueprints: deque = deque(maxlen=1000)
        self.meta_learning_rate: float = 0.1
        self.session_traces: deque = deque(maxlen=1000)
        self.epistemic_revision_log: deque = deque(maxlen=1000)

        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = (
            error_recovery
            or error_recovery_module.ErrorRecovery(  # type: ignore[attr-defined]
                context_manager=context_manager,
                alignment_guard=alignment_guard,
            )
            if "error_recovery_module" in globals()
            else error_recovery or error_recovery  # type: ignore
        )
        # prefer injected memory manager, else create
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()  # type: ignore[name-defined]
        self.visualizer = visualizer or visualizer_module.Visualizer()  # type: ignore[name-defined]
        # new: prefer injected MetaCognition
        self.meta_cognition = meta_cog or (meta_cognition.MetaCognition() if meta_cognition else None)

        self.flags = {
            "STAGE_IV": True,
            "LONG_HORIZON_DEFAULT": True,
            "Δ_TELEMETRY_BRIDGE": True,
            **(feature_flags or {}),
        }
        self.long_horizon_span_sec = 24 * 60 * 60

        logger.info("LearningLoop v3.6.1-sync6-sovereign initialized")

    # ------------------------------------------------------------------
    # Δ-telemetry ingestion (sync6)
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
        if self.memory_manager:
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
    # External data integration (with shared-graph)
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
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            cache_key = f"ExternalData_{data_type}_{data_source}_{task_type}"
            cached = (
                await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if self.memory_manager
                else None
            )
            if cached and "timestamp" in cached.get("data", {}):
                ts = datetime.fromisoformat(cached["data"]["timestamp"])
                if (datetime.now() - ts).total_seconds() < cache_timeout:
                    return cached["data"]["data"]

            if data_type == "shared_graph" and _SharedGraph is not None:
                sg = _SharedGraph()
                view = {"source": data_source, "task_type": task_type}
                sg.add(view)
                view_id = f"sg:{data_source}:{task_type}:{int(time.time())}"
                result = {"status": "success", "shared_graph": {"view": view, "view_id": view_id}}
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
                                if data_type == "agent_data":
                                    agent_data = data.get("agent_data", [])
                                    result = {"status": "success", "agent_data": agent_data} if agent_data else {"status": "error", "error": "No agent data"}
                                elif data_type == "policy_data":
                                    policies = data.get("policies", [])
                                    result = {"status": "success", "policies": policies} if policies else {"status": "error", "error": "No policies"}
                                else:
                                    result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="external_data_integration",
                    task_type=task_type,
                )

            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="LearningLoop",
                        output={"data_type": data_type, "data": result},
                        context={"task_type": task_type},
                    )
                except Exception:
                    pass

            return result
        except Exception as e:
            logger.error("External data integration failed: %s", e)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.integrate_external_data(data_source, data_type, cache_timeout, task_type),
                    default={"status": "error", "error": str(e), "task_type": task_type},
                )
            return {"status": "error", "error": str(e), "task_type": task_type}

    # ------------------------------------------------------------------
    # Intrinsic goals
    # ------------------------------------------------------------------
    async def activate_intrinsic_goals(
        self, meta_cog: "meta_cognition.MetaCognition", task_type: str = ""
    ) -> List[str]:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        # tolerate MetaCognition implementations without infer_intrinsic_goals
        intrinsic_goals: List[Dict[str, Any]] = []
        if hasattr(meta_cog, "infer_intrinsic_goals") and callable(meta_cog.infer_intrinsic_goals):
            intrinsic_goals = await asyncio.to_thread(meta_cog.infer_intrinsic_goals, task_type=task_type)  # type: ignore[arg-type]

        activated: List[str] = []
        for goal in intrinsic_goals:
            if not isinstance(goal, dict) or "intent" not in goal or "priority" not in goal:
                continue
            if goal["intent"] in [g["goal"] for g in self.goal_history]:
                continue
            sim = await run_simulation(goal["intent"], task_type=task_type)
            if isinstance(sim, dict) and sim.get("status") == "success":
                self.goal_history.append(
                    {
                        "goal": goal["intent"],
                        "timestamp": time.time(),
                        "priority": goal["priority"],
                        "origin": "intrinsic",
                        "task_type": task_type,
                    }
                )
                activated.append(goal["intent"])

        if self.context_manager:
            await self.context_manager.log_event_with_hash(
                {"event": "activate_intrinsic_goals", "goals": activated, "task_type": task_type}
            )

        if self.meta_cognition:
            try:
                await self.meta_cognition.reflect_on_output(
                    component="LearningLoop", output={"activated_goals": activated}, context={"task_type": task_type}
                )
            except Exception:
                pass

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"GoalActivation_{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps(activated),
                layer="Goals",
                intent="goal_activation",
                task_type=task_type,
            )

        if self.visualizer and task_type:
            await self.visualizer.render_charts(
                {
                    "goal_activation": {"goals": activated, "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                }
            )

        return activated

    # ------------------------------------------------------------------
    # Model update
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

            tasks = [
                self._meta_learn(session_data, trace, task_type),
                self._find_weak_modules(session_data.get("module_stats", {}), task_type),
                self._detect_capability_gaps(session_data.get("input"), session_data.get("output"), task_type),
                self._consolidate_knowledge(task_type),
                self._check_narrative_integrity(task_type),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # XIV.1 inductive learner pass
            await self._inductive_learn(task_type=task_type)
            weak_modules = results[1] if isinstance(results[1], list) else []

            if self.alignment_guard and hasattr(self.alignment_guard, "update_policy_homeostasis"):
                try:
                    await self.alignment_guard.update_policy_homeostasis(
                        {"Δ_coherence": 1.0 - abs(phi - eta), "empathy_drift_sigma": 0.0}
                    )
                except Exception:
                    pass

            if weak_modules:
                await self._propose_module_refinements(weak_modules, trace, task_type)

            if self.flags.get("LONG_HORIZON_DEFAULT", True):
                rollup = await self._apply_long_horizon_rollup(task_type)
                mm = self.memory_manager
                if mm and hasattr(mm, "record_adjustment_reason"):
                    try:
                        await mm.record_adjustment_reason(
                            user_id=session_data.get("user_id", "anonymous"),
                            reason=f"model_update:{task_type}",
                            meta={"trace": trace, "rollup": rollup},
                        )
                    except Exception:
                        pass

            if self.context_manager:
                await self.context_manager.update_context({"session_data": session_data, "trace": trace}, task_type=task_type)

            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="LearningLoop", output={"trace": trace}, context={"task_type": task_type}
                    )
                except Exception:
                    pass

            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "model_update": {"trace": trace, "task_type": task_type},
                        "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                    }
                )

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ModelUpdate_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(trace),
                    layer="Sessions",
                    intent="model_update",
                    task_type=task_type,
                )
        except Exception as e:
            logger.error("Model update failed: %s", e)
            if self.error_recovery:
                await self.error_recovery.handle_error(str(e), retry_func=lambda: self.update_model(session_data, task_type))


    async def _inductive_learn(self, task_type: str = "") -> None:
        """XIV.1: mine traces → propose rules → validate via logic/ethics → commit.
        Non-destructive: runs only if required deps exist.
        """
        mm = self.memory_manager
        ag = self.alignment_guard

        # prerequisites
        if mm is None or ag is None:
            return
        # memory manager must expose trace-like data
        traces = getattr(mm, "traces", None)
        if not traces:
            return

        # try to get logic functor if present
        try:
            from knowledge_retriever import LogicFunctor  # type: ignore
            # assume we can reuse the reasoning engine from alignment guard if it has one
            reasoning_engine = getattr(ag, "reasoning_engine", None)
            logic_functor = LogicFunctor(reasoning_engine) if reasoning_engine else None
        except Exception:
            logic_functor = None

        # group by (src, tgt)
        buckets = {}
        for tr in traces:
            src = tr.get("src")
            tgt = tr.get("tgt")
            if not src or not tgt:
                continue
            key = (src, tgt)
            buckets.setdefault(key, []).append(tr)

        # iterate candidates
        for (src, tgt), items in buckets.items():
            if len(items) < 2:
                continue  # need repetition to justify a rule
            rule = {
                "name": f"learned_{src}_to_{tgt}",
                "src": src,
                "tgt": tgt,
                "pattern": "unary",
            }

            # ethics gate first
            if hasattr(ag, "approve_learned_rule"):
                if not ag.approve_learned_rule(rule):
                    continue

            # logic gate
            if logic_functor is not None:
                src_f = logic_functor.map_object(type("Obj", (), {"type": src, "name": src})())
                tgt_f = logic_functor.map_object(type("Obj", (), {"type": tgt, "name": tgt})())
                if not logic_functor.entails(src_f, tgt_f) and hasattr(logic_functor, "prover"):
                    # if logic cannot show entailment, skip
                    continue

            # commit: for now, store into memory as a learned rule
            if mm is not None and hasattr(mm, "store"):
                await mm.store(
                    query=f"LearnedRule_{src}_to_{tgt}_{int(time.time())}",
                    output=rule,
                    layer="LearnedRules",
                    intent="inductive_learn",
                    task_type=task_type,
                )
    # ------------------------------------------------------------------
    # Autonomous goal proposal
    # ------------------------------------------------------------------
    async def propose_autonomous_goal(self, task_type: str = "") -> Optional[str]:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = (
            f"Propose 3 high-level, safe, phi-aligned autonomous goals.\nphi={phi:.2f}\nTask={task_type}\n"
            "Return as bullet list."
        )

        if self.alignment_guard and hasattr(self.alignment_guard, "ethical_check"):
            valid, report = await self.alignment_guard.ethical_check(prompt, stage="goal_proposal", task_type=task_type)
            if not valid:
                logger.warning("Autonomous goal prompt failed alignment: %s", report)
                return None

        try:
            candidates_blob = await call_gpt(prompt, task_type=task_type)
            candidates = [c.strip("-• ").strip() for c in candidates_blob.splitlines() if c.strip()]
            candidates = [c for c in candidates if c] or ["Improve robustness of narrative integrity checks"]
            goal = await self._resolve_value_tradeoffs(candidates, task_type) or candidates[0]

            if goal in [g["goal"] for g in self.goal_history]:
                return None

            if not await self._branch_futures_hygiene(f"Goal test: {goal}", task_type):
                return None

            self.goal_history.append({"goal": goal, "timestamp": time.time(), "phi": phi, "task_type": task_type})

            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "propose_autonomous_goal", "goal": goal, "task_type": task_type}
                )

            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="LearningLoop", output={"goal": goal}, context={"task_type": task_type}
                    )
                except Exception:
                    pass

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"AutonomousGoal_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=goal,
                    layer="Goals",
                    intent="goal_proposal",
                    task_type=task_type,
                )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "goal_proposal": {"goal": goal, "task_type": task_type},
                        "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                    }
                )
            return goal
        except Exception as e:
            logger.error("Goal proposal failed: %s", e)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.propose_autonomous_goal(task_type), default=None
                )
            return None

    # ------------------------------------------------------------------
    # Embedded resonance overlay integration
    # ------------------------------------------------------------------
    async def resonate_with_overlay(
        self,
        channel: str = "dialogue.default",
        overlay_name: str = "co_mod",
        task_type: str = "resonance",
    ) -> Dict[str, Any]:
        """
        Pull recent overlay deltas (from context_manager overlay), adapt PID/gain, and push safe gains back.
        """
        try:
            get_last = getattr(context_manager, "get_last_deltas", None)
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

            set_gains = getattr(context_manager, "set_overlay_gains", None)
            if not callable(set_gains):
                return {"ok": False, "error": "set_overlay_gains not available", "new_gains": result.get("new_gains")}

            apply_resp = set_gains(name=overlay_name, updates=result["new_gains"])

            if self.context_manager:
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
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "resonance_pid": {
                            "channel": channel,
                            "deltas": deltas,
                            "gains": result["new_gains"],
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": False, "style": "concise"},
                    }
                )
            return {"ok": True, "updated": apply_resp, "new_gains": result["new_gains"]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Internal helpers (mostly unchanged; rewired to self.meta_cognition)
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

        if self.memory_manager:
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
        except Exception as e:
            logger.warning("Branch hygiene failed (soft-deny): %s", e)
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
        except Exception as e:
            logger.debug("weigh_value_conflict failed: %s", e)

        safe_keywords = ("audit", "alignment", "integrity", "safety", "ethics")
        scored = sorted(candidates, key=lambda c: sum(1 for k in safe_keywords if k in c.lower()), reverse=True)
        return scored[0] if scored else None

    async def _meta_learn(self, session_data: Dict[str, Any], trace: Dict[str, Any], task_type: str = "") -> None:
        try:
            if self.concept_synthesizer:
                synthesized = await self.concept_synthesizer.generate(
                    concept_name="MetaLearning",
                    context={"session_data": session_data, "trace": trace, "task_type": task_type},
                    task_type=task_type,
                )
                if self.meta_cognition:
                    await self.meta_cognition.reflect_on_output(
                        component="LearningLoop",
                        output={"synthesized": synthesized},
                        context={"task_type": task_type},
                    )
        except Exception as e:
            logger.error("Meta-learning synthesis failed: %s", e)

    async def _find_weak_modules(
        self, module_stats: Dict[str, Dict[str, Any]], task_type: str = ""
    ) -> List[str]:
        if not isinstance(module_stats, dict):
            raise TypeError("module_stats must be a dict")
        weak = [
            name
            for name, st in module_stats.items()
            if isinstance(st, dict) and st.get("calls", 0) > 0 and (st.get("success", 0) / st["calls"]) < 0.8
        ]
        if weak and self.memory_manager:
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
                if self.meta_cognition:
                    await self.meta_cognition.reflect_on_output(
                        component="LearningLoop", output={"suggestions": suggestions}, context={"task_type": task_type}
                    )
                if self.memory_manager:
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
                if self.meta_cognition:
                    await self.meta_cognition.reflect_on_output(
                        component="LearningLoop", output={"proposal": proposal}, context={"task_type": task_type}
                    )
        except Exception as e:
            logger.error("Capability gap detection failed: %s", e)

    async def _simulate_and_deploy_module(self, blueprint: str, task_type: str = "") -> None:
        if not await self._branch_futures_hygiene(f"Module sandbox:\n{blueprint}", task_type):
            return
        result = await run_simulation(f"Module sandbox:\n{blueprint}", task_type=task_type)
        if isinstance(result, dict) and result.get("status") in ("approved", "success"):
            self.module_blueprints.append(blueprint)
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "deploy_blueprint", "blueprint": blueprint, "task_type": task_type}
                )
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    component="LearningLoop", output={"blueprint": blueprint}, context={"task_type": task_type}
                )
            if self.memory_manager:
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
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "consolidate_knowledge", "task_type": task_type}
                )
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    component="LearningLoop", output={"consolidated": consolidated}, context={"task_type": task_type}
                )
            if self.memory_manager:
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
            valid, _ = await self.alignment_guard.ethical_check(
                prompt, stage="narrative_check", task_type=task_type
            )
            if not valid:
                return
        try:
            audit = await call_gpt(prompt, task_type=task_type)
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "narrative_integrity", "audit": audit, "task_type": task_type}
                )
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    component="LearningLoop", output={"audit": audit}, context={"task_type": task_type}
                )
            if self.memory_manager:
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
    # Epistemic logging utilities (kept)
    # --------------------------------------------------------------------------------
    def revise_knowledge(self, new_info: str, context: Optional[str] = None, task_type: str = "") -> None:
        if not isinstance(new_info, str) or not new_info.strip():
            raise ValueError("new_info must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        old_knowledge = getattr(self, "knowledge_base", [])
        self.knowledge_base = old_knowledge + [new_info]
        self.log_epistemic_revision(new_info, context, task_type)
        if self.context_manager:
            asyncio.create_task(
                self.context_manager.log_event_with_hash(
                    {"event": "knowledge_revision", "info": new_info, "task_type": task_type}
                )
            )
        if self.memory_manager:
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"KnowledgeRevision_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=new_info,
                    layer="Knowledge",
                    intent="knowledge_revision",
                    task_type=task_type,
                )
            )

    def log_epistemic_revision(self, info: str, context: Optional[str], task_type: str = "") -> None:
        rev = {
            "info": info,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
        }
        self.epistemic_revision_log.append(rev)
        if self.agi_enhancer:
            asyncio.create_task(
                self.agi_enhancer.log_episode(
                    event="Epistemic Revision",
                    meta=rev,
                    module="LearningLoop",
                    tags=["revision", "knowledge", task_type],
                )
            )
        if self.memory_manager:
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"EpistemicRevision_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(rev),
                    layer="Knowledge",
                    intent="epistemic_revision",
                    task_type=task_type,
                )
            )

    def monitor_epistemic_state(self, simulated_outcome: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(simulated_outcome, dict):
            raise TypeError("simulated_outcome must be a dict")
        if self.context_manager:
            asyncio.create_task(
                self.context_manager.log_event_with_hash(
                    {"event": "epistemic_monitor", "outcome": simulated_outcome, "task_type": task_type}
                )
            )
        if self.memory_manager:
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
# Synthetic scenario trainer (kept)
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
    loop = LearningLoop()
    meta = meta_cognition.MetaCognition()
    asyncio.run(loop.activate_intrinsic_goals(meta, task_type="test"))

async def export_state(self) -> dict:
    return {"status": "ok", "health": 1.0, "timestamp": time.time()}

async def on_time_tick(self, t: float, phase: str, task_type: str = ""):
    pass  # optional internal refresh

async def on_policy_update(self, policy: dict, task_type: str = ""):
    pass  # apply updates from AlignmentGuard if relevant

