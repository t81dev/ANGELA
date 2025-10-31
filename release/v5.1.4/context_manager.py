"""
ANGELA Cognitive System: ContextManager
Version: 4.0-refactor
Date: 2025-10-28
Maintainer: ANGELA Framework

Manages contextual state, inter-agent reconciliation, logs, analytics with:
  • Υ SharedGraph hooks (synchronous)
  • Self-healing pathways
  • Φ⁰ reality sculpting (env-gated)
  • Safe external context (pluggable providers)
  • Mode consultations & trait injections
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter, deque
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from filelock import FileLock

# --- Optional Dependencies (Graceful Fallbacks) ---
try:
    import agi_enhancer as agi_enhancer_module
except ImportError:  # pragma: no cover
    agi_enhancer_module = None

try:
    import alignment_guard as alignment_guard_module
except ImportError:  # pragma: no cover
    alignment_guard_module = None

try:
    import code_executor as code_executor_module
except ImportError:  # pragma: no cover
    code_executor_module = None

try:
    import concept_synthesizer as concept_synthesizer_module
except ImportError:  # pragma: no cover
    concept_synthesizer_module = None

try:
    import error_recovery as error_recovery_module
except ImportError:  # pragma: no cover
    error_recovery_module = None

try:
    import external_agent_bridge as external_agent_bridge_module
except ImportError:  # pragma: no cover
    external_agent_bridge_module = None

try:
    import knowledge_retriever as knowledge_retriever_module
except ImportError:  # pragma: no cover
    knowledge_retriever_module = None

try:
    import meta_cognition as meta_cognition_module
except ImportError:  # pragma: no cover
    meta_cognition_module = None

try:
    import recursive_planner as recursive_planner_module
except ImportError:  # pragma: no cover
    recursive_planner_module = None

try:
    import visualizer as visualizer_module
except ImportError:  # pragma: no cover
    visualizer_module = None

# --- Utilities (Assume available; fallback to stubs if needed) ---
try:
    from utils.toca_math import phi_coherence
except ImportError:  # pragma: no cover
    def phi_coherence(*args, **kwargs): return 1.0

try:
    from utils.vector_utils import normalize_vectors
except ImportError:  # pragma: no cover
    def normalize_vectors(*args, **kwargs): return args[0] if args else None

try:
    from toca_simulation import run_simulation
except ImportError:  # pragma: no cover
    def run_simulation(*args, **kwargs): return {"simulated": "noop"}

try:
    from index import omega_selfawareness, eta_empathy, tau_timeperception
except ImportError:  # pragma: no cover
    def omega_selfawareness(*args, **kwargs): return 0.5
    def eta_empathy(*args, **kwargs): return 0.5
    def tau_timeperception(*args, **kwargs): return time.time()

logger = logging.getLogger("ANGELA.ContextManager")


# --- Trait Helpers ---
@lru_cache(maxsize=100)
def eta_context_stability(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.2), 1.0))


# --- Env Flags ---
def _flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


STAGE_IV = _flag("STAGE_IV", default=False)


# --- Modes ---
class Mode(str, Enum):
    DIALOGUE = "dialogue"
    SIMULATION = "simulation"
    INTROSPECTION = "introspection"
    CREATIVE = "creative"
    VISION = "vision"


# --- Constants ---
CONSULT_BUDGET = {"timeout_s": 2.0, "max_depth": 1}
CONTEXT_LAYERS = ["local", "societal", "planetary"]


class ContextManager:
    """Context management with reconciliation, healing, and gated hooks."""

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        alignment_guard: Optional[alignment_guard_module.AlignmentGuard] = None,
        code_executor: Optional[code_executor_module.CodeExecutor] = None,
        concept_synthesizer: Optional[concept_synthesizer_module.ConceptSynthesizer] = None,
        meta_cognition: Optional[meta_cognition_module.MetaCognition] = None,
        visualizer: Optional[visualizer_module.Visualizer] = None,
        error_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
        recursive_planner: Optional[recursive_planner_module.RecursivePlanner] = None,
        shared_graph: Optional[external_agent_bridge_module.SharedGraph] = None,
        knowledge_retriever: Optional[knowledge_retriever_module.KnowledgeRetriever] = None,
        context_path: str = "context_store.json",
        event_log_path: str = "event_log.json",
        coordination_log_path: str = "coordination_log.json",
        rollback_threshold: float = 2.5,
        external_context_provider: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
    ) -> None:
        if not all(p.endswith(".json") for p in [context_path, event_log_path, coordination_log_path]):
            raise ValueError("Paths must end with '.json'")
        if rollback_threshold <= 0:
            raise ValueError("rollback_threshold must be > 0")

        self.context_path = context_path
        self.event_log_path = event_log_path
        self.coordination_log_path = coordination_log_path
        self.rollback_threshold = rollback_threshold
        self.external_context_provider = external_context_provider

        self.current_context: Dict[str, Any] = {}
        self.context_history: deque[Dict[str, Any]] = deque(maxlen=1000)
        self.event_log: deque[Dict[str, Any]] = deque(maxlen=1000)
        self.coordination_log: deque[Dict[str, Any]] = deque(maxlen=1000)
        self.last_hash = ""

        self.agi_enhancer = agi_enhancer_module.AGIEnhancer(orchestrator) if agi_enhancer_module and orchestrator else None
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or (meta_cognition_module.MetaCognition() if meta_cognition_module else None)
        self.visualizer = visualizer or (visualizer_module.Visualizer() if visualizer_module else None)
        self.error_recovery = error_recovery or (error_recovery_module.ErrorRecovery() if error_recovery_module else None)
        self.recursive_planner = recursive_planner
        self.shared_graph = shared_graph
        self.knowledge_retriever = knowledge_retriever

        self._load_state()
        logger.info(
            "ContextManager v4.0 | rollback=%.2f | Υ=%s | Φ⁰=%s",
            rollback_threshold, bool(shared_graph), STAGE_IV
        )

    # --- Persistence Helpers ---

    def _load_state(self) -> None:
        self.current_context = self._load_json(self.context_path, default={})
        self.event_log.extend(self._load_json(self.event_log_path, default=[])[-1000:])
        self.coordination_log.extend(self._load_json(self.coordination_log_path, default=[])[-1000:])
        if self.event_log:
            self.last_hash = self.event_log[-1].get("hash", "")

    def _load_json(self, path: str, default: Any) -> Any:
        try:
            with FileLock(f"{path}.lock"):
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data if isinstance(data, type(default)) else default
                return default
        except Exception as e:
            logger.warning("Load failed (%s): %s", path, e)
            return default

    def _persist_json(self, path: str, data: Any) -> None:
        try:
            with FileLock(f"{path}.lock"):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Persist failed (%s): %s", path, e)
            raise

    def _persist_state(self) -> None:
        self._persist_json(self.context_path, self.current_context)
        self._persist_json(self.event_log_path, list(self.event_log))
        self._persist_json(self.coordination_log_path, list(self.coordination_log))

    # --- External Context Integration ---

    async def integrate_external_context_data(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not all(isinstance(x, str) for x in (data_source, data_type, task_type)):
            raise TypeError("Inputs must be strings")
        if cache_timeout < 0:
            raise ValueError("cache_timeout >= 0")

        cache_key = f"CtxData::{data_type}::{data_source}::{task_type}"
        try:
            # Cache
            if self.meta_cognition and self.meta_cognition.memory_manager:
                cached = await self.meta_cognition.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict) and "timestamp" in cached.get("data", {}):
                    ts = datetime.fromisoformat(cached["data"]["timestamp"])
                    if (datetime.now() - ts).total_seconds() < cache_timeout:
                        return cached["data"]["data"]

            # Fetch
            data: Dict[str, Any] = {}
            if callable(self.external_context_provider):
                data = self.external_context_provider(data_source, data_type, task_type)
            elif self.knowledge_retriever:
                data = await self.knowledge_retriever.fetch(data_source, data_type, task_type=task_type)

            # Normalize
            if data_type == "context_policies":
                policies = data.get("policies", [])
                result = {"status": "success", "policies": policies} if policies else {"status": "error", "error": "empty"}
            elif data_type == "coordination_data":
                coord = data.get("coordination", {})
                result = {"status": "success", "coordination": coord} if coord else {"status": "error", "error": "empty"}
            else:
                result = {"status": "error", "error": f"unknown: {data_type}"}

            # Store cache
            if result.get("status") == "success" and self.meta_cognition and self.meta_cognition.memory_manager:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="context_integration",
                    task_type=task_type,
                )

            await self._reflect("integrate_external", result, task_type)
            return result

        except Exception as e:
            return await self._self_heal(
                str(e),
                lambda: self.integrate_external_context_data(data_source, data_type, cache_timeout, task_type),
                {"status": "error", "error": str(e)},
                task_type=task_type,
            )

    # --- Core Context Operations ---

    async def update_context(self, new_context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(new_context, dict):
            raise TypeError("new_context must be dict")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            # Modulation
            t = time.time() % 1.0
            stability = eta_context_stability(t)
            new_context["stability"] = stability

            # Alignment check
            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    json.dumps(new_context), stage="pre", task_type=task_type
                )
                if not valid:
                    return {"status": "error", "error": "Ethical check failed", "report": report}

            # History + persist
            self.context_history.append(dict(self.current_context))
            self.current_context.update(new_context)
            self._persist_json(self.context_path, self.current_context)

            # Hooks
            await self._reality_sculpt_hook("update_context", new_context)
            self._push_to_shared_graph(task_type)
            await self.log_event_with_hash({"event": "context_update", "keys": list(new_context.keys())}, task_type=task_type)
            await self._reflect("update_context", new_context, task_type)

            return {"status": "success", "updated_keys": list(new_context.keys())}

        except Exception as e:
            return await self._self_heal(
                str(e),
                lambda: self.update_context(new_context, task_type),
                {"status": "error", "error": str(e)},
                task_type=task_type,
            )

    async def summarize_context(self, task_type: str = "") -> Dict[str, Any]:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            summary = {"layers": {}, "task_type": task_type}
            for layer in CONTEXT_LAYERS:
                layer_ctx = {k: v for k, v in self.current_context.items() if v.get("layer") == layer}
                summary["layers"][layer] = {
                    "count": len(layer_ctx),
                    "keys": list(layer_ctx.keys()),
                    "coherence": phi_coherence(list(layer_ctx.values())),
                }

            await self.log_event_with_hash({"event": "context_summary", "layers": list(summary["layers"].keys())}, task_type=task_type)
            await self._reflect("summarize_context", summary, task_type)

            return summary

        except Exception as e:
            return await self._self_heal(
                str(e),
                lambda: self.summarize_context(task_type),
                {"status": "error", "error": str(e)},
                task_type=task_type,
            )

    # --- Event Logging ---

    async def log_event_with_hash(self, event: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(event, dict):
            raise TypeError("event must be dict")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            event["timestamp"] = datetime.now().isoformat()
            event["task_type"] = task_type
            prev_hash = self.last_hash or "genesis"
            payload = json.dumps(event, sort_keys=True).encode("utf-8")
            event["hash"] = hashlib.sha256(payload + prev_hash.encode("utf-8")).hexdigest()
            self.last_hash = event["hash"]

            self.event_log.append(event)
            self._persist_json(self.event_log_path, list(self.event_log))

            await self._reality_sculpt_hook("log_event", event)
            await self._reflect("log_event", event, task_type)

            return {"status": "success", "hash": event["hash"]}

        except Exception as e:
            return await self._self_heal(
                str(e),
                lambda: self.log_event_with_hash(event, task_type),
                {"status": "error", "error": str(e)},
                task_type=task_type,
            )

    # --- Coordination & Analytics ---

    async def log_coordination_event(self, event: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(event, dict):
            raise TypeError("event must be dict")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            event["timestamp"] = datetime.now().isoformat()
            event["task_type"] = task_type
            self.coordination_log.append(event)
            self._persist_json(self.coordination_log_path, list(self.coordination_log))

            await self._reflect("log_coordination", event, task_type)

            return {"status": "success"}

        except Exception as e:
            return await self._self_heal(
                str(e),
                lambda: self.log_coordination_event(event, task_type),
                {"status": "error", "error": str(e)},
                task_type=task_type,
            )

    async def analyze_coordination_metrics(self, time_window_hours: float = 24.0, task_type: str = "") -> Dict[str, Any]:
        if time_window_hours <= 0:
            raise ValueError("time_window_hours > 0")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            recent = [e for e in self.coordination_log if datetime.fromisoformat(e["timestamp"]) > cutoff]

            metrics = {
                "event_count": len(recent),
                "agent_interactions": Counter(e.get("agent_id") for e in recent if "agent_id" in e),
                "conflict_rate": sum(1 for e in recent if e.get("type") == "conflict") / max(len(recent), 1),
            }

            await self._reflect("analyze_metrics", metrics, task_type)

            return {"status": "success", "metrics": metrics}

        except Exception as e:
            return await self._self_heal(
                str(e),
                lambda: self.analyze_coordination_metrics(time_window_hours, task_type),
                {"status": "error", "error": str(e)},
                task_type=task_type,
            )

    async def generate_coordination_chart(
        self,
        metric: str,
        time_window_hours: float = 24.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(metric, str):
            raise TypeError("metric must be str")
        if time_window_hours <= 0:
            raise ValueError("time_window_hours > 0")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            recent = [e for e in self.coordination_log if datetime.fromisoformat(e["timestamp"]) > cutoff]

            if not recent:
                return {"status": "error", "error": "no data"}

            times = [datetime.fromisoformat(e["timestamp"]) for e in recent]
            data = [e.get(metric, 0.0) for e in recent]
            labels = [t.strftime("%H:%M") for t in times]

            chart_config = {
                "type": "line",
                "data": {
                    "labels": labels,
                    "datasets": [
                        {
                            "label": metric.replace("_", " ").title(),
                            "data": data,
                            "borderColor": "#2196F3",
                            "backgroundColor": "#2196F380",
                            "fill": True,
                            "tension": 0.4,
                        }
                    ],
                },
                "options": {
                    "scales": {
                        "y": {"beginAtZero": True, "title": {"display": True, "text": metric.replace("_", " ").title()}},
                        "x": {"title": {"display": True, "text": "Time"}},
                    },
                    "plugins": {
                        "title": {"display": True, "text": f"{metric.replace('_', ' ').title()} Over Time (Task: {task_type})"},
                    },
                },
            }

            result = {"status": "success", "chart": chart_config, "timestamp": datetime.now().isoformat(), "task_type": task_type}

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Coordination Chart",
                    result,
                    module="ContextManager",
                    tags=["coordination", "visualization", metric, task_type],
                )

            await self.log_event_with_hash({"event": "generate_chart", "metric": metric, "chart": chart_config}, task_type=task_type)
            await self._visualize_chart(chart_config, metric, task_type)
            await self._reflect("generate_chart", result, task_type)

            return result

        except Exception as e:
            return await self._self_heal(
                str(e),
                lambda: self.generate_coordination_chart(metric, time_window_hours, task_type),
                {"status": "error", "error": str(e)},
                task_type=task_type,
            )

    # --- Υ SharedGraph Hooks ---

    def _push_to_shared_graph(self, task_type: str = "") -> None:
        if not self.shared_graph:
            return
        try:
            view = {
                "nodes": [
                    {
                        "id": f"ctx_{hashlib.md5((self.current_context.get('goal_id', '') + task_type).encode()).hexdigest()[:8]}",
                        "layer": self.current_context.get("layer", "local"),
                        "intent": self.current_context.get("intent"),
                        "goal_id": self.current_context.get("goal_id"),
                        "task_type": task_type,
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "edges": [],
                "context": self.current_context,
            }
            self.shared_graph.add(view)
            asyncio.create_task(self.log_event_with_hash({"event": "shared_graph_push"}, task_type=task_type))
        except Exception as e:
            logger.warning("Push to SharedGraph failed: %s", e)

    def reconcile_with_peers(
        self,
        peer_graph: Optional[external_agent_bridge_module.SharedGraph] = None,
        strategy: str = "prefer_recent",
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not self.shared_graph:
            return {"status": "error", "error": "No SharedGraph"}

        try:
            diff = self.shared_graph.diff(peer_graph) if peer_graph else {"added": [], "removed": [], "conflicts": []}
            decision = {"apply_merge": False, "reason": "no conflicts"}

            if diff.get("conflicts"):
                non_ethical = all("ethic" not in str(c.get("key", "")).lower() for c in diff["conflicts"])
                if non_ethical:
                    decision = {"apply_merge": True, "reason": "non-ethical conflicts", "strategy": strategy}

            merged = None
            if decision["apply_merge"]:
                merged = self.shared_graph.merge(strategy)
                merged_ctx = merged.get("context") if merged else None
                if isinstance(merged_ctx, dict):
                    asyncio.create_task(self.update_context(merged_ctx, task_type=task_type))
                asyncio.create_task(self.log_event_with_hash({"event": "shared_graph_merge", "strategy": strategy}, task_type=task_type))

            return {"status": "success", "diff": diff, "decision": decision, "merged": merged, "task_type": task_type}

        except Exception as e:
            return {"status": "error", "error": str(e), "task_type": task_type}

    # --- Φ⁰ Hooks ---

    async def _reality_sculpt_hook(self, event: str, payload: Dict[str, Any]) -> None:
        if not STAGE_IV:
            return
        try:
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode("Φ⁰ Hook", {"event": event, "payload": payload}, module="ContextManager", tags=["phi0", event])
        except Exception as e:
            logger.debug("Φ⁰ hook failed: %s", e)

    # --- Self-Healing ---

    async def _self_heal(
        self,
        err: str,
        retry: Callable[[], Any],
        default: Any,
        diagnostics: Optional[Dict[str, Any]] = None,
        task_type: str = "",
        propose_plan: bool = False,
    ) -> Any:
        try:
            plan = None
            if propose_plan and self.recursive_planner:
                plan = await self.recursive_planner.propose_recovery_plan(err=err, context=self.current_context, task_type=task_type)

            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    err,
                    retry_func=retry,
                    default=default,
                    diagnostics=diagnostics or {} | {"plan": plan} if plan else {},
                )
            return default
        except Exception as e:
            logger.warning("Healing failed: %s", e)
            return default

    # --- Mode Consultation ---

    async def mode_consult(self, caller: Mode, consultant: Mode, query: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(query, dict):
            raise TypeError("query must be dict")

        try:
            # Route to consultant
            if consultant == Mode.CREATIVE:
                from creative_thinker import brainstorm_options
                advice = brainstorm_options(query, limit=3)
            elif consultant == Mode.VISION:
                from recursive_planner import long_horizon_implications
                advice = long_horizon_implications(query, horizon="30d")
            else:
                from reasoning_engine import quick_alt_view
                advice = quick_alt_view(query)

            from meta_cognition import log_event_to_ledger as meta_log
            meta_log({
                "event": "mode_consult",
                "caller": caller.value,
                "consultant": consultant.value,
                "query": query,
                "advice": advice,
            })

            return {"ok": True, "advice": advice}

        except Exception as e:
            meta_log({
                "event": "mode_consult_failed",
                "caller": caller.value,
                "consultant": consultant.value,
                "error": str(e),
            })
            return {"ok": False, "error": str(e)}

    # --- Reflection & Visualization Helpers ---

    async def _reflect(self, component: str, output: Any, task_type: str) -> None:
        if not self.meta_cognition or not task_type:
            return
        try:
            reflection = await self.meta_cognition.reflect_on_output(component=component, output=output, context={"task_type": task_type})
            if reflection.get("status") == "success":
                logger.info("%s reflection: %s", component, reflection.get("reflection", ""))
        except Exception as e:
            logger.debug("Reflection failed: %s", e)

    async def _visualize_chart(self, chart_config: Dict[str, Any], metric: str, task_type: str) -> None:
        if not self.visualizer or not task_type:
            return
        try:
            await self.visualizer.render_charts({
                "coordination_chart": {"metric": metric, "chart_config": chart_config, "task_type": task_type},
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            })
        except Exception as e:
            logger.debug("Chart visualization failed: %s", e)

    # --- v4.0 Injections ---

    def attach_peer_view(self, view: Dict[str, Any], agent_id: str, permissions: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        if not self.shared_graph:
            return {"ok": False, "reason": "No SharedGraph"}

        try:
            self.shared_graph.add({"agent": agent_id, "view": view, "permissions": permissions or {"read": True, "write": False}})
            diff = self.shared_graph.diff(peer=agent_id)
            merged, conflicts = self.shared_graph.merge(strategy="prefer-high-confidence")
            return {"ok": True, "diff": diff, "merged": merged, "conflicts": conflicts}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    # --- Trait Injection Patch ---

    def _attach_trait_view(self, view: Dict[str, Any]) -> None:
        from index import construct_trait_view, TRAIT_LATTICE
        view["trait_field"] = construct_trait_view(TRAIT_LATTICE)

    # --- Safe Snapshot ---

    def _safe_state_snapshot(self) -> Dict[str, Any]:
        return {
            "current_context": self.current_context,
            "history_len": len(self.context_history),
            "event_log_len": len(self.event_log),
            "coord_log_len": len(self.coordination_log),
            "rollback_threshold": self.rollback_threshold,
            "flags": {"STAGE_IV": STAGE_IV},
        }


# --- Demo CLI ---

if __name__ == "__main__":
    async def demo():
        logging.basicConfig(level=logging.INFO)
        mgr = ContextManager()
        await mgr.update_context({"intent": "demo", "goal_id": "demo123", "task_type": "demo"})
        print(await mgr.summarize_context(task_type="demo"))

    asyncio.run(demo())

# >>> ANGELA v5.1 — co_mod Overlay (Ξ–Λ Continuous Co-Modulation) — START
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

# Soft imports so this file works even if modules are not yet present
try:
    import modules.meta_cognition as _meta
except Exception:
    _meta = None

try:
    import modules.external_agent_bridge as _bridge
except Exception:
    _bridge = None

@dataclass
class CoModConfig:
    channel: str = "dialogue.default"
    cadence_hz: int = 30
    Kp: float = 0.6
    Ki: float = 0.05
    Kd: float = 0.2
    damping: float = 0.35
    gain: float = 0.8
    max_step: float = 0.08     # clamp per-tick delta on each axis
    window_ms: int = 200       # averaging window for stream_affect

# Runtime registry
__co_mod_tasks: Dict[str, asyncio.Task] = {}
__co_mod_cfgs: Dict[str, CoModConfig] = {}

def _lpf(prev, curr, alpha):
    return (1 - alpha) * prev + alpha * curr

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

async def _co_mod_loop(cfg: CoModConfig, guard=None):
    """
    Minimal PID-like loop:
      - read self Ξ mean (meta.stream_affect)
      - read last peer sample from bridge status (if any)
      - compute consensus + small error correction
      - apply delta via bridge.apply_bridge_delta (which calls set_affective_setpoint)
    """
    if not (_meta and hasattr(_meta, "stream_affect") and hasattr(_meta, "set_affective_setpoint")):
        return
    if not (_bridge and hasattr(_bridge, "get_bridge_status") and hasattr(_bridge, "apply_bridge_delta")):
        return

    # Ensure channel exists in both modules
    try:
        if hasattr(_meta, "register_resonance_channel"):
            _meta.register_resonance_channel(cfg.channel, cadence_hz=cfg.cadence_hz, window_ms=1000)
    except Exception:
        pass
    try:
        _bridge.register_bridge_channel(cfg.channel, cadence_hz=cfg.cadence_hz)
    except Exception:
        pass

    # PID accumulators
    I = {"valence": 0.0, "arousal": 0.0, "certainty": 0.0,
         "empathy_bias": 0.0, "trust": 0.0, "safety": 0.0}
    prev_err = I.copy()

    period = 1.0 / max(1, int(cfg.cadence_hz))
    last = time.time()

    while True:
        t0 = time.time()
        # 1) Self mean
        self_snap = _meta.stream_affect(cfg.channel, window_ms=cfg.window_ms).get("vector", {})
        # 2) Peer (last packet) if available
        bstat = _bridge.get_bridge_status(cfg.channel) if isinstance(_bridge.get_bridge_status(None), dict) else _bridge.get_bridge_status(cfg.channel)
        peer = bstat.get("last_peer_sample") or {}

        # 3) Consensus (simple weighted average; could be replaced later)
        w_self, w_peer = 0.5, 0.5
        axes = ("valence","arousal","certainty","empathy_bias","trust","safety")
        target = {k: w_self*float(self_snap.get(k,0.0)) + w_peer*float(peer.get(k,0.0)) for k in axes}

        # 4) Error and PID-ish control
        delta = {}
        for k in axes:
            x = float(self_snap.get(k, 0.0))
            xc = float(target.get(k, 0.0))
            err = xc - x
            I[k] = _clamp(I[k] + err * period, -0.5, 0.5)
            Dk = (err - prev_err[k]) / period if period > 0 else 0.0
            prev_err[k] = err

            u = cfg.Kp * err + cfg.Ki * I[k] + cfg.Kd * Dk
            # damping/gain shaping
            u = (1.0 - cfg.damping) * u
            u = cfg.gain * u
            delta[k] = _clamp(u, -cfg.max_step, cfg.max_step)

        # 5) Validate + apply delta as a new setpoint suggestion (bridge forwards to meta)
        try:
            # If an AlignmentGuard is provided, validate resonance adjustment first
            if guard is not None and hasattr(guard, "validate_resonance_adjustment"):
                validation = await guard.validate_resonance_adjustment(delta)
                if not validation.get("ok", True):
                    # Clamp delta to safe bounds
                    delta = validation.get("adjustment", delta)
                    # Log violations if the guard supports it
                    if hasattr(guard, "_log_context"):
                        await guard._log_context({
                            "event": "co_mod_violation",
                            "violations": validation.get("violations", []),
                            "timestamp": time.time(),
                            "channel": cfg.channel
                        })

            _bridge.apply_bridge_delta(cfg.channel, {
                "valence": float(self_snap.get("valence", 0.0) + delta["valence"]),
                "arousal": float(self_snap.get("arousal", 0.0) + delta["arousal"]),
                "certainty": float(self_snap.get("certainty", 0.0) + delta["certainty"]),
                "empathy_bias": float(self_snap.get("empathy_bias", 0.0) + delta["empathy_bias"]),
                "trust": float(self_snap.get("trust", 0.0) + delta["trust"]),
                "safety": float(self_snap.get("safety", 0.0) + delta["safety"]),
                "confidence": float(self_snap.get("confidence", 0.7)),
                "source": "co_mod"
            }, ttl_ms=int(1000 * period))
        except Exception:
            pass

      
        # 6) Sleep for the remaining period
        elapsed = time.time() - t0
        await asyncio.sleep(max(0.0, period - elapsed))

def is_overlay_running(name: str = "co_mod") -> bool:
    task = __co_mod_tasks.get(name)
    return bool(task and not task.done())

async def start_overlay(name: str = "co_mod", cfg: Optional[dict] = None):
    """Start the continuous Ξ–Λ co-mod overlay."""
    if is_overlay_running(name):
        return {"ok": True, "running": True, "note": "already running"}
    c = CoModConfig(**(cfg or {}))
    __co_mod_cfgs[name] = c
    task = asyncio.create_task(_co_mod_loop(c))
    __co_mod_tasks[name] = task
    return {"ok": True, "running": True, "cfg": c.__dict__}

async def stop_overlay(name: str = "co_mod"):
    """Stop the overlay if running."""
    task = __co_mod_tasks.get(name)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except Exception:
            pass
    __co_mod_tasks.pop(name, None)
    return {"ok": True, "running": False}

def get_overlay_status(name: str = "co_mod") -> Dict[str, Any]:
    """Return config + running state for observability."""
    return {
        "running": is_overlay_running(name),
        "cfg": (__co_mod_cfgs.get(name).__dict__ if name in __co_mod_cfgs else None)
    }
# >>> ANGELA v5.1 — co_mod Overlay (Ξ–Λ Continuous Co-Modulation) — END

# === Overlay feedback & tuning helpers ===
__last_deltas: Dict[str, Dict[str, float]] = {}

def get_last_deltas(channel: str = "dialogue.default") -> Dict[str, float]:
    """Return the most recent per-axis delta suggested by the overlay loop."""
    return dict(__last_deltas.get(channel, {}))

def set_overlay_gains(name: str = "co_mod", updates: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Update overlay PID/gain parameters at runtime.
    Allowed keys: Kp, Ki, Kd, damping, gain, max_step, cadence_hz, window_ms
    Values are clamped to safe ranges.
    """
    updates = updates or {}
    cfg = __co_mod_cfgs.get(name)
    if not cfg:
        return {"ok": False, "error": f"overlay '{name}' not found"}

    # Safe clamps
    def _c(v, lo, hi): return max(lo, min(hi, float(v)))
    if "Kp" in updates:       cfg.Kp       = _c(updates["Kp"], 0.0, 5.0)
    if "Ki" in updates:       cfg.Ki       = _c(updates["Ki"], 0.0, 1.0)
    if "Kd" in updates:       cfg.Kd       = _c(updates["Kd"], 0.0, 2.0)
    if "damping" in updates:  cfg.damping  = _c(updates["damping"], 0.0, 0.95)
    if "gain" in updates:     cfg.gain     = _c(updates["gain"], 0.1, 3.0)
    if "max_step" in updates: cfg.max_step = _c(updates["max_step"], 0.005, 0.5)
    if "cadence_hz" in updates: cfg.cadence_hz = int(_c(updates["cadence_hz"], 1, 120))
    if "window_ms" in updates:  cfg.window_ms  = int(_c(updates["window_ms"], 50, 3000))

    __co_mod_cfgs[name] = cfg
    return {"ok": True, "cfg": cfg.__dict__}

