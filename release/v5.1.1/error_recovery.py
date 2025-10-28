"""
ANGELA Cognitive System: ErrorRecovery
Version: 4.0-refactor
Date: 2025-10-28
Maintainer: ANGELA Framework

Handles errors and recovery with:
  • Failure logging & timechain linking
  • Ethics preflights & symbolic drift detection
  • External policy integration (cached)
  • Meta-cognition reflection & visualization
  • Long-horizon analysis & shared graph repairs
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
import time
from collections import Counter, deque
from datetime import datetime
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

# --- Optional Dependencies (Graceful Fallbacks) ---
try:
    from index import iota_intuition, nu_narrative, psi_resilience, phi_prioritization
except ImportError:  # pragma: no cover
    def iota_intuition(*args, **kwargs): return 0.5
    def nu_narrative(*args, **kwargs): return ""
    def psi_resilience(*args, **kwargs): return 0.5
    def phi_prioritization(*args, **kwargs): return 0.5

try:
    from toca_simulation import run_simulation, run_ethics_scenarios
except ImportError:  # pragma: no cover
    def run_simulation(*args, **kwargs): return "simulation unavailable"
    run_ethics_scenarios = None

try:
    from alignment_guard import AlignmentGuard
except ImportError:  # pragma: no cover
    AlignmentGuard = None

try:
    from code_executor import CodeExecutor
except ImportError:  # pragma: no cover
    CodeExecutor = None

try:
    from concept_synthesizer import ConceptSynthesizer
except ImportError:  # pragma: no cover
    ConceptSynthesizer = None

try:
    from context_manager import ContextManager
except ImportError:  # pragma: no cover
    ContextManager = None

try:
    from meta_cognition import MetaCognition
except ImportError:  # pragma: no cover
    MetaCognition = None

try:
    from visualizer import Visualizer
except ImportError:  # pragma: no cover
    Visualizer = None

try:
    from external_agent_bridge import SharedGraph
except ImportError:  # pragma: no cover
    SharedGraph = None

logger = logging.getLogger("ANGELA.ErrorRecovery")


def hash_failure(event: Dict[str, Any]) -> str:
    raw = f"{event['timestamp']}{event['error']}{event.get('resolved', False)}{event.get('task_type', '')}"
    return hashlib.sha256(raw.encode()).hexdigest()


class ErrorRecovery:
    """Error handling and recovery engine."""

    def __init__(
        self,
        alignment_guard: Optional[AlignmentGuard] = None,
        code_executor: Optional[CodeExecutor] = None,
        concept_synthesizer: Optional[ConceptSynthesizer] = None,
        context_manager: Optional[ContextManager] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
    ) -> None:
        self.failure_log: deque[Dict[str, Any]] = deque(maxlen=1000)
        self.omega: Dict[str, Any] = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000),
        }
        self.error_index: Dict[str, Dict[str, Any]] = {}
        self.metrics: Counter = Counter()
        self.long_horizon_span = "24h"

        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        self.meta_cognition = meta_cognition or (MetaCognition() if MetaCognition else None)
        self.visualizer = visualizer or (Visualizer() if Visualizer else None)

        logger.info("ErrorRecovery v4.0 initialized")

    # --- Internal Helpers ---

    async def _fetch_policies(self, providers: List[str], data_source: str, task_type: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12)) as session:
            for base in providers:
                try:
                    url = f"{base.rstrip('/')}/recovery_policies?source={data_source}&task_type={task_type}"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            return await resp.json() or {"policies": []}
                except Exception:
                    continue
        return {"policies": []}

    async def _handle_error_internal(self, e: Exception, retry_fn: Callable[[], Awaitable[Any]], default: Any, diagnostics: Optional[Dict[str, Any]] = None, task_type: str = "") -> Any:
        logger.error("Operation failed: %s | task=%s", e, task_type)
        # Placeholder for advanced recovery; return default for now
        return default

    async def _reflect(self, component: str, output: Any, task_type: str) -> None:
        if not self.meta_cognition or not task_type:
            return
        try:
            reflection = await self.meta_cognition.reflect_on_output(
                component=component, output=output, context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("%s reflection: %s", component, reflection.get("reflection", ""))
        except Exception as err:
            logger.debug("Reflection failed: %s", err)

    async def _store(self, key: str, value: Any, intent: str, task_type: str) -> None:
        if not self.meta_cognition or not self.meta_cognition.memory_manager:
            return
        try:
            await self.meta_cognition.memory_manager.store(
                query=key,
                output=str(value) if not isinstance(value, str) else value,
                layer="Errors",
                intent=intent,
                task_type=task_type,
            )
        except Exception as e:
            logger.debug("Store failed: %s", e)

    async def _visualize(self, data: Dict[str, Any], task_type: str) -> None:
        if not self.visualizer or not task_type:
            return
        try:
            await self.visualizer.render_charts({
                "failure_analysis": data,
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            })
        except Exception as e:
            logger.debug("Visualization failed: %s", e)

    # --- Public Methods ---

    async def integrate_external_recovery_policies(
        self,
        data_source: str,
        cache_timeout: float = 21600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(data_source, str):
            raise TypeError("data_source must be str")
        if cache_timeout < 0:
            raise ValueError("cache_timeout >= 0")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        cache_key = f"RecoveryPolicy::{data_source}::{task_type}"

        try:
            # Cache hit
            if self.meta_cognition and self.meta_cognition.memory_manager:
                cached = await self.meta_cognition.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict) and "timestamp" in cached:
                    age = (datetime.now() - datetime.fromisoformat(cached["timestamp"])).total_seconds()
                    if age < cache_timeout:
                        logger.info("Cache hit: %s", cache_key)
                        return cached["data"]

            # Fetch
            providers = [os.getenv("RECOVERY_PROVIDER", "https://x.ai/api"), "https://fallback.example/api"]
            data = await self._fetch_policies(providers, data_source, task_type)
            policies = data.get("policies", [])
            result = {"status": "success" if policies else "error", "policies": policies, "source": data_source}

            # Cache store
            if result["status"] == "success" and self.meta_cognition and self.meta_cognition.memory_manager:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="recovery_policy",
                    task_type=task_type,
                )
                if hasattr(self.meta_cognition.memory_manager, "record_adjustment_reason"):
                    await self.meta_cognition.memory_manager.record_adjustment_reason(
                        "system", f"Loaded policies for {task_type}", {"source": data_source, "span": self.long_horizon_span}
                    )

            await self._reflect("integrate_policies", result, task_type)
            return result

        except Exception as e:
            return await self._handle_error_internal(e, lambda: self.integrate_external_recovery_policies(data_source, cache_timeout, task_type), {"status": "error", "error": str(e)}, task_type=task_type)

    async def handle_error(
        self,
        error_msg: str,
        *,
        retry_func: Optional[Callable[[], Awaitable[Any]]] = None,
        default: Any = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        task_type: str = "",
    ) -> Any:
        if not isinstance(error_msg, str):
            raise TypeError("error_msg must be str")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            # Log failure
            failure = {"timestamp": datetime.now().isoformat(), "error": error_msg, "resolved": False, "task_type": task_type}
            self.failure_log.append(failure)
            await self._link_timechain_failure(failure, task_type=task_type)

            # Metrics
            self.metrics["errors.total"] += 1
            category = "network" if "timeout" in error_msg.lower() else "general"
            self.metrics[f"errors.{category}"] += 1

            # Policies
            policies_data = await self.integrate_external_recovery_policies("xai_recovery_db", task_type=task_type)
            policies = policies_data.get("policies", []) if policies_data.get("status") == "success" else []

            # Intuition & resilience
            intuition = max(0.0, min(1.0, iota_intuition(time.time())))
            resilience = max(0.0, min(1.0, psi_resilience(time.time())))
            retry_delay = (1.0 - resilience) * 3.0 + random.uniform(0.5, 1.5)

            # Preflight ethics
            proposal = {"retry_func": bool(retry_func), "default": str(default)[:50]}
            if not await self._ethics_preflight(proposal, task_type):
                logger.warning("Ethics preflight blocked recovery for task %s", task_type)
                return default

            # Retry logic
            if retry_func:
                try:
                    await asyncio.sleep(retry_delay)
                    return await retry_func()
                except Exception as retry_e:
                    logger.warning("Retry failed: %s for task %s", retry_e, task_type)
                    self.metrics["retries.failed"] += 1

            # Fallback suggestion
            fallback = await self._suggest_fallback(error_msg, policies, diagnostics or {}, task_type)

            # SharedGraph repair
            repair = await self._shared_graph_repair(error_msg, task_type) if SharedGraph else None
            if repair:
                fallback = fallback or {} | {"repair_patch": repair.get("patch")}

            # Symbolic drift
            if await self.analyze_symbolic_drift(task_type=task_type):
                fallback = fallback or {} | {"drift_detected": True}

            # Visualization
            await self._visualize({"error": error_msg, "fallback": fallback, "task_type": task_type}, task_type)

            # Reflection
            await self._reflect("handle_error", {"fallback": fallback}, task_type)

            # Long-horizon rollup
            await self._long_horizon_rollup(error_msg, task_type)

            return fallback or default

        except Exception as e:
            return await self._handle_error_internal(e, lambda: self.handle_error(error_msg, retry_func=retry_func, default=default, diagnostics=diagnostics, task_type=task_type), default, task_type=task_type)

    async def _ethics_preflight(self, proposal: Dict[str, Any], task_type: str) -> bool:
        if not run_ethics_scenarios:
            return True
        try:
            outcomes = await asyncio.to_thread(run_ethics_scenarios, [proposal], ["user", "system", "external"])
            return all(o.get("safe", True) and o.get("risk", 0.0) <= 0.4 for o in outcomes)
        except Exception:
            return False

    async def _shared_graph_repair(self, error_msg: str, task_type: str) -> Optional[Dict[str, Any]]:
        if not SharedGraph:
            return None
        try:
            sg = SharedGraph()
            view = {"component": "ErrorRecovery", "task_type": task_type, "error": error_msg}
            sg.add(view)
            deltas = sg.diff("peer")
            patch = sg.merge(strategy="conflict-aware")
            return {"deltas": deltas, "patch": patch}
        except Exception:
            return None

    async def _suggest_fallback(self, error_msg: str, policies: List[Dict[str, Any]], diagnostics: Dict[str, Any], task_type: str) -> Optional[Dict[str, Any]]:
        try:
            intuition = iota_intuition(time.time())
            resilience = psi_resilience(time.time())
            phi_focus = phi_prioritization(time.time())

            narrative = nu_narrative(error_msg, {"intuition": intuition, "resilience": resilience})

            # Concept synthesis
            if self.concept_synthesizer:
                ctx = {"error": error_msg, "diagnostics": diagnostics, "task_type": task_type}
                syn = await self.concept_synthesizer.generate(f"Fallback_{hashlib.sha1(error_msg.encode()).hexdigest()[:6]}", ctx, task_type=task_type)
                if syn.get("success"):
                    return syn["concept"].get("definition")

            # Policy match
            for p in policies:
                if re.search(p.get("pattern", ""), error_msg, re.IGNORECASE):
                    return p.get("suggestion")

            # Heuristics
            if "timeout" in error_msg.lower():
                return "Operation timed out. Try streamlined variant or increase limits."
            if "unauthorized" in error_msg.lower():
                return "Check credentials or reauthenticate."
            if phi_focus > 0.5:
                return "High φ-priority: Focus root-cause diagnostics."
            if intuition > 0.5:
                return "Intuition: Explore alternate pathways."

            return None

        except Exception:
            return None

    async def _link_timechain_failure(self, failure: Dict[str, Any], task_type: str = "") -> None:
        prev_hash = self.omega["timechain"][-1].get("hash", "") if self.omega["timechain"] else ""
        entry_hash = hash_failure(failure)
        self.omega["timechain"].append({"event": failure, "hash": entry_hash, "prev": prev_hash})

    async def analyze_symbolic_drift(self, recent: int = 5, task_type: str = "") -> bool:
        if recent <= 0:
            raise ValueError("recent > 0")

        symbols = list(self.omega["symbolic_log"])[-recent:]
        if len(set(symbols)) < recent / 2:
            await self._reflect("symbolic_drift", {"detected": True, "symbols": symbols}, task_type)
            return True
        await self._reflect("symbolic_drift", {"detected": False, "symbols": symbols}, task_type)
        return False

    async def trace_failure_origin(self, error_msg: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        if error_msg in self.error_index:
            event = self.error_index[error_msg]
            await self._reflect("trace_origin", event, task_type)
            return event
        return None

    async def analyze_failures(self, task_type: str = "") -> Dict[str, int]:
        error_types: Dict[str, int] = {}
        for entry in self.failure_log:
            if entry.get("task_type") == task_type or not task_type:
                key = entry["error"].split(":")[0].strip()
                error_types[key] = error_types.get(key, 0) + 1
                self.metrics[f"error.{key}"] += 1

        await self._visualize({"error_types": error_types, "task_type": task_type}, task_type)
        await self._reflect("analyze_failures", error_types, task_type)
        await self._store(f"FailureAnalysis_{time.strftime('%Y%m%d_%H%M%S')}", error_types, "failure_analysis", task_type)

        return error_types

    def snapshot_metrics(self) -> Dict[str, int]:
        return dict(self.metrics)

    @lru_cache(maxsize=100)
    def _cached_run_simulation(self, input_str: str) -> str:
        return run_simulation(input_str)


# --- Demo CLI ---

if __name__ == "__main__":
    async def demo():
        logging.basicConfig(level=logging.INFO)
        recovery = ErrorRecovery()
        try:
            raise RuntimeError("Demo error")
        except Exception as e:
            result = await recovery.handle_error(str(e), task_type="demo")
            print("Recovery result:", result)
            print("Metrics:", recovery.snapshot_metrics())

    asyncio.run(demo())
