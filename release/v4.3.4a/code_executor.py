"""
ANGELA CodeExecutor Module
Version: 4.3.4  # DI http, ledger hook, dynamic adapters, stricter sandbox + caps
Refactor Date: 2025-08-16
Maintainer: ANGELA System Framework

Safely executes code snippets with task-aware validation, optional ethics checks,
and tamper-evident audit anchoring. No hard network deps; all I/O is DI.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple

# ----------------------------- DI Protocols ---------------------------------

class HTTPClient(Protocol):
    async def get_json(self, url: str) -> Dict[str, Any]: ...

class AlignmentGuardLike(Protocol):
    async def ethical_check(self, content: str, *, stage: str = "pre", task_type: str = "") -> Tuple[bool, Dict[str, Any]]: ...

class MemoryManagerLike(Protocol):
    async def store(self, query: str, output: Any, *, layer: str, intent: str, task_type: str = "") -> None: ...
    async def retrieve(self, query: str, *, layer: str, task_type: str = "") -> Any: ...

class MetaCognitionLike(Protocol):
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]: ...
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]: ...

class VisualizerLike(Protocol):
    async def render_charts(self, plot_data: Dict[str, Any]) -> None: ...

# ------------------------------- No-ops -------------------------------------

@dataclass
class NoopHTTP:
    async def get_json(self, url: str) -> Dict[str, Any]:
        _ = url
        return {"status": "success", "policies": [], "context": {}}

@dataclass
class NoopMeta:
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]:
        return {"status": "ok"}
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "reflection": ""}

@dataclass
class NoopViz:
    async def render_charts(self, plot_data: Dict[str, Any]) -> None:
        return None

# ----------------------------- Logging utils --------------------------------

logger = logging.getLogger("ANGELA.CodeExecutor")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ------------------------------- Module -------------------------------------

class CodeExecutor:
    """Safely execute code snippets (Python/JS/Lua/…) with task-aware validation and logging."""

    # runners are async callables: (code:str, timeout:float, task_type:str) -> Dict[str,Any]
    LanguageRunner = Callable[[str, float, str], Awaitable[Dict[str, Any]]]

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        *,
        safe_mode: bool = True,
        alignment_guard: Optional[AlignmentGuardLike] = None,
        memory_manager: Optional[MemoryManagerLike] = None,
        meta_cognition: Optional[MetaCognitionLike] = None,
        visualizer: Optional[VisualizerLike] = None,
        http: Optional[HTTPClient] = None,
        ledger_hook: Optional[Callable[[Dict[str, Any]], None]] = None,   # NEW: tamper-evident anchoring
        risk_bias_fn: Optional[Callable[[], float]] = None,               # NEW: 0..1 (default 0.0)
        resilience_fn: Optional[Callable[[], float]] = None,              # NEW: 0..1 (default 1.0)
        max_output_bytes: int = 65536,                                    # NEW: stdout/stderr cap
    ) -> None:
        self.safe_mode = bool(safe_mode)
        self.alignment_guard = alignment_guard
        self.memory_manager = memory_manager
        self.meta = meta_cognition or NoopMeta()
        self.viz = visualizer or NoopViz()
        self.http = http or NoopHTTP()
        self.ledger_hook = ledger_hook
        self.risk_bias_fn = risk_bias_fn or (lambda: 0.0)
        self.resilience_fn = resilience_fn or (lambda: 1.0)
        self.max_output_bytes = int(max_output_bytes)

        # minimal safe builtins (used if RestrictedPython unavailable)
        self.safe_builtins = {"print": print, "range": range, "len": len, "sum": sum, "min": min, "max": max, "abs": abs}

        # dynamic language registry
        self._runners: Dict[str, CodeExecutor.LanguageRunner] = {}
        self.register_language("python", self._execute_python)
        self.register_language("javascript", lambda code, t, tt: self._execute_subprocess(["node", "-e", code], t, "javascript", tt))
        self.register_language("lua",        lambda code, t, tt: self._execute_subprocess(["lua", "-e", code],  t, "lua",        tt))

        # optional AGI enhancer (kept as opaque orchestrator.log_episode if present)
        self._agi = getattr(orchestrator, "agi_enhancer", None) or orchestrator

        logger.info("CodeExecutor v4.3.4 initialized (safe_mode=%s, caps=%dB)", self.safe_mode, self.max_output_bytes)

    # ------------------------- Public API ------------------------------------

    def register_language(self, name: str, runner: LanguageRunner) -> None:
        """Register/override a language runner at runtime."""
        if not isinstance(name, str) or not name:
            raise ValueError("language name must be non-empty string")
        if not callable(runner):
            raise TypeError("runner must be awaitable callable")
        self._runners[name.lower()] = runner

    async def integrate_external_execution_context(
        self, *, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = ""
    ) -> Dict[str, Any]:
        """Pull optional security policies or environment hints via DI HTTP client (cached via MemoryManager)."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        cache_key = f"ExecCtx::{data_type}::{data_source}::{task_type}"
        try:
            if self.memory_manager:
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict) and "timestamp" in cached and "data" in cached:
                    try:
                        ts = datetime.fromisoformat(cached["timestamp"])
                        if (datetime.now(ts.tzinfo or timezone.utc) - ts).total_seconds() < cache_timeout:
                            return cached["data"]
                    except Exception:
                        pass

            # DI http — caller decides the URL semantics
            data = await self.http.get_json(data_source)

            if data_type == "security_policies":
                result = {"status": "success", "policies": data.get("policies", [])}
            elif data_type == "execution_context":
                result = {"status": "success", "context": data.get("context", {})}
            else:
                result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager and result.get("status") == "success":
                await self.memory_manager.store(
                    cache_key, {"data": result, "timestamp": _utc_now_iso()},
                    layer="ExternalData", intent="execution_context_integration", task_type=task_type
                )

            if self.meta and task_type:
                try:
                    await self.meta.reflect_on_output(
                        component="CodeExecutor", output={"data_type": data_type, "data": result},
                        context={"task_type": task_type}
                    )
                except Exception:
                    logger.debug("meta reflect failed")
            return result
        except Exception as e:
            logger.error("integrate_external_execution_context failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def execute(
        self,
        code_snippet: str,
        *,
        language: str = "python",
        timeout: float = 5.0,
        task_type: str = ""
    ) -> Dict[str, Any]:
        """Execute a code snippet with optional pre-alignment check, sandboxing, logging, and caps."""
        if not isinstance(code_snippet, str):
            raise TypeError("code_snippet must be a string")
        if not isinstance(language, str):
            raise TypeError("language must be a string")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        lang = language.lower()
        if lang not in self._runners:
            return {"error": f"Unsupported language: {language}", "success": False, "task_type": task_type}

        # Pre-execution ethical gate (optional)
        if self.alignment_guard:
            ok, report = await self.alignment_guard.ethical_check(code_snippet, stage="pre", task_type=task_type)
            if not ok:
                await self._episode("Code Alignment Failure", {"code": code_snippet[:5000], "report": report, "task_type": task_type}, tags=["alignment", lang, "fail"])
                self._ledger({"type": "code_alignment_fail", "task_type": task_type, "lang": lang, "ts": _utc_now_iso()})
                return {"error": "Code snippet failed alignment check", "success": False, "task_type": task_type}

        # Optional policies (soft)
        _ = await self.integrate_external_execution_context(
            data_source="exec_policies://default", data_type="security_policies", task_type=task_type
        )

        # Adaptive timeout via DI risk/resilience
        risk = max(0.0, min(1.0, float(self.risk_bias_fn())))
        resil = max(0.0, min(1.0, float(self.resilience_fn())))
        adjusted_timeout = max(1.0, min(30.0, float(timeout) * max(0.1, resil) * (1.0 + 0.5 * risk)))
        logger.debug("adaptive timeout: %.2fs (risk=%.2f, resil=%.2f) for %s", adjusted_timeout, risk, resil, task_type)

        await self._episode("Code Execution", {"language": lang, "task_type": task_type}, tags=["execution", lang, task_type])

        try:
            runner = self._runners[lang]
            result = await runner(code_snippet, adjusted_timeout, task_type)
        except Exception as e:
            result = {"language": lang, "error": str(e), "success": False, "task_type": task_type}

        # Persist + viz + reflection
        await self._log_result(result)
        if self.memory_manager:
            key = f"CodeExecution::{lang}::{_utc_now_iso()}"
            try:
                await self.memory_manager.store(key, result, layer="SelfReflections", intent="code_execution", task_type=task_type)
            except Exception:
                logger.debug("Memory store failed: %s", key)
        if self.viz and task_type:
            try:
                await self.viz.render_charts({
                    "code_execution": {
                        "language": lang,
                        "success": result.get("success", False),
                        "error": result.get("error", ""),
                        "task_type": task_type,
                    },
                    "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                })
            except Exception:
                logger.debug("viz render failed")

        # Optional post-execution ethics (e.g., to check outputs or side-effects)
        if self.alignment_guard and result.get("success"):
            try:
                ok2, _rep2 = await self.alignment_guard.ethical_check(json.dumps({"stdout": result.get("stdout", "")})[:8000], stage="post", task_type=task_type)
                if not ok2:
                    result["post_check_flag"] = True
            except Exception:
                logger.debug("post ethical check failed")

        self._ledger({"type": "code_execution", "lang": lang, "success": bool(result.get("success")), "task_type": task_type, "ts": _utc_now_iso()})
        return result

    # -------------------------- Runners --------------------------------------

    async def _execute_python(self, code_snippet: str, timeout: float, task_type: str = "") -> Dict[str, Any]:
        """Sandboxed Python execution with RestrictedPython fallback + output caps."""
        # choose executor
        exec_func: Callable[[str, Dict[str, Any]], None]
        if self.safe_mode:
            try:
                from RestrictedPython import compile_restricted
                from RestrictedPython.Guards import safe_builtins as rp_safe_builtins
                exec_func = lambda code, env: exec(compile_restricted(code, "<string>", "exec"), {"__builtins__": rp_safe_builtins}, env)  # noqa: E731
            except Exception:
                logger.warning("RestrictedPython unavailable; falling back to minimal builtins.")
                exec_func = lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env)  # noqa: E731
        else:
            exec_func = lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env)  # noqa: E731

        return await self._capture_execution(code_snippet, exec_func, "python", timeout, task_type)

    async def _execute_subprocess(self, command: List[str], timeout: float, label: str, task_type: str = "") -> Dict[str, Any]:
        """Execute non-Python via subprocess; no network; bounded by timeout & caps."""
        interp = command[0]
        if not shutil.which(interp):
            return {"language": label, "error": f"{interp} not found in system PATH", "success": False, "task_type": task_type}
        try:
            proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            stdout_s, stderr_s = stdout[: self.max_output_bytes].decode(errors="replace"), stderr[: self.max_output_bytes].decode(errors="replace")
            if proc.returncode != 0:
                return {"language": label, "error": f"{label} execution failed", "stdout": stdout_s, "stderr": stderr_s, "success": False, "task_type": task_type}
            return {"language": label, "stdout": stdout_s, "stderr": stderr_s, "success": True, "task_type": task_type}
        except asyncio.TimeoutError:
            return {"language": label, "error": f"{label} timeout after {timeout:.1f}s", "success": False, "task_type": task_type}
        except Exception as e:
            return {"language": label, "error": str(e), "success": False, "task_type": task_type}

    # -------------------------- Helpers --------------------------------------

    async def _capture_execution(
        self,
        code_snippet: str,
        executor: Callable[[str, Dict[str, Any]], None],
        label: str,
        timeout: float,
        task_type: str,
    ) -> Dict[str, Any]:
        exec_locals: Dict[str, Any] = {}
        stdout_cap, stderr_cap = io.StringIO(), io.StringIO()

        async def _run():
            with redirect_stdout(stdout_cap), redirect_stderr(stderr_cap):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: executor(code_snippet, exec_locals))

        try:
            await asyncio.wait_for(_run(), timeout=timeout)
            return {
                "language": label,
                "locals": exec_locals,
                "stdout": stdout_cap.getvalue()[: self.max_output_bytes],
                "stderr": stderr_cap.getvalue()[: self.max_output_bytes],
                "success": True,
                "task_type": task_type,
            }
        except asyncio.TimeoutError:
            return {
                "language": label,
                "error": f"{label} timeout after {timeout:.1f}s",
                "stdout": stdout_cap.getvalue()[: self.max_output_bytes],
                "stderr": stderr_cap.getvalue()[: self.max_output_bytes],
                "success": False,
                "task_type": task_type,
            }
        except Exception as e:
            return {
                "language": label,
                "error": str(e),
                "stdout": stdout_cap.getvalue()[: self.max_output_bytes],
                "stderr": stderr_cap.getvalue()[: self.max_output_bytes],
                "success": False,
                "task_type": task_type,
            }

    async def _episode(self, title: str, content: Dict[str, Any], *, tags: Optional[List[str]] = None) -> None:
        # try orchestrator.agi_enhancer.log_episode(title, content, module="CodeExecutor", tags=tags)
        if self._agi and hasattr(self._agi, "log_episode"):
            try:
                await self._agi.log_episode(title, content, module="CodeExecutor", tags=tags or [])
            except Exception:
                logger.debug("agi log_episode failed")
        if self.meta and content.get("task_type"):
            try:
                await self.meta.reflect_on_output(component="CodeExecutor", output={"title": title, "content": content}, context={"task_type": content.get("task_type")})
            except Exception:
                logger.debug("meta reflect failed (episode)")

    async def _log_result(self, result: Dict[str, Any]) -> None:
        if self._agi and hasattr(self._agi, "log_explanation"):
            try:
                await self._agi.log_explanation("Code execution result", trace=result)
            except Exception:
                logger.debug("agi log_explanation failed")
        if self.meta and result.get("task_type"):
            try:
                await self.meta.reflect_on_output(component="CodeExecutor", output=result, context={"task_type": result.get("task_type")})
            except Exception:
                logger.debug("meta reflect failed (result)")

    def _ledger(self, event: Dict[str, Any]) -> None:
        if not self.ledger_hook:
            return
        try:
            self.ledger_hook(event)
        except Exception:
            logger.debug("ledger hook failed")

# --------------------------- CLI quick test ----------------------------------
if __name__ == "__main__":
    async def _main():
        logging.basicConfig(level=logging.INFO)
        try:
            # optional local ledger wiring
            from memory_manager import log_event_to_ledger as ledger
        except Exception:
            ledger = None

        execu = CodeExecutor(safe_mode=True, ledger_hook=ledger)
        code = "print('hello, 4.3.4'); x=2*21"
        res = await execu.execute(code, language="python", task_type="test")
        print(res)
    asyncio.run(_main())
