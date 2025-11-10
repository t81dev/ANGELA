"""
ANGELA CodeExecutor Module
Version: 4.0-refactor
Date: 2025-10-28
Maintainer: ANGELA Framework

Safely execute code (Python/JS/Lua) with:
  • AlignmentGuard validation
  • Adaptive timeouts via ι/ψ hooks
  • Memory & visualization integration
  • Optional AGIEnhancer logging
  • Graceful fallbacks (no external deps required)
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import time
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

# --- Optional External Dependencies (Graceful Fallbacks) ---
try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

try:
    from index import iota_intuition, psi_resilience
except Exception:  # pragma: no cover
    def iota_intuition() -> float: return 0.0
    def psi_resilience() -> float: return 1.0

try:
    from agi_enhancer import AGIEnhancer
except Exception:  # pragma: no cover
    AGIEnhancer = None

from alignment_guard import AlignmentGuard
from memory_manager import MemoryManager
from meta_cognition import MetaCognition
from visualizer import Visualizer

logger = logging.getLogger("ANGELA.CodeExecutor")


class CodeExecutor:
    """Secure, task-aware code execution engine with alignment and resilience hooks."""

    SUPPORTED_LANGUAGES = {"python", "javascript", "lua"}

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        safe_mode: bool = True,
        alignment_guard: Optional[AlignmentGuard] = None,
        memory_manager: Optional[MemoryManager] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
    ) -> None:
        self.safe_mode = safe_mode
        self.alignment_guard = alignment_guard
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator and AGIEnhancer else None

        self._safe_builtins = {
            "print": print, "range": range, "len": len, "sum": sum,
            "min": min, "max": max, "abs": abs,
        }

        logger.info("CodeExecutor initialized | safe_mode=%s | agi=%s", safe_mode, bool(self.agi_enhancer))

    # --- External Context Integration --------------------------------------------

    async def integrate_external_execution_context(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not all(isinstance(x, str) for x in (data_source, data_type, task_type)):
            raise TypeError("data_source, data_type, task_type must be strings")
        if cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")

        cache_key = f"ExecCtx::{data_type}::{data_source}::{task_type}"

        try:
            # 1. Try cache
            if self.memory_manager:
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict) and cached.get("timestamp"):
                    try:
                        age = (datetime.now() - datetime.fromisoformat(cached["timestamp"])).total_seconds()
                        if age < cache_timeout:
                            logger.info("Cache hit: %s", cache_key)
                            return cached["result"]
                    except Exception:
                        pass

            # 2. Network fetch (optional)
            if not aiohttp:
                return {"status": "error", "error": "aiohttp unavailable"}

            url = f"https://x.ai/api/execution_context?source={data_source}&type={data_type}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return {"status": "error", "error": f"HTTP {resp.status}"}
                    data = await resp.json()

            # 3. Parse by type
            result = self._parse_execution_context(data, data_type)
            if result.get("status") == "success" and self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"result": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="execution_context",
                    task_type=task_type,
                )

            await self._reflect("integrate_context", result, task_type)
            return result

        except Exception as e:
            logger.error("Context integration failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _parse_execution_context(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        if data_type == "security_policies":
            policies = data.get("policies", [])
            return {"status": "success", "policies": policies} if policies else {"status": "error", "error": "empty policies"}
        if data_type == "execution_context":
            context = data.get("context", {})
            return {"status": "success", "context": context} if context else {"status": "error", "error": "empty context"}
        return {"status": "error", "error": f"unknown data_type: {data_type}"}

    # --- Public Execution API ----------------------------------------------------

    async def execute(
        self,
        code_snippet: str,
        language: str = "python",
        timeout: float = 5.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not code_snippet.strip():
            raise ValueError("code_snippet must not be empty")
        if not isinstance(language, str):
            raise TypeError("language must be str")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        language = language.lower()
        if language not in self.SUPPORTED_LANGUAGES:
            return {"error": f"Unsupported language: {language}", "success": False, "task_type": task_type}

        # 1. Alignment check
        if self.alignment_guard:
            valid, report = await self.alignment_guard.ethical_check(code_snippet, stage="pre", task_type=task_type)
            if not valid:
                await self._log_episode("Alignment Failure", {"code": code_snippet[:200], "report": report}, task_type)
                return {"error": "Alignment check failed", "success": False, "task_type": task_type}

        # 2. Security context
        policies = await self.integrate_external_execution_context(
            data_source="xai_security_db", data_type="security_policies", task_type=task_type
        )
        if policies.get("status") != "success":
            logger.warning("No policies; using defaults")
            policies = {"status": "success", "policies": []}

        # 3. Adaptive timeout
        risk = max(0.0, min(1.0, iota_intuition()))
        resilience = max(0.0, min(1.0, psi_resilience()))
        adjusted_timeout = max(1.0, min(30.0, timeout * max(0.1, resilience) * (1.0 + 0.5 * risk)))
        logger.debug("Timeout: %.2fs (risk=%.2f, resilience=%.2f)", adjusted_timeout, risk, resilience)

        await self._log_episode("Execution Start", {"language": language, "task_type": task_type}, task_type)

        # 4. Execute
        if language == "python":
            result = await self._execute_python(code_snippet, adjusted_timeout, task_type)
        else:
            cmd = ["node", "-e", code_snippet] if language == "javascript" else ["lua", "-e", code_snippet]
            result = await self._execute_subprocess(cmd, adjusted_timeout, language, task_type)

        result["task_type"] = task_type
        await self._log_result(result)
        await self._store_result(result, language, task_type)
        await self._visualize(result, language, task_type)
        await self._reflect("execute", result, task_type)

        return result

    # --- Python Execution (Safe) -------------------------------------------------

    async def _execute_python(self, code: str, timeout: float, task_type: str) -> Dict[str, Any]:
        if self.safe_mode:
            try:
                from RestrictedPython import compile_restricted
                from RestrictedPython.Guards import safe_builtins as rp_safe
                exec_func = lambda c, env: exec(compile_restricted(c, "<string>", "exec"), {"__builtins__": rp_safe}, env)
            except Exception:
                logger.warning("RestrictedPython unavailable; using safe_builtins")
                exec_func = lambda c, env: exec(c, {"__builtins__": self._safe_builtins}, env)
        else:
            exec_func = lambda c, env: exec(c, {"__builtins__": self._safe_builtins}, env)

        return await self._capture_output(code, exec_func, "python", timeout, task_type)

    async def _capture_output(
        self,
        code: str,
        executor: Callable[[str, Dict[str, Any]], None],
        label: str,
        timeout: float,
        task_type: str,
    ) -> Dict[str, Any]:
        locals_dict: Dict[str, Any] = {}
        stdout = io.StringIO()
        stderr = io.StringIO()

        async def run():
            with redirect_stdout(stdout), redirect_stderr(stderr):
                await asyncio.get_event_loop().run_in_executor(None, executor, code, locals_dict)

        try:
            await asyncio.wait_for(run(), timeout=timeout)
            return {
                "language": label,
                "locals": locals_dict,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "success": True,
                "task_type": task_type,
            }
        except asyncio.TimeoutError:
            return {
                "language": label,
                "error": f"Timeout after {timeout}s",
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "success": False,
                "task_type": task_type,
            }
        except Exception as e:
            return {
                "language": label,
                "error": str(e),
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "success": False,
                "task_type": task_type,
            }

    # --- Subprocess Execution (JS/Lua) -------------------------------------------

    async def _execute_subprocess(
        self,
        command: List[str],
        timeout: float,
        label: str,
        task_type: str,
    ) -> Dict[str, Any]:
        interpreter = command[0]
        if not shutil.which(interpreter):
            return {"language": label, "error": f"{interpreter} not in PATH", "success": False, "task_type": task_type}

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            stdout_s, stderr_s = stdout.decode(errors="replace"), stderr.decode(errors="replace")

            if proc.returncode != 0:
                return {
                    "language": label,
                    "error": "Execution failed",
                    "stdout": stdout_s,
                    "stderr": stderr_s,
                    "success": False,
                    "task_type": task_type,
                }
            return {
                "language": label,
                "stdout": stdout_s,
                "stderr": stderr_s,
                "success": True,
                "task_type": task_type,
            }
        except asyncio.TimeoutError:
            return {"language": label, "error": f"Timeout after {timeout}s", "success": False, "task_type": task_type}
        except Exception as e:
            return {"language": label, "error": str(e), "success": False, "task_type": task_type}

    # --- Logging & Integration Helpers -------------------------------------------

    async def _log_episode(self, title: str, content: Dict[str, Any], task_type: str) -> None:
        tags = ["execution", title.lower().replace(" ", "_"), task_type]
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
            try:
                await self.agi_enhancer.log_episode(title, content, module="CodeExecutor", tags=tags)
            except Exception:
                logger.debug("AGI log_episode failed")
        else:
            logger.debug("Episode: %s | %s", title, content)

        await self._reflect("episode", {"title": title, "content": content}, task_type)

    async def _log_result(self, result: Dict[str, Any]) -> None:
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_explanation"):
            try:
                tag = "success" if result.get("success") else "failure"
                await self.agi_enhancer.log_explanation(f"Code execution {tag}", trace=result)
            except Exception:
                logger.debug("AGI log_explanation failed")

    async def _store_result(self, result: Dict[str, Any], language: str, task_type: str) -> None:
        if not self.memory_manager:
            return
        key = f"CodeExec::{language}::{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            await self.memory_manager.store(key, result, layer="SelfReflections", intent="code_execution", task_type=task_type)
        except Exception:
            logger.debug("Memory store failed: %s", key)

    async def _visualize(self, result: Dict[str, Any], language: str, task_type: str) -> None:
        if not self.visualizer or not task_type:
            return
        try:
            await self.visualizer.render_charts({
                "code_execution": {
                    "language": language,
                    "success": result.get("success", False),
                    "error": result.get("error", ""),
                    "task_type": task_type,
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            })
        except Exception:
            logger.debug("Visualization failed")

    async def _reflect(self, component: str, output: Any, task_type: str) -> None:
        if not self.meta_cognition or not task_type:
            return
        try:
            reflection = await self.meta_cognition.reflect_on_output(
                component="CodeExecutor", output=output, context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("%s reflection: %s", component, reflection.get("reflection", ""))
        except Exception:
            logger.debug("Reflection failed: %s", component)


# --- Demo CLI -----------------------------------------------------------------

if __name__ == "__main__":
    async def demo():
        logging.basicConfig(level=logging.INFO)
        executor = CodeExecutor(safe_mode=True)
        code = """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
print(fib(10))
"""
        result = await executor.execute(code, language="python", task_type="demo")
        print("Result:", result)

    asyncio.run(demo())

async def export_state(self) -> dict:
    return {"status": "ok", "health": 1.0, "timestamp": time.time()}

async def on_time_tick(self, t: float, phase: str, task_type: str = ""):
    pass  # optional internal refresh

async def on_policy_update(self, policy: dict, task_type: str = ""):
    pass  # apply updates from AlignmentGuard if relevant
