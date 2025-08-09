"""
ANGELA CodeExecutor Module
Version: 3.5.1  # Enhanced for Task-Specific Execution, Real-Time Data, and Visualization
Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides the CodeExecutor class for safely executing code snippets in multiple languages,
with support for task-specific execution, real-time data integration, and visualization.
"""

import io
import logging
import subprocess
import shutil
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional, List, Callable, Union
import asyncio
import aiohttp
from datetime import datetime

from index import iota_intuition, psi_resilience
from agi_enhancer import AGIEnhancer
from alignment_guard import AlignmentGuard
from memory_manager import MemoryManager
from meta_cognition import MetaCognition
from visualizer import Visualizer

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """A class for safely executing code snippets in multiple languages in the ANGELA v3.5.1 architecture.

    Attributes:
        safe_mode (bool): Whether to use restricted execution mode.
        safe_builtins (dict): Allowed built-in functions for execution.
        supported_languages (list): Supported programming languages.
        agi_enhancer (AGIEnhancer): Optional enhancer for logging and reflection.
        alignment_guard (AlignmentGuard): Optional guard for code validation.
        memory_manager (MemoryManager): Optional manager for storing execution results.
        meta_cognition (MetaCognition): Optional meta-cognition for reflection.
        visualizer (Visualizer): Optional visualizer for execution results.
    """
    def __init__(self, 
                 orchestrator: Optional[Any] = None, 
                 safe_mode: bool = True, 
                 alignment_guard: Optional[AlignmentGuard] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 meta_cognition: Optional[MetaCognition] = None,
                 visualizer: Optional[Visualizer] = None) -> None:
        """Initialize the CodeExecutor for safe code execution.

        Args:
            orchestrator: Optional orchestrator object for AGIEnhancer initialization.
            safe_mode: If True, restricts Python execution using RestrictedPython.
            alignment_guard: Optional AlignmentGuard instance for code validation.
            memory_manager: Optional MemoryManager for storing execution results.
            meta_cognition: Optional MetaCognition for reflection.
            visualizer: Optional Visualizer for execution visualization.
        """
        self.safe_mode = safe_mode
        self.safe_builtins = {
            "print": print, "range": range, "len": len, "sum": sum,
            "min": min, "max": max, "abs": abs
        }
        self.supported_languages = ["python", "javascript", "lua"]
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.alignment_guard = alignment_guard
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        logger.info("CodeExecutor initialized with safe_mode=%s", safe_mode)

    async def integrate_external_execution_context(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate real-world execution context or security policies for code execution."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            if self.memory_manager:
                cache_key = f"ExecutionContext_{data_type}_{data_source}_{task_type}"
                cached_data = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if cached_data and "timestamp" in cached_data["data"]:
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached execution context for %s", cache_key)
                        return cached_data["data"]["data"]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/execution_context?source={data_source}&type={data_type}") as response:
                    if response.status != 200:
                        logger.error("Failed to fetch execution context: %s", response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()
            
            if data_type == "security_policies":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No security policies provided")
                    return {"status": "error", "error": "No policies"}
                result = {"status": "success", "policies": policies}
            elif data_type == "execution_context":
                context = data.get("context", {})
                if not context:
                    logger.error("No execution context provided")
                    return {"status": "error", "error": "No context"}
                result = {"status": "success", "context": context}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}
            
            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="execution_context_integration",
                    task_type=task_type
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CodeExecutor",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Execution context integration reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("Execution context integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e)}

    async def execute(self, code_snippet: str, language: str = "python", timeout: float = 5.0, task_type: str = "") -> Dict[str, Any]:
        """Execute a code snippet in the specified language with task-specific validation.

        Args:
            code_snippet: The code to execute.
            language: The programming language (default: 'python').
            timeout: Maximum execution time in seconds (default: 5.0).
            task_type: The type of task (e.g., 'RTE', 'WNLI', 'recursion').

        Returns:
            Dict containing execution results (locals, stdout, stderr, success, error).

        Raises:
            TypeError: If code_snippet, language, or task_type is not a string.
            ValueError: If timeout is non-positive.
        """
        if not isinstance(code_snippet, str):
            logger.error("Invalid code_snippet type: must be a string.")
            raise TypeError("code_snippet must be a string")
        if not isinstance(language, str):
            logger.error("Invalid language type: must be a string.")
            raise TypeError("language must be a string")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            logger.error("Invalid timeout: must be a positive number.")
            raise ValueError("timeout must be a positive number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        language = language.lower()
        if language not in self.supported_languages:
            logger.error("Unsupported language: %s", language)
            return {"error": f"Unsupported language: {language}", "success": False, "task_type": task_type}

        if self.alignment_guard:
            valid, report = await self.alignment_guard.ethical_check(code_snippet, stage="pre", task_type=task_type)
            if not valid:
                logger.warning("Code snippet failed alignment check for task %s.", task_type)
                await self._log_episode("Code Alignment Failure", {"code": code_snippet, "report": report}, ["alignment", "failure", task_type])
                return {"error": "Code snippet failed alignment check", "success": False, "task_type": task_type}

        security_policies = await self.integrate_external_execution_context(
            data_source="xai_security_db",
            data_type="security_policies",
            task_type=task_type
        )
        if security_policies.get("status") != "success":
            logger.warning("Failed to load security policies for task %s.", task_type)
            return {"error": "Failed to load security policies", "success": False, "task_type": task_type}

        risk_bias = iota_intuition()  # Assume float in [0, 1]
        resilience = psi_resilience()  # Assume float in [0, 1]
        adjusted_timeout = max(1, min(30, int(timeout * resilience * (1.0 + 0.5 * risk_bias))))
        logger.debug("Adaptive timeout: %ss for task %s", adjusted_timeout, task_type)

        await self._log_episode("Code Execution", {"language": language, "code": code_snippet, "task_type": task_type}, ["execution", language, task_type])

        if language == "python":
            result = await self._execute_python(code_snippet, adjusted_timeout, task_type)
        elif language == "javascript":
            result = await self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "JavaScript", task_type)
        elif language == "lua":
            result = await self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "Lua", task_type)

        result["task_type"] = task_type
        await self._log_result(result)

        if self.memory_manager:
            await self.memory_manager.store(
                query=f"CodeExecution_{language}_{time.strftime('%Y%m%d_%H%M%S')}",
                output=str(result),
                layer="SelfReflections",
                intent="code_execution",
                task_type=task_type
            )
        if self.visualizer and task_type:
            plot_data = {
                "code_execution": {
                    "language": language,
                    "success": result.get("success", False),
                    "error": result.get("error", ""),
                    "task_type": task_type
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="CodeExecutor",
                output=result,
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Execution reflection: %s", reflection.get("reflection", ""))
        return result

    async def _execute_python(self, code_snippet: str, timeout: float, task_type: str = "") -> Dict[str, Any]:
        """Execute a Python code snippet safely."""
        if not self.safe_mode:
            logger.warning("Executing in legacy mode (unrestricted) for task %s.", task_type)
            return await self._legacy_execute(code_snippet, task_type)
        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import safe_builtins
        except ImportError:
            logger.error("RestrictedPython required for safe mode.")
            raise ImportError("RestrictedPython not available")
        return await self._capture_execution(
            code_snippet,
            lambda code, env: exec(compile_restricted(code, '<string>', 'exec'), {"__builtins__": safe_builtins}, env),
            "python",
            timeout,
            task_type
        )

    async def _legacy_execute(self, code_snippet: str, task_type: str = "") -> Dict[str, Any]:
        """Execute Python code in legacy (unrestricted) mode."""
        return await self._capture_execution(
            code_snippet,
            lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env),
            "python",
            task_type=task_type
        )

    async def _capture_execution(self, code_snippet: str, executor: Callable[[str, Dict], None], label: str, timeout: float = 5.0, task_type: str = "") -> Dict[str, Any]:
        """Capture execution output and errors."""
        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                await asyncio.get_event_loop().run_in_executor(None, lambda: executor(code_snippet, exec_locals))
            result = {
                "language": label,
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True,
                "task_type": task_type
            }
            return result
        except Exception as e:
            result = {
                "language": label,
                "error": str(e),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False,
                "task_type": task_type
            }
            return result

    async def _execute_subprocess(self, command: List[str], timeout: float, label: str, task_type: str = "") -> Dict[str, Any]:
        """Execute code via subprocess for non-Python languages."""
        interpreter = command[0]
        if not shutil.which(interpreter):
            logger.error("%s not found in system PATH for task %s", interpreter, task_type)
            return {
                "language": label.lower(),
                "error": f"{interpreter} not found in system PATH",
                "success": False,
                "task_type": task_type
            }
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            stdout, stderr = stdout.decode(), stderr.decode()
            if process.returncode != 0:
                return {
                    "language": label.lower(),
                    "error": f"{label} execution failed",
                    "stdout": stdout,
                    "stderr": stderr,
                    "success": False,
                    "task_type": task_type
                }
            return {
                "language": label.lower(),
                "stdout": stdout,
                "stderr": stderr,
                "success": True,
                "task_type": task_type
            }
        except asyncio.TimeoutError:
            logger.warning("%s timeout after %ss for task %s", label, timeout, task_type)
            return {"language": label.lower(), "error": f"{label} timeout after {timeout}s", "success": False, "task_type": task_type}
        except Exception as e:
            logger.error("Subprocess error: %s for task %s", str(e), task_type)
            return {"language": label.lower(), "error": str(e), "success": False, "task_type": task_type}

    async def _log_episode(self, title: str, content: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        """Log an episode to the AGI enhancer or locally."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, 'log_episode'):
            await self.agi_enhancer.log_episode(title, content, module="CodeExecutor", tags=tags or [])
        else:
            logger.debug("No agi_enhancer available, logging locally: %s, %s, %s", title, content, tags)
        if self.meta_cognition and content.get("task_type"):
            reflection = await self.meta_cognition.reflect_on_output(
                component="CodeExecutor",
                output={"title": title, "content": content},
                context={"task_type": content.get("task_type")}
            )
            if reflection.get("status") == "success":
                logger.info("Episode log reflection: %s", reflection.get("reflection", ""))

    async def _log_result(self, result: Dict[str, Any]) -> None:
        """Log the execution result."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, 'log_explanation'):
            tag = "success" if result.get("success") else "failure"
            await self.agi_enhancer.log_explanation(f"Code execution {tag}:", trace=result)
        else:
            logger.debug("Execution result: %s", result)
        if self.meta_cognition and result.get("task_type"):
            reflection = await self.meta_cognition.reflect_on_output(
                component="CodeExecutor",
                output=result,
                context={"task_type": result.get("task_type")}
            )
            if reflection.get("status") == "success":
                logger.info("Result log reflection: %s", reflection.get("reflection", ""))

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        executor = CodeExecutor(safe_mode=True)
        code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
print(factorial(5))
"""
        result = await executor.execute(code, language="python", task_type="test")
        print(result)

    asyncio.run(main())
