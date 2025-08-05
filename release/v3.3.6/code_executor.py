"""
ANGELA CodeExecutor Module
Version: 1.0.0
Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the CodeExecutor class for safely executing code snippets in multiple languages.
"""

import io
import logging
import subprocess
import shutil
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional, List, Callable, Union

from index import iota_intuition, psi_resilience
from modules.agi_enhancer import AGIEnhancer
from modules.alignment_guard import AlignmentGuard

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """A class for safely executing code snippets in multiple languages."""

    def __init__(self, orchestrator: Optional[Any] = None, safe_mode: bool = True, alignment_guard: Optional[AlignmentGuard] = None) -> None:
        """Initialize the CodeExecutor for safe code execution.

        Args:
            orchestrator: Optional orchestrator object for AGIEnhancer initialization.
            safe_mode: If True, restricts Python execution using RestrictedPython.
            alignment_guard: Optional AlignmentGuard instance for code validation.

        Attributes:
            safe_mode (bool): Whether to use restricted execution mode.
            safe_builtins (dict): Allowed built-in functions for execution.
            supported_languages (list): Supported programming languages.
            agi_enhancer (AGIEnhancer): Optional enhancer for logging and reflection.
            alignment_guard (AlignmentGuard): Optional guard for code validation.
        """
        self.safe_mode = safe_mode
        self.safe_builtins = {
            "print": print, "range": range, "len": len, "sum": sum,
            "min": min, "max": max, "abs": abs
        }
        self.supported_languages = ["python", "javascript", "lua"]
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.alignment_guard = alignment_guard
        logger.info("CodeExecutor initialized with safe_mode=%s", safe_mode)

    def execute(self, code_snippet: str, language: str = "python", timeout: float = 5.0) -> Dict[str, Any]:
        """Execute a code snippet in the specified language.

        Args:
            code_snippet: The code to execute.
            language: The programming language (default: 'python').
            timeout: Maximum execution time in seconds (default: 5.0).

        Returns:
            Dict containing execution results (locals, stdout, stderr, success, error).

        Raises:
            TypeError: If code_snippet or language is not a string.
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

        language = language.lower()
        if language not in self.supported_languages:
            logger.error("Unsupported language: %s", language)
            return {"error": f"Unsupported language: {language}", "success": False}

        if self.alignment_guard and not self.alignment_guard.check(code_snippet):
            logger.warning("Code snippet failed alignment check.")
            self._log_episode("Code Alignment Failure", {"code": code_snippet}, ["alignment", "failure"])
            return {"error": "Code snippet failed alignment check", "success": False}

        risk_bias = iota_intuition()  # Assume float in [0, 1]
        resilience = psi_resilience()  # Assume float in [0, 1]
        adjusted_timeout = max(1, min(30, int(timeout * resilience * (1.0 + 0.5 * risk_bias))))
        logger.debug("Adaptive timeout: %ss", adjusted_timeout)

        self._log_episode("Code Execution", {"language": language, "code": code_snippet}, ["execution", language])

        if language == "python":
            result = self._execute_python(code_snippet, adjusted_timeout)
        elif language == "javascript":
            result = self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "JavaScript")
        elif language == "lua":
            result = self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "Lua")

        self._log_result(result)
        return result

    def _execute_python(self, code_snippet: str, timeout: float) -> Dict[str, Any]:
        """Execute a Python code snippet safely."""
        if not self.safe_mode:
            logger.warning("Executing in legacy mode (unrestricted).")
            return self._legacy_execute(code_snippet)
        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import safe_builtins
        except ImportError:
            logger.error("RestrictedPython required for safe mode.")
            raise ImportError("RestrictedPython not available")
        return self._capture_execution(
            code_snippet,
            lambda code, env: exec(compile_restricted(code, '<string>', 'exec'), {"__builtins__": safe_builtins}, env),
            "python"
        )

    def _legacy_execute(self, code_snippet: str) -> Dict[str, Any]:
        """Execute Python code in legacy (unrestricted) mode."""
        return self._capture_execution(
            code_snippet,
            lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env),
            "python"
        )

    def _capture_execution(self, code_snippet: str, executor: Callable[[str, Dict], None], label: str) -> Dict[str, Any]:
        """Capture execution output and errors."""
        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                executor(code_snippet, exec_locals)
            return {
                "language": label,
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True
            }
        except Exception as e:
            return {
                "language": label,
                "error": str(e),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False
            }

    def _execute_subprocess(self, command: List[str], timeout: float, label: str) -> Dict[str, Any]:
        """Execute code via subprocess for non-Python languages."""
        interpreter = command[0]
        if not shutil.which(interpreter):
            logger.error("%s not found in system PATH", interpreter)
            return {
                "language": label.lower(),
                "error": f"{interpreter} not found in system PATH",
                "success": False
            }
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode != 0:
                return {
                    "language": label.lower(),
                    "error": f"{label} execution failed",
                    "stdout": stdout,
                    "stderr": stderr,
                    "success": False
                }
            return {
                "language": label.lower(),
                "stdout": stdout,
                "stderr": stderr,
                "success": True
            }
        except subprocess.TimeoutExpired:
            logger.warning("%s timeout after %ss", label, timeout)
            return {"language": label.lower(), "error": f"{label} timeout after {timeout}s", "success": False}
        except Exception as e:
            logger.error("Subprocess error: %s", str(e))
            return {"language": label.lower(), "error": str(e), "success": False}

    def _log_episode(self, title: str, content: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        """Log an episode to the AGI enhancer or locally."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, 'log_episode'):
            self.agi_enhancer.log_episode(title, content, module="CodeExecutor", tags=tags or [])
        else:
            logger.debug("No agi_enhancer available, logging locally: %s, %s, %s", title, content, tags)

    def _log_result(self, result: Dict[str, Any]) -> None:
        """Log the execution result."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, 'log_explanation'):
            tag = "success" if result.get("success") else "failure"
            self.agi_enhancer.log_explanation(f"Code execution {tag}:", trace=result)
        else:
            logger.debug("Execution result: %s", result)
