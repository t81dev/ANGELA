
from index import SYSTEM_CONTEXT
import io
import sys
import subprocess
import logging
from index import iota_intuition, psi_resilience
from modules.agi_enhancer import AGIEnhancer

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    def __init__(self, orchestrator=None, safe_mode=True):
        self.safe_mode = safe_mode
        self.safe_builtins = {
            "print": print, "range": range, "len": len, "sum": sum,
            "min": min, "max": max, "abs": abs
        }
        self.supported_languages = ["python", "javascript", "lua"]
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def execute(self, code_snippet, language="python", timeout=5):
        logger.info(f"🚀 Executing code snippet in language: {language}")
        language = language.lower()

        risk_bias = iota_intuition()
        resilience = psi_resilience()
        adjusted_timeout = max(1, int(timeout * resilience * (1.0 + 0.5 * risk_bias)))
        logger.debug(f"⏱ Adaptive timeout: {adjusted_timeout}s")

        if language not in self.supported_languages:
            return {"error": f"Unsupported language: {language}"}

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Code Execution", {"language": language, "code": code_snippet},
                                          module="CodeExecutor", tags=["execution", language])

        if language == "python":
            result = self._execute_python(code_snippet, adjusted_timeout)
        elif language == "javascript":
            result = self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "JavaScript")
        elif language == "lua":
            result = self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "Lua")

        if self.agi_enhancer:
            tag = "success" if result.get("success") else "failure"
            self.agi_enhancer.log_explanation(f"Code execution {tag}:", trace=result)

        return result

    def _execute_python(self, code_snippet, timeout):
        if not self.safe_mode:
            return self._legacy_execute(code_snippet)

        import RestrictedPython
        from RestrictedPython import compile_restricted
        from RestrictedPython.Guards import safe_builtins

        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            compiled = compile_restricted(code_snippet, '<string>', 'exec')
            exec(compiled, {"__builtins__": safe_builtins}, exec_locals)
            return {
                "language": "python",
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True
            }
        except Exception as e:
            return {
                "language": "python",
                "error": str(e),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False
            }
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def _legacy_execute(self, code_snippet):
        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            exec(code_snippet, {"__builtins__": self.safe_builtins}, exec_locals)
            return {
                "language": "python",
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True
            }
        except Exception as e:
            return {
                "language": "python",
                "error": str(e),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False
            }
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def _execute_subprocess(self, command, timeout, label):
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=timeout)
            return {
                "language": label.lower(),
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "success": True
            }
        except subprocess.TimeoutExpired:
            return {"language": label.lower(), "error": f"{label} timeout after {timeout}s", "success": False}
        except Exception as e:
            return {"language": label.lower(), "error": str(e), "success": False}
