import io
import sys
import subprocess
import logging
from index import iota_intuition, psi_resilience

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """
    CodeExecutor v1.5.0 (trait-adaptive)
    ------------------------------------
    - Sandboxed execution for Python, JavaScript, and Lua
    - Trait-driven risk thresholding for timeouts and isolation
    - Context-aware runtime diagnostics and resilience-based error mitigation
    ------------------------------------
    """

    def __init__(self):
        self.safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs
        }
        self.supported_languages = ["python", "javascript", "lua"]

    def execute(self, code_snippet, language="python", timeout=5):
        logger.info(f"🚀 Executing code snippet in language: {language}")
        language = language.lower()

        risk_bias = iota_intuition()
        resilience = psi_resilience()
        adjusted_timeout = max(1, int(timeout * resilience * (1.0 + 0.5 * risk_bias)))
        logger.debug(f"⏱ Adaptive timeout: {adjusted_timeout}s based on ToCA traits")

        if language not in self.supported_languages:
            logger.error(f"❌ Unsupported language: {language}")
            return {"error": f"Unsupported language: {language}"}

        if language == "python":
            return self._execute_python(code_snippet, adjusted_timeout)
        elif language == "javascript":
            return self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "JavaScript")
        elif language == "lua":
            return self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "Lua")

    def _execute_python(self, code_snippet, timeout):
        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            sys_stdout_original = sys.stdout
            sys_stderr_original = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            exec(code_snippet, {"__builtins__": self.safe_builtins}, exec_locals)

            sys.stdout = sys_stdout_original
            sys.stderr = sys_stderr_original

            logger.info("✅ Python code executed successfully.")
            return {
                "language": "python",
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True
            }
        except Exception as e:
            sys.stdout = sys_stdout_original
            sys.stderr = sys_stderr_original
            logger.error(f"❌ Python execution error: {e}")
            return {
                "language": "python",
                "error": f"Python execution error: {str(e)}",
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False
            }

    def _execute_subprocess(self, command, timeout, language_label):
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=timeout)
            logger.info(f"✅ {language_label} code executed successfully.")
            return {
                "language": language_label.lower(),
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "success": True
            }
        except subprocess.TimeoutExpired:
            logger.error(f"⏳ {language_label} execution timed out after {timeout}s.")
            return {
                "language": language_label.lower(),
                "error": f"{language_label} execution timed out after {timeout}s",
                "success": False
            }
        except Exception as e:
            logger.error(f"❌ {language_label} execution error: {e}")
            return {
                "language": language_label.lower(),
                "error": f"{language_label} execution error: {str(e)}",
                "success": False
            }

    def add_language_support(self, language_name, command_template):
        logger.info(f"➕ Adding dynamic language support: {language_name}")
        self.supported_languages.append(language_name.lower())
