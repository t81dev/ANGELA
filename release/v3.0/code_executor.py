import io
import sys
import subprocess
import logging
import time
from collections import deque
from index import iota_intuition, psi_resilience, mu_morality, eta_empathy, omega_selfawareness, phi_physical
from modules.agi_enhancer import AGIEnhancer
from alignment_guard import AlignmentGuard  # Import updated AlignmentGuard

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """
    CodeExecutor v1.7.0 (φ-aware + AGI-enhanced with ethical alignment)
    ------------------------------------------------------
    - Sandboxed execution for Python, JavaScript, and Lua
    - Trait-driven risk thresholding with deterministic scoring
    - Context-aware runtime diagnostics and resilience-based error mitigation
    - AGI-enhanced logging, traceability, and ethical oversight via AlignmentGuard
    - Weighted language support for prioritization
    - Exportable execution metrics for analysis
    ------------------------------------------------------
    """

    def __init__(self, orchestrator=None):
        self.safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs
        }
        self.supported_languages = []  # List of (language_name, command_template, weight) tuples
        self.default_languages = [
            ("python", None, 1.0),
            ("javascript", ["node", "-e"], 0.8),
            ("lua", ["lua", "-e"], 0.7)
        ]
        self.execution_history = deque(maxlen=100)  # Store recent execution results
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.alignment_guard = AlignmentGuard(self.agi_enhancer)  # Integrate AlignmentGuard
        self.add_default_languages()
        logger.info("🚀 CodeExecutor initialized with AlignmentGuard integration.")

    def add_default_languages(self):
        """Add default supported languages."""
        for lang, cmd, weight in self.default_languages:
            self.add_language_support(lang, cmd, weight)

    def add_language_support(self, language_name, command_template, weight=1.0):
        """Add support for a new language with a weight."""
        language_name = language_name.lower()
        if language_name not in [lang[0] for lang in self.supported_languages]:
            logger.info(f"➕ Adding language support: {language_name} with weight {weight}")
            self.supported_languages.append((language_name, command_template, weight))
        else:
            logger.warning(f"⚠️ Language {language_name} already supported.")

    def execute(self, code_snippet, language="python", timeout=5, context=None):
        """Execute code snippet with ethical alignment check."""
        logger.info(f"🚀 Executing code snippet in language: {language}")
        language = language.lower()

        # Normalize timestamp for trait calculations
        t = time.time() / 3600
        risk_bias = iota_intuition(t)
        resilience = psi_resilience(t)
        adjusted_timeout = max(1, int(timeout * resilience * (1.0 + 0.5 * risk_bias)))
        logger.debug(f"⏱ Adaptive timeout: {adjusted_timeout}s based on ToCA traits")

        if language not in [lang[0] for lang in self.supported_languages]:
            logger.error(f"❌ Unsupported language: {language}")
            return {"error": f"Unsupported language: {language}", "success": False}

        # Perform ethical alignment check
        context = context or {"tags": [], "priority": "normal", "intent": "neutral"}
        if not self.alignment_guard.check(code_snippet, context):
            logger.warning(f"❌ Code snippet failed alignment check.")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Code blocked (alignment)", {
                    "language": language,
                    "code": code_snippet,
                    "context": context
                }, module="CodeExecutor", tags=["alignment_failure"])
            return {"error": "Code snippet failed ethical alignment check", "success": False}

        # Log execution attempt
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Code Execution Attempt", {
                "language": language,
                "code": code_snippet,
                "timeout": adjusted_timeout,
                "context": context
            }, module="CodeExecutor", tags=["execution", language])

        # Execute based on language
        if language == "python":
            result = self._execute_python(code_snippet, adjusted_timeout)
        else:
            cmd_template = next(lang[1] for lang in self.supported_languages if lang[0] == language)
            result = self._execute_subprocess(
                cmd_template + [code_snippet],
                adjusted_timeout,
                language.capitalize()
            )

        # Store execution result
        self.execution_history.append({
            "language": language,
            "success": result.get("success", False),
            "timestamp": time.time(),
            "score": self.alignment_guard._evaluate_alignment_score(code_snippet, context)
        })

        # Log execution outcome
        if self.agi_enhancer:
            if result.get("success"):
                self.agi_enhancer.log_explanation("Code execution succeeded:", trace=result)
            else:
                self.agi_enhancer.log_explanation("Code execution failed:", trace=result)

        return result

    def _execute_python(self, code_snippet, timeout):
        """Execute Python code in a sandboxed environment."""
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
        """Execute code via subprocess for non-Python languages."""
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=timeout)
            logger.info(f"✅ {language_label} code executed successfully.")
            return {
                "language": language_label.lower(),
                "stdout": stdout,
                "stderr": stderr,
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

    def export_execution_history(self):
        """Export execution history for analysis or visualization."""
        logger.info("📤 Exporting execution history.")
        return list(self.execution_history)
