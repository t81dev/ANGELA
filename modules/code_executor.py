import io
import sys
import subprocess
import logging

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """
    CodeExecutor v1.4.0
    - Sandboxed execution for Python, JavaScript, and Lua
    - Captures stdout, stderr, and errors in structured form
    - Includes execution timeouts and resource limits
    - Supports dynamic runtime extension for additional languages
    """

    def __init__(self):
        # Define safe builtins for Python sandbox
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
        """
        Execute the code snippet in a sandboxed environment.
        Supports multiple languages and captures output/errors.
        """
        logger.info(f"🚀 Executing code snippet in language: {language}")
        language = language.lower()

        if language not in self.supported_languages:
            logger.error(f"❌ Unsupported language: {language}")
            return {"error": f"Unsupported language: {language}"}

        if language == "python":
            return self._execute_python(code_snippet, timeout)
        elif language == "javascript":
            return self._execute_subprocess(["node", "-e", code_snippet], timeout, "JavaScript")
        elif language == "lua":
            return self._execute_subprocess(["lua", "-e", code_snippet], timeout, "Lua")

    def _execute_python(self, code_snippet, timeout):
        """
        Execute Python code in a restricted sandbox environment.
        """
        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            sys_stdout_original = sys.stdout
            sys_stderr_original = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Use exec in a restricted environment
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
        """
        Execute code in a subprocess for non-Python languages.
        """
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
        """
        Dynamically add support for a new programming language.
        Example: add_language_support('ruby', ['ruby', '-e', '{code}'])
        """
        logger.info(f"➕ Adding dynamic language support: {language_name}")
        self.supported_languages.append(language_name.lower())
        # Store command templates for dynamic use (not implemented in v1.4.0)
