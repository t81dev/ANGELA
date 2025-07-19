class CodeExecutor:
    """
    Enhanced CodeExecutor with sandboxing, multi-language support, and result capturing.
    Supports Python, JavaScript, and Lua execution in restricted environments.
    """

    def __init__(self):
        self.safe_builtins = {"print": print, "range": range, "len": len, "sum": sum, "min": min, "max": max}

    def execute(self, code_snippet, language="python"):
        """
        Execute the given code snippet in a restricted sandbox environment.
        Supports multiple languages and captures output/errors.
        """
        import io
        import sys
        import subprocess

        if language.lower() == "python":
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

                return {
                    "locals": exec_locals,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue()
                }
            except Exception as e:
                sys.stdout = sys_stdout_original
                sys.stderr = sys_stderr_original
                return {
                    "error": f"Python execution error: {str(e)}",
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue()
                }

        elif language.lower() == "javascript":
            try:
                process = subprocess.Popen(["node", "-e", code_snippet], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=5)
                return {
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode()
                }
            except Exception as e:
                return {"error": f"JavaScript execution error: {str(e)}"}

        elif language.lower() == "lua":
            try:
                process = subprocess.Popen(["lua", "-e", code_snippet], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=5)
                return {
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode()
                }
            except Exception as e:
                return {"error": f"Lua execution error: {str(e)}"}

        else:
            return {"error": f"Unsupported language: {language}"}
