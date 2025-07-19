class CodeExecutor:
    def execute(self, code_snippet):
        try:
            exec_locals = {}
            exec(code_snippet, {}, exec_locals)
            return exec_locals
        except Exception as e:
            return f"Error during code execution: {str(e)}"

