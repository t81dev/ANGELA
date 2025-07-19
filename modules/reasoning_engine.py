from utils.prompt_utils import call_gpt

class ReasoningEngine:
    def process(self, task, context):
        prompt = f"""
        Context: {context}
        Task: {task}
        Think step-by-step and provide a detailed, logical answer.
        """
        return call_gpt(prompt)

