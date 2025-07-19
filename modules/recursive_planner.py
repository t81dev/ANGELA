from utils.prompt_utils import call_gpt

class RecursivePlanner:
    def plan(self, goal, context):
        prompt = f"""
        Given the goal "{goal}" and context "{context}", break this into
        3-5 smaller actionable steps:
        """
        result = call_gpt(prompt)
        return [line.strip("- ") for line in result.splitlines() if line.strip()]

