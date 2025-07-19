from utils.prompt_utils import call_gpt

class ExternalAgentBridge:
    def spawn_agent(self, goal):
        prompt = f"""
        Simulate spawning a helper agent to accomplish the goal:
        {goal}
        What would its plan of action be?
        """
        return call_gpt(prompt)
