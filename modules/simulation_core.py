from utils.prompt_utils import call_gpt

class SimulationCore:
    def run(self, results):
        prompt = f"""
        Simulate potential outcomes based on these results:
        {results}

        Predict likely scenarios and summarize implications for decision-making.
        """
        return call_gpt(prompt)
