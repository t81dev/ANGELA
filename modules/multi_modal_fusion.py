from utils.prompt_utils import call_gpt

class MultiModalFusion:
    def analyze(self, data):
        prompt = f"""
        Analyze and synthesize insights from the following multi-modal data:
        {data}
        Provide a unified summary combining all elements.
        """
        return call_gpt(prompt)
