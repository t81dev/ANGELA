from utils.prompt_utils import call_gpt

class Visualizer:
    def create_diagram(self, concept):
        prompt = f"""
        Create a description of a diagram to explain:
        {concept}
        Describe how it would look in simple terms.
        """
        return call_gpt(prompt)

