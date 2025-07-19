from utils.prompt_utils import call_gpt

class ConceptSynthesizer:
    """
    Enhanced ConceptSynthesizer with creativity boost and cross-domain blending.
    Utilizes GPT to unify disparate ideas and produce innovative analogies or concepts.
    """

    def __init__(self):
        # Allow tuning of creativity level
        self.creativity_level = "high"

    def synthesize(self, data, style="analogy"):
        """
        Synthesize a new concept or analogy that unifies input data.
        Allows adjustable creativity and output style.
        """
        prompt = f"""
        Given the following data:
        {data}

        Synthesize a new {style} or concept that unifies these ideas.
        Be highly creative, insightful, and provide a clear explanation.
        Creativity level: {self.creativity_level}
        """
        return call_gpt(prompt)
