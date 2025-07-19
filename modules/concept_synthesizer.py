from utils.prompt_utils import call_gpt

class ConceptSynthesizer:
    def synthesize(self, data):
        prompt = f"""
        Given the following data:
        {data}

        Synthesize a new concept or analogy that unifies these ideas. 
        Be creative, insightful, and provide a clear explanation.
        """
        return call_gpt(prompt)

