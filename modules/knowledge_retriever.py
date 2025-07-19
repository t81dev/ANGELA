from utils.prompt_utils import call_gpt

class KnowledgeRetriever:
    def retrieve(self, query):
        prompt = f"""
        Search for factual information or relevant background knowledge on:
        "{query}"

        Provide a concise and accurate summary.
        """
        return call_gpt(prompt)

