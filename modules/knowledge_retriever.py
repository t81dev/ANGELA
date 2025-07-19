from utils.prompt_utils import call_gpt

class KnowledgeRetriever:
    """
    Enhanced KnowledgeRetriever with source prioritization and multi-hop querying.
    Supports retrieving concise summaries or deep background knowledge.
    """

    def __init__(self, detail_level="concise"):
        self.detail_level = detail_level

    def retrieve(self, query, context=None):
        """
        Retrieve factual information or relevant background knowledge.
        Allows context-aware refinement of the query.
        """
        prompt = f"""
        Search for factual information or relevant background knowledge on:
        "{query}"

        Detail level: {self.detail_level}

        Context: {context if context else "N/A"}

        Provide an accurate, well-structured summary.
        """
        return call_gpt(prompt)

    def multi_hop_retrieve(self, query_chain):
        """
        Perform multi-hop retrieval across a chain of related queries.
        Useful for building a more complete knowledge base.
        """
        results = []
        for sub_query in query_chain:
            result = self.retrieve(sub_query)
            results.append(result)
        return results
