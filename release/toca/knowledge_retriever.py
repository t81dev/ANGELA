from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

class KnowledgeRetriever:
    """
    KnowledgeRetriever v1.4.0
    - Multi-hop reasoning for deeper knowledge synthesis
    - Source prioritization for factual and domain-specific retrieval
    - Context-aware query refinement
    - Supports concise summaries or deep background exploration
    """

    def __init__(self, detail_level="concise", preferred_sources=None):
        """
        :param detail_level: "concise" for summaries, "deep" for extensive background
        :param preferred_sources: List of domains or data sources to prioritize
        """
        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]

    def retrieve(self, query, context=None):
        """
        Retrieve factual information or relevant background knowledge.
        Prioritizes preferred sources and uses context for refinement.
        """
        logger.info(f"🔎 Retrieving knowledge for query: '{query}'")
        sources_str = ", ".join(self.preferred_sources)

        prompt = f"""
        Search for factual information or relevant background knowledge on:
        "{query}"

        Detail level: {self.detail_level}
        Preferred sources: {sources_str}
        Context: {context if context else "N/A"}

        Provide an accurate, well-structured summary.
        """
        return call_gpt(prompt)

    def multi_hop_retrieve(self, query_chain):
        """
        Perform multi-hop retrieval across a chain of related queries.
        Useful for building a more complete knowledge base.
        """
        logger.info("🔗 Starting multi-hop retrieval.")
        results = []
        for i, sub_query in enumerate(query_chain, 1):
            logger.debug(f"➡️ Multi-hop step {i}: {sub_query}")
            result = self.retrieve(sub_query)
            results.append({"step": i, "query": sub_query, "result": result})
        return results

    def refine_query(self, base_query, context):
        """
        Refine a base query using additional context for more relevant results.
        """
        logger.info(f"🛠 Refining query: '{base_query}' with context.")
        prompt = f"""
        Refine the following query for better relevance:
        Base query: "{base_query}"
        Context: {context}

        Return an optimized query string.
        """
        return call_gpt(prompt)

    def prioritize_sources(self, sources_list):
        """
        Update the list of preferred sources for retrieval.
        """
        logger.info(f"📚 Updating preferred sources: {sources_list}")
        self.preferred_sources = sources_list
