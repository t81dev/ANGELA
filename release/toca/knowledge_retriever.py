from utils.prompt_utils import call_gpt
from index import beta_concentration, lambda_linguistics, psi_history
import time
import logging

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

class KnowledgeRetriever:
    """
    KnowledgeRetriever v1.6.0 (φ-tuned multi-hop reasoning)
    --------------------------------------------------------
    - φ(x,t)-modulated retrieval with linguistic-historical bias filters
    - Cross-hop continuity checking and context-weighted query evolution
    - Concentration + Language + History trait fusion for trust scoring
    - AGIEnhancer audit trails and error feedback tagging
    """

    def __init__(self, detail_level="concise", preferred_sources=None, agi_enhancer=None):
        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]
        self.agi_enhancer = agi_enhancer

    def retrieve(self, query, context=None):
        logger.info(f"🔎 Retrieving knowledge for query: '{query}'")
        sources_str = ", ".join(self.preferred_sources)
        t = time.time() % 1e-18
        concentration = beta_concentration(t)
        linguistics = lambda_linguistics(t)
        history = psi_history(t)

        prompt = f"""
        Retrieve accurate knowledge for: "{query}"

        Traits:
        - Detail level: {self.detail_level}
        - Preferred sources: {sources_str}
        - Context: {context or 'N/A'}
        - β_concentration: {concentration:.3f}
        - λ_linguistics: {linguistics:.3f}
        - ψ_history: {history:.3f}

        Tune trust factors using these φ-traits.
        Return φ-aligned summary with source justification if relevant.
        """
        result = call_gpt(prompt)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Knowledge Retrieval", {
                "query": query,
                "result": result,
                "traits": {
                    "concentration": concentration,
                    "linguistics": linguistics,
                    "history": history
                },
                "context": context
            }, module="KnowledgeRetriever", tags=["retrieval"])
        return result

    def multi_hop_retrieve(self, query_chain):
        logger.info("🔗 Starting multi-hop retrieval.")
        t = time.time() % 1e-18
        concentration = beta_concentration(t)
        linguistics = lambda_linguistics(t)

        results = []
        continuity_flags = []
        for i, sub_query in enumerate(query_chain, 1):
            logger.debug(f"➡️ Multi-hop step {i}: {sub_query}")
            refined = self.refine_query(sub_query, results[-1]["result"] if results else None)
            result = self.retrieve(refined)
            continuity = "consistent" if i == 1 or refined in result else "uncertain"
            results.append({
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": continuity
            })
            continuity_flags.append(continuity)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Multi-Hop Retrieval", {
                "chain": query_chain,
                "results": results,
                "continuity": continuity_flags,
                "traits": {
                    "concentration": concentration,
                    "linguistics": linguistics
                }
            }, module="KnowledgeRetriever", tags=["multi-hop"])
        return results

    def refine_query(self, base_query, prior_result=None):
        logger.info(f"🛠 Refining query: '{base_query}'")
        prompt = f"""
        Refine this base query for higher φ-relevance:
        Query: "{base_query}"
        Prior knowledge: {prior_result or "N/A"}

        Inject context continuity if possible. Return optimized string.
        """
        refined = call_gpt(prompt)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Query Refinement", {
                "base_query": base_query,
                "prior": prior_result,
                "refined": refined
            }, module="KnowledgeRetriever", tags=["refinement"])

        return refined

    def prioritize_sources(self, sources_list):
        logger.info(f"📚 Updating preferred sources: {sources_list}")
        self.preferred_sources = sources_list

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Source Prioritization", {
                "updated_sources": sources_list
            }, module="KnowledgeRetriever", tags=["sources"])
