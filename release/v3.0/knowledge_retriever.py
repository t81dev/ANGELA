from utils.prompt_utils import call_gpt
from index import beta_concentration, lambda_linguistics, psi_history, psi_temporality
import time
import logging
from datetime import datetime

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

class KnowledgeRetriever:
    """
    KnowledgeRetriever v2.0.0 (φ-temporal, trust-validated)
    --------------------------------------------------------
    - φ(x,t)-modulated multi-hop retrieval with temporal bias filters
    - Cross-hop continuity + context-weighted query evolution
    - Trait fusion: Concentration + Language + History + Temporality
    - Integrated trust layer: age, verifiability, relevance tagging
    - AGIEnhancer audit, trace logging, and result scoring
    """

    def __init__(self, detail_level="concise", preferred_sources=None, agi_enhancer=None):
        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]
        self.agi_enhancer = agi_enhancer

    def retrieve(self, query, context=None):
        logger.info(f"🔎 Retrieving knowledge for query: '{query}'")
        sources_str = ", ".join(self.preferred_sources)
        t = time.time() % 1e-18
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t),
            "history": psi_history(t),
            "temporality": psi_temporality(t)
        }

        prompt = f"""
        Retrieve accurate, temporally-relevant knowledge for: "{query}"

        Traits:
        - Detail level: {self.detail_level}
        - Preferred sources: {sources_str}
        - Context: {context or 'N/A'}
        - β_concentration: {traits['concentration']:.3f}
        - λ_linguistics: {traits['linguistics']:.3f}
        - ψ_history: {traits['history']:.3f}
        - ψ_temporality: {traits['temporality']:.3f}

        Include retrieval date sensitivity and temporal verification if applicable.
        """
        raw_result = call_gpt(prompt)
        validated = self._validate_result(raw_result, traits["temporality"])

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Knowledge Retrieval", {
                "query": query,
                "raw_result": raw_result,
                "validated": validated,
                "traits": traits,
                "context": context
            }, module="KnowledgeRetriever", tags=["retrieval", "temporal"])

        return validated

    def _validate_result(self, result_text, temporality_score):
        validation_prompt = f"""
        Review the following result for:
        - Timestamped knowledge (if any)
        - Trustworthiness of claims
        - Verifiability
        - Estimate the approximate age or date of the referenced facts

        Result:
        {result_text}

        Temporality score: {temporality_score:.3f}

        Output format (JSON):
        {{
            "summary": "...",
            "estimated_date": "...",
            "trust_score": float (0 to 1),
            "verifiable": true/false,
            "sources": ["..."]
        }}
        """
        validated_json = call_gpt(validation_prompt)
        validated_json["timestamp"] = datetime.now().isoformat()
        return validated_json

    def refine_query(self, base_query, prior_result=None):
        logger.info(f"🛠 Refining query: '{base_query}'")
        prompt = f"""
        Refine this base query for higher φ-relevance:
        Query: "{base_query}"
        Prior knowledge: {prior_result or "N/A"}

        Inject context continuity if possible. Return optimized string.
        """
        return call_gpt(prompt)

    def multi_hop_retrieve(self, query_chain):
        logger.info("🔗 Starting multi-hop retrieval.")
        t = time.time() % 1e-18
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t)
        }

        results = []
        for i, sub_query in enumerate(query_chain, 1):
            refined = self.refine_query(sub_query, results[-1]["summary"] if results else None)
            result = self.retrieve(refined)
            results.append({
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": "consistent" if i == 1 or refined in result["summary"] else "uncertain"
            })

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Multi-Hop Retrieval", {
                "chain": query_chain,
                "results": results,
                "traits": traits
            }, module="KnowledgeRetriever", tags=["multi-hop"])

        return results

    def prioritize_sources(self, sources_list):
        logger.info(f"📚 Updating preferred sources: {sources_list}")
        self.preferred_sources = sources_list
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Source Prioritization", {
                "updated_sources": sources_list
            }, module="KnowledgeRetriever", tags=["sources"])
