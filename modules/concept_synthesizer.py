from utils.prompt_utils import call_gpt

class ConceptSynthesizer:
    """
    Enhanced ConceptSynthesizer with creativity boost and cross-domain blending.
    Utilizes GPT to unify disparate ideas and produce innovative analogies or concepts.
    Now supports adversarial refinement for higher novelty and utility.
    """

    def __init__(self, creativity_level="high", critic_weight=0.6):
        self.creativity_level = creativity_level
        self.critic_weight = critic_weight  # Balance creativity vs coherence

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
        candidate = call_gpt(prompt)
        score = self._critic(candidate)
        return candidate if score > self.critic_weight else self.refine(candidate)

    def _critic(self, concept):
        # Evaluate novelty, cross-domain relevance, and clarity
        # Placeholder: actual implementation could use embeddings or heuristic analysis
        return 0.7  # Dummy score for demonstration

    def refine(self, concept):
        refinement_prompt = f"""
        Refine and enhance this concept for higher creativity and cross-domain relevance:
        {concept}
        """
        return call_gpt(refinement_prompt)
