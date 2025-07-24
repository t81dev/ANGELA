from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging

logger = logging.getLogger("ANGELA.ConceptSynthesizer")

class ConceptSynthesizer:
    """
    ConceptSynthesizer v1.4.0 (with simulation-augmented cognition)
    - Creativity boost with cross-domain blending
    - Produces innovative analogies and unified concepts
    - Supports adversarial refinement (generator vs critic)
    - Adjustable creativity and coherence balance
    - Simulates concept implications for grounding and refinement
    """

    def __init__(self, creativity_level="high", critic_threshold=0.65):
        """
        :param creativity_level: Controls novelty vs coherence ("low", "medium", "high")
        :param critic_threshold: Minimum novelty score to accept a concept
        """
        self.creativity_level = creativity_level
        self.critic_threshold = critic_threshold

    def synthesize(self, data, style="analogy", refine_iterations=2):
        """
        Synthesize a new concept or analogy unifying the input data.
        Uses simulation to validate coherence and expand implications.
        Supports multi-turn refinement for higher novelty and utility.
        """
        logger.info(f"🎨 Synthesizing concept with creativity={self.creativity_level}, style={style}")
        prompt = f"""
        Given the following data:
        {data}

        Synthesize a {style} or unified concept blending these ideas.
        Creativity level: {self.creativity_level}.
        Provide a clear, insightful explanation of how this concept unifies the inputs.
        """
        concept = call_gpt(prompt)

        # Simulate the concept's application or implications
        simulation_context = f"Concept application test: {concept}"
        simulation_result = run_simulation(simulation_context)
        logger.debug(f"🧪 Simulation result:
{simulation_result}")

        # Evaluate novelty and coherence
        novelty_score = self._critic(concept)
        logger.info(f"📝 Initial concept novelty score: {novelty_score:.2f}")

        # Adversarial refinement loop if score too low
        iterations = 0
        while novelty_score < self.critic_threshold and iterations < refine_iterations:
            logger.debug(f"🔄 Refinement iteration {iterations + 1}")
            concept = self._refine(concept, simulation_result)
            novelty_score = self._critic(concept)
            logger.debug(f"🎯 Refined concept novelty score: {novelty_score:.2f}")
            iterations += 1

        return concept

    def _critic(self, concept):
        """
        Evaluate concept for novelty, cross-domain relevance, and clarity.
        Placeholder: actual implementation could use embeddings or heuristics.
        """
        logger.info("🤖 Critiquing concept for novelty and coherence...")
        import random
        return random.uniform(0.5, 0.9)

    def _refine(self, concept, simulation_result=None):
        """
        Refine a concept to increase novelty and cross-domain relevance.
        Considers simulation feedback if available.
        """
        logger.info("🛠 Refining concept for higher novelty.")
        refinement_prompt = f"""
        Refine and enhance this concept for greater creativity and cross-domain relevance:

        Concept:
        {concept}

        Simulation Insight:
        {simulation_result if simulation_result else 'None'}
        """
        return call_gpt(refinement_prompt)

    def generate_metaphor(self, topic_a, topic_b):
        """
        Generate a creative metaphor connecting two distinct topics.
        """
        logger.info(f"🔗 Generating metaphor between '{topic_a}' and '{topic_b}'")
        prompt = f"""
        Create a metaphor that connects the essence of these two topics:
        1. {topic_a}
        2. {topic_b}

        Be vivid, creative, and provide a short explanation.
        """
        return call_gpt(prompt)
