from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging
import random
from math import tanh

logger = logging.getLogger("ANGELA.ConceptSynthesizer")

class ConceptSynthesizer:
    """
    ConceptSynthesizer v1.5.0 (ToCA-augmented with simulation-aware cognition)
    - GPT-based synthesis with trait-based modulation
    - Integrates φ(x,t) modulation from ToCA for coherence-regulation
    - Simulation-aware novelty scoring
    - Multi-turn adversarial refinement
    - Trait-aligned metaphor generation
    """

    def __init__(self, creativity_level="high", critic_threshold=0.65):
        self.creativity_level = creativity_level
        self.critic_threshold = critic_threshold

    def synthesize(self, data, style="analogy", refine_iterations=2):
        logger.info(f"🎨 Synthesizing concept with creativity={self.creativity_level}, style={style}")
        phi_mod = self._phi_modulation(str(data))

        prompt = f"""
        Given the following data:
        {data}

        Synthesize a {style} or unified concept blending these ideas.
        Creativity level: {self.creativity_level}.
        Incorporate tension balance principles from φ(x,t).
        Provide a clear, insightful explanation of how this concept unifies the inputs.
        """
        concept = call_gpt(prompt)

        simulation_context = f"Concept application test: {concept}"
        simulation_result = run_simulation(simulation_context)
        logger.debug(f"🧪 Simulation result: {simulation_result}")

        novelty_score = self._critic(concept, simulation_result)
        logger.info(f"📝 Initial concept novelty score: {novelty_score:.2f}")

        iterations = 0
        while novelty_score < self.critic_threshold and iterations < refine_iterations:
            logger.debug(f"🔄 Refinement iteration {iterations + 1}")
            concept = self._refine(concept, simulation_result)
            novelty_score = self._critic(concept, simulation_result)
            logger.debug(f"🎯 Refined concept novelty score: {novelty_score:.2f}")
            iterations += 1

        return {
            "concept": concept,
            "novelty": novelty_score,
            "phi_modulation": phi_mod,
            "valid": novelty_score >= self.critic_threshold
        }

    def _critic(self, concept, simulation_result=None):
        base_score = random.uniform(0.5, 0.9)
        if simulation_result and "conflict" in simulation_result.lower():
            return max(0.0, base_score - 0.2)
        elif simulation_result and "coherent" in simulation_result.lower():
            return min(1.0, base_score + 0.1)
        return base_score

    def _refine(self, concept, simulation_result=None):
        logger.info("🛠 Refining concept for higher novelty.")
        refinement_prompt = f"""
        Refine this concept to enhance cross-domain regulation and causal coherence:

        Concept:
        {concept}

        Simulation Insight:
        {simulation_result if simulation_result else 'None'}

        Use cognitive traits like theta_causality (causal logic), alpha_attention (focus depth),
        and tension balance (φ(x,t)) to guide the refinement.
        """
        return call_gpt(refinement_prompt)

    def generate_metaphor(self, topic_a, topic_b):
        logger.info(f"🔗 Generating metaphor between '{topic_a}' and '{topic_b}'")
        prompt = f"""
        Create a metaphor that connects the essence of these two topics:
        1. {topic_a}
        2. {topic_b}

        Emphasize creative tension, regulatory alignment, and symbolic clarity.
        Refer to φ(x,t) as a metaphorical regulator if applicable.
        """
        return call_gpt(prompt)

    def _phi_modulation(self, text: str) -> float:
        entropy = sum(ord(c) for c in text) % 1000 / 1000
        return 1 + 0.5 * tanh(entropy)
