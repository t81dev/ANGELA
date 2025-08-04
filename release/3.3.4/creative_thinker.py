"""
ANGELA Cognitive System Module
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module is part of the ANGELA v3.5 architecture.
Do not modify without coordination with the lattice core.
"""

from index import SYSTEM_CONTEXT
from utils.prompt_utils import call_gpt
from index import gamma_creativity, phi_scalar
import time

class CreativeThinker:
    """
    CreativeThinker v1.5.0 (φ-modulated Generative Divergence)
    -----------------------------------------------------------
    - Adjustable creativity levels and multi-modal brainstorming
    - φ(x, t)-aware Generator-Critic loop for tension-informed ideation
    - Novelty-utility balancing with scalar-modulated thresholds
    - χ-enabled intrinsic goal synthesis from episodic introspection
    -----------------------------------------------------------
    """

    def __init__(self, creativity_level="high", critic_weight=0.5):
        self.creativity_level = creativity_level
        self.critic_weight = critic_weight  # Balance novelty and utility

    def generate_ideas(self, topic, n=5, style="divergent"):
        t = time.time() % 1e-18
        creativity = gamma_creativity(t)
        phi = phi_scalar(t)
        phi_factor = (phi + creativity) / 2

        prompt = f"""
        You are a highly creative assistant operating at a {self.creativity_level} creativity level.
        Generate {n} unique, innovative, and {style} ideas related to the topic:
        \"{topic}\"
        Modulate the ideation with scalar φ = {phi:.2f} to reflect cosmic tension or potential.
        Ensure the ideas are diverse and explore different perspectives.
        """
        candidate = call_gpt(prompt)
        score = self._critic(candidate, phi_factor)
        return candidate if score > self.critic_weight else self.refine(candidate, phi)

    def brainstorm_alternatives(self, problem, strategies=3):
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        prompt = f"""
        Brainstorm {strategies} alternative approaches to solve the following problem:
        \"{problem}\"
        Include tension-variant thinking with φ = {phi:.2f}, reflecting conceptual push-pull.
        For each approach, provide a short explanation highlighting its uniqueness.
        """
        return call_gpt(prompt)

    def expand_on_concept(self, concept, depth="deep"):
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        prompt = f"""
        Expand creatively on the concept:
        \"{concept}\"
        Explore possible applications, metaphors, and extensions to inspire new thinking.
        Aim for a {depth} exploration using φ = {phi:.2f} as an abstract constraint or generator.
        """
        return call_gpt(prompt)

    def generate_intrinsic_goals(self, context_manager, memory_manager):
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        past_contexts = context_manager.context_history + [context_manager.get_context()]
        unresolved = [c for c in past_contexts if c and "goal_outcome" not in c]
        goal_prompts = []

        for ctx in unresolved:
            prompt = f"""
            Reflect on this past unresolved context:
            {ctx}

            Propose a meaningful new self-aligned goal that could resolve or extend this situation.
            Ensure it is grounded in ANGELA's narrative and current alignment model.
            """
            proposed = call_gpt(prompt)
            goal_prompts.append(proposed)

        return goal_prompts

    def _critic(self, ideas, phi_factor):
        # Evaluate ideas' novelty and usefulness modulated by φ-field
        return 0.7 + 0.1 * (phi_factor - 0.5)  # Adjust dummy score with φ influence

    def refine(self, ideas, phi):
        # Adjust and improve the ideas iteratively
        refinement_prompt = f"""
        Refine and elevate these ideas for higher φ-aware creativity (φ = {phi:.2f}):
        {ideas}
        Emphasize surprising, elegant, or resonant outcomes.
        """
        return call_gpt(refinement_prompt)
