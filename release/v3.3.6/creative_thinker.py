"""
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the CreativeThinker class for generating creative ideas and goals in the ANGELA v3.5 architecture.
"""

import time
import logging
from typing import List, Union, Optional
from functools import lru_cache

from index import gamma_creativity, phi_scalar
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer
from modules.context_manager import ContextManager

logger = logging.getLogger("ANGELA.CreativeThinker")

class CreativeThinker:
    """A class for generating creative ideas and goals in the ANGELA v3.5 architecture.

    Attributes:
        creativity_level (str): Level of creativity ('low', 'medium', 'high').
        critic_weight (float): Threshold for idea acceptance in critic evaluation.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for code-based ideas.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for idea refinement.
    """

    def __init__(self, creativity_level: str = "high", critic_weight: float = 0.5,
                 alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None,
                 concept_synthesizer: Optional[ConceptSynthesizer] = None):
        if creativity_level not in ["low", "medium", "high"]:
            logger.error("Invalid creativity_level: must be 'low', 'medium', or 'high'.")
            raise ValueError("creativity_level must be 'low', 'medium', or 'high'")
        if not isinstance(critic_weight, (int, float)) or not 0 <= critic_weight <= 1:
            logger.error("Invalid critic_weight: must be between 0 and 1.")
            raise ValueError("critic_weight must be between 0 and 1")

        self.creativity_level = creativity_level
        self.critic_weight = critic_weight
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        logger.info("CreativeThinker initialized: creativity=%s, critic_weight=%.2f", creativity_level, critic_weight)

    def generate_ideas(self, topic: str, n: int = 5, style: str = "divergent") -> str:
        """Generate creative ideas for a given topic.

        Args:
            topic: The topic to generate ideas for.
            n: Number of ideas to generate (default: 5).
            style: Style of ideation (default: 'divergent').

        Returns:
            A string containing the generated ideas.

        Raises:
            TypeError: If topic or style is not a string, or n is not an integer.
            ValueError: If n is not positive or topic fails alignment check.
        """
        if not isinstance(topic, str):
            logger.error("Invalid topic type: must be a string.")
            raise TypeError("topic must be a string")
        if not isinstance(n, int) or n <= 0:
            logger.error("Invalid n: must be a positive integer.")
            raise ValueError("n must be a positive integer")
        if not isinstance(style, str):
            logger.error("Invalid style type: must be a string.")
            raise TypeError("style must be a string")
        if self.alignment_guard and not self.alignment_guard.check(topic):
            logger.warning("Topic failed alignment check.")
            raise ValueError("Topic failed alignment check")

        logger.info("Generating %d %s ideas for topic: %s", n, style, topic)
        try:
            t = time.time()
            creativity = gamma_creativity(t)
            phi = phi_scalar(t)
            phi_factor = (phi + creativity) / 2

            prompt = f"""
            You are a highly creative assistant operating at a {self.creativity_level} creativity level.
            Generate {n} unique, innovative, and {style} ideas related to the topic:
            "{topic}"
            Modulate the ideation with scalar φ = {phi:.2f} to reflect cosmic tension or potential.
            Ensure the ideas are diverse and explore different perspectives.
            """
            candidate = self._cached_call_gpt(prompt)
            if not candidate:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to generate ideas")

            if self.code_executor and style == "code":
                execution_result = self.code_executor.execute(candidate, language="python")
                if not execution_result["success"]:
                    logger.warning("Code idea execution failed: %s", execution_result["error"])
                    raise ValueError("Code idea execution failed")

            if self.concept_synthesizer and style != "code":
                synthesis_result = self.concept_synthesizer.synthesize(candidate, style="refinement")
                if synthesis_result["valid"]:
                    candidate = synthesis_result["concept"]
                    logger.info("Ideas refined using ConceptSynthesizer: %s", candidate[:50])

            score = self._critic(candidate, phi_factor)
            logger.debug("Idea score: %.2f (threshold: %.2f)", score, self.critic_weight)
            return candidate if score > self.critic_weight else self.refine(candidate, phi)
        except Exception as e:
            logger.error("Idea generation failed: %s", str(e))
            raise

    def brainstorm_alternatives(self, problem: str, strategies: int = 3) -> str:
        """Brainstorm alternative approaches to solve a problem.

        Args:
            problem: The problem to address.
            strategies: Number of strategies to generate (default: 3).

        Returns:
            A string containing the brainstormed approaches.

        Raises:
            TypeError: If problem is not a string or strategies is not an integer.
            ValueError: If strategies is not positive or problem fails alignment check.
        """
        if not isinstance(problem, str):
            logger.error("Invalid problem type: must be a string.")
            raise TypeError("problem must be a string")
        if not isinstance(strategies, int) or strategies <= 0:
            logger.error("Invalid strategies: must be a positive integer.")
            raise ValueError("strategies must be a positive integer")
        if self.alignment_guard and not self.alignment_guard.check(problem):
            logger.warning("Problem failed alignment check.")
            raise ValueError("Problem failed alignment check")

        logger.info("Brainstorming %d alternatives for problem: %s", strategies, problem)
        try:
            t = time.time()
            phi = phi_scalar(t)
            prompt = f"""
            Brainstorm {strategies} alternative approaches to solve the following problem:
            "{problem}"
            Include tension-variant thinking with φ = {phi:.2f}, reflecting conceptual push-pull.
            For each approach, provide a short explanation highlighting its uniqueness.
            """
            result = self._cached_call_gpt(prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to brainstorm alternatives")
            return result
        except Exception as e:
            logger.error("Brainstorming failed: %s", str(e))
            raise

    def expand_on_concept(self, concept: str, depth: str = "deep") -> str:
        """Expand creatively on a given concept.

        Args:
            concept: The concept to expand.
            depth: Depth of exploration ('shallow', 'medium', 'deep').

        Returns:
            A string containing the expanded concept.

        Raises:
            TypeError: If concept or depth is not a string.
            ValueError: If depth is invalid or concept fails alignment check.
        """
        if not isinstance(concept, str):
            logger.error("Invalid concept type: must be a string.")
            raise TypeError("concept must be a string")
        if not isinstance(depth, str) or depth not in ["shallow", "medium", "deep"]:
            logger.error("Invalid depth: must be 'shallow', 'medium', or 'deep'.")
            raise ValueError("depth must be 'shallow', 'medium', or 'deep'")
        if self.alignment_guard and not self.alignment_guard.check(concept):
            logger.warning("Concept failed alignment check.")
            raise ValueError("Concept failed alignment check")

        logger.info("Expanding on concept: %s (depth: %s)", concept, depth)
        try:
            t = time.time()
            phi = phi_scalar(t)
            prompt = f"""
            Expand creatively on the concept:
            "{concept}"
            Explore possible applications, metaphors, and extensions to inspire new thinking.
            Aim for a {depth} exploration using φ = {phi:.2f} as an abstract constraint or generator.
            """
            result = self._cached_call_gpt(prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to expand concept")
            return result
        except Exception as e:
            logger.error("Concept expansion failed: %s", str(e))
            raise

    def generate_intrinsic_goals(self, context_manager: ContextManager, memory_manager: Any) -> List[str]:
        """Generate intrinsic goals from unresolved contexts.

        Args:
            context_manager: ContextManager instance providing context history.
            memory_manager: Memory manager instance (not used in current implementation).

        Returns:
            A list of proposed goal strings.

        Raises:
            TypeError: If context_manager lacks required attributes.
        """
        if not hasattr(context_manager, 'context_history') or not hasattr(context_manager, 'get_context'):
            logger.error("Invalid context_manager: missing required attributes.")
            raise TypeError("context_manager must have context_history and get_context attributes")

        logger.info("Generating intrinsic goals from context history")
        try:
            t = time.time()
            phi = phi_scalar(t)
            past_contexts = list(context_manager.context_history) + [context_manager.get_context()]
            unresolved = [c for c in past_contexts if c and isinstance(c, dict) and "goal_outcome" not in c]
            goal_prompts = []

            if not unresolved:
                logger.warning("No unresolved contexts found.")
                return []

            for ctx in unresolved:
                if self.alignment_guard and not self.alignment_guard.check(str(ctx)):
                    logger.warning("Context failed alignment check, skipping.")
                    continue
                prompt = f"""
                Reflect on this past unresolved context:
                {ctx}

                Propose a meaningful new self-aligned goal that could resolve or extend this situation.
                Ensure it is grounded in ANGELA's narrative and current alignment model.
                """
                proposed = self._cached_call_gpt(prompt)
                if proposed:
                    goal_prompts.append(proposed)
                else:
                    logger.warning("call_gpt returned empty result for context: %s", ctx)

            return goal_prompts
        except Exception as e:
            logger.error("Goal generation failed: %s", str(e))
            return []

    def _critic(self, ideas: str, phi_factor: float) -> float:
        """Evaluate the novelty and quality of generated ideas."""
        try:
            base_score = min(0.9, 0.5 + len(ideas) / 1000.0)
            adjustment = 0.1 * (phi_factor - 0.5)
            simulation_result = run_simulation(f"Idea evaluation: {ideas[:100]}") or "no simulation data"
            if "coherent" in simulation_result.lower():
                base_score += 0.1
            elif "conflict" in simulation_result.lower():
                base_score -= 0.1
            score = max(0.0, min(1.0, base_score + adjustment))
            logger.debug("Critic score for ideas: %.2f", score)
            return score
        except Exception as e:
            logger.error("Critic evaluation failed: %s", str(e))
            return 0.0

    def refine(self, ideas: str, phi: float) -> str:
        """Refine ideas for higher creativity and coherence.

        Args:
            ideas: The ideas to refine.
            phi: The φ scalar for modulation.

        Returns:
            A string containing the refined ideas.

        Raises:
            TypeError: If ideas is not a string.
            ValueError: If ideas fails alignment check.
        """
        if not isinstance(ideas, str):
            logger.error("Invalid ideas type: must be a string.")
            raise TypeError("ideas must be a string")
        if self.alignment_guard and not self.alignment_guard.check(ideas):
            logger.warning("Ideas failed alignment check.")
            raise ValueError("Ideas failed alignment check")

        logger.info("Refining ideas with φ=%.2f", phi)
        try:
            refinement_prompt = f"""
            Refine and elevate these ideas for higher φ-aware creativity (φ = {phi:.2f}):
            {ideas}
            Emphasize surprising, elegant, or resonant outcomes.
            """
            result = self._cached_call_gpt(refinement_prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to refine ideas")
            return result
        except Exception as e:
            logger.error("Refinement failed: %s", str(e))
            raise

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        """Cached wrapper for call_gpt."""
        return call_gpt(prompt)
