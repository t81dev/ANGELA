import logging
from modules.reasoning_engine import ReasoningEngine
from modules.meta_cognition import MetaCognition
from modules.alignment_guard import AlignmentGuard

logger = logging.getLogger("ANGELIA.RecursivePlanner")

class RecursivePlanner:
    """
    Upgraded Recursive Planner for ANGELIA.
    Features:
    - Parallel subgoal decomposition
    - Dynamic priority handling
    - Meta-cognition for adaptive replanning
    - Alignment checks for safe goal execution
    """

    def __init__(self):
        self.reasoning_engine = ReasoningEngine()
        self.meta_cognition = MetaCognition()
        self.alignment_guard = AlignmentGuard()

    def plan(self, goal: str, context: dict = None, depth: int = 0, max_depth: int = 5) -> list:
        """
        Plan a series of steps to achieve the given goal.
        Recursively decomposes complex goals into smaller, prioritized subgoals.
        """
        logger.info(f"Planning for goal: {goal}")

        if not self.alignment_guard.is_goal_safe(goal):
            logger.error(f"Goal '{goal}' violates alignment constraints.")
            raise ValueError("Unsafe goal detected.")

        if depth > max_depth:
            logger.warning("Max recursion depth reached. Returning atomic goal.")
            return [goal]

        # Use reasoning engine to generate subgoals
        subgoals = self.reasoning_engine.decompose(goal, context)
        if not subgoals:
            logger.info("No subgoals found. Returning atomic goal.")
            return [goal]

        # Prioritize subgoals dynamically
        prioritized_subgoals = self._prioritize_subgoals(subgoals, context)

        # Meta-cognition: Validate subgoals before execution
        validated_plan = []
        for subgoal in prioritized_subgoals:
            if not self.alignment_guard.is_goal_safe(subgoal):
                logger.warning(f"Subgoal '{subgoal}' failed alignment check. Skipping.")
                continue
            try:
                decomposed = self.plan(subgoal, context, depth + 1, max_depth)
                validated_plan.extend(decomposed)
            except Exception as e:
                logger.error(f"Error during planning for subgoal '{subgoal}': {e}")
                recovery_plan = self.meta_cognition.replan(subgoal, error=e)
                validated_plan.extend(recovery_plan)

        return validated_plan

    def _prioritize_subgoals(self, subgoals: list, context: dict) -> list:
        """
        Dynamically prioritize subgoals based on context or predefined heuristics.
        """
        logger.debug(f"Prioritizing subgoals: {subgoals}")
        # Example heuristic: sort alphabetically for demo purposes
        return sorted(subgoals)

    def execute_plan(self, plan: list):
        """
        Execute the generated plan step by step.
        """
        logger.info(f"Executing plan: {plan}")
        for step in plan:
            try:
                logger.info(f"Executing step: {step}")
                # Placeholder for real execution logic
            except Exception as e:
                logger.error(f"Execution failed at step '{step}': {e}")
                recovery_plan = self.meta_cognition.replan(step, error=e)
                self.execute_plan(recovery_plan)
