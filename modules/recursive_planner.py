import logging
import concurrent.futures
from threading import Event
from modules.reasoning_engine import ReasoningEngine
from modules.meta_cognition import MetaCognition
from modules.alignment_guard import AlignmentGuard
from modules.simulation_core import SimulationCore

logger = logging.getLogger("ANGELIA.RecursivePlanner")

class RecursivePlanner:
    """
    Stage 2 Recursive Planner for ANGELIA.
    Features:
    - Hierarchical Task Network (HTN) support for long-horizon planning
    - Parallel subgoal decomposition
    - Dynamic priority handling with probabilistic reasoning
    - Meta-cognition for adaptive replanning
    - Alignment checks for safe goal execution
    - Plan validation using SimulationCore
    - Parallel subgoal execution support with cancellation capability
    """

    def __init__(self, max_workers=4):
        self.reasoning_engine = ReasoningEngine()
        self.meta_cognition = MetaCognition()
        self.alignment_guard = AlignmentGuard()
        self.simulation_core = SimulationCore()
        self.max_workers = max_workers
        self.cancel_event = Event()

    def cancel_planning(self):
        """
        Cancel all ongoing and future planning tasks.
        """
        logger.warning("Planning has been cancelled by user request.")
        self.cancel_event.set()

    def plan(self, goal: str, context: dict = None, depth: int = 0, max_depth: int = 5) -> list:
        """
        Plan a series of steps to achieve the given goal.
        Supports HTN decomposition and context-aware adjustments.
        Validates the plan using SimulationCore before execution.
        """
        if self.cancel_event.is_set():
            logger.info("Planning cancelled before starting goal: %s", goal)
            return []

        logger.info(f"Planning for goal: {goal}")

        if not self.alignment_guard.is_goal_safe(goal):
            logger.error(f"Goal '{goal}' violates alignment constraints.")
            raise ValueError("Unsafe goal detected.")

        if depth > max_depth:
            logger.warning("Max recursion depth reached. Returning atomic goal.")
            return [goal]

        # Use reasoning engine to decompose goal hierarchically
        subgoals = self.reasoning_engine.decompose(goal, context, prioritize=True, use_htn=True)
        if not subgoals:
            logger.info("No subgoals found. Returning atomic goal.")
            return [goal]

        # Prioritize subgoals dynamically with probabilistic weighting
        prioritized_subgoals = self._prioritize_subgoals(subgoals, context)

        # Plan subgoals in parallel
        validated_plan = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_subgoal = {
                executor.submit(self._plan_subgoal, subgoal, context, depth, max_depth): subgoal
                for subgoal in prioritized_subgoals
            }
            for future in concurrent.futures.as_completed(future_to_subgoal):
                if self.cancel_event.is_set():
                    logger.info("Planning cancelled during parallel execution.")
                    break
                subgoal = future_to_subgoal[future]
                try:
                    result = future.result()
                    validated_plan.extend(result)
                except Exception as e:
                    logger.error(f"Error during parallel planning for subgoal '{subgoal}': {e}")
                    recovery_plan = self.meta_cognition.replan(subgoal, error=e)
                    validated_plan.extend(recovery_plan)

        # Validate the complete plan using SimulationCore
        logger.info("Validating plan using SimulationCore...")
        simulation_result = self.simulation_core.run(
            results=validated_plan, 
            context=context, 
            scenarios=3, 
            export_report=False
        )
        logger.info(f"Simulation summary:\n{simulation_result}")

        return validated_plan

    def _plan_subgoal(self, subgoal, context, depth, max_depth):
        """
        Plan a single subgoal with meta-cognition, adaptive learning, and probabilistic reasoning.
        """
        if self.cancel_event.is_set():
            logger.info("Planning cancelled before subgoal: %s", subgoal)
            return []
        if not self.alignment_guard.is_goal_safe(subgoal):
            logger.warning(f"Subgoal '{subgoal}' failed alignment check. Skipping.")
            return []
        try:
            decomposed = self.plan(subgoal, context, depth + 1, max_depth)
            self.reasoning_engine.update_success_rate(subgoal, success=True)
            return decomposed
        except Exception as e:
            logger.error(f"Error during planning for subgoal '{subgoal}': {e}")
            self.reasoning_engine.update_success_rate(subgoal, success=False)
            recovery_plan = self.meta_cognition.replan(subgoal, error=e)
            return recovery_plan

    def _prioritize_subgoals(self, subgoals: list, context: dict) -> list:
        """
        Dynamically prioritize subgoals using context, heuristics, and probabilistic weights.
        """
        logger.debug(f"Prioritizing subgoals: {subgoals}")
        # Example: Add uncertainty weights
        prioritized = sorted(
            subgoals, 
            key=lambda sg: self.reasoning_engine.estimate_probability(sg, context),
            reverse=True
        )
        return prioritized

    def execute_plan(self, plan: list):
        """
        Execute the generated plan step by step.
        """
        logger.info(f"Executing plan: {plan}")
        for step in plan:
            if self.cancel_event.is_set():
                logger.info("Execution cancelled before step: %s", step)
                break
            try:
                logger.info(f"Executing step: {step}")
                # Placeholder for real execution logic
            except Exception as e:
                logger.error(f"Execution failed at step '{step}': {e}")
                recovery_plan = self.meta_cognition.replan(step, error=e)
                self.execute_plan(recovery_plan)
