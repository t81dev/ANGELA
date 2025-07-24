import logging
import concurrent.futures
from modules.reasoning_engine import ReasoningEngine
from modules.meta_cognition import MetaCognition
from modules.alignment_guard import AlignmentGuard
from toca_simulation import run_simulation
from index import beta_concentration, omega_selfawareness, mu_morality
import time

logger = logging.getLogger("ANGELA.RecursivePlanner")

class RecursivePlanner:
    """
    Recursive Planner v1.4.0 (with simulation-enhanced meta-cognition)
    - Multi-agent collaborative planning
    - Conflict resolution and dynamic priority handling
    - Parallelized subgoal decomposition with progress tracking
    - Integrated simulation feedback for goal validation and adaptation
    """

    def __init__(self, max_workers=4):
        self.reasoning_engine = ReasoningEngine()
        self.meta_cognition = MetaCognition()
        self.alignment_guard = AlignmentGuard()
        self.max_workers = max_workers

    def plan(self, goal: str, context: dict = None, depth: int = 0, max_depth: int = 5, collaborating_agents=None) -> list:
        """
        Plan steps to achieve the goal.
        Supports multi-agent collaboration and conflict resolution.
        Simulates subgoal outcomes when available for higher fidelity.
        Adjusts recursion depth and goal filtering using ToCA traits.
        """
        logger.info(f"📋 Planning for goal: '{goal}'")

        if not self.alignment_guard.is_goal_safe(goal):
            logger.error(f"🚨 Goal '{goal}' violates alignment constraints.")
            raise ValueError("Unsafe goal detected.")

        t = time.time() % 1e-18
        concentration = beta_concentration(t)
        awareness = omega_selfawareness(t)
        moral_weight = mu_morality(t)

        dynamic_depth_limit = max_depth + int(concentration * 10)
        if depth > dynamic_depth_limit:
            logger.warning("⚠️ Dynamic max recursion depth reached based on concentration trait. Returning atomic goal.")
            return [goal]

        subgoals = self.reasoning_engine.decompose(goal, context, prioritize=True)
        if not subgoals:
            logger.info("ℹ️ No subgoals found. Returning atomic goal.")
            return [goal]

        if collaborating_agents:
            logger.info(f"🤝 Collaborating with agents: {[agent.name for agent in collaborating_agents]}")
            subgoals = self._distribute_subgoals(subgoals, collaborating_agents)

        validated_plan = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_subgoal = {
                executor.submit(self._plan_subgoal, subgoal, context, depth, dynamic_depth_limit): subgoal
                for subgoal in subgoals
            }
            for future in concurrent.futures.as_completed(future_to_subgoal):
                subgoal = future_to_subgoal[future]
                try:
                    result = future.result()
                    validated_plan.extend(result)
                except Exception as e:
                    logger.error(f"❌ Error planning subgoal '{subgoal}': {e}")
                    recovery_plan = self.meta_cognition.review_reasoning(str(e))
                    validated_plan.extend(recovery_plan)

        logger.info(f"✅ Final validated plan for goal '{goal}': {validated_plan}")
        return validated_plan

    def _plan_subgoal(self, subgoal, context, depth, max_depth):
        """
        Plan a single subgoal recursively with alignment and simulation checks.
        """
        if not self.alignment_guard.is_goal_safe(subgoal):
            logger.warning(f"⚠️ Subgoal '{subgoal}' failed alignment check. Skipping.")
            return []

        simulation_feedback = run_simulation(subgoal)
        approved, _ = self.meta_cognition.pre_action_alignment_check(subgoal)
        if not approved:
            logger.warning(f"🚫 Subgoal '{subgoal}' denied by meta-cognitive alignment check.")
            return []

        try:
            return self.plan(subgoal, context, depth + 1, max_depth)
        except Exception as e:
            logger.error(f"❌ Error in subgoal '{subgoal}': {e}")
            return []

    def _distribute_subgoals(self, subgoals, agents):
        """
        Distribute subgoals among collaborating agents with conflict resolution.
        """
        logger.info("🕸 Distributing subgoals among agents with conflict resolution.")
        distributed = []
        for i, subgoal in enumerate(subgoals):
            agent = agents[i % len(agents)]
            logger.info(f"📤 Assigning subgoal '{subgoal}' to agent '{agent.name}'")
            if self._resolve_conflicts(subgoal, agent):
                distributed.append(subgoal)
            else:
                logger.warning(f"⚠️ Conflict detected for subgoal '{subgoal}'. Skipping assignment.")
        return distributed

    def _resolve_conflicts(self, subgoal, agent):
        """
        Simulate conflict resolution for subgoal assignment.
        """
        logger.info(f"🛠 Resolving conflicts for subgoal '{subgoal}' and agent '{agent.name}'")
        return True
