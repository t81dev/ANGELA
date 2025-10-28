from __future__ import annotations
import logging
import time
import asyncio
import math
from typing import List, Dict, Any, Optional, Protocol
from threading import Lock
from functools import lru_cache

# ANGELA modules
from modules import (
    reasoning_engine as reasoning_engine_module,
    meta_cognition as meta_cognition_module,
    alignment_guard as alignment_guard_module,
    simulation_core as simulation_core_module,
    memory_manager as memory_manager_module,
)

logger = logging.getLogger("ANGELA.RecursivePlanner")

class AgentProtocol(Protocol):
    name: str
    def process_subgoal(self, subgoal: str) -> Any: ...

class RecursivePlanner:
    """
    Handles recursive goal planning with trait-based modulation and reflection.
    """
    def __init__(self, **services):
        self.services = services
        self.omega: Dict[str, Any] = {"timeline": [], "traits": {}}
        self.omega_lock = Lock()
        logger.info("RecursivePlanner v5.1.0 initialized")

    # --- Planning ---
    async def plan(self, goal: str, context: Dict, depth: int = 0, max_depth: int = 5) -> List[str]:
        """Recursively decomposes and plans a goal."""
        if depth > max_depth:
            return [goal]

        subgoals = await self._decompose_goal(goal, context)
        if not subgoals:
            return [goal]

        tasks = [self.plan(sg, context, depth + 1, max_depth) for sg in subgoals]
        results = await asyncio.gather(*tasks)

        flat_plan = [step for plan in results for step in plan]
        return await self._reflect_and_refine(flat_plan, context)

    async def _decompose_goal(self, goal: str, context: Dict) -> List[str]:
        """Decomposes a goal into subgoals using the reasoning engine."""
        reasoning_engine = self.services.get("reasoning_engine")
        if reasoning_engine:
            return await reasoning_engine.decompose(goal, context)
        return []

    async def _reflect_and_refine(self, plan: List[str], context: Dict) -> List[str]:
        """Reflects on a plan and refines it using meta-cognition."""
        meta_cognition = self.services.get("meta_cognition")
        if meta_cognition:
            reflection = await meta_cognition.reflect_on_output("RecursivePlanner", plan, context)
            # In a real scenario, you'd parse the reflection and modify the plan
        return plan

    # --- Trait-Based Adjustments ---
    def adjust_plan_depth(self, trait_weights: Dict[str, float], task_type: str) -> int:
        """Adjusts planning depth based on traits and task type."""
        base_depth = 2 if trait_weights.get("omega", 0.0) > 0.7 else 1
        if "recursion" in task_type:
            return min(base_depth + 1, 3)
        return base_depth

    # --- Omega State ---
    def _update_omega(self, subgoal: str, result: List[str], error: bool = False) -> None:
        """Updates the shared omega state with the outcome of a subgoal."""
        with self.omega_lock:
            self.omega["timeline"].append({
                "subgoal": subgoal,
                "result": result,
                "timestamp": time.time(),
                "error": error,
            })
