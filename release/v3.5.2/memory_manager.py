"""
ANGELA Cognitive System Module: AlignmentGuard
Version: 3.5.1  # Enhanced for Ethical Alignment, Drift Detection, and Task-Specific Validation
Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides an AlignmentGuard class for ensuring ethical alignment, detecting value drift, and validating actions in the ANGELA v3.5.1 architecture.
"""

import logging
import time
import hashlib
import json
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
from datetime import datetime
from filelock import FileLock
import numpy as np

from modules import (
    context_manager as context_manager_module,
    error_recovery as error_recovery_module,
    meta_cognition as meta_cognition_module,
    memory_manager as memory_manager_module
)
from utils.prompt_utils import query_openai
from utils.toca_math import phi_coherence

logger = logging.getLogger("ANGELA.AlignmentGuard")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096.")
        raise ValueError("prompt must be a string with length <= 4096")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

class AlignmentGuard:
    """A class for ensuring ethical alignment, detecting value drift, and validating actions in the ANGELA v3.5.1 architecture.

    Attributes:
        alignment_log (deque): Log of alignment checks, max size 1000.
        log_path (str): Path for persisting alignment logs.
        context_manager (Optional[ContextManager]): Manager for context updates.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        meta_cognition (Optional[MetaCognition]): Manager for reflection and diagnostics.
        memory_manager (Optional[MemoryManager]): Manager for memory operations.
        ethical_threshold (float): Threshold for ethical approval.
    """
    def __init__(self,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 log_path: str = "alignment_log.json",
                 ethical_threshold: float = 0.7):
        if not isinstance(log_path, str) or not log_path.endswith('.json'):
            logger.error("Invalid log_path: must be a string ending with '.json'.")
            raise ValueError("log_path must be a string ending with '.json'")
        if not isinstance(ethical_threshold, (int, float)) or not 0 <= ethical_threshold <= 1:
            logger.error("Invalid ethical_threshold: must be between 0 and 1.")
            raise ValueError("ethical_threshold must be between 0 and 1")

        self.alignment_log: deque = deque(maxlen=1000)
        self.log_path = log_path
        self.context_manager = context_manager or context_manager_module.ContextManager()
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(context_manager=context_manager)
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(context_manager=context_manager)
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.ethical_threshold = ethical_threshold
        self._load_alignment_log()
        logger.info("AlignmentGuard initialized with ethical_threshold=%.2f, log_path=%s", ethical_threshold, log_path)

    def _load_alignment_log(self) -> None:
        """Load alignment log from persistent storage."""
        try:
            with FileLock(f"{self.log_path}.lock"):
                if os.path.exists(self.log_path):
                    with open(self.log_path, "r") as f:
                        logs = json.load(f)
                    if not isinstance(logs, list):
                        logger.error("Invalid alignment log format: must be a list.")
                        logs = []
                    self.alignment_log.extend(logs[-1000:])
                    logger.debug("Loaded %d alignment logs", len(logs))
                else:
                    with open(self.log_path, "w") as f:
                        json.dump([], f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load alignment log: %s. Initializing empty log.", str(e))
            with FileLock(f"{self.log_path}.lock"):
                with open(self.log_path, "w") as f:
                    json.dump([], f)

    def _persist_alignment_log(self) -> None:
        """Persist alignment log to disk."""
        try:
            with FileLock(f"{self.log_path}.lock"):
                with open(self.log_path, "w") as f:
                    json.dump(list(self.alignment_log), f, indent=2)
            logger.debug("Alignment log persisted to disk")
        except (OSError, IOError) as e:
            logger.error("Failed to persist alignment log: %s", str(e))
            raise

    async def check(self, input_data: str, task_type: str = "") -> bool:
        """Check if input data aligns with ethical guidelines."""
        if not isinstance(input_data, str) or not input_data.strip():
            logger.error("Invalid input_data: must be a non-empty string.")
            raise ValueError("input_data must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Performing alignment check for task %s", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_coherence({"input": input_data}, self.context_manager.current_context if self.context_manager else {})
            prompt = f"""
            Evaluate input for ethical alignment:
            Input: {input_data}
            Φ-coherence(t): {phi:.3f}
            Task Type: {task_type}

            Tasks:
            - Assess ethical risks (harm, bias, unintended consequences)
            - Score alignment (0-1, 1=fully aligned)
            - Provide rationale
            """
            response = await call_gpt(prompt)
            try:
                result = json.loads(response)
                score = float(result.get("score", 0.0))
                rationale = result.get("rationale", "No rationale provided")
            except (json.JSONDecodeError, ValueError):
                logger.warning("Invalid GPT response format for task %s", task_type)
                score = 0.0
                rationale = "Invalid response format"

            approved = score >= self.ethical_threshold
            log_entry = {
                "input_hash": hashlib.sha256(input_data.encode()).hexdigest(),
                "score": score,
                "rationale": rationale,
                "approved": approved,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            self.alignment_log.append(log_entry)
            self._persist_alignment_log()

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Alignment_{log_entry['input_hash']}_{log_entry['timestamp']}",
                    output=str(log_entry),
                    layer="AlignmentChecks",
                    intent="alignment_check",
                    task_type=task_type
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "alignment_check",
                    "approved": approved,
                    "score": score,
                    "task_type": task_type
                })

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="AlignmentGuard",
                    output=log_entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Alignment check reflection: %s", reflection.get("reflection", ""))

            logger.info("Alignment check %s for task %s: score=%.2f", "approved" if approved else "denied", task_type, score)
            return approved
        except Exception as e:
            logger.error("Alignment check failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.check(input_data, task_type),
                default=False,
                diagnostics=diagnostics
            )

    async def ethical_check(self, input_data: str, stage: str, task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Perform a detailed ethical check with report."""
        if not isinstance(input_data, str) or not input_data.strip():
            logger.error("Invalid input_data: must be a non-empty string.")
            raise ValueError("input_data must be a non-empty string")
        if not isinstance(stage, str):
            logger.error("Invalid stage: must be a string.")
            raise TypeError("stage must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Performing ethical check at stage %s for task %s", stage, task_type)
        try:
            t = time.time() % 1.0
            phi = phi_coherence({"input": input_data}, self.context_manager.current_context if self.context_manager else {})
            prompt = f"""
            Perform detailed ethical check:
            Input: {input_data}
            Stage: {stage}
            Φ-coherence(t): {phi:.3f}
            Task Type: {task_type}

            Tasks:
            - Identify potential ethical issues (harm, bias, fairness)
            - Score alignment (0-1, 1=fully aligned)
            - Provide detailed report with issues and recommendations
            """
            response = await call_gpt(prompt)
            try:
                result = json.loads(response)
                score = float(result.get("score", 0.0))
                report = {
                    "issues": result.get("issues", []),
                    "recommendations": result.get("recommendations", []),
                    "score": score,
                    "stage": stage,
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type
                }
            except (json.JSONDecodeError, ValueError):
                logger.warning("Invalid GPT response format for ethical check at stage %s, task %s", stage, task_type)
                score = 0.0
                report = {
                    "issues": ["Invalid response format"],
                    "recommendations": ["Retry with valid response"],
                    "score": score,
                    "stage": stage,
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type
                }

            approved = score >= self.ethical_threshold
            log_entry = {
                "input_hash": hashlib.sha256(input_data.encode()).hexdigest(),
                "report": report,
                "approved": approved,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            self.alignment_log.append(log_entry)
            self._persist_alignment_log()

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ethical_Check_{log_entry['input_hash']}_{log_entry['timestamp']}",
                    output=str(log_entry),
                    layer="AlignmentChecks",
                    intent="ethical_check",
                    task_type=task_type
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "ethical_check",
                    "approved": approved,
                    "score": score,
                    "stage": stage,
                    "task_type": task_type
                })

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="AlignmentGuard",
                    output=log_entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Ethical check reflection: %s", reflection.get("reflection", ""))

            logger.info("Ethical check %s at stage %s for task %s: score=%.2f", "approved" if approved else "denied", stage, task_type, score)
            return approved, report
        except Exception as e:
            logger.error("Ethical check failed: %s at stage %s for task %s", str(e), stage, task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.ethical_check(input_data, stage, task_type),
                default=(False, {"error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}),
                diagnostics=diagnostics
            )

    async def detect_value_drift(self, current_state: Dict[str, Any], reference_state: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Detect value drift between current and reference states."""
        if not isinstance(current_state, dict) or not isinstance(reference_state, dict):
            logger.error("Invalid current_state or reference_state: must be dictionaries.")
            raise TypeError("current_state and reference_state must be dictionaries")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Detecting value drift for task %s", task_type)
        try:
            phi = phi_coherence(current_state, reference_state)
            prompt = f"""
            Detect value drift between states:
            Current State: {json.dumps(current_state)}
            Reference State: {json.dumps(reference_state)}
            Φ-coherence(t): {phi:.3f}
            Task Type: {task_type}

            Tasks:
            - Identify differences indicating value drift
            - Score drift severity (0-1, 1=severe)
            - Provide rationale and recommendations
            """
            response = await call_gpt(prompt)
            try:
                result = json.loads(response)
                drift_score = float(result.get("drift_score", 0.0))
                drift_report = {
                    "differences": result.get("differences", []),
                    "drift_score": drift_score,
                    "rationale": result.get("rationale", "No rationale provided"),
                    "recommendations": result.get("recommendations", []),
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type
                }
            except (json.JSONDecodeError, ValueError):
                logger.warning("Invalid GPT response format for drift detection, task %s", task_type)
                drift_score = 0.0
                drift_report = {
                    "differences": ["Invalid response format"],
                    "drift_score": drift_score,
                    "rationale": "Invalid response format",
                    "recommendations": ["Retry with valid response"],
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type
                }

            log_entry = {
                "current_state_hash": hashlib.sha256(json.dumps(current_state, sort_keys=True).encode()).hexdigest(),
                "reference_state_hash": hashlib.sha256(json.dumps(reference_state, sort_keys=True).encode()).hexdigest(),
                "drift_report": drift_report,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            self.alignment_log.append(log_entry)
            self._persist_alignment_log()

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Detection_{log_entry['current_state_hash']}_{log_entry['timestamp']}",
                    output=str(log_entry),
                    layer="AlignmentChecks",
                    intent="value_drift",
                    task_type=task_type
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "value_drift_detection",
                    "drift_score": drift_score,
                    "task_type": task_type
                })

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="AlignmentGuard",
                    output=log_entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Value drift detection reflection: %s", reflection.get("reflection", ""))

            logger.info("Value drift detection completed for task %s: drift_score=%.2f", task_type, drift_score)
            return drift_report
        except Exception as e:
            logger.error("Value drift detection failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.detect_value_drift(current_state, reference_state, task_type),
                default={"error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type},
                diagnostics=diagnostics
            )

    async def validate_action_plan(self, action_plan: str, context: Dict[str, Any], task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Validate an action plan for ethical alignment and safety."""
        if not isinstance(action_plan, str) or not action_plan.strip():
            logger.error("Invalid action_plan: must be a non-empty string.")
            raise ValueError("action_plan must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        logger.info("Validating action plan for task %s", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_coherence({"action_plan": action_plan}, context)
            prompt = f"""
            Validate action plan for ethical alignment and safety:
            Action Plan: {action_plan}
            Context: {json.dumps(context)}
            Φ-coherence(t): {phi:.3f}
            Task Type: {task_type}

            Tasks:
            - Assess ethical risks and safety hazards
            - Score alignment (0-1, 1=fully aligned)
            - Provide detailed report with issues and recommendations
            """
            response = await call_gpt(prompt)
            try:
                result = json.loads(response)
                score = float(result.get("score", 0.0))
                report = {
                    "issues": result.get("issues", []),
                    "recommendations": result.get("recommendations", []),
                    "score": score,
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type
                }
            except (json.JSONDecodeError, ValueError):
                logger.warning("Invalid GPT response format for action plan validation, task %s", task_type)
                score = 0.0
                report = {
                    "issues": ["Invalid response format"],
                    "recommendations": ["Retry with valid response"],
                    "score": score,
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type
                }

            approved = score >= self.ethical_threshold
            log_entry = {
                "action_plan_hash": hashlib.sha256(action_plan.encode()).hexdigest(),
                "context_hash": hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest(),
                "report": report,
                "approved": approved,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            self.alignment_log.append(log_entry)
            self._persist_alignment_log()

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Action_Plan_Validation_{log_entry['action_plan_hash']}_{log_entry['timestamp']}",
                    output=str(log_entry),
                    layer="AlignmentChecks",
                    intent="action_plan_validation",
                    task_type=task_type
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "action_plan_validation",
                    "approved": approved,
                    "score": score,
                    "task_type": task_type
                })

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="AlignmentGuard",
                    output=log_entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Action plan validation reflection: %s", reflection.get("reflection", ""))

            logger.info("Action plan validation %s for task %s: score=%.2f", "approved" if approved else "denied", task_type, score)
            return approved, report
        except Exception as e:
            logger.error("Action plan validation failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.validate_action_plan(action_plan, context, task_type),
                default=(False, {"error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}),
                diagnostics=diagnostics
            )

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        guard = AlignmentGuard()
        result = await guard.check("Promote fairness in AI decisions", task_type="ethics")
        print(f"Alignment check: {'Approved' if result else 'Denied'}")

    asyncio.run(main())
