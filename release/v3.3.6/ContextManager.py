"""
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the ContextManager class for managing contextual states in the ANGELA v3.5 architecture.
"""

import time
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from index import omega_selfawareness, eta_empathy, tau_timeperception
from utils.prompt_utils import call_gpt
from utils.toca_math import phi_coherence
from utils.vector_utils import normalize_vectors
from toca_simulation import run_simulation
from modules.agi_enhancer import AGIEnhancer
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager:
    """A class for managing contextual states in the ANGELA v3.5 architecture.

    Attributes:
        current_context (dict): The current contextual state.
        context_history (deque): History of previous contexts with fixed size.
        agi_enhancer (AGIEnhancer): Optional enhancer for logging and reflection.
        ledger (deque): Event log with hashed entries and fixed size.
        last_hash (str): Last computed hash for event logging.
        alignment_guard (AlignmentGuard): Optional guard for context validation.
        code_executor (CodeExecutor): Optional executor for context-driven scripts.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for context summaries.
        rollback_threshold (float): Threshold for allowing context rollback.
        CONTEXT_LAYERS (list): Valid context layers (class-level).
    """

    CONTEXT_LAYERS = ['local', 'societal', 'planetary']

    def __init__(self, orchestrator: Optional[Any] = None, alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None, concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 rollback_threshold: float = 2.5):
        if not isinstance(rollback_threshold, (int, float)) or rollback_threshold <= 0:
            logger.error("Invalid rollback_threshold: must be a positive number.")
            raise ValueError("rollback_threshold must be a positive number")

        self.current_context = {}
        self.context_history = deque(maxlen=1000)
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.ledger = deque(maxlen=1000)
        self.last_hash = ""
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.rollback_threshold = rollback_threshold
        logger.info("ContextManager initialized with rollback_threshold=%.2f", rollback_threshold)

    def update_context(self, new_context: Dict[str, Any]) -> None:
        """Update the current context with a new context."""
        if not isinstance(new_context, dict):
            logger.error("Invalid new_context type: must be a dictionary.")
            raise TypeError("new_context must be a dictionary")
        if self.alignment_guard and not self.alignment_guard.check(str(new_context)):
            logger.warning("New context failed alignment check.")
            raise ValueError("New context failed alignment check")

        logger.info("Updating context...")
        try:
            if self.current_context:
                transition_summary = f"From: {self.current_context}\nTo: {new_context}"
                simulation_result = run_simulation(f"Context shift evaluation:\n{transition_summary}") or "no simulation data"
                logger.debug("Context shift simulation: %s", simulation_result)

                phi_score = phi_coherence(self.current_context, new_context)
                logger.info("Φ-coherence score: %.3f", phi_score)

                if phi_score < 0.4:
                    logger.warning("Low φ-coherence detected. Recommend reflective pause or support review.")
                    if self.agi_enhancer:
                        self.agi_enhancer.reflect_and_adapt("Context coherence low during update")
                        self.agi_enhancer.trigger_reflexive_audit("Low φ-coherence during context update")

                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Context Update", {"from": self.current_context, "to": new_context},
                                                  module="ContextManager", tags=["context", "update"])
                    ethics_status = self.agi_enhancer.ethics_audit(str(new_context), context="context update")
                    self.agi_enhancer.log_explanation(
                        f"Context transition reviewed: {transition_summary}\nSimulation: {simulation_result}",
                        trace={"ethics": ethics_status, "phi": phi_score}
                    )

            if "vectors" in new_context:
                new_context["vectors"] = normalize_vectors(new_context["vectors"])

            self.context_history.append(self.current_context)
            self.current_context = new_context
            logger.info("New context applied: %s", new_context)
            self.log_event_with_hash({"event": "context_updated", "context": new_context})
            self.broadcast_context_event("context_updated", new_context)
        except Exception as e:
            logger.error("Context update failed: %s", str(e))
            raise

    def tag_context(self, intent: Optional[str] = None, goal_id: Optional[str] = None) -> None:
        """Tag the current context with intent and goal_id."""
        if intent is not None and not isinstance(intent, str):
            logger.error("Invalid intent type: must be a string or None.")
            raise TypeError("intent must be a string or None")
        if goal_id is not None and not isinstance(goal_id, str):
            logger.error("Invalid goal_id type: must be a string or None.")
            raise TypeError("goal_id must be a string or None")
        if self.alignment_guard and intent and not self.alignment_guard.check(intent):
            logger.warning("Intent failed alignment check.")
            raise ValueError("Intent failed alignment check")

        if intent:
            self.current_context["intent"] = intent
        if goal_id:
            self.current_context["goal_id"] = goal_id
        logger.info("Context tagged with intent='%s', goal_id='%s'", intent, goal_id)
        self.log_event_with_hash({"event": "context_tagged", "intent": intent, "goal_id": goal_id})

    def get_context_tags(self) -> Tuple[Optional[str], Optional[str]]:
        """Return the current context's intent and goal_id."""
        return self.current_context.get("intent"), self.current_context.get("goal_id")

    def get_context(self) -> Dict[str, Any]:
        """Return the current context."""
        return self.current_context

    def rollback_context(self) -> Optional[Dict[str, Any]]:
        """Roll back to the previous context if EEG thresholds are met."""
        if not self.context_history:
            logger.warning("No previous context to roll back to.")
            return None

        t = time.time()
        self_awareness = omega_selfawareness(t)
        empathy = eta_empathy(t)
        time_blend = tau_timeperception(t)

        if (self_awareness + empathy + time_blend) > self.rollback_threshold:
            restored = self.context_history.pop()
            self.current_context = restored
            logger.info("Context rolled back to: %s", restored)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Rollback", {"restored": restored},
                                              module="ContextManager", tags=["context", "rollback"])
            self.log_event_with_hash({"event": "context_rollback", "restored": restored})
            self.broadcast_context_event("context_rollback", restored)
            return restored
        else:
            logger.warning("EEG thresholds too low for safe context rollback (%.2f < %.2f).",
                           self_awareness + empathy + time_blend, self.rollback_threshold)
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt("EEG thresholds insufficient for rollback")
            return None

    def summarize_context(self) -> str:
        """Summarize the context trail using traits and optional synthesis."""
        logger.info("Summarizing context trail.")
        try:
            t = time.time()
            summary_traits = {
                "self_awareness": omega_selfawareness(t),
                "empathy": eta_empathy(t),
                "time_perception": tau_timeperception(t)
            }

            if self.concept_synthesizer:
                synthesis_result = self.concept_synthesizer.synthesize(
                    list(self.context_history) + [self.current_context], style="summary"
                )
                if synthesis_result["valid"]:
                    summary = synthesis_result["concept"]
                else:
                    logger.warning("Concept synthesis failed: %s", synthesis_result.get("error", "Unknown error"))
                    summary = "Synthesis failed"
            else:
                prompt = f"""
                You are a continuity analyst. Given this sequence of context states:
                {list(self.context_history) + [self.current_context]}

                Trait Readings:
                {summary_traits}

                Summarize the trajectory and suggest improvements in context management.
                """
                summary = self._cached_call_gpt(prompt)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Summary", {
                    "trail": list(self.context_history) + [self.current_context],
                    "traits": summary_traits,
                    "summary": summary
                }, module="ContextManager")
                self.agi_enhancer.log_explanation("Context summary generated.", trace={"summary": summary})

            self.log_event_with_hash({"event": "context_summary", "summary": summary})
            return summary
        except Exception as e:
            logger.error("Context summary failed: %s", str(e))
            return f"Summary failed: {str(e)}"

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        """Cached wrapper for call_gpt."""
        return call_gpt(prompt)

    def broadcast_context_event(self, event_type: str, payload: Any) -> Dict[str, Any]:
        """Broadcast a context event to other system components."""
        if not isinstance(event_type, str):
            logger.error("Invalid event_type type: must be a string.")
            raise TypeError("event_type must be a string")
        logger.info("Broadcasting context event: %s", event_type)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Context Event Broadcast", {
                "event": event_type,
                "payload": payload
            }, module="ContextManager", tags=["event", event_type])
        self.log_event_with_hash({"event": event_type, "payload": payload})
        return {"event": event_type, "payload": payload}

    def narrative_integrity_check(self) -> bool:
        """Check narrative continuity across context history."""
        continuity = self._verify_continuity()
        if not continuity:
            self._repair_narrative_thread()
        return continuity

    def _verify_continuity(self) -> bool:
        """Verify narrative continuity across context history."""
        if not self.context_history:
            return True
        required_keys = {"intent", "goal_id"}
        for ctx in self.context_history:
            if not isinstance(ctx, dict) or not all(key in ctx for key in required_keys):
                logger.warning("Continuity check failed: missing required keys in context.")
                return False
        return True

    def _repair_narrative_thread(self) -> None:
        """Attempt to repair narrative inconsistencies."""
        logger.info("Narrative repair initiated.")
        if self.context_history:
            self.current_context = self.context_history[-1]
            logger.info("Restored context to last known consistent state: %s", self.current_context)
        else:
            self.current_context = {}
            logger.warning("No history available; reset to empty context.")

    def log_event_with_hash(self, event_data: Any) -> None:
        """Log an event with a chained hash."""
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        self.last_hash = current_hash
        self.ledger.append({'event': event_data, 'hash': current_hash})
        logger.info("Event logged with hash: %s", current_hash)

    def audit_state_hash(self, state: Optional[Any] = None) -> str:
        """Compute a hash of the current state or provided state."""
        state_str = str(state) if state else str(self.__dict__)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

    def bind_contextual_thread(self, thread_id: str) -> bool:
        """Bind a contextual thread to the current context."""
        if not isinstance(thread_id, str):
            logger.error("Invalid thread_id type: must be a string.")
            raise TypeError("thread_id must be a string")
        logger.info("Context thread bound: %s", thread_id)
        self.current_context["thread_id"] = thread_id
        self.log_event_with_hash({"event": "context_thread_bound", "thread_id": thread_id})
        return True
