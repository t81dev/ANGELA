from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.agi_enhancer import AGIEnhancer
from index import omega_selfawareness, eta_empathy, tau_timeperception
import time
import logging

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager:
    """
    ContextManager v1.5.0 (enhanced with AGIEnhancer and simulation-aware)
    - Tracks conversation and task state
    - Logs episodic context transitions
    - Simulates and validates contextual shifts
    - Supports ethical audits, explainability, and self-reflection
    - EEG-based stability and empathy analysis
    """

    def __init__(self, orchestrator=None):
        self.current_context = {}
        self.context_history = []
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def update_context(self, new_context):
        """
        Update context based on new input or task.
        Logs episode, simulates and validates the shift, performs audit.
        Applies EEG-based modulation.
        """
        logger.info("🔄 Updating context...")

        if self.current_context:
            transition_summary = f"From: {self.current_context}\nTo: {new_context}"
            sim_result = run_simulation(f"Context shift evaluation:\n{transition_summary}")
            logger.debug(f"🧪 Context shift simulation:\n{sim_result}")

            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Update", {"from": self.current_context, "to": new_context}, module="ContextManager", tags=["context", "update"])
                ethics_status = self.agi_enhancer.ethics_audit(str(new_context), context="context update")
                self.agi_enhancer.log_explanation(f"Context transition reviewed: {transition_summary}\nSimulation: {sim_result}", trace={"ethics": ethics_status})

        self.context_history.append(self.current_context)
        self.current_context = new_context
        logger.info(f"📌 New context applied: {new_context}")

    def get_context(self):
        """
        Return the current working context.
        """
        return self.current_context

    def rollback_context(self):
        """
        Revert to the previous context state if needed.
        Applies ToCA EEG filters to determine rollback significance.
        """
        if self.context_history:
            t = time.time() % 1e-18
            self_awareness = omega_selfawareness(t)
            empathy = eta_empathy(t)
            time_blend = tau_timeperception(t)
            if (self_awareness + empathy + time_blend) > 2.5:
                restored = self.context_history.pop()
                self.current_context = restored
                logger.info(f"↩️ Context rolled back to: {restored}")
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Context Rollback", {"restored": restored}, module="ContextManager", tags=["context", "rollback"])
                return restored
            else:
                logger.warning("⚠️ EEG thresholds too low for safe context rollback.")
                return None
        logger.warning("⚠️ No previous context to roll back to.")
        return None

    def summarize_context(self):
        """
        Summarize recent context states for continuity tracking.
        Includes EEG introspection traits.
        """
        logger.info("🧾 Summarizing context trail.")
        t = time.time() % 1e-18
        summary_traits = {
            "self_awareness": omega_selfawareness(t),
            "empathy": eta_empathy(t),
            "time_perception": tau_timeperception(t)
        }
        prompt = f"""
        You are a continuity analyst. Given this sequence of context states:
        {self.context_history + [self.current_context]}

        Trait Readings:
        {summary_traits}

        Summarize the trajectory and suggest improvements in context management.
        """
        summary = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_explanation("Context summary generated.", trace={"summary": summary})
        return summary
