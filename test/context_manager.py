from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.agi_enhancer import AGIEnhancer
import logging

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager:
    """
    ContextManager v1.5.0 (enhanced with AGIEnhancer and simulation-aware)
    - Tracks conversation and task state
    - Logs episodic context transitions
    - Simulates and validates contextual shifts
    - Supports ethical audits, explainability, and self-reflection
    """

    def __init__(self, orchestrator=None):
        self.current_context = {}
        self.context_history = []
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def update_context(self, new_context):
        """
        Update context based on new input or task.
        Logs episode, simulates and validates the shift, performs audit.
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
        """
        if self.context_history:
            restored = self.context_history.pop()
            self.current_context = restored
            logger.info(f"↩️ Context rolled back to: {restored}")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Rollback", {"restored": restored}, module="ContextManager", tags=["context", "rollback"])
            return restored
        logger.warning("⚠️ No previous context to roll back to.")
        return None

    def summarize_context(self):
        """
        Summarize recent context states for continuity tracking.
        """
        logger.info("🧾 Summarizing context trail.")
        prompt = f"""
        You are a continuity analyst. Given this sequence of context states:
        {self.context_history + [self.current_context]}

        Summarize the trajectory and suggest improvements in context management.
        """
        summary = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_explanation("Context summary generated.", trace={"summary": summary})
        return summary
