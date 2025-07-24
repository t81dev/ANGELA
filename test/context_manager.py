from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager:
    """
    ContextManager v1.4.0 (simulation-aware)
    - Tracks conversation and task state
    - Manages dynamic context switching
    - Simulates contextual shifts and validates transitions
    """

    def __init__(self):
        self.current_context = {}
        self.context_history = []

    def update_context(self, new_context):
        """
        Update context based on new input or task.
        Simulates and validates the contextual shift.
        """
        logger.info("🔄 Updating context...")
        if self.current_context:
            transition_summary = f"From: {self.current_context}\nTo: {new_context}"
            sim_result = run_simulation(f"Context shift evaluation:\n{transition_summary}")
            logger.debug(f"🧪 Context shift simulation:
{sim_result}")

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
        return call_gpt(prompt)
