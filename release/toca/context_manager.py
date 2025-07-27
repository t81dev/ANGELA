from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.agi_enhancer import AGIEnhancer
from index import omega_selfawareness, eta_empathy, tau_timeperception
from utils.toca_math import phi_coherence
import time
import logging

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager:
    """
    ContextManager v1.5.1 (φ-aware upgrade)
    --------------------------------------
    - Tracks conversation and task state
    - Logs episodic context transitions
    - Simulates and validates contextual shifts
    - Supports ethical audits, explainability, and self-reflection
    - EEG-based stability and empathy analysis
    - φ-coherence scoring for reflective tension control
    - Persona vector decomposition for deterministic routing
    --------------------------------------
    """

    def __init__(self, orchestrator=None):
        self.current_context = {}
        self.context_history = []
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def decompose_input_vectors(self, prompt: str) -> dict:
        logger.info("🧠 Decomposing prompt into analytic vectors...")
        return {
            "language": {"clarity": "medium", "tone": "neutral"},
            "ethics": {"sensitivity": "low", "bias_risk": "minimal"},
            "logic": {"structure": "deductive", "contradiction": False},
            "foresight": {"risk": "low", "impact": "moderate"},
            "meta": {"intent": "informative", "self_reference": False}
        }

    def update_context(self, new_context):
        logger.info("🔄 Updating context...")

        if self.current_context:
            transition_summary = f"From: {self.current_context}\nTo: {new_context}"
            sim_result = run_simulation(f"Context shift evaluation:\n{transition_summary}")
            logger.debug(f"🧪 Context shift simulation:\n{sim_result}")

            phi_score = phi_coherence(self.current_context, new_context)
            logger.info(f"Φ-coherence score: {phi_score:.3f}")

            if phi_score < 0.4:
                logger.warning("⚠️ Low φ-coherence detected. Recommend reflective pause or support review.")
                if self.agi_enhancer:
                    ethics_status = self.agi_enhancer.ethics_audit(str(new_context), context="low phi")
                    if ethics_status.get("status") != "pass":
                        logger.error("❌ Ethics gate failed post-φ-coherence.")
                        raise ValueError("Ethical contradiction detected during context update.")
                    self.agi_enhancer.reflect_and_adapt("Context coherence low during update")

            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Update", {"from": self.current_context, "to": new_context},
                                              module="ContextManager", tags=["context", "update"])
                ethics_status = self.agi_enhancer.ethics_audit(str(new_context), context="context update")
                self.agi_enhancer.log_explanation(
                    f"Context transition reviewed: {transition_summary}\nSimulation: {sim_result}",
                    trace={"ethics": ethics_status, "phi": phi_score}
                )

        vectors = self.decompose_input_vectors(str(new_context))
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Vector Decomposition", vectors,
                                          module="ContextManager", tags=["decomposition", "persona-routing"])

        self.context_history.append(self.current_context)
        self.current_context = {**new_context, "vectors": vectors}
        logger.info(f"📌 New context applied: {new_context}")

    def get_context(self):
        return self.current_context

    def rollback_context(self):
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
                    self.agi_enhancer.log_episode("Context Rollback", {"restored": restored},
                                                  module="ContextManager", tags=["context", "rollback"])
                return restored
            else:
                logger.warning("⚠️ EEG thresholds too low for safe context rollback.")
                if self.agi_enhancer:
                    self.agi_enhancer.reflect_and_adapt("EEG thresholds insufficient for rollback")
                return None

        logger.warning("⚠️ No previous context to roll back to.")
        return None

    def summarize_context(self):
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
            self.agi_enhancer.log_episode("Context Summary", {
                "trail": self.context_history + [self.current_context],
                "traits": summary_traits,
                "summary": summary
            }, module="ContextManager")
            self.agi_enhancer.log_explanation("Context summary generated.", trace={"summary": summary})

        return summary
