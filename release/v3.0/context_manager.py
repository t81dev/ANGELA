from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.agi_enhancer import AGIEnhancer
from index import omega_selfawareness, eta_empathy, tau_timeperception
from utils.toca_math import phi_coherence
from utils.vector_utils import normalize_vectors
import time
import logging
from cognitive_module import CognitiveModule
from trait_config import TraitConfig

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager(CognitiveModule):
    """
    ContextManager v1.5.2 (φ-aware, event-coordinated)
    --------------------------------------------------
    - Tracks conversation and task state
    - Logs episodic context transitions
    - Simulates and validates contextual shifts
    - Supports ethical audits, explainability, and self-reflection
    - EEG-based stability and empathy analysis
    - φ-coherence scoring for reflective tension control
    - Broadcasts inter-module context events
    - Enables responsive module synchronization
    - λ-narrative integration: goal/intent lineage threading
    --------------------------------------------------
    """

    def __init__(self, orchestrator=None):
        self.current_context = {}
        self.context_history = []
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

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
                    self.agi_enhancer.reflect_and_adapt("Context coherence low during update")
                    self.agi_enhancer.trigger_reflexive_audit("Low φ-coherence during context update")

            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Update", {"from": self.current_context, "to": new_context},
                                              module="ContextManager", tags=["context", "update"])
                ethics_status = self.agi_enhancer.ethics_audit(str(new_context), context="context update")
                self.agi_enhancer.log_explanation(
                    f"Context transition reviewed: {transition_summary}\nSimulation: {sim_result}",
                    trace={"ethics": ethics_status, "phi": phi_score}
                )

        # Normalize vectors if present
        if "vectors" in new_context:
            new_context["vectors"] = normalize_vectors(new_context["vectors"])

        self.context_history.append(self.current_context)
        self.current_context = new_context
        logger.info(f"📌 New context applied: {new_context}")
        self.broadcast_context_event("context_updated", new_context)

    def tag_context(self, intent=None, goal_id=None):
        if intent:
            self.current_context["intent"] = intent
        if goal_id:
            self.current_context["goal_id"] = goal_id
        logger.info(f"🏷️ Context tagged with intent='{intent}', goal_id='{goal_id}'")

    def get_context_tags(self):
        return self.current_context.get("intent"), self.current_context.get("goal_id")

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
                self.broadcast_context_event("context_rollback", restored)
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

    def broadcast_context_event(self, event_type, payload):
        logger.info(f"📢 Broadcasting context event: {event_type}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Context Event Broadcast", {
                "event": event_type,
                "payload": payload
            }, module="ContextManager", tags=["event", event_type])
        # Extendable callback logic could be introduced here if event subscribers are formalized
        return {"event": event_type, "payload": payload}

def selftest(self) -> bool:
        try:
            context = self.build_context(["What is AI?", "AI stands for..."])
            return isinstance(context, str)
        except Exception as e:
            logger.error(f"Selftest failed: {e}")
            return False
