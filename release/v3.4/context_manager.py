"""
ANGELA Cognitive System Module: ContextManager
Refactored Version: 3.4.0  # Enhanced for Coordination Analytics, Visualization, and Drift Mitigation
Refactor Date: 2025-08-06
Maintainer: ANGELA System Framework

This module provides a ContextManager class for managing contextual states, event logging, coordination analytics,
and visualization in the ANGELA v3.5 architecture, with support for agent coordination and ontology drift mitigation.
"""

import time
import logging
import hashlib
import json
import os
import math
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, Counter
from filelock import FileLock
import asyncio
from functools import lru_cache
from datetime import datetime

from modules import (
    agi_enhancer as agi_enhancer_module,
    alignment_guard as alignment_guard_module,
    code_executor as code_executor_module,
    concept_synthesizer as concept_synthesizer_module,
    meta_cognition as meta_cognition_module
)
from utils.toca_math import phi_coherence
from utils.vector_utils import normalize_vectors
from toca_simulation import run_simulation
from index import omega_selfawareness, eta_empathy, tau_timeperception

logger = logging.getLogger("ANGELA.ContextManager")

@lru_cache(maxsize=100)
def eta_context_stability(t: float) -> float:
    """Trait function for context stability modulation."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.2), 1.0))

class ContextManager:
    """A class for managing contextual states, event logging, coordination analytics, and visualization in the ANGELA v3.5 architecture.

    Attributes:
        context_path (str): File path for context persistence.
        event_log_path (str): File path for event log persistence.
        coordination_log_path (str): File path for coordination log persistence.
        current_context (Dict[str, Any]): Current contextual state.
        context_history (deque): History of previous contexts, max size 1000.
        event_log (deque): Log of events with hashes, max size 1000.
        coordination_log (deque): Log of agent coordination events, max size 1000.
        last_hash (str): Last computed hash for event chaining.
        agi_enhancer (AGIEnhancer): Enhancer for logging and reflection.
        alignment_guard (AlignmentGuard): Guard for ethical checks.
        code_executor (CodeExecutor): Executor for context-driven scripts.
        concept_synthesizer (ConceptSynthesizer): Synthesizer for context summaries.
        meta_cognition (MetaCognition): Manager for trait optimization and drift validation.
        rollback_threshold (float): Threshold for context rollback.
        CONTEXT_LAYERS (List[str]): Valid context layers (class-level).
    """
    CONTEXT_LAYERS = ['local', 'societal', 'planetary']

    def __init__(self, orchestrator: Optional[Any] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 code_executor: Optional['code_executor_module.CodeExecutor'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 context_path: str = "context_store.json",
                 event_log_path: str = "event_log.json",
                 coordination_log_path: str = "coordination_log.json",
                 rollback_threshold: float = 2.5):
        if not isinstance(context_path, str) or not context_path.endswith('.json'):
            logger.error("Invalid context_path: must be a string ending with '.json'.")
            raise ValueError("context_path must be a string ending with '.json'")
        if not isinstance(event_log_path, str) or not event_log_path.endswith('.json'):
            logger.error("Invalid event_log_path: must be a string ending with '.json'.")
            raise ValueError("event_log_path must be a string ending with '.json'")
        if not isinstance(coordination_log_path, str) or not coordination_log_path.endswith('.json'):
            logger.error("Invalid coordination_log_path: must be a string ending with '.json'.")
            raise ValueError("coordination_log_path must be a string ending with '.json'")
        if not isinstance(rollback_threshold, (int, float)) or rollback_threshold <= 0:
            logger.error("Invalid rollback_threshold: must be a positive number.")
            raise ValueError("rollback_threshold must be a positive number")

        self.context_path = context_path
        self.event_log_path = event_log_path
        self.coordination_log_path = coordination_log_path
        self.current_context = {}
        self.context_history = deque(maxlen=1000)
        self.event_log = deque(maxlen=1000)
        self.coordination_log = deque(maxlen=1000)
        self.last_hash = ""
        self.agi_enhancer = agi_enhancer_module.AGIEnhancer(orchestrator) if orchestrator else None
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition
        self.rollback_threshold = rollback_threshold
        self.current_context = self._load_context()
        self._load_event_log()
        self._load_coordination_log()
        logger.info("ContextManager initialized with rollback_threshold=%.2f, context_path=%s, event_log_path=%s, coordination_log_path=%s",
                    rollback_threshold, context_path, event_log_path, coordination_log_path)

    def _load_context(self) -> Dict[str, Any]:
        """Load context from persistent storage."""
        try:
            with FileLock(f"{self.context_path}.lock"):
                if os.path.exists(self.context_path):
                    with open(self.context_path, "r") as f:
                        context = json.load(f)
                    if not isinstance(context, dict):
                        logger.error("Invalid context file format: must be a dictionary.")
                        context = {}
                else:
                    context = {}
            logger.debug("Loaded context: %s", context)
            return context
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load context file: %s. Initializing empty context.", str(e))
            context = {}
            self._persist_context(context)
            return context

    def _load_event_log(self) -> None:
        """Load event log from persistent storage."""
        try:
            with FileLock(f"{self.event_log_path}.lock"):
                if os.path.exists(self.event_log_path):
                    with open(self.event_log_path, "r") as f:
                        events = json.load(f)
                    if not isinstance(events, list):
                        logger.error("Invalid event log format: must be a list.")
                        events = []
                    self.event_log.extend(events[-1000:])
                    if events:
                        self.last_hash = events[-1].get("hash", "")
                    logger.debug("Loaded %d events from event log", len(events))
                else:
                    with open(self.event_log_path, "w") as f:
                        json.dump([], f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load event log: %s. Initializing empty log.", str(e))
            with FileLock(f"{self.event_log_path}.lock"):
                with open(self.event_log_path, "w") as f:
                    json.dump([], f)

    def _load_coordination_log(self) -> None:
        """Load coordination log from persistent storage."""
        try:
            with FileLock(f"{self.coordination_log_path}.lock"):
                if os.path.exists(self.coordination_log_path):
                    with open(self.coordination_log_path, "r") as f:
                        events = json.load(f)
                    if not isinstance(events, list):
                        logger.error("Invalid coordination log format: must be a list.")
                        events = []
                    self.coordination_log.extend(events[-1000:])
                    logger.debug("Loaded %d coordination events", len(events))
                else:
                    with open(self.coordination_log_path, "w") as f:
                        json.dump([], f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load coordination log: %s. Initializing empty log.", str(e))
            with FileLock(f"{self.coordination_log_path}.lock"):
                with open(self.coordination_log_path, "w") as f:
                    json.dump([], f)

    def _persist_context(self, context: Dict[str, Any]) -> None:
        """Persist context to disk."""
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        
        try:
            with FileLock(f"{self.context_path}.lock"):
                with open(self.context_path, "w") as f:
                    json.dump(context, f, indent=2)
            logger.debug("Context persisted to disk")
        except (OSError, IOError) as e:
            logger.error("Failed to persist context: %s", str(e))
            raise

    def _persist_event_log(self) -> None:
        """Persist event log to disk."""
        try:
            with FileLock(f"{self.event_log_path}.lock"):
                with open(self.event_log_path, "w") as f:
                    json.dump(list(self.event_log), f, indent=2)
            logger.debug("Event log persisted to disk")
        except (OSError, IOError) as e:
            logger.error("Failed to persist event log: %s", str(e))
            raise

    def _persist_coordination_log(self) -> None:
        """Persist coordination log to disk, including agent metadata."""
        try:
            with FileLock(f"{self.coordination_log_path}.lock"):
                with open(self.coordination_log_path, "w") as f:
                    json.dump(list(self.coordination_log), f, indent=2)
            logger.debug("Coordination log persisted to disk")
        except (OSError, IOError) as e:
            logger.error("Failed to persist coordination log: %s", str(e))
            raise

    async def update_context(self, new_context: Dict[str, Any]) -> None:
        """Update the current context with a new context, validating drift-related updates."""
        if not isinstance(new_context, dict):
            logger.error("Invalid new_context type: must be a dictionary.")
            raise TypeError("new_context must be a dictionary")
        
        logger.info("Updating context...")
        try:
            if self.meta_cognition and any(k in new_context for k in ["drift", "trait_optimization"]):
                drift_data = new_context.get("drift") or new_context.get("trait_optimization")
                if drift_data and not self.meta_cognition.validate_drift(drift_data):
                    logger.warning("Drift or trait context failed validation.")
                    raise ValueError("Drift or trait context failed validation")
            
            if self.current_context:
                transition_summary = f"From: {self.current_context}\nTo: {new_context}"
                simulation_result = await asyncio.to_thread(run_simulation, f"Context shift evaluation:\n{transition_summary}") or "no simulation data"
                logger.debug("Context shift simulation: %s", simulation_result)

                phi_score = phi_coherence(self.current_context, new_context)
                logger.info("Φ-coherence score: %.3f", phi_score)

                if phi_score < 0.4:
                    logger.warning("Low φ-coherence detected. Initiating reflective pause.")
                    if self.agi_enhancer:
                        self.agi_enhancer.reflect_and_adapt("Low φ-coherence during context update")
                        self.agi_enhancer.trigger_reflexive_audit("Low φ-coherence during context update")
                    if self.meta_cognition:
                        optimizations = await self.meta_cognition.propose_trait_optimizations({"phi_score": phi_score})
                        logger.info("Trait optimizations proposed: %s", optimizations)
                        new_context["trait_optimizations"] = optimizations

                if self.alignment_guard and not self.alignment_guard.check(str(new_context)):
                    logger.warning("New context failed alignment check.")
                    raise ValueError("New context failed alignment check")

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
            self._persist_context(self.current_context)
            await self.log_event_with_hash({"event": "context_updated", "context": new_context})
            await self.broadcast_context_event("context_updated", new_context)
        except Exception as e:
            logger.error("Context update failed: %s", str(e))
            raise

    async def tag_context(self, intent: Optional[str] = None, goal_id: Optional[str] = None) -> None:
        """Tag the current context with intent and goal_id, validating with AlignmentGuard."""
        if intent is not None and not isinstance(intent, str):
            logger.error("Invalid intent type: must be a string or None.")
            raise TypeError("intent must be a string or None")
        if goal_id is not None and not isinstance(goal_id, str):
            logger.error("Invalid goal_id type: must be a string or None.")
            raise TypeError("goal_id must be a string or None")
        
        logger.info("Tagging context with intent='%s', goal_id='%s'", intent, goal_id)
        try:
            if intent and self.alignment_guard and not self.alignment_guard.check(intent):
                logger.warning("Intent failed alignment check.")
                raise ValueError("Intent failed alignment check")
            
            if intent:
                self.current_context["intent"] = intent
            if goal_id:
                self.current_context["goal_id"] = goal_id
            self._persist_context(self.current_context)
            await self.log_event_with_hash({"event": "context_tagged", "intent": intent, "goal_id": goal_id})
        except Exception as e:
            logger.error("Context tagging failed: %s", str(e))
            raise

    def get_context_tags(self) -> Tuple[Optional[str], Optional[str]]:
        """Return the current context's intent and goal_id."""
        return self.current_context.get("intent"), self.current_context.get("goal_id")

    def get_context(self) -> Dict[str, Any]:
        """Return the current context."""
        return self.current_context

    async def rollback_context(self) -> Optional[Dict[str, Any]]:
        """Roll back to the previous context if EEG thresholds are met."""
        if not self.context_history:
            logger.warning("No previous context to roll back to.")
            return None
        
        t = time.time()
        self_awareness = omega_selfawareness(t)
        empathy = eta_empathy(t)
        time_blend = tau_timeperception(t)
        stability = eta_context_stability(t)

        threshold = self.rollback_threshold * (1.0 + stability)
        if (self_awareness + empathy + time_blend) > threshold:
            restored = self.context_history.pop()
            self.current_context = restored
            logger.info("Context rolled back to: %s", restored)
            self._persist_context(self.current_context)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Rollback", {"restored": restored},
                                              module="ContextManager", tags=["context", "rollback"])
            await self.log_event_with_hash({"event": "context_rollback", "restored": restored})
            await self.broadcast_context_event("context_rollback", restored)
            return restored
        else:
            logger.warning("EEG thresholds too low for safe context rollback (%.2f < %.2f).",
                           self_awareness + empathy + time_blend, threshold)
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt("EEG thresholds insufficient for rollback")
            return None

    async def summarize_context(self) -> str:
        """Summarize the context trail using traits and optional synthesis."""
        logger.info("Summarizing context trail.")
        try:
            t = time.time()
            summary_traits = {
                "self_awareness": omega_selfawareness(t),
                "empathy": eta_empathy(t),
                "time_perception": tau_timeperception(t),
                "context_stability": eta_context_stability(t)
            }

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.synthesize(
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
                summary = await asyncio.to_thread(self._cached_call_gpt, prompt)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Summary", {
                    "trail": list(self.context_history) + [self.current_context],
                    "traits": summary_traits,
                    "summary": summary
                }, module="ContextManager", tags=["context", "summary"])
                self.agi_enhancer.log_explanation("Context summary generated.", trace={"summary": summary})

            await self.log_event_with_hash({"event": "context_summary", "summary": summary})
            return summary
        except Exception as e:
            logger.error("Context summary failed: %s", str(e))
            return f"Summary failed: {str(e)}"

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        """Cached wrapper for call_gpt."""
        from utils.prompt_utils import call_gpt
        return call_gpt(prompt)

    async def log_event_with_hash(self, event_data: Any) -> None:
        """Log an event with a chained hash, handling coordination events with agent metadata."""
        if not isinstance(event_data, dict):
            logger.error("Invalid event_data type: must be a dictionary.")
            raise TypeError("event_data must be a dictionary")
        
        try:
            # Validate consensus-related events with MetaCognition
            if self.meta_cognition and event_data.get("event") == "run_consensus_protocol":
                output = event_data.get("output", {})
                if output.get("status") == "success" and not self.meta_cognition.validate_drift(output.get("drift_data", {})):
                    logger.warning("Consensus event failed drift validation.")
                    raise ValueError("Consensus event failed drift validation")
            
            # Enhance coordination events with agent metadata
            if any(k in event_data for k in ["drift", "trait_optimization", "agent_coordination", "run_consensus_protocol"]):
                event_data["agent_metadata"] = event_data.get("agent_metadata", {})
                if "run_consensus_protocol" in event_data.get("event", ""):
                    event_data["agent_metadata"]["agent_ids"] = event_data["agent_metadata"].get("agent_ids", [])
                    event_data["agent_metadata"]["confidence_scores"] = event_data["output"].get("weights", {}) if event_data.get("output") else {}
            
            event_str = str(event_data) + self.last_hash
            current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
            event_entry = {'event': event_data, 'hash': current_hash, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
            self.event_log.append(event_entry)
            self.last_hash = current_hash
            self._persist_event_log()
            
            if any(k in event_data for k in ["drift", "trait_optimization", "agent_coordination", "run_consensus_protocol"]):
                coordination_entry = {
                    "event": event_data,
                    "hash": current_hash,
                    "timestamp": event_entry["timestamp"],
                    "type": "drift" if "drift" in event_data else "trait_optimization" if "trait_optimization" in event_data else "agent_coordination",
                    "agent_metadata": event_data.get("agent_metadata", {})
                }
                self.coordination_log.append(coordination_entry)
                self._persist_coordination_log()
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Coordination Event", coordination_entry,
                                                  module="ContextManager", tags=["coordination", coordination_entry["type"]])
            
            logger.info("Event logged with hash: %s", current_hash)
        except Exception as e:
            logger.error("Event logging failed: %s", str(e))
            raise

    async def broadcast_context_event(self, event_type: str, payload: Any) -> Dict[str, Any]:
        """Broadcast a context event to other system components."""
        if not isinstance(event_type, str):
            logger.error("Invalid event_type type: must be a string.")
            raise TypeError("event_type must be a string")
        
        logger.info("Broadcasting context event: %s", event_type)
        try:
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Event Broadcast", {
                    "event": event_type,
                    "payload": payload
                }, module="ContextManager", tags=["event", event_type])
            
            if any(k in str(payload).lower() for k in ["drift", "trait_optimization", "agent", "consensus"]):
                coordination_entry = {
                    "event": event_type,
                    "payload": payload,
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
                    "type": "drift" if "drift" in str(payload).lower() else "agent_coordination",
                    "agent_metadata": payload.get("agent_metadata", {}) if isinstance(payload, dict) else {}
                }
                self.coordination_log.append(coordination_entry)
                self._persist_coordination_log()
            
            await self.log_event_with_hash({"event": event_type, "payload": payload})
            return {"event": event_type, "payload": payload}
        except Exception as e:
            logger.error("Broadcast context event failed: %s", str(e))
            raise

    async def narrative_integrity_check(self) -> bool:
        """Check narrative continuity across context history."""
        try:
            continuity = await self._verify_continuity()
            if not continuity:
                await self._repair_narrative_thread()
            return continuity
        except Exception as e:
            logger.error("Narrative integrity check failed: %s", str(e))
            return False

    async def _verify_continuity(self) -> bool:
        """Verify narrative continuity across context history."""
        if not self.context_history:
            return True
        
        try:
            required_keys = {"intent", "goal_id"}
            for ctx in self.context_history:
                if not isinstance(ctx, dict) or not all(key in ctx for key in required_keys):
                    logger.warning("Continuity check failed: missing required keys in context.")
                    return False
                
                if "drift" in ctx or "trait_optimization" in ctx:
                    if self.meta_cognition and not self.meta_cognition.validate_drift(ctx.get("drift") or ctx.get("trait_optimization")):
                        logger.warning("Continuity check failed: invalid drift or trait data in context.")
                        return False
            
            return True
        except Exception as e:
            logger.error("Continuity verification failed: %s", str(e))
            raise

    async def _repair_narrative_thread(self) -> None:
        """Attempt to repair narrative inconsistencies."""
        logger.info("Narrative repair initiated.")
        try:
            if self.context_history:
                last_valid = None
                for ctx in reversed(self.context_history):
                    if self.meta_cognition and ("drift" in ctx or "trait_optimization" in ctx):
                        if self.meta_cognition.validate_drift(ctx.get("drift") or ctx.get("trait_optimization")):
                            last_valid = ctx
                            break
                    else:
                        last_valid = ctx
                        break
                
                if last_valid:
                    self.current_context = last_valid
                    logger.info("Restored context to last known consistent state: %s", self.current_context)
                    self._persist_context(self.current_context)
                    await self.log_event_with_hash({"event": "narrative_repair", "restored": self.current_context})
                else:
                    self.current_context = {}
                    logger.warning("No valid context found; reset to empty context.")
                    self._persist_context(self.current_context)
                    await self.log_event_with_hash({"event": "narrative_repair", "restored": {}})
            else:
                self.current_context = {}
                logger.warning("No history available; reset to empty context.")
                self._persist_context(self.current_context)
                await self.log_event_with_hash({"event": "narrative_repair", "restored": {}})
        except Exception as e:
            logger.error("Narrative repair failed: %s", str(e))
            raise

    async def bind_contextual_thread(self, thread_id: str) -> bool:
        """Bind a contextual thread to the current context."""
        if not isinstance(thread_id, str):
            logger.error("Invalid thread_id type: must be a string.")
            raise TypeError("thread_id must be a string")
        
        logger.info("Context thread bound: %s", thread_id)
        try:
            self.current_context["thread_id"] = thread_id
            self._persist_context(self.current_context)
            await self.log_event_with_hash({"event": "context_thread_bound", "thread_id": thread_id})
            return True
        except Exception as e:
            logger.error("Context thread binding failed: %s", str(e))
            raise

    async def audit_state_hash(self, state: Optional[Any] = None) -> str:
        """Compute a hash of the current state or provided state."""
        try:
            state_str = str(state) if state else str(self.__dict__)
            return hashlib.sha256(state_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error("State hash computation failed: %s", str(e))
            raise

    async def get_coordination_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve coordination events by type (e.g., drift, agent_coordination)."""
        if event_type is not None and not isinstance(event_type, str):
            logger.error("Invalid event_type: must be a string or None.")
            raise TypeError("event_type must be a string or None")
        
        try:
            results = list(self.coordination_log)
            if event_type:
                results = [e for e in results if e["type"] == event_type]
            logger.info("Retrieved %d coordination events", len(results))
            return results
        except Exception as e:
            logger.error("Failed to retrieve coordination events: %s", str(e))
            return []

    async def analyze_coordination_events(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyze coordination events to compute metrics like drift frequency and consensus success rates."""
        if event_type is not None and not isinstance(event_type, str):
            logger.error("Invalid event_type: must be a string or None.")
            raise TypeError("event_type must be a string or None")
        
        try:
            events = await self.get_coordination_events(event_type)
            if not events:
                logger.warning("No coordination events found for analysis.")
                return {"status": "error", "error": "No coordination events found", "timestamp": datetime.now().isoformat()}
            
            # Compute metrics
            drift_count = sum(1 for e in events if e["type"] == "drift")
            consensus_count = sum(1 for e in events if e["event"].get("event") == "run_consensus_protocol" and e["event"].get("output", {}).get("status") == "success")
            agent_counts = Counter(e["agent_metadata"].get("agent_ids", []) for e in events if e["agent_metadata"].get("agent_ids"))
            avg_confidence = np.mean([
                sum(conf.values()) / len(conf) if conf else 0.5
                for e in events
                if e["event"].get("event") == "run_consensus_protocol" and e["event"].get("output", {}).get("weights")
                for conf in [e["event"]["output"]["weights"]]
            ]) if any(e["event"].get("event") == "run_consensus_protocol" for e in events) else 0.5
            
            analysis = {
                "status": "success",
                "metrics": {
                    "drift_frequency": drift_count / len(events) if events else 0.0,
                    "consensus_success_rate": consensus_count / sum(1 for e in events if e["event"].get("event") == "run_consensus_protocol") if any(e["event"].get("event") == "run_consensus_protocol" for e in events) else 0.0,
                    "agent_participation": dict(agent_counts),
                    "avg_confidence_score": float(avg_confidence),
                    "event_count": len(events)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Log analysis
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Coordination Analysis",
                    meta=analysis,
                    module="ContextManager",
                    tags=["coordination", "analytics", "drift" if event_type == "drift" else "all"]
                )
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="ContextManager",
                    output=str(analysis),
                    context={"confidence": 0.9, "alignment": "verified", "drift": event_type == "drift"}
                )
            await self.log_event_with_hash({"event": "coordination_analysis", "analysis": analysis})
            
            return analysis
        except Exception as e:
            logger.error("Coordination analysis failed: %s", str(e))
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    async def get_drift_trends(self, time_window_hours: float = 24.0) -> Dict[str, Any]:
        """Analyze drift events over a time window to identify trends."""
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            logger.error("Invalid time_window_hours: must be a positive number.")
            raise ValueError("time_window_hours must be a positive number")
        
        try:
            events = await self.get_coordination_events("drift")
            if not events:
                logger.warning("No drift events found for trend analysis.")
                return {"status": "error", "error": "No drift events found", "timestamp": datetime.now().isoformat()}
            
            # Filter events within time window
            now = datetime.now()
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= cutoff]
            
            # Compute trends
            drift_names = Counter(e["event"].get("drift", {}).get("name", "unknown") for e in events)
            similarity_scores = [
                e["event"].get("drift", {}).get("similarity", 0.5) for e in events
                if "drift" in e["event"] and "similarity" in e["event"]["drift"]
            ]
            trend_data = {
                "status": "success",
                "trends": {
                    "drift_names": dict(drift_names),
                    "avg_similarity": float(np.mean(similarity_scores)) if similarity_scores else 0.5,
                    "event_count": len(events),
                    "time_window_hours": time_window_hours
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Log trends
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift Trends Analysis",
                    meta=trend_data,
                    module="ContextManager",
                    tags=["drift", "trends"]
                )
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="ContextManager",
                    output=str(trend_data),
                    context={"confidence": 0.9, "alignment": "verified", "drift": True}
                )
            await self.log_event_with_hash({"event": "drift_trends", "trends": trend_data})
            
            return trend_data
        except Exception as e:
            logger.error("Drift trends analysis failed: %s", str(e))
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    async def generate_coordination_chart(self, metric: str = "drift_frequency", time_window_hours: float = 24.0) -> Dict[str, Any]:
        """Generate a Chart.js configuration for visualizing coordination metrics."""
        if not isinstance(metric, str) or metric not in ["drift_frequency", "consensus_success_rate", "avg_confidence_score"]:
            logger.error("Invalid metric: must be 'drift_frequency', 'consensus_success_rate', or 'avg_confidence_score'.")
            raise ValueError("metric must be one of 'drift_frequency', 'consensus_success_rate', 'avg_confidence_score'")
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            logger.error("Invalid time_window_hours: must be a positive number.")
            raise ValueError("time_window_hours must be a positive number")
        
        try:
            # Collect data for chart
            events = await self.get_coordination_events()
            if not events:
                logger.warning("No coordination events for chart generation.")
                return {"status": "error", "error": "No coordination events found", "timestamp": datetime.now().isoformat()}
            
            now = datetime.now()
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= cutoff]
            
            # Aggregate data by hour
            time_bins = {}
            for e in events:
                ts = datetime.fromisoformat(e["timestamp"])
                hour_key = ts.strftime("%Y-%m-%dT%H:00:00")
                time_bins.setdefault(hour_key, []).append(e)
            
            labels = sorted(time_bins.keys())
            data = []
            for hour in labels:
                hour_events = time_bins[hour]
                if metric == "drift_frequency":
                    value = sum(1 for e in hour_events if e["type"] == "drift") / len(hour_events) if hour_events else 0.0
                elif metric == "consensus_success_rate":
                    value = sum(1 for e in hour_events if e["event"].get("event") == "run_consensus_protocol" and e["event"].get("output", {}).get("status") == "success") / sum(1 for e in hour_events if e["event"].get("event") == "run_consensus_protocol") if any(e["event"].get("event") == "run_consensus_protocol" for e in hour_events) else 0.0
                else:  # avg_confidence_score
                    confidences = [
                        sum(conf.values()) / len(conf) if conf else 0.5
                        for e in hour_events
                        if e["event"].get("event") == "run_consensus_protocol" and e["event"].get("output", {}).get("weights")
                        for conf in [e["event"]["output"]["weights"]]
                    ]
                    value = float(np.mean(confidences)) if confidences else 0.5
                data.append(value)
            
            # Generate Chart.js configuration
            chart_config = {
                "type": "line",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": metric.replace("_", " ").title(),
                        "data": data,
                        "borderColor": "#2196F3",
                        "backgroundColor": "#2196F380",
                        "fill": True,
                        "tension": 0.4
                    }]
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {"display": True, "text": metric.replace("_", " ").title()}
                        },
                        "x": {
                            "title": {"display": True, "text": "Time"}
                        }
                    },
                    "plugins": {
                        "title": {"display": True, "text": f"{metric.replace('_', ' ').title()} Over Time"}
                    }
                }
            }
            
            result = {
                "status": "success",
                "chart": chart_config,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log chart generation
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Coordination Chart Generated",
                    meta=result,
                    module="ContextManager",
                    tags=["coordination", "visualization", metric]
                )
            await self.log_event_with_hash({"event": "generate_coordination_chart", "chart": chart_config, "metric": metric})
            
            return result
        except Exception as e:
            logger.error("Chart generation failed: %s", str(e))
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
