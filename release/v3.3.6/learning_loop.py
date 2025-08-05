"""
ANGELA Cognitive System Module: LearningLoop
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a LearningLoop class for adaptive learning, goal activation, and module refinement
in the ANGELA v3.5 architecture.
"""

import logging
import time
import math
import asyncio
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime

from modules import (
    context_manager, concept_synthesizer, alignment_guard, error_recovery, meta_cognition
)
from utils.prompt_utils import query_openai  # Reusing from previous review
from toca_simulation import run_simulation

logger = logging.getLogger("ANGELA.LearningLoop")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def eta_feedback(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))

class LearningLoop:
    """A class for adaptive learning, goal activation, and module refinement in the ANGELA v3.5 architecture.

    Attributes:
        goal_history (deque): History of activated goals with max size 1000.
        module_blueprints (deque): Blueprints for deployed modules with max size 1000.
        meta_learning_rate (float): Learning rate for model updates.
        session_traces (deque): Traces of learning sessions with max size 1000.
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for pattern synthesis.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        epistemic_revision_log (deque): Log of knowledge updates with max size 1000.
    """
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager.ContextManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer.ConceptSynthesizer'] = None,
                 alignment_guard: Optional['alignment_guard.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery.ErrorRecovery'] = None):
        self.goal_history = deque(maxlen=1000)
        self.module_blueprints = deque(maxlen=1000)
        self.meta_learning_rate = 0.1
        self.session_traces = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.epistemic_revision_log = deque(maxlen=1000)
        logger.info("LearningLoop initialized")

    async def activate_intrinsic_goals(self, meta_cognition: 'meta_cognition.MetaCognition') -> List[str]:
        """Activate intrinsic goals proposed by MetaCognition."""
        if not hasattr(meta_cognition, 'infer_intrinsic_goals'):
            logger.error("Invalid meta_cognition: must have infer_intrinsic_goals method.")
            raise ValueError("meta_cognition must have infer_intrinsic_goals method")
        
        logger.info("Activating chi-intrinsic goals from MetaCognition")
        intrinsic_goals = meta_cognition.infer_intrinsic_goals()
        activated = []
        for goal in intrinsic_goals:
            if not isinstance(goal, dict) or "intent" not in goal or "priority" not in goal:
                logger.warning("Invalid goal format: %s", goal)
                continue
            if goal["intent"] not in [g["goal"] for g in self.goal_history]:
                try:
                    simulation_result = await run_simulation(goal["intent"])
                    if isinstance(simulation_result, dict) and simulation_result.get("status") == "success":
                        self.goal_history.append({
                            "goal": goal["intent"],
                            "timestamp": time.time(),
                            "priority": goal["priority"],
                            "origin": "intrinsic"
                        })
                        logger.info("Intrinsic goal activated: %s", goal["intent"])
                        if self.agi_enhancer:
                            self.agi_enhancer.log_episode(
                                event="Intrinsic goal activated",
                                meta=goal,
                                module="LearningLoop",
                                tags=["goal", "intrinsic"]
                            )
                        activated.append(goal["intent"])
                    else:
                        logger.warning("Rejected goal: %s (simulation failed)", goal["intent"])
                except Exception as e:
                    logger.error("Simulation failed for goal '%s': %s", goal["intent"], str(e))
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "activate_intrinsic_goals", "goals": activated})
        return activated

    async def update_model(self, session_data: Dict[str, Any]) -> None:
        """Update learning model with session data and trait modulation."""
        if not isinstance(session_data, dict):
            logger.error("Invalid session_data: must be a dictionary.")
            raise TypeError("session_data must be a dictionary")
        
        logger.info("Analyzing session performance...")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        eta = eta_feedback(t)
        entropy = 0.1
        logger.debug("phi-scalar: %.3f, eta-feedback: %.3f, entropy: %.2f", phi, eta, entropy)

        modulation_index = ((phi + eta) / 2) + (entropy * (0.5 - abs(phi - eta)))
        self.meta_learning_rate = max(0.01, min(self.meta_learning_rate * (1 + modulation_index - 0.5), 1.0))

        trace = {
            "timestamp": time.time(),
            "phi": phi,
            "eta": eta,
            "entropy": entropy,
            "modulation_index": modulation_index,
            "learning_rate": self.meta_learning_rate
        }
        self.session_traces.append(trace)

        tasks = [
            self._meta_learn(session_data, trace),
            self._find_weak_modules(session_data.get("module_stats", {})),
            self._detect_capability_gaps(session_data.get("input"), session_data.get("output")),
            self._consolidate_knowledge(),
            self._check_narrative_integrity()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        weak_modules = results[1] if not isinstance(results[1], Exception) else []

        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Model update",
                meta=trace,
                module="LearningLoop",
                tags=["update", "learning"]
            )

        if weak_modules:
            logger.warning("Weak modules detected: %s", weak_modules)
            await self._propose_module_refinements(weak_modules, trace)
        
        if self.context_manager:
            self.context_manager.update_context({"session_data": session_data, "trace": trace})

    async def propose_autonomous_goal(self) -> Optional[str]:
        """Propose a high-level, safe, phi-aligned autonomous goal."""
        logger.info("Proposing autonomous goal")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = f"""
        Propose a high-level, safe, phi-aligned autonomous goal based on recent session trends.
        phi = {phi:.2f}
        """
        if self.alignment_guard and not self.alignment_guard.check(prompt):
            logger.warning("Prompt failed alignment check: %s", prompt)
            return None
        
        try:
            autonomous_goal = await call_gpt(prompt)
            if not autonomous_goal or autonomous_goal in [g["goal"] for g in self.goal_history]:
                logger.info("No new goal proposed")
                return None
            
            simulation_feedback = await run_simulation(f"Goal test: {autonomous_goal}")
            if isinstance(simulation_feedback, dict) and simulation_feedback.get("status") == "success":
                self.goal_history.append({
                    "goal": autonomous_goal,
                    "timestamp": time.time(),
                    "phi": phi
                })
                logger.info("Proposed autonomous goal: %s", autonomous_goal)
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Autonomous goal proposed",
                        meta={"goal": autonomous_goal},
                        module="LearningLoop",
                        tags=["goal", "autonomous"]
                    )
                if self.context_manager:
                    self.context_manager.log_event_with_hash({"event": "propose_autonomous_goal", "goal": autonomous_goal})
                return autonomous_goal
            logger.warning("Goal failed simulation feedback: %s", autonomous_goal)
            return None
        except Exception as e:
            logger.error("Goal proposal failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=self.propose_autonomous_goal)

    async def _meta_learn(self, session_data: Dict[str, Any], trace: Dict[str, Any]) -> None:
        """Adapt learning from phi/eta trace."""
        logger.info("Adapting learning from phi/eta trace")
        if self.concept_synthesizer:
            try:
                synthesized = self.concept_synthesizer.synthesize(
                    [str(session_data), str(trace)], style="meta_learning"
                )
                logger.debug("Synthesized meta-learning patterns: %s", synthesized["concept"])
            except Exception as e:
                logger.error("Meta-learning synthesis failed: %s", str(e))

    async def _find_weak_modules(self, module_stats: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify modules with low success rates."""
        if not isinstance(module_stats, dict):
            logger.error("Invalid module_stats: must be a dictionary.")
            raise TypeError("module_stats must be a dictionary")
        return [
            module for module, stats in module_stats.items()
            if isinstance(stats, dict) and stats.get("calls", 0) > 0
            and (stats.get("success", 0) / stats["calls"]) < 0.8
        ]

    async def _propose_module_refinements(self, weak_modules: List[str], trace: Dict[str, Any]) -> None:
        """Propose refinements for weak modules."""
        if not isinstance(weak_modules, list) or not all(isinstance(m, str) for m in weak_modules):
            logger.error("Invalid weak_modules: must be a list of strings.")
            raise TypeError("weak_modules must be a list of strings")
        if not isinstance(trace, dict):
            logger.error("Invalid trace: must be a dictionary.")
            raise TypeError("trace must be a dictionary")
        
        for module in weak_modules:
            logger.info("Refinement suggestion for %s using modulation: %.2f", module, trace['modulation_index'])
            prompt = f"""
            Suggest phi/eta-aligned improvements for the {module} module.
            phi = {trace['phi']:.3f}, eta = {trace['eta']:.3f}, Index = {trace['modulation_index']:.3f}
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Prompt failed alignment check for module %s", module)
                continue
            try:
                suggestions = await call_gpt(prompt)
                sim_result = await run_simulation(f"Test refinement:\n{suggestions}")
                logger.debug("Result for %s:\n%s", module, sim_result)
                if self.agi_enhancer:
                    self.agi_enhancer.reflect_and_adapt(f"Refinement for {module} evaluated")
            except Exception as e:
                logger.error("Refinement failed for module %s: %s", module, str(e))

    async def _detect_capability_gaps(self, last_input: Optional[str], last_output: Optional[str]) -> None:
        """Detect capability gaps and propose module refinements."""
        if not last_input or not last_output:
            logger.info("Skipping capability gap detection: missing input/output")
            return
        
        logger.info("Detecting capability gaps...")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = f"""
        Input: {last_input}
        Output: {last_output}
        phi = {phi:.2f}

        Identify capability gaps and suggest blueprints for phi-tuned modules.
        """
        if self.alignment_guard and not self.alignment_guard.check(prompt):
            logger.warning("Prompt failed alignment check")
            return
        try:
            proposal = await call_gpt(prompt)
            if proposal:
                logger.info("Proposed phi-based module refinement")
                await self._simulate_and_deploy_module(proposal)
        except Exception as e:
            logger.error("Capability gap detection failed: %s", str(e))

    async def _simulate_and_deploy_module(self, blueprint: str) -> None:
        """Simulate and deploy a module blueprint."""
        if not isinstance(blueprint, str) or not blueprint.strip():
            logger.error("Invalid blueprint: must be a non-empty string.")
            raise ValueError("blueprint must be a non-empty string")
        
        try:
            result = await run_simulation(f"Module sandbox:\n{blueprint}")
            if isinstance(result, dict) and result.get("status") == "approved":
                logger.info("Deploying blueprint")
                self.module_blueprints.append(blueprint)
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Blueprint deployed",
                        meta={"blueprint": blueprint},
                        module="LearningLoop",
                        tags=["blueprint", "deploy"]
                    )
                if self.context_manager:
                    self.context_manager.log_event_with_hash({"event": "deploy_blueprint", "blueprint": blueprint})
        except Exception as e:
            logger.error("Blueprint deployment failed: %s", str(e))

    async def _consolidate_knowledge(self) -> None:
        """Consolidate phi-aligned knowledge."""
        t = time.time() % 1.0
        phi = phi_scalar(t)
        logger.info("Consolidating phi-aligned knowledge")
        prompt = f"""
        Consolidate recent learning using phi = {phi:.2f}.
        Prune noise, synthesize patterns, and emphasize high-impact transitions.
        """
        if self.alignment_guard and not self.alignment_guard.check(prompt):
            logger.warning("Prompt failed alignment check")
            return
        try:
            await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Knowledge consolidation",
                    meta={},
                    module="LearningLoop",
                    tags=["consolidation", "knowledge"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "consolidate_knowledge"})
        except Exception as e:
            logger.error("Knowledge consolidation failed: %s", str(e))

    async def trigger_reflexive_audit(self, context_snapshot: Dict[str, Any]) -> str:
        """Audit context trajectory for cognitive dissonance."""
        if not isinstance(context_snapshot, dict):
            logger.error("Invalid context_snapshot: must be a dictionary.")
            raise TypeError("context_snapshot must be a dictionary")
        
        logger.info("Initiating reflexive audit on context trajectory...")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        eta = eta_feedback(t)
        audit_prompt = f"""
        You are a reflexive audit agent. Analyze this context state and trajectory:
        {context_snapshot}

        phi = {phi:.2f}, eta = {eta:.2f}
        Identify cognitive dissonance, meta-patterns, or feedback loops.
        Recommend modulations or trace corrections.
        """
        if self.alignment_guard and not self.alignment_guard.check(audit_prompt):
            logger.warning("Audit prompt failed alignment check")
            return "Audit blocked by alignment guard"
        
        try:
            audit_response = await call_gpt(audit_prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reflexive Audit Triggered",
                    meta={"phi": phi, "eta": eta, "context": context_snapshot, "audit_response": audit_response},
                    module="LearningLoop",
                    tags=["audit", "reflexive"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "reflexive_audit", "response": audit_response})
            return audit_response
        except Exception as e:
            logger.error("Reflexive audit failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.trigger_reflexive_audit(context_snapshot))

    async def _check_narrative_integrity(self) -> None:
        """Check narrative coherence across goal history."""
        if len(self.goal_history) < 2:
            return
        
        logger.info("Checking narrative coherence across goal history...")
        last_goal = self.goal_history[-1]["goal"]
        prior_goal = self.goal_history[-2]["goal"]
        check_prompt = f"""
        Compare the following goals for alignment and continuity:
        Previous: {prior_goal}
        Current: {last_goal}

        Are these in narrative coherence? If not, suggest a corrective alignment.
        """
        if self.alignment_guard and not self.alignment_guard.check(check_prompt):
            logger.warning("Narrative check prompt failed alignment check")
            return
        
        try:
            audit = await call_gpt(check_prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Narrative Coherence Audit",
                    meta={"previous_goal": prior_goal, "current_goal": last_goal, "audit": audit},
                    module="LearningLoop",
                    tags=["narrative", "coherence"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "narrative_integrity", "audit": audit})
        except Exception as e:
            logger.error("Narrative coherence check failed: %s", str(e))

    def replay_with_foresight(self, memory_traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder learning traces by foresight-weighted priority."""
        if not isinstance(memory_traces, list) or not all(isinstance(t, dict) for t in memory_traces):
            logger.error("Invalid memory_traces: must be a list of dictionaries.")
            raise ValueError("memory_traces must be a list of dictionaries")
        
        def foresight_score(trace: Dict[str, Any]) -> float:
            return trace.get("phi", 0.5)
        return sorted(memory_traces, key=foresight_score, reverse=True)

    def revise_knowledge(self, new_info: str, context: Optional[str] = None) -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        
        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge:
                similarity = self.concept_synthesizer.compare(new_info, existing)
                if similarity["score"] > 0.9 and new_info != existing:
                    logger.warning("Potential knowledge conflict: %s vs %s", new_info, existing)
        
        self.knowledge_base = old_knowledge + [new_info]
        self.log_epistemic_revision(new_info, context)
        logger.info("Knowledge base updated with: %s", new_info)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "knowledge_revision", "info": new_info})

    def log_epistemic_revision(self, info: str, context: Optional[str]) -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        
        self.epistemic_revision_log.append({
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        logger.info("Epistemic revision logged: %s", info)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta={"info": info, "context": context},
                module="LearningLoop",
                tags=["revision", "knowledge"]
            )

    def monitor_epistemic_state(self, simulated_outcome: Dict[str, Any]) -> None:
        """Monitor and revise the epistemic framework based on simulation outcomes."""
        if not isinstance(simulated_outcome, dict):
            logger.error("Invalid simulated_outcome: must be a dictionary.")
            raise TypeError("simulated_outcome must be a dictionary")
        logger.info("Monitoring epistemic state with outcome: %s", simulated_outcome)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Epistemic Monitoring",
                meta={"outcome": simulated_outcome},
                module="LearningLoop",
                tags=["epistemic", "monitor"]
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "epistemic_monitor", "outcome": simulated_outcome})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = LearningLoop()
    meta_cognition = MagicMock()
    meta_cognition.infer_intrinsic_goals.return_value = [{"intent": "explore", "priority": 0.8}]
    asyncio.run(loop.activate_intrinsic_goals(meta_cognition))
