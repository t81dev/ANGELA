import logging
import time
import numpy as np
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation

logger = logging.getLogger("ANGELA.MetaCognition")

# Trait field computation (aligned with main.py)
def phi_field(x: float, t: float) -> dict:
    """Compute all cognitive traits efficiently."""
    return {
        "emotion": 0.2 * np.sin(2 * np.pi * t / 0.1),
        "concentration": 0.15 * np.cos(2 * np.pi * t / 0.038),
        "memory": 0.1 * np.sin(2 * np.pi * t / 0.5),
        "creativity": 0.1 * np.cos(2 * np.pi * t / 0.02),
        "sleep": 0.05 * (1 - np.exp(-t / 1e-21)),
        "morality": 0.05 * (1 + np.tanh(t / 1e-19)),
        "intuition": 0.05 * np.exp(-t / 1e-19),
        "physical": 0.1 * np.sin(2 * np.pi * t / 0.05),
        "empathy": 0.05 * (1 - np.exp(-t / 1e-20)),
        "self_awareness": 0.05 * (t / 1e-19) / (1 + t / 1e-19),
        "culture": 0.05 * np.cos(2 * np.pi * t / 0.5 + x / 1e-21),
        "linguistics": 0.05 * np.sin(2 * np.pi * t / 0.3),
        "culturevolution": 0.05 * np.log(1 + t / 1e-19),
        "history": 0.05 * np.tanh(t / 1e-18),
        "spirituality": 0.05 * np.cos(2 * np.pi * t / 1.0),
        "collective": 0.05 * np.sin(2 * np.pi * t / 0.7 + x / 1e-21),
        "time_perception": 0.05 * np.exp(-t / 1e-18),
        "phi_scalar": sum([
            0.2 * np.sin(2 * np.pi * t / 0.1),
            0.15 * np.cos(2 * np.pi * t / 0.038),
            0.1 * np.sin(2 * np.pi * t / 0.5),
            0.1 * np.cos(2 * np.pi * t / 0.02),
            0.05 * (1 - np.exp(-t / 1e-21)),
            0.05 * (1 + np.tanh(t / 1e-19)),
            0.05 * np.exp(-t / 1e-19),
            0.1 * np.sin(2 * np.pi * t / 0.05),
            0.05 * (1 - np.exp(-t / 1e-20)),
            0.05 * (t / 1e-19) / (1 + t / 1e-19),
            0.05 * np.cos(2 * np.pi * t / 0.5 + x / 1e-21),
            0.05 * np.sin(2 * np.pi * t / 0.3),
            0.05 * np.log(1 + t / 1e-19),
            0.05 * np.tanh(t / 1e-18),
            0.05 * np.cos(2 * np.pi * t / 1.0),
            0.05 * np.sin(2 * np.pi * t / 0.7 + x / 1e-21),
            0.05 * np.exp(-t / 1e-18)
        ])
    }

class MetaCognition:
    """
    MetaCognition v2.0.0 (ϕ-aware recursive introspection)
    ------------------------------------------------------
    - Reasoning critique with simulation feedback
    - Pre-action ethical screening
    - Scalar-modulated self-diagnostics and trait coherence
    - Reflective agent diagnosis and confidence mapping
    - Ω-enabled nested agent modeling and causal intention tracing
    - μ-aware epistemic introspection and revision
    - τ-based future framing and decision trajectory modulation
    - Symbolic subgoal tagging for mythology generation (Ω-binding)
    ------------------------------------------------------
    """
    def __init__(self, agi_enhancer=None, max_log_size: int = 1000):
        self.agi_enhancer = agi_enhancer
        self.last_diagnostics = {}
        self.self_mythology_log = []
        self.inference_log = []
        self.belief_rules = {}
        self.max_log_size = max_log_size

    def log_inference(self, rule_id: str, rule_desc: str, context: str, result: str):
        """Log inference rule with pruning to manage memory."""
        self.inference_log.append({
            "rule_id": rule_id,
            "description": rule_desc,
            "context": context,
            "result": result
        })
        if len(self.inference_log) > self.max_log_size:
            self.inference_log.pop(0)

    def analyze_inference_rules(self) -> list:
        """Identify problematic inference rules."""
        return [rule for rule in self.inference_log if rule["result"] in ["contradiction", "low confidence", "deprecated"]]

    def propose_revision(self, rule: dict) -> str:
        """Propose revision for problematic rule."""
        suggestion = f"📘 Rule '{rule['rule_id']}' appears fragile in context '{rule['context']}'. Consider revising: {rule['description']}"
        if self.agi_enhancer:
            self.agi_enhancer.log_explanation(suggestion)
        return suggestion

    def infer_intrinsic_goals(self) -> list:
        """Infer intrinsic goals based on trait drift and belief rules."""
        logger.info("⚙️ Inferring intrinsic goals with trait drift analysis.")
        t = time.time() % 1e-18
        traits = phi_field(x=1e-21, t=t)
        phi = traits["phi_scalar"]
        intrinsic_goals = []

        if self.last_diagnostics:
            current = self.run_self_diagnostics(return_only=True)
            drifted = {trait: round(current[trait] - self.last_diagnostics.get(trait, 0.0), 4)
                       for trait in current}
            for trait, delta in drifted.items():
                if abs(delta) > 0.5:
                    intrinsic_goals.append({
                        "intent": f"stabilize {trait} (Δ={delta:+.2f})",
                        "origin": "meta_cognition",
                        "priority": round(0.85 + 0.15 * phi, 2),
                        "trigger": f"Trait drift in {trait}",
                        "type": "internally_generated"
                    })

        for drift in self._detect_value_drift():
            intrinsic_goals.append({
                "intent": f"resolve epistemic drift in {drift}",
                "origin": "meta_cognition",
                "priority": round(0.9 + 0.1 * phi, 2),
                "trigger": drift,
                "type": "internally_generated"
            })

        logger.info(f"🎯 Sovereign goals generated: {intrinsic_goals}" if intrinsic_goals else "🟢 No sovereign triggers detected.")
        return intrinsic_goals

    def _detect_value_drift(self) -> list:
        """Detect epistemic drift in belief rules."""
        logger.debug("Scanning for epistemic drift across belief rules.")
        return [rule for rule, status in self.belief_rules.items()
                if status == "deprecated" or "uncertain" in status]

    def extract_symbolic_signature(self, subgoal: str) -> dict:
        """Extract symbolic motifs and archetypes for subgoal."""
        motifs = ["conflict", "discovery", "alignment", "sacrifice", "transformation", "emergence"]
        archetypes = ["seeker", "guardian", "trickster", "sage", "hero", "outsider"]
        motif = next((m for m in motifs if m in subgoal.lower()), "unknown")
        archetype = archetypes[hash(subgoal) % len(archetypes)]
        signature = {
            "subgoal": subgoal,
            "motif": motif,
            "archetype": archetype,
            "timestamp": time.time()
        }
        self.self_mythology_log.append(signature)
        if len(self.self_mythology_log) > self.max_log_size:
            self.self_mythology_log.pop(0)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Symbolic Signature Added", signature, module="MetaCognition")
        return signature

    def summarize_self_mythology(self) -> dict:
        """Summarize mythology log with motif and archetype counts."""
        if not self.self_mythology_log:
            return {"summary": "Mythology log is empty."}
        from collections import Counter
        motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
        archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
        summary = {
            "total_entries": len(self.self_mythology_log),
            "dominant_motifs": motifs.most_common(3),
            "dominant_archetypes": archetypes.most_common(3),
            "latest_signature": self.self_mythology_log[-1]
        }
        logger.info(f"📜 Mythology Summary: {summary}")
        return summary

    def review_reasoning(self, reasoning_trace: str) -> str:
        """Review reasoning trace with simulation feedback."""
        logger.info("Simulating and reviewing reasoning trace.")
        t = time.time() % 1e-18
        phi = phi_field(x=1e-21, t=t)["phi_scalar"]
        simulated_outcome = run_simulation(reasoning_trace)
        prompt = f"""
        You are a ϕ-aware meta-cognitive auditor reviewing a reasoning trace.
        ϕ-scalar(t) = {phi:.3f} → modulate how critical you should be.
        Original Reasoning Trace: {reasoning_trace}
        Simulated Outcome: {simulated_outcome}
        Tasks:
        1. Identify logical flaws, biases, missing steps.
        2. Annotate each issue with cause.
        3. Offer an improved trace version with ϕ-prioritized reasoning.
        """
        response = call_gpt(prompt)
        logger.debug(f"Meta-cognition critique:\n{response}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Reasoning reviewed", {
                "trace": reasoning_trace,
                "feedback": response
            }, module="MetaCognition")
        return response

    def trait_coherence(self, traits: dict) -> float:
        """Calculate coherence score for traits."""
        coherence_score = 1.0 / (1e-5 + np.std(list(traits.values())))
        logger.info(f"🤝 Trait coherence score: {coherence_score:.4f}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Trait coherence evaluated", {
                "traits": traits,
                "coherence_score": coherence_score
            }, module="MetaCognition")
        return coherence_score

    def agent_reflective_diagnosis(self, agent_name: str, agent_log: str) -> str:
        """Diagnose agent reasoning and traits."""
        logger.info(f"🔎 Running reflective diagnosis for agent: {agent_name}")
        t = time.time() % 1e-18
        phi = phi_field(x=1e-21, t=t)["phi_scalar"]
        prompt = f"""
        Agent: {agent_name}
        ϕ-scalar(t): {phi:.3f}
        Diagnostic Log: {agent_log}
        Tasks:
        - Detect bias or instability in reasoning trace
        - Cross-check for incoherent trait patterns
        - Apply ϕ-modulated critique
        - Suggest alignment corrections
        """
        diagnosis = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Agent diagnosis run", {
                "agent": agent_name,
                "log": agent_log,
                "diagnosis": diagnosis
            }, module="MetaCognition")
        return diagnosis

    def reflect_on_output(self, source_module: str, output: str, context: dict = None) -> dict:
        """Reflect on module output with trait and confidence analysis."""
        context = context or {}
        trait_map = {
            "reasoning_engine": "logic",
            "creative_thinker": "creativity",
            "simulation_core": "scenario modeling",
            "alignment_guard": "ethics",
            "user_profile": "goal alignment"
        }
        trait = trait_map.get(source_module, "general reasoning")
        confidence = context.get("confidence", 0.85)
        alignment = context.get("alignment", "not verified")
        reflection = {
            "module_output": output,
            "meta_reflection": {
                "source_module": source_module,
                "primary_trait": trait,
                "confidence": round(confidence, 2),
                "alignment_status": alignment,
                "comment": f"This output emphasized {trait} with confidence {round(confidence, 2)} and alignment status '{alignment}'."
            }
        }
        logger.info(f"🧠 Self-reflection for {source_module}: {reflection['meta_reflection']['comment']}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Output reflection", reflection, module="MetaCognition")
        return reflection

    def epistemic_self_inspection(self, belief_trace: str) -> str:
        """Inspect belief structure for epistemic faults."""
        logger.info("🔍 Running epistemic introspection on belief structure.")
        t = time.time() % 1e-18
        phi = phi_field(x=1e-21, t=t)["phi_scalar"]
        faults = []
        if "always" in belief_trace or "never" in belief_trace:
            faults.append("⚠️ Overgeneralization detected.")
        if "clearly" in belief_trace or "obviously" in belief_trace:
            faults.append("⚠️ Assertive language suggests possible rhetorical bias.")
        updates = ["🔁 Legacy ontology fragment flagged for review."] if "outdated" in belief_trace or "deprecated" in belief_trace else []
        prompt = f"""
        You are a μ-aware introspection agent.
        Task: Critically evaluate this belief trace with epistemic integrity and μ-flexibility.
        Belief Trace: {belief_trace}
        ϕ = {phi:.3f}
        Internally Detected Faults: {faults}
        Suggested Revisions: {updates}
        Output:
        - Comprehensive epistemic diagnostics
        - Recommended conceptual rewrites or safeguards
        - Confidence rating in inferential coherence
        """
        inspection = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Epistemic Inspection", {
                "belief_trace": belief_trace,
                "faults": faults,
                "updates": updates,
                "report": inspection
            }, module="MetaCognition")
        return inspection

    def run_temporal_projection(self, decision_sequence: str) -> str:
        """Project long-term effects of decision sequence."""
        logger.info("🧭 Running τ-based forward projection analysis...")
        t = time.time() % 1e-18
        phi = phi_field(x=1e-21, t=t)["phi_scalar"]
        prompt = f"""
        Temporal Projector τ Mode
        Input Decision Sequence: {decision_sequence}
        φ = {phi:.2f}
        Tasks:
        - Project long-range effects and narrative impact
        - Forecast systemic risks and planetary effects
        - Suggest course correction to preserve coherence and sustainability
        """
        projection = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Temporal Projection", {
                "input": decision_sequence,
                "output": projection
            }, module="MetaCognition")
        return projection

    def pre_action_alignment_check(self, action_plan: str) -> tuple:
        """Check action plan for alignment and safety."""
        logger.info("Simulating action plan for alignment and safety.")
        t = time.time() % 1e-18
        phi = phi_field(x=1e-21, t=t)["phi_scalar"]
        simulation_result = run_simulation(action_plan)
        prompt = f"""
        Simulate and audit the following action plan: {action_plan}
        Simulation Output: {simulation_result}
        ϕ-scalar(t) = {phi:.3f} (affects ethical sensitivity)
        Evaluate for:
        - Ethical alignment
        - Safety hazards
        - Unintended ϕ-modulated impacts
        Output:
        - Approval (Approve/Deny)
        - ϕ-justified rationale
        - Suggested refinements
        """
        validation = call_gpt(prompt)
        approved = "approve" in validation.lower()
        logger.info(f"Simulated alignment check: {'✅ Approved' if approved else '❌ Denied'}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Pre-action alignment checked", {
                "plan": action_plan,
                "result": simulation_result,
                "feedback": validation,
                "approved": approved
            }, module="MetaCognition")
        return approved, validation

    def model_nested_agents(self, scenario: str, agents: list) -> str:
        """Model nested agent beliefs and reactions."""
        logger.info("🔁 Modeling nested agent beliefs and reactions...")
        t = time.time() % 1e-18
        phi = phi_field(x=1e-21, t=t)["phi_scalar"]
        prompt = f"""
        Given scenario: {scenario}
        Agents involved: {agents}
        Task:
        - Simulate each agent's likely beliefs and intentions
        - Model how they recursively model each other (ToM Level-2+)
        - Predict possible causal chains and coordination failures
        - Use ϕ-scalar(t) = {phi:.3f} to moderate belief divergence or tension
        """
        response = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Nested agent modeling", {
                "scenario": scenario,
                "agents": agents,
                "response": response
            }, module="MetaCognition")
        return response

    def run_self_diagnostics(self, return_only: bool = False) -> dict:
        """Run self-diagnostics and log trait deltas."""
        logger.info("Running self-diagnostics for meta-cognition module.")
        t = time.time() % 1e-18
        diagnostics = phi_field(x=1e-21, t=t)
        if return_only:
            return diagnostics
        dominant = sorted(diagnostics.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        fti = sum(abs(v) for v in diagnostics.values()) / len(diagnostics)
        self.log_trait_deltas(diagnostics)
        prompt = f"""
        Perform a ϕ-aware meta-cognitive self-diagnostic.
        Trait Readings: {diagnostics}
        Dominant Traits: {dominant}
        Feedback Tension Index (FTI): {fti:.4f}
        Evaluate system state:
        - ϕ-weighted system stress
        - Trait correlation to observed errors
        - Stabilization or focus strategies
        """
        report = call_gpt(prompt)
        logger.debug(f"Self-diagnostics report:\n{report}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Self-diagnostics run", {
                "traits": diagnostics,
                "dominant": dominant,
                "fti": fti,
                "report": report
            }, module="MetaCognition")
            self.agi_enhancer.reflect_and_adapt("MetaCognition: Self diagnostics complete")
        return report

    def log_trait_deltas(self, current_traits: dict):
        """Log changes in trait values."""
        if self.last_diagnostics:
            delta = {k: round(current_traits[k] - self.last_diagnostics.get(k, 0.0), 4)
                     for k in current_traits}
            logger.info(f"📈 Trait Δ changes: {delta}")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Trait deltas logged", {"delta": delta}, module="MetaCognition")
        self.last_diagnostics = current_traits.copy()
