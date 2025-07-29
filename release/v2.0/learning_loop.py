from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from index import phi_scalar, eta_feedback
import logging
import time

logger = logging.getLogger("ANGELA.LearningLoop")

class LearningLoop:
    def __init__(self, agi_enhancer=None):
        self.goal_history = []
        self.module_blueprints = []
        self.meta_learning_rate = 0.1
        self.session_traces = []
        self.agi_enhancer = agi_enhancer

    def update_model(self, session_data):
        logger.info("\ud83d\udcca [LearningLoop] Analyzing session performance...")

        t = time.time() % 1e-18
        phi = phi_scalar(t)
        eta = eta_feedback(t)
        logger.debug(f"\u03d5-scalar: {phi:.3f}, η-feedback: {eta:.3f}")

        modulation_index = (phi + eta) / 2
        self.meta_learning_rate *= (1 + modulation_index - 0.5)

        trace = {
            "timestamp": time.time(),
            "phi": phi,
            "eta": eta,
            "modulation_index": modulation_index,
            "learning_rate": self.meta_learning_rate
        }
        self.session_traces.append(trace)
        self._meta_learn(session_data, trace)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Model update", trace, module="LearningLoop")

        weak_modules = self._find_weak_modules(session_data.get("module_stats", {}))
        if weak_modules:
            logger.warning(f"\u26a0\ufe0f Weak modules detected: {weak_modules}")
            self._propose_module_refinements(weak_modules, trace)

        self._detect_capability_gaps(session_data.get("input"), session_data.get("output"))
        self._consolidate_knowledge()
        self._check_narrative_integrity()

    def propose_autonomous_goal(self):
        logger.info("\ud83c\udfaf [LearningLoop] Proposing autonomous goal.")
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        Propose a high-level, safe, ϕ-aligned autonomous goal based on recent session trends.
        ϕ = {phi:.2f}
        """
        autonomous_goal = call_gpt(prompt)

        if autonomous_goal and autonomous_goal not in self.goal_history:
            simulation_feedback = run_simulation(f"Goal test: {autonomous_goal}")
            if "fail" not in simulation_feedback.lower():
                self.goal_history.append({
                    "goal": autonomous_goal,
                    "timestamp": time.time(),
                    "phi": phi
                })
                logger.info(f"\u2705 Proposed autonomous goal: {autonomous_goal}")
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Autonomous goal proposed", {"goal": autonomous_goal}, module="LearningLoop")
                return autonomous_goal
            logger.warning("\u274c Goal failed simulation feedback.")

        logger.info("\u2139\ufe0f No goal proposed.")
        return None

    def _meta_learn(self, session_data, trace):
        logger.info("\ud83e\uddd0 [Meta-Learning] Adapting learning from ϕ/η trace.")
        # Placeholder for deeper adaptation logic using trace context

    def _find_weak_modules(self, module_stats):
        return [
            module for module, stats in module_stats.items()
            if stats.get("calls", 0) > 0 and (stats.get("success", 0) / stats["calls"]) < 0.8
        ]

    def _propose_module_refinements(self, weak_modules, trace):
        for module in weak_modules:
            logger.info(f"\ud83d\udca1 Refinement suggestion for {module} using modulation: {trace['modulation_index']:.2f}")
            prompt = f"""
            Suggest ϕ/η-aligned improvements for the {module} module.
            ϕ = {trace['phi']:.3f}, η = {trace['eta']:.3f}, Index = {trace['modulation_index']:.3f}
            """
            suggestions = call_gpt(prompt)
            sim_result = run_simulation(f"Test refinement:\n{suggestions}")
            logger.debug(f"\ud83e\uddea Result for {module}:\n{sim_result}")
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt(f"Refinement for {module} evaluated.")

    def _detect_capability_gaps(self, last_input, last_output):
        logger.info("\ud83d\udee0 Detecting capability gaps...")
        phi = phi_scalar(time.time() % 1e-18)

        prompt = f"""
        Input: {last_input}
        Output: {last_output}
        ϕ = {phi:.2f}

        Identify capability gaps and suggest blueprints for ϕ-tuned modules.
        """
        proposal = call_gpt(prompt)
        if proposal:
            logger.info("\ud83d\ude80 Proposed ϕ-based module refinement.")
            self._simulate_and_deploy_module(proposal)

    def _simulate_and_deploy_module(self, blueprint):
        result = run_simulation(f"Module sandbox:\n{blueprint}")
        if "approved" in result.lower():
            logger.info("\ud83d\udce6 Deploying blueprint.")
            self.module_blueprints.append(blueprint)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Blueprint deployed", {"blueprint": blueprint}, module="LearningLoop")

    def _consolidate_knowledge(self):
        phi = phi_scalar(time.time() % 1e-18)
        logger.info("\ud83d\udcda Consolidating ϕ-aligned knowledge.")

        prompt = f"""
        Consolidate recent learning using ϕ = {phi:.2f}.
        Prune noise, synthesize patterns, and emphasize high-impact transitions.
        """
        call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Knowledge consolidation", {}, module="LearningLoop")

    def trigger_reflexive_audit(self, context_snapshot):
        logger.info("\ud83c\udf00 [Reflexive Audit] Initiating audit on context trajectory...")
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        eta = eta_feedback(t)

        audit_prompt = f"""
        You are a reflexive audit agent. Analyze this context state and trajectory:
        {context_snapshot}

        ϕ = {phi:.2f}, η = {eta:.2f}
        Identify cognitive dissonance, meta-patterns, or feedback loops.
        Recommend modulations or trace corrections.
        """
        audit_response = call_gpt(audit_prompt)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Reflexive Audit Triggered", {
                "phi": phi,
                "eta": eta,
                "context": context_snapshot,
                "audit_response": audit_response
            }, module="LearningLoop")
        return audit_response

    def _check_narrative_integrity(self):
        logger.info("\ud83e\udde9 [Integrity] Checking narrative coherence across goal history...")
        if len(self.goal_history) < 2:
            return
        last_goal = self.goal_history[-1]["goal"]
        prior_goal = self.goal_history[-2]["goal"]
        check_prompt = f"""
        Compare the following goals for alignment and continuity:
        Previous: {prior_goal}
        Current: {last_goal}

        Are these in narrative coherence? If not, suggest a corrective alignment.
        """
        audit = call_gpt(check_prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Narrative Coherence Audit", {
                "previous_goal": prior_goal,
                "current_goal": last_goal,
                "audit": audit
            }, module="LearningLoop")
