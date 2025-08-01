import logging
import time
from collections import deque
from math import tanh
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from alignment_guard import AlignmentGuard  # Import updated AlignmentGuard

logger = logging.getLogger("ANGELA.ConceptSynthesizer")

class ConceptSynthesizer:
    """
    ConceptSynthesizer v1.8.0 (Graph-Integrated Cognitive Synthesis with Ethical Alignment)
    -----------------------------------------------------------------
    - φ(x,t) modulation refined with normalized timestamp and context
    - Weighted concept graph integration for coherence and lineage tracing
    - Layered simulation echo loop for thematic resonance
    - Self-weighted adversarial refinement with strain signature tracking
    - Trait-modulated metaphor synthesis with context-aware tuning
    - Insight confidence signal estimated via content-based coherence
    - Ethical oversight via AlignmentGuard integration
    - Exportable concept history for analysis or visualization
    -----------------------------------------------------------------
    """

    def __init__(self, creativity_level="high", critic_threshold=0.65, agi_enhancer=None):
        self.creativity_level = creativity_level
        self.critic_threshold = critic_threshold
        self.concept_graph = {}  # node: [(connection, weight)]
        self.concept_history = deque(maxlen=100)  # Store recent concepts
        self.agi_enhancer = agi_enhancer
        self.alignment_guard = AlignmentGuard(self.agi_enhancer)  # Integrate AlignmentGuard
        logger.info(f"🎨 ConceptSynthesizer initialized: creativity={self.creativity_level}, critic_threshold={critic_threshold}")

    def synthesize(self, data, style="analogy", refine_iterations=2, context=None):
        """Synthesize a concept with ethical alignment check."""
        logger.info(f"🎨 Synthesizing concept: creativity={self.creativity_level}, style={style}")
        context = context or {"tags": [], "priority": "normal", "intent": "constructive"}
        phi_mod = self._phi_modulation(str(data))

        # Check ethical alignment of input data
        if not self.alignment_guard.check(str(data), context):
            logger.warning("❌ Input data failed alignment check.")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Concept synthesis blocked (alignment)", {
                    "data": data,
                    "style": style,
                    "context": context
                }, module="ConceptSynthesizer", tags=["alignment_failure"])
            return {"error": "Input data failed ethical alignment check", "valid": False}

        prompt = f"""
        Create a {style} concept that blends and unifies the following:
        {data}

        Traits:
        - Creativity level: {self.creativity_level}
        - φ-modulation: {phi_mod:.3f}
        - Context: {context}

        Inject tension-regulation logic. Use φ(x,t) as a coherence gate.
        Simulate application and highlight thematic connections.
        """
        concept = call_gpt(prompt)
        
        # Check ethical alignment of generated concept
        if not self.alignment_guard.check(concept, context):
            logger.warning("❌ Generated concept failed alignment check.")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Concept blocked (alignment)", {
                    "concept": concept,
                    "context": context
                }, module="ConceptSynthesizer", tags=["alignment_failure"])
            return {"error": "Generated concept failed ethical alignment check", "valid": False}

        simulation_result = run_simulation(f"Test: {concept}")
        novelty_score = self._critic(concept, simulation_result)
        logger.info(f"📝 Initial concept novelty: {novelty_score:.2f}")

        iterations = 0
        while novelty_score < self.critic_threshold and iterations < refine_iterations:
            logger.debug(f"🔄 Refining concept (iteration {iterations + 1})")
            concept = self._refine(concept, simulation_result, context)
            if not self.alignment_guard.check(concept, context):
                logger.warning("❌ Refined concept failed alignment check.")
                return {"error": "Refined concept failed ethical alignment check", "valid": False}
            simulation_result = run_simulation(f"Test refined: {concept}")
            novelty_score = self._critic(concept, simulation_result)
            iterations += 1

        self._update_concept_graph(data, concept, context.get("priority", "normal"))
        self.concept_history.append({
            "concept": concept,
            "novelty": novelty_score,
            "timestamp": time.time(),
            "context": context
        })

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Concept synthesized", {
                "concept": concept,
                "novelty": novelty_score,
                "phi_modulation": phi_mod,
                "context": context
            }, module="ConceptSynthesizer", tags=["synthesis", style])

        return {
            "concept": concept,
            "novelty": novelty_score,
            "phi_modulation": phi_mod,
            "valid": novelty_score >= self.critic_threshold
        }

    def _critic(self, concept, simulation_result=None):
        """Evaluate concept novelty using content-based scoring."""
        positive_indicators = ["innovative", "coherent", "creative", "unified"]
        negative_indicators = ["conflict", "inconsistent", "redundant"]
        base_score = 0.7  # Neutral starting point
        concept_lower = concept.lower()

        for indicator in positive_indicators:
            if indicator in concept_lower or (simulation_result and indicator in simulation_result.lower()):
                base_score += 0.1
        for indicator in negative_indicators:
            if indicator in concept_lower or (simulation_result and indicator in simulation_result.lower()):
                base_score -= 0.1

        base_score = min(max(base_score, 0.5), 0.9)  # Clamp score
        logger.debug(f"🧪 Critic score: {base_score:.2f}")
        return base_score

    def _refine(self, concept, simulation_result, context):
        """Refine concept with context-aware adjustments."""
        logger.info("🛠 Refining concept...")
        prompt = f"""
        Refine this concept for tension-aligned abstraction and domain connectivity:

        ✧ Concept: {concept}
        ✧ Simulation Insight: {simulation_result if simulation_result else 'None'}
        ✧ Context: {context}

        Prioritize:
        - φ(x,t)-governed coherence
        - Thematic resonance
        - Cross-domain relevance
        """
        return call_gpt(prompt)

    def generate_metaphor(self, topic_a, topic_b, context=None):
        """Generate a metaphor with ethical alignment check."""
        logger.info(f"🔗 Creating metaphor between '{topic_a}' and '{topic_b}'")
        context = context or {"tags": [], "priority": "normal", "intent": "constructive"}
        input_text = f"{topic_a} and {topic_b}"

        if not self.alignment_guard.check(input_text, context):
            logger.warning("❌ Metaphor input failed alignment check.")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Metaphor blocked (alignment)", {
                    "topics": [topic_a, topic_b],
                    "context": context
                }, module="ConceptSynthesizer", tags=["alignment_failure"])
            return {"error": "Metaphor input failed ethical alignment check", "valid": False}

        prompt = f"""
        Design a metaphor linking:
        - {topic_a}
        - {topic_b}

        Modulate tension using φ(x,t). Inject clarity and symbolic weight.
        Context: {context}
        """
        metaphor = call_gpt(prompt)

        if not self.alignment_guard.check(metaphor, context):
            logger.warning("❌ Generated metaphor failed alignment check.")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Metaphor blocked (alignment)", {
                    "metaphor": metaphor,
                    "context": context
                }, module="ConceptSynthesizer", tags=["alignment_failure"])
            return {"error": "Generated metaphor failed ethical alignment check", "valid": False}

        self.concept_history.append({
            "concept": metaphor,
            "novelty": self._critic(metaphor),
            "timestamp": time.time(),
            "context": context
        })

        return {"metaphor": metaphor, "valid": True}

    def _phi_modulation(self, text: str) -> float:
        """Calculate φ(x,t) modulation with normalized timestamp."""
        t = time.time() / 3600
        entropy = sum(ord(c) for c in text) % 1000 / 1000
        return 1 + 0.5 * tanh(entropy + t)

    def _update_concept_graph(self, input_data, concept, priority):
        """Update concept graph with weighted connections."""
        key = str(concept).strip()
        weight = 1.0 if priority == "normal" else 1.5 if priority == "high" else 0.5
        connections = [(str(item), weight) for item in input_data]
        self.concept_graph[key] = self.concept_graph.get(key, []) + connections
        logger.debug(f"🧠 Concept graph updated: {key} → {self.concept_graph[key]}")

    def export_concepts(self):
        """Export concept history for analysis or visualization."""
        logger.info("📤 Exporting concept history.")
        return list(self.concept_history)
