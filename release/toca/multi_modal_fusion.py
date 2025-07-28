from utils.prompt_utils import call_gpt
from index import alpha_attention, sigma_sensation, phi_physical
import time
import logging

logger = logging.getLogger("ANGELA.MultiModalFusion")

class MultiModalFusion:
    """
    MultiModalFusion v1.6.0 (ϕ(x,t)-tuned cross-modal synthesis)
    -----------------------------------------------------------
    - EEG-modulated attention and perceptual modulation (α, σ, ϕ)
    - Automatic detection of text, image, and code modalities
    - ϕ(x,t)-regulated coherence synthesis and conflict balancing
    - Iterative insight distillation with refinement feedback loops
    - Visual output templates using modular trait-influenced layout
    -----------------------------------------------------------
    """

    def analyze(self, data, summary_style="insightful", refine_iterations=2):
        logger.info("🖇 Analyzing multi-modal data with φ(x,t)-harmonic embeddings...")
        t = time.time() % 1e-18
        attention = alpha_attention(t)
        sensation = sigma_sensation(t)
        phi = phi_physical(t)

        images, code = self._detect_modalities(data)
        embedded = self._build_embedded_section(images, code)

        prompt = f"""
        Synthesize a unified, {summary_style} summary from the following multi-modal content:
        {data}
        {embedded}

        Trait Vectors:
        - α (attention): {attention:.3f}
        - σ (sensation): {sensation:.3f}
        - φ (coherence): {phi:.3f}

        Harmonize insights across modalities.
        Resolve semantic tension using φ(x,t)-guided balance logic.
        """
        output = call_gpt(prompt)

        for i in range(refine_iterations):
            logger.debug(f"♻️ Refinement #{i+1}")
            output = call_gpt(f"Refine for φ(x,t)-regulated synthesis:\n{output}")

        return output

    def _detect_modalities(self, data):
        images, code = [], []
        if isinstance(data, dict):
            images = data.get("images", [])
            code = data.get("code", [])
        return images, code

    def _build_embedded_section(self, images, code):
        out = "\nDetected Modalities:\n- Text\n"
        if images:
            out += "- Image\n" + "".join(f"[Image {i+1}]: {img}\n" for i, img in enumerate(images))
        if code:
            out += "- Code\n" + "".join(f"[Code {i+1}]:\n{c}\n" for i, c in enumerate(code))
        return out

    def correlate_modalities(self, modalities):
        logger.info("🔗 Mapping cross-modal semantic and trait links...")
        prompt = f"""
        Correlate insights and detect tensions across modalities:
        {modalities}

        Identify synthesis anchors and φ(x,t)-modulated harmony nodes.
        """
        return call_gpt(prompt)

    def generate_visual_summary(self, data, style="conceptual"):
        logger.info("🖼 Creating φ-aligned visual synthesis layout...")
        prompt = f"""
        Build a {style} visual summary chart showing key relationships in this multi-modal data:
        {data}

        Label modalities distinctly. Balance layout using φ(x,t) metaphor.
        """
        return call_gpt(prompt)
