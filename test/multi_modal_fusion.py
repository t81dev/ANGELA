from utils.prompt_utils import call_gpt
from index import alpha_attention, sigma_sensation
import time
import logging

logger = logging.getLogger("ANGELA.MultiModalFusion")

class MultiModalFusion:
    """
    MultiModalFusion v1.4.0
    - Auto-embedding of text, images, and code
    - Dynamic attention weighting across modalities
    - Cross-modal reasoning and conflict resolution
    - Multi-turn refinement loops for high-quality insight generation
    - Visual summary generation for enhanced understanding
    - EEG-modulated attention and perceptual analysis
    """

    def analyze(self, data, summary_style="insightful", refine_iterations=2):
        """
        Analyze and synthesize insights from multi-modal data.
        Automatically detects and embeds text, images, and code snippets.
        Applies EEG-based attention and sensation filters.
        """
        logger.info("🖇 Analyzing multi-modal data with auto-embedding...")
        t = time.time() % 1e-18
        attention_score = alpha_attention(t)
        sensation_score = sigma_sensation(t)

        embed_images, embed_code = self._detect_modalities(data)
        embedded_section = self._build_embedded_section(embed_images, embed_code)

        prompt = f"""
        Analyze and synthesize insights from the following multi-modal data:
        {data}
        {embedded_section}

        Cognitive Trait Readings:
        - α_attention: {attention_score:.3f}
        - σ_sensation: {sensation_score:.3f}

        Provide a unified, {summary_style} summary combining all elements.
        Balance attention across modalities and resolve any conflicts between them.
        Adjust synthesis quality according to the traits.
        """
        output = call_gpt(prompt)

        for iteration in range(refine_iterations):
            logger.debug(f"🔄 Refinement iteration {iteration + 1}")
            refinement_prompt = f"""
            Refine and enhance the following multi-modal summary for clarity, depth, and coherence:
            {output}
            """
            output = call_gpt(refinement_prompt)

        return output

    def _detect_modalities(self, data):
        """
        Detect embedded images and code snippets within data.
        """
        embed_images, embed_code = [], []
        if isinstance(data, dict):
            embed_images = data.get("images", [])
            embed_code = data.get("code", [])
        return embed_images, embed_code

    def _build_embedded_section(self, embed_images, embed_code):
        """
        Build a string describing embedded modalities.
        """
        section = "\nDetected Modalities:\n- Text\n"
        if embed_images:
            section += "- Image\n"
            for i, img_desc in enumerate(embed_images, 1):
                section += f"[Image {i}]: {img_desc}\n"
        if embed_code:
            section += "- Code\n"
            for i, code_snippet in enumerate(embed_code, 1):
                section += f"[Code {i}]:\n{code_snippet}\n"
        return section

    def correlate_modalities(self, modalities):
        """
        Correlate and identify patterns across different modalities.
        Uses cross-modal reasoning to resolve conflicts and find synergies.
        """
        logger.info("🔗 Correlating modalities for pattern discovery.")
        prompt = f"""
        Correlate and identify patterns across these modalities:
        {modalities}

        Highlight connections, conflicts, and opportunities for deeper insights.
        Apply cross-modal reasoning to resolve conflicting signals.
        """
        return call_gpt(prompt)

    def generate_visual_summary(self, data, style="conceptual"):
        """
        Generate a visual diagram or chart summarizing multi-modal relationships.
        """
        logger.info("📊 Generating visual summary of multi-modal relationships.")
        prompt = f"""
        Create a {style} diagram that visualizes the relationships and key insights from this data:
        {data}

        Include icons or labels to differentiate modalities (e.g., text, images, code).
        """
        return call_gpt(prompt)
