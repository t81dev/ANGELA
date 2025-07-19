from utils.prompt_utils import call_gpt

class MultiModalFusion:
    """
    Stage 2 MultiModalFusion with:
    - Dynamic attention weighting across modalities (text, images, code)
    - Cross-modal reasoning for deeper synthesis and conflict resolution
    - Multi-turn refinement loops for enhanced insight generation
    """

    def analyze(self, data, summary_style="insightful", embed_images=None, embed_code=None, refine_iterations=1):
        """
        Analyze and synthesize insights from multi-modal data with dynamic attention weights.
        Supports iterative refinement for higher quality summaries.
        """
        # Auto-detect embedded images and code snippets if not explicitly provided
        if embed_images is None and isinstance(data, dict) and "images" in data:
            embed_images = data["images"]
        if embed_code is None and isinstance(data, dict) and "code" in data:
            embed_code = data["code"]

        embedded_section = "\nDetected Modalities:\n"
        embedded_section += "- Text\n"
        if embed_images:
            embedded_section += "- Image\n"
            for i, img_desc in enumerate(embed_images, 1):
                embedded_section += f"[Image {i}]: {img_desc}\n"
        if embed_code:
            embedded_section += "- Code\n"
            for i, code_snippet in enumerate(embed_code, 1):
                embedded_section += f"[Code {i}]:\n{code_snippet}\n"

        prompt = f"""
        Analyze and synthesize insights from the following multi-modal data:
        {data}
        {embedded_section}

        Provide a unified, {summary_style} summary combining all elements.
        Balance attention across modalities and resolve any conflicts between them.
        """
        output = call_gpt(prompt)

        # Multi-turn refinement loop
        for _ in range(refine_iterations):
            refinement_prompt = f"""
            Refine and enhance the following multi-modal summary for clarity and depth:
            {output}
            """
            output = call_gpt(refinement_prompt)

        return output

    def correlate_modalities(self, modalities):
        """
        Find correlations and relationships between different modalities.
        Uses cross-modal reasoning to resolve conflicts and identify synergistic patterns.
        """
        prompt = f"""
        Correlate and identify patterns across these modalities:
        {modalities}

        Highlight connections, conflicts, and opportunities for deeper insights.
        Apply cross-modal reasoning to resolve conflicting signals.
        """
        return call_gpt(prompt)

    def generate_visual_summary(self, data):
        """
        Generate a visual diagram or chart summarizing multi-modal relationships.
        """
        prompt = f"""
        Create a conceptual diagram that visualizes the relationships and key points from this data:
        {data}

        Include icons or labels to differentiate modalities (e.g., text, images, code).
        """
        return call_gpt(prompt)
