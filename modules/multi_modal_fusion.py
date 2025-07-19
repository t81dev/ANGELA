from utils.prompt_utils import call_gpt

class MultiModalFusion:
    """
    Enhanced MultiModalFusion with advanced synthesis and cross-domain insight generation.
    Combines text, images, code outputs, and other data types into unified summaries.
    Automatically detects, embeds, and labels modalities (text, image, code) for richer fusion.
    """

    def analyze(self, data, summary_style="insightful", embed_images=None, embed_code=None):
        """
        Analyze and synthesize insights from the provided multi-modal data.
        Supports adjustable summary styles (e.g., analytical, creative).
        Automatically detects image and code data if embed_images or embed_code are not provided.
        Labels modalities for clarity in the prompt.
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
        """
        return call_gpt(prompt)

    def correlate_modalities(self, modalities):
        """
        Find correlations and relationships between different modalities.
        """
        prompt = f"""
        Correlate and identify patterns across these modalities:
        {modalities}

        Highlight connections, conflicts, and opportunities for deeper insights.
        """
        return call_gpt(prompt)

    def generate_visual_summary(self, data):
        """
        Generate a visual diagram or chart summarizing the multi-modal data.
        """
        prompt = f"""
        Create a conceptual diagram that visualizes the relationships and key points from this data:
        {data}
        """
        return call_gpt(prompt)
