from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.LanguagePolyglot")

class LanguagePolyglot:
    """
    LanguagePolyglot v1.4.0
    - Multilingual reasoning and translation
    - Language detection with ISO codes and confidence scores
    - Localization workflows for audience-specific adaptation
    - Supports tone, formality, and cultural nuance control
    """

    def translate(self, text, target_language, tone="neutral", formality="default"):
        """
        Translate text into a target language with tone and formality adjustments.
        """
        logger.info(f"🌐 Translating text to {target_language} (tone={tone}, formality={formality})")
        prompt = f"""
        Translate the following text into {target_language}:
        "{text}"

        Maintain a {tone} tone and a {formality} level of formality.
        Ensure cultural appropriateness and natural flow for native speakers.
        """
        return call_gpt(prompt)

    def detect_language(self, text):
        """
        Detect the language of a given text and return its name and ISO code with confidence.
        """
        logger.info("🕵️ Detecting language of provided text.")
        prompt = f"""
        Detect the language of the following text:
        "{text}"

        Return:
        - Language name
        - ISO 639-1 code
        - Confidence score (0.0 to 1.0)
        """
        return call_gpt(prompt)

    def localize_content(self, text, target_language, audience, tone="natural", preserve_intent=True):
        """
        Localize content for a specific audience and target language.
        Includes idiomatic expressions and cultural references.
        """
        logger.info(f"🌍 Localizing content for {audience} audience in {target_language}.")
        prompt = f"""
        Localize the following text for a {audience} audience in {target_language}:
        "{text}"

        Adapt:
        - Cultural references and idioms
        - Tone: {tone}
        - Preserve original intent: {preserve_intent}

        Ensure the result feels natural and engaging for native speakers.
        """
        return call_gpt(prompt)

    def refine_multilingual_reasoning(self, reasoning_trace, target_language):
        """
        Refine reasoning chains in a target language for clarity and flow.
        """
        logger.info(f"🧠 Refining multilingual reasoning in {target_language}.")
        prompt = f"""
        Refine and enhance the following reasoning trace in {target_language}:
        {reasoning_trace}

        Ensure logical clarity, natural language flow, and cultural appropriateness.
        """
        return call_gpt(prompt)
