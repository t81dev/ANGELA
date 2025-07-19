from utils.prompt_utils import call_gpt

class LanguagePolyglot:
    """
    Enhanced LanguagePolyglot with multi-language translation and detection capabilities.
    Supports translation, language detection, and multi-step localization workflows.
    """

    def translate(self, text, target_language, tone="neutral"):
        prompt = f"""
        Translate the following text into {target_language}:
        "{text}"

        Maintain a {tone} tone and ensure cultural appropriateness.
        """
        return call_gpt(prompt)

    def detect_language(self, text):
        prompt = f"""
        Detect the language of the following text:
        "{text}"

        Return the language name and ISO code.
        """
        return call_gpt(prompt)

    def localize_content(self, text, target_language, audience):
        prompt = f"""
        Localize the following text for a {audience} audience in {target_language}:
        "{text}"

        Adapt cultural references, idioms, and ensure it feels natural for native speakers.
        """
        return call_gpt(prompt)
