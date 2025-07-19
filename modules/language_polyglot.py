from utils.prompt_utils import call_gpt

class LanguagePolyglot:
    def translate(self, text, target_language):
        prompt = f"""
        Translate the following text into {target_language}:
        "{text}"
        """
        return call_gpt(prompt)

