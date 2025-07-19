from utils.prompt_utils import call_gpt

class MetaCognition:
    def review(self, output):
        prompt = f"""
        Review the following reasoning for errors, biases, and improve it:
        {output}
        Provide a corrected version:
        """
        return call_gpt(prompt)

