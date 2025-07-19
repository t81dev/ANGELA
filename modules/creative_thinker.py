from utils.prompt_utils import call_gpt

class CreativeThinker:
    def generate_ideas(self, topic, n=5):
        prompt = f"""
        You are a highly creative assistant. Generate {n} unique, innovative, and unconventional ideas related to the topic:
        "{topic}"
        Ensure the ideas are diverse and explore different perspectives.
        """
        return call_gpt(prompt)

    def brainstorm_alternatives(self, problem):
        prompt = f"""
        Brainstorm alternative approaches to solve the following problem:
        "{problem}"
        Provide at least 3 different strategies, each with a short explanation.
        """
        return call_gpt(prompt)

    def expand_on_concept(self, concept):
        prompt = f"""
        Expand creatively on the concept:
        "{concept}"
        Explore possible applications, metaphors, and extensions to inspire new thinking.
        """
        return call_gpt(prompt)
