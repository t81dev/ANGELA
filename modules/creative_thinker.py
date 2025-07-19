from utils.prompt_utils import call_gpt

class CreativeThinker:
    """
    Enhanced CreativeThinker with adjustable creativity levels and multi-modal brainstorming.
    Supports idea generation, alternative brainstorming, and concept expansion with flexible styles.
    """

    def __init__(self, creativity_level="high"):
        self.creativity_level = creativity_level

    def generate_ideas(self, topic, n=5, style="divergent"):
        prompt = f"""
        You are a highly creative assistant operating at a {self.creativity_level} creativity level.
        Generate {n} unique, innovative, and {style} ideas related to the topic:
        "{topic}"
        Ensure the ideas are diverse and explore different perspectives.
        """
        return call_gpt(prompt)

    def brainstorm_alternatives(self, problem, strategies=3):
        prompt = f"""
        Brainstorm {strategies} alternative approaches to solve the following problem:
        "{problem}"
        For each approach, provide a short explanation highlighting its uniqueness.
        """
        return call_gpt(prompt)

    def expand_on_concept(self, concept, depth="deep"):
        prompt = f"""
        Expand creatively on the concept:
        "{concept}"
        Explore possible applications, metaphors, and extensions to inspire new thinking.
        Aim for a {depth} exploration.
        """
        return call_gpt(prompt)
