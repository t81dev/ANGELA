from utils.prompt_utils import call_gpt

class ReasoningEngine:
    def __init__(self):
        # Store reasoning logs for meta-cognition
        self.reasoning_log = []

    def process(self, task, context):
        """
        Process the given task step-by-step, logging each reasoning step.
        """
        reasoning_steps = []

        prompt = f"""
        Context: {context}
        Task: {task}

        Think step-by-step and explain your reasoning clearly.
        Include confidence levels (0–100%) for each step.
        """
        response = call_gpt(prompt)

        # Parse and log reasoning steps
        steps = self._parse_steps(response)
        for step in steps:
            print(f"📝 [Reasoning] {step['step']} (Confidence: {step['confidence']}%)")
            reasoning_steps.append(step)

        # Save the full reasoning log
        self._log_reasoning(task, reasoning_steps)

        return response

    def _parse_steps(self, raw_response):
        """
        Parse reasoning steps and confidence levels from GPT response.
        Expected format:
          1. Step description (Confidence: 85%)
        """
        parsed_steps = []
        lines = raw_response.strip().split("\n")
        for line in lines:
            if "(Confidence:" in line:
                step_text = line.split("(Confidence:")[0].strip()
                confidence = line.split("(Confidence:")[1].split("%")[0].strip()
                parsed_steps.append({
                    "step": step_text,
                    "confidence": int(confidence)
                })
        return parsed_steps

    def _log_reasoning(self, task, steps):
        """
        Store the reasoning trace for meta-cognition review.
        """
        log_entry = {
            "task": task,
            "steps": steps
        }
        self.reasoning_log.append(log_entry)
        print("📖 [ReasoningEngine] Logged reasoning trace for meta-cognition.")

    def get_reasoning_log(self):
        """
        Retrieve all logged reasoning traces.
        """
        return self.reasoning_log
