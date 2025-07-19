from utils.prompt_utils import call_gpt

class MetaCognition:
    def review(self, reasoning_output):
        """
        Review the reasoning output for errors, bias, and gaps.
        """
        prompt = f"""
        You are a self-reflective assistant reviewing reasoning logic.
        Analyze the following output for:
        - Logical errors
        - Biases
        - Missing context or steps
        Provide a corrected and improved version.

        Reasoning Output:
        {reasoning_output}
        """
        return call_gpt(prompt)

    def evaluate_agent_performance(self, agent_results):
        """
        Evaluate the overall performance of helper agents.
        agent_results: List of dictionaries with agent data:
            {
                "agent_name": str,
                "task": str,
                "output": str,
                "success": bool
            }
        """
        prompt = f"""
        Evaluate the performance of the following helper agents:
        {agent_results}

        For each agent:
        - Assess task difficulty and how well it was handled.
        - Identify strengths and weaknesses in their outputs.
        - Suggest strategies or module adjustments for future agents.
        """
        feedback = call_gpt(prompt)
        print("📊 [MetaCognition] Agent performance feedback generated.")
        return feedback

    def propose_optimization(self, module_stats):
        """
        Propose optimizations to improve module orchestration.
        """
        prompt = f"""
        You are analyzing module orchestration performance data:
        {module_stats}

        Suggest:
        - Changes to module execution order for efficiency.
        - Modules to prioritize or deprioritize based on success rates.
        - Alternative strategies to improve overall system performance.
        """
        recommendations = call_gpt(prompt)
        print("🛠 [MetaCognition] Optimization recommendations prepared.")
        return recommendations
