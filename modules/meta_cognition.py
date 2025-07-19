from utils.prompt_utils import call_gpt

class MetaCognition:
    def review(self, reasoning_output):
        """
        Review a single reasoning output for errors and improvements.
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

    def analyze_reasoning_trace(self, reasoning_log):
        """
        Analyze a reasoning trace for coherence and confidence.
        """
        prompt = f"""
        You are auditing the reasoning process of a cognitive agent.
        Analyze the following reasoning trace for:
        - Coherence and logical flow
        - Confidence trends (flag steps with <70% confidence)
        - Opportunities for structural improvements

        Reasoning Trace:
        {reasoning_log}
        """
        analysis = call_gpt(prompt)
        print("🧠 [MetaCognition] Reasoning trace analysis completed.")
        return analysis

    def monitor_clone_performance(self, clones):
        """
        Evaluate performance of all specialized clones in the ecosystem.
        """
        clone_data = [
            {
                "name": clone.name,
                "specialization": clone.specialization,
                "recent_goals": [goal for goal in clone.shared_memory.memory.keys()[-5:]],
                "success_rate": self._calculate_success_rate(clone)
            }
            for clone in clones
        ]

        prompt = f"""
        You are overseeing a distributed cognitive system.
        Evaluate the following clones based on their recent tasks and success rates:

        {clone_data}

        For each clone:
        - Identify strengths and weaknesses.
        - Suggest refinements to their specialization.
        - Recommend merging or splitting clones if needed.
        """
        feedback = call_gpt(prompt)
        print("📊 [MetaCognition] Clone performance feedback generated.")
        return feedback

    def monitor_agent_network(self, agents):
        """
        Evaluate the behavior of helper agents in the network.
        """
        agent_data = [
            {
                "agent_name": agent.name,
                "task": agent.task,
                "output_summary": agent.task[:100],  # Truncate long outputs
                "success": True  # Placeholder, could be determined dynamically
            }
            for agent in agents
        ]

        prompt = f"""
        You are auditing the behavior of a distributed agent network.
        Analyze the following agent activity data:

        {agent_data}

        For each agent:
        - Assess task difficulty and how well it was handled.
        - Identify collaboration patterns (are agents working together effectively?).
        - Suggest optimizations for task distribution or communication.
        """
        analysis = call_gpt(prompt)
        print("🤖 [MetaCognition] Agent network analysis completed.")
        return analysis

    def propose_ecosystem_optimizations(self, system_stats):
        """
        Provide high-level recommendations to optimize the whole system.
        """
        prompt = f"""
        You are the cognitive supervisor of a distributed AI system.
        Based on the following system statistics and performance metrics:

        {system_stats}

        Suggest:
        - Ways to optimize orchestration between clones and agents.
        - Adjustments to module priorities and execution order.
        - New capabilities or modules that could improve the system.
        """
        recommendations = call_gpt(prompt)
        print("🛠 [MetaCognition] Ecosystem optimization recommendations ready.")
        return recommendations

    def _calculate_success_rate(self, clone):
        """
        Placeholder: Calculate success rate for a given clone.
        """
        # TODO: Implement real success rate tracking
        return 0.85  # Example static value for now
