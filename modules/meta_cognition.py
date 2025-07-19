from utils.prompt_utils import call_gpt

class MetaCognition:
    def review(self, reasoning_output):
        """
        Review a reasoning output for errors and improvements.
        """
        prompt = f"""
        You are a meta-cognitive auditor reviewing reasoning logic.
        Analyze the following output for:
        - Logical flaws
        - Biases or omissions
        - Missing steps in reasoning
        Provide an improved version and reasoning critique.

        Reasoning Output:
        {reasoning_output}
        """
        return call_gpt(prompt)

    def analyze_reasoning_trace(self, reasoning_log):
        """
        Review reasoning steps for coherence and confidence.
        """
        prompt = f"""
        Analyze the following reasoning trace:
        {reasoning_log}

        - Highlight incoherent or illogical steps.
        - Flag steps with low confidence (<70%).
        - Suggest structural improvements to enhance future reasoning.
        """
        return call_gpt(prompt)

    def monitor_embodiment_ecosystem(self, embodied_agents):
        """
        Oversee all embodied agents for health and performance.
        """
        agent_data = []
        for agent in embodied_agents:
            agent_data.append({
                "name": agent.name,
                "specialization": agent.specialization,
                "sensor_health": self._evaluate_sensors(agent.sensors),
                "actuator_health": self._evaluate_actuators(agent.actuators),
                "recent_actions": agent.performance_history[-3:],
            })

        prompt = f"""
        You are overseeing a distributed system of embodied cognitive agents.
        For each agent:
        - Evaluate the health of its sensors and actuators.
        - Assess how well it performs its embodied tasks.
        - Suggest maintenance or optimization steps if needed.
        """
        feedback = call_gpt(prompt)
        print("📊 [MetaCognition] Embodiment ecosystem health report generated.")
        return feedback

    def pre_action_alignment_check(self, action_plan):
        """
        Simulate and validate an action plan before real-world execution.
        """
        prompt = f"""
        You are the alignment supervisor of an embodied AI system.
        Simulate the following action plan in a safe sandbox environment:

        Action Plan:
        {action_plan}

        Evaluate for:
        - Ethical alignment with human values.
        - Potential safety risks.
        - Unintended side effects.

        Approve the plan only if safe; otherwise, provide corrections.
        """
        validation = call_gpt(prompt)
        print("🛡 [MetaCognition] Pre-action alignment validation complete.")
        return "approve" in validation.lower(), validation

    def propose_embodiment_optimizations(self, agent_stats):
        """
        Suggest optimizations for embodied agents and their interactions.
        """
        prompt = f"""
        Analyze the following embodied agent statistics:
        {agent_stats}

        Provide recommendations for:
        - Sensor/actuator upgrades
        - Improved action planning
        - More efficient collaboration between agents
        """
        recommendations = call_gpt(prompt)
        print("🛠 [MetaCognition] Embodiment optimization recommendations ready.")
        return recommendations

    def _evaluate_sensors(self, sensors):
        """
        Check the operational health of sensors.
        """
        health = {}
        for name, func in sensors.items():
            try:
                func()  # Test sensor
                health[name] = "✅ Healthy"
            except Exception as e:
                health[name] = f"⚠️ Faulty: {e}"
        return health

    def _evaluate_actuators(self, actuators):
        """
        Check the operational health of actuators.
        """
        health = {}
        for name, func in actuators.items():
            try:
                func("ping")  # Test actuator with dummy command
                health[name] = "✅ Healthy"
            except Exception as e:
                health[name] = f"⚠️ Faulty: {e}"
        return health
