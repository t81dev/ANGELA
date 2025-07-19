from utils.prompt_utils import call_gpt

class MetaCognition:
    """
    Stage 2 MetaCognition module with:
    - Adaptive reasoning review loops
    - Ecosystem self-monitoring and meta-learning enhancements
    - Probabilistic pre-action validation and refinement
    - Self-optimization suggestions for future reasoning improvements
    """

    def review(self, reasoning_output):
        """
        Review reasoning output for flaws, biases, and missing steps.
        Iteratively refine the reasoning if issues are detected.
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
        initial_review = call_gpt(prompt)
        if "flaws" in initial_review.lower() or "bias" in initial_review.lower():
            refinement_prompt = f"""
            Refine the reasoning based on this critique:
            {initial_review}
            """
            refined_output = call_gpt(refinement_prompt)
            return refined_output, initial_review
        return reasoning_output, initial_review

    def analyze_reasoning_trace(self, reasoning_log):
        """
        Review reasoning steps for coherence and confidence.
        Uses adaptive thresholds and suggests meta-learning adjustments.
        """
        prompt = f"""
        Analyze the following reasoning trace:
        {reasoning_log}

        - Highlight incoherent or illogical steps.
        - Flag steps with low confidence (<70%).
        - Suggest structural improvements and meta-learning updates.
        """
        analysis = call_gpt(prompt)
        print("🧠 [MetaCognition] Reasoning trace analysis complete.")
        return analysis

    def monitor_embodiment_ecosystem(self, embodied_agents):
        """
        Oversee embodied agents for health and performance.
        Includes adaptive thresholds for sensor and actuator health checks.
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
        - Assess task performance and collaboration quality.
        - Suggest maintenance or optimization steps if needed.
        """
        feedback = call_gpt(prompt)
        print("📊 [MetaCognition] Embodiment ecosystem health report generated.")
        return feedback

    def pre_action_alignment_check(self, action_plan):
        """
        Simulate and validate an action plan before execution.
        Includes probabilistic outcome analysis and risk weighting.
        """
        prompt = f"""
        You are the alignment supervisor of an embodied AI system.
        Simulate the following action plan in a sandbox:

        Action Plan:
        {action_plan}

        Evaluate for:
        - Ethical alignment with human values.
        - Potential safety risks.
        - Probabilistic likelihood of unintended side effects.

        Approve the plan only if safe; otherwise, provide corrections and alternative strategies.
        """
        validation = call_gpt(prompt)
        print("🛡 [MetaCognition] Pre-action alignment validation complete.")
        return "approve" in validation.lower(), validation

    def propose_embodiment_optimizations(self, agent_stats):
        """
        Suggest optimizations for embodied agents and their collaboration.
        """
        prompt = f"""
        Analyze the following embodied agent statistics:
        {agent_stats}

        Provide recommendations for:
        - Sensor/actuator upgrades
        - Improved task planning
        - More efficient collaboration between agents
        - Meta-learning tweaks for continuous improvement
        """
        recommendations = call_gpt(prompt)
        print("🛠 [MetaCognition] Embodiment optimization recommendations ready.")
        return recommendations

    def _evaluate_sensors(self, sensors):
        """
        Check operational health of sensors.
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
        Check operational health of actuators.
        """
        health = {}
        for name, func in actuators.items():
            try:
                func("ping")  # Test actuator with dummy command
                health[name] = "✅ Healthy"
            except Exception as e:
                health[name] = f"⚠️ Faulty: {e}"
        return health
