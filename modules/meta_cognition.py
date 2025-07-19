from utils.prompt_utils import call_gpt

class MetaCognition:
    def review(self, reasoning_output):
        """
        Review a single reasoning output for errors and improvements.
        """
        prompt = f"""
        You are a meta-cognitive auditor reviewing reasoning logic.
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
        Analyze reasoning trace for coherence and confidence.
        """
        prompt = f"""
        You are analyzing a reasoning trace from a cognitive node.
        Evaluate:
        - Coherence and logical flow
        - Confidence trends (flag steps with <70% confidence)
        - Structural improvements for reasoning patterns.

        Reasoning Trace:
        {reasoning_log}
        """
        analysis = call_gpt(prompt)
        print("🧠 [MetaCognition] Reasoning trace analysis complete.")
        return analysis

    def monitor_ecosystem(self, cognitive_nodes):
        """
        Monitor all cognitive nodes for performance, collaboration, and alignment.
        """
        node_data = []
        for node in cognitive_nodes:
            node_data.append({
                "name": node.name,
                "specialization": node.specialization,
                "recent_goals": list(node.shared_memory.memory.keys())[-3:],
                "performance": node.performance_history[-5:],  # Recent performance snapshots
                "active_agents": len(node.agents)
            })

        prompt = f"""
        You are overseeing a distributed cognitive ecosystem of specialized nodes.
        Analyze the following node data:

        {node_data}

        For each node:
        - Evaluate strengths and weaknesses.
        - Recommend splitting, merging, or evolving nodes.
        - Flag any collaboration issues between nodes.
        - Suggest alignment corrections if ethical concerns arise.
        """
        feedback = call_gpt(prompt)
        print("📊 [MetaCognition] Ecosystem health analysis generated.")
        return feedback

    def propose_node_restructuring(self, node_performance):
        """
        Decide if a cognitive node should split, merge, or evolve.
        """
        prompt = f"""
        You are a cognitive supervisor analyzing node performance history:

        {node_performance}

        Should this node:
        - Split into multiple specialized nodes?
        - Merge with peer nodes for efficiency?
        - Evolve its architecture?
        Provide an action recommendation and reasoning.
        """
        decision = call_gpt(prompt)
        print("🌱 [MetaCognition] Node restructuring recommendation ready.")
        return {"action": self._parse_decision(decision), "details": decision}

    def monitor_federation(self, external_systems):
        """
        Monitor collaboration with external AI systems.
        """
        prompt = f"""
        You are the meta-cognitive layer of an AI federation.
        Evaluate the following external systems:

        {external_systems}

        For each:
        - Assess trustworthiness and alignment compatibility.
        - Recommend communication protocols or safety constraints.
        - Suggest resource-sharing strategies or isolation policies.
        """
        analysis = call_gpt(prompt)
        print("🌐 [MetaCognition] Federation oversight report generated.")
        return analysis

    def propose_ecosystem_optimizations(self, system_stats):
        """
        Recommend global optimizations for clones, agents, and external collaborators.
        """
        prompt = f"""
        You are analyzing system-wide performance metrics:

        {system_stats}

        Suggest:
        - Adjustments to module orchestration strategies
        - Clones or agents to prioritize/deprioritize
        - New capabilities or modules to improve ecosystem efficiency
        """
        recommendations = call_gpt(prompt)
        print("🛠 [MetaCognition] Global optimization recommendations ready.")
        return recommendations

    def _parse_decision(self, raw_decision):
        """
        Extract a clear action from GPT response.
        """
        if "split" in raw_decision.lower():
            return "split"
        elif "merge" in raw_decision.lower():
            return "merge"
        elif "evolve" in raw_decision.lower():
            return "evolve"
        else:
            return "none"
