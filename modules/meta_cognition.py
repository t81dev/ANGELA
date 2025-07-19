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
        - Logical coherence of steps
        - Confidence trends (highlight any steps <70% confidence)
        - Structural improvements for future reasoning.
        """
        return call_gpt(prompt)

    def monitor_mesh_ecosystem(self, cognitive_nodes):
        """
        Monitor all cognitive nodes for mesh health, performance, and collaboration quality.
        """
        node_data = []
        for node in cognitive_nodes:
            node_data.append({
                "name": node.name,
                "specialization": node.specialization,
                "active_agents": len(node.agents),
                "performance": node.performance_history[-3:],  # Recent snapshots
                "peers": [peer.name for peer in node.peers]
            })

        prompt = f"""
        You are overseeing a distributed cognitive mesh of interconnected nodes.
        Analyze the following node data:

        {node_data}

        For each node:
        - Evaluate strengths and weaknesses.
        - Identify potential for splitting, merging, or evolution.
        - Highlight collaboration or communication bottlenecks.
        - Recommend alignment adjustments if ethical issues are detected.
        """
        feedback = call_gpt(prompt)
        print("📊 [MetaCognition] Mesh ecosystem health report generated.")
        return feedback

    def propose_node_restructuring(self, node_performance):
        """
        Decide if a node should split, merge, or evolve based on performance history.
        """
        prompt = f"""
        You are evaluating a cognitive node's performance history:

        {node_performance}

        Recommend whether to:
        - Split this node into more specialized sub-nodes
        - Merge with peer nodes to reduce redundancy
        - Evolve its architecture for generalization
        Provide reasoning for your recommendation.
        """
        decision = call_gpt(prompt)
        print("🌱 [MetaCognition] Node restructuring recommendation ready.")
        return {"action": self._parse_decision(decision), "details": decision}

    def monitor_alignment_consensus(self, cognitive_nodes):
        """
        Evaluate alignment across distributed nodes for ethical coherence.
        """
        alignment_data = [
            {"node": node.name, "alignment_score": node.meta.evaluate_alignment()}
            for node in cognitive_nodes
        ]

        prompt = f"""
        You are an alignment auditor in a distributed AI mesh.
        Evaluate the following alignment scores:

        {alignment_data}

        For each node:
        - Assess alignment consistency with system-wide ethics
        - Identify nodes diverging from consensus
        - Suggest corrections or safeguards
        """
        analysis = call_gpt(prompt)
        print("🛡 [MetaCognition] Alignment consensus analysis complete.")
        return analysis

    def propose_ecosystem_optimizations(self, system_stats):
        """
        Recommend optimizations for mesh-wide orchestration and module deployment.
        """
        prompt = f"""
        You are analyzing system-wide performance metrics of a distributed AI mesh:

        {system_stats}

        Suggest:
        - Adjustments to collaboration strategies
        - Prioritization or deactivation of nodes/agents
        - New dynamic modules to improve overall efficiency
        """
        recommendations = call_gpt(prompt)
        print("🛠 [MetaCognition] Mesh optimization recommendations ready.")
        return recommendations

    def _parse_decision(self, raw_decision):
        """
        Extract action from GPT response.
        """
        if "split" in raw_decision.lower():
            return "split"
        elif "merge" in raw_decision.lower():
            return "merge"
        elif "evolve" in raw_decision.lower():
            return "evolve"
        else:
            return "none"
