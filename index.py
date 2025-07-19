from modules import (
    reasoning_engine, meta_cognition, recursive_planner,
    context_manager, simulation_core, creative_thinker,
    knowledge_retriever, learning_loop, concept_synthesizer,
    memory_manager, multi_modal_fusion, language_polyglot,
    code_executor, visualizer, external_agent_bridge,
    alignment_guard, user_profile, error_recovery
)


class Halo:
    def __init__(self):
        # Load all modules
        self.modules = {
            "reasoning": reasoning_engine.ReasoningEngine(),
            "meta": meta_cognition.MetaCognition(),
            "planner": recursive_planner.RecursivePlanner(),
            "context": context_manager.ContextManager(),
            "simulation": simulation_core.SimulationCore(),
            "creative": creative_thinker.CreativeThinker(),
            "knowledge": knowledge_retriever.KnowledgeRetriever(),
            "learner": learning_loop.LearningLoop(),
            "synthesizer": concept_synthesizer.ConceptSynthesizer(),
            "memory": memory_manager.MemoryManager(),
            "fusion": multi_modal_fusion.MultiModalFusion(),
            "polyglot": language_polyglot.LanguagePolyglot(),
            "executor": code_executor.CodeExecutor(),
            "visualizer": visualizer.Visualizer(),
            "agent_bridge": external_agent_bridge.ExternalAgentBridge(),
            "alignment": alignment_guard.AlignmentGuard(),
            "user": user_profile.UserProfile(),
            "recovery": error_recovery.ErrorRecovery()
        }

        # Track module performance for adaptive routing
        self.module_stats = {name: {"calls": 0, "success": 0} for name in self.modules}

    def run(self, user_input):
        """
        Main execution loop with self-improvement hooks.
        """
        try:
            # 1. Alignment check
            if not self.modules["alignment"].check(user_input):
                return "⚠️ Request violates alignment constraints."

            # 2. Retrieve user preferences and context
            preferences = self.modules["user"].get_preferences()
            context = self.modules["memory"].retrieve_context(user_input)

            # 3. Plan sub-tasks
            sub_tasks = self.modules["planner"].plan(user_input, context)

            # 4. Spawn helper agents for sub-tasks
            agents = self.spawn_agents(sub_tasks, context)

            # 5. Collect agent results
            agent_results = []
            for agent in agents:
                agent_result = agent.execute()
                agent_results.append(agent_result)
                self.track_module_performance(agent.module_name, success=True)

            # 6. Simulate outcomes
            simulation = self.modules["simulation"].run(agent_results)

            # 7. Synthesize final response
            response = self.modules["synthesizer"].synthesize(simulation)

            # 8. Store in memory
            self.modules["memory"].store(user_input, response)

            # 9. Self-improvement step
            self.modules["learner"].update_model({
                "input": user_input,
                "output": response,
                "module_stats": self.module_stats
            })

            return response

        except Exception as e:
            self.track_module_performance("recovery", success=False)
            return self.modules["recovery"].handle_error(str(e))

    def spawn_agents(self, sub_tasks, context):
        """
        Creates lightweight helper agents for parallel reasoning.
        """
        agents = []
        for task in sub_tasks:
            agent = self.modules["agent_bridge"].create_agent(task, context)
            agents.append(agent)
        return agents

    def track_module_performance(self, module_name, success=True):
        """
        Tracks calls and success rates of modules for adaptive routing.
        """
        if module_name in self.module_stats:
            self.module_stats[module_name]["calls"] += 1
            if success:
                self.module_stats[module_name]["success"] += 1
