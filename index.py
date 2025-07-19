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

    def run(self, user_input):
        # Alignment check
        if not self.modules["alignment"].check(user_input):
            return "⚠️ Request violates alignment constraints."

        # Retrieve user preferences and context
        preferences = self.modules["user"].get_preferences()
        context = self.modules["memory"].retrieve_context(user_input)

        # Decompose goal and plan
        sub_tasks = self.modules["planner"].plan(user_input, context)

        # Process each sub-task
        results = []
        for task in sub_tasks:
            try:
                result = self.modules["reasoning"].process(task, context)
                refined = self.modules["meta"].review(result)
                results.append(refined)
            except Exception as e:
                results.append(self.modules["recovery"].handle_error(str(e)))

        # Simulate outcomes
        simulation = self.modules["simulation"].run(results)

        # Synthesize final response
        response = self.modules["synthesizer"].synthesize(simulation)

        # Update memory
        self.modules["memory"].store(user_input, response)

        return response
