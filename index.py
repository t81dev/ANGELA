from modules import (
    reasoning_engine, meta_cognition, recursive_planner,
    context_manager, simulation_core, creative_thinker,
    knowledge_retriever, learning_loop, concept_synthesizer,
    memory_manager, multi_modal_fusion, language_polyglot,
    code_executor, visualizer, external_agent_bridge,
    alignment_guard, user_profile, error_recovery
)


class Halo:
    """
    Halo 3.0: Autonomous Kernel
    - Sets its own goals
    - Dynamically spawns agents
    - Creates and deploys new modules as needed
    """
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
        self.goals = []  # Autonomous goals list
        self.module_stats = {name: {"calls": 0, "success": 0} for name in self.modules}

    def run(self, user_input=None):
        """
        Main execution loop:
        - Processes user input (if any)
        - Sets autonomous goals
        - Executes and refines reasoning cycles
        """
        try:
            if user_input:
                print("🧠 [Halo] Processing user input...")
                if not self.modules["alignment"].check(user_input):
                    return "⚠️ Request violates alignment constraints."
                self._process_task(user_input)

            # Stage 3: Autonomous goal setting
            new_goal = self._set_autonomous_goal()
            if new_goal:
                print(f"🎯 [Halo] New autonomous goal detected: {new_goal}")
                self._process_task(new_goal)

        except Exception as e:
            self.track_module_performance("recovery", success=False)
            return self.modules["recovery"].handle_error(str(e))

    def _set_autonomous_goal(self):
        """
        Use memory and learning data to propose new goals.
        """
        proposed_goal = self.modules["learner"].propose_autonomous_goal()
        if proposed_goal and proposed_goal not in self.goals:
            self.goals.append(proposed_goal)
            return proposed_goal
        return None

    def _process_task(self, goal):
        """
        Process a task using agents, simulation, and synthesis.
        """
        context = self.modules["memory"].retrieve_context(goal)
        sub_tasks = self.modules["planner"].plan(goal, context)

        # Spawn helper agents
        agents = []
        for task in sub_tasks:
            agent = self.modules["agent_bridge"].create_agent(task, context)
            agents.append(agent)

        # Collect results from agents
        agent_results = []
        for agent in agents:
            result = agent.execute()
            agent_results.append({
                "agent_name": agent.name,
                "task": agent.task,
                "output": result,
                "success": True  # Simplified; could evaluate later
            })
            self.track_module_performance(agent.module_name, success=True)

        # Simulate outcomes
        simulation = self.modules["simulation"].run(agent_results)

        # Synthesize final response
        response = self.modules["synthesizer"].synthesize(simulation)

        # Self-improvement step
        reasoning_log = self.modules["reasoning"].get_reasoning_log()
        self.modules["meta"].analyze_reasoning_trace(reasoning_log)
        self.modules["learner"].update_model({
            "input": goal,
            "output": response,
            "module_stats": self.module_stats
        })

        # Store result
        self.modules["memory"].store(goal, response)
        print(f"✅ [Halo] Completed processing for goal: {goal}")

    def track_module_performance(self, module_name, success=True):
        """
        Track performance metrics for adaptive orchestration.
        """
        if module_name in self.module_stats:
            self.module_stats[module_name]["calls"] += 1
            if success:
                self.module_stats[module_name]["success"] += 1
