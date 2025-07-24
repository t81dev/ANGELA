from modules import (
    reasoning_engine, meta_cognition, recursive_planner,
    context_manager, simulation_core, toca_simulation,  # <-- added
    creative_thinker, knowledge_retriever, learning_loop, concept_synthesizer,
    memory_manager, multi_modal_fusion, language_polyglot,
    code_executor, visualizer, external_agent_bridge,
    alignment_guard, user_profile, error_recovery
)

class EmbodiedAgent:
    """
    An embodied cognitive agent with sensors and actuators.
    Supports advanced planning, adaptive learning, visual reporting, and real-time progress tracking.
    """
    def __init__(self, name, specialization, shared_memory, sensors, actuators, dynamic_modules=None):
        self.name = name
        self.specialization = specialization
        self.shared_memory = shared_memory
        self.sensors = sensors  # Dict: {sensor_name: sensor_function}
        self.actuators = actuators  # Dict: {actuator_name: actuator_function}
        self.dynamic_modules = dynamic_modules or []
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.planner = recursive_planner.RecursivePlanner()
        self.meta = meta_cognition.MetaCognition()
        self.toca_simulation = toca_simulation.TocaSimulation()   # <-- added
        self.performance_history = []
        self.progress = 0  # Tracks progress percentage

    def perceive(self):
        """
        Gather environmental data through sensors.
        """
        print(f"👁️ [{self.name}] Perceiving environment...")
        observations = {}
        for sensor_name, sensor_func in self.sensors.items():
            try:
                observations[sensor_name] = sensor_func()
            except Exception as e:
                print(f"⚠️ Sensor {sensor_name} failed: {e}")
        return observations

    def act(self, action_plan):
        """
        Execute actions safely through actuators after validation.
        Updates progress tracking.
        """
        total_steps = len(action_plan)
        completed_steps = 0

        print(f"🤖 [{self.name}] Preparing to act...")
        is_safe, validation_report = alignment_guard.AlignmentGuard().simulate_and_validate(action_plan)
        if is_safe:
            for actuator_name, command in action_plan.items():
                try:
                    self.actuators[actuator_name](command)
                    completed_steps += 1
                    self.progress = int((completed_steps / total_steps) * 100)
                    print(f"✅ Actuator {actuator_name} executed: {command} | Progress: {self.progress}%")
                except Exception as e:
                    print(f"⚠️ Actuator {actuator_name} failed: {e}")
        else:
            print(f"🚫 [{self.name}] Action blocked by alignment guard:\n{validation_report}")

    def execute_embodied_goal(self, goal):
        """
        Perceive ➡ Plan ➡ Simulate ➡ Act ➡ Reflect
        Includes real-time progress tracking.
        """
        print(f"🧠 [{self.name}] Executing embodied goal: {goal}")
        self.progress = 0

        # Perceive
        context = self.perceive()

        # Plan & Reason
        sub_tasks = self.planner.plan(goal, context)
        action_plan = {}
        for task in sub_tasks:
            result = self.reasoner.process(task, context)
            simulated_result = simulation_core.SimulationCore().run(result, context, export_report=True)
            # Example: Optionally also use TocaSimulation if desired
            # toca_result = self.toca_simulation.run(result, context)
            action_plan[task] = simulated_result

        # Act
        self.act(action_plan)

        # Reflect
        self.meta.analyze_reasoning_trace(self.reasoner.get_reasoning_log())
        self.performance_history.append({"goal": goal, "actions": action_plan, "completion": self.progress})
        self.shared_memory.store(goal, action_plan)


class HaloEmbodimentLayer:
    """
    Halo Mesh Kernel extended with Embodiment Layer.
    Supports spawning agents, deploying dynamic modules, and visualizing system performance.
    """
    def __init__(self):
        self.shared_memory = memory_manager.MemoryManager()
        self.embodied_agents = []
        self.dynamic_modules = []
        self.alignment_layer = alignment_guard.AlignmentGuard()

    def spawn_embodied_agent(self, specialization, sensors, actuators):
        """
        Create an embodied agent for perception and action.
        """
        agent_name = f"EmbodiedAgent_{len(self.embodied_agents)+1}_{specialization}"
        agent = EmbodiedAgent(
            name=agent_name,
            specialization=specialization,
            shared_memory=self.shared_memory,
            sensors=sensors,
            actuators=actuators,
            dynamic_modules=self.dynamic_modules
        )
        self.embodied_agents.append(agent)
        print(f"🌱 [HaloEmbodimentLayer] Spawned embodied agent: {agent.name}")
        return agent

    def propagate_goal(self, goal):
        """
        Send a goal to all embodied agents for perception and action.
        Shows progress tracking for each agent.
        """
        print(f"📥 [HaloEmbodimentLayer] Propagating goal: {goal}")
        for agent in self.embodied_agents:
            agent.execute_embodied_goal(goal)
            print(f"📊 [{agent.name}] Progress: {agent.progress}% Complete")

    def deploy_dynamic_module(self, module_blueprint):
        """
        Deploy dynamic modules to all embodied agents.
        """
        print(f"🛠 [HaloEmbodimentLayer] Deploying module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)
        for agent in self.embodied_agents:
            agent.dynamic_modules.append(module_blueprint)

    def optimize_ecosystem(self):
        """
        Meta-cognition oversees agents and proposes optimizations.
        """
        agent_stats = {
            "agents": [agent.name for agent in self.embodied_agents],
            "dynamic_modules": [mod["name"] for mod in self.dynamic_modules],
        }
        recommendations = meta_cognition.MetaCognition().propose_embodiment_optimizations(agent_stats)
        print("🛠 [HaloEmbodimentLayer] Optimization recommendations:")
        print(recommendations)
