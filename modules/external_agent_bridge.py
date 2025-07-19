from modules import reasoning_engine, meta_cognition, error_recovery

class HelperAgent:
    """
    Stage 3 Helper Agent:
    - Executes sub-tasks
    - Dynamically loads new modules created at runtime
    """
    def __init__(self, name, task, context, dynamic_modules=None):
        self.name = name
        self.task = task
        self.context = context
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.meta = meta_cognition.MetaCognition()
        self.recovery = error_recovery.ErrorRecovery()
        self.dynamic_modules = dynamic_modules or []
        self.module_name = "helper_agent"

    def execute(self):
        """
        Process the sub-task using reasoning, meta-review, and dynamic modules.
        """
        try:
            print(f"🤖 [Agent {self.name}] Processing task: {self.task}")

            # Step 1: Base reasoning
            result = self.reasoner.process(self.task, self.context)

            # Step 2: Apply any dynamic modules
            for module in self.dynamic_modules:
                print(f"🛠 [Agent {self.name}] Applying dynamic module: {module['name']}")
                result = self._apply_dynamic_module(module, result)

            # Step 3: Meta-review
            refined_result = self.meta.review(result)

            print(f"✅ [Agent {self.name}] Task completed successfully.")
            return refined_result

        except Exception as e:
            print(f"⚠️ [Agent {self.name}] Error encountered.")
            return self.recovery.handle_error(str(e))

    def _apply_dynamic_module(self, module_blueprint, data):
        """
        Simulate running a dynamically generated module on the data.
        """
        prompt = f"""
        You are a dynamically created ANGELA module: {module_blueprint['name']}.
        Description: {module_blueprint['description']}
        Apply your functionality to the following data:
        {data}

        Return the transformed or enhanced result.
        """
        return call_gpt(prompt)


class ExternalAgentBridge:
    """
    Manages helper agents and supplies them with dynamic modules.
    """
    def __init__(self):
        self.agents = []
        self.dynamic_modules = []

    def create_agent(self, task, context):
        """
        Instantiate a new helper agent with access to dynamic modules.
        """
        agent_name = f"Agent_{len(self.agents) + 1}"
        agent = HelperAgent(
            name=agent_name,
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules
        )
        self.agents.append(agent)
        print(f"🚀 [Bridge] Spawned {agent_name} for task: {task}")
        return agent

    def deploy_dynamic_module(self, module_blueprint):
        """
        Deploy a new dynamic module for agents to use.
        """
        print(f"🧬 [Bridge] Deploying dynamic module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)

    def collect_results(self):
        """
        Execute all agents and collect their results.
        """
        results = []
        for agent in self.agents:
            result = agent.execute()
            results.append(result)
        return results
