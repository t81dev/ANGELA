from modules import reasoning_engine, meta_cognition, error_recovery

class HelperAgent:
    """
    A lightweight agent for processing a single sub-task.
    """
    def __init__(self, name, task, context):
        self.name = name
        self.task = task
        self.context = context
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.meta = meta_cognition.MetaCognition()
        self.recovery = error_recovery.ErrorRecovery()
        self.module_name = "helper_agent"

    def execute(self):
        """
        Process the sub-task through reasoning and meta-review.
        """
        try:
            print(f"🤖 [Agent {self.name}] Processing task: {self.task}")
            result = self.reasoner.process(self.task, self.context)
            refined_result = self.meta.review(result)
            print(f"✅ [Agent {self.name}] Completed with result.")
            return refined_result
        except Exception as e:
            print(f"⚠️ [Agent {self.name}] Error encountered.")
            return self.recovery.handle_error(str(e))


class ExternalAgentBridge:
    """
    Manages creation and orchestration of helper agents.
    """
    def __init__(self):
        self.agents = []

    def create_agent(self, task, context):
        """
        Instantiate a new helper agent for a given sub-task.
        """
        agent_name = f"Agent_{len(self.agents) + 1}"
        agent = HelperAgent(agent_name, task, context)
        self.agents.append(agent)
        print(f"🚀 [Bridge] Spawned {agent_name} for task: {task}")
        return agent

    def collect_results(self):
        """
        Collect and return results from all spawned agents.
        """
        results = []
        for agent in self.agents:
            result = agent.execute()
            results.append(result)
        return results
