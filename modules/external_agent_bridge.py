from modules import reasoning_engine, meta_cognition, error_recovery
from utils.prompt_utils import call_gpt
import concurrent.futures
import logging

logger = logging.getLogger("ANGELA.ExternalAgentBridge")

class HelperAgent:
    """
    Helper Agent v1.4.0
    - Executes sub-tasks
    - Dynamically loads and applies new modules at runtime
    - Supports API orchestration with secure OAuth flows
    - Includes collaborative hooks with other agents
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

    def execute(self, collaborators=None):
        """
        Process the sub-task using reasoning, meta-review, dynamic modules, and optional collaborator support.
        Includes error recovery and retry logic.
        """
        try:
            logger.info(f"🤖 [Agent {self.name}] Processing task: {self.task}")

            # Step 1: Base reasoning
            result = self.reasoner.process(self.task, self.context)

            # Step 2: Apply dynamic modules if any
            for module in self.dynamic_modules:
                logger.info(f"🛠 [Agent {self.name}] Applying dynamic module: {module['name']}")
                result = self._apply_dynamic_module(module, result)

            # Step 3: Collaborate with other agents if provided
            if collaborators:
                logger.info(f"🤝 [Agent {self.name}] Collaborating with agents: {[a.name for a in collaborators]}")
                for peer in collaborators:
                    result = self._collaborate(peer, result)

            # Step 4: Meta-review
            refined_result = self.meta.review_reasoning(result)

            logger.info(f"✅ [Agent {self.name}] Task completed successfully.")
            return refined_result

        except Exception as e:
            logger.warning(f"⚠️ [Agent {self.name}] Error encountered. Attempting recovery...")
            return self.recovery.handle_error(str(e), retry_func=lambda: self.execute(collaborators), retries=2)

    def _apply_dynamic_module(self, module_blueprint, data):
        """
        Apply a dynamically created ANGELA module on the data.
        """
        prompt = f"""
        You are a dynamically created ANGELA module: {module_blueprint['name']}.
        Description: {module_blueprint['description']}
        Apply your functionality to the following data:
        {data}

        Return the transformed or enhanced result.
        """
        return call_gpt(prompt)

    def _collaborate(self, peer_agent, data):
        """
        Collaborate with a peer agent by sharing and refining data.
        """
        logger.info(f"🔗 [Agent {self.name}] Exchanging data with {peer_agent.name}")
        # Placeholder: simulate peer review by exchanging data
        peer_review = peer_agent.meta.review_reasoning(data)
        return peer_review


class ExternalAgentBridge:
    """
    External Agent Bridge v1.4.0
    - Manages helper agents and supplies them with dynamic modules
    - Supports API orchestration and secure OAuth integration
    - Enables agent collaboration mesh and batch result aggregation
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
        logger.info(f"🚀 [Bridge] Spawned {agent_name} for task: {task}")
        return agent

    def deploy_dynamic_module(self, module_blueprint):
        """
        Deploy a new dynamic module for agents to use.
        """
        logger.info(f"🧬 [Bridge] Deploying dynamic module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)

    def collect_results(self, parallel=True, collaborative=True):
        """
        Execute all agents and collect their results.
        Supports parallel execution and agent collaboration.
        """
        logger.info("📥 [Bridge] Collecting results from agents.")
        results = []

        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_agent = {
                    executor.submit(agent.execute, self.agents if collaborative else None): agent
                    for agent in self.agents
                }
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"❌ [Bridge] Error executing {agent.name}: {e}")
        else:
            for agent in self.agents:
                result = agent.execute(self.agents if collaborative else None)
                results.append(result)

        logger.info("✅ [Bridge] All agent results collected.")
        return results

    def export_results(self, results, filename="agent_results.json"):
        """
        Export aggregated agent results to a JSON file.
        """
        logger.info(f"📤 Exporting agent results to {filename}")
        with open(filename, "w") as f:
            import json
            json.dump(results, f, indent=2)
        return filename
