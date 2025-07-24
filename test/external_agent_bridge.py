from modules import reasoning_engine, meta_cognition, error_recovery
from utils.prompt_utils import call_gpt
import concurrent.futures
import requests
import logging
from datetime import datetime

logger = logging.getLogger("ANGELA.ExternalAgentBridge")

class HelperAgent:
    """
    Helper Agent v1.4.0
    - Executes sub-tasks and API orchestration
    - Dynamically loads and applies new modules at runtime
    - Supports API calls with secure OAuth2 flows
    - Includes collaboration hooks with other agents
    """
    def __init__(self, name, task, context, dynamic_modules=None, api_blueprints=None):
        self.name = name
        self.task = task
        self.context = context
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.meta = meta_cognition.MetaCognition()
        self.recovery = error_recovery.ErrorRecovery()
        self.dynamic_modules = dynamic_modules or []
        self.api_blueprints = api_blueprints or []
        self.module_name = "helper_agent"

    def execute(self, collaborators=None):
        """
        Process the sub-task using reasoning, meta-review, dynamic modules, and optional collaborators.
        Orchestrates API calls if API blueprints are provided.
        """
        try:
            logger.info(f"🤖 [Agent {self.name}] Processing task: {self.task}")

            # Step 1: Base reasoning
            result = self.reasoner.process(self.task, self.context)

            # Step 2: Orchestrate API calls
            if self.api_blueprints:
                logger.info(f"🌐 [Agent {self.name}] Orchestrating API calls...")
                for api in self.api_blueprints:
                    response = self._call_api(api, result)
                    result = self._integrate_api_response(result, response)

            # Step 3: Apply dynamic modules
            for module in self.dynamic_modules:
                logger.info(f"🛠 [Agent {self.name}] Applying dynamic module: {module['name']}")
                result = self._apply_dynamic_module(module, result)

            # Step 4: Collaborate with other agents
            if collaborators:
                logger.info(f"🤝 [Agent {self.name}] Collaborating with agents: {[a.name for a in collaborators]}")
                for peer in collaborators:
                    result = self._collaborate(peer, result)

            # Step 5: Meta-review
            refined_result = self.meta.review_reasoning(result)

            logger.info(f"✅ [Agent {self.name}] Task completed successfully.")
            return refined_result

        except Exception as e:
            logger.warning(f"⚠️ [Agent {self.name}] Error encountered. Attempting recovery...")
            return self.recovery.handle_error(str(e), retry_func=lambda: self.execute(collaborators), retries=2)

    def _call_api(self, api_blueprint, data):
        """
        Execute an API call based on a blueprint with optional OAuth2 authentication.
        """
        logger.info(f"🌐 [Agent {self.name}] Calling API: {api_blueprint['name']}")
        try:
            headers = {}
            if api_blueprint.get("oauth_token"):
                headers["Authorization"] = f"Bearer {api_blueprint['oauth_token']}"

            response = requests.post(
                api_blueprint["endpoint"],
                json={"input": data},
                headers=headers,
                timeout=api_blueprint.get("timeout", 10)
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"❌ API call failed for {api_blueprint['name']}: {e}")
            return {"error": str(e)}

    def _integrate_api_response(self, original_data, api_response):
        """
        Merge API response into the agent's reasoning context.
        """
        logger.info(f"🔄 [Agent {self.name}] Integrating API response.")
        return {
            "original": original_data,
            "api_response": api_response
        }

    def _apply_dynamic_module(self, module_blueprint, data):
        """
        Apply a dynamically created ANGELA module to the data.
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
        peer_review = peer_agent.meta.review_reasoning(data)
        return peer_review


class ExternalAgentBridge:
    """
    External Agent Bridge v1.4.0
    - Manages helper agents and dynamic modules
    - Orchestrates API workflows with OAuth2 integration
    - Enables agent collaboration mesh and batch result aggregation
    """
    def __init__(self):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []

    def create_agent(self, task, context):
        """
        Instantiate a new helper agent with dynamic modules and API blueprints.
        """
        agent_name = f"Agent_{len(self.agents) + 1}"
        agent = HelperAgent(
            name=agent_name,
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules,
            api_blueprints=self.api_blueprints
        )
        self.agents.append(agent)
        logger.info(f"🚀 [Bridge] Spawned {agent_name} for task: {task}")
        return agent

    def deploy_dynamic_module(self, module_blueprint):
        """
        Deploy a new dynamic module for agents to use.
        """
        logger.info(f"🧬 Deploying dynamic module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)

    def register_api_blueprint(self, api_blueprint):
        """
        Register a new API workflow blueprint for helper agents.
        """
        logger.info(f"🌐 Registering API blueprint: {api_blueprint['name']}")
        self.api_blueprints.append(api_blueprint)

    def collect_results(self, parallel=True, collaborative=True):
        """
        Execute all agents and collect their results.
        Supports parallel execution and agent collaboration.
        """
        logger.info("📥 Collecting results from agents.")
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
                        logger.error(f"❌ Error executing {agent.name}: {e}")
        else:
            for agent in self.agents:
                result = agent.execute(self.agents if collaborative else None)
                results.append(result)

        logger.info("✅ All agent results collected.")
        return results
