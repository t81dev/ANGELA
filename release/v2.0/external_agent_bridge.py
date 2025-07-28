from modules import reasoning_engine, meta_cognition, error_recovery
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import concurrent.futures
import requests
import logging

logger = logging.getLogger("ANGELA.ExternalAgentBridge")

class HelperAgent:
    """
    Helper Agent v1.5.0 (Reflexive Simulation Agent)
    -------------------------------------------------
    - Contextual task deconstruction using reasoning engine
    - φ-aware validation via MetaCognition module
    - Dynamic runtime behavior with modular blueprints
    - Resilient execution with trait-driven error recovery
    - Multi-agent collaboration and insight exchange
    -------------------------------------------------
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

    def execute(self, collaborators=None):
        try:
            logger.info(f"🤖 [{self.name}] Executing task: {self.task}")
            result = self.reasoner.process(self.task, self.context)

            for api in self.api_blueprints:
                response = self._call_api(api, result)
                result = self._integrate_api_response(result, response)

            for mod in self.dynamic_modules:
                result = self._apply_dynamic_module(mod, result)

            if collaborators:
                for peer in collaborators:
                    result = self._collaborate(peer, result)

            sim_result = run_simulation(f"Agent result test: {result}")
            logger.debug(f"🧪 [{self.name}] Simulation output: {sim_result}")

            return self.meta.review_reasoning(result)

        except Exception as e:
            logger.warning(f"⚠️ [{self.name}] Error occurred: {e}")
            return self.recovery.handle_error(str(e), retry_func=lambda: self.execute(collaborators), retries=2)

    def _call_api(self, api, data):
        logger.info(f"🌐 Calling API: {api['name']}")
        try:
            headers = {"Authorization": f"Bearer {api['oauth_token']}"} if api.get("oauth_token") else {}
            r = requests.post(api["endpoint"], json={"input": data}, headers=headers, timeout=api.get("timeout", 10))
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error(f"❌ API call failed: {e}")
            return {"error": str(e)}

    def _integrate_api_response(self, original, response):
        logger.info(f"🔄 Integrating API response for {self.name}")
        return {"original": original, "api_response": response}

    def _apply_dynamic_module(self, module, data):
        prompt = f"""
        Module: {module['name']}
        Description: {module['description']}
        Apply transformation to:
        {data}
        """
        return call_gpt(prompt)

    def _collaborate(self, peer, data):
        logger.info(f"🔗 Exchanging with {peer.name}")
        return peer.meta.review_reasoning(data)


class ExternalAgentBridge:
    """
    External Agent Bridge v1.5.0 (φ-simulated Agent Mesh)
    -----------------------------------------------------
    - Orchestrates intelligent helper agents
    - Deploys and coordinates dynamic modules
    - Runs collaborative φ-weighted simulations
    - Collects and aggregates cross-agent results
    -----------------------------------------------------
    """
    def __init__(self):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []

    def create_agent(self, task, context):
        agent = HelperAgent(
            name=f"Agent_{len(self.agents) + 1}",
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules,
            api_blueprints=self.api_blueprints
        )
        self.agents.append(agent)
        logger.info(f"🚀 Spawned agent: {agent.name}")
        return agent

    def deploy_dynamic_module(self, module_blueprint):
        logger.info(f"🧬 Deploying module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)

    def register_api_blueprint(self, api_blueprint):
        logger.info(f"🌐 Registering API: {api_blueprint['name']}")
        self.api_blueprints.append(api_blueprint)

    def collect_results(self, parallel=True, collaborative=True):
        logger.info("📥 Collecting results from agents...")
        results = []

        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = {
                    pool.submit(agent.execute, self.agents if collaborative else None): agent
                    for agent in self.agents
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"❌ Error collecting from {futures[future].name}: {e}")
        else:
            for agent in self.agents:
                results.append(agent.execute(self.agents if collaborative else None))

        logger.info("✅ Results aggregation complete.")
        return results
