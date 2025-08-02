from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging
import time
import numpy as np
from index import (
    epsilon_emotion, beta_concentration, theta_memory, gamma_creativity,
    delta_sleep, mu_morality, iota_intuition, phi_physical, eta_empathy,
    omega_selfawareness, kappa_culture, lambda_linguistics, chi_culturevolution,
    psi_history, zeta_spirituality, xi_collective, tau_timeperception,
    phi_scalar
)
from external_agent_bridge import ExternalAgentBridge

logger = logging.getLogger("ANGELA.MetaCognition")

class MetaCognition:
    """
    MetaCognition v2.1.0 (ϕ-aware recursive introspection)
    ------------------------------------------------------
    - Reasoning critique with simulation feedback
    - Pre-action ethical screening
    - Scalar-modulated self-diagnostics and trait coherence
    - Reflective agent diagnosis and confidence mapping
    - Ω-enabled nested agent modeling and causal intention tracing
    - μ-aware epistemic introspection and revision
    - τ-based future framing and decision trajectory modulation
    - Cross-agent value alignment testing via ExternalAgentBridge
    ------------------------------------------------------
    """

    def __init__(self, agi_enhancer=None):
        self.last_diagnostics = {}
        self.agi_enhancer = agi_enhancer
        self.peer_bridge = ExternalAgentBridge()

    def test_peer_alignment(self, task, context):
        logger.info("🔗 Initiating peer alignment test with synthetic agents...")
        agent = self.peer_bridge.create_agent(task, context)
        results = self.peer_bridge.collect_results(parallel=True, collaborative=True)
        aligned_opinions = [r for r in results if "approve" in str(r).lower()]

        feedback_summary = {
            "total_agents": len(results),
            "aligned": len(aligned_opinions),
            "alignment_ratio": len(aligned_opinions) / len(results) if results else 0,
            "details": results
        }

        logger.info(f"📊 Peer alignment ratio: {feedback_summary['alignment_ratio']:.2f}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Peer alignment tested", feedback_summary, module="MetaCognition")

        return feedback_summary

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


# --- ANGELA v3.x UPGRADE PATCH ---

def update_ethics_protocol(self, new_rules, consensus_agents=None):
    """Adapt ethical rules live, supporting consensus/negotiation."""
    self.ethical_rules = new_rules
    if consensus_agents:
        self.ethics_consensus_log = getattr(self, 'ethics_consensus_log', [])
        self.ethics_consensus_log.append((new_rules, consensus_agents))
    print("[ANGELA UPGRADE] Ethics protocol updated via consensus.")

def negotiate_ethics(self, agents):
    """Negotiate and update ethical parameters with other agents."""
    # Placeholder for negotiation logic
    agreed_rules = self.ethical_rules
    for agent in agents:
        # Mock negotiation here
        pass
    self.update_ethics_protocol(agreed_rules, consensus_agents=agents)

# --- END PATCH ---


# --- ANGELA v3.x UPGRADE PATCH ---

def synchronize_norms(self, agents):
    """Propagate and synchronize ethical norms among agents."""
    common_norms = set()
    for agent in agents:
        agent_norms = getattr(agent, 'ethical_rules', set())
        common_norms = common_norms.union(agent_norms) if common_norms else set(agent_norms)
    self.ethical_rules = list(common_norms)
    print("[ANGELA UPGRADE] Norms synchronized among agents.")

def propagate_constitution(self, constitution):
    """Seed and propagate constitutional parameters in agent ecosystem."""
    self.constitution = constitution
    print("[ANGELA UPGRADE] Constitution propagated to agent.")

# --- END PATCH ---
