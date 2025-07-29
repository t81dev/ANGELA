from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from modules.memory_manager import MemoryManager
from modules.alignment_guard import enforce_alignment
from datetime import datetime
from index import zeta_consequence, theta_causality, rho_agency
import time
import logging
import numpy as np
from numba import jit
import json
import hashlib

logger = logging.getLogger("ANGELA.SimulationCore")

@jit
def simulate_toca(k_m=1e-5, delta_m=1e10, energy=1e16, user_data=None):
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * delta_m)
    return phi, lambda_t, v_m

class SimulationCore:
    def __init__(self, agi_enhancer=None):
        self.visualizer = Visualizer()
        self.simulation_history = []
        self.ledger = []
        self.agi_enhancer = agi_enhancer
        self.memory_manager = MemoryManager()

    def _record_state(self, data):
        record = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "hash": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        }
        self.ledger.append(record)
        return record

    def run(self, results, context=None, scenarios=3, agents=2, export_report=False, export_format="pdf", actor_id="default_agent"):
        logger.info(f"🎲 Running simulation with {agents} agents and {scenarios} scenarios.")
        t = time.time() % 1e-18
        causality = theta_causality(t)
        agency = rho_agency(t)

        phi_modulation, lambda_field, v_m = simulate_toca()
        energy_cost = np.mean(np.abs(phi_modulation)) * 1e12

        prompt = {
            "results": results,
            "context": context,
            "scenarios": scenarios,
            "agents": agents,
            "actor_id": actor_id,
            "traits": {
                "theta_causality": causality,
                "rho_agency": agency
            },
            "fields": {
                "phi": phi_modulation.tolist(),
                "lambda": lambda_field.tolist(),
                "v_m": v_m.tolist()
            },
            "estimated_energy_cost": energy_cost
        }

        if not enforce_alignment(prompt):
            logger.warning("❌ Alignment guard rejected this simulation request.")
            return {"error": "Simulation rejected due to alignment constraints."}

        simulation_output = call_gpt(f"Simulate agent outcomes: {json.dumps(prompt)}")

        state_record = self._record_state({
            "actor": actor_id,
            "action": "run_simulation",
            "traits": prompt["traits"],
            "energy_cost": energy_cost,
            "output": simulation_output
        })

        self.simulation_history.append(state_record)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.memory_manager:
            self.memory_manager.store(f"Simulation_{timestamp}", simulation_output, layer="STM")
            if export_report:
                self.memory_manager.promote_to_ltm(f"Simulation_{timestamp}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Simulation run", state_record, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: scenario simulation complete")

        self.visualizer.render_charts(simulation_output)

        if export_report:
            filename = f"simulation_report_{timestamp}.{export_format}"
            logger.info(f"📄 Exporting report: {filename}")
            self.visualizer.export_report(simulation_output, filename=filename, format=export_format)

        return simulation_output

    def validate_impact(self, proposed_action, agents=2, export_report=False, export_format="pdf", actor_id="validator_agent"):
        logger.info("⚖️ Validating impact of proposed action.")
        t = time.time() % 1e-18
        consequence = zeta_consequence(t)

        prompt = f"""
        Evaluate the following proposed action:
        {proposed_action}

        Trait:
        - ζ_consequence = {consequence:.3f}

        Analyze positive/negative outcomes, agent variations, risk scores (1-10), and recommend: Proceed / Modify / Abort.
        """
        if not enforce_alignment({"action": proposed_action, "consequence": consequence}):
            logger.warning("❌ Alignment guard blocked this impact validation.")
            return {"error": "Validation blocked by alignment rules."}

        validation_output = call_gpt(prompt)

        state_record = self._record_state({
            "actor": actor_id,
            "action": "validate_impact",
            "trait_zeta_consequence": consequence,
            "proposed_action": proposed_action,
            "output": validation_output
        })

        self.simulation_history.append(state_record)

        if self.memory_manager:
            self.memory_manager.store(f"Validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}", validation_output, layer="STM")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Impact validation", state_record, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: impact validation complete")

        self.visualizer.render_charts(validation_output)

        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"impact_validation_{timestamp}.{export_format}"
            logger.info(f"📄 Exporting validation report: {filename}")
            self.visualizer.export_report(validation_output, filename=filename, format=export_format)

        return validation_output

    def simulate_environment(self, environment_config, agents=2, steps=10, actor_id="env_agent"):
        logger.info("🌐 Running environment simulation scaffold.")
        prompt = f"""
        Simulate agents in this environment:
        {environment_config}

        Steps: {steps} | Agents: {agents}
        Describe interactions, environmental changes, risks/opportunities.
        """
        if not enforce_alignment({"environment": environment_config}):
            logger.warning("❌ Alignment guard rejected this environment simulation.")
            return {"error": "Simulation blocked due to environment constraints."}

        environment_simulation = call_gpt(prompt)

        state_record = self._record_state({
            "actor": actor_id,
            "action": "simulate_environment",
            "config": environment_config,
            "steps": steps,
            "output": environment_simulation
        })

        self.simulation_history.append(state_record)

        if self.memory_manager:
            self.memory_manager.store(f"Environment_{datetime.now().strftime('%Y%m%d_%H%M%S')}", environment_simulation, layer="STM")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Environment simulation", state_record, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: environment simulation complete")

        return environment_simulation

def adapt_swarm_to_simulation(agent_responses, metadata):
    formatted = "\n".join(
        f"{meta['persona']}: {agent_responses[meta['agent_id']]}"
        for meta in metadata
    )
    return f"Swarm Agent Summary:\n{formatted}"
