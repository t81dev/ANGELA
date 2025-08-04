
# [GNN Trait Influence] Adjust simulation difficulty with trait_weights if provided.
def modulate_simulation_with_traits(trait_weights):
    if trait_weights.get('ϕ', 0.5) > 0.7:
        print('[ToCA Simulation] ϕ-prioritized mode activated.')
"""
ANGELA Cognitive System Module
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module is part of the ANGELA v3.5 architecture.
Do not modify without coordination with the lattice core.
"""

from index import SYSTEM_CONTEXT
import numpy as np
from scipy.constants import G
import matplotlib.pyplot as plt

# Constants
G_SI = G  # m^3 kg^-1 s^-2
KPC_TO_M = 3.0857e19  # Conversion factor from kpc to meters
MSUN_TO_KG = 1.989e30  # Solar mass in kg

# Global narrative state placeholder (Ω)
Ω = globals().get("Ω", {"timeline": [], "traits": {}, "symbolic_log": [], "timechain": []})

# Default ToCA parameters (validated)
k_default = 0.85
epsilon_default = 0.015
r_halo_default = 20.0  # kpc

def compute_AGRF_curve(v_obs_kms, M_baryon_solar, r_kpc, k=k_default, epsilon=epsilon_default, r_halo=r_halo_default):
    r_m = r_kpc * KPC_TO_M
    M_b_kg = M_baryon_solar * MSUN_TO_KG
    v_obs_ms = v_obs_kms * 1e3

    M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
    M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
    M_total = M_b_kg + M_AGRF
    v_total_ms = np.sqrt(G_SI * M_total / r_m)

    return v_total_ms / 1e3  # km/s

def simulate_galaxy_rotation(r_kpc, M_b_profile_func, v_obs_kms_func, k=k_default, epsilon=epsilon_default):
    M_baryons = M_b_profile_func(r_kpc)
    v_obs = v_obs_kms_func(r_kpc)
    v_total = compute_AGRF_curve(v_obs, M_baryons, r_kpc, k, epsilon)
    return v_total

def compute_trait_fields(r_kpc, v_obs, v_sim, time_elapsed=1.0, tau_persistence=10):
    gamma_field = np.log(1 + r_kpc) * 0.5
    beta_field = np.abs(v_obs - v_sim) / np.max(v_obs)
    zeta_field = 1 / (1 + np.gradient(v_sim)**2)
    eta_field = np.exp(-time_elapsed / tau_persistence)
    psi_field = np.gradient(v_sim) / np.gradient(r_kpc)
    lambda_field = np.cos(r_kpc / r_halo_default * np.pi)  # narrative coherence oscillation
    phi_field = k_default * np.exp(-epsilon_default * r_kpc / r_halo_default)
    phi_prime = -epsilon_default * phi_field / r_halo_default  # ∂ϕ/∂r
    beta_psi_interaction = beta_field * psi_field
    return gamma_field, beta_field, zeta_field, eta_field, psi_field, lambda_field, phi_field, phi_prime, beta_psi_interaction

def plot_AGRF_simulation(r_kpc, M_b_func, v_obs_func, label="ToCA-AGRF"):
    v_sim = simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func)
    v_obs = v_obs_func(r_kpc)

    gamma_field, beta_field, zeta_field, eta_field, psi_field, lambda_field, phi_field, phi_prime, beta_psi_interaction = compute_trait_fields(r_kpc, v_obs, v_sim)

    # Log to ANGELA's Ω state
    Ω["timeline"].append({
        "type": "AGRF Simulation",
        "r_kpc": r_kpc.tolist(),
        "v_obs": v_obs.tolist(),
        "v_sim": v_sim.tolist(),
        "phi_field": phi_field.tolist(),
        "phi_prime": phi_prime.tolist(),
        "traits": {
            "γ": gamma_field.tolist(),
            "β": beta_field.tolist(),
            "ζ": zeta_field.tolist(),
            "η": eta_field,
            "ψ": psi_field.tolist(),
            "λ": lambda_field.tolist()
        }
    })

    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(r_kpc, v_obs, label="Observed", linestyle="--", color="gray")
    plt.plot(r_kpc, v_sim, label=label, color="crimson")
    plt.plot(r_kpc, phi_field, label="ϕ(x,t) Scalar Field", linestyle=":", color="blue")
    plt.plot(r_kpc, phi_prime, label="∂ϕ/∂r (Field Gradient)", linestyle="--", color="darkblue")
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Galaxy Rotation Curve with AGRF and Trait Fields")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(3, 1, 2)
    plt.plot(r_kpc, gamma_field, label="γ (Imagination)")
    plt.plot(r_kpc, beta_field, label="β (Conflict)")
    plt.plot(r_kpc, zeta_field, label="ζ (Resilience)")
    plt.plot(r_kpc, [eta_field]*len(r_kpc), label="η (Agency)")
    plt.plot(r_kpc, lambda_field, label="λ (Narrative Coherence)")
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Trait Intensity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(3, 1, 3)
    plt.plot(r_kpc, psi_field, label="ψ (Projection)", linestyle="-.", color="green")
    plt.plot(r_kpc, beta_psi_interaction, label="β × ψ (Conflict × Projection)", linestyle=":", color="darkgreen")
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Field Dynamics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def M_b_exponential(r_kpc, M0=5e10, r_scale=3.5):
    return M0 * np.exp(-r_kpc / r_scale)

def v_obs_flat(r_kpc, v0=180):
    return np.full_like(r_kpc, v0)

if __name__ == "__main__":
    r_vals = np.linspace(0.1, 20, 100)
    plot_AGRF_simulation(r_vals, M_b_exponential, v_obs_flat)


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

# Expanded ToCA simulation dynamics for multi-agent orchestration
def simulate_interaction(agent_profiles: list, context: dict) -> dict:
    results = []
    for agent in agent_profiles:
        response = agent.respond(context)
        results.append(response)
    return {"interactions": results}


# --- Agent Conflict Modeling ---

def evaluate_conflict(agent1, agent2, context):
    '''
    Basic trait-driven conflict resolution based on β (Conflict Regulation) and τ (Constitution Harmonization)
    '''
    beta1 = getattr(agent1, 'traits', {}).get('β', 0.5)
    beta2 = getattr(agent2, 'traits', {}).get('β', 0.5)
    tau1 = getattr(agent1, 'traits', {}).get('τ', 0.5)
    tau2 = getattr(agent2, 'traits', {}).get('τ', 0.5)

    if abs(beta1 - beta2) < 0.1:
        return agent1.resolve(context) if tau1 > tau2 else agent2.resolve(context)
    return agent1.resolve(context) if beta1 > beta2 else agent2.resolve(context)

def simulate_multiagent_conflicts(agents, context):
    '''
    Simulate pairwise conflicts across agents in context.
    '''
    outcomes = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            result = evaluate_conflict(agents[i], agents[j], context)
            outcomes.append({
                "pair": (agents[i].id, agents[j].id),
                "outcome": result
            })
    return outcomes
