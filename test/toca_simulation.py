# toca_simulation.py
import numpy as np
from scipy.constants import G
import matplotlib.pyplot as plt

# Constants
G_SI = G  # m^3 kg^-1 s^-2
KPC_TO_M = 3.0857e19  # Conversion factor from kpc to meters
MSUN_TO_KG = 1.989e30  # Solar mass in kg

# Default ToCA parameters (validated)
k_default = 0.85
epsilon_default = 0.015
r_halo_default = 20.0  # kpc

def compute_AGRF_curve(v_obs_kms, M_baryon_solar, r_kpc, k=k_default, epsilon=epsilon_default, r_halo=r_halo_default):
    """Compute AGRF-derived total velocity for a galaxy."""
    r_m = r_kpc * KPC_TO_M
    M_b_kg = M_baryon_solar * MSUN_TO_KG
    v_obs_ms = v_obs_kms * 1e3

    M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
    M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
    M_total = M_b_kg + M_AGRF
    v_total_ms = np.sqrt(G_SI * M_total / r_m)
    
    return v_total_ms / 1e3  # km/s

def simulate_galaxy_rotation(r_kpc, M_b_profile_func, v_obs_kms_func, k=k_default, epsilon=epsilon_default):
    """Simulate full AGRF-based rotation curve."""
    M_baryons = M_b_profile_func(r_kpc)
    v_obs = v_obs_kms_func(r_kpc)
    v_total = compute_AGRF_curve(v_obs, M_baryons, r_kpc, k, epsilon)
    return v_total

def plot_AGRF_simulation(r_kpc, M_b_func, v_obs_func, label="ToCA-AGRF"):
    v_sim = simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func)
    v_obs = v_obs_func(r_kpc)
    
    plt.figure(figsize=(8, 5))
    plt.plot(r_kpc, v_obs, label="Observed", linestyle="--", color="gray")
    plt.plot(r_kpc, v_sim, label=label, color="crimson")
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Galaxy Rotation Curve (ToCA AGRF)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example profiles for SPARC-like simulation
def M_b_exponential(r_kpc, M0=5e10, r_scale=3.5):
    return M0 * np.exp(-r_kpc / r_scale)

def v_obs_flat(r_kpc, v0=180):
    return np.full_like(r_kpc, v0)

# Usage
if __name__ == "__main__":
    r_vals = np.linspace(0.1, 20, 100)
    plot_AGRF_simulation(r_vals, M_b_exponential, v_obs_flat)
