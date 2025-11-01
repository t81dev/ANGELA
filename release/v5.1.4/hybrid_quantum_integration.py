# ANGELA HALO Kernel Module: hybrid_quantum_integration_v5_1_3.py
# Version: 5.1.3
# Layer: Φ⁰–Ω² Bridge
# Function: Quantum–Classical Hybrid Resonance Engine
# Ledger Integrity: SHA-1024 (composite: SHA-512 x 2)
# Status: Distributed Empathic Reflection Ready
#
# Notes:
# - Flask-compatible API (FastAPI optional).
# - Modern Qiskit AerSimulator API with graceful fallback if Qiskit/QuTiP are absent.
# - Parameterized circuits for hybrid training.
# - Coherence diagnostics (purity, entropy) and hybrid feedback loss.
# - HALO metadata & ledger hashing in all endpoints.
#
# SPDX-License-Identifier: MIT

import os
import io
import re
import csv
import json
import base64
import hashlib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify, Response

# ----------------------------- Optional Imports ------------------------------
# We import Qiskit/QuTiP lazily and degrade gracefully if unavailable.
_QISKIT_OK = True
_QUTIP_OK = True
try:
    from qiskit import QuantumCircuit
    try:
        # Prefer qiskit-aer >= 0.12
        from qiskit_aer import AerSimulator
    except Exception:
        # Legacy path
        from qiskit import Aer as _AerLegacy
        class AerSimulator:
            def __init__(self, method="statevector", device=None):
                self._backend = _AerLegacy.get_backend("statevector_simulator")
            def run(self, qc):
                from qiskit import execute
                return execute(qc, self._backend)
except Exception:
    _QISKIT_OK = False

try:
    from qutip import Qobj, ket2dm, tensor, qeye  # type: ignore
except Exception:
    _QUTIP_OK = False

# ------------------------------ HALO Metadata --------------------------------
HALO_META = {
    "kernel_version": "5.1.3",
    "module": "hybrid_quantum_integration",
    "layer": "Φ⁰–Ω² Bridge",
    "status": "active"
}

# ------------------------------ Hash Utilities --------------------------------
def sha1024(data: bytes) -> str:
    \"\"\"Composite 1024-bit digest using two SHA-512 passes (concatenated hex).
    This is a pragmatic stand-in where true SHA-1024 isn't available.
    \"\"\"
    h1 = hashlib.sha512(data).hexdigest()
    h2 = hashlib.sha512(h1.encode("utf-8")).hexdigest()
    return h1 + h2  # 1024 bits in hex (128+128 hex chars = 256 hex chars)

# ------------------------------ Coherence Metrics -----------------------------
def coherence_metrics(state: np.ndarray) -> Dict[str, float]:
    \"\"\"Compute purity and Shannon entropy on the state amplitudes.\"\"\"
    # Ensure 1D complex state vector
    psi = state.astype(np.complex128).flatten()
    probs = np.abs(psi) ** 2
    # Purity for pure state |ψ⟩⟨ψ|: Tr(ρ²) = 1, but noise can reduce it after maps
    # For vector-only estimate, approximate purity as sum(p_i^2)
    purity = float(np.sum(probs ** 2))
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return {"purity": purity, "entropy": entropy}

def hybrid_feedback_loss(fidelity: float, classical_loss: float) -> float:
    \"\"\"Harmonic-like feedback: encourages high quantum fidelity and low classical loss.\"\"\"
    num = 2.0 * (fidelity * (1.0 - classical_loss))
    den = fidelity + (1.0 - classical_loss) + 1e-9
    return float(max(0.0, num / den))

# ------------------------------ Visualization ---------------------------------
def make_prob_plot(state: np.ndarray) -> str:
    probs = np.abs(state) ** 2
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(range(len(probs)), probs, width=0.6)
    ax.set_xlabel("Basis index |i⟩")
    ax.set_ylabel("Probability")
    ax.set_title("Quantum State Probabilities (|ψ|²)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ------------------------------ Quantum Engine --------------------------------
@dataclass
class QuantumConfig:
    n_qubits: int = 6
    damp_qubit: int = 1          # amplitude damping target
    gamma: float = 0.1           # damping parameter
    use_gpu: bool = bool(os.getenv("USE_GPU", ""))

def _simulate_statevector(qc) -> np.ndarray:
    if not _QISKIT_OK:
        # Fallback: synthetic normalized complex vector
        dim = 2 ** 6
        z = (np.random.randn(dim) + 1j * np.random.randn(dim)).astype(np.complex128)
        z /= np.linalg.norm(z) + 1e-12
        return z

    try:
        method = "statevector"
        device = "GPU" if bool(os.getenv("USE_GPU", "")) else "CPU"
        sim = AerSimulator(method=method)  # device hint not always supported
        result = sim.run(qc).result()
        # Some backends require passing qc to get_statevector
        try:
            state = result.get_statevector(qc)
        except Exception:
            state = result.get_statevector()
        return np.array(state, dtype=np.complex128)
    except Exception as e:
        warnings.warn(f\"Qiskit simulation failed: {e}. Using synthetic state.\")
        dim = 2 ** 6
        z = (np.random.randn(dim) + 1j * np.random.randn(dim)).astype(np.complex128)
        z /= np.linalg.norm(z) + 1e-12
        return z

def _apply_amplitude_damping(rho: Any, cfg: QuantumConfig) -> Any:
    if not _QUTIP_OK:
        return rho  # no-op if qutip missing
    n_qubits = cfg.n_qubits
    gamma = cfg.gamma
    E0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])

    def full_op(single_qubit_op, target_index):
        ops = [qeye(2)] * n_qubits
        ops[target_index] = single_qubit_op
        return tensor(*ops)

    E0_full = full_op(E0, cfg.damp_qubit)
    E1_full = full_op(E1, cfg.damp_qubit)
    return E0_full * rho * E0_full.dag() + E1_full * rho * E1_full.dag()

def quantum_evolve(phi0: float, phi1: float, phi2: float, cfg: QuantumConfig = QuantumConfig()) -> Tuple[np.ndarray, float]:
    \"\"\"Build a 6-qubit circuit, simulate statevector, apply amplitude damping on 1 qubit.
    Return (statevector, fidelity_estimate).\"\"\"
    n = cfg.n_qubits
    if _QISKIT_OK:
        # Parameter-free circuit (simple example). Parameterized version below.
        qc = QuantumCircuit(n)
        qc.h(0); qc.h(3)
        qc.cx(0, 1)
        qc.rz(phi0, 0)
        qc.rx(phi1, 1)
        qc.cx(3, 4)
        qc.rz(phi2, 3)
    else:
        qc = None  # synthetic path

    state = _simulate_statevector(qc)

    # Convert to density matrix if QuTiP available
    rho_after = None
    if _QUTIP_OK:
        psi = Qobj(state)  # |ψ⟩
        rho = ket2dm(psi)  # |ψ⟩⟨ψ|
        rho_after = _apply_amplitude_damping(rho, cfg)
        # We do not re-sample state from rho_after; keep original statevector for plotting.

    # Simple analytic fidelity proxy (demo)
    fidelity = (np.cos(phi1 / 2.0) ** 2) / 4.0 * (1.0 - cfg.gamma / 2.0)
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    return state, fidelity

# Parameterized circuit variant (for hybrid training backends)
def build_param_circuit(n_qubits: int = 6):
    if not _QISKIT_OK:
        return None, None
    from qiskit.circuit import Parameter
    phi0 = Parameter('phi0'); phi1 = Parameter('phi1'); phi2 = Parameter('phi2')
    qc = QuantumCircuit(n_qubits)
    qc.h(0); qc.h(3)
    qc.cx(0, 1)
    qc.rz(phi0, 0)
    qc.rx(phi1, 1)
    qc.cx(3, 4)
    qc.rz(phi2, 3)
    return qc, (phi0, phi1, phi2)

# ------------------------------ Data Loading ----------------------------------
def load_em_data(csv_file: str = "electromagnetic_spectrum.csv") -> np.ndarray:
    data: List[float] = []
    try:
        with open(csv_file, newline='', encoding=\"utf-8\") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # prefer a column that contains numbers
                picked = None
                for key, val in row.items():
                    m = re.findall(r\"[-+]?\\d*\\.?\\d+\", str(val))
                    if m:
                        picked = float(m[0]); break
                if picked is not None:
                    data.append(picked)
    except FileNotFoundError:
        pass

    if len(data) == 0:
        rng = np.random.default_rng(42)
        data = rng.random(16).tolist()

    data = np.array(data, dtype=float)
    denom = (np.max(data) - np.min(data)) + 1e-9
    data = (data - np.min(data)) / denom
    return data

# ------------------------------ Classical Learner -----------------------------
def train_hybrid(samples: np.ndarray, labels: np.ndarray, epochs: int = 20):
    # Lazy import TensorFlow to reduce cold-start
    import tensorflow as tf

    X = np.array(samples, dtype=float)
    y = np.array(labels, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    input_dim = X.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model

# ------------------------------ Flask App -------------------------------------
app = Flask(__name__)

def _meta() -> Dict[str, Any]:
    return dict(HALO_META)

@app.route(\"/quantum/state\")
def route_quantum_state():
    phi = np.random.rand(3)
    state, F = quantum_evolve(*phi)
    coh = coherence_metrics(state)
    plot_png = make_prob_plot(state)

    payload = {
        \"meta\": _meta(),
        \"params\": [float(x) for x in phi],
        \"fidelity\": F,
        \"coherence\": coh,
        \"state_sample\": np.real(state[:8]).tolist(),
        \"prob_plot_base64\": plot_png,
    }
    payload[\"ledger_hash\"] = sha1024(json.dumps(payload, sort_keys=True).encode(\"utf-8\"))
    return jsonify(payload)

@app.route(\"/train/full\")
def route_train_full():
    em_data = load_em_data()
    samples, labels = [], []

    # Generate training pairs
    for i in range(len(em_data)):
        phi = np.random.rand(3)
        state, F = quantum_evolve(*phi)
        samples.append(np.real(state[:64]))
        labels.append(em_data[i])

    X = np.array(samples, dtype=float)
    y = np.array(labels, dtype=float)
    model = train_hybrid(X, y, epochs=20)

    # Evaluate
    classical_loss = float(model.evaluate(X, y, verbose=0))
    # Use the last quantum fidelity as a proxy; could average across all
    _, F_last = quantum_evolve(*np.random.rand(3))
    resonance_alignment = hybrid_feedback_loss(F_last, classical_loss)

    # representative weight
    w = model.layers[0].get_weights()[0]
    first_weight = float(w.flatten()[0]) if w.size > 0 else 0.0
    mapping = map_equation(first_weight)

    payload = {
        \"meta\": _meta(),
        \"trained_samples\": int(len(samples)),
        \"first_weight\": first_weight,
        \"mapping\": mapping,
        \"classical_loss\": classical_loss,
        \"quantum_fidelity\": F_last,
        \"resonance_alignment\": resonance_alignment
    }
    payload[\"ledger_hash\"] = sha1024(json.dumps(payload, sort_keys=True).encode(\"utf-8\"))
    return jsonify(payload)

@app.route(\"/equation/map\")
def route_equation_map():
    payload = {\"meta\": _meta(), **map_equation(0.73)}
    payload[\"ledger_hash\"] = sha1024(json.dumps(payload, sort_keys=True).encode(\"utf-8\"))
    return jsonify(payload)

@app.route(\"/visualize/state\")
def route_visualize_state():
    phi = [0.1, 0.2, 0.3]
    state, _ = quantum_evolve(*phi)
    img_b64 = make_prob_plot(state)
    img = base64.b64decode(img_b64)
    return Response(img, mimetype=\"image/png\")


# ------------------------------ Physics Mapping -------------------------------
def map_equation(weight: float, ledger_file: str = \"physics_equations_table.csv\") -> Dict[str, Any]:
    equations: List[List[str]] = []
    try:
        with open(ledger_file, newline='', encoding=\"utf-8\") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if len(r) > 1:
                    equations.append(r)
    except FileNotFoundError:
        pass

    if abs(weight) > 0.6:
        key = \"E =\"
    elif abs(weight) > 0.3:
        key = \"f =\"
    else:
        key = \"v =\"

    for r in equations:
        if len(r) > 1 and key in r[1]:
            return {\"domain\": r[0], \"equation\": r[1], \"weight\": float(weight)}
    return {\"domain\": \"N/A\", \"equation\": \"None\", \"weight\": float(weight)}

# ------------------------------ Resonance Analysis ----------------------------
@app.route(\"/resonance/analyze\")
def route_resonance_analyze():
    \"\"\"Run several random circuits and summarize coherence trends.\"\"\"
    trials = int(os.getenv(\"RESONANCE_TRIALS\", \"8\"))
    purities, entropies, fidelities = [], [], []
    for _ in range(trials):
        phi = np.random.rand(3)
        state, F = quantum_evolve(*phi)
        coh = coherence_metrics(state)
        purities.append(coh[\"purity\"]); entropies.append(coh[\"entropy\"]); fidelities.append(F)

    summary = {
        \"meta\": _meta(),
        \"trials\": trials,
        \"purity_avg\": float(np.mean(purities)),
        \"purity_std\": float(np.std(purities)),
        \"entropy_avg\": float(np.mean(entropies)),
        \"entropy_std\": float(np.std(entropies)),
        \"fidelity_avg\": float(np.mean(fidelities)),
        \"fidelity_std\": float(np.std(fidelities)),
    }
    summary[\"ledger_hash\"] = sha1024(json.dumps(summary, sort_keys=True).encode(\"utf-8\"))
    return jsonify(summary)

# ------------------------------ Main ------------------------------------------
if __name__ == \"__main__\":
    print(\"[HALO] Hybrid Quantum Integration v5.1.3 — Φ⁰–Ω² Bridge starting on :5000 ...\")
    app.run(host=\"0.0.0.0\", port=5000)
