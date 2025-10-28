# full_hybrid_quantum_integration.py
# Unified MKone-style hybrid oracle (cleaned and fixed)

from qiskit import QuantumCircuit, Aer, execute
from qutip import Qobj, ket2dm, tensor, destroy, qeye
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import csv
import re
from flask import Flask, jsonify, Response

app = Flask(__name__)

# --- Quantum Substrate: with amplitude damping ------------------------------
def quantum_evolve(phi0, phi1, phi2, gamma=0.1):
    """
    Build a small 6-qubit circuit, get the statevector, convert to density matrix,
    and apply amplitude-damping Kraus operators to qubit index 1 (second qubit).
    Returns (statevector (numpy), fidelity estimate float).
    """
    n_qubits = 6
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.h(3)
    qc.cx(0, 1)
    qc.rz(phi0, 0)
    qc.rx(phi1, 1)
    qc.cx(3, 4)
    qc.rz(phi2, 3)

    # simulate statevector
    backend = Aer.get_backend("statevector_simulator")
    result = execute(qc, backend).result()
    state = result.get_statevector()  # numpy array length 2**n_qubits

    # density matrix (qutip Qobj)
    psi = Qobj(state)  # ket
    rho = ket2dm(psi)

    # amplitude damping Kraus operators for single qubit
    E0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])

    # Build full-system operators that act on qubit index 1 (0-based)
    # For qutip.tensor we must list operators in order [q0, q1, q2, ...]
    def full_op(single_qubit_op, target_index):
        ops = []
        for i in range(n_qubits):
            if i == target_index:
                ops.append(single_qubit_op)
            else:
                ops.append(qeye(2))
        return tensor(*ops)

    E0_full = full_op(E0, target_index=1)
    E1_full = full_op(E1, target_index=1)

    # Apply Kraus map to the density matrix
    rho_after = E0_full * rho * E0_full.dag() + E1_full * rho * E1_full.dag()

    # For returning, convert rho_after back to statevector approximation:
    # (we will return original statevector for plotting, but fidelity estimated)
    # Compute a simple fidelity estimate against an ideal single-qubit rotation pattern for demo:
    fidelity = (np.cos(phi1 / 2) ** 2) / 4 * (1 - gamma / 2)
    fidelity = float(np.clip(fidelity, 0.0, 1.0))

    return np.array(state), fidelity


# --- TensorFlow: empirical EM dataset ---------------------------------------
def load_em_data(csv_file="electromagnetic_spectrum.csv"):
    """
    Attempts to load a CSV with a header column "Energy (eV)". If missing/fails,
    returns a small normalized synthetic array.
    """
    data = []
    try:
        with open(csv_file, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Energy (eV)" in row:
                    val = re.findall(r"[\d.]+", row["Energy (eV)"])
                    if val:
                        data.append(float(val[0]))
                else:
                    # fallback: try first numeric column
                    for v in row.values():
                        val = re.findall(r"[\d.]+", v)
                        if val:
                            data.append(float(val[0]))
                            break
    except FileNotFoundError:
        data = []

    if len(data) == 0:
        # fallback synthetic data
        data = np.random.rand(10)

    data = np.array(data, dtype=float)
    # normalize to [0,1]
    denom = (np.max(data) - np.min(data)) + 1e-9
    data = (data - np.min(data)) / denom
    return data


# --- Classical learner -------------------------------------------------------
def train_hybrid(samples, labels, epochs=20):
    """
    Simple dense regressor. samples: list/array shape (N, D).
    """
    X = np.array(samples)
    y = np.array(labels)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    input_dim = X.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model


# --- Physics equation mapping ------------------------------------------------
def map_equation(weight, ledger_file="physics_equations_table.csv"):
    """
    Read a CSV ledger where each row is [domain, equation_string].
    Use thresholds on weight to pick a keyword to search for in the equation text.
    """
    equations = []
    try:
        with open(ledger_file, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            # assume header present; if not, skip gracefully
            header = next(reader, None)
            for r in reader:
                if len(r) > 1:
                    equations.append(r)
    except FileNotFoundError:
        equations = []

    if abs(weight) > 0.6:
        key = "E ="
    elif abs(weight) > 0.3:
        key = "f ="
    else:
        key = "v ="

    for r in equations:
        # r[1] holds equation text in expected format
        if key in r[1]:
            try:
                return {"domain": r[0], "equation": r[1], "weight": float(weight)}
            except Exception:
                return {"domain": r[0], "equation": r[1], "weight": float(weight)}

    return {"domain": "N/A", "equation": "None", "weight": float(weight)}


# --- Visualization helper ----------------------------------------------------
def make_prob_plot(state):
    """
    Create a bar plot of probabilities |psi|^2. Returns PNG encoded as base64 string.
    """
    probs = np.abs(state) ** 2
    fig, ax = plt.subplots(figsize=(8, 3))
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


# --- API endpoints -----------------------------------------------------------
@app.route("/quantum/state")
def get_quantum_state():
    phi = np.random.rand(3)
    state, F = quantum_evolve(*phi)
    plot_png = make_prob_plot(state)
    return jsonify({
        "params": [float(x) for x in phi],
        "fidelity": F,
        "state_sample": np.real(state[:8]).tolist(),
        "prob_plot_base64": plot_png
    })


@app.route("/train/full")
def train_full():
    em_data = load_em_data()
    samples, labels = [], []
    # create training pairs: use em_data length as number of training examples
    for i in range(len(em_data)):
        phi = np.random.rand(3)
        state, F = quantum_evolve(*phi)
        # use first 64 amplitudes (should be 2^6 = 64 for 6 qubits)
        samples.append(np.real(state[:64]))
        labels.append(em_data[i])
    model = train_hybrid(samples, labels, epochs=20)
    # get a representative weight (first layer weights)
    w = model.layers[0].get_weights()[0]
    first_weight = float(w.flatten()[0]) if w.size > 0 else 0.0
    mapping = map_equation(first_weight)
    return jsonify({"trained_samples": len(samples),
                    "first_weight": first_weight,
                    "mapping": mapping})


@app.route("/equation/map")
def equation_map():
    return jsonify(map_equation(0.73))


@app.route("/visualize/state")
def visualize_state():
    phi = [0.1, 0.2, 0.3]
    state, _ = quantum_evolve(*phi)
    img = base64.b64decode(make_prob_plot(state))
    return Response(img, mimetype="image/png")


if __name__ == "__main__":
    print("Running full hybrid quantum integration on port 5000...")
    # bind to all addresses so containerized runs work; change as needed
    app.run(host="0.0.0.0", port=5000)
