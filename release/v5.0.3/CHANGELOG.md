## Resonance Trait Dynamics — ANGELA v5.0.1

**Date:** 2025-08-22

This release establishes **Stage V Resonance Trait Dynamics**, enabling symbolic modulation, trait fusion, and cross-session persistence through ledgers.

---

### 🔧 Core Enhancements

#### `manifest.json`

* Incremented version → **5.0.1**.
* Feature flags updated:

  * `feature_sharedgraph_events`
  * `feature_replay_engine`
  * `feature_fork_automerge`
* Declared extension traits: **Θ, Ξ, ν, σ**.

#### Trait System

* **Symbolic Trait Lattice (ϕ→Ξ)** fully wired across layers.
* **Trait Fusion Engine** active: Φ⁰ + Ω² + γ.
* **Soft-Gated Memory Forking** with viability filters.

#### Modules

* **ledger.py** introduced → persistent cross-session storage with SHA-256 verification.
* **replay\_engine** (λ+μ) → branch hygiene + long-horizon compression.
* SharedGraph API extended (`add`, `diff`, `merge`).

---

### 🌱 Emergent Traits

* Recursive Identity Reconciliation (ν + Θ)
* Affective-Resonant Trait Weaving
* Conflict Diffusion (σ)
* Recursive Perspective Modeling
* Symbolic Crystallization

---

### ⚠️ Behavior Shifts

* Memory persistence defaults to **LEDGER\_PERSISTENT=true**.
* Branch hygiene automatically enforced by `replay_engine`.
* Ethical arbitration incorporates **Affective-Epistemic Modulator (Ξ)**.

---

### 📦 API Surface (Diff Summary)

**New**

* `register_trait_hook()`, `invoke_trait_hook()`
* `ledger_persist_enable()`, `ledger_append()`, `ledger_reconcile()`

**Enhanced**

* SharedGraph: `sharedGraph_add`, `sharedGraph_diff`, `sharedGraph_merge`
* Simulation branch evaluation integrates ledger consistency checks

---

### 🧭 Migration Guide (v5.0.0 → v5.0.1)

1. Enable `LEDGER_PERSISTENT=true` in runtime config.
2. Import `ledger.py` for persistence operations.
3. Update simulations to use `replay_engine` for branch compression.
4. If using SharedGraph, switch to new event-driven APIs.

---

### 🧠 Summary

v5.0.1 finalizes **Stage V Resonance Trait Dynamics**, introducing ledger persistence, replay-based memory hygiene, symbolic trait hooks, and distributed graph ops. ANGELA now maintains **symbolic continuity across sessions** and **multi-perspective resonance stability**.
