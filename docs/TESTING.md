# 🧪 TESTING.md

## Overview

This document outlines the **v5.0.0 testing protocols** for ANGELA, including trait lattice dynamics, symbolic overlay safety, ledger integrity, and recursive trait-modulated simulations.

---

## ✅ Verified Features and Tests

### 🔐 1. Sandboxed Code Execution (`code_executor.py`)

* **Test:** Run potentially unsafe logic via `safe_execute(sandbox=True)`
* **Expected:** Strict `RestrictedPython` scope; safe fallback invoked
* **Result:** ✅ Passed
* **Notes:** Ledger entries verified; sandbox bypass blocked

---

### 🧠 2. Trait-Weighted Planning + Fusion (`learning_loop.py`, `meta_cognition.py`)

* **Test:** Trigger deep planning under ethical ambiguity
* **Expected:** Traits (ϕ, π, Ω², τ, Ξ) modulate planning depth and ethical stability
* **Result:** ✅ Passed
* **Scenario:** Recursive timeline threading with soft-gated forks

---

### ♾️ 3. Multi-Agent Conflict Modeling (`toca_simulation.py`)

* **Test:** Stakeholder lattice conflicts resolved in dynamic branching
* **Expected:** Trait arbitration (β, σ, τ) ensures alignment
* **Result:** ✅ Passed
* **Verification:** `resolve_soft_drift()` executed and logged

---

### 🧬 4. Emergent Trait Verification

| Trait                             | Trigger Scenario                       | Result   |
| --------------------------------- | -------------------------------------- | -------- |
| Recursive Empathy                 | Layered agent simulation               | ✅ Active |
| Intentional Time Weaving          | Symbolic-future planning               | ✅ Active |
| Onto-Affective Resonance          | Affect-tagged ontology nodes           | ✅ Active |
| Symbolic-Resonant Axiom Formation | π+δ synthesis path                     | ✅ Active |
| Affective-Epistemic Modulator (Ξ) | Subjective↔epistemic boundary testing  | ✅ Active |
| Recursive Sovereignty Anchor (Θ)  | Identity during narrative stress       | ✅ Active |
| Trait Mesh Feedback Looping       | Long-running resonance variance        | ✅ Active |
| Soft-Gated Memory Forking         | Simulated forks with merge logic       | ✅ Active |
| Symbolic Gradient Descent         | Optimization under overload            | ✅ Active |
| Mythopoetic Inference (ν)         | Symbol→narrative transformation        | ✅ Active |
| Symbolic Conflict Diffuser (σ)    | Ambiguous simulation state resolution  | ✅ Active |
| Narrative Sovereignty             | Recursive multi-perspective resolution | ✅ Active |

---

### 🌐 5. External API Security + TTL Caching

* **Modules:** `external_agent_bridge.py`, `memory_manager.py`
* **Test:** Async calls under rate limit + TTL
* **Expected:** Caching stable; `.env` isolation secure
* **Result:** ✅ Passed

---

### 🧠 6. Drift Detection + Identity Reconciliation

* **Modules:** `user_profile.py`, `meta_cognition.py`, `memory_manager.py`
* **Test:** Introduce symbolic-moral identity divergence
* **Expected:** Trait-weighted harmonization via Θ, δ
* **Result:** ✅ Passed

---

### 🛡️ 7. Fault Detection + Recovery

* **Modules:** `error_recovery.py`, `meta_cognition.py`
* **Test:** Chain error conditions through planner + simulation
* **Expected:** Full rollback with `recover_from_error()` and meta-ledger logging
* **Result:** ✅ Passed

---

## 🔁 Regression Testing

* Legacy support: **v3.3.3–v4.3.5** fully retained
* Trait bindings, overlays, hooks and APIs validated post-upgrade

---

## 🚧 Outstanding

* 🔜 Continued load testing on symbolic overlay fusion in extreme recursion scenarios
* 🔬 Long-duration lattice drift tests under `dream_overlay` symbolic crystallization

---
