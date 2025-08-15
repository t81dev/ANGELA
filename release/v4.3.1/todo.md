# 📝 TODO.md

*Angela — Symbolic Meta-Synthesis Engine*

---

## **Post-4.3.1 Release Follow-Ups**

### 🔍 API Exposure Verification

* [x] Test `safe_execute` in **code\_executor.py** for sandbox integrity & exception safety
* [x] Validate `execute_code` handles both trusted/untrusted inputs without leak
* [x] Run `train_on_experience` from **learning\_loop.py** with diverse datasets
* [x] Confirm `retrieve_knowledge` returns deterministic, scoped results
* [x] Verify `fuse_modalities` integrates all supported sensory inputs without data loss
* [x] Stress-test `run_simulation` for multi-branch stability

---

### 🧙‍♂️ Trait-Role Mapping Enhancements

* [x] Confirm Σ integration with `user_profile` matches spec
* [x] Verify Υ (SharedGraph) renders correct multi-agent views under load
* [x] Check Φ⁰ visualizer link routing & accessibility

---

### 🛠️ Simulation Fix Validation

* [x] Run regression tests for `evaluateBranches` in **simulation\_core.py**
* [x] Create edge-case scenarios to confirm branch resolution logic

---

### 🔐 Ledger Enhancements

* [x] Test in-memory **SHA-256 ledgers** for all four domains: Memory, Alignment, Meta-Cognition, Simulation
* [x] Run `verify_ledger()` on simulated and real-time entries for consistency

---

### 🧠 Stage IV Hook Integrations

* [x] Validate **dream\_mode** symbolic recursion output matches expected synthesis depth
* [x] Test **axiom\_filter** with philosophical paradox inputs
* [x] Simulate SharedGraph diff/merge with multiple agents and overlapping state changes
* [x] Check ethical sandbox responses under extreme and borderline scenarios

---

### ⚙️ Config Updates

* [x] Confirm **Long-Horizon memory** persistence and retrieval for `24h` span
* [x] Ensure runtime trait modulators fire on correct triggers:

  * [x] `ψ` via dream\_sync
  * [x] `π` via axiom\_fusion
  * [x] `Ω` via recursive\_resonance
