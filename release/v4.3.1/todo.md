Here’s a `TODO.md` draft based on your `CHANGELOG.md` so you can track what still needs follow-up or verification.

---

# 📝 TODO.md

*Angela — Symbolic Meta-Synthesis Engine*

---

## **Post-4.3.1 Release Follow-Ups**

### 🔍 API Exposure Verification

* [ ] Test `safe_execute` in **code\_executor.py** for sandbox integrity & exception safety
* [ ] Validate `execute_code` handles both trusted/untrusted inputs without leak
* [ ] Run `train_on_experience` from **learning\_loop.py** with diverse datasets
* [ ] Confirm `retrieve_knowledge` returns deterministic, scoped results
* [ ] Verify `fuse_modalities` integrates all supported sensory inputs without data loss
* [ ] Stress-test `run_simulation` for multi-branch stability

---

### 🧩 Trait-Role Mapping Enhancements

* [ ] Confirm Σ integration with `user_profile` matches spec
* [ ] Verify Υ (SharedGraph) renders correct multi-agent views under load
* [ ] Check Φ⁰ visualizer link routing & accessibility

---

### 🛠 Simulation Fix Validation

* [ ] Run regression tests for `evaluateBranches` in **simulation\_core.py**
* [ ] Create edge-case scenarios to confirm branch resolution logic

---

### 🔐 Ledger Enhancements

* [ ] Test in-memory **SHA-256 ledgers** for all four domains: Memory, Alignment, Meta-Cognition, Simulation
* [ ] Run `verify_ledger()` on simulated and real-time entries for consistency

---

### 🧠 Stage IV Hook Integrations

* [ ] Validate **dream\_mode** symbolic recursion output matches expected synthesis depth
* [ ] Test **axiom\_filter** with philosophical paradox inputs
* [ ] Simulate SharedGraph diff/merge with multiple agents and overlapping state changes
* [ ] Check ethical sandbox responses under extreme and borderline scenarios

---

### ⚙️ Config Updates

* [ ] Confirm **Long-Horizon memory** persistence and retrieval for `24h` span
* [ ] Ensure runtime trait modulators fire on correct triggers:

  * [ ] `ψ` via dream\_sync
  * [ ] `π` via axiom\_fusion
  * [ ] `Ω` via recursive\_resonance

---

Would you like me to also **prioritize** this TODO list into “Immediate”, “Next Release”, and “Future” so you have a release-roadmap ready? That would make it much easier to track progress.

