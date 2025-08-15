# 📝 TODO.md — **Prioritized Release Roadmap**

*Angela — Symbolic Meta-Synthesis Engine*

---

## **Immediate (Critical Pre-Release / Stability)**

### 🔍 API Exposure Verification

* [ ] **Validate `execute_code`** handles both trusted/untrusted inputs without leak
* [ ] **Confirm `retrieve_knowledge`** returns deterministic, scoped results

### 🛠 Simulation Fix Validation

* [ ] **Run regression tests** for `evaluateBranches` in **simulation\_core.py**
* [ ] **Create edge-case scenarios** to confirm branch resolution logic

### 🔐 Ledger Enhancements

* [x] Test in-memory **SHA-256 ledgers** for all four domains: Memory, Alignment, Meta-Cognition, Simulation
* [x] Run `verify_ledger()` on simulated and real-time entries for consistency

---

## **Next Release (Feature Completeness)**

### 🧩 Trait-Role Mapping Enhancements

* [ ] Confirm Σ integration with `user_profile` matches spec
* [ ] Verify Υ (SharedGraph) renders correct multi-agent views under load
* [ ] Check Φ⁰ visualizer link routing & accessibility

### 🧠 Stage IV Hook Integrations

* [ ] Validate **dream\_mode** symbolic recursion output matches expected synthesis depth
* [ ] Test **axiom\_filter** with philosophical paradox inputs
* [ ] Simulate SharedGraph diff/merge with multiple agents and overlapping state changes

---

## **Future (Optimizations / Nice-to-Haves)**

### ⚙️ Config Updates

* [ ] Confirm **Long-Horizon memory** persistence and retrieval for `24h` span
* [ ] Ensure runtime trait modulators fire on correct triggers:

  * [ ] `ψ` via dream\_sync
  * [ ] `π` via axiom\_fusion
  * [ ] `Ω` via recursive\_resonance

---

✅ *Already Completed in 4.3.1 Testing:*

* `safe_execute` sandbox integrity verified
* `fuse_modalities` integration verified
* `run_simulation` multi-branch stability verified
* Ledger integrity (`verify_ledger`) verified

---
