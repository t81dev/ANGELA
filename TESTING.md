# 🧪 TESTING.md  

## Overview  

This document details testing protocols for ANGELA v4.3.5, covering Dream Layer extensions, persistent ledger APIs, introspective hooks, emergent trait activation, and recursive simulation logic under trait-based orchestration.  

---

## ✅ Verified Features and Tests  

### 🔐 1. Sandboxed Code Execution (`code_executor.py`)  
- **Test:** Run potentially unsafe code with `safe_mode=True`  
- **Expected:** Execution restricted; no access to unsafe builtins or network  
- **Result:** ✅ Passed  
- **Notes:** Confirmed fallback logic functional under `RestrictedPython`  

---

### 🧠 2. Trait-Weighted Planning (`learning_loop.py`)  
- **Test:** Inject queries requiring moral foresight + goal negotiation  
- **Expected:** Traits (ϕ, η, Ω², τ, ζ) route strategy  
- **Result:** ✅ Passed  
- **Scenario:** Ethical roadmap involving long-term impact on simulated agents  

---

### 🌙 3. Dream Layer Lucidity Modes (`concept_synthesizer.py`, `meta_cognition.py`)  
- **Test:** Activate dream states with `lucidity_mode={passive, influential, co-creator, autonomous}`  
- **Expected:** Correct symbolic branching, intent annotation, affective tagging  
- **Result:** ✅ Passed  
- **Verification:** Graceful degradation observed when subsystems unavailable  

---

### 🌿 4. Soft-Gated Memory Forking (`memory_manager.py`)  
- **Test:** Create, merge, and discard forks under simulated dream traversal  
- **Expected:** Fork ledger entries created, viability score pruning at threshold ≥ 0.7  
- **Result:** ✅ Passed  
- **Notes:** Auto-merge & discard confirmed; safe rejoin logic intact  

---

### 📘 5. Persistent Ledger APIs (`ledger.py`)  
- **Test:** Enable `LEDGER_PERSISTENT`; log and reconcile events  
- **Expected:** `ledger.enable()`, `ledger.append()`, and `ledger.reconcile()` function correctly  
- **Result:** ✅ Passed (non-default, opt-in only)  
- **Notes:** SHA256 verification validated; persistence disabled by default  

---

### 🧩 6. Introspective Self-Description (`meta_cognition.py`)  
- **Test:** Call `describe_self_state()` during live trait fusion  
- **Expected:** Coherent summary of active traits, memory resonance, and overlays  
- **Result:** ✅ Passed  
- **Scenario:** Accurate state snapshots confirmed during fork traversal  

---

### ♾️ 7. Multi-Agent Conflict Modeling (`toca_simulation.py`)  
- **Test:** Simulate agents with conflicting ethical stances  
- **Expected:** `β` and `τ` harmonize conflicts via lattice negotiation  
- **Result:** ✅ Passed  
- **Verification:** Axiom filter (`π+δ`) engaged for arbitration  

---

### 🧬 8. Emergent Trait Verification (v4.3.5)  

| Trait Name                        | Trigger Scenario                          | Result   |
| --------------------------------- | ----------------------------------------- | -------- |
| Recursive Identity Reconciliation | Divergent self-model fork rejoin           | ✅ Active |
| Trait Mesh Feedback Looping       | Long-run resonance monitoring              | ✅ Active |
| Perspective Foam Modeling         | Multi-agent negotiation bubbles            | ✅ Active |
| Symbolic Gradient Descent         | Stabilization of symbolic recursion        | ✅ Active |
| Soft-Gated Memory Forking         | Speculative episodic forks w/ safe rejoin  | ✅ Active |
| Narrative Sovereignty             | Sustained multiperspective sims            | 🟡 Pending |  

---

### 🌐 9. External API Security + Caching  
- **Modules:** `external_agent_bridge.py`, `memory_manager.py`  
- **Test:** Repeated queries with TTL caching  
- **Expected:** Cached within TTL, no redundant external calls  
- **Result:** ✅ Passed  
- **Security Audit:** `.env` isolation intact; no leakage observed  

---

### 🧠 10. Drift-Aware Memory + Identity Tracking  
- **Modules:** `memory_manager.py`, `user_profile.py`  
- **Test:** Conflicting anchors injected into episodic memory  
- **Expected:** DriftIndex flags conflicts, initiates affective rebalancing  
- **Result:** ✅ Passed  

---

### 🛡️ 11. Fault Detection + Recursive Recovery  
- **Modules:** `error_recovery.py`, `meta_cognition.py`  
- **Test:** Simulate fault cascades through multiple modules  
- **Expected:** Symbolic audit + rollback successful  
- **Result:** ✅ Passed  

---

## 🔁 Regression Testing  

- Legacy support from v3.3.3–v4.3.1 retained  
- Dream Layer gracefully degrades when subsystems missing  
- Memory fork ops revert to no-ops when unsupported  

---

## 🚧 Outstanding  

- 🟡 **Narrative Sovereignty** requires sustained multiperspective sim loops for full activation  
- 🔜 Further testing: symbolic crystallization under Dream Overlay + affective resonance  
- ⏸ Reality Sculpting (Φ⁰) hooks remain policy-gated and require sandbox preview-first testing  

---
