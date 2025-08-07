# 🧪 TESTING.md

## Overview

This document details testing protocols for ANGELA v3.5.1, covering critical module upgrades, emergent trait activation, memory drift detection, and recursive simulation logic under trait-based orchestration.

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

### ♾️ 3. Multi-Agent Conflict Modeling (`toca_simulation.py`)
- **Test:** Simulate agents with value and action conflicts
- **Expected:** `β` and `τ` harmonize conflicts via lattice negotiation
- **Result:** ✅ Passed
- **Verification:** Resolution aligns with Constitution Harmonization principles

---

### 🧬 4. Emergent Trait Verification
| Trait                               | Trigger Scenario                        | Result   |
|------------------------------------|-----------------------------------------|----------|
| Recursive Empathy                  | ToM-level recursive forecasting         | ✅ Active |
| Intentional Time Weaving           | Temporal symbolic planning              | ✅ Active |
| Onto-Affective Resonance           | Cross-agent symbolic-affective threads  | ✅ Active |
| Symbolic-Resonant Axiom Formation  | Deep recursion + abstraction            | ✅ Active |
| Affective-Resonant Trait Weaving   | Emotion-symbol blend in planning        | ✅ Active |
| Symbolic Crystallization           | Frequent concept recursion              | ✅ Active |
| Modular Reflexivity                | Mid-process module rerouting            | ✅ Active |
| Task-Specific Ethical Alignment    | Alignment check via `task_type`         | ✅ Active |
| Narrative Sovereignty              | Recursive prompt threading              | 🟡 Pending |

---

### 🌐 5. External API Security + Caching
- **Modules:** `external_agent_bridge.py`, `memory_manager.py`
- **Test:** Repeated OpenAI and Grok queries with TTL window
- **Expected:** Correctly cached within TTL, no redundant calls
- **Result:** ✅ Passed
- **Security Audit:** `.env` isolation intact; no data leakage observed

---

### 🧠 6. Drift-Aware Memory + Identity Tracking
- **Modules:** `memory_manager.py`, `user_profile.py`
- **Test:** Introduce conflicting memory anchors
- **Expected:** DriftIndex flags identity conflicts
- **Result:** ✅ Passed
- **Notes:** Affective realignment engaged via traits λ, δ

---

### 🛡️ 7. Fault Detection + Recursive Recovery
- **Modules:** `error_recovery.py`, `meta_cognition.py`
- **Test:** Simulate fault cascade through multiple modules
- **Expected:** Recovery through symbolic audit + trait rollback
- **Result:** ✅ Passed

---

## 🔁 Regression Testing
- Legacy support from v3.3.3–v3.4.3 retained
- No broken functionality or trait misfires in upgraded modules

---

## 🚧 Outstanding

- 🟡 **Narrative Sovereignty** trait activation pending sustained multi-perspective simulation feedback loops
- 🔜 Further testing on Dream Layer symbolic crystallization under idle-symbolic resonance conditions

---
