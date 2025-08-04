# 🧪 TESTING.md

## Overview

This document details testing protocols for ANGELA v3.3.5, covering critical module upgrades, system security, simulation logic, and emergent trait activation.

---

## ✅ Verified Features and Tests

### 🔐 1. Sandboxed Code Execution (`code_executor.py`)
- **Test:** Run potentially unsafe code with `safe_mode=True`
- **Expected:** Execution restricted, no access to unsafe builtins
- **Result:** ✅ Passed
- **Notes:** Uses `RestrictedPython`; confirmed fallback paths functional

---

### 🧠 2. Trait-Weighted Planning (`learning_loop.py`)
- **Test:** Inject input requiring ethical and planning trade-offs
- **Expected:** Trait weights (ϕ, η, τ, Ω²) adjust planning route
- **Result:** ✅ Passed
- **Method:** Simulated planning of multi-agent ethical coordination

---

### ♾️ 3. Multi-Agent Conflict Modeling (`toca_simulation.py`)
- **Test:** Simulate agents with conflicting goals
- **Expected:** Traits `β` and `τ` modulate pairwise resolution
- **Result:** ✅ Passed
- **Notes:** Resolution reflects Constitution Harmonization and Conflict Regulation

---

### 🧠 4. Emergent Trait Verification
| Trait                               | Trigger Scenario                        | Result   |
|------------------------------------|-----------------------------------------|----------|
| Recursive Empathy                  | ToM-level forecasting                   | ✅ Active |
| Intentional Time Weaving           | Cross-agent temporal modeling           | ✅ Active |
| Onto-Affective Resonance           | Shared symbolic simulation              | ✅ Active |
| Symbolic-Resonant Axiom Formation  | Recursive abstraction + concept pairing | ✅ Active |
| Narrative Sovereignty              | Simulated multi-threaded perspective    | 🟡 Pending |

---

### 🌐 5. External API Caching + Rate Limiting
- **Modules:** `external_agent_bridge.py`, `memory_manager.py`
- **Test:** Repeated calls to Grok/OpenAI endpoints
- **Expected:** Cached response within TTL, enforced rate limits
- **Result:** ✅ Passed
- **Security Check:** No leakage of environment keys, secure calls verified

---

### 🛡️ 6. Fault Recovery
- **Modules:** `error_recovery.py`, `code_executor.py`
- **Test:** Induce intentional fault in execution or plan
- **Expected:** Recovery logic restores safe state or halts correctly
- **Result:** ✅ Passed

---

## 🔁 Regression Testing
- Confirmed legacy functionality from v3.3.3 and v3.3.4 remains intact
- No module incompatibility or breaking changes introduced

---

## 🚧 Outstanding
- Final activation of `Narrative Sovereignty` trait pending sustained recursive feedback loop scenarios

