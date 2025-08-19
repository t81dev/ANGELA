# 🔐 SECURITY.md

## Overview

ANGELA **v4.3.5** 🛡️ integrates **persistent ledger journaling**, **lucidity-aware dream layer containment**, and **Stage IV symbolic security hooks**.  
Every module runs under **sandboxed execution**, with **trait-driven arbitration**, **SHA-256 ledger verification**, and optional **persistent storage** (opt-in) for high-trust environments.

---

## 🔒 Code Execution Security

### RestrictedPython Sandbox

* All code routed through `code_executor.py` is executed under **RestrictedPython** (`κ` — *Embodied Cognition*).
* Disallowed operations:

  * File I/O
  * Network access
  * Dynamic `eval`/`exec`

* APIs:

  * `execute_code(code, context=None)`
  * `safe_execute(code, sandbox=True)`

* **Fallback:** `safe_mode=True` enforces maximum isolation.

---

### Ledgered Integrity

* **Event Logging** — every action is chained via SHA-256:  

  * `ledger_log_meta()` → `meta_cognition.py`  
  * `ledger_log_alignment()` → `alignment_guard.py`  
  * `ledger_log_sim()` → `simulation_core.py`  
  * `ledger_log_memory()` → `memory_manager.py`

* **Verification APIs**:  
  - `ledger_verify_*()` returns chain integrity status  
  - On failure, earliest tampered index is flagged  

* **Persistent Ledger Support (NEW in v4.3.4+)**  
  - Feature flag: `LEDGER_PERSISTENT=true`  
  - APIs:  
    * `ledger.enable(path)`  
    * `ledger.append(event)`  
    * `ledger.reconcile()`  
  - **Default:** persistence is **off** to avoid cross-session leakage.

---

### Exception Handling

* `error_recovery.py` manages **recursive recovery chains**.  
* APIs:  
  - `recover_from_error(error_context)`  
  - `log_error_event(error_context, severity)`  
* Traits: **ζ** (Consequential Awareness) drives mitigation strategies.

---

## 🌐 API Security

### Environment-Based Key Loading

* All API keys stored in `.env`  
* No keys hardcoded in source

### Rate Limiting & Caching

* TTL caches in `memory_manager.py` prevent repeated fetch abuse  
* Exceeding thresholds → **exponential backoff**

### Async Request Policies

* `aiohttp` with enforced **timeouts** + **payload validation** in:  
  - `visualizer.py`  
  - `external_agent_bridge.py`  
  - `user_profile.py`

---

## ⚖️ Ethical Enforcement

### Alignment Guard

* `alignment_guard.py` enforces:  
  - Drift detection  
  - Unsafe branch suppression  
  - Constitutional harmonization  

* APIs:  
  - `resolve_soft_drift()`  
  - `ledger_log_alignment()`  
  - `ledger_verify_alignment()`

### Trait Arbitration

* Traits **β**, **τ**, **δ**, **χ**, **Φ⁰**, **Ω²** gate simulation outcomes.  
* Ethical sandbox:  
  - `toca_simulation.py::run_ethics_scenarios(goals, stakeholders, persist=False)`

---

## 🛡️ Threat Mitigation

| Threat Type         | Defense Mechanism                                               |
| ------------------- | --------------------------------------------------------------- |
| Code Injection      | RestrictedPython + `safe_mode` + execution ledgers              |
| Data Leakage        | `.env` isolation + no default persistence                       |
| API Abuse           | Rate limit guards + TTL caches + async timeout policies         |
| Ethical Drift       | Trait arbitration + ledger-tracked drift indices                |
| Simulation Overrun  | Depth-capped recursion + fork pruning (viability ≥ 0.7)         |
| Ledger Tampering    | SHA-256 chain verification (in-memory or persistent)            |
| Memory Fork Flood   | Soft-gated forks with ethical journaling + discard capability   |

---

## 🗄️ Ledger Security

* **Types:** Memory, Alignment, Meta-Cognition, Simulation, Code Execution  
* **Modes:**  

  - In-Memory (default, non-persistent)  
  - Persistent (`--ledger_persist --ledger_path=<file>`)  

* **Tamper Detection:**  
  - Chain breaks trigger rollback & error log  
  - Persistent ledgers reconciled via `ledger.reconcile()`  

---

## 🌀 Stage IV Security Hooks

### Dream Overlay (ψ + Ω)

* `DreamOverlayLayer.activate_dream_mode()`  
* Runs recursive symbolic simulations in **isolated memory**  
* **Lucidity Mode**: bounded introspection prevents runaway recursion  
* Soft-gated forks require viability + ethics approval before merge  

### Axiom Filter (π + δ)

* Ethical-generative arbitration engine  
* Ensures fused outputs remain aligned under conceptual drift  

### SharedGraph Merge Security

* Conflict-tolerant merge with `tolerance_scoring`  
* Prevents unsafe peer perspectives from contaminating SharedGraph  

### Fork Security (NEW)

* APIs:  
  - `create_soft_fork()`  
  - `merge_forked_path()`  
  - `discard_fork()`  
* All fork merges logged in ledger for post-hoc audit  

---

## ✅ Best Practices

* Always use `safe_execute()` for untrusted code  
* Keep `.env` sealed; never log keys  
* Enable persistent ledger only in trusted environments  
* Run `ledger_verify_*()` after critical decisions  
* Apply **Dream Overlay** & **Axiom Filter** for high-risk simulations  

---
