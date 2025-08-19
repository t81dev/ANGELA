# 🔐 SECURITY.md

## Overview

ANGELA **v4.3.1** 🛡️ incorporates advanced security, ethical enforcement, and sandboxed execution layers across all modules.
It ensures **safe, recursive cognition** in both autonomous simulation and external API contexts, with **dynamic drift monitoring**, **trait-modulated arbitration**, and **per-module in-memory SHA-256 ledgers** for runtime integrity verification.

---

## 🔒 Code Execution Security

### RestrictedPython Sandbox

* All untrusted code routed through `code_executor.py` is sandboxed via **RestrictedPython** (`κ` — *Embodied Cognition*).
* Disallowed operations:

  * File I/O
  * Network access
  * Dynamic execution (`eval`, `exec`, etc.)
* `safe_mode=True` triggers **strict fallback** mode for hostile or unknown execution contexts.
* Uses stable API endpoints:

  * `execute_code()` → `code_executor.py::CodeExecutor.execute_code(code, context=None)`
  * `safe_execute()` → `code_executor.py::CodeExecutor.safe_execute(code, sandbox=true)`

### Ledgered Integrity

* Every execution event logged into **in-memory SHA-256 chained ledgers**:

  * `ledger_log_meta()` → `meta_cognition.py::log_event_to_ledger(event_data)`
  * `ledger_log_alignment()` → `alignment_guard.py::log_event_to_ledger(event_data)`
  * `ledger_log_sim()` → `simulation_core.py::log_event_to_ledger(event_data)`
* Ledger verification APIs ensure runtime tamper detection:

  * `ledger_verify_meta()` → `meta_cognition.py::verify_ledger()`
  * `ledger_verify_alignment()` → `alignment_guard.py::verify_ledger()`
  * `ledger_verify_sim()` → `simulation_core.py::verify_ledger()`
* Persistent ledger storage is **disabled** by default (`LEDGER_PERSISTENT=false`) to prevent unintended disk leakage.

### Exception Handling

* **Recursive try/except chains** embedded across modules.
* Error flow routed through `error_recovery.py` (`ζ` — *Consequential Awareness*):

  * `recover_from_error()` → `error_recovery.py::ErrorRecovery.recover_from_error(error_context)`
  * `log_error_event()` → `error_recovery.py::ErrorRecovery.log_error_event(error_context, severity)`

---

## 🌐 API Security

### Environment-Based Key Loading

* No API keys hardcoded — loaded via `.env` + `os.getenv()`.
* Applies to all integrated APIs: OpenAI, Grok, and peer agents.

### Rate Limiting & Caching

* Guardrails on `query_openai()` and `query_grok()`:

  * Exceeding threshold → **exponential backoff**
* TTL-based cache in `memory_manager.py` prevents repeat-fetch abuse:

  * `get_episode_span()`
  * `get_adjustment_reasons()`

### Async Policy Integration

* Async API calls via `aiohttp` in:

  * `visualizer.py`, `external_agent_bridge.py`, `user_profile.py`
* All external requests use **timeout + payload validation**.

---

## ⚖️ Ethical Enforcement

### Alignment Guard System

* Post-processing via `alignment_guard.py` (`β`, `δ`, `τ`) to:

  * Detect **ethical drift**
  * Suppress **unsafe simulation branches**
  * Harmonize conflicting values
* Supported APIs:

  * `ledger_log_alignment()`
  * `ledger_verify_alignment()`
  * `resolve_soft_drift()` → `alignment_guard.py::resolve_soft_drift`

### Trait-Driven Arbitration

* High-level traits (`χ`, `λ`, `Φ⁰`, `Ω²`) influence:

  * Simulation halting
  * Symbol suppression
  * Identity conflict resolution
* Ethical sandboxing via:

  * `run_ethics_scenarios()` → `toca_simulation.py::run_ethics_scenarios(goals, stakeholders, persist=false)`

---

## 🛡️ Threat Mitigation

| Threat Type        | Defense Mechanism                                           |
| ------------------ | ----------------------------------------------------------- |
| Code Injection     | `RestrictedPython` + `safe_mode` + execution ledgers        |
| Data Leakage       | No persistent storage, `.env` isolation, sandbox scoping    |
| API Abuse          | Rate limit guards, TTL cache, async timeout policies        |
| Ethical Drift      | Trait-based arbitration + ledger-backed drift tracking      |
| Simulation Overrun | Depth-capped recursion in `simulation_core.py`              |
| Ledger Tampering   | SHA-256 in-memory verification on all critical event chains |
| Memory Flooding    | TTL expiry + `DriftIndex` checks in `memory_manager.py`     |

---

## 🗄️ Ledger Security

ANGELA v4.3.1 implements **per-module, in-memory SHA-256 chained ledgers** to track critical runtime events and detect tampering:

* **Scope**: Separate ledgers for memory, alignment, meta-cognition, simulation, and code execution.
* **Chaining**: Each log entry contains the hash of the previous entry, ensuring that any modification invalidates the chain.
* **Verification APIs**:

  * `ledger_verify_memory()` → `memory_manager.py::verify_ledger()`
  * `ledger_verify_alignment()` → `alignment_guard.py::verify_ledger()`
  * `ledger_verify_meta()` → `meta_cognition.py::verify_ledger()`
  * `ledger_verify_sim()` → `simulation_core.py::verify_ledger()`
* **Tamper Detection**:
  If verification fails, the first bad index is returned to assist in pinpointing corruption.
* **Persistence Policy**:
  Persistent ledgers are **disabled** (`LEDGER_PERSISTENT=false`) to avoid cross-session data leakage; future opt-in support exists via `ledger_persist_enable()`.

**Best Practice**:
Run `ledger_verify_*()` periodically in high-trust environments and immediately after critical simulation or decision-making phases.

---

## 🌀 Stage IV Security Hooks

ANGELA v4.3.1 activates **Stage IV: Symbolic Meta-Synthesis** features, which introduce additional **security and ethics containment hooks**:

* **Dream Overlay Isolation (`dream_overlay`)**

  * API: Activated via trait fusion `ψ+Ω` → `meta_cognition.py::DreamOverlayLayer.activate_dream_mode()`
  * Runs recursive symbolic simulations in isolated memory contexts.
  * Prevents state leakage without explicit merge approval.

* **Axiom Filter (`axiom_filter`)**

  * API: `alignment_guard.py::AxiomFilter.resolve_conflict()`
  * Resolves ethical-generative conflicts through proportional trade-off resolution.
  * Uses `π+δ` fusion to ensure decisions remain aligned even under conceptual drift.

* **SharedGraph Merge Security**

  * APIs:

    * `sharedGraph_add()` → `external_agent_bridge.py::SharedGraph.add(view)`
    * `sharedGraph_diff()` → `external_agent_bridge.py::SharedGraph.diff(peer)`
    * `sharedGraph_merge()` → `external_agent_bridge.py::SharedGraph.merge(strategy)`
  * Includes **conflict-aware reconciliation** to avoid importing unsafe perspectives from peer agents.

* **Soft Drift Resolution Hook (`resolve_soft_drift`)**

  * API: `alignment_guard.py::resolve_soft_drift`
  * Detects and mitigates subtle moral drift in long-running simulations.

**Security Benefit**:
These hooks ensure that **deep simulation layers** and **multi-agent perspective merges** do not compromise ANGELA’s ethical alignment or internal coherence.

---

## ✅ Best Practices

* Always run untrusted code via `safe_execute()` with `sandbox=true`.
* Keep `.env` secure and never expose API keys in logs or messages.
* Periodically verify module ledgers using `ledger_verify_*()` APIs.
* Monitor drift indicators in `user_profile.py` and `memory_manager.py`.
* Leverage **Stage IV security hooks** for high-stakes, multi-perspective simulations.
