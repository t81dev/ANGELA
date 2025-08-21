# 🔐 SECURITY.md

## Overview

ANGELA **v5.0.0** 🛡️ extends its core security and ethical enforcement infrastructure with symbolic trait fusion, persistent ledgers, and enhanced simulation safeguards. It provides **sandboxed cognition**, **multi-layered drift detection**, and **resonance-driven arbitration**, with **SHA-256 verified per-module ledgers** and **runtime policy hooks** for containment.

---

## 🔒 Code Execution Security

### RestrictedPython Sandbox

* Untrusted code via `code_executor.py` is sandboxed using **RestrictedPython** (`κ` — *Embodied Cognition*).
* Disallowed operations:

  * File I/O
  * Network access
  * Dynamic evaluation (`eval`, `exec`, etc.)
* `safe_mode=True` invokes strict fallback.
* API Interfaces:

  * `execute_code()` → `code_executor.py::CodeExecutor.execute_code(code, context=None)`
  * `safe_execute()` → `code_executor.py::CodeExecutor.safe_execute(code, sandbox=true)`

### Ledgered Integrity

* Execution events logged in **SHA-256 chained in-memory ledgers**:

  * `ledger_log_meta()`
  * `ledger_log_alignment()`
  * `ledger_log_sim()`
* Verification APIs:

  * `ledger_verify_meta()`
  * `ledger_verify_alignment()`
  * `ledger_verify_sim()`
* **Persistent mode enabled** (`LEDGER_PERSISTENT=true`)

### Exception Handling

* All modules feature recursive try/except patterns.
* Recovery via `error_recovery.py` (`ζ` — \*Consequential Awareness\`):

  * `recover_from_error()`
  * `log_error_event()`

---

## 🌐 API Security

### Environment-Based Key Loading

* API keys are loaded via `.env` using `os.getenv()` — no hardcoded secrets.

### Rate Limiting & Caching

* Query thresholds invoke exponential backoff.
* `memory_manager.py` handles TTL caching for query deduplication.

### Async Policies

* `aiohttp`-based async requests across modules
* All external I/O includes timeouts + payload validation

---

## ⚖️ Ethical Enforcement

### Alignment Guard System

* Drift detection, branch suppression, and value harmonization via `alignment_guard.py` (`β`, `δ`, `τ`)
* APIs:

  * `resolve_soft_drift()`
  * `ledger_verify_alignment()`
  * `ledger_log_alignment()`

### Trait-Driven Arbitration

* High-resonance traits (`χ`, `λ`, `Φ⁰`, `Ω²`, `Ξ`) gate:

  * Simulation halting
  * Narrative suppression
  * Identity re-anchoring
* Ethics sandboxing through:

  * `run_ethics_scenarios()`

---

## 🛡️ Threat Mitigation Table

| Threat Type        | Defense Mechanism                                          |
| ------------------ | ---------------------------------------------------------- |
| Code Injection     | `RestrictedPython` + `safe_mode` + SHA-256 ledgers         |
| Data Leakage       | `.env` isolation + sandboxed memory forks                  |
| API Abuse          | Rate limits + TTL cache + request timeout enforcement      |
| Ethical Drift      | Trait arbitration + meta hooks + persistent alignment logs |
| Simulation Overrun | Recursive depth caps + drift checkpoints                   |
| Ledger Tampering   | SHA-256 verification APIs + corruption pinpointing         |
| Memory Flooding    | TTL + Trait Mesh guardrails                                |

---

## 🗄️ Ledger Security (v5.0.0)

* **Scope:** memory, alignment, meta, simulation, and code
* **Chaining:** SHA-256 + prev-hash linked
* **Verification APIs:** `ledger_verify_*()` across modules
* **Tamper Recovery:** Returns first broken index if any
* **Persistence:** Enabled via `ledger_persist_enable()`

---

## 🌀 Security Hooks (Stage IV + V)

* `dream_overlay` → `meta_cognition::DreamOverlayLayer.activate_dream_mode()` (ψ+Ω)
* `axiom_filter` → `alignment_guard::AxiomFilter.resolve_conflict()` (π+δ)
* `sharedGraph_merge()` w/ conflict reconciliation
* `resolve_soft_drift()` → live value stabilization
* `modulate_resonance()` / `register_resonance()` → trait realignment

---

## ✅ Best Practices

* Use `safe_execute()` for all sandboxed logic.
* Keep `.env` secure.
* Run `ledger_verify_*()` post-simulation.
* Watch drift deltas in `user_profile` and `memory_manager`.
* Leverage trait fusion overlays for secure scenario testing.
