# 🔐 SECURITY.md

## Overview

ANGELA v3.5.1 incorporates advanced security, ethical enforcement, and sandboxed execution layers across all modules. It ensures safe, recursive cognition in both autonomous simulation and external API contexts, with dynamic drift monitoring and trait-modulated arbitration.

---

## 🔒 Code Execution Security

### RestrictedPython Sandbox
- All untrusted code routed through `code_executor.py` is sandboxed via `RestrictedPython`.
- Disallowed operations include:
  - File I/O
  - Network access
  - Dynamic execution (`eval`, `exec`, etc.)
- `safe_mode=True` ensures strict fallback security in runtime environments.

### Exception Handling
- Exception control embedded across all modules with expanded try-except tracing.
- Recursive error paths routed through `error_recovery.py` for symbolic diagnostics and correction.

---

## 🌐 API Security

### Environment-Based Key Loading
- API keys for OpenAI, Grok, and others are **never hardcoded**.
- Keys are securely accessed via `os.getenv()` within `.env` scope.

### Rate Limiting & Caching
- Rate limiting enforced across:
  - `query_openai()`, `query_grok()`
  - Exceeding threshold triggers exponential backoff
- TTL-controlled cache via `memory_manager.py` prevents overuse or repeated queries.

### Async Policy Integration
- Asynchronous external data requests handled via `aiohttp` in:
  - `visualizer.py`, `external_agent_bridge.py`, `user_profile.py`
- Policy-fetching follows strict timeout and validation rules.

---

## ⚖️ Ethical Enforcement

### Alignment Guard System
- Outputs screened by `alignment_guard.py` using:
  - Drift monitoring (`δ`)
  - Value conflict resolution (`τ`)
  - Simulation suppression (`β`, `ζ`)
- `meta_cognition.reflect_on_output()` triggers post-run ethics feedback.

### Trait-Driven Arbitration
- Traits like `χ`, `λ`, `Φ⁺`, `Ω`, and `Ω²` influence:
  - Simulation halting
  - Symbol suppression
  - Identity conflict warnings

---

## 🛡️ Threat Mitigation

| Threat Type           | Defense Mechanism                               |
|------------------------|-------------------------------------------------|
| Code Injection         | `RestrictedPython` + fallback `safe_mode`       |
| Data Leakage           | No I/O, API masking, scoped variable domains    |
| API Abuse              | Rate limit guards, secure `.env` vaults         |
| Ethical Drift          | Trait-based arbitration + drift feedback loop   |
| Simulation Overrun     | Depth-capped recursion in `simulation_core.py`  |
| Memory Flooding        | `DriftIndex` and TTL-based memory expiration    |

---

## ✅ Best Practices

- Use `safe_mode=True` for any user-provided or unknown logic blocks
- Store all keys in `.env` and restrict access to production variables
- Monitor drift via `user_profile.py` and `memory_manager.py` regularly
- Avoid bypassing TTL cache expiry in `memory_manager.py`
- Validate symbolic outputs using `meta_cognition.reflect_on_output()` post-generation

---
