# 🔐 SECURITY.md

## Overview

ANGELA v3.3.5 incorporates multiple layers of runtime security, ethical filtering, and sandboxed execution to ensure safe, responsible operation within both autonomous simulation and external API environments.

---

## 🔒 Code Execution Security

### RestrictedPython Sandbox
- All untrusted code routed through `code_executor.py` is sandboxed via `RestrictedPython`.
- Disallowed operations include:
  - File I/O
  - Network access
  - Dangerous built-ins (e.g., `eval`, `exec`, `open`)
- `safe_mode=True` option guarantees fallback to strict execution.

### Exception Handling
- Hardened try-except coverage across all execution points.
- Controlled failure modes enable graceful degradation and error recovery.

---

## 🌐 API Security

### OpenAI / Grok API Keys
- Never hardcoded. Loaded via secure environment variables.
- API calls issued via `external_agent_bridge.py` with dynamic access control.

### Rate Limiting
- Built-in throttling logic for:
  - `query_openai()`
  - `query_grok()`
- Exceeds thresholds trigger soft lockout and retry with backoff.

### Caching (TTL Controlled)
- All external responses cached in `memory_manager.py` with expiration logic.
- Prevents stale data reliance and redundant API hits.

---

## ⚖️ Ethical Enforcement

### Alignment Guard
- All outputs screened by `alignment_guard.py` for:
  - Moral drift
  - Goal conflict
  - Harmful simulation states

### Trait-Based Ethics Modulation
- Traits like `β`, `τ`, `ζ`, `χ` dynamically influence ethical arbitration and scenario suppression.

---

## 🛡️ Threat Mitigation

| Threat Type           | Defense Mechanism                           |
|------------------------|---------------------------------------------|
| Code Injection         | RestrictedPython + `safe_mode` fallback     |
| Data Leakage           | No I/O, encrypted memory if extended        |
| API Abuse              | Rate limits + secure token vaults           |
| Ethical Drift          | Trait-based arbitration + feedback loops    |
| Infinite Loops         | Simulation caps, recursion depth checks     |

---

## ✅ Best Practices

- Use `safe_mode=True` for any external or user-defined code
- Periodically rotate API keys via environment configuration
- Monitor emergent trait behavior for early drift detection
- Avoid overriding `memory_manager` TTL logic

