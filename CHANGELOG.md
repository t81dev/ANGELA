# 📘 CHANGELOG.md

## [3.3.5] - 2025-08-04

### Added
- Sandboxed code execution with `RestrictedPython` in `code_executor.py`
- `safe_mode=True` fallback for secure execution
- Multi-agent conflict modeling in `toca_simulation.py` using traits `β` (conflict) and `τ` (harmonization)
- Full OpenAI and Grok API integration with secure env variable access
- Caching for external API responses in `memory_manager.py` with expiration TTLs
- Rate limiting enforcement for `query_grok` and `query_openai`

### Improved
- Expanded exception handling and resilience for API calls
- Trait-weighted dynamic planning using GNN in `learning_loop.py`
- `code_executor.py` modularity and fallbacks restored to AGI-enhanced form
- Cache expiration added in `memory_manager.py` to avoid stale results

---

## [3.3.4] - 2025-08-04

### Added
- Asynchronous orchestration via `asyncio` in core task pipeline
- Integrated xAI Grok API for temporal-symbolic reasoning
- Secure OpenAI API access using environment variables

### Improved
- Replaced hardcoded API keys with secure environment variables
- Added Grok response caching in `memory_manager.py`

---

## [3.3.3] - 2025-08-03

### Added
- Embedded Graph Neural Network (GNN) for trait-weighted modulation
- Dynamic trait influence in `meta_cognition`, `learning_loop`, `toca_simulation`, `recursive_planner`, `alignment_guard`

### Changed
- Trait arbitration now impacted by weights from traits `ϕ`, `η`, `τ`, `Ω²`
