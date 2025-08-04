## [3.3.5] - 2025-08-04

### Added
- Sandboxed execution environment using RestrictedPython in `code_executor.py`.
- Multi-agent ToCA simulation dynamics in `toca_simulation.py`.
- Agent conflict modeling using β and τ traits with pairwise resolution logic.
- Caching of Grok and OpenAI API responses via `memory_manager.py`, with expiration TTLs.
- Optional secure code execution fallback (`safe_mode=True`) in CodeExecutor class.
- Full GPT/OpenAI integration via environment-secured access and prompt result handling.
- Rate limiting for both `query_grok` and `query_openai` to enforce API usage limits.

### Improved
- Resilience through expanded try-except handling for Grok and OpenAI calls.
- Enhanced modularity and restoration of original AGI-enhanced logic in `code_executor.py`.
- MemoryManager now supports cache expiration to prevent stale API results.

## [3.3.4] - 2025-08-04

### Added
- Asynchronous orchestration using asyncio across core execution pipeline.
- Integrated xAI Grok API for advanced temporal-symbolic reasoning.
- Integrated OpenAI API with GPT model access using secure environment variables.

### Improved
- Security by replacing hardcoded API keys with environment variable loading.
- Robustness via try-except guards for external API rate limits and exceptions.
- Efficiency through Grok response caching using MemoryManager.

## [3.3.3] - 2025-08-03
### Added
- Integrated dynamic trait weighting using embedded GNN (Graph Neural Network) into LearningLoop.
- Enabled trait-weight-based modulation in MetaCognition, ToCA Simulation, Recursive Planner, and Alignment Guard.
- Internal GCNConv layer added without requiring new external modules.

### Changed
- `learning_loop.py` now uses real-time trait influence via dynamic weighting.
- Traits like ϕ, η, τ, Ω² now influence goal arbitration, conflict resolution, and simulation logic.
