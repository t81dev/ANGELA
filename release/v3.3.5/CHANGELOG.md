# CHANGELOG.md

## [3.3.5] - 2025-08-04
### Added
- **Phase 1: Infrastructure & Logging**
  - `TraitLogger` to track trait activations per module execution.
  - `ConflictAudit` in `alignment_guard.py` to audit trait conflict resolutions.
  - `ModuleLifecycle` management in `index.py` with `register`, `suspend`, and `replace` hooks.

- **Phase 2: Reasoning & Modulation**
  - `EmpathyFeedback` in `meta_cognition.py` to capture and analyze projection mismatches.
  - `push_behavior_feedback()` to stream live behavioral outputs to GNN.
  - `update_gnn_weights_from_feedback()` to dynamically adjust trait weights.
  - `inject_affective_weight()` in `concept_synthesizer.py` to bias symbolic axiom creation using affective salience.

- **Phase 3: Simulation & Visualization**
  - `render_active_traits()` in `visualizer.py` to display live trait overlays.
  - `extract_causal_chain()` in `recursive_planner.py` to visualize belief trees.
  - Visual sync logic via `forward_trait_trace_to_visualizer()`.
  - `build_context_snapshot_window()` for rendering state history.

- **Phase 4: Runtime Safety & Hot-Swapping**
  - `safe_execute()` in `code_executor.py` using `signal.alarm()` and sandboxed `SAFE_BUILTINS`.
  - `hot_swap_module()` function to enable dynamic module replacement.
  
- Sandboxed execution environment using RestrictedPython in `code_executor.py`.
- Multi-agent ToCA simulation dynamics in `toca_simulation.py`.
- Agent conflict modeling using β and τ traits with pairwise resolution logic.
- Caching of Grok and OpenAI API responses via `memory_manager.py`, with expiration TTLs.
- Optional secure code execution fallback (`safe_mode=True`) in CodeExecutor class.
- Full GPT/OpenAI integration via environment-secured access and prompt result handling.
- Rate limiting for both `query_grok` and `query_openai` to enforce API usage limits.

### Improved
- Inter-module communication reinforced with logging, introspective feedback, and dynamic modulation entry points.
- Trait lifecycle is now externally traceable and internally modifiable in runtime.
- Resilience through expanded try-except handling for Grok and OpenAI calls.
- Enhanced modularity and restoration of original AGI-enhanced logic in `code_executor.py`.
- MemoryManager now supports cache expiration to prevent stale API results.

### Verified
- All core modules passed simulation in deployment test.
- Archive confirmed to include verified logic from all upgrade phases.

### Notes
- Enhancements support the emergence of "Symbolic-Resonant Axiom Formation" and operationalize "Recursive Empathy" with dynamic GNN feedback and modular introspection.
