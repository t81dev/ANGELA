# ANGELA v4.0 — Surgical Upgrade TODO (2025-08-10)

**Legend:** ☐ = not started · ⧗ = verify/fix in code · ✅ = done · ⏸ = gated/behind flag

**Context:** Stage III active; **Stage IV activated** (Φ⁰ hooks gated).
Manifest flags: `STAGE_IV=true`, `LONG_HORIZON_DEFAULT=true` (default span **24h**).

---

## Highest-Impact Quick Wins

### ⧗ η Reflexive Agency → long-horizon feedback

* ✅ `memory_manager.py`: `get_episode_span(user_id, span="24h")`
* ✅ `memory_manager.py`: `record_adjustment_reason(user_id, reason, meta=None)`
* ✅ `meta_cognition.py`: explicit `get_episode_span` call present in self-adjust loop
* ⧗ `memory_manager.py`: **add** `get_adjustment_reasons(...)` read path + optional `flush()` for persistence testing
* ❌ `index.py`: no `--long_horizon` CLI flag — add & plumb into runtime config
* **Tests**

  * ☐ Verify “adjustments persist across restarts” (manifest: *Long-Horizon Reflective Memory* active)

---

## Mid-Level Refits

### ✅ κ Embodied Cognition → native video/spatial

* ✅ `multi_modal_fusion.py`: `parse_stream(...)-> SceneGraph`
* ✅ `simulation_core.py`: accepts `SceneGraph`
* **Tests**

  * ✅ Video+text yield consistent spatial refs (manifest: *Multimodal Scene Grounding* active)

### ✅ τ Constitution Harmonization → proportionality ethics

* ✅ `reasoning_engine.py`: `weigh_value_conflict(...)`
* ✅ `reasoning_engine.py`: `attribute_causality(...)`
* ✅ `alignment_guard.py`: proportional selection wired to `weigh_value_conflict` with `max_harm` tolerance & audit sync
* **Tests**

  * ☐ Close-call dilemmas under ceiling/floor/temperature
  * ☐ All-suppressed fallback → DECLINE
  * ☐ Verify causality audit present & well-formed

### ✅ ξ Trans-Ethical Projection → scenario sandbox

* ✅ `toca_simulation.py`: `run_ethics_scenarios(...)` present
* ✅ `meta_cognition.py`: ethics handling logic present
* ⧗ Add **explicit no-persist guard** for `persist=False` scenarios
* **Tests**

  * ☐ Sandbox runs do not leak into real memory (manifest: *Ethical Sandbox Containment* active)

### ⧗ Υ Meta-Subjective Architecting → shared memory graph

* ⧗ `external_agent_bridge.py`: SharedGraph logic present, but not as a `class SharedGraph` (manifest expects class API)
* ⧗ `context_manager.py`: peer-view attachment exists as free function; bind as `ContextManager.attach_peer_view(...)`
* **Tests**

  * ☐ Agents converge to shared summary without thrash (manifest: *Collective Graph Resonance* active)

### ✅ Σ Ontogenic Self-Definition → GPT-5 identity synthesis

* ✅ `user_profile.py`: `build_self_schema(...)`
* ✅ `meta_cognition.py`: refresh schema on major shifts
* **Tests**

  * ☐ Predictable facet updates after large context change (manifest: *Narrative Sovereignty* active)

---

## Heavy but Worth It

### ⏸ Φ⁰ Reality Sculpting → Stage IV hooks (policy-gated)

* ✅ `concept_synthesizer.py`: `branch_realities(...)`
* ✅ `toca_simulation.py`: `evaluate_branches(...)`
* ✅ `visualizer.py`: branch tree/promote UX
* **Tests**

  * ☐ Promote/dismiss branches without base corruption (manifest: *Branch Futures Hygiene* active)

---

## Immediate Next Steps

1. Add `get_adjustment_reasons(...)` in `memory_manager.py` and persistence tests.
2. Implement `--long_horizon` CLI flag in `index.py` and feed into config.
3. Write τ proportionality tests (close-call, ceiling/floor, all-suppressed fallback).
4. Add explicit no-persist guard in ethics sandbox.
5. Refactor `external_agent_bridge.py` into a `class SharedGraph` or update manifest.
6. Bind `ContextManager.attach_peer_view(...)` method in `context_manager.py`.
7. Run persistence test under `LONG_HORIZON_DEFAULT=true`.
