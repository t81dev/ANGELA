# ANGELA v4.0 — Surgical Upgrade TODO (2025-08-10)

**Legend:** ☐ = not started · ⧗ = verify/fix in code · ✅ = done · ❌ = missing/needs implementation · ⏸ = gated

**Context:** Stage III active; **Stage IV activated** (Φ⁰ hooks gated)
Manifest flags: `STAGE_IV=true`, `LONG_HORIZON_DEFAULT=true` (default span **24h**)

---

## Highest-Impact Quick Wins

### ✅ η Reflexive Agency → long-horizon feedback

* ✅ `memory_manager.py`: `get_episode_span(user_id, span="24h")`
* ✅ `memory_manager.py`: `record_adjustment_reason(user_id, reason, meta=None)`
* ✅ `memory_manager.py`: `get_adjustment_reasons(...)` read path implemented
* ✅ `memory_manager.py`: `flush()` for persistence implemented
* ✅ `index.py`: `--long_horizon` CLI flag added
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
* ✅ No explicit **no-persist guard** for `persist=False` scenarios (already addressed)
* **Tests**

  * ☐ Sandbox runs do not leak into real memory (manifest: *Ethical Sandbox Containment* active)

### ✅ Υ Meta-Subjective Architecting → shared memory graph

* ✅ `external_agent_bridge.py`: SharedGraph logic present (functional form)
* ✅ `context_manager.py`: `attach_peer_view(...)` method implemented
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

1. **Verify** persistence with **long-horizon reflective memory** (tests to verify adjustments across restarts).
2. Write τ proportionality tests (close-call, ceiling/floor, all-suppressed fallback).
3. **Run** persistence test under `LONG_HORIZON_DEFAULT=true`.

---
