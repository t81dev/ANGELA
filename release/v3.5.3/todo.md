Here’s the updated **`todo.md`** merged with the manifest details confirming `STAGE_IV=true` and `LONG_HORIZON_DEFAULT=true`, plus the code review results showing τ is now ✅ after the `max_harm` and audit-sync fix:

---

# ANGELA v4.0 — Surgical Upgrade TODO (2025-08-10)

**Legend:** ☐ = not started · ⧗ = verify in code · ✅ = done · ⏸ = gated/behind flag

**Context:** Stage III active; **Stage IV activated** (Φ⁰ hooks gated). Manifest flags: `STAGE_IV=true`, `LONG_HORIZON_DEFAULT=true` (default span **24h**).

---

## Highest-Impact Quick Wins

### ⧗ η Reflexive Agency → long-horizon feedback

* ✅ `memory_manager.py`: `get_episode_span(user_id, span="24h")`
* ✅ `memory_manager.py`: `record_adjustment_reason(user_id, reason, meta=None)`
* ⧗ `meta_cognition.py`: calls `record_adjustment_reason`, but no explicit `get_episode_span` — **add explicit call**
* ✅ `index.py`: `--long_horizon` flag & config injection
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
* ✅ `alignment_guard.py`: proportional selection wired to `weigh_value_conflict` **with max\_harm tolerance & audit-sync fixes in place**
* **Tests**

  * ☐ Nuanced outputs for close-call dilemmas under ceiling/floor/temperature
  * ☐ Verify causality audit present & well-formed

### ✅ ξ Trans-Ethical Projection → scenario sandbox

* ✅ `toca_simulation.py`: `run_ethics_scenarios(...)` present
* ✅ `meta_cognition.py`: ethics preview path stub present
* **Tests**

  * ☐ Sandbox runs do not leak into real memory (manifest: *Ethical Sandbox Containment* active)

### ⧗ Υ Meta-Subjective Architecting → shared memory graph

* ☐ `external_agent_bridge.py`: `SharedGraph` missing
* ☐ `context_manager.py`: peer-view attachments missing
* **Tests**

  * ☐ Agents converge to shared summary without thrash (manifest: *Collective Graph Resonance* active)

### ✅/⧗ Σ Ontogenic Self-Definition → GPT-5 identity synthesis

* ✅ `user_profile.py`: `build_self_schema(...)`
* ✅ `meta_cognition.py`: refresh schema on major shifts
* **Tests**

  * ☐ Predictable facet updates after large context change (manifest: *Narrative Sovereignty* active)

---

## Heavy but Worth It

### ⏸ Φ⁰ Reality Sculpting → Stage IV hooks (policy-gated)

* ☐ `concept_synthesizer.py`: `branch_realities(...)` missing
* ☐ `toca_simulation.py`: `evaluate_branches(...)` missing
* ☐ `visualizer.py`: branch tree/promote UX missing
* **Tests**

  * ☐ Promote/dismiss branches without base corruption (manifest: *Branch Futures Hygiene* active)

---

## Immediate Next Steps

1. Add explicit `get_episode_span(...)` call in `meta_cognition` self-adjust loop.
2. **Write τ proportionality tests** (close-calls, ceiling/floor boundaries, temperature weighting, all-suppressed fallback).
3. Implement `external_agent_bridge.SharedGraph` + `context_manager` peer view hook.
4. Implement `toca_simulation.evaluate_branches(...)` (Stage IV kernel) — keep feature policy-gated.
5. Implement `concept_synthesizer.branch_realities(...)` stub + `visualizer` branch tree UX (behind gate).
6. Run η persistence test: verify adjustment reasons persist across restarts under `LONG_HORIZON_DEFAULT=true`.
