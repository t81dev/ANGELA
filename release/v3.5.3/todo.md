Here’s your updated **`todo.md`** with the alignment\_guard findings merged in and status changes reflecting manifest + code review results:

---

# ANGELA v4.0 — Surgical Upgrade TODO (2025-08-10)

**Legend:** ☐ = not started · ⧗ = verify in code · ✅ = done · ⏸ = gated/behind flag

**Context:** Stage III active; Stage IV not yet activated. Manifest confirms partial η, completed κ, planned τ/ξ/Υ/Σ, gated Φ⁺.

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

### ⧗ τ Constitution Harmonization → proportionality ethics

* ✅ `reasoning_engine.py`: `weigh_value_conflict(...)`
* ✅ `reasoning_engine.py`: `attribute_causality(...)`
* ✅ `alignment_guard.py`: proportional selection wired to `weigh_value_conflict` — **edge fixes needed**

  * ⧗ Patch `max_harm` propagation to avoid false suppressions
  * ⧗ Align audit numeric strings with calculation
* **Tests**

  * ☐ Nuanced outputs for close-call dilemmas under ceiling/floor/temperature
  * ☐ Verify causality audit present & well-formed

### ✅ ξ Trans-Ethical Projection → scenario sandbox

* ✅ `toca_simulation.py`: `run_ethics_scenarios(...)` present
* ✅ `meta_cognition.py`: ethics preview path stub present
* **Tests**

  * ☐ Sandbox runs do not leak into real memory (manifest: *Ethical Sandbox Containment* active)

### ☐ Υ Meta-Subjective Architecting → shared memory graph

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

### ⏸ Φ⁺ Reality Sculpting → Stage IV hooks (flag: `STAGE_IV=false`)

* ☐ `concept_synthesizer.py`: `branch_realities(...)` missing
* ☐ `toca_simulation.py`: `evaluate_branches(...)` missing
* ☐ `visualizer.py`: branch tree/promote UX missing
* **Tests**

  * ☐ Promote/dismiss branches without base corruption (manifest: *Branch Futures Hygiene* active)

---

## Immediate Next Steps

1. Add explicit `get_episode_span(...)` call in `meta_cognition` self-adjust loop.
2. Patch `alignment_guard` with robust `max_harm` calculation & audit numeric sync.
3. Add τ proportionality close-call / ceiling / floor test cases.
4. Implement `toca_simulation.evaluate_branches(...)` Stage-IV method.
5. Implement `external_agent_bridge.SharedGraph` + `context_manager` peer view hook.
6. Land Stage-IV branch stubs behind flag.

---

Do you want me to go ahead and **write the exact diff for `alignment_guard.py`** to fix the `max_harm` and audit issues so we can tick ⧗ to ✅ for τ? That’ll also make the test work smoother.
