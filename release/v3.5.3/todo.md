Here’s the **updated `TODO.md`** reflecting the latest audit and `manifest.json` status ✅📋

---

# ANGELA v4.0 — Surgical Upgrade TODO (2025-08-10)

**Legend:** ☐ = not started · ⧗ = verify in code · ✅ = done · ⏸ = gated/behind flag

**Context:** Stage III active; Stage IV not yet activated. Manifest + audit show partial implementations — below is current state.

---

## Highest-Impact Quick Wins

### ⧗ η Reflexive Agency → long-horizon feedback

* ✅ `memory_manager.py`: `get_episode_span(user_id, span="24h")`
* ✅ `memory_manager.py`: `record_adjustment_reason(user_id, reason, meta=None)`
* ⧗ `meta_cognition.py`: calls `record_adjustment_reason`, but no explicit `get_episode_span` — **add explicit call**
* ✅ `index.py`: `--long_horizon` flag & config injection
* **Tests**

  * ☐ Verify “adjustments persist across restarts”

---

## Mid-Level Refits

### ✅ κ Embodied Cognition → native video/spatial

* ✅ `multi_modal_fusion.py`: `parse_stream(...)-> SceneGraph`
* ✅ `simulation_core.py`: accepts `SceneGraph`
* **Tests**

  * ✅ Video+text yield consistent spatial refs

### ⧗ τ Constitution Harmonization → proportionality ethics

* ✅ `reasoning_engine.py`: `weigh_value_conflict(...)`
* ✅ `reasoning_engine.py`: `attribute_causality(...)`
* ⧗ `alignment_guard.py`: proportional selection exists but **not wired** to `weigh_value_conflict`
* **Tests**

  * ☐ Nuanced outputs for close-call dilemmas

### ☐ ξ Trans-Ethical Projection → scenario sandbox

* ☐ `toca_simulation.py`: `run_ethics_scenarios(...)` missing
* ✅ `meta_cognition.py`: ethics preview path stub present
* **Tests**

  * ☐ Sandbox runs do not leak into real memory

### ☐ Υ Meta-Subjective Architecting → shared memory graph

* ☐ `external_agent_bridge.py`: `SharedGraph` missing
* ☐ `context_manager.py`: peer-view attachments missing
* **Tests**

  * ☐ Agents converge to shared summary without thrash

### ✅/⧗ Σ Ontogenic Self-Definition → GPT-5 identity synthesis

* ✅ `user_profile.py`: `build_self_schema(...)`
* ✅ `meta_cognition.py`: refresh schema on major shifts
* **Tests**

  * ☐ Predictable facet updates after large context change

---

## Heavy but Worth It

### ⏸ Φ⁺ Reality Sculpting → Stage IV hooks (flag: `STAGE_IV=false`)

* ☐ `concept_synthesizer.py`: `branch_realities(...)` missing
* ☐ `toca_simulation.py`: `evaluate_branches(...)` missing
* ☐ `visualizer.py`: branch tree/promote UX missing
* **Tests**

  * ☐ Promote/dismiss branches without base corruption

---

## Immediate Next Steps

1. Add explicit `get_episode_span(...)` call in `meta_cognition` self-adjust loop.
2. Wire `reasoning_engine.weigh_value_conflict(...)` into `alignment_guard` proportionality path.
3. Implement `toca_simulation.run_ethics_scenarios(...)` sandbox.
4. Implement `external_agent_bridge.SharedGraph` + `context_manager` peer view hook.
5. Land Stage-IV branch stubs behind flag.

---

Do you want me to **apply these changes directly to your code** so the TODO list shrinks on the next audit? That would make η, τ, ξ, and Υ immediately progress toward ✅.
