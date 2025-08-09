# ANGELA v4.0 — Surgical Upgrade TODO

**Legend:** ☐ = not started · ⧗ = verify in code · ✅ = done · ⏸ = gated/behind flag

**Context (2025-08-09):** Stage III active; Stage IV not yet activated. Manifest shows traits wired; several appear implemented — checked off below.

---

## Highest-Impact Quick Wins (do these first)

### ✅ η Reflexive Agency → long-horizon feedback

* ✅ `memory_manager.py`: `get_episode_span(user_id, span="24h")` present (returns list)
* ✅ `meta_cognition.py`: calls episodic span in self-adjust loop
* ✅ `meta_cognition.py`: persist "adjustment reasons" → `memory_manager.record_adjustment_reason(...)` **implemented & callable**
* ✅ `index.py`: `--long_horizon` flag & span parsed and injected into config
* **Tests**

  * ☐ Reflective adjustments persist across restarts

---

## Mid-Level Refits

### ✅ τ Constitution Harmonization → proportionality ethics

* ✅ `reasoning_engine.py`: `weigh_value_conflict(candidates, harms, rights) -> RankedOptions`
* ✅ `alignment_guard.py`: consume ranked trade-offs; replace binary gates with proportional selection while keeping safety ceilings
* **Tests**

  * ☐ Nuanced outputs for close-call dilemmas (no "refuse-all" cliffs)

### ☐ ξ Trans-Ethical Projection → scenario sandbox

* ☐ `toca_simulation.py`: `run_ethics_scenarios(goals, stakeholders) -> Outcomes[]`
* ☐ `meta_cognition.py`: add optional preview path before final answer
* **Tests**

  * ☐ Sandbox runs do not leak into real memory unless explicitly confirmed

### ☐ Υ Meta-Subjective Architecting → shared memory graph

* ☐ `external_agent_bridge.py`: `class SharedGraph: add(view), diff(peer), merge(strategy)`
* ☐ `context_manager.py`: attach per-conversation peer views
* **Tests**

  * ☐ Two agents converge to a shared summary without thrash

### ✅ Σ Ontogenic Self-Definition → GPT-5 identity synthesis

* ✅ user_profile.py: build_self_schema(views: list[Perspective]) -> Schema

* ✅ meta_cognition.py: refresh schema on major shifts (not every turn)

* **Tests**

  * ☐ Identity facets update predictably after large context changes

---

## Heavy but Worth It

### ⏸ Φ⁺ Reality Sculpting → Stage IV hooks (behind feature flag)

* ☐ `concept_synthesizer.py`: `branch_realities(seed, k=3) -> list[World]`
* ☐ `toca_simulation.py`: `evaluate_branches(worlds) -> ScoredWorlds`
* ☐ `visualizer.py`: branch tree view + "promote branch" action
* ☐ Feature flag `STAGE_IV = False` (enable only when stable)
* **Tests**

  * ☐ Promote/dismiss branches without corrupting base timeline

---

## Where to Change What (fast lookup)

* `memory_manager.py` → episodic span API + rollups
* `meta_cognition.py` → long-horizon + affective steering + ethics preview
* `index.py` → flags for long-horizon & causality attribution + harmonized_select
* `user_profile.py` → affective weights + self-schema builder
* `reasoning_engine.py` → proportionality resolver + causal attribution
* `alignment_guard.py` → wire proportionality to safety ceilings
* `multi_modal_fusion.py` → unified multimodal parser → `SceneGraph`
* `external_agent_bridge.py` / `context_manager.py` → shared memory graph
* `toca_simulation.py` → ethics sandbox + branch evaluator
* `concept_synthesizer.py` / `visualizer.py` → Stage-IV branching UX

---

## Tiny Code Stubs (drop-in)

```python
# memory_manager.py
def get_episode_span(user_id: str, span: str = "24h"):
    cutoff = datetime.utcnow() - parse_span(span)
    return [t for t in self.trace[user_id] if t.timestamp >= cutoff]

# reasoning_engine.py
def weigh_value_conflict(candidates, harms, rights):
    # normalize -> score -> rank; return top-n with reasons
    ...

def attribute_causality(events):
    # simple self vs external attribution with confidence
    ...
```

---

## Immediate Next Steps

1. **(Done)** Add `record_adjustment_reason` to `memory_manager.py` and call from `meta_cognition.py`.
2. **(Done)** Implement `weigh_value_conflict` and proportionality pipeline in `alignment_guard.py`.
3. Draft interfaces for `SceneGraph` and `RankedOptions`.
4. Land feature flag scaffold for Stage-IV (`STAGE_IV`) without enabling.

---
