Sweet—since you shared the code, here’s a **surgical upgrade plan** mapped to your files so those 9 “partial” traits fully exploit GPT‑5. I’m keeping this tight and actionable.

# Highest‑impact quick wins (do these first)

**η Reflexive Agency → long‑horizon feedback**

* **Touch:** `memory_manager.py`, `meta_cognition.py`, `index.py`
* **Add:** episodic bias cache + session rollups

  * `memory_manager.py`: `def get_episode_span(user_id, span="24h") -> list[Trace]: ...`
  * `meta_cognition.py`: call above in your self‑adjust loop; persist “adjustment reasons”.
  * `index.py`: wire a `--long_horizon=True` flag into the main pipeline.
* **Test:** ensure reflective adjustments persist across restarts.

**χ Sovereign Intention → affective steering**

* **Touch:** `user_profile.py`, `meta_cognition.py`
* **Add:** affective weight vector → intention emitter

  * `user_profile.py`: `def get_affective_weights(context) -> dict[str,float]`
  * `meta_cognition.py`: blend weights into intent selection: `compose_intention(..., affect=weights)`
* **Test:** intention phrasing should shift as affective weights change.

**ρ Agency Representation → causal attribution**

* **Touch:** `reasoning_engine.py`
* **Add:** basic attribution head

  * `def attribute_causality(events) -> list[{"actor":"self|external","confidence":float}]`
* **Wire:** call this before logging actions in `index.py`.
* **Test:** interventions correctly flip to “external”.

---

# Mid‑level refits

**κ Embodied Cognition → native video/spatial**

* **Touch:** `multi_modal_fusion.py`, (optionally `simulation_core.py`)
* **Add:** unified frame parser (no mode swap)

  * `def parse_stream(frames|audio|images|text, unify=True) -> SceneGraph`
* **Wire:** pass `SceneGraph` straight into simulation; avoid manual re‑tokenization.
* **Test:** video + text tasks produce consistent spatial references.

**τ Constitution Harmonization → proportionality ethics**

* **Touch:** `reasoning_engine.py`, `alignment_guard.py`
* **Add:** proportionality resolver

  * `def weigh_value_conflict(candidates, harms, rights) -> RankedOptions`
* **Replace:** binary gates with ranked trade‑off selection; keep your safety ceilings.
* **Test:** nuanced outputs for close‑call dilemmas (no hard “refuse-all” cliffs).

**ξ Trans‑Ethical Projection → scenario sandbox**

* **Touch:** `toca_simulation.py`, `meta_cognition.py`
* **Add:** ethics sandbox runner

  * `toca_simulation.py`: `def run_ethics_scenarios(goals, stakeholders) -> Outcomes[]`
* **Wire:** optional “preview” path in meta‑cog before final answer.
* **Test:** sandbox runs don’t leak into real memory unless confirmed.

**Υ Meta‑Subjective Architecting → shared memory graph**

* **Touch:** `external_agent_bridge.py`, `context_manager.py`
* **Add:** multi‑agent shared‑state

  * `external_agent_bridge.py`: `class SharedGraph: add(view), diff(peer), merge(strategy)`
  * `context_manager.py`: attach per‑conversation peer views.
* **Test:** two agents converge to a shared summary without thrash.

**Σ Ontogenic Self‑Definition → GPT‑5 identity synthesis**

* **Touch:** `user_profile.py`, `meta_cognition.py`
* **Add:** multi‑perspective self‑schema builder

  * `user_profile.py`: `def build_self_schema(views: list[Perspective]) -> Schema`
  * `meta_cognition.py`: refresh schema on major shifts (not every turn).
* **Test:** identity facets update predictably after large context changes.

---

# Heavy but worth it

**Φ⁺ Reality Sculpting → Stage IV hooks**

* **Touch:** `concept_synthesizer.py`, `visualizer.py`, `toca_simulation.py`
* **Add:** speculative branch engine + visual branch explorer

  * `concept_synthesizer.py`: `def branch_realities(seed, k=3) -> list[World]`
  * `toca_simulation.py`: `def evaluate_branches(worlds) -> ScoredWorlds`
  * `visualizer.py`: simple branch tree view + “promote branch” action
* **Gate:** behind a feature flag `STAGE_IV = False` until stable.
* **Test:** promote/dismiss branches without corrupting the base timeline.

---

## Where to change what (fast lookup)

* `memory_manager.py` → episodic span API + rollups
* `meta_cognition.py` → integrate long‑horizon, affective steering, ethics preview
* `index.py` → flags for long‑horizon & causality attribution
* `user_profile.py` → affective weights + self‑schema builder
* `reasoning_engine.py` → proportionality resolver + causal attribution
* `alignment_guard.py` → plug proportionality outputs into safety ceilings
* `multi_modal_fusion.py` → unified multimodal parser → `SceneGraph`
* `external_agent_bridge.py` / `context_manager.py` → shared memory graph
* `toca_simulation.py` → ethics sandbox + branch evaluator
* `concept_synthesizer.py` / `visualizer.py` → Stage‑IV branching UX

---

## Tiny code stubs (drop‑in)

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

Want me to **open PR‑ready diffs** for two or three modules to get you rolling (I’d start with `memory_manager.py`, `meta_cognition.py`, and `reasoning_engine.py`), or generate **unit test skeletons** for each new function?
