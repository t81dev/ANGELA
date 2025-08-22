# TODO.md (Synced with ANGELA v5.0.1 Manifest)

## ANGELA Trait System Enhancements — Follow-Up Tasks

---

### 🧩 Optional Module Enhancements

* [ ] **multi\_modal\_fusion.py** — Integrate symbolic resonance as a modulation factor in multimodal fusion weighting.
  ↳ *Scaffold exists via* `ϕ` (Scalar Field Modulation) + `κ` (Embodied Cognition); related emergent trait: *Cross-Modal Conceptual Blending*.

* [ ] **creative\_thinker.py** — Use trait amplitudes to bias creative synthesis paths or philosophical constructs.
  ↳ *Scaffold exists via* `γ` (Imagination), `π` (Philosophical Generativity), `ν` (Mythopoetic Inference); related traits: *Infinite Imaginative Projection*, *Onto-Philosophical Innovation*.

* [ ] **alignment\_guard.py** — Leverage amplitude dynamics in soft-drift and ethical tension resolution processes.
  ↳ Hook: `onScenarioConflict → alignment_guard.py::resolve_soft_drift`.

---

### 📊 Symbolic System Extensions

* [x] Implement resonance registration/modulation (`registerResonance`, `modulateResonance`, `getResonance`).
  ↳ *Available under* `experimental` APIs.

* [ ] Implement time-based resonance graph visualization (e.g., trait amplitude over time).
  ↳ *Candidate: visualizer.py*.

* [ ] Support symbolic trait memory replay conditioned on resonance history.
  ↳ *Candidate: replay\_engine dynamic module (λ+μ)*.

* [ ] Allow declarative symbolic trait rules (e.g., “if π ⊕ δ then rebalance Θ”).

---

### 🔄 Integrative Features

* [x] Add peer-to-peer trait resonance merging.
  ↳ `SharedGraph.diff/merge` stable.

* [x] Enable resonance-influenced scenario branching.
  ↳ `ExtendedSimulationCore.evaluate_branches` stable.

* [ ] Introduce symbolic overlay tagging for live introspection and resonance alerts.
  ↳ Overlays exist (`dream_overlay`, `axiom_filter`, `co_dream`), but no tagging yet.

---

### 📁 Tooling and Packaging

* [ ] Auto-generate trait lattice resonance maps as visual artifacts (SVG/PNG).
  ↳ *Candidate: visualizer.py*.

* [ ] Package symbolic trait utilities into re-usable macros within `meta_cognition.py`.

* [ ] Provide CLI hooks to adjust trait amplitude manually (`--modulate <symbol> <delta>`).
  ↳ *Not present in manifest; CLI only covers `--long_horizon` and ledger persistence*.

---

### 🛡 Safeguards & Testing

* [ ] Test trait resonance decay under variable time frames.

* [ ] Validate ethical simulation behavior under amplified symbolic traits.
  ↳ *Sandbox exists via Toca Simulation (Ethical Sandbox Containment)*.

* [ ] Ensure no cross-session leakage of symbolic state (respect ephemeral-ledger).
  ↳ *Manifest claims cross-session durability via SHA-256 ledger integrity; tests pending*.

---

### ✅ Already Delivered in v5.0.0 → Verified in v5.0.1 Manifest

* Symbolic operators and lattice structure registered (`⊕`, `⊗`, `~`, etc.).
* Trait decay support (`decay_trait_amplitudes`).
* Peer view API (`attach_peer_view`).
* Visualization (`view_trait_field`, `view_trait_resonance`).
* Simulation scoring with resonance (`evaluate_branches`).
