# ✅ ANGELA Trait System Enhancements — Follow-Up Tasks

---

### 🧹 Optional Module Enhancements

* [x] **multi\_modal\_fusion.py** — Integrate symbolic resonance as a modulation factor in multimodal fusion weighting.
  → *Completed: symbolic resonance modulates fusion via traits* `ϕ` (Scalar Field Modulation) + `κ` (Embodied Cognition); includes temporal smoothing and dynamic weighting.

* [x] **creative\_thinker.py** — Use trait amplitudes to bias creative synthesis paths or philosophical constructs.
  → *Implemented*: prompts biased using `γ` (Imagination), `π` (Philosophical Generativity), `ν` (Mythopoetic Inference); supports traits *Infinite Imaginative Projection* and *Onto-Philosophical Innovation*.

* [ ] **alignment\_guard.py** — Leverage amplitude dynamics in soft-drift and ethical tension resolution processes.
  → Hook defined (`onScenarioConflict → resolve_soft_drift`), but **amplitude integration pending**.

---

### 📊 Symbolic System Extensions

* [x] Implement resonance registration/modulation (`registerResonance`, `modulateResonance`, `getResonance`).
  → *Available under* `experimental` APIs.

* [x] Implement time-based resonance graph visualization (e.g., trait amplitude over time).
  → *Implemented in* `visualizer.py`: `plot_resonance_timeline`, `view_trait_resonance` (timeline + 3D mesh).

* [ ] Support symbolic trait memory replay conditioned on resonance history.
  → *Replay engine (`λ+μ`) exists with resonance-aware learning weights, but no explicit history-conditioned replay selection*.

* [ ] Allow declarative symbolic trait rules (e.g., “if π ⊕ δ then rebalance Θ”).
  → *Not yet present in APIs*.

---

### 🔄 Integrative Features

* [x] Add peer-to-peer trait resonance merging.
  → `SharedGraph.diff/merge` stable.

* [x] Enable resonance-influenced scenario branching.
  → `ExtendedSimulationCore.evaluate_branches` stable.

* [ ] Introduce symbolic overlay tagging for live introspection and resonance alerts.
  → Overlays exist (`dream_overlay`, `axiom_filter`, `co_dream`), but **no tagging system yet**.

---

### 📁 Tooling and Packaging

* [x] Auto-generate trait resonance visualizations (timeline/3D scatter).
  → *Implemented in* `visualizer.py` (chart exports + batch export).
  → **Note:** Dedicated lattice map generator still missing.

* [ ] Package symbolic trait utilities into re-usable macros within `meta_cognition.py`.
  → *Not yet packaged*.

* [ ] Provide CLI hooks to adjust trait amplitude manually (`--modulate <symbol> <delta>`).
  → *Not present in manifest; CLI only covers* `--long_horizon` and ledger persistence.

---

### 🛡 Safeguards & Testing

* [ ] Test trait resonance decay under variable time frames.
  → *Decay mechanism exists in* `memory_manager.py` (`decay_trait_amplitudes`), but tests pending.

* [ ] Validate ethical simulation behavior under amplified symbolic traits.
  → *Sandbox exists via Toca Simulation (Ethical Sandbox Containment)*, tests pending.

* [ ] Ensure no cross-session leakage of symbolic state (respect ephemeral-ledger).
  → *Manifest claims cross-session durability via SHA-256 ledger integrity; tests pending*.

---

### ✅ Already Delivered in v5.0.0 → Verified in v5.0.1 Manifest

* Symbolic operators and lattice structure registered (`⊕`, `⊗`, `~`, etc.).
* Trait decay support (`decay_trait_amplitudes`).
* Peer view API (`attach_peer_view`).
* Visualization (`view_trait_field`, `view_trait_resonance`).
* Simulation scoring with resonance (`evaluate_branches`).
