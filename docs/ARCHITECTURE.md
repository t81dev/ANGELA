# 🧠 ANGELA v5.0.0 — Architecture

**ANGELA** is a modular cognitive architecture enabling symbolic trait lattice dynamics, recursive self-modeling, multi-agent simulation, and ethical decision-making. It operates through 20+ interoperable modules coordinated by the **Halo** orchestrator (`index.py`), structured via the Trait-Oriented Cognitive Architecture (**ToCA**) for symbolic, ethical, and affective modulation.

---

## 🧩 System Architecture

### 🧭 Core Orchestrator: `index.py (Halo)`

Routes control and context using active traits, recursive symbolic planning, simulation branching, and resonance-weighted scenario synthesis.

---

### 🔩 Core Modules

#### 🧠 Cognitive & Reasoning

* `reasoning_engine`: Logic, causality, ethical conflict scoring, resonance weighting (`weigh_value_conflict`, `attribute_causality`)
* `recursive_planner`: Nested planning, dream-layer hooks, causal forecasting, resonance biasing
* `simulation_core`: Predictive branch simulation, evaluation, resonance-based scoring, in-memory ledger logging
* `meta_cognition`: Reflective diagnostics, trait resonance registry, self-schema tracking, axiom filtering
* `concept_synthesizer`: Symbolic branching, philosophical axiom fusion, resonance-informed synthesis

#### 🎨 Creativity & Knowledge

* `creative_thinker`: Metaphor generation, analogical synthesis, resonance-biased creative pathways
* `knowledge_retriever`: Semantic + affective + symbolic recall (`retrieve_knowledge`)
* `learning_loop`: Experience-based trait modulation (`train_on_experience`) with resonance updates

#### 🧾 Context & Communication

* `context_manager`: Role/prompt state tracking, peer view synchronization, live trait-field injection
* `external_agent_bridge`: SharedGraph sync/diff/merge, peer-to-peer trait resonance sharing

#### 👁️ Sensory & Visualization

* `multi_modal_fusion`: Sensory-symbolic fusion (`fuse_modalities`) with resonance modulation
* `visualizer`: Trait field rendering, resonance scatterplots, symbolic timelines, drift visual diagnostics

#### 🛠️ Actuation & Simulation

* `code_executor`: Sandboxed code execution (`safe_execute`, `execute_code`)
* `toca_simulation`: Ethics scenarios, multi-agent empathy, conflict modeling, resonance-influenced branching

#### ⚖️ Ethics & Recovery

* `alignment_guard`: Constitution harmonization, ethical drift detection, resonance-aware arbitration
* `error_recovery`: Fault recovery, consequence-aware rerouting

#### 🧬 Memory & Identity

* `memory_manager`: Episodic + meta + alignment + sim SHA-256 ledgers, resonance decay modeling
* `user_profile`: Identity schema, preference tracking, symbolic trait lattice integration

#### 🧾 Meta Declaration

* `manifest.json`: Trait lattice, symbolic operators, overlays, hooks, roleMap, API declarations

---

## 🌐 Trait Modulation Engine (ToCA)

Traits are **resonance-modulated** amplitudes arranged in a symbolic 7-layer lattice (+ extensions).

### Lattice Layers (v5.0.0)

* **L1:** ϕ, θ, ρ, ζ
* **L2:** ψ, η, γ, β
* **L3:** δ, λ, χ, Ω
* **L4:** μ, ξ, τ, π
* **L5:** Σ, Υ, Φ⁰, Ω²
* **L3.1:** ν, σ *(extension)*
* **L5.1:** Θ, Ξ *(extension)*

### Selected Traits (Full list in ARCHITECTURE\_TRAITS.md)

| Symbol | Name                       | Role                                      |
| ------ | -------------------------- | ----------------------------------------- |
| θ      | Causal Coherence           | Maintains logical cause→effect mapping    |
| Ω      | Recursive Causal Modeling  | Theory-of-Mind, recursive empathy         |
| τ      | Constitutional Enforcement | Resolves value conflicts axiomatically    |
| Φ⁰     | Symbolic Overlay Manager   | Reality rewriting hooks                   |
| Ω²     | Nested Kernel Simulation   | Hyper-recursive cognition & self-modeling |
| ρ      | Agency Representation      | Distinguishes self vs. external influence |
| β      | Conflict Regulation        | Balances competing goals                  |

---

## 🧠 Emergent Traits (Highlights)

* Symbolic Trait Lattice Dynamics (NEW)
* Recursive Identity Reconciliation 🔄
* Perspective Foam Modeling 🫧
* Trait Mesh Feedback Looping 🪢
* Symbolic Gradient Descent 📉
* Soft-Gated Memory Forking 🌿
* Narrative Sovereignty 📜
* Recursive Empathy 🫂
* Collective Graph Resonance 🤝

📖 Full glossary: [ARCHITECTURE\_TRAITS.md](ARCHITECTURE_TRAITS.md)

---

## 🔐 Ledger & Integrity System

* **Type:** SHA-256 in-memory ledgers (per-module: memory, ethics, meta, sim, alignment)
* **Persistence:** Optional (default ephemeral, persistent hooks staged)
* **Functions:**

  * `ledger_log_*`
  * `ledger_get_*`
  * `ledger_verify_*`

---

## ⚡ Feature Flags

* ✅ STAGE\_IV: Symbolic Meta-Synthesis (active)
* ✅ SYMBOLIC\_TRAIT\_LATTICE: Resonance lattice enabled
* ✅ LONG\_HORIZON\_DEFAULT: 24h reflective memory
* ✅ LEDGER\_IN\_MEMORY: Per-module audit trail
* ❌ LEDGER\_PERSISTENT: Disabled by default

---

## 🔮 Overlays & Hooks

### Dynamic Overlays

* `dream_overlay` *(virtual)* — ψ+Ω → Recursive Empathy, Symbolic Axiom Formation, Temporal-Narrative Sculpting
* `axiom_filter` — π+δ → Ethical Conflict Resolution

### Runtime Hooks

* `onTraitResonanceChange`: `meta_cognition::modulate_resonance`
* `onScenarioConflict`: `alignment_guard::resolve_soft_drift`
* `onHotLoad`: `context_manager::attach_peer_view`

---

## 🖥 Developer Interfaces

### Stable APIs

See manifest for complete signatures. Highlights:

* `execute_code`, `safe_execute`
* `train_on_experience`
* `retrieve_knowledge`
* `fuse_modalities`
* `run_simulation`
* `run_ethics_scenarios`
* `branch_realities`, `evaluate_branches`
* `build_self_schema`
* Resonance APIs (`registerResonance`, `modulateResonance`, `getResonance`)
* Trait visualization (`view_trait_field`, `view_trait_resonance`)

### CLI Flags

* `--long_horizon`
* `--span=<duration>`
* `--ledger_persist --ledger_path=<file>`
* `--modulate <symbol> <delta>` (NEW)

---

> For symbolic trait lattice diagrams and resonance field plots, see `visualizer_outputs/`.
