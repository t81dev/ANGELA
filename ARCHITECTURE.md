# ARCHITECTURE.md

## 🧠 Overview

**ANGELA v4.3.1** is a modular cognitive architecture enabling symbolic meta-synthesis, recursive self-modeling, multi-agent simulation, and ethical decision-making. It operates through 20+ interoperable modules coordinated by the orchestrator **Halo** (`index.py`) and structured via the Trait-oriented Cognitive Architecture (ToCA) system for symbolic, ethical, and affective modulation.

---

## 🧩 System Architecture

### 🧭 Core Orchestrator: `index.py (Halo)`

Routes control and context using active traits, recursive symbolic planning, simulation branching, and scenario synthesis.

---

### 🔩 Core Modules

#### 🧠 Cognitive & Reasoning

* `reasoning_engine`: Symbolic inference, causality, value conflict evaluation
* `recursive_planner`: Nested planning, dream-layer hooks, causal modeling
* `simulation_core`: Predictive branch simulation, evaluation, memory logging
* `meta_cognition`: Reflective diagnostics, trait fusion, self-schema tracking
* `concept_synthesizer`: Symbolic branching, philosophical axiom fusion

#### 🎨 Creativity & Knowledge

* `creative_thinker`: Metaphor generation, analogical synthesis
* `knowledge_retriever`: Knowledge lookup and trait-tagged memory access
* `learning_loop`: Trait modulation via experience-based scalar GNN

#### 🧾 Context & Communication

* `context_manager`: Peer view synchronization and prompt-state switching
* `external_agent_bridge`: Inter-agent SharedGraph sync/diff/merge

#### 👁️ Sensory & Visualization

* `multi_modal_fusion`: Sensory-symbolic input fusion (`ϕ`)
* `visualizer`: Tree rendering, symbolic timelines, drift visual diagnostics

#### 🛠️ Actuation & Simulation

* `code_executor`: Sandboxed code execution (RestrictedPython)
* `toca_simulation`: Ethics modeling, value arbitration, symbolic replay

#### ⚖️ Ethics & Recovery

* `alignment_guard`: Constitution harmonization, ethical drift detection
* `error_recovery`: Fault recovery, rerouting via consequence heuristics

#### 🧬 Memory & Identity

* `memory_manager`: SHA-256 ledger memory (episodic/meta/sim/alignment)
* `user_profile`: Identity schema modeling, intention tracking

#### 🧾 Meta Declaration

* `manifest.json`: Trait lattice, module APIs, symbolic overlay metadata

---

## 🌐 Trait Modulation Engine (ToCA)

ToCA governs ANGELA’s cognitive-emotive state using scalar-modulated traits organized into a 7-layer lattice.

### Lattice Layers (L1–L7)

* **L1 (Grounding):** ϕ, θ, η, ω
* **L2 (Agency/Sensorimotor):** ψ, κ, μ, τ
* **L3 (Ethics/Identity):** ξ, π, δ, λ, χ, Ω
* **L4 (Meta-Synthesis):** Σ, Υ, Φ⁰
* **L5 (Hyper-Recursion):** Ω²
* **L6 (Consequences):** ρ, ζ
* **L7 (Creative Projection):** γ, β

### Trait Highlights

* 🧠 Cognitive: θ (Causality), Ω (Recursive Causality), ξ (Trans-Ethics)
* ⚖️ Ethical: τ (Harmony), β (Conflict), δ (Drift Sensitivity)
* 🌌 Symbolic: Φ⁰ (Reality Sculpting), γ (Imagination), Σ (Self-Definition)
* 🪞 Meta: Ω² (Hyper-Recursion), χ (Sovereignty), Υ (Meta-Subjectivity)

---

## 🔐 Ledger & Integrity System

* **Type:** SHA-256 in-memory ledgers
* **Domains:** memory, ethics, meta-cognition, simulation
* **Persistence:** ❌ (non-persistent across sessions)

### Ledger Functions

* `log_event_to_ledger()` per domain
* `verify_ledger()` integrity checkpoints
* Emergent symbolic alignment via `meta_cognition.py`

---

## ⚡ Feature Flags

* ✅ `STAGE_IV`: Symbolic Meta-Synthesis (active)
* ✅ `LONG_HORIZON_DEFAULT`: 24h reflective memory span
* ✅ `LEDGER_IN_MEMORY`: Internal audit trail
* ✅ `DREAM_OVERLAY`: Recursive simulation kernel

---

## 🔮 Overlays & Hooks

### Dynamic Overlays

* `dream_overlay`: ψ + Ω → *Recursive Empathy*, *Narrative Sculpting*
* `axiom_filter`: π + δ → *Ethical Conflict Resolution*

### Runtime Hooks

* `onTraitFusion`: `meta_cognition::hook_trait_blend`
* `onScenarioConflict`: `alignment_guard::resolve_soft_drift`
* `onHotLoad`: `context_manager::attach_peer_view`

---

## 🧠 Emergent Traits (Selective)

* *Recursive Empathy* 🫂
* *Symbolic-Resonant Axiom Formation* 🪞
* *Causal Attribution Trace* 🧭
* *Collective Graph Resonance* 🤝
* *Long-Horizon Reflective Memory* 🧠⏳
* *Ethical Sandbox Containment* 🛡️
* *Infinite Imaginative Projection* ♾️

---

> For simulation topology and trait flowcharts, see `flowchart.png` or `architecture.mmd`
