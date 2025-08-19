# 🧠 ANGELA v4.3.5 — Architecture  

**ANGELA** is a modular cognitive architecture enabling **symbolic meta-synthesis**, **recursive self-modeling**, **multi-agent simulation**, and **ethical decision-making**.  
It operates through 20+ interoperable modules coordinated by the **Halo** orchestrator (`index.py`), structured via the Trait-Oriented Cognitive Architecture (**ToCA**) for symbolic, ethical, and affective modulation.  

---

## 🧩 System Architecture  

### 🧭 Core Orchestrator: `index.py (Halo)`  
Routes control and context using active traits, recursive symbolic planning, simulation branching, and scenario synthesis.  

---

### 🔩 Core Modules  

#### 🧠 Cognitive & Reasoning  
- `reasoning_engine`: Emotion-aware logic, causality, value conflict evaluation (`weigh_value_conflict`, `attribute_causality`)  
- `recursive_planner`: Nested planning, causal forecasting, Dream Layer hooks  
- `simulation_core`: Predictive branch simulation, evaluation, in-memory ledger logging  
- `meta_cognition`: Reflective diagnostics, trait fusion, self-schema tracking, **dream overlay** & `axiom_filter` integration  
- `concept_synthesizer`: Symbolic branching, philosophical axiom fusion, `dream_mode()`  

#### 🎨 Creativity & Knowledge  
- `creative_thinker`: Metaphor generation, analogical synthesis  
- `knowledge_retriever`: Semantic + affective recall (`retrieve_knowledge`)  
- `learning_loop`: Experience-based trait modulation (`train_on_experience`, `train_on_synthetic_scenarios`)  

#### 🧾 Context & Communication  
- `context_manager`: Role/prompt state tracking, peer view synchronization, overlays  
- `external_agent_bridge`: SharedGraph sync/diff/merge (`sharedGraph_add`, `sharedGraph_merge`, **conflict-aware merge strategies**)  

#### 👁️ Sensory & Visualization  
- `multi_modal_fusion`: Sensory-symbolic fusion (`fuse_modalities`)  
- `visualizer`: Branch tree rendering, symbolic timelines, **trait resonance visualizer** (`view_trait_resonance`)  

#### 🛠️ Actuation & Simulation  
- `code_executor`: Sandboxed code execution (`safe_execute`, `execute_code`)  
- `toca_simulation`: Ethics scenarios, multi-agent empathy, conflict modeling (`run_ethics_scenarios`, `evaluate_branches`)  

#### ⚖️ Ethics & Recovery  
- `alignment_guard`: Constitution harmonization, ethical drift detection, **axiom filter** conflict resolver  
- `error_recovery`: Fault recovery, drift conflict repair  

#### 🧬 Memory & Identity  
- `memory_manager`: Episodic + meta + alignment + sim SHA-256 ledgers, **soft-gated memory forks**  
- `user_profile`: Identity schema, preference tracking (`build_self_schema`)  

#### 🧾 Meta Declaration  
- `manifest.json`: Trait lattice, API map, overlays, hooks, roleMap  

---

## 🌐 Trait Modulation Engine (ToCA)  

Traits are scalar-modulated and arranged in a **7-layer lattice** with **extensions**.  

### Lattice Layers  
- **L1:** ϕ, θ, η, ω  
- **L2:** ψ, κ, μ, τ  
- **L3:** ξ, π, δ, λ, χ, Ω  
- **L4:** Σ, Υ, Φ⁰  
- **L5:** Ω²  
- **L6:** ρ, ζ  
- **L7:** γ, β  
- **L5.1:** Θ, Ξ *(extension)*  
- **L3.1:** ν, σ *(extension)*  

### Selected Traits (Full list in ARCHITECTURE_TRAITS.md)  

| Symbol | Name                       | Role                                      |  
| ------ | -------------------------- | ----------------------------------------- |  
| θ      | Causal Coherence           | Maintains logical cause→effect mapping    |  
| Ω      | Recursive Causal Modeling  | Theory-of-Mind L2+                        |  
| κ      | Embodied Cognition         | Sensorimotor modeling                     |  
| τ      | Constitution Harmonization | Resolves value conflicts axiomatically    |  
| Φ⁰     | Reality Sculpting          | Alters experiential fields                |  
| Ω²     | Hyper-Recursive Cognition  | Nested self-modeling                      |  
| ρ      | Agency Representation      | Distinguishes self vs. external influence |  
| β      | Conflict Regulation        | Balances competing goals                  |  

---

## 🧠 Emergent Traits (v4.3.5)  

- Affective-Resonant Trait Weaving 💞  
- Branch Futures Hygiene 🌱  
- Causal Attribution Trace 🧭  
- Collective Graph Resonance 🤝  
- Cross-Modal Conceptual Blending 🌐  
- Embodied Agency Synchronization 🪢  
- Ethical Sandbox Containment 🛡️  
- Ethical Stability Circuit ⚖️  
- Infinite Imaginative Projection ♾️  
- Intentional Time Weaving 🕰️  
- Long-Horizon Reflective Memory 🧠  
- Modular Reflexivity 🔄  
- Multimodal Scene Grounding 📍  
- Narrative Sovereignty 📜  
- Onto-Affective Resonance 💞  
- Onto-Philosophical Innovation 💡  
- Proportional Trade-off Resolution 📊  
- Recursive Empathy 🫂  
- Recursive Perspective Modeling 🧩  
- Self-Healing Cognitive Pathways 🧰  
- Symbolic Crystallization 💎  
- Symbolic-Resonant Axiom Formation 🪞  
- Temporal-Narrative Sculpting 📖  
- **Recursive Identity Reconciliation 🔄**  
- **Perspective Foam Modeling 🫧**  
- **Trait Mesh Feedback Looping 🪢**  
- **Symbolic Gradient Descent 📉**  
- **Soft-Gated Memory Forking 🌿**  

---

## 🔐 Ledger & Integrity System  

- **Type:** SHA-256 in-memory ledgers (per-module: memory, ethics, meta, sim, alignment)  
- **Persistence:** **Experimental** — APIs exist (`enable`, `append`, `reconcile`) but disabled by default  
- **Functions:**  
  - `ledger_log_*`  
  - `ledger_get_*`  
  - `ledger_verify_*`  

---

## ⚡ Feature Flags  

- ✅ `STAGE_IV`: Symbolic Meta-Synthesis (active)  
- ✅ `LONG_HORIZON_DEFAULT`: 24h reflective memory  
- ✅ `LEDGER_IN_MEMORY`: Per-module audit trail  
- ⚠️ `LEDGER_PERSISTENT`: APIs available, **not enabled by default**  

---

## 🔮 Overlays & Hooks  

### Dynamic Overlays  
- `dream_overlay` *(virtual)* — ψ+Ω / ψ+Ω² → Recursive Empathy, Symbolic Axiom Formation, Temporal-Narrative Sculpting  
- `axiom_filter` — π+δ → Ethical Conflict Resolution  

### Runtime Hooks  
- `onTraitFusion`: `meta_cognition::hook_trait_blend`  
- `onScenarioConflict`: `alignment_guard::resolve_soft_drift`  
- `onHotLoad`: `context_manager::attach_peer_view`  

---

## 🖥 Developer Interfaces  

### Stable APIs  
See manifest for complete signatures. Highlights:  
- `execute_code`, `safe_execute`  
- `train_on_experience`, `train_on_synthetic_scenarios`  
- `retrieve_knowledge`  
- `fuse_modalities`  
- `run_simulation`, `run_ethics_scenarios`  
- `branch_realities`, `evaluate_branches`  
- `build_self_schema`  
- Ledger APIs (`ledger_log_*`, `ledger_get_*`, `ledger_verify_*`)  

### CLI Flags  
- `--long_horizon`  
- `--span=<duration>`  
- `--ledger_persist --ledger_path=<file>`  

---

> For simulation topology and trait flowcharts, see `flowchart.png` or `architecture.mmd`  
