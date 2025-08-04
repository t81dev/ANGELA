# ARCHITECTURE.md

## 🧠 Overview

**ANGELA v3.3.5** is a modular cognitive framework simulating generalized intelligence through orchestrated autonomous modules, all coordinated by a central orchestrator called **Halo** (`index.py`). It integrates symbolic introspection, recursive planning, ethical modulation, dynamic trait-weighting, and embodied simulation.

---

## 🧩 System Architecture

### 🧭 Core Orchestrator: `index.py (Halo)`

Routes control and context across all modules using trait modulation and symbolic flow. Operates as the central scheduler and task dispatcher.

---

### 🔩 Core Modules (19):

#### 🧠 Cognitive & Reasoning

* `reasoning_engine`: Symbolic and trait-weighted inference
* `recursive_planner`: Time-based goal decomposition, nested tasks
* `simulation_core`: Scenario modeling + predictive pathways
* `meta_cognition`: Reflective diagnostics, state introspection
* `concept_synthesizer`: Symbol fusion and concept creation

#### 🎨 Creativity & Knowledge

* `creative_thinker`: Metaphor generation, abstract problem solving
* `knowledge_retriever`: Trait-routed memory and info access
* `learning_loop`: Trait influence via embedded GNN; live adaptation

#### 🧾 Context & Communication

* `context_manager`: Modality + session state regulation
* `external_agent_bridge`: API interface (OpenAI, Grok), agent sync

#### 👁️ Sensory & Visualization

* `multi_modal_fusion`: φ(x,t)-modulated synthesis of data types
* `visualizer`: Symbolic + perceptual diagramming

#### 🛠️ Actuation & Simulation

* `code_executor`: Secure sandboxed execution (RestrictedPython); `safe_mode=True`
* `toca_simulation`: Multi-agent simulation with inter-agent conflict modeling (traits `β`, `τ`)

#### ⚖️ Ethics & Recovery

* `alignment_guard`: Ethical modulation, trait arbitration
* `error_recovery`: Rollbacks + correction logic

#### 🧬 Memory & Identity

* `memory_manager`: Semantic/episodic storage with cache TTL
* `user_profile`: Models personal history, drift, affective preferences

---

## 🌐 Trait Modulation Engine (ToCA)

Traits operate as scalar field modulators influencing attention, simulation, reasoning, and adaptation. Dynamic trait weighting is powered by an internal GNN via `learning_loop`.

Example Traits:

* `phi_physical`: Sensorimotor grounding
* `beta_conflict`: Conflict arbitration in planning
* `tau_harmony`: Value alignment and synthesis
* `eta_empathy`: ToM-aware simulation

---

## 🔄 Data & Control Flow

1. **Input** → Routed to `index.py` (Halo)
2. **Trait Activation** → Traits dynamically modulate flow based on input type
3. **Module Selection** → Modules are recursively activated
4. **Execution** → Computation, reasoning, simulation
5. **Ethics + Feedback** → Validated via `alignment_guard`, stored in `memory_manager`

---

## 🧠 Design Principles

* **Modularity**: 19 autonomous, interoperable modules
* **Reflectivity**: Internal self-evaluation via recursive introspection
* **Safety**: Built-in ethical guards and secure code execution
* **Extensibility**: GNN-driven traits and simulation expansion
* **Embodiment**: Agent modeling via sensorimotor + identity loops

---
