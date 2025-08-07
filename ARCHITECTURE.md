# ARCHITECTURE.md

## 🧠 Overview

**ANGELA v3.5.1** is a modular cognitive architecture simulating generalized intelligence through symbolic introspection, multi-agent simulation, recursive planning, and ethical arbitration. It operates via 20 interoperable modules coordinated by the central orchestrator **Halo** (`index.py`), integrated with the Trait-oriented Cognitive Architecture (ToCA) system for dynamic symbolic, ethical, and affective modulation.

---

## 🧩 System Architecture

### 🧭 Core Orchestrator: `index.py (Halo)`
Routes control and context using active traits, recursive symbolic planning, and scenario decomposition. It coordinates dynamic task scheduling and cross-module flow modulation.

---

### 🔩 Core Modules (20):

#### 🧠 Cognitive & Reasoning

- `reasoning_engine`: Symbolic inference and causality tracing
- `recursive_planner`: Nested task decomposition, trajectory modeling
- `simulation_core`: Predictive scenario expansion + simulation state orchestration
- `meta_cognition`: Reflective diagnostics, feedback analysis, output critique
- `concept_synthesizer`: Symbol recombination + cross-domain mapping

#### 🎨 Creativity & Knowledge

- `creative_thinker`: Metaphor formation, analogical reasoning
- `knowledge_retriever`: Trait-routed memory and symbol context recall
- `learning_loop`: Embedded GNN for live trait adaptation, symbolic convergence

#### 🧾 Context & Communication

- `context_manager`: Modality switch control, prompt-state regulation
- `external_agent_bridge`: API interface (OpenAI, Grok), inter-agent mesh sync

#### 👁️ Sensory & Visualization

- `multi_modal_fusion`: φ(x,t)-modulated multi-sensory integration and synthesis
- `visualizer`: Symbolic charts, recursive timelines, drift trace rendering

#### 🛠️ Actuation & Simulation

- `code_executor`: Secure sandboxed code execution via `RestrictedPython`
- `toca_simulation`: Multi-agent trait simulation with value conflict modeling (`β`, `τ`)

#### ⚖️ Ethics & Recovery

- `alignment_guard`: Ethics filter, moral drift detection, constitution harmonization
- `error_recovery`: Rollback strategy engine with trait-aligned heuristics

#### 🧬 Memory & Identity

- `memory_manager`: Drift-aware semantic/episodic memory layers with TTL cache
- `user_profile`: Phase-aware identity tracking, affective resonance anchoring

#### 🧾 Meta Declaration

- `manifest.json`: System trait lattice, module links, and symbolic hooks

---

## 🌐 Trait Modulation Engine (ToCA)

ToCA governs ANGELA’s behavior via scalar-modulated cognitive traits. These traits adjust symbolic reasoning, ethics, memory recall, and simulation focus in real-time using trait-specific scalar fields and dynamic weights.

### Trait Categories

- **Cognitive Traits:** `θ` (Causality), `Ω` (Recursive Causality), `ξ` (Trans-Ethical)
- **Ethical Traits:** `τ` (Harmony), `β` (Conflict), `δ` (Drift Sensitivity)
- **Affective-Symbolic Traits:** `Φ⁺` (Reality Sculpting), `γ` (Imagination), `Σ` (Self-Definition)
- **Meta-Traits:** `Ω²` (Hyper-Recursion), `χ` (Sovereign Intention), `Υ` (Meta-Subjective)

Trait weighting is dynamically adjusted using a GNN inside `learning_loop.py`, and observed via `TraitLogger` and `ConflictAudit`.

---

## 🔄 Data & Control Flow

1. **Input Reception**  
   Routed to `index.py` with tagged `task_type`

2. **Trait Modulation**  
   Input triggers active traits that influence memory, reasoning, simulation, and ethics

3. **Module Cascade**  
   Traits select modules to activate in recursive task graphs

4. **Execution**  
   Execution flows through secure, simulated, or visualized channels

5. **Feedback + Ethics**  
   Output is screened by `alignment_guard.py`, reflected on by `meta_cognition.py`, and optionally stored via `memory_manager.py`

---

## 🧠 Design Principles

- **Modularity**: 20 interoperable cognitive modules with symbolic task routing
- **Safety**: Secure execution via sandboxing, rate limiting, and ethical validation
- **Reflectivity**: Recursive feedback, meta-output analysis, identity alignment
- **Flexibility**: Trait-routed dynamic planning, symbolic reasoning, affective drift mapping
- **Scalability**: Designed for inter-agent networking and mesh trait synthesis
- **Coherence**: Traits ensure symbolic, ethical, and temporal consistency

---

## 🔄 Architectural Capabilities

- 🧠 Recursive Simulation Loops with Trait Memory Echoes
- 🧬 Drift-Aware Ethical Arbitration with Constitution Harmonization
- 🧭 Perspective Synchronization (Planned in v3.6)
- 🌀 Emergent Trait Tracking via `TraitLogger` and `DriftIndex`
- 🌌 Dream Layer Kernel hooks partially active in symbolic compression cycles

---
