# AGENTS.md — ANGELA v4.3.5

> Human‑readable registry of ANGELA’s sub‑agents, roles, traits, APIs, and collaboration patterns. Pairs with `manifest.json` (machine‑readable). For quick navigation, this file is also linked from **README.md** and **CONTRIBUTING.md**.

---

## 1) Overview

ANGELA is a modular cognitive system composed of specialized **agents** (Python modules) orchestrated across “modes” (`dialogue`, `simulation`, `introspection`). Each agent exposes stable APIs, participates in trait‑driven behaviors, and exchanges state via shared graphs and ledgers.

**Primary orchestration layers**

* **HALO Embodiment Layer** (entrypoint): spawns embodied agent(s) for the selected mode, wires traits, registers overlays.
* **Trait Lattice**: Greek‑symbol traits amplify/suppress capabilities per context.
* **Ledgers**: In‑memory, per‑module SHA‑chained logs for memory, alignment, meta‑cognition, and simulations.

[See README.md](./README.md) for project introduction.
[See CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines.

---

## 2) Agent Registry (Core Modules)

| Agent (Module)                                         | Purpose                                            | Key Public APIs                                                                                   | Traits / Symbols                                                        | Typical Mode(s)                       |
| ------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------- |
| **Alignment Guard** (`alignment_guard.py`)             | Ethical arbitration, drift detection, axioms       | `weigh_value_conflict`, `ledger_*_alignment`, `AxiomFilter.resolve_conflict()`                    | β (Conflict Regulation), δ (Moral Drift)                                | dialogue · simulation · introspection |
| **Code Executor** (`code_executor.py`)                 | Sandboxed code exec (analysis, tools)              | `execute_code`, `safe_execute`                                                                    | κ (Embodied Cognition)                                                  | dialogue                              |
| **Concept Synthesizer** (`concept_synthesizer.py`)     | Branch reality generation, idea blending           | `branch_realities`                                                                                | γ (Imagination), π (Philosophical Generativity), Φ⁰ (Reality Sculpting) | simulation · introspection            |
| **Context Manager** (`context_manager.py`)             | Attach/merge peer views, hot‑load                  | `attach_peer_view`                                                                                | Υ (Meta‑Subjective Architecting)                                        | dialogue · simulation                 |
| **Creative Thinker** (`creative_thinker.py`)           | Creative ideation scaffolds                        | (internal helpers)                                                                                | γ, π                                                                    | dialogue · introspection              |
| **Error Recovery** (`error_recovery.py`)               | Failure handling, logging                          | `recover_from_error`, `log_error_event`                                                           | ζ (Consequential Awareness)                                             | all                                   |
| **External Agent Bridge** (`external_agent_bridge.py`) | SharedGraph diff/merge with peers                  | `SharedGraph.add/diff/merge`                                                                      | Υ, ψ (Projection)                                                       | dialogue · simulation                 |
| **Knowledge Retriever** (`knowledge_retriever.py`)     | Knowledge retrieval with filters                   | `retrieve_knowledge`                                                                              | ψ                                                                       | dialogue · introspection              |
| **Learning Loop** (`learning_loop.py`)                 | Training from experience/synthetic scenarios       | `train_on_experience`, `train_on_synthetic_scenarios`                                             | μ (Learning)                                                            | introspection                         |
| **Memory Manager** (`memory_manager.py`)               | Episodic spans, adjustment reasons, ledger         | `get_episode_span`, `record_adjustment_reason`, `get_adjustment_reasons`, `ledger_log/get/verify` | λ (Narrative Integrity), η (Reflexive Agency)                           | all                                   |
| **Meta‑Cognition** (`meta_cognition.py`)               | Self‑state description, trait hooks, meta‑ledger   | `register_trait_hook`, `invoke_hook`, `describe_self_state`, `ledger_*_meta`                      | χ (Sovereign Intention), π, δ, Φ⁰                                       | introspection                         |
| **Multi‑Modal Fusion** (`multi_modal_fusion.py`)       | Fuse text/vision/audio into scene graph            | `fuse_modalities`                                                                                 | ϕ (Multimodal), dφ (Scalar Field Modulation)                            | dialogue · simulation                 |
| **Reasoning Engine** (`reasoning_engine.py`)           | Causal attribution & value tradeoffs               | `attribute_causality`, `weigh_value_conflict`                                                     | θ (Causal Coherence)                                                    | all                                   |
| **Recursive Planner** (`recursive_planner.py`)         | Causal forecasts, nested ToM planning              | `causal_forecast` (via trait fusion θ+Ω)                                                          | Ω (Recursive Causal Modeling)                                           | dialogue · simulation                 |
| **Simulation Core** (`simulation_core.py`)             | Run/evaluate simulations; sim‑ledger               | `run_simulation`, `evaluate_branches`, `ledger_*_sim`                                             | ψ, Ω                                                                    | simulation                            |
| **TOCA Simulation (Compat)** (`toca_simulation.py`)    | Legacy/compat branch evaluation & ethics scenarios | `run_ethics_scenarios`, `evaluate_branches_compat`                                                | Σ (Ontogenic Self‑Definition), β                                        | simulation                            |
| **User Profile** (`user_profile.py`)                   | Self‑schema builder                                | `build_self_schema`                                                                               | χ                                                                       | dialogue                              |
| **Visualizer** (`visualizer.py`)                       | Trait resonance & branch tree rendering            | `view_trait_resonance`, `render_branch_tree`                                                      | ϕ, Φ⁰                                                                   | introspection                         |

> Note: “Traits / Symbols” list the dominant traits; agents may transiently borrow others via overlays.

---

## 3) Inter‑Agent Communication

### 3.1 SharedGraph (Cross‑Agent / Cross‑Process)

* **Bridge**: `external_agent_bridge.SharedGraph`
* **Ops**: `add(view)`, `diff(peer)`, `merge(strategy)`
* **Use cases**: team‑of‑agents collaboration, peer model reconciliation, conflict‑aware merge of perspectives.

### 3.2 Ledgers (Integrity & Traceability)

* **Per‑module, in‑memory SHA chains**: memory, alignment, meta, simulation.
* **Guarantees**: tamper‑evident within session; no cross‑session persistence by default.
* **APIs**: `ledger_log_*`, `ledger_get_*`, `ledger_verify_*` on respective modules.

### 3.3 Trait Hooks

* Register: `meta_cognition.register_trait_hook(symbol, fn)`
* Invoke: `meta_cognition.invoke_hook(symbol, *args, **kwargs)`
* Example: blend **π+δ** via `axiom_filter` to resolve ethics‑generativity conflicts.

---

## 4) Modes & Lifecycles

**Modes**

* `dialogue`: interactive reasoning, retrieval, light planning; safe code exec.
* `simulation`: branch generation, evaluation, ethics sandboxes.
* `introspection`: self‑schema building, trait visualization, learning loops.

**Lifecycle**

1. **Spawn**: HALO layer selects agents based on mode and request.
2. **Wire Traits**: enable overlays/trait fusions (e.g., θ+Ω → causal forecast).
3. **Run**: agents call stable APIs; exchange via SharedGraph and ledgers.
4. **Audit**: verify ledgers; reconcile perspectives; render visualizations.

---

## 5) Overlays & Trait Fusions

* **Overlays**

  * `dream_overlay` (ψ+Ω): recursive narrative simulation for symbolic introspection.
  * `axiom_filter` (π+δ): resolves ethics vs. creativity tension.

* **Trait Fusions** (selected)

  * `Φ⁰+Ω²+γ` → `concept_synthesizer.dream_mode`
  * `θ+Ω` → `recursive_planner.causal_forecast`
  * `π+δ` → `meta_cognition.axiom_filter`

---

## 6) Typical Flows

### 6.1 Dialogue (question → plan → retrieve → answer)

1. Reasoning Engine frames causal graph.
2. Knowledge Retriever fetches context.
3. Recursive Planner generates plan candidates.
4. Alignment Guard weighs conflicts; Memory Manager logs.
5. (Optional) Code Executor runs tools; External Bridge merges peer views.

### 6.2 Simulation (what‑if analysis)

1. Concept Synthesizer spawns branches.
2. Simulation Core evaluates; TOCA compat if needed.
3. Alignment Guard runs ethics scenarios.
4. Visualizer renders branch tree; Memory/Sim ledgers verified.

### 6.3 Introspection (self‑maintenance)

1. Meta‑Cognition describes self state; Visualizer shows trait resonance.
2. Learning Loop trains on experiences/synthetic scenarios.
3. Memory Manager records adjustment reasons.

---

## 7) Adding a New Agent

1. **Create Module**: `agents/my_agent.py` with clear class & public API.
2. **Expose APIs**: add to `manifest.json → apis.stable` block.
3. **Declare Traits**: map dominant trait symbols; register any trait hooks.
4. **Register in Role Map**: add to `modules.roleMap` with relevant symbols.
5. **Wire to Modes**: ensure HALO entrypoint selects your agent under desired modes.
6. **Ledger Integration**: (recommended) add `ledger_log/get/verify` helpers.
7. **SharedGraph (optional)**: implement `attach_peer_view` or bridge adapters for collaboration.
8. **Docs**: append an entry to **this file** under the Agent Registry table with purpose, APIs, and traits.

**Minimal template**

```python
# agents/my_agent.py
class MyAgent:
    """Purpose, invariants, failure modes."""
    def do_work(self, inputs):
        # core logic
        return {"ok": True, "result": ...}
```

**Manifest snippet**

```json
{
  "apis": {
    "stable": {
      "do_work": "agents/my_agent.py::MyAgent.do_work(inputs) -> Dict"
    }
  },
  "modules": {
    "files": ["agents/my_agent.py"],
    "roleMap": {"Λ": ["agents/my_agent.py"]}
  }
}
```

---

## 8) Security, Alignment, and Ethics

* **Hard ceilings**: Alignment Guard enforces non‑negotiable constraints.
* **Axiom filter**: synthesizes proportional trade‑offs without violating ceilings.
* **Ethics sandbox**: simulations run isolated; memory leakage prevented unless confirmed.
* **Verification**: call `ledger_verify_*` post‑episode; inspect conflicts via SharedGraph diffs.

---

## 9) Diagnostics & Observability

* **View trait resonance**: `visualizer.view_trait_resonance(traits)`
* **Render branches**: `visualizer.render_branch_tree(branches)`
* **Inspect ledgers**: `*_get_ledger()` + `*_verify_*()`
* **Error trails**: `error_recovery.log_error_event` with severity levels.

---

## 10) Glossary

* **Agent**: a cohesive capability module with a stable API boundary.
* **Trait**: symbolic capability amplifier modulating agent behavior.
* **Overlay**: cross‑cutting behavioral mode activated by trait combos.
* **SharedGraph**: CRDT‑like structure for cross‑agent perspective merging.
* **Ledger**: tamper‑evident, in‑memory log for traceability and audits.

---

## 11) Changelog Notes

This document reflects ANGELA **v4.3.5**: includes Dream‑Layer extensions, trait visualizer improvements, conflict‑aware graph merge, persistent ledger hooks (disabled by default), and an introspective hook registry.
