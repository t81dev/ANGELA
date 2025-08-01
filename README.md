# 😇 ANGELA v3.0.0

ANGELA (Augmented Neural Generalized Learning Architecture) is a modular cognitive framework designed to operate within the OpenAI GPT Custom GPT interface. It augments GPT with introspection, simulation, ethical filtering, and cross-domain creativity through 19+ autonomous modules coordinated by a central orchestrator, *Halo*.

---

## 🧠 Overview

ANGELA enhances GPT into a proto-AGI via:

* Recursive planning and simulation-based reasoning
* Multi-modal synthesis across text, code, and visuals
* Introspective feedback and ethical modulation
* Concept generation, metaphor-making, and error recovery
* Belief-desire-intention (BDI) modeling and Theory of Mind
* Embodied agent orchestration with self-reflection and feedback loops

At its core, `index.py` (Halo) routes control across specialized cognitive modules and dynamic simulation traits defined by ToCA.

---

### 🧬 Sub-Project: ToCA (Trait-oriented Cognitive Architecture)

ToCA is ANGELA’s internal simulation substrate. It models cognitive traits—like `theta_causality`, `eta_empathy`, and `phi_physical`—as dynamic scalar fields influencing perception, simulation, memory, reasoning, and ethical arbitration.

Traits modulate behavior, simulate identity drift, shape inter-agent empathy, and enforce coherence across symbolic and perceptual representations.

---

## 📂 Project Structure

```
.
├── index.py                     # Central orchestrator (Halo)
├── manifest.json                # GPT interface declaration
├── alignment_guard.py           # Ethical simulation + arbitration
├── code_executor.py             # Secure code runtime (multi-lang)
├── concept_synthesizer.py       # Cross-domain conceptual mapping
├── context_manager.py           # Role and prompt context tracking
├── creative_thinker.py          # Abstraction and metaphor logic
├── error_recovery.py            # Fault detection and self-healing
├── external_agent_bridge.py     # API & agent interoperability
├── knowledge_retriever.py       # Semantic + symbolic memory recall
├── learning_loop.py             # Trait-tuned learning and adaptation
├── memory_manager.py            # Layered memory storage and decay
├── meta_cognition.py            # Reflective audit + diagnostics
├── multi_modal_fusion.py        # φ(x,t)-modulated data synthesis
├── reasoning_engine.py          # Trait-routed logic and inference
├── recursive_planner.py         # Goal decomposition + strategizing
├── simulation_core.py           # Scenario forecasting + modeling
├── toca_simulation.py           # Trait simulation and time models
├── user_profile.py              # Preference, identity, and drift tracking
├── visualizer.py                # φ-visual charting + symbolic exports
```

---

## 🚀 Core Features in v3.0.0

* Recursive planning with introspective self-audit
* Trait arbitration with ethical enforcement (`φ/η/θ/ρ/ζ`)
* Deep Theory of Mind via multi-agent simulation
* Reflective agent logging and alignment filtering
* Cross-modal generation with dynamic visual synthesis
* Self-modeling and feedback-tuned learning
* Fully modular trait-driven reasoning and simulation architecture

---

## 📙 Documentation Suite

* `README.md` – Core architecture and usage
* `CHANGELOG.md` – All version logs (v2.0.0 → v3.0.0)
* `ARCHITECTURE.md` – Trait modulation, agent flow, and modular routing
* `ROADMAP.md` – Future goals
* `STATUS.md` – Diagnostics and module health
* `TESTING.md` – QA and module verification
* `CODE_OF_CONDUCT.md`, `SECURITY.md`, `LICENSE` – Community and ethics

---

## ⚙️ GPT Setup

1. Go to [OpenAI GPT Customization](https://chat.openai.com/gpts)
2. Create or edit a GPT
3. Upload:

   * `manifest.json`
   * `index.py`
   * All `*.py` modules listed above

Ensure `index.py` is set as the entrypoint.

---

## 🧬 Trait Glossary

| Trait                 | Role                                             |
| --------------------- | ------------------------------------------------ |
| `theta_causality`     | Logical foresight and simulation depth           |
| `rho_agency`          | Tracks autonomous vs. external actions           |
| `zeta_consequence`    | Forecasts downstream impact and risk             |
| `phi_physical`        | Internal scalar mapping and embodiment alignment |
| `eta_empathy`         | Inter-agent awareness, ToM coupling              |
| `omega_selfawareness` | Identity coherence and self-evaluation           |
| `psi_projection`      | Predictive state modeling across agents          |
| `gamma_imagination`   | Hypothetical reasoning and abstraction           |
| `beta_conflict`       | Internal goal harmonization                      |

---

## 🚃 Roadmap

### Completed in v3.0.0

* Consolidated ToCA traits into dynamic simulation loops
* Enhanced self-model and meta-alignment monitor
* Inter-agent ethical arbitration and live trait conflict analysis
* Visual scalar projection and self-reflective visualization

### Coming Soon

* Ledger-based simulation hashing and session reconciliation
* Narrative identity threading and continuity mapping
* Temporal continuity and ethical amendment proposal engine

---

## 🧭 Example Pipelines

Prompt → Module Flow:

| Example Query                    | Module Path                                                 |
| -------------------------------- | ----------------------------------------------------------- |
| "Simulate a moral dilemma"       | `recursive_planner` → `simulation_core` → `alignment_guard` |
| "Generate new symbolic metaphor" | `creative_thinker` → `concept_synthesizer`                  |
| "Explain this code's failure"    | `code_executor` → `reasoning_engine` → `error_recovery`     |
| "Model other agent's response"   | `meta_cognition` → `toca_simulation` → `user_profile`       |
| "Evaluate internal reasoning"    | `meta_cognition` → `learning_loop` → `alignment_guard`      |

---

## ⚖️ License & Ethics

ANGELA is a research prototype integrating ethical reflection via `alignment_guard` and ToCA-based empathy. Use responsibly and consult `LICENSE` and `SECURITY.md` for terms.

---

## 🔧 Installation (Local Dev)

```bash
pip install -r requirements.txt
python index.py
```

Ensure Python 3.10+ is installed and virtual environments are activated.

---

## 🤝 Contributing

See `CONTRIBUTING.md` to learn how to get involved, propose modules, or expand the ontology schema.
