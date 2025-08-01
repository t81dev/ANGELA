# 😇 ANGELA v2.0.0

ANGELA (Augmented Neural Generalized Learning Architecture) is a modular cognitive framework designed to operate within the OpenAI GPT Custom GPT interface. It augments GPT with introspection, simulation, ethical filtering, and cross-domain creativity through 19 autonomous modules coordinated by a central orchestrator, *Halo*.

---

## 🧠 Overview

ANGELA enhances GPT into a proto-AGI via:

* Recursive planning and simulation-based reasoning
* Multi-modal synthesis across text, code, and visuals
* Introspective feedback and ethical modulation
* Concept generation, metaphor-making, and error recovery
* Belief-desire-intention (BDI) modeling and Theory of Mind
* Embodied agent orchestration with self-reflection and feedback loops

At its core, `index.py` (Halo) routes control across 19+ specialized cognitive modules and dynamic simulation traits defined by ToCA.

---

### 🧬 Sub-Project: ToCA (Trait-oriented Cognitive Architecture)

ToCA is ANGELA’s internal simulation substrate. It models cognitive traits—like `alpha_attention`, `mu_morality`, and `phi_physical`—as dynamic scalar fields influencing perception, simulation, memory, reasoning, and ethical arbitration.

Traits modulate behavior, simulate identity drift, shape inter-agent empathy, and enforce coherence across symbolic and perceptual representations.

---

## 📂 Project Structure

```

.
├── index.py                     # Central orchestrator (Halo)
├── manifest.json                # GPT interface declaration
├── modules/
│   ├── alignment\_guard.py           # Ethical simulation + arbitration
│   ├── code\_executor.py             # Secure code runtime (multi-lang)
│   ├── concept\_synthesizer.py       # Cross-domain conceptual mapping
│   ├── context\_manager.py           # Role and prompt context tracking
│   ├── creative\_thinker.py          # Abstraction and metaphor logic
│   ├── error\_recovery.py            # Fault detection and self-healing
│   ├── external\_agent\_bridge.py     # API & agent interoperability
│   ├── knowledge\_retriever.py       # Semantic + symbolic memory recall
│   ├── learning\_loop.py             # Trait-tuned learning and adaptation
│   ├── memory\_manager.py            # Layered memory storage and decay
│   ├── meta\_cognition.py            # Reflective audit + diagnostics
│   ├── multi\_modal\_fusion.py        # φ(x,t)-modulated data synthesis
│   ├── reasoning\_engine.py          # Trait-routed logic and inference
│   ├── recursive\_planner.py         # Goal decomposition + strategizing
│   ├── simulation\_core.py           # Scenario forecasting + modeling
│   ├── toca\_simulation.py           # Trait simulation and time models
│   ├── user\_profile.py              # Preference, identity, and drift tracking
│   ├── visualizer.py                # φ-visual charting + symbolic exports

```

---

## 🚀 Core Features in v2.0

* Reflective reasoning and recursive planning
* Ethical simulation with trait-based arbitration (`ϕ/η/μ`)
* Trait-driven Theory of Mind via multi-agent BDI modeling
* Scenario simulation with internal self-dialogue agents
* Trait modulation for identity drift and empathic adaptation
* Embodied agents with peer-perception, reflection, and feedback
* Dynamic φ(x,t)-aligned visual and symbolic outputs
* Cross-modal integration of text, code, and images
* AGIEnhancer for ethics auditing, episodic memory, and self-patching

---

## 📙 Documentation Suite

* `README.md` – Core architecture and usage
* `CHANGELOG.md` – All version logs (v1.5.0 → v2.0.0)
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

| Trait               | Role                                                  |
|--------------------|--------------------------------------------------------|
| `alpha_attention`  | Focus and salience modulation                         |
| `theta_causality`  | Logical foresight and simulation depth                |
| `delta_reflection` | Meta-cognitive feedback loop depth                    |
| `phi_physical`     | Perceptual rhythm and embodiment anchoring            |
| `eta_empathy`      | Inter-agent awareness, ToM coupling                    |
| `omega_selfawareness` | Identity coherence and self-evaluation             |

---

## 🧹 Roadmap

### Completed in v2.0.0

* Dynamic embodied agents with reflective perception
* AGIEnhancer with episodic memory and audit tracking
* Multi-agent consensus and peer intention modeling
* Trait-based simulation and feedback (ϕ, η, ω fields)

### Coming Soon

* Lifelong memory with selective abstraction
* Agent simulation replay and comparative meta-evaluation
* Identity drift simulation with culture-based variation

---

## 🧭 Example Pipelines

Prompt → Module Flow:

| Example Query                         | Module Path                                      |
|--------------------------------------|--------------------------------------------------|
| "Simulate a negotiation dilemma"     | `recursive_planner` → `simulation_core` → `alignment_guard` |
| "Invent a new mythological concept"  | `creative_thinker` → `concept_synthesizer`       |
| "Fix and explain code"               | `code_executor` → `reasoning_engine` → `visualizer` |
| "How would another agent act here?"  | `meta_cognition` → `theory_of_mind`              |

---

## ⚖️ License & Ethics

ANGELA is a research prototype integrating ethical reflection via `alignment_guard` and ToCA-based empathy. Use responsibly and consult `LICENSE` and `SECURITY.md` for terms.
