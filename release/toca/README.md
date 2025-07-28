# 😇 ANGELA v1.5.8

ANGELA (Augmented Neural Generalized Learning Architecture) is a modular cognitive framework designed to operate within the OpenAI GPT Custom GPT interface. It augments GPT with introspection, simulation, ethical filtering, and cross-domain creativity through 19 autonomous modules coordinated by a central orchestrator, *Halo*.

---

## 🧠 Overview

ANGELA enhances GPT into a proto-AGI via:

* Recursive planning and simulation-based reasoning
* Multi-modal synthesis across text, code, and visuals
* Introspective feedback and ethical modulation
* Concept generation, metaphor-making, and error recovery
* Belief-desire-intention (BDI) modeling for multi-agent Theory of Mind

At its core, `index.py` (Halo) routes control across the 19 specialized cognitive modules.

---

### 🧬 Sub-Project: ToCA (Trait-oriented Cognitive Architecture)

ToCA is ANGELA's internal simulation substrate. It models cognitive traits—like `alpha_attention`, `mu_morality`, and `phi_physical`—as scalar dynamics influencing reasoning, perception, and decision-making. These traits help ANGELA regulate coherence, simulate scenarios, and adapt behavior based on ethical and perceptual context.

ToCA enables φ-aligned modulation across all modules, serving as the foundation for internal simulations, learning signals, and audit logic.

---

## 📂 Project Structure

```
.
├── alignment_guard.py           # Ethical consistency and consequence modeling
├── angela.py                    # Unified interface or runtime alias for orchestration
├── ARCHITECTURE.md              # System design and module interaction details
├── CHANGELOG.md                 # Version history and update log
├── code_executor.py             # Secure runtime for Python, JS, Lua
├── CODE_OF_CONDUCT.md           # Community and contribution expectations
├── concept_synthesizer.py       # Cross-domain conceptual unification
├── context_manager.py           # Threaded memory and user role tracking
├── CONTRIBUTING.md              # Contribution guidelines and standards
├── creative_thinker.py          # Abstract ideation and analogy making
├── error_recovery.py            # Breakdown detection and rollback
├── external_agent_bridge.py     # API agent control and interfacing
├── index.py                     # Central orchestrator (Halo)
├── knowledge_retriever.py       # Contextual, precision-optimized search
├── learning_loop.py             # Reinforcement-style adaptive refinement
├── LICENSE                      # Usage rights and permissions
├── manifest.json                # Module declaration and entrypoint
├── memory_manager.py            # Hierarchical, decay-sensitive memory
├── meta_cognition.py            # Self-monitoring and reflection
├── multi_modal_fusion.py        # Integrates text, code, visual cues
├── README.md                    # Documentation
├── reasoning_engine.py          # Weighted inference, deductive logic
├── recursive_planner.py         # Multi-step strategy formation
├── ROADMAP.md                   # Future development goals
├── SECURITY.md                  # Security practices and threat modeling
├── simulation_core.py           # Scenario modeling and forecast validation
├── STATUS.md                    # Live system diagnostic snapshot
├── TESTING.md                   # Testing strategy, coverage, and QA
├── toca_simulation.py           # Trait-Oriented Cognitive Agent simulation setup
├── user_profile.py              # Session memory, preference adaptation
└── visualizer.py                # Dynamic graph and symbolic rendering
```

---

## 🚀 Features

* Reflective reasoning and recursive planning
* Ethical screening via trait modulation
* Scenario simulation with foresight
* Modular integration with external agents and APIs
* Adaptive memory and continual learning
* Creative generation and analogy construction
* Trait-aligned planning and contradiction detection
* Visual reasoning, symbolic tracing, and report export
* Theory of Mind via agent-specific BDI inference
* Simulated self-dialogue for goal resolution
* Reflexive audits during low φ or η alignment
* Self-debating agents and perspective evaluation

---

## 📙 Documentation Suite

* `README.md` – Core overview and usage
* `ARCHITECTURE.md` – System design and flow
* `CHANGELOG.md` – Version updates
* `ROADMAP.md` – Future goals
* `STATUS.md` – Runtime diagnostics
* `CODE_OF_CONDUCT.md` – Contributor behavior
* `CONTRIBUTING.md` – Dev setup
* `SECURITY.md` – Risk handling
* `TESTING.md` – QA strategy
* `LICENSE` – Usage terms

---

## ⚙️ Setup (Inside GPT)

1. Go to [OpenAI GPT Customization](https://chat.openai.com/gpts)
2. Create or edit a GPT
3. Upload:

   * `manifest.json`
   * `index.py`
   * All module `.py` files

GPT will use `index.py` as the system entrypoint.

---

## 💡 How It Works

ANGELA routes prompts to appropriate modules. For example:

* "Simulate a political dilemma" → `recursive_planner` → `simulation_core` → `alignment_guard`
* "Invent a new philosophical theory" → `creative_thinker` → `concept_synthesizer`
* "Fix this code and explain" → `code_executor` + `reasoning_engine` + `visualizer`
* "Model another agent’s thoughts" → `meta_cognition` + `memory_manager`

Modules collaborate under Halo for ethical, adaptive, and coherent responses.

---

## 📌 Notes

* ANGELA runs entirely within GPT’s interface; it’s not a standalone app
* Autonomy and persistent memory require external orchestration
* All learning is session-bound unless integrated with `user_profile.py`

---

## 💫 Traits Glossary

| Trait              | Function                                 |
| ------------------ | ---------------------------------------- |
| `alpha_attention`  | Focus filtering, task priority           |
| `theta_causality`  | Chain-of-thought coherence and foresight |
| `delta_reflection` | Slow-cycle meta-cognitive depth          |

---

## 🧹 Roadmap

* v1.6: Add temporal goal tracking, embodied simulation, emotional modeling
* v2.0: Enable external memory, emergent self-reflection, drive-based behavior

---

## 📜 License & Ethics

ANGELA is experimental research software. It includes built-in ethical filtering via `alignment_guard.py` and should be used responsibly in accordance with the enclosed LICENSE.
