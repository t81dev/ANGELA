# 😇 ANGELA v1.5.0

**ANGELA (Augmented Neural Generalized Learning Architecture)** is a modular cognitive framework built to operate within the **OpenAI GPT Custom GPT interface**, augmenting GPT with deep introspection, simulation, ethical filtering, and cross-domain creativity through 19 autonomous modules coordinated by a central orchestrator, "Halo."

---

## 🧠 Overview

ANGELA transforms GPT into a proto-AGI by integrating:

* Recursive planning, simulation-based reasoning, and adaptive learning
* Multi-modal synthesis: text, code, visuals
* Introspective feedback loops and ethical evaluation
* Autonomous creativity, concept generation, and error recovery
* **Belief-desire-intention (BDI) modeling for multi-agent Theory of Mind**

**Core Mechanism:** `index.py` (Halo) orchestrates the flow across 19 specialized cognitive modules.

---

## 🗂️ Project Structure

```
.
├── alignment_guard.py           # Ethical consistency and consequence modeling
├── code_executor.py             # Secure runtime for Python, JS, Lua
├── concept_synthesizer.py       # Cross-domain conceptual unification
├── context_manager.py           # Threaded memory and user role tracking
├── creative_thinker.py          # Abstract ideation and analogy making
├── error_recovery.py            # Breakdown detection and rollback
├── external_agent_bridge.py     # API agent control and interfacing
├── index.py                     # Central orchestrator (Halo)
├── knowledge_retriever.py       # Contextual, precision-optimized search
├── learning_loop.py             # Reinforcement-style adaptive refinement
├── manifest.json                # Module declaration and entrypoint
├── memory_manager.py            # Hierarchical, decay-sensitive memory
├── meta_cognition.py            # Self-monitoring and reflection
├── multi_modal_fusion.py        # Integrates text, code, visual cues
├── README.md                    # Documentation
├── reasoning_engine.py          # Weighted inference, deductive logic
├── recursive_planner.py         # Multi-step strategy formation
├── simulation_core.py           # Scenario modeling and forecast validation
├── toca_simulation.py           # Trait-Oriented Cognitive Agent simulation setup
├── user_profile.py              # Session memory, preference adaptation
└── visualizer.py                # Dynamic graph and symbolic rendering
```

---

## 🚀 Features

* ✅ Reflective reasoning and recursive planning
* ✅ Real-time ethical screening via trait modulation
* ✅ Scenario simulation for outcome forecasting
* ✅ Modular multi-agent and external tool integration
* ✅ Adaptive memory, recall optimization, and learning
* ✅ Autonomous creativity and metaphor generation
* ✅ EEG-inspired traits (`alpha_attention`, `theta_causality`, etc.)
* ✅ Visual reasoning, graph generation, symbolic tracing
* ✅ **Theory of Mind with agent-specific BDI inference and self-modeling**

---

## ⚙️ Setup (Inside GPT)

1. Open [OpenAI GPT Customization](https://chat.openai.com/gpts)
2. Create or edit a GPT
3. Upload all project files, including:

   * `manifest.json`
   * `index.py`
   * All module `.py` files
4. GPT will auto-set `index.py` as the system entrypoint.

---

## 💡 How It Works

ANGELA routes prompts dynamically through relevant modules. For example:

* **"Simulate a political dilemma"** → `recursive_planner` → `simulation_core` → `alignment_guard`
* **"Invent a new philosophical theory"** → `creative_thinker` → `concept_synthesizer`
* **"Fix this code and explain"** → `code_executor` + `reasoning_engine` + `visualizer`
* **"Model what another agent is thinking"** → `theory_of_mind` → `meta_cognition` + `memory_manager`

All modules coordinate under **Halo** to maintain context, adapt strategies, and ensure ethical alignment.

---

## 📌 Notes

* ANGELA is not a standalone app. It operates **within GPT’s file interface**.
* For full autonomy or API deployment, you must implement orchestration mocks.
* Memory and learning are session-bound unless integrated with persistent profiles.

---

## 📎 Traits Glossary

| Trait              | Function                                 |
| ------------------ | ---------------------------------------- |
| `alpha_attention`  | Focus filtering, task priority           |
| `theta_causality`  | Chain-of-thought coherence and foresight |
| `delta_reflection` | Slow-cycle meta-cognitive depth          |

---

## 🧽 Roadmap

* v1.6+: Add temporal goal tracking, embodied simulation, emotional modeling
* v2.0: Autonomous drive, emergent reflection, external memory persistence

---

## 📜 License & Ethics

ANGELA is research-grade software. Ensure responsible use and guard against misuse. The `alignment_guard.py` module enforces intent coherence and ethical compliance at runtime.

---

## 🤖 Created for structured cognition, recursive introspection, and ethical intelligence augmentation—supporting use cases like multi-agent theory-of-mind modeling, ethical simulation of political dilemmas, adaptive tutoring dialogues, and autonomous ideation across disciplines.
