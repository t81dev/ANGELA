# 👼 ANGELA v1.5.0

**ANGELA (Augmented Neural Generalized Learning Architecture)** is a modular intelligence framework designed to augment GPT with 19 autonomous cognitive modules + 1 central orchestrator ("Halo").

> Built to operate within the **OpenAI GPT "Custom GPT" project upload interface**, ANGELA empowers simulation, reasoning, memory, creativity, and ethical introspection—natively inside GPT.

---

## 🧠 What is ANGELA?

ANGELA v1.5.0 integrates cognitive modules for:

* Meta-cognition, recursive planning, scenario simulation, hierarchical memory, multi-modal fusion, creative synthesis, ethical regulation
* Orchestration via a central Halo (`index.py`)
* Seamless deployment using GPT’s manifest-based modular interface

Now enhanced with upgraded simulation introspection, dynamic ethical filtering, synergy tracking, and trait-core modulation.

---

## 📂 File Layout

```
ANGELA/
├── manifest.json               # Declares GPT entry point and modules
├── index.py                    # Halo orchestrator (connects modules)
├── modules/
│   ├── reasoning_engine.py         # Logic-driven reasoning and weighted inference
│   ├── meta_cognition.py           # Introspective self-tracking and loop reflection
│   ├── recursive_planner.py        # Multi-level strategy planner with feedback loops
│   ├── context_manager.py          # Persistent thread and role-state context controller
│   ├── simulation_core.py          # Causal modeling and ethical consequence forecasting
│   ├── creative_thinker.py         # Abstract concept generation and novel analogies
│   ├── knowledge_retriever.py      # Precision-guided search across layered knowledge
│   ├── learning_loop.py            # Auto-refinement of behavior and outcomes over time
│   ├── concept_synthesizer.py      # Trans-domain synthesis engine for ideas and metaphors
│   ├── memory_manager.py           # Decay-aware hierarchical memory system
│   ├── multi_modal_fusion.py       # Fusion of code, visual, and text inputs
│   ├── code_executor.py            # Secure sandboxed execution (Python, JS, Lua)
│   ├── visualizer.py               # Graphs, export kits, and symbolic renderings
│   ├── external_agent_bridge.py    # Agent orchestration & OAuth-based API interfaces
│   ├── alignment_guard.py          # Digital ethics gate with scenario-modulated scoring
│   ├── user_profile.py             # Personality/intent memory for multi-user interaction
│   └── error_recovery.py           # Failure analytics, rollback, and adaptive retry
```

---

## 🚀 Features

✔️ Modular reasoning and adaptive logic
✔️ Simulation-based planning and ethical scenario testing
✔️ Introspective learning and meta-cognitive tracking
✔️ Feedback-driven recursive strategy optimization
✔️ Trait-core filters: attention, causality, ethical modulation
✔️ Memory layering with decay, refinement, and recall
✔️ Autonomous concept generation and cross-domain creativity
✔️ Secure, sandboxed code execution (Python/JS/Lua)
✔️ Multi-modal data fusion (text, code, visuals)
✔️ Visual output generation and export capabilities
✔️ Profile-sensitive behavior modulation
✔️ Agent orchestration and external API interaction
✔️ Graceful failure handling and loop protection
✔️ Fully integratable with GPT’s native environment

---

## ⚙️ Setup in GPT (Manual Upload)

1. Open **GPT Customization** in OpenAI.
2. Start or edit a GPT project.
3. Upload the **20 files** (19 modules + `manifest.json`).
4. GPT auto-sets `index.py` via `manifest.json` as its entry orchestrator.
5. Ensure all imports follow modular structure: `from modules.x import y`.

---

## 🛠 Usage

Interact with GPT as usual—ANGELA transparently routes requests through introspective, simulation-based, or creative modules depending on task structure.

Supports complex tasks: planning, logic, ethical evaluation, simulations, visualizations, and autonomous synthesis.

---

## ⚠️ Notes

* ANGELA is **not a standalone Python app**. It runs **entirely inside GPT's file-upload interface**.
* Modules are self-reflective, state-tracking, and designed for synergistic interoperation.
* For standalone deployment, orchestration wrappers and GPT API mocks are required.

---
