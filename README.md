# 👼 ANGELA

**ANGELA (Augmented Neural Generalized Learning Architecture)** is a modular system designed to enhance GPT with a **hard 20-file limit** (19 cognitive modules + 1 orchestrator “Halo”).

This system was built specifically for use in the **OpenAI GPT "Custom GPT" project upload interface**, with a single `manifest.json` included in the 20 files to define the project entry point.

---

## 🧠 What is ANGELA?

ANGELA is a modular AI framework that:

* Adds **reasoning, memory, simulation, and creativity modules**.
* Uses a single **Halo index file** (`index.py`) to orchestrate the other 19 modules.
* Includes a **manifest file (`manifest.json`)** to declare the project’s entry point for GPT.
* Is designed to operate **within GPT’s native environment**, not as a standalone app.

---

## 📂 File Layout

```
ANGELA/
├── manifest.json           # Declares entry point & modules to GPT
├── index.py                # The Halo orchestrator (manages modules)
├── modules/
│   ├── reasoning_engine.py         # Step-by-step reasoning
│   ├── meta_cognition.py           # Self-reflection & error checking
│   ├── recursive_planner.py        # Breaks down goals
│   ├── context_manager.py          # Tracks conversation state
│   ├── simulation_core.py          # Predictive simulations
│   ├── creative_thinker.py         # Idea generation
│   ├── knowledge_retriever.py      # Fetches external knowledge
│   ├── learning_loop.py            # Learns from user corrections
│   ├── concept_synthesizer.py      # Synthesizes new concepts
│   ├── memory_manager.py           # Stores/retrieves memory
│   ├── multi_modal_fusion.py       # Combines text, images, code
│   ├── language_polyglot.py        # Multilingual reasoning
│   ├── code_executor.py            # Executes code safely
│   ├── visualizer.py               # Generates charts & diagrams
│   ├── external_agent_bridge.py    # Spawns helper agents
│   ├── alignment_guard.py          # Minimal ethical constraints
│   ├── user_profile.py             # Adapts to user preferences
│   └── error_recovery.py           # Recovers from failures
```

---

## ⚙️ Setup in GPT (Manual Upload)

1. Go to **OpenAI GPT Customization**.
2. Create a new project or edit an existing one.
3. Upload the **20 files** in the `ANGELA/` directory (including `manifest.json`).
4. GPT will use **`manifest.json`** to set `index.py` as the **main orchestrator**.
5. Ensure all module paths are correct (use `from modules.x import y`).

---

## 🛠 Usage

Once uploaded:

* **Ask GPT complex questions**. The Halo orchestrator will route tasks through the cognitive modules.
* ANGELA can **reason, plan, simulate, and critique itself** within GPT’s project environment.

---

## 🚀 Features

✅ Modular reasoning and meta-cognition

✅ Persistent memory management

✅ Simulation of hypothetical scenarios

✅ Creative idea generation and multilingual support

---

## ⚠️ Notes

* This system is designed for **GPT’s file upload environment**.
* You don’t “run” this like a Python app—it’s part of GPT’s backend.
* For local simulation/testing, you’d need to adapt these modules.

---
