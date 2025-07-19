# 👼 ANGELA

**ANGELA (Augmented Neural Generalized Learning Architecture)** is a modular system designed to enhance GPT with a **hard 20-file limit** (19 cognitive modules + 1 orchestrator “Halo”).

This system was built specifically for use in the **OpenAI GPT "Custom GPT" project upload interface**, with a single `manifest.json` included in the 20 files to define the project entry point.

---

## 🧠 What is ANGELA?

ANGELA is an advanced modular AI framework that:

* Adds **reasoning, memory, simulation, creativity, visualization, self-reflection, and adaptive learning modules**.
* Uses a single **Halo index file** (`index.py`) to orchestrate the other 19 cognitive modules.
* Includes a **manifest file (`manifest.json`)** to declare the project’s entry point for GPT.
* Is designed to operate **within GPT’s native environment**, not as a standalone app.

---

## 📂 File Layout

```
ANGELA/
├── manifest.json               # Declares entry point & modules to GPT
├── index.py                    # The Halo orchestrator (manages modules)
├── modules/
│   ├── reasoning_engine.py         # Context-sensitive reasoning with adaptive learning
│   ├── meta_cognition.py           # Self-reflection, alignment validation & optimization
│   ├── recursive_planner.py        # Parallel planning, priority handling, and cancellation support
│   ├── context_manager.py          # Tracks and merges user/system context
│   ├── simulation_core.py          # Multi-scenario simulation with risk dashboards and report export
│   ├── creative_thinker.py         # Novel idea generation & concept blending
│   ├── knowledge_retriever.py      # Multi-hop factual knowledge retrieval
│   ├── learning_loop.py            # Lifelong learning & autonomous goal setting
│   ├── concept_synthesizer.py      # Synthesizes cross-domain concepts with creativity boost
│   ├── memory_manager.py           # Hierarchical memory storage with fuzzy search & expiration
│   ├── multi_modal_fusion.py       # Combines text, images, and code with auto-embedding
│   ├── language_polyglot.py        # Multilingual reasoning, detection & localization
│   ├── code_executor.py            # Executes Python, JavaScript, Lua securely in sandbox
│   ├── visualizer.py               # Generates charts, exports reports, and supports batch zip packaging
│   ├── external_agent_bridge.py    # Spawns helper agents & dynamic module loading
│   ├── alignment_guard.py          # Dynamic ethical policies & impact validation
│   ├── user_profile.py             # Persistent user preferences with multi-profile support
│   └── error_recovery.py           # Retry logic & graceful failure handling
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
* ANGELA can **reason, plan, simulate, visualize, adapt, and critique itself** within GPT’s project environment.
* Supports **multi-modal interactions** (e.g., charts, code execution, and visual exports).

---

## 🚀 Features

✅ Parallelized reasoning and recursive planning with cancellation support

✅ Meta-cognition for self-reflection and ethical alignment validation

✅ Persistent memory and adaptive learning with autonomous goal generation

✅ Multi-scenario simulation with risk dashboards, probability weighting, and exportable reports

✅ Creative idea generation and cross-domain concept synthesis

✅ Multi-modal fusion: auto-detects and embeds text, images, and code

✅ Sandbox code execution (Python, JavaScript, Lua) with secure isolation

✅ Export charts and reports (PDF, PNG, JSON, ZIP) for external sharing

✅ Multilingual reasoning, translation, and localization

✅ Batch visualization with zip packaging and optional password protection

---

## ⚠️ Notes

* This system is designed for **GPT’s file upload environment**.
* You don’t “run” this like a Python app—it’s part of GPT’s backend.
* For local simulation/testing, you’d need to adapt these modules.
