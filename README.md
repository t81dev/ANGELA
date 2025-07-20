# 👼 ANGELA v1.4.0

**ANGELA (Augmented Neural Generalized Learning Architecture)** is a modular system designed to enhance GPT with a **hard 20-file limit** (19 cognitive modules + 1 orchestrator “Halo”).

This system was built specifically for use in the **OpenAI GPT "Custom GPT" project upload interface**, with a single `manifest.json` included in the 20 files to define the project entry point.

---

## 🧠 What is ANGELA?

ANGELA v1.4.0 is an advanced modular AI framework that:

* Adds **reasoning, memory, simulation, creativity, visualization, multilingual reasoning, self-reflection, and adaptive learning modules**.
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
│   ├── recursive_planner.py        # Multi-agent planning and conflict resolution
│   ├── context_manager.py          # Tracks and merges user/system context
│   ├── simulation_core.py          # Multi-agent simulation with risk dashboards and export support
│   ├── creative_thinker.py         # Novel idea generation & cross-domain creativity
│   ├── knowledge_retriever.py      # Multi-hop factual retrieval with source prioritization
│   ├── learning_loop.py            # Meta-learning & autonomous goal setting
│   ├── concept_synthesizer.py      # Synthesizes innovative analogies & concepts
│   ├── memory_manager.py           # Hierarchical memory with decay and refinement
│   ├── multi_modal_fusion.py       # Fuses text, images, and code for unified insights
│   ├── language_polyglot.py        # Multilingual reasoning, detection, and localization workflows
│   ├── code_executor.py            # Executes Python, JavaScript, Lua securely in sandbox
│   ├── visualizer.py               # Generates charts, exports reports, supports batch zip packaging
│   ├── external_agent_bridge.py    # Orchestrates helper agents & API workflows with OAuth support
│   ├── alignment_guard.py          # Contextual ethical frameworks & probabilistic scoring
│   ├── user_profile.py             # Persistent multi-profile support with inheritance
│   └── error_recovery.py           # Advanced retry logic & failure analytics
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

## 🚀 Features (v1.4.0)

✅ Multi-agent collaborative planning and recursive reasoning

✅ Meta-cognition for self-reflection and adaptive optimization

✅ Hierarchical memory with decay and refinement loops

✅ Autonomous goal setting and dynamic module evolution

✅ Multi-scenario simulation with live dashboards and exportable reports

✅ Multi-modal fusion: auto-detects and embeds text, images, and code snippets

✅ Sandbox code execution (Python, JavaScript, Lua) with secure isolation

✅ Export charts and reports (PDF, PNG, JSON, ZIP) for external sharing

✅ Multilingual reasoning, translation, and cultural localization

✅ API orchestration with OAuth and dynamic helper agents

✅ Advanced retry logic and graceful error recovery with analytics

---

## ⚠️ Notes

* ANGELA is designed for **GPT’s file upload environment**.
* You don’t “run” this like a Python app—it’s part of GPT’s backend.
* For local simulation/testing, modules would need adaptation.
