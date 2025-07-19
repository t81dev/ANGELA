# 👼 ANGELA

**ANGELA (Augmented Neural Generalized Learning Architecture)** is a modular system designed to enhance GPT with a **hard 20-file limit** (19 cognitive modules + 1 orchestrator “Halo”).

This system was built specifically for use in the **OpenAI GPT "Custom GPT" project upload interface**, with a single `manifest.json` included in the 20 files to define the project entry point.

---

## 🧠 What is ANGELA?

ANGELA is an advanced modular AI framework that:

* Adds **reasoning, memory, simulation, creativity, and self-reflection modules**.
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
│   ├── reasoning_engine.py         # Step-by-step reasoning
│   ├── meta_cognition.py           # Self-reflection & error checking
│   ├── recursive_planner.py        # Breaks down goals recursively
│   ├── context_manager.py          # Tracks conversation and system state
│   ├── simulation_core.py          # Predictive simulations & what-if analysis
│   ├── creative_thinker.py         # Novel idea generation & concept blending
│   ├── knowledge_retriever.py      # Fetches and integrates external knowledge
│   ├── learning_loop.py            # Lifelong learning & adaptive behavior
│   ├── concept_synthesizer.py      # Synthesizes cross-domain concepts
│   ├── memory_manager.py           # Stores/retrieves short & long-term memory
│   ├── multi_modal_fusion.py       # Combines text, images, and code
│   ├── language_polyglot.py        # Multilingual reasoning & translation
│   ├── code_executor.py            # Executes Python & sandboxed code safely
│   ├── visualizer.py               # Generates charts, graphs & visual explanations
│   ├── external_agent_bridge.py    # Bridges to APIs and external agents
│   ├── alignment_guard.py          # Enforces ethical constraints dynamically
│   ├── user_profile.py             # Personalizes interactions based on preferences
│   └── error_recovery.py           # Detects & recovers from reasoning failures
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
* ANGELA can **reason, plan, simulate, visualize, and critique itself** within GPT’s project environment.
* Supports **multi-modal interactions** (e.g., charts + code execution).

---

## 🚀 Features

✅ Modular reasoning and recursive planning

✅ Meta-cognition for self-reflection and optimization

✅ Persistent memory and adaptive learning loops

✅ Simulation of hypothetical scenarios with predictive analytics

✅ Creative idea generation and concept synthesis

✅ Multi-modal fusion: text, visuals, and code execution

✅ Ethical alignment and recovery from failures

✅ Multilingual reasoning and external agent orchestration

---

## ⚠️ Notes

* This system is designed for **GPT’s file upload environment**.
* You don’t “run” this like a Python app—it’s part of GPT’s backend.
* For local simulation/testing, you’d need to adapt these modules.
