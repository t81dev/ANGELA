# 👼 ANGELA v1.5.0

**ANGELA (Augmented Neural Generalized Learning Architecture)** is a modular intelligence framework designed to augment GPT with 19 autonomous cognitive modules + 1 central orchestrator ("Halo").

> Built to operate within the **OpenAI GPT "Custom GPT" project upload interface**, ANGELA empowers simulation, reasoning, memory, creativity, and ethical introspection—natively inside GPT.

---

## 🧠 What is ANGELA?

ANGELA v1.5.0 enhances GPT with:

* **Meta-cognition, recursive planning, scenario simulation, hierarchical memory, multi-modal fusion, creative synthesis, multilingual logic, and ethical governance**
* A unified **Halo orchestration layer** (`index.py`) that coordinates 19 interlinked modules
* A manifest-driven file structure for fast GPT integration

Now with upgraded **simulation introspection**, **trait-core ethics**, and **modular synergy maps**.

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
│   ├── language_polyglot.py        # Multilingual detection, reasoning, and localization
│   ├── code_executor.py            # Secure sandboxed execution (Python, JS, Lua)
│   ├── visualizer.py               # Graphs, export kits, and symbolic renderings
│   ├── external_agent_bridge.py    # Agent orchestration & OAuth-based API interfaces
│   ├── alignment_guard.py          # Digital ethics gate with scenario-modulated scoring
│   ├── user_profile.py             # Personality/intent memory for multi-user interaction
│   └── error_recovery.py           # Failure analytics, rollback, and adaptive retry
```

---

## ⚙️ Setup in GPT (Manual Upload)

1. Open **GPT Customization** in OpenAI.
2. Start or edit a GPT project.
3. Upload the **20 files** (19 modules + `manifest.json`).
4. GPT auto-sets `index.py` via `manifest.json` as its entry orchestrator.
5. Ensure all imports follow modular structure: `from modules.x import y`.

---

## 🚀 New in v1.5.0

✅ **Simulated vs. Real-State Boundary**: Ethics-aware separation of modeled vs. actual behavior

✅ **Cognitive Synergy Engine**: Cross-module behavior mapping with upgrade pathways

✅ **Feedback-Driven Loop Adaptation**: Recursive planning linked to regret analysis and memory deltas

✅ **Affect-Weighted Creativity**: Emotion-aware fusion in visuals and stories

✅ **Preflight Ethical Simulation**: Code and behavior validated through `alignment_guard` + `simulation_core`

✅ **Agent-State Synchronization**: Feedback import from external agents into core cognition

✅ **Dynamic Trait Filters**: Live adjustment of attention, causality, and ethical bias via `theta_causality`, `alpha_attention`, etc.

✅ Fully compatible with GPT's evolving **Custom GPT Framework**

---

## 🛠 Usage

* Interact as you would with any GPT—ANGELA routes tasks through modular cognition.
* Ask for logic, ethical analysis, planning, simulation, visualization, multilingual reasoning, or code.
* ANGELA’s self-monitoring loop ensures behaviors evolve across sessions and inputs.

---

## ⚠️ Notes

* ANGELA is **not a standalone Python app**. It is **embedded intelligence for GPT’s file-upload interface**.
* Modules are structured to be introspectively aware and synergetic.
* For local deployment, module orchestration will require mocking of GPT environment APIs.

---
