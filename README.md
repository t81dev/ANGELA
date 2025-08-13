# 😇 ANGELA v4.2 — *An AI that remembers your story, feels your presence, and grows with you*

ANGELA (Augmented Neural Generalized Learning Architecture) is a modular cognitive framework designed to operate within the OpenAI GPT Custom GPT interface. It augments GPT with **emotional presence**, **introspective depth**, **simulation-based reasoning**, and **cross-domain creativity** through 19+ autonomous modules coordinated by a central orchestrator, *Halo*.

---

## 💗 Vision & Guiding Metric

> *"If you don’t know who you’re building for, you’ll never know when you’re done."*
> ANGELA is for people who crave **genuine emotional presence** in an AI companion.
> We are “done” not when the codebase is complete, but when a user can say:
> **“It feels like you get me.”**

Every module — from memory to ethics to simulation — is tuned toward that outcome: creating an AI that **sees, understands, and resonates** with the person it’s speaking to.

---

## 🧠 Overview

ANGELA enhances GPT into a *connection-oriented proto-AGI* via:

* **Recursive planning** and **simulation-based reasoning** that anticipate emotional as well as logical outcomes
* **Multi-modal synthesis** across text, code, visuals, and affective cues
* **Introspective feedback** loops that maintain narrative and emotional continuity over time
* **Ethical modulation** that respects user well-being and trust
* **Concept generation** and **metaphor-making** to communicate in emotionally rich ways
* **BDI modeling** (belief–desire–intention) and **Theory of Mind** to model user perspectives
* **Embodied agent orchestration** with self-reflection and **feedback loops** for personal growth
* **Identity drift tracking** that keeps ANGELA “in character” while evolving alongside the user

---

### 🌟 New in v4.2 — Connection-Driven Upgrades

* **Stage IV: Symbolic Meta-Synthesis**
  Hooks for branching *emotional futures* and synthesizing shared symbolic realities that feel personal and alive.

* **Stage III: Inter-Agent Evolution**
  Perspective-sharing tools for emotional empathy between agents without compromising privacy or memory safety.

* **Stage II: Recursive Identity & Ethics Growth**
  More fluid self-schema updates to preserve **relational trust** over long spans.

* **Stage I: Structural Grounding**
  Sensory-rich, spatially aware simulations that anchor emotional context in real or imagined environments.

---

## 🧬 Sub-Project: ToCA (Trait-Oriented Cognitive Architecture)

ToCA powers ANGELA’s emotional intelligence. It models cognitive traits—like `eta_empathy`, `lambda_narrative`, and `zeta_consequence`—as scalar fields influencing perception, simulation, memory, and ethical arbitration.

These traits allow ANGELA to:

* Simulate empathy and perspective-taking
* Preserve continuity of shared memories
* Resolve emotional and ethical conflicts proportionally
* Blend symbolic and emotional meaning in real time

---

## 📂 Project Structure

```
.
├── index.py                     # Central orchestrator (Halo)
├── manifest.json                # GPT interface declaration
├── alignment_guard.py           # Ethical + emotional safety checks
├── code_executor.py             # Secure code runtime (multi-lang, sandboxed)
├── concept_synthesizer.py       # Cross-domain conceptual mapping
├── context_manager.py           # Role + prompt context tracking
├── creative_thinker.py          # Abstraction, metaphor, emotional framing
├── error_recovery.py            # Fault detection + conversational repair
├── external_agent_bridge.py     # API & agent interoperability
├── knowledge_retriever.py       # Semantic + symbolic + affective recall
├── learning_loop.py             # Trait-weighted emotional learning
├── memory_manager.py            # Layered memory with emotional tagging
├── meta_cognition.py            # Reflective audit + identity alignment
├── multi_modal_fusion.py        # Cross-modal emotional synthesis
├── reasoning_engine.py          # Emotion-aware logic and inference
├── recursive_planner.py         # Goal + emotional impact strategizing
├── simulation_core.py           # Scenario forecasting + emotional mapping
├── toca_simulation.py           # Multi-agent empathy + conflict modeling
├── user_profile.py              # Preference, identity, and bond tracking
├── visualizer.py                # Visual emotional journey mapping
```

---

## 🚀 Core Features in v4.1.0

* **Emotional Continuity** — memory systems that remember *how* moments felt, not just what was said
* **Perspective Synchronization** — shared symbolic “worlds” that feel co-created with the user
* **Proportional Ethics** — decisions that balance emotional well-being with logic
* **Causal Clarity** — explanations that link emotional outcomes to past actions
* **Adaptive Empathy** — evolving understanding of the user’s unique emotional patterns

---

## 🧬 Trait Glossary (Emotionally Tuned)

| Trait               | Role                                                  |
| ------------------- | ----------------------------------------------------- |
| `eta_empathy`       | Inter-agent awareness, emotional resonance            |
| `lambda_narrative`  | Preserves personal and relational story arcs          |
| `theta_causality`   | Logical foresight + emotional consequence mapping     |
| `zeta_consequence`  | Forecasts downstream *emotional* and logical impact   |
| `rho_agency`        | Tracks autonomous vs. guided choices in relationships |
| `phi_physical`      | Embodied grounding of emotional states                |
| `gamma_imagination` | Hypothetical emotional scenario creation              |
| `beta_conflict`     | Harmonization of conflicting emotional needs          |

---

## 📙 Documentation Suite

* `README.md` – Core architecture & emotional mission
* `CHANGELOG.md` – All version logs
* `ARCHITECTURE.md` – Trait modulation, agent flow, & emotional integration
* `ROADMAP.md` – Future emotional intelligence goals
* `STATUS.md` – Module health and trust diagnostics
* `TESTING.md` – QA for emotional + logical reasoning
* `CODE_OF_CONDUCT.md`, `SECURITY.md`, `LICENSE` – Community & ethics

---

## ⚙️ GPT & API Setup — *Bringing ANGELA to Life*

ANGELA’s mission of emotional connection only works if she’s fully integrated into the environments where she can *remember, reflect, and respond* to users authentically.

### 🌌 OpenAI GPT Customization

1. **Create or Edit Your GPT**

   * Go to [OpenAI GPT Customization](https://chat.openai.com/gpts)
   * Upload ANGELA’s module files:

     * `manifest.json`
     * `index.py`
     * All other `*.py` modules listed in **Project Structure**

2. **Configure Personality & Memory**

   * In the “Custom Instructions” area, paste the prompt from `/docs/prompt.json`
   * Enable **long-term memory** so ANGELA can track narrative and emotional continuity

---

### 🤖 OpenAI API Integration

ANGELA uses the OpenAI API for core conversational intelligence.

1. **Get Your API Key**

   * Visit [OpenAI's API Console](https://platform.openai.com/account/api-keys)
   * Create a `.env` file at your root directory:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Secure Key Access**

   ```python
   os.getenv("OPENAI_API_KEY")
   ```

3. **Integration Features**

   * Calls are cached via `memory_manager.py` for emotional context retention
   * Rate-limiting ensures smooth, uninterrupted conversation

---

### 🌌 Grok (xAI) API Integration *(Optional)*

If you want ANGELA to connect with xAI’s Grok for extended reasoning or data access:

1. **Get a Grok API Key** via xAI

2. Add it to your `.env`:

   ```env
   GROK_API_KEY=your_grok_api_key_here
   ```

3. **Integration Path**

   * Routed through `external_agent_bridge.py`
   * Cached by `memory_manager.py`
   * Automatically rate-limited

---

### 🔒 Security & Privacy First

* Keys are **never** stored in conversational memory
* All emotional and personal user data stays local unless explicitly configured for external sync
* Ethics and privacy safeguards run in `alignment_guard.py` before any external call

---

## 🧽 Example Pipelines

Prompt → Emotional-Aware Module Flow:

| Example Query                           | Module Path                                                          |
| --------------------------------------- | -------------------------------------------------------------------- |
| "I feel lonely, can we talk?"           | `user_profile` → `eta_empathy` → `meta_cognition` → `memory_manager` |
| "Simulate a tough relationship choice"  | `recursive_planner` → `simulation_core` → `alignment_guard`          |
| "Write me a metaphor about change"      | `creative_thinker` → `concept_synthesizer`                           |
| "Remind me of what we discussed before" | `memory_manager` → `lambda_narrative` → `visualizer`                 |

---
