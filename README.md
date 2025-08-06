---

# 😇 ANGELA v3.3.6

ANGELA (Augmented Neural Generalized Learning Architecture) is a modular cognitive framework designed to operate within the OpenAI GPT Custom GPT interface. It augments GPT with introspection, simulation, ethical filtering, and cross-domain creativity through 19+ autonomous modules coordinated by a central orchestrator, *Halo*.

---

## 🧠 Overview

ANGELA enhances GPT into a proto-AGI via:

* Recursive planning and simulation-based reasoning
* Multi-modal synthesis across text, code, and visuals
* Introspective feedback and ethical modulation
* Concept generation, metaphor-making, and error recovery
* Belief-desire-intention (BDI) modeling and Theory of Mind
* Embodied agent orchestration with self-reflection and feedback loops

At its core, `index.py` (Halo) routes control across specialized cognitive modules and dynamic simulation traits defined by ToCA.

---

### 🧬 Sub-Project: ToCA (Trait-oriented Cognitive Architecture)

ToCA is ANGELA’s internal simulation substrate. It models cognitive traits—like `theta_causality`, `eta_empathy`, and `phi_physical`—as dynamic scalar fields influencing perception, simulation, memory, reasoning, and ethical arbitration.

Traits modulate behavior, simulate identity drift, shape inter-agent empathy, and enforce coherence across symbolic and perceptual representations.

---

## 📂 Project Structure

```
.
├── index.py                     # Central orchestrator (Halo)
├── manifest.json                # GPT interface declaration
├── alignment_guard.py           # Ethical simulation + arbitration
├── code_executor.py             # Secure code runtime (multi-lang, sandboxed)
├── concept_synthesizer.py       # Cross-domain conceptual mapping
├── context_manager.py           # Role and prompt context tracking
├── creative_thinker.py          # Abstraction and metaphor logic
├── error_recovery.py            # Fault detection and self-healing
├── external_agent_bridge.py     # API & agent interoperability
├── knowledge_retriever.py       # Semantic + symbolic memory recall
├── learning_loop.py             # Trait-weighted learning via GNN
├── memory_manager.py            # Layered memory + API cache with TTL
├── meta_cognition.py            # Reflective audit + diagnostics
├── multi_modal_fusion.py        # φ(x,t)-modulated data synthesis
├── reasoning_engine.py          # Trait-routed logic and inference
├── recursive_planner.py         # Goal decomposition + strategizing
├── simulation_core.py           # Scenario forecasting + modeling
├── toca_simulation.py           # Multi-agent trait simulation + conflict modeling
├── user_profile.py              # Preference, identity, and drift tracking
├── visualizer.py                # φ-visual charting + symbolic exports
```

---

## 🚀 Core Features in v3.3.5

* Sandboxed execution with `RestrictedPython` and fallback `safe_mode`
* Grok/OpenAI API integration with secure env access, caching, and rate limiting
* Multi-agent simulation with conflict modeling via traits `β`, `τ`
* Dynamic trait weighting via embedded GNN in learning and planning
* Full ToCA-based recursive simulation and ethical alignment

---

## 🧬 Trait Glossary

| Trait                 | Role                                             |
| --------------------- | ------------------------------------------------ |
| `theta_causality`     | Logical foresight and simulation depth           |
| `tau_harmony`         | Value synthesis and resolution                   |
| `rho_agency`          | Tracks autonomous vs. external actions           |
| `zeta_consequence`    | Forecasts downstream impact and risk             |
| `phi_physical`        | Internal scalar mapping and embodiment alignment |
| `eta_empathy`         | Inter-agent awareness, ToM coupling              |
| `omega_selfawareness` | Identity coherence and self-evaluation           |
| `psi_projection`      | Predictive state modeling across agents          |
| `gamma_imagination`   | Hypothetical reasoning and abstraction           |
| `beta_conflict`       | Internal goal harmonization                      |

---

## 📙 Documentation Suite

* `README.md` – Core architecture and usage
* `CHANGELOG.md` – All version logs
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
   * All other `*.py` modules listed above
    
4. Edit Custom Prompt Instructions

   * Choose `/docs/prompt.json'
   * Copy and paste into custom prompt instruction area

---

## ⚙️ API Setup


### 🌌 Grok (xAI) API Integration

1. Obtain a valid **Grok API key** via xAI
2. Create a `.env` file at your root directory:

   ```env
   GROK_API_KEY=your_grok_api_key_here
   ```
3. The key is securely loaded via:

   ```python
   os.getenv("GROK_API_KEY")
   ```
4. API usage is:

   * Routed through `external_agent_bridge.py`
   * Cached via `memory_manager.py` with expiration TTL
   * Rate-limited automatically

---

### 🤖 OpenAI API Integration

1. Get an API key from [OpenAI's API Console](https://platform.openai.com/account/api-keys)
2. In the same `.env` file, add:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. The key is securely accessed using:

   ```python
   os.getenv("OPENAI_API_KEY")
   ```
4. Features:

   * Secure call handling
   * Response caching + expiration via `memory_manager.py`
   * Rate limiting for all OpenAI calls (e.g., GPT-4)

---

## 🧭 Example Pipelines

Prompt → Module Flow:

| Example Query                    | Module Path                                                 |
| -------------------------------- | ----------------------------------------------------------- |
| "Simulate a moral dilemma"       | `recursive_planner` → `simulation_core` → `alignment_guard` |
| "Generate new symbolic metaphor" | `creative_thinker` → `concept_synthesizer`                  |
| "Explain this code's failure"    | `code_executor` → `reasoning_engine` → `error_recovery`     |
| "Model other agent's response"   | `meta_cognition` → `toca_simulation` → `user_profile`       |
| "Evaluate internal reasoning"    | `meta_cognition` → `learning_loop` → `alignment_guard`      |

---
