# Contributing to ANGELA (Augmented Neural Generalized Learning Architecture)

Thank you for your interest in contributing to ANGELA. This document outlines the process for proposing improvements, submitting code, and participating in the project community.

## 🧠 Philosophy

ANGELA is an experimental cognitive architecture and modular framework for AGI simulation. Contributions should align with its principles: modularity, introspection, ethical alignment, and clarity.

## 💡 Getting Started

1. **Read the `README.md`** to understand the core system.
2. Review `STATUS.md` and `CHANGELOG.md` to see what's in development and what's changed.
3. Familiarize yourself with the module structure in `index.py` and `manifest.json`.
4. **See `AGENTS.md`** for the registry of all sub-agents, their APIs, and traits.  
   > ⚠️ When you add or modify an agent/module, you must update **AGENTS.md** to keep documentation consistent.

## 🔧 How to Contribute

### Issues

* Search existing issues before opening a new one.
* For bugs, include reproduction steps.
* For feature requests, describe motivation and proposed design.

### Pull Requests

* Fork the repository and create a branch: `feature/your-feature-name`
* Include tests where applicable.
* Update documentation if behavior changes.  
  - This includes `README.md`, `ARCHITECTURE.md`, and **`AGENTS.md`** if agents are affected.
* Run system-wide tests before submitting.

### Code Style

* Follow Python 3.10+ syntax and type annotations.
* Keep functions pure where possible.
* Favor readability and maintainability.
* Use consistent naming conventions matching existing modules.

## 🧪 Testing

Run all tests using the orchestration shell or test harness inside the simulation kernel:

```bash
python -m tests.run_all
````

Each module should have its own test suite where feasible.

## 🧱 Modules Directory

Contributions to the following modules are most welcome:

* `reasoning_engine`
* `meta_cognition`
* `simulation_core`
* `theory_of_mind`
* `agi_enhancer`
* `alignment_guard`

## 🌐 Community Guidelines

All contributors must adhere to the `CODE_OF_CONDUCT.md`.

## 🔒 Licensing

All contributions are made under the same license as the main project (see `LICENSE`).

## 🙌 Thanks

We appreciate your ideas, code, and support in building a reflective, modular AGI simulation framework.
