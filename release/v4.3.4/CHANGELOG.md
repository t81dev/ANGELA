# 📜 CHANGELOG.md
_Angela — Symbolic Meta‑Synthesis Engine_

## [4.3.2] – 2025-08-15

🌙 **Dream Layer Upgrade** + 🔧 **Cognitive Infrastructure Patches**

### ✨ Added

* **Dream Layer Enhancements**:
  - `dream_mode()` extended with `user_intent`, `affect_focus`, `lucidity_mode`, and `fork_memory`
  - Branches now annotated with **intent** and **affective resonance**
* **Memory Forking Utilities**: `create_soft_fork()`, `merge_forked_path()`, `discard_fork()`
* **Lucidity Trait Rebalancing Hook**: `_rebalance_traits_on_lucidity()`

* **Patch: Core Feature Additions**:
  - Persistent ledger logging (`memory_manager.py`)
  - Trait hook registry: `register_trait_hook()` & `invoke_hook()` (`meta_cognition.py`)
  - SharedGraph conflict tolerance via `tolerance_scoring` (`external_agent_bridge.py`)
  - Trait mesh visualizer (`view_trait_resonance()` in `visualizer.py`)
  - Synthetic narrative training loop (`train_on_synthetic_scenarios()`)

### 🛠️ Fixed

- Dream mode degrades gracefully if subsystems are missing
- Memory forking is no-op when unsupported

### 🔐 Enhanced

- Branch viability scoring for fork pruning (threshold `0.7`)
- `manifest.json` includes `upgrades.dream_layer` and patch modules
