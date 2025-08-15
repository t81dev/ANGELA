## \[4.3.2] – 2025-08-15

🌙 **Dream Layer Upgrade** — *Lucidity Controls & Soft-Gated Memory Forking*

### ✨ Added

* **Dream Layer Enhancements**:

  * New parameters for `dream_mode` in `concept_synthesizer.py`:

    * `user_intent`
    * `affect_focus`
    * `lucidity_mode` (`passive`, `influential`, `co-creator`, `autonomous`)
    * `fork_memory` (soft-gated episodic branching)
  * Dream branches now annotated with **intent** and **affective resonance tags** in `visualizer.py`.
* **Memory Forking Utilities** (`memory_manager.py`):

  * `create_soft_fork()`, `merge_forked_path()`, `discard_fork()` — with ledger logging when enabled.
* **Lucidity Trait Rebalancing Hook** in `meta_cognition.py`:

  * `_rebalance_traits_on_lucidity()` adjusts internal lattice amplitudes based on lucidity mode.

### 🛠️ Fixed

* **Backward Compatibility**:

  * Dream mode gracefully degrades when certain subsystems are unavailable (no hard failures).
  * Memory fork functions act as no-ops if unsupported.

### 🔐 Enhanced

* **Dream Branch Viability Filter**:

  * Auto-merge or discard forks based on branch viability score (default threshold: `0.7`).
* **Manifest Update**:

  * `manifest.json` now reflects `upgrades.dream_layer` entry with version bump.
