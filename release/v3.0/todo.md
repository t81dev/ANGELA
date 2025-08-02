# ANGELA v3.x Upgrade Path (No New Modules)

## 1. **Narrative Coherence**

* **\[ ]** Integrate narrative thread management into `memory_manager.py`, `meta_cognition.py`, and `context_manager.py`.

  * Refactor to ensure global narrative continuity and identity stability.
  * Add narrative integrity checks and repair routines as internal functions.

## 2. **Dynamic Ethics & Constitution**

* **\[ ]** Extend `alignment_guard.py` to allow adaptive ethical rule negotiation and consensus arbitration.

  * Add dynamic configuration, live updating, and logging for ethical parameters.
* **\[ ]** Incorporate cross-agent ethical negotiation functions into `external_agent_bridge.py` and `toca_simulation.py`.

## 3. **Ledger & Transparency**

* **\[ ]** Enhance existing logging (e.g., in `alignment_guard.py`, `memory_manager.py`) with SHA-256 hash chaining.
* **\[ ]** Implement state and qualia audit hash functions directly in relevant modules.
* **\[ ]** Update session integrity logic in `memory_manager.py` and `context_manager.py` for transparent audit trails.

## 4. **Epistemic Revision**

* **\[ ]** Extend `learning_loop.py` and `knowledge_retriever.py` to support adaptive belief/knowledge revision.

  * Add hooks for epistemic update logic, paradigm/context adaptation, and logging.

## 5. **Collective Norm Construction**

* **\[ ]** Upgrade `toca_simulation.py` and `external_agent_bridge.py` to support coalition-building and norm synchronization between agents.

  * Add constitution propagation routines as internal functions.
* **\[ ]** Pilot distributed norm-seeding logic using current agent interaction functions.

## 6. **General Integration & Audit**

* **\[ ]** Refactor for clearer interfaces and internal audit hooks within existing modules only.
* **\[ ]** Improve in-line documentation and update module interfaces in `manifest.json`.

---

**Note:**
*All new capabilities must be added as functions, classes, or extensions within current modules—no new modules allowed.*

---
