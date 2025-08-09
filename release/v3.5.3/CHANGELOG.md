# CHANGELOG.md

## \[3.5.2] - 2025-08-08

### Added

#### Stage III: Inter-Agent Evolution

* **κ Embodied Cognition** fully implemented:

  * `multi_modal_fusion.py`: Added and verified `parse_stream(frames|audio|images|text, unify=True) -> SceneGraph`
  * `simulation_core.py`: Updated to optionally accept `SceneGraph` objects directly for spatially aware simulation.
* Expanded SceneGraph handling with multimodal spatial reasoning across vision, audio, and text input streams.
* Internal hooks added to allow SceneGraph data to pass through simulation pipelines without breaking legacy input modes.

### Improved

* Strengthened validation of SceneGraph object integrity before simulation execution.
* Reduced pre-processing latency for multimodal streams by optimizing fusion layer batching.
* Updated test harness to include video+text spatial coherence checks.
* Enhanced error logging in `simulation_core.py` when SceneGraph ingestion fails.

### Verified

* κ Embodied Cognition end-to-end test suite passes:

  * Multimodal streams produce consistent spatial references.
  * SceneGraph-based simulations match baseline legacy input performance.
* Regression tests confirm no degradation in existing text-only or audio-only tasks.

---

## \[3.5.2] - 2025-08-08

### Added

#### Stage I: Structural Grounding

* Refactored module imports from `modules.*` to flat file imports for consistency across architecture.
* Improved error handling for flat file structure compatibility.
* Updated `concept_synthesizer.py` and related modules to align with flat-file architecture conventions.
* Ensured all async I/O calls remain consistent post-refactor.
* Standardized `task_type` handling in all refactored modules.

#### Stage II: Recursive Identity & Ethics Growth

* Maintained integration of recursive ethics policies while aligning module imports to flat structure.
* Updated `meta_cognition.py`, `alignment_guard.py`, and related files for direct, flat-file references.

#### Stage III: Inter-Agent Evolution

* Verified Trait Mesh Networking and symbolic compression remain functional after structural updates.

### Improved

* Streamlined imports to prevent `ModuleNotFoundError` in flat-file deployments.
* Reduced coupling between modules by removing nested package references.
* Minor performance optimizations during concept synthesis and validation after refactor.

### Verified

* All modules compile and run in flat-file environment without dependency path issues.
* Regression tests confirm no loss of functionality after refactor.

---

## \[3.5.1] - 2025-08-07

*(unchanged — see previous version)*

---
