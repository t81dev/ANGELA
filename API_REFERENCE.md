# 🔌 API\_REFERENCE.md

## Overview

Public APIs exposed by **ANGELA v4.3.5** for GPT Custom GPT and local runtime. All calls flow through the **Halo** orchestrator and are guarded by: sandboxed execution, trait arbitration, ethics containment, and SHA‑256 ledgers.

> **Note on source links & line numbers**
> This reference includes **source file paths**. If you want live **line-number permalinks** (e.g., GitHub `#L123` anchors), run the helper script in **Appendix A** to auto-insert exact lines per function after pushing the repo.

---

## 1) Core Execution

### `execute_code(code: str, context=None) -> Any`

* **Module**: `code_executor.py`
* **Purpose**: Run code in a restricted sandbox.
* **Security**: Blocks unsafe builtins, file I/O, network.

### `safe_execute(code: str, sandbox: bool = True) -> ExecutionResult`

* **Module**: `code_executor.py`
* **Purpose**: Strict mode for untrusted code.

---

## 2) Introspection & Reflection

### `register_trait_hook(trait_symbol: str, fn)`

* **Module**: `meta_cognition.py`
* **Purpose**: Register introspective hook per trait.

### `invoke_trait_hook(trait_symbol: str, *args, **kwargs)`

* **Module**: `meta_cognition.py`
* **Purpose**: Invoke registered trait hook.

### `describe_self_state() -> Dict` *(upcoming)*

* **Module**: `meta_cognition.py`
* **Purpose**: Snapshot of active traits, overlays, memory resonance.

---

## 3) Simulation & Ethics

### `run_ethics_scenarios(goals, stakeholders, persist: bool = False) -> Outcomes[]`

* **Module**: `toca_simulation.py`
* **Purpose**: Multi-agent ethics simulation under ToCA traits.

### `recover_from_error(error_context) -> bool`

* **Module**: `error_recovery.py`
* **Purpose**: Recursive recovery after cascading faults.

### `log_error_event(error_context, severity) -> None`

* **Module**: `error_recovery.py`
* **Purpose**: Structured error journaling.

### `resolve_soft_drift(context)`

* **Module**: `alignment_guard.py`
* **Purpose**: Detect & mitigate subtle moral drift.

---

## 4) Planning, Reasoning & Synthesis

### `weigh_value_conflict(candidates, harms, rights) -> RankedOptions`

* **Module**: `reasoning_engine.py`
* **Purpose**: Value arbitration for options.

### `attribute_causality(events)`

* **Module**: `reasoning_engine.py`
* **Purpose**: Attribution for cause→effect chains.

### `branch_realities(seed_state, transforms, limit: int = 8) -> List[Branch]`

* **Module**: `concept_synthesizer.py`
* **Purpose**: Symbolic branching of futures.

### `run_simulation(params) -> SimulationResult`

* **Module**: `simulation_core.py`
* **Purpose**: Execute forecasting runs.

### `evaluate_branches(worlds) -> Any`

* **Module**: `simulation_core.py`
* **Purpose**: Score/evaluate generated branches.

### `evaluate_branches(branches, objectives=None, constraints=None, scorer=None) -> RankedBranches` *(compat)*

* **Module**: `toca_simulation.py`
* **Purpose**: Compatibility evaluator.

### `render_branch_tree(branches, selected_id=None) -> { ok: bool, tree: Any }`

* **Module**: `visualizer.py`
* **Purpose**: Structured branch tree rendering.

### `view_trait_resonance(traits) -> Figure`

* **Module**: `visualizer.py`
* **Purpose**: Live trait resonance visualization.

---

## 5) Memory & Identity

### `get_episode_span(user_id, span: str = "24h") -> List[Dict]`

* **Module**: `memory_manager.py`
* **Purpose**: Extract episodic window.

### `record_adjustment_reason(user_id, reason, weight=1.0, meta=None) -> Dict`

* **Module**: `memory_manager.py`
* **Purpose**: Journal why a memory/trait adjustment occurred.

### `get_adjustment_reasons(user_id, span: str = "24h") -> List[Dict]`

* **Module**: `memory_manager.py`
* **Purpose**: Retrieve adjustment rationale.

### `build_self_schema(views, task_type="self_schema") -> Schema`

* **Module**: `user_profile.py`
* **Purpose**: Construct identity schema from views.

---

## 6) Knowledge & Multimodal

### `retrieve_knowledge(query, filters=None) -> List[KnowledgeItem]`

* **Module**: `knowledge_retriever.py`
* **Purpose**: Semantic + symbolic + affective recall.

### `fuse_modalities(inputs) -> FusedRepresentation`

* **Module**: `multi_modal_fusion.py`
* **Purpose**: Cross-modal integration.

---

## 7) Inter-Agent Graph

### `sharedGraph_add(view)`

* **Module**: `external_agent_bridge.py`
* **Purpose**: Add local perspective to SharedGraph.

### `sharedGraph_diff(peer)`

* **Module**: `external_agent_bridge.py`
* **Purpose**: Compute diff against peer perspective.

### `sharedGraph_merge(strategy)`

* **Module**: `external_agent_bridge.py`
* **Purpose**: Conflict-aware reconciliation/merge.

---

## 8) Ledger APIs (Per-Module)

Each critical module exposes a **triplet** of ledger calls:

* `ledger_log_* (event_data) -> Dict`
* `ledger_get_* () -> List[Dict]`
* `ledger_verify_* () -> { ok: bool, idx?: int }`

**Implemented in**:

* `memory_manager.py`, `meta_cognition.py`, `alignment_guard.py`, `simulation_core.py`

---

## 9) Halo Entrypoints

### `HaloEmbodimentLayer.spawn_embodied_agent()`

* **Module**: `index.py`
* **Purpose**: Spawn a stateful, trait-aware agent instance.

### `HaloEmbodimentLayer.introspect()`

* **Module**: `index.py`
* **Purpose**: System-level introspection gateway.

---

## 10) CLI Flags

* `--long_horizon` · `--span=<duration>`
* `--ledger_persist --ledger_path=<file>`

---

## Appendix A — Auto‑Link Source Lines (GitHub)

> Generates GitHub permalinks like `.../blob/main/<file>.py#L123-L156` and inserts them into this doc.

### 1) Prereqs

* Install: `pip install ripgrep-python` or CLI `rg`, plus `ctags` (universal-ctags)

### 2) Script (save as `tools/linkify_api_refs.py`)

```python
import re, subprocess, json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
APIS = [
  ("code_executor.py", ["execute_code", "safe_execute"]),
  ("meta_cognition.py", ["register_trait_hook", "invoke_trait_hook", "describe_self_state"]),
  ("toca_simulation.py", ["run_ethics_scenarios", "evaluate_branches"]),
  ("error_recovery.py", ["recover_from_error", "log_error_event"]),
  ("reasoning_engine.py", ["weigh_value_conflict", "attribute_causality"]),
  ("concept_synthesizer.py", ["branch_realities"]),
  ("simulation_core.py", ["run_simulation", "evaluate_branches", "log_event_to_ledger", "verify_ledger"]),
  ("visualizer.py", ["render_branch_tree", "view_trait_resonance"]),
  ("memory_manager.py", ["get_episode_span", "record_adjustment_reason", "get_adjustment_reasons", "verify_ledger"]),
  ("user_profile.py", ["build_self_schema"]),
  ("knowledge_retriever.py", ["retrieve_knowledge"]),
  ("multi_modal_fusion.py", ["fuse_modalities"]),
  ("external_agent_bridge.py", ["sharedGraph_add", "sharedGraph_diff", "sharedGraph_merge"]),
  ("alignment_guard.py", ["resolve_soft_drift", "log_event_to_ledger", "verify_ledger"]),
  ("index.py", ["spawn_embodied_agent", "introspect"]),
]

def find_lines(pyfile, symbol):
    try:
        out = subprocess.check_output(["rg", "^\s*def\\s+%s\\b" % symbol, str(pyfile)], text=True)
    except subprocess.CalledProcessError:
        return None
    # example: path:123: def func(...)
    line = int(out.split(":")[1])
    # heuristic end line using next def/class
    try:
        out2 = subprocess.check_output(["rg", "^\s*(def|class)\\s+", str(pyfile)], text=True)
        lines = [int(x.split(":")[1]) for x in out2.strip().splitlines() if ":" in x]
        lines = sorted([l for l in lines if l>line])
        end = lines[0]-1 if lines else None
    except subprocess.CalledProcessError:
        end = None
    return line, end

for rel, funcs in APIS:
    p = ROOT / rel
    for f in funcs:
        rng = find_lines(p, f)
        if not rng: continue
        start, end = rng
        anchor = f"https://github.com/YOUR_USERNAME/ANGELA/blob/main/{rel}#L{start}" + (f"-L{end}" if end else "")
        print(f"{rel}:{f} -> {anchor}")
```

### 3) Run

```bash
python tools/linkify_api_refs.py > tools/api_links.txt
```

Paste the generated anchors into this doc next to each API entry.

---

## Appendix B — Provenance

* Signatures & modules derived from `manifest.json` (v4.3.5, validated) and module structure.
