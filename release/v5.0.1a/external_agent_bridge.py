# ExternalAgentBridge.md

## Overview of the ExternalAgentBridge Module

The ExternalAgentBridge module is a core component of the ANGELA Cognitive System (v3.5.3), enabling coordination and collaboration among helper agents, dynamic modules, APIs, and trait-based networking. It manages a shared perspective graph (Υ) for multi-agent workflows, enforces constitutional alignment (τ) with harm ceilings, and supports long-horizon memory for reflective logging. The module ensures ethical compliance through sandboxed scenarios, integrates external data, and handles drift mitigation. It provides robust error handling, visualization, and arbitration of results, maintaining system coherence and transparency.

**Version**: 3.5.3 (2025-08-10)

## Key Components

### SharedGraph (Υ)
- **Purpose**: Maintains a shared perspective graph for agent coordination.
- **Functionality**:
  - **Add**: Stores views (payloads with nodes/edges) with unique IDs and timestamps.
  - **Diff**: Compares graphs with a peer, identifying added/removed nodes and attribute conflicts.
  - **Merge**: Combines views using "prefer_recent" (newest values) or "prefer_majority" (most frequent values).
  - **Ingest Events**: Processes events with deduplication (SHA-256 hashing), tracks conflicts, and updates a vector clock.
- **Output**: View ID (add), diff summary, or merged nodes.

### EthicalSandbox
- **Purpose**: Runs isolated "what-if" ethics scenarios without memory leakage.
- **Process**:
  - Takes goals and stakeholders; validates with AlignmentGuard.
  - Executes scenarios via `toca_simulation.run_ethics_scenarios` (if available).
  - Persists outcomes only if requested.
- **Output**: {"status": "success/error", "outcomes": [...] or "error": "..."}.

### HelperAgent
- **Purpose**: Executes tasks with collaboration and integrates dynamic modules/APIs.
- **Process**:
  - Initialized with name, task, context, modules, APIs, meta-cognition, and task type.
  - Executes via MetaCognition, supporting collaborative execution with other agents.
- **Output**: Task result or error.

### MetaCognition
- **Purpose**: Oversees task execution, reflection, and diagnostics with v3.5.3 upgrades.
- **Functionality**:
  - Integrates SharedGraph, ethical sandbox, and long-horizon memory.
  - Executes tasks with context updates, external data integration, and reasoning (simulation for "drift" tasks).
  - Applies APIs/modules, collaborates with peers, and arbitrates results.
  - Stores adjustment reasons and task summaries; visualizes via SharedGraph.
  - Reflects on outputs; runs diagnostics.
- **Output**: Reviewed result or error with diagnostics.

### ExternalAgentBridge
- **Purpose**: Orchestrates agents, modules, APIs, and trait networking.
- **Functionality**:
  - Manages agent lifecycle, module/API registration, result collection, trait broadcasting, and drift mitigation.
  - Enforces τ harm ceiling; logs to memory; visualizes results.
- **Components**:
  - Agents list, dynamic modules, API blueprints, network graph, trait states.
  - SharedGraph for Υ workflows; CodeExecutor for safe execution.

## Main Processes

### Agent Lifecycle
- **Create Agent**: Creates a HelperAgent with task, context, and task type; logs to ContextManager; adds to network graph.
- **Deploy Module**: Adds module blueprint (name, description); logs event.
- **Register API**: Adds API blueprint (endpoint, name); logs event.
- **Collect Results**: Runs agents in parallel or sequentially, optionally collaborative; adds results to SharedGraph; logs event.
- **Output**: HelperAgent or results list.

### Trait Broadcasting & Sync
- **Broadcast Trait State**:
  - Sends ψ or Υ state to HTTPS URLs; checks harm ceiling and alignment.
  - Caches state; adds edges to network graph; logs audit.
  - Output: List of responses or error.
- **Synchronize Trait States**:
  - Aligns agent’s trait state with peers; runs simulation for coherence.
  - Arbitrates states; caches and stores aligned state.
  - Output: {"status": "success/error", "aligned_state": {...}}.

### Drift Coordination
- **Coordinate Drift Mitigation**:
  - Validates drift data; creates agent for mitigation task.
  - Decomposes task into subgoals; runs simulation (if ReasoningEngine available).
  - Collects/ arbitrates results; shares via SharedGraph; broadcasts ψ state.
  - Logs to memory.
- **Output**: {"drift_data": {...}, "subgoals": [...], "results": [...], ...}.

### Arbitration & Feedback
- **Arbitrate**: Selects best result (max similarity or majority vote); validates with simulation; logs event.
- **Push Behavior Feedback**: Logs feedback to ContextManager.
- **Update GNN Weights**: Logs feedback-driven weight updates.

### ConstitutionSync (τ)
- **Purpose**: Synchronizes constitutional values with harm ceiling.
- **Process**: Updates peer agent’s constitution if harm is within limit; logs audit.
- **Output**: True (success) or False.

## Key Features
- **Shared Perspective (Υ)**: Graph-based coordination with conflict-aware merging.
- **Ethical Compliance**: Sandboxed scenarios; τ harm ceiling; alignment checks.
- **Long-Horizon Memory**: Stores adjustment reasons, task summaries, and audits.
- **Robustness**: Defensive checks, fallbacks, and error recovery integration.
- **Visualization**: Shares results via SharedGraph; logs for transparency.

## Example Workflow
1. Creates agent for “Mitigate drift” with context; registers API/module.
2. Broadcasts ψ state to peers; synchronizes states via simulation.
3. Runs ethical sandbox; collects parallel results; arbitrates best outcome.
4. Logs to memory; visualizes via SharedGraph; reflects (“Results coherent”).
5. Updates constitution if harm < ceiling; stores audit.

## Notes
- All network calls use HTTPS; guarded by AlignmentGuard.
- Stage IV (Φ⁰) hooks are gated; no-op if disabled.
- Integrates with CreativeThinker, ErrorRecovery, and ReasoningEngine for robust workflows.
