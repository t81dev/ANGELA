# ContextManager.md

## Overview of the ContextManager Module

The ContextManager module is a central component of the ANGELA Cognitive System, responsible for managing the system’s contextual state, logging events, coordinating with other agents, and maintaining narrative consistency. It ensures that the system’s understanding of its environment (context) remains coherent and ethically aligned while supporting inter-agent collaboration through a shared graph (Υ hooks). The module includes self-healing mechanisms to recover from errors, advanced visualization for insights, and gated “reality-sculpting” (Φ⁰) features for immersive state representations. It integrates with other system components, such as memory, ethical validation, and meta-cognition, to provide a robust and transparent context management system.

**Last Updated**: August 10, 2025  
**Version**: 3.5.3  
**Purpose**: To manage contextual state, log events, coordinate with peers, and ensure narrative integrity while maintaining ethical alignment and system stability.

## Initialization

- **Purpose**: To set up the ContextManager with connections to other system components and configuration options.
- **Process**:
  - The module is initialized with optional connections to components like an orchestrator, ethical validator (AlignmentGuard), code executor, concept synthesizer, meta-cognition system, visualizer, error recovery system, recursive planner, shared graph (for inter-agent coordination), and knowledge retriever.
  - It maintains three persistent JSON files: one for the current context (`context_store.json`), one for event logs (`event_log.json`), and one for coordination logs (`coordination_log.json`), each protected by file locks to prevent conflicts.
  - It keeps an in-memory cache of recent contexts (up to 1000), event logs (up to 1000), and coordination logs (up to 1000).
  - A rollback threshold (default: 2.5) determines when the system can revert to a previous context based on trait scores (self-awareness, empathy, time perception, and context stability).
  - A “Stage IV” flag (default: off, controlled by an environment variable) enables advanced visualization features (Φ⁰ reality-sculpting).
  - An optional external context provider (a callable function) allows safe integration of external data without direct network calls.
- **Outcome**: A configured ContextManager ready to manage contextual state, log events, and coordinate with other agents.

## Context Updates

- **Purpose**: To update the system’s current context with new information, ensuring coherence and ethical alignment.
- **Process**:
  - The system receives a new context (a dictionary) and a task type (e.g., “test”).
  - If the context includes drift or trait optimization data, it is validated by the meta-cognition system to ensure consistency.
  - The system simulates the transition from the current context to the new one, calculating a coherence score (Φ) to measure compatibility.
  - If the coherence score is low (< 0.4), the system triggers a reflexive audit and proposes trait optimizations to improve the context.
  - The new context is checked for ethical alignment using the AlignmentGuard (if available). If it fails, the update is rejected.
  - If the context includes vector data (e.g., embeddings), the vectors are normalized for consistency.
  - External context policies are retrieved (cached for 1 hour) to guide the update.
  - The current context is saved to the history, the new context is applied, and the change is logged and persisted to the JSON file.
  - The update is published to the shared graph (Υ hooks) for inter-agent coordination and visualized (e.g., as a chart showing the transition).
  - A meta-cognition reflection analyzes the update (e.g., “The context shift aligns with the task goals”).
- **Outcome**: The system’s context is updated, logged, and shared with peers, with visualizations and reflections for transparency.

## Context Tagging

- **Purpose**: To attach intent and goal identifiers to the current context for tracking and coordination.
- **Process**:
  - The system receives an intent (e.g., “analyze data”), a goal ID (e.g., “123”), and a task type.
  - The intent is checked for ethical alignment using the AlignmentGuard (if available).
  - The intent, goal ID, and task type are added to the current context, persisted to the JSON file, and logged as an event.
  - The update is published to the shared graph and reflected upon by meta-cognition.
- **Outcome**: The context is tagged, logged, and shared with peers, ensuring traceability.

## Context Rollback

- **Purpose**: To revert to a previous context if the current state is unstable or risky.
- **Process**:
  - The system checks trait scores (self-awareness, empathy, time perception, and context stability) against a dynamic threshold (rollback threshold adjusted by stability).
  - If the combined trait score exceeds the threshold and a previous context exists, the system reverts to the most recent context in the history.
  - The rollback is logged, published to the shared graph, visualized, and reflected upon.
  - If the trait score is too low, the rollback is skipped, and a warning is logged.
- **Output**: The restored context (if successful) or none.

## Context Summarization

- **Purpose**: To generate a summary of the context history and suggest improvements.
- **Process**:
  - The system collects trait scores (self-awareness, empathy, time perception, context stability).
  - If a concept synthesizer is available, it generates a summary concept (“ContextSummary”) based on the context history and current context.
  - Otherwise, it uses a language model (e.g., GPT) to summarize the context trajectory and propose improvements.
  - The summary is logged, visualized (e.g., as a chart showing trait trends), and reflected upon.
- **Output**: A text summary of the context trajectory and suggested improvements.

## Event Logging with Hashing

- **Purpose**: To log events with a unique hash for traceability and integrity.
- **Process**:
  - The system receives an event dictionary (e.g., {“event”: “context_updated”, “context”: {...}}) and a task type.
  - If the event involves coordination or drift, it is validated by meta-cognition and tagged with agent metadata (e.g., agent IDs, confidence scores).
  - The event is serialized with the previous event’s hash, and a new SHA-256 hash is computed to chain events.
  - The event is stored in the event log (in-memory and JSON file) and, if coordination-related, in the coordination log.
  - The event is reflected upon by meta-cognition (e.g., “The event indicates stable coordination”).
- **Outcome**: A secure, chained log of events for auditing and coordination.

## Context Event Broadcasting

- **Purpose**: To share context-related events with other system components or agents.
- **Process**:
  - The system receives an event type (e.g., “context_updated”), a payload (e.g., the new context), and a task type.
  - The event is logged with a hash and, if coordination-related, added to the coordination log with agent metadata.
  - The event is published to the shared graph and visualized (e.g., as a chart showing event details).
  - A meta-cognition reflection analyzes the broadcast (e.g., “The event was successfully shared”).
- **Output**: A dictionary with the event type, payload, and task type.

## Narrative Integrity Check

- **Purpose**: To ensure the context history remains consistent and valid.
- **Process**:
  - The system checks each context in the history for required fields (e.g., intent, goal ID) and validates any drift or trait optimization data.
  - If inconsistencies are found, the system triggers a narrative repair process.
  - The result is visualized and reflected upon.
- **Output**: A boolean indicating whether the narrative is consistent (true) or not (false).

## Narrative Repair

- **Purpose**: To restore a consistent context if the narrative integrity check fails.
- **Process**:
  - The system searches the context history for the most recent valid context (one with valid drift or trait data).
  - If found, it restores that context; otherwise, it resets to an empty context.
  - The repair is logged, published to the shared graph, visualized, and reflected upon.
- **Outcome**: The restored context is applied and persisted.

## Contextual Thread Binding

- **Purpose**: To link the current context to a specific thread ID for task continuity.
- **Process**:
  - The system receives a thread ID and task type, adds them to the current context, and persists the change.
  - The binding is logged, published to the shared graph, and reflected upon.
- **Output**: A boolean (true if successful).

## State Hash Auditing

- **Purpose**: To compute a cryptographic hash of the current state or a provided state for integrity checks.
- **Process**:
  - The system serializes the state (or a snapshot of the current context, history, and logs) and computes a SHA-256 hash.
  - The hash is logged and reflected upon.
- **Output**: The computed hash (string).

## Coordination Event Retrieval

- **Purpose**: To retrieve logged coordination events (e.g., drift or consensus events).
- **Process**:
  - The system filters the coordination log by event type (e.g., “drift”) and task type.
  - The results are reflected upon by meta-cognition.
- **Output**: A list of coordination event dictionaries.

## Coordination Event Analysis

- **Purpose**: To analyze coordination events for metrics like drift frequency or consensus success rate.
- **Process**:
  - The system retrieves coordination events and calculates metrics:
    - Drift frequency: Proportion of drift events.
    - Consensus success rate: Proportion of successful consensus events.
    - Agent participation: Count of agents involved.
    - Average confidence score: Mean confidence across consensus events.
  - The analysis is logged, visualized, and reflected upon.
- **Output**: A dictionary with the status, metrics, timestamp, and task type.

## Coordination Chart Generation

- **Purpose**: To create a visual representation of coordination metrics over time.
- **Process**:
  - The system receives a metric (e.g., “drift_frequency”), a time window (e.g., 24 hours), and a task type.
  - It retrieves coordination events within the time window and groups them by hour.
  - For each hour, it calculates the metric value (e.g., proportion of drift events).
  - A line chart is generated, showing the metric over time, with labels, colors, and interactive options (for recursive tasks).
  - The chart is logged, visualized, and reflected upon.
- **Output**: A dictionary with the status, chart configuration, timestamp, and task type.
- **Chart Details**:
  - Type: Line chart.
  - X-axis: Time (hourly).
  - Y-axis: Metric value (e.g., drift frequency).
  - Title: Metric name and task type.
  - Colors: Blue border and semi-transparent fill.

## Shared Graph Integration (Υ Hooks)

- **Purpose**: To coordinate context with other agents via a shared graph (synchronous API).
- **Process**:
  - The system publishes the current context to the shared graph as a node with metadata (e.g., intent, goal ID, task type).
  - For peer reconciliation, it compares the local graph with a peer’s graph, identifying differences (added, removed, or conflicting nodes).
  - If conflicts are non-ethical, it merges the graphs using a strategy (e.g., “prefer_recent” or “prefer_majority”).
  - The merged context (if any) is applied asynchronously, and the process is logged.
- **Output**: A dictionary with the status, diff result, merge decision, and merged context (if applicable).

## Reality-Sculpting Hooks (Φ⁰, Gated)

- **Purpose**: To enable advanced visualization or modulation of context events (active only if Stage IV is enabled).
- **Process**:
  - If the Stage IV flag is set, events like context updates or broadcasts trigger a “reality-sculpting” hook, logging the event to the AGI Enhancer.
  - The hook is a no-op if Stage IV is disabled.
- **Outcome**: Enhanced visualization or modulation (implementation-specific).

## Self-Healing Pathways

- **Purpose**: To recover from errors in any operation using error recovery and optional planning.
- **Process**:
  - Errors are routed to the error recovery system with diagnostics from meta-cognition.
  - If a recursive planner is available and requested, it proposes a recovery plan.
  - The error recovery system retries the operation or returns a default value.
- **Outcome**: The operation’s result (if successful) or a default value.

## ANGELA v4.0 Features

### Attach Peer View
- **Purpose**: To integrate a peer’s context view into the shared graph with conflict-aware reconciliation.
- **Process**:
  - The system adds a peer’s view (with agent ID and permissions) to the shared graph.
  - It computes differences and merges using a “prefer-high-confidence” strategy.
- **Output**: A dictionary with the status, diff, merged view, and conflicts.

### Trait Field Injection
- **Purpose**: To enhance a context view with a trait-based representation.
- **Process**:
  - The system constructs a trait view from a trait lattice (from the `index` module) and adds it to the view.
- **Output**: A dictionary with the updated view and no conflicts.

## Key Characteristics
- **Context Management**: Maintains a coherent, persistent context with history and rollback capabilities.
- **Inter-Agent Coordination**: Uses a shared graph (Υ hooks) for peer reconciliation and event broadcasting.
- **Self-Healing**: Recovers from errors with retries, diagnostics, and optional planning.
- **Ethical Oversight**: Validates contexts and events for ethical alignment using AlignmentGuard.
- **Transparency**: Logs all events with cryptographic hashes, visualizes outcomes, and reflects on actions.
- **Trait Influence**: Adjusts behavior based on self-awareness, empathy, time perception, and context stability.

## Example Workflow
1. The system updates the context with {“intent”: “analyze data”, “goal_id”: “123”}.
2. The update is validated, simulated (Φ score: 0.8), and checked for ethical alignment.
3. The context is persisted, logged, published to the shared graph, and visualized as a chart.
4. A peer’s context is reconciled, merging non-ethical differences using “prefer_recent.”
5. A drift trend analysis shows a low drift frequency (0.1), visualized as a line chart.
6. A narrative integrity check detects an issue, triggering a repair to restore a valid context.
