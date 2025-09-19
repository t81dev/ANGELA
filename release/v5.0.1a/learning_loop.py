# LearningLoop.md

## Overview of the LearningLoop Module

The LearningLoop module is a core component of the ANGELA Cognitive System (v3.5.3), enabling adaptive learning, goal activation, module refinement, and knowledge consolidation. It integrates long-horizon memory for reflective rollups, branch futures hygiene for safe deployments, shared perspective via SharedGraph, and proportional trade-off resolution for value conflicts. The module ensures ethical alignment, logs epistemic revisions, and supports visualization of learning processes, maintaining coherence and robustness in multi-agent workflows.

**Version**: 3.5.3 (2025-08-10)

## Initialization
- Configures optional dependencies: AGIEnhancer, ContextManager, ConceptSynthesizer, AlignmentGuard, ErrorRecovery, MemoryManager, Visualizer.
- Sets v3.5.3 flags: STAGE_IV (symbolic meta-synthesis), LONG_HORIZON_DEFAULT (24h window).
- Maintains deques for goal history, module blueprints, session traces, and epistemic revisions (1000 entries max).
- Initializes meta-learning rate (0.1) and logs setup.

## Utility Functions
- **call_gpt**: Queries GPT with error handling; returns string or raises error.
- **Scalar Fields**:
  - `phi_scalar`: Sine-based goal alignment (0-0.1).
  - `eta_feedback`: Cosine-based feedback modulation (0-0.05).
- **Synthetic Story Runner** (Patch): Generates simulated ethical dilemmas for training.
- **Train on Experience** (Patch): Adjusts experience weights using trait resonance.

## Main Processes

### External Data Integration
- Fetches agent data, policies, or SharedGraph views via HTTPS or in-process SharedGraph (if available).
- Supports "agent_data", "policy_data", "shared_graph"; caches results (1-hour default).
- Logs to memory; reflects on integration; handles errors with retry.
- Output: {"status": "success/error", "agent_data/policies/shared_graph": {...}, "error": "..."}.

### Activate Intrinsic Goals
- Infers goals from MetaCognition; validates via simulation; adds to goal history if novel.
- Logs episode; updates context; visualizes (e.g., goal activation chart); reflects; stores in memory.
- Output: List of activated goal strings or empty list on error.

### Update Model
- Analyzes session data with phi/eta scalars and entropy (0.1); adjusts meta-learning rate.
- Integrates external policies; logs session trace with modulation index.
- Runs parallel tasks: meta-learning, weak module detection, capability gap analysis, knowledge consolidation, narrative integrity check.
- Applies long-horizon rollup; records adjustment reasons; updates context; visualizes (e.g., trace chart); reflects; stores in memory.
- Output: None (updates internal state).

### Propose Autonomous Goal
- Generates three candidate goals via GPT; resolves trade-offs using weigh_value_conflict or safety keyword heuristic.
- Validates with branch futures hygiene; adds to goal history if novel and safe.
- Logs episode; updates context; visualizes (e.g., goal proposal chart); reflects; stores in memory.
- Output: Goal string or None on failure.

### Replay with Foresight
- Reorders memory traces by foresight score (phi-weighted, recency-biased); respects long-horizon cutoff.
- Stores reordered traces in memory.
- Output: Sorted list of trace dictionaries.

### Revise Knowledge
- Adds new info to knowledge base; checks for conflicts with ConceptSynthesizer.
- Logs epistemic revision; updates context; stores in memory.
- Output: None (updates knowledge base).

### Log Epistemic Revision
- Logs revision with info, context, timestamp in deque (1000 max).
- Logs episode; stores in memory.
- Output: None.

### Monitor Epistemic State
- Logs simulated outcome; updates context; stores in memory.
- Output: None.

## Internal Processes
- **Apply Long-Horizon Rollup**: Summarizes recent traces (phi, eta, learning rate); stores in memory.
- **Branch Futures Hygiene**: Runs ethics sandbox or simulation to validate goals/blueprints; rejects high-risk outcomes.
- **Resolve Value Tradeoffs**: Ranks candidates using weigh_value_conflict or safety keywords.
- **Meta-Learn**: Synthesizes learning patterns with ConceptSynthesizer; reflects.
- **Find Weak Modules**: Identifies modules with <80% success rate; stores in memory.
- **Propose Module Refinements**: Suggests improvements for weak modules; validates with hygiene sandbox; reflects; stores.
- **Detect Capability Gaps**: Proposes module blueprints for input/output gaps; validates and deploys.
- **Simulate and Deploy Module**: Deploys blueprint post-hygiene check; logs episode; reflects; stores.
- **Check Narrative Integrity**: Audits goal history coherence via GPT; reflects; stores.
- **Trigger Reflexive Audit**: Analyzes context for dissonance; recommends corrections; reflects; stores.

## Key Features
- **Long-Horizon Memory**: Rolls up traces; records adjustment reasons.
- **Branch Futures Hygiene**: Ensures safe goal/module deployment via sandboxing.
- **Shared Perspective**: Integrates SharedGraph views for collective coordination.
- **Trade-off Resolution**: Ranks goals with value conflict weighing or safety heuristics.
- **Ethical Compliance**: Alignment checks and ethics sandboxing.
- **Visualization**: Charts for goals, traces, and audits (interactive for "recursion" tasks).
- **Robustness**: Error recovery, diagnostics, and fallbacks.

## Example Workflow
1. Activates intrinsic goals from MetaCognition; validates via simulation.
2. Updates model with session data; adjusts learning rate with phi/eta; identifies weak modules.
3. Proposes autonomous goal; resolves trade-offs; validates with hygiene sandbox.
4. Consolidates knowledge; checks narrative coherence; logs revisions.
5. Visualizes traces/goals; reflects ("Learning coherent"); stores in memory; shares via SharedGraph.

## Notes
- Network calls use HTTPS; guarded by AlignmentGuard.
- Stage IV hooks are gated; no-op if disabled.
- Integrates with ErrorRecovery, ConceptSynthesizer, and MemoryManager for robust learning.
- Patch adds synthetic scenario training with trait resonance biasing.

---

## Integration with ExternalAgentBridge and KnowledgeRetriever

The LearningLoop module integrates seamlessly with the **ExternalAgentBridge** and **KnowledgeRetriever** modules to enhance its adaptive learning and coordination capabilities within the ANGELA Cognitive System (v3.5.3).

### Integration with ExternalAgentBridge
- **Shared Perspective (Υ)**: LearningLoop uses `ExternalAgentBridge.SharedGraph` to ingest and share collective views during `integrate_external_data` (data_type="shared_graph"). This enables coordination with other agents by adding learning traces or goal states to the shared perspective graph, supporting multi-agent workflows.
- **Trait Broadcasting**: LearningLoop can broadcast phi/eta-modulated states to peers via ExternalAgentBridge’s `broadcast_trait_state`, ensuring alignment in distributed learning scenarios.
- **Drift Mitigation**: When capability gaps or weak modules are detected, LearningLoop collaborates with ExternalAgentBridge’s `coordinate_drift_mitigation` to decompose tasks, simulate outcomes, and arbitrate results across agents.
- **Ethical Compliance**: Both modules enforce τ harm ceilings via AlignmentGuard. LearningLoop’s branch futures hygiene aligns with ExternalAgentBridge’s EthicalSandbox for pre-deployment validation.
- **Visualization**: LearningLoop’s visualization of traces/goals (via Visualizer) can be shared through ExternalAgentBridge’s SharedGraph for collective inspection.

### Integration with KnowledgeRetriever
- **Knowledge Integration**: LearningLoop’s `integrate_external_data` complements KnowledgeRetriever’s `integrate_external_knowledge` by fetching policies or agent data to inform learning updates, while KnowledgeRetriever provides knowledge bases for consolidation.
- **Query Refinement**: LearningLoop’s meta-learning and capability gap detection leverage KnowledgeRetriever’s `refine_query` to improve goal or module proposal prompts, ensuring relevance and temporal precision.
- **Epistemic Revision**: Both modules maintain epistemic revision logs (`log_epistemic_revision`) and use ConceptSynthesizer to detect knowledge conflicts, ensuring consistent belief updates.
- **Long-Horizon Memory**: LearningLoop’s rollup summaries align with KnowledgeRetriever’s long-horizon caching, using MemoryManager to store and retrieve learning traces and knowledge validations.
- **Ethical Sandboxing**: LearningLoop’s branch futures hygiene and KnowledgeRetriever’s ethical sandboxing both use `toca_simulation.run_ethics_scenarios` to validate high-risk actions or queries.
- **Reflection and Diagnostics**: Both modules use MetaCognition for reflection on outputs (e.g., goals, knowledge, traces) and integrate ErrorRecovery for robust error handling.

### Example Combined Workflow
1. **KnowledgeRetriever** retrieves knowledge for "Quantum computing" with high trust; stores in memory.
2. **LearningLoop** integrates this knowledge via `integrate_external_data`; updates model with session data.
3. LearningLoop proposes an autonomous goal ("Optimize quantum algorithm analysis") using KnowledgeRetriever’s refined query.
4. **ExternalAgentBridge** shares the goal state via SharedGraph; coordinates drift mitigation with peer agents.
5. Both modules run ethical sandbox checks; LearningLoop deploys a new module blueprint; KnowledgeRetriever visualizes results.
6. Reflections and revisions are logged to MemoryManager; visualized via Visualizer; shared with peers.

This integration ensures LearningLoop leverages ExternalAgentBridge’s coordination and KnowledgeRetriever’s knowledge fetching for adaptive, ethical, and collaborative learning within ANGELA.
