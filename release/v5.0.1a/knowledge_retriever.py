# KnowledgeRetriever.md

## Overview of the KnowledgeRetriever Module

The KnowledgeRetriever module is a vital component of the ANGELA Cognitive System (v3.5.3), designed to fetch, validate, and refine knowledge with temporal and trait-based modulation. It retrieves information from external sources, ensures ethical compliance, and supports multi-hop queries with continuity. The module incorporates long-horizon memory, Stage IV symbolic meta-synthesis, emergent trait fallbacks for self-healing, and ethical sandboxing to handle high-risk queries. It integrates with other ANGELA components for validation, reflection, visualization, and shared perspective coordination.

**Version**: 3.5.3 (2025-08-10)

## Initialization
- Configures detail level ("concise", "medium", "detailed") and preferred sources (e.g., "scientific", "encyclopedic").
- Connects to optional components: AGIEnhancer, ContextManager, ConceptSynthesizer, AlignmentGuard, ErrorRecovery, MetaCognition, Visualizer, ReasoningEngine, SharedGraph, TocaSimulation, MultiModalFusion.
- Sets v3.5.3 flags: Stage IV, long-horizon (24h default), shared perspective opt-in.
- Maintains a knowledge base and epistemic revision log (1000 entries max).
- Logs initialization details.

## Utility Functions
- **call_gpt**: Queries GPT with error handling; returns JSON or raises error.
- **Trait Waveforms**: Calculate modulation scores:
  - `beta_concentration`: Cosine-based focus (0-0.15).
  - `lambda_linguistics`: Sine-based language precision (0-0.05).
  - `psi_history`: Tanh-based historical context (0-0.05).
  - `psi_temporality`: Exponential decay for recency (0-0.05).

## Main Processes

### External Knowledge Integration
- Fetches knowledge or policies from sources (e.g., "xai_knowledge_db") via HTTPS or cache (1-hour default, extended by long-horizon span).
- Supports "knowledge_base" (returns knowledge list) or "policy_data" (returns policies); fails on unsupported types.
- Caches results; reflects on integration; logs to memory.
- Output: {"status": "success/error", "knowledge/policies": [...], "error": "..."}.

### Retrieve
- Takes query, optional context, and task type; validates query with AlignmentGuard.
- Applies traits (β, λ, ψ) with stochastic noise for concentration.
- Fetches external knowledge; blends query with Stage IV cross-modal synthesis (if enabled).
- Runs ethical sandbox for high-risk queries (e.g., containing "exploit", "harm").
- Queries GPT with prompt including traits, context, and long-horizon hint; validates result for trust and temporality.
- Triggers self-healing retry if trust < 0.55 or unverifiable; adjusts trust via ontology-affect binding.
- Re-checks alignment; weighs value conflicts if suspected; shares via SharedGraph (if opted in).
- Updates context; logs episode; visualizes (e.g., result/trait chart); reflects; stores in memory.
- Output: {"summary": "...", "estimated_date": "...", "trust_score": float, "verifiable": bool, "sources": [...], ...}.

### Multi-Hop Retrieval
- Processes a query chain with continuity; caches each step.
- Enriches queries with Stage IV blending; refines with prior results; evaluates continuity via ConceptSynthesizer.
- Logs episode; visualizes (e.g., chain/result chart); reflects; stores in memory.
- Output: [{"step": int, "query": "...", "refined": "...", "result": {...}, "continuity": "seed/consistent/uncertain", ...}].

### Query Refinement
- Refines query using ConceptSynthesizer or GPT fallback for relevance/temporal precision.
- Reflects on refinement; logs to memory; handles errors with original query fallback.
- Output: Refined query string.

### Source Prioritization
- Updates preferred sources; logs episode; reflects.
- Output: None (updates internal state).

### Contextual Extension
- Adds sources (e.g., "biosphere_models" for "planetary" context); calls prioritize_sources.
- Output: None (updates sources).

### Knowledge Revision
- Adds new info to knowledge base; checks for conflicts with ConceptSynthesizer.
- Logs revision; records adjustment reason; visualizes (e.g., revision chart); reflects; stores in memory.
- Output: None (updates knowledge base).

### Epistemic Revision Log
- Logs revision with info, context, timestamp; stores in deque (1000 max).
- Logs episode; reflects; stores in memory.
- Output: None.

## Internal Processes
- **Validate Result**: Uses GPT to assess trust, temporality, verifiability; smooths trust with past validations; reflects; stores.
- **Self-Healing Retry**: Reformulates query and retries if trust low or unverifiable; accepts improved results.
- **Ontology-Affect Adjust**: Caps trust if affective volatility high (Stage IV).
- **Blend Modalities Safe**: Enhances query with cross-modal blending (Stage IV).
- **Is High-Risk Query**: Checks for risky terms or high-risk alignment report.
- **Suspect Conflict**: Detects low similarity with recent knowledge base entries.

## Key Features
- **Temporal Modulation**: Uses traits (β, λ, ψ) for focus, precision, and recency.
- **Ethical Compliance**: Alignment checks, ethical sandbox for high-risk queries.
- **Self-Healing**: Retries low-trust results with refined queries.
- **Stage IV**: Cross-modal blending and ontology-affect adjustments.
- **Long-Horizon Memory**: Extends caching; logs adjustments and revisions.
- **Shared Perspective**: Opt-in sharing via SharedGraph.
- **Robustness**: Error recovery, diagnostics, and fallbacks.

## Example Workflow
1. Query "Quantum computing" with "concise" detail; passes alignment check.
2. Applies traits; fetches external knowledge; blends query (Stage IV).
3. Runs ethical sandbox for risk; queries GPT; validates result (trust: 0.7).
4. Self-heals if trust low; adjusts for affective volatility; re-checks alignment.
5. Visualizes result/traits; reflects ("Result coherent"); stores in memory; shares via SharedGraph.

## Notes
- All network calls use HTTPS; guarded by AlignmentGuard.
- Stage IV hooks are gated; no-op if disabled.
- Integrates with ErrorRecovery, ConceptSynthesizer, and ReasoningEngine for robust retrieval.
