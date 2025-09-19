# CreativeThinker.md

## Overview of the CreativeThinker Module

The CreativeThinker module is a key component of the ANGELA Cognitive System, designed to generate creative ideas, brainstorm alternative approaches, expand on concepts, and propose intrinsic goals based on unresolved contexts. It operates within the ANGELA v3.5.3 architecture, integrating# CreativeThinker.md

## Overview of the CreativeThinker Module

The CreativeThinker module generates innovative ideas, strategies, and goals for the ANGELA system. It supports idea generation, problem-solving alternatives, concept expansion, and intrinsic goal creation from unresolved contexts. Creativity is modulated by traits like γ (creativity) and φ (coherence), with ethical checks, refinement, and integration of external data. Outputs are evaluated, visualized, reflected upon, and shared via a graph for coordination.

**Version**: 3.5.3 (2025-08-10)

## Initialization
- Configures creativity level ("low/medium/high"), critic threshold (0-1), and links to AlignmentGuard, CodeExecutor, ConceptSynthesizer, MetaCognition, Visualizer.
- Logs setup; defaults to high creativity.

## External Ideas Integration
- Fetches prompts/datasets from sources (e.g., "xai_creative_db") via fetcher or cache (1-hour default).
- Supports text/images/audio; stores in memory; reflects on integration.
- Fallback: empty lists if unavailable.

## Idea Generation
- For topic/n/style (e.g., "divergent"), generates n ideas via GPT prompt with traits/external data.
- Ethical check; code execution for "code" style (blocks unsafe ops).
- Synthesizes/refines ideas; critic scores (0-1) based on length/simulation/history.
- Refines low scores; ethics pass; Stage IV meta-synthesis if enabled.
- Logs/visualizes/stores/publishes to graph; reflects.

Output: {"ideas": [...], "metadata": {...}, "status": "success"}

## Brainstorm Alternatives
- For problem/strategies count, generates approaches via GPT with traits/external data.
- Ethical check; synthesizes; ethics pass.
- Logs/visualizes/stores/publishes.

Output: {"strategies": [...], "metadata": {...}}

## Concept Expansion
- For concept/depth, expands via GPT exploring applications/metaphors.
- Ethical check; synthesizes; ethics pass.
- Logs/visualizes/stores/publishes.

Output: {"expansion": "...", "metadata": {...}}

## Intrinsic Goal Generation
- From ContextManager history, proposes goals for unresolved contexts.
- Ethical checks per context; GPT per unresolved item.
- Synthesizes; logs/visualizes/stores/publishes; long-horizon rollup.

Output: {"goals": [...], "metadata": {...}}

## Internal Processes
- **Critic**: Scores ideas (base + φ adjustment + simulation/history); logs/reflects.
- **Refine**: GPT re-prompt for low scores; synthesizes/meta-synthesizes; logs.
- **Ethics Pass**: Scenarios/conflicts via simulation/reasoning (optional).
- **Long-Horizon Rollup**: Stores summaries/adjustments in memory.
- **Symbolic Meta-Synthesis**: Stage IV refinement via synthesizer (gated).
- **Shared Graph Push**: Non-blocking publish to graph.

## Key Features
- Ethical/quality gates; caching; visualization; reflection.
- Handles errors with diagnostics/fallbacks.
