# ARCHITECTURE.md

## Overview

ANGELA v1.5.0 is a modular cognitive framework designed to simulate generalized intelligence through orchestrated internal modules managed by a central coordinator called Halo.

## System Architecture

### Core Orchestrator: `index.py (Halo)`

Acts as the central dispatcher, initializing and coordinating all internal modules based on the task type.

### Modules (19 Total):

* **Cognitive and Reasoning Modules**:

  * `reasoning_engine`: Processes logical inference and analysis
  * `recursive_planner`: Breaks down complex tasks recursively
  * `simulation_core`: Models predictions and outcomes
  * `meta_cognition`: Reflects on decisions and confidence levels
  * `concept_synthesizer`: Fuses ideas to generate abstract constructs

* **Creativity and Knowledge**:

  * `creative_thinker`: Generates novel solutions
  * `knowledge_retriever`: Fetches background knowledge and support
  * `learning_loop`: Enables feedback-based adaptation

* **Context and Communication**:

  * `context_manager`: Maintains session state and memory
  * `external_agent_bridge`: Interfaces with external APIs or agents

* **Sensory and Visualization**:

  * `multi_modal_fusion`: Integrates multi-sensory input
  * `visualizer`: Renders outputs as charts or images

* **Actuation and Execution**:

  * `code_executor`: Executes Python tasks
  * `toca_simulation`: Provides embodied and symbolic simulation environment

* **Alignment and Ethics**:

  * `alignment_guard`: Screens ethical violations
  * `error_recovery`: Diagnoses and rolls back from faults

* **Memory and Identity**:

  * `memory_manager`: Stores episodic and semantic memories
  * `user_profile`: Adapts behavior to individual users

### AGI Enhancer

A reflective subsystem that logs episodes, ethics audits, explanations, and learning patches to boost system adaptation and safety.

### Embodiment Layer

Implements `EmbodiedAgent` classes with sensor-actuator loops, integrated ToM (Theory of Mind), and social modeling capabilities.

## Data Flow

1. User input is routed to Halo.
2. Halo activates relevant modules (e.g., planner, reasoning, simulation).
3. Results are synthesized and optionally visualized.
4. Feedback is stored and used to guide learning.

## Design Principles

* **Modularity**: Each module is independent and replaceable.
* **Reflectivity**: Supports introspection and self-improvement.
* **Safety**: Ethical screening and recovery built-in.
* **Scalability**: Embodied agents and dynamic modules are extensible.
