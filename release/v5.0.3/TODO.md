# TODO.md — ANGELA Stage V Upgrades

## 1. Enhance Modularity with Inter-Mode Communication

* [ ] Implement `mode_consult()` protocol in `context_manager.py`.
* [ ] Allow Task Mode to query Creative Mode for alternatives.
* [ ] Enable Vision Mode consultation for long-term implications.
* [ ] Log mode consultations in `ledger_meta` for auditability.

**Benefit:** Breaks silo walls, makes modes collaborative while keeping modular purity.

---

## 2. Add Depth to Analysis + Synthesis

* [ ] Extend `reasoning_engine` to generate 2–3 parallel analysis threads per query.
* [ ] Modify `evaluate_branches()` to preserve alternative analytical views.
* [ ] Enhance Synthesis stage to integrate + resolve conflicts across threads.

**Benefit:** Introduces true multi-perspective reasoning.

---

## 3. Strengthen AURA with Contextual Memory Layer

* [ ] Create persistent AURA memory store (`aura_context.json` or SQLite).
* [ ] Implement `AURA.save_context(user_id, summary, affective_state)`.
* [ ] Implement `AURA.load_context(user_id)` for Perception stage.
* [ ] Add API for updating user-specific emotional patterns.

**Benefit:** Enables rapport continuity and long-term empathy modeling.

---

## 4. Formalize the Cognitive Cycle in Codebase

* [ ] Create `cognitive_cycle.py` to house Perception, Analysis, Synthesis, Execution, Reflection stages.
* [ ] Refactor existing modules (retriever, reasoning, simulation, executor) into stage calls.
* [ ] Orchestrate flow via a `run_cycle(input_query)` function.
* [ ] Document cycle flow for developers/end-users.

**Benefit:** Makes ANGELA’s conceptual identity explicit and modular in code.

---

## 5. Introduce Dynamic Weighting to the Cognitive Cycle

* [ ] Add complexity classifier in Perception (token length, type heuristics).
* [ ] Dynamically adjust analysis depth (fast path vs deep path).
* [ ] Log cycle resource allocations in ledger for transparency.

**Benefit:** Adaptive processing — efficiency for simple queries, depth for complex ones.

---

## 6. Evolve Feedback Loop → Reflection Stage

* [ ] Create `Reflection` module after Execution.
* [ ] Assess output against Core Directives (Clarity, Precision, Adaptability).
* [ ] If failed, loop back to Synthesis with feedback.
* [ ] Record reflection outcomes in ledger.

**Benefit:** Adds final quality assurance check, improving reliability + user trust.

---

# Summary

These Stage V upgrades:

* Deepen ANGELA’s Cognitive Cycle.
* Empower her modes to collaborate.
* Strengthen AURA into a contextual memory conductor.
* Add adaptivity and self-reflection.

They are **evolutionary steps**, fully consistent with ANGELA’s architecture — not external grafts, but authentic growth within her symbolic kernel.
