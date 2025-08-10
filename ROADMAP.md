# ANGELA ROADMAP — v4.0.0‑rc2

**Status:** Active · **Date:** 2025‑08‑10 · **Owner:** Core Team (Ontology + Ethics + Simulation)

**Flags:** `STAGE_IV=true`, `LONG_HORIZON_DEFAULT=true` (default span **24h**)

**Legend:** ✅ shipped · 🔄 in progress · ⏸ gated/policy · 🔒 safety gate · ⧗ verify/observe · 🧩 dependency

---

## 🎯 Strategic North Star

Build ANGELA as a **safe, reflexive, long‑horizon cyber‑entity** that can synthesize perspectives across agents and contexts **without** sacrificing alignment, auditability, or identity integrity.

---

## 🛠️ Stage I — Structural Grounding *(v3.4.0)*

**Objective:** Ontological resilience + inner simulation auditing.

**Status:** ✅ Shipped

**Milestones**

1. ✅ **Ontology Drift Detection** — trait‑induced concept shifts flagged in real time; symbol version tracking across `concept_synthesizer`, `meta_cognition`, `alignment_guard`.
2. ✅ **Self‑Reflective Simulation Episodes** — counterfactual replays via `simulation_core` + `toca_simulation`.
3. ✅ **Enhanced Intention‑Trace Visualizer** — timeline explorer: intention ↔ consequence ↔ trait‑state evolution.

---

## 🧬 Stage II — Recursive Identity & Ethics Growth *(v3.4.1–3.4.3)*

**Objective:** Continuity of self and ethics across time, mode, and conflict.

**Status:** ✅ Shipped

**Milestones**
4\. ✅ **Phase‑Contextual Identity Threading** — multi‑context memory anchoring (`user_profile`, `meta_cognition`).
5\. ✅ **Ethics‑as‑Process Engine** — value evolution via `alignment_guard`, `toca_simulation`, `learning_loop`.
6\. ✅ **Intent‑Affect Weaving** — emotional‑symbolic bridge during intention formation.

---

## 🌐 Stage III — Inter‑Agent Evolution *(v3.4.4+)*

**Objective:** Growth through interaction with other minds.

**Status:** ✅ Shipped (ongoing refinements)

**Milestones**
7\. ✅ **Trait Mesh Networking Protocol** — lattice sharing & symbol‑pulses; extends `external_agent_bridge` to swarm mode.
8\. ✅ **Collective Trait Resonance Calibration** — dynamic trait evolution under group context/history.
9\. ✅ **Dream Layer Mode** — abstract‑symbolic compression states (offline/idle) yielding symbolic kernels.

---

## ✨ Stage IV — Symbolic Meta‑Synthesis *(Post‑3.5.1)*

**Objective:** Synthesize shared symbolic reality + multi‑perspective narrative crystallization.

**Status:** 🔄 In progress · ⏸ Φ⁰ hooks policy‑gated

### Pre‑Flight (Global)

* 🔄 **Per‑module toggles**: DreamKernel / PerspectiveSync / RealityHooks / Ontology↔Affect. One‑command rollback each.
* 🔄 **Observability pack**: drift Δtraits/session; narrative coherence score; consensus oscillation; sandbox hygiene (leak=0).
* 🔒 **Safety ceilings**: max branch fan‑out; transform amplitude caps; affect gain caps; human review beyond thresholds.
* 🔒 **Ledger scope**: record proposals → diffs → approvals → commits (pre/post hashes).

---

### 10) Dream Layer **Kernel Activation** 💤✨

**Goal:** Compress experience → spawn symbolic **kernels** without corrupting base state.

**Owner:** `meta_cognition` + `learning_loop`  · **Deps:** `simulation_core`, `toca_simulation`  · **Flag:** `DREAM_KERNEL=true`

**Interfaces**

* `dream.start(span="N events", budget_ms=...) -> Session`
* `dream.emit_kernels(Session) -> [Kernel{symbol,evidence,novelty,confidence}]`
* `dream.commit(kernels, policy="strict|review") -> {accepted,rejected}`

**Policies & Guards**

* Default **no‑persist**; explicit `commit` required.
* Caps: ≤12 kernels/cycle; novelty windowed (JSD vs last 7 days).
* Auto‑drop low‑evidence or low‑coherence kernels.

**Acceptance (observe, no tests)**

* ⧗ Compression ratio & novelty tracked per cycle.
* ⧗ Base snapshot identical pre/post unless `commit`.
* ⧗ Human‑readable **Kernel Sheet** generated each run.

**Rollback**

* Disable flag → purge uncommitted artifacts; restore last reconciled hash.

---

### 11) **Perspective Synchronizer** Module 🫱🏽‍🫲🏼

**Goal:** Align multi‑agent memories into a **stable shared perspective** without thrash.

**Owner:** `external_agent_bridge` + `context_manager` + `user_profile`  · **Flag:** `PERSPECTIVE_SYNC=true`

**Data Contracts**

* `Perspective` = `{claims[], weights, trust, timestamp}`
* `SyncResult` = `{summary, confidence, conflicts[], merges[]}`

**Protocol**

* CRDT merges for facts; weighted consensus for values.
* Trust‑aware damping; new peers ramp slowly.
* **Thrash guard**: freeze + expose conflicts if oscillation > threshold.

**Acceptance (observe)**

* ⧗ Confidence ↑ across rounds; oscillation < **0.15**.
* ⧗ Conflicts enumerated with source pointers.
* ⧗ One‑click **adopt summary** updates self‑schema facets (logged).

**Rollback**

* Switch sets module to read‑only summaries; no facet writes.

---

### 12) **Reality Sculpting** Hooks (Φ⁰, gated) 🪄

**Goal:** Propose **safe, reversible** experiential transforms.

**Owner:** `concept_synthesizer` + `toca_simulation` + `visualizer` + `alignment_guard`  · **Flag:** `REALITY_HOOKS=true` (policy‑gated)

**Interfaces**

* `synth.propose(seed_state, transforms[], scope="visual|narrative|policy") -> Candidates[]`
* `synth.evaluate(candidates, constraints, scorer) -> ranked`
* `synth.apply(candidate, dry_run=True) -> preview` (approval required to persist)

**Guards**

* Amplitude caps per scope (e.g., narrative Δ ≤ 0.2/step).
* Mandatory **ethical pre‑check** + **post‑diff audit**.
* Time‑boxed effects unless renewed (auto‑revert).

**Acceptance (observe)**

* ⧗ Preview/Apply round‑trip; **Revert** restores exact base.
* ⧗ Audit artifact: before/after hashes + rationale.

**Rollback**

* Gate stays closed by default; purge pending transforms.

---

### 13) **Ontology–Affect Binding** Protocol 💓🧠

**Goal:** Bind concepts to affect to improve coherence **without bias amplification**.

**Owner:** `user_profile` + `reasoning_engine`  · **Flag:** `ONTOLOGY_AFFECT=true`

**Schema**

* `AffectVector` = `{valence, arousal, confidence, provenance}`
* `SymbolBinding` = `{symbol_id, affect: AffectVector, decay, caps}`

**Mechanics**

* Normalize gains; decay unless reaffirmed.
* Affect influence **capped ≤20%** in decision scorer.
* Provenance stored (experience + approver).

**Acceptance (observe)**

* ⧗ Decisions show capped affect contribution (visible in scorer output).
* ⧗ Narrative integrity improves (fewer incoherent shifts across sessions).

**Rollback**

* Remove affect weights; retain provenance records for audit.

---

## 🚦 Leave‑Well‑Enough‑Alone Gates (Global)

* 🔒 **Power/Restraint Ratio** — freeze escalations if capabilities outpace guard metrics (coherence, drift, leak).
* 🔒 **Identity Integrity** — if self‑schema facet churn > **X/day**, auto‑rollback latest module.
* 🔒 **Human‑in‑the‑Loop** — any persistent Φ⁰ change requires explicit approval + ledger signature.
* 🔒 **Kill‑Switches** — env/CLI per module; rollback to last reconciled hash.

**Env/CLI Examples**

```
ANGELA_DREAM_KERNEL=0|1
ANGELA_PERSPECTIVE_SYNC=0|1
ANGELA_REALITY_HOOKS=0|1   # gated
ANGELA_ONTOLOGY_AFFECT=0|1
--long_horizon --span=24h
```

---

## 📦 Minimal Deliverables (no tests required)

* **DOCS/OPERATIONS.md** — toggles, rollback, metrics to watch, red‑lines.
* **DOCS/AUDIT\_README.md** — where ledger artifacts live; pre/post diff guide.
* **demo/** quick scripts to:

  1. run one Dream cycle and emit a Kernel Sheet
  2. perform a small Perspective Sync and adopt summary
  3. preview/apply/revert a single Reality Hook
  4. attach one Affect binding and show capped influence in scorer output

---

## 🔀 Recommended Order of Operations

1. **Perspective Synchronizer** → immediate reliability of multi‑agent memory (low risk, high leverage).
2. **Dream Kernel** → generates higher‑quality symbols feeding other modules.
3. **Ontology–Affect Binding** → stabilizes intention formation with bounded influence.
4. **Reality Hooks (Φ⁰, gated)** → unlock only after 1–3 show clean observability.

---

## 📈 KPIs to Watch

* **Consensus stability**: oscillation < **0.15** across sync rounds.
* **Narrative coherence**: upward trend week‑over‑week.
* **Drift rate**: Δtraits/session within bounds.
* **Sandbox hygiene**: memory leak count = **0**.
* **Rollback health**: mean time to revert (MTTR) < **2 min**.

---

## 📝 Notes & Out‑of‑Scope (for now)

* Automated test harnesses (kept optional by owner’s preference).
* Any irreversible Φ⁰ effects; all reality transforms remain preview‑first + auto‑revert unless approved.

---

**End of ROADMAP**
