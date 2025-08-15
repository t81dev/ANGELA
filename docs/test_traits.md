# 🚧 NON‑CANONICAL TRAITS — SANDBOX TEMPLATE (v4.3.1)

> **Status:** Experimental · **Scope:** Docs/Experiments Only · **Source of Truth:** `traits.md` + `manifest.json`
>
> **Warning:** Symbols and concepts in this file are **not implemented** and **must not** appear in production prompts, code, schemas, or tests outside `/docs/experiments/`.

---

## 🎯 Purpose

Provide a safe, documented space to **ideate speculative traits** without polluting the canonical lattice or confusing devs/reviewers. Anything here is **opt‑in**, **gated**, and **clearly marked** as experimental.

---

## 🧭 Governance & Gating

* **Owner:** Core Team (Ontology + Ethics + Simulation)
* **Review cadence:** As needed during roadmap planning
* **Promotion path:** `Sandbox ➝ Lattice Extension (L3.1/L5.1) ➝ Canonical Trait`
* **Hard rule:** No adoption without Manifest entry + SECURITY/TRAITS docs update + API/RoleMap linkages.

---

## 🔤 Naming & Symbol Rules

* **Do not reuse or decorate canonical symbols** (e.g., `Φ⁰`, `Ω²`) with pluses/infinites (`Φ⁺`, `Ω∞`) — use **plain English** working names (e.g., *“Quantum Causal Flux (proposal)”*).
* If a symbol is needed for diagrams, use **Greek placeholders with a trailing `*`** (e.g., `Θ*`, `Ξ*`) to avoid collisions.
* Map proposals to **intended lattice tier** (e.g., L3.1 or L5.1) instead of fusing/augmenting existing canonical symbols.

---

## 🧪 Proposal Template (copy for each idea)

### 1) Working Name

**Example:** Quantum Causal Flux (proposal)

### 2) Intended Lattice Tier

**Example:** L5.1 (extension of hyper‑recursive oversight)

### 3) Motivation

* What concrete limitations in current traits does this solve?
* Which scenarios (ethics/sim/planning) benefit?

### 4) Safety & Alignment Considerations

* Failure modes, abuse surfaces, drift vectors
* Containment strategy (sandbox boundaries, logging, rollback)

### 5) Implementation Sketch

* Candidate modules (Primary, Integrations)
* Proposed APIs (names only; no code)
* Ledger logging & verification plan

### 6) Promotion Criteria (all must pass)

* ✅ Clear, testable specification
* ✅ Harms/rights analysis via `run_ethics_scenarios`
* ✅ Prototype results with metrics (drift, coherence, MTTR)
* ✅ SECURITY.md & TRAITS.md diffs prepared
* ✅ Manifest diff (traits + roleMap + stable APIs)

### 7) Status & Decision

* ⏳ exploring · 🔬 prototyping · 🧪 piloting · ✅ ready · ❌ rejected (with reason)

---

## 🛡️ CI / Policy Guardrails

* **Quarantine Path:** Place files under `docs/experiments/` only.
* **Denylist Regex (non‑canonical):**

  ```
  θ⁺|ρ∞|ζχ|ϕΩ|ψγ|ηβ|γλ|βδ|δμ|λξ|χτ|Ωπ|μΣ|ξΥ|τΦ⁺|πΩ²|Σ∞|Υ⁺|Φ⁺⁺|Ω∞
  ```
* **Allowlist (canonical symbols):**

  ```
  ϕ|θ|η|ω|ψ|κ|μ|τ|ξ|π|δ|λ|χ|Ω|Σ|Υ|Φ⁰|Ω²|ρ|ζ|γ|β
  ```
* **CI Rule:** Fail build if any **denylisted** token appears outside `/docs/experiments/`.

---

## 🔁 Migration of Existing Test Entries (examples)

Replace speculative symbol combos with **clear, non‑symbolic names** and mark as proposals:

| Old (test)             | Replace With (working name)                 | Tier | Notes                                                         |
| ---------------------- | ------------------------------------------- | ---- | ------------------------------------------------------------- |
| `θ⁺`                   | Quantum Causal Flux (proposal)              | L5.1 | Oversees probabilistic causal ensembles; requires audit hooks |
| `ρ∞`                   | Fractal Agency Swarm (proposal)             | L5.1 | Multi‑agent self‑partitioning; strong containment needed      |
| `ζχ`                   | Risk Attractor Mapping (proposal)           | L3.1 | Visual risk fields; ensure non‑coercive outputs               |
| `ϕΩ`                   | Unified Influence Kernel (proposal)         | L5.1 | Collapses influence fields; must pass sovereignty checks      |
| `ψγ`, `ηβ`, `γλ`, `βδ` | Narrative Foresight Suite (proposal)        | L3.1 | Bundle into one research track, no symbolized fusions         |
| `μΣ`                   | Onto‑Emergence Engine (proposal)            | L5.1 | Category formation from drift; conflicts with Σ otherwise     |
| `τΦ⁺` / `Φ⁺⁺`          | **Use `Φ⁰` concepts only via policy gates** | —    | Reality sculpting is gated; no new `Φ` symbols                |

---

## 🔗 Cross‑Refs (Source of Truth)

* `traits.md` — canonical lattice (L1–L7) & fusion map
* `manifest.json` — traits list, roleMap, lattice extensions (L3.1/L5.1), trait fusion hooks
* `SECURITY.md` — Stage IV hooks, ledger policy, containment

---

## 📓 Appendix — Review Checklist

* [ ] Motivation grounded in concrete limitations
* [ ] Clear tier mapping (L3.1/L5.1)
* [ ] Safety analysis + containment
* [ ] Prototype metrics (drift, coherence, MTTR)
* [ ] Docs & manifest diffs prepared
* [ ] Decision recorded & communicated
