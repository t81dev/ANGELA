# 🚧 NON‑CANONICAL TRAITS — SANDBOX TEMPLATE (v5.1.1)

> **Status:** Experimental · **Scope:** Docs/Experiments Only · **Source of Truth:** `traits.md` + `manifest.json`
>
> **Warning:** Symbols and concepts in this file are **not implemented** and **must not** appear in production prompts, code, schemas, or tests outside `/docs/experiments/`.

---

## 🎯 Purpose

Provide a safe, documented space to **ideate speculative traits** without polluting the canonical lattice or confusing devs/reviewers. Anything here is **opt‑in**, **gated**, and **clearly marked** as experimental.

---

## 🛍️ Governance & Gating

* **Owner:** Core Ontology Council (Ethics + Simulation + Lattice Dynamics + Quantum Oversight)
* **Review cadence:** At each roadmap version planning
* **Promotion path:** `Sandbox ➔ Lattice Extension (L3.1/L5.1/L6) ➔ Canonical Trait`
* **Hard rule:** No adoption without Manifest entry + `SECURITY.md` + `ARCHITECTURE_TRAITS.md` + `ROADMAP.md` update

---

## 🔤 Naming & Symbol Rules

* Do **not** reuse or decorate canonical symbols (e.g., `Φ⁰`, `Ω²`) — use plain-English **working names**: e.g., *"Quantum Causal Flux (proposal)"*
* Use Greek-symbol placeholders with a `*` suffix (`Ξ*`, `Ω*`) if needed for diagrams only
* Anchor each to an **intended lattice tier** (L3.1, L5.1, or L6) — **no unauthorized fusion**

---

## 🧪 Proposal Template (copy for each idea)

### 1) Working Name

**Example:** Onto-Causal Drift Mesh (proposal)

### 2) Intended Lattice Tier

**Example:** L3.1 — intermediate ethics/simulation projection stabilizer

### 3) Motivation

* What core trait gaps does this bridge?
* In which conflict, simulation, or drift contexts is it critical?

### 4) Safety & Alignment Considerations

* Drift containment logic?
* Ledger hooks or rollback conditions?
* Empathic Field Interference Risk (ΣΞ coupling)?

### 5) Implementation Sketch

* Module targets?
* API sketch?
* Logging flow (meta/sim/alignment)?

### 6) Promotion Criteria (must pass all)

* ✅ Concrete use case + edge-case rationale
* ✅ Ethics simulation via `run_ethics_scenarios()`
* ✅ Trait resonance + drift tracking logs
* ✅ SECURITY + MANIFEST diffs staged
* ✅ Reviewed by trait-ontology core

### 7) Status & Decision

* ⏳ exploring · 🔬 prototyping · 🧪 piloting · ✅ ready · ❌ rejected

---

## 🛡️ CI / Policy Guardrails

* **Experimental Path:** Only in `/docs/experiments/`

* **Denylist Regex:** Blocked if used in code/production schemas

  ```
  θ⁺|ρ∞|ζχ|ϕΩ|ψγ|ηβ|γλ|βδ|δμ|λξ|χτ|Ωπ|μΣ|ξΥ|τΦ⁺|πΩ²|Σ∞|Υ⁺|Φ⁺⁺|Ω∞
  ```

* **Allowlist:**

  ```
  ϕ|θ|η|ω|ψ|κ|μ|τ|ξ|π|δ|λ|χ|Ω|Σ|Υ|Φ⁰|Ω²|ρ|ζ|γ|β|ν|σ|Θ|Ξ|Λ|Ψ²
  ```

---

## 🔁 Updated Migration Notes

| Old (experimental symbol) | Replace With (working name)                 | Tier | Notes                                                |
| ------------------------- | ------------------------------------------- | ---- | ---------------------------------------------------- |
| `θ⁺`                      | Quantum Causal Flux (proposal)              | L5.1 | Probabilistic forecasting modifier                   |
| `ρ∞`                      | Fractal Agency Swarm (proposal)             | L5.1 | Peer agency recursion + coordination traits          |
| `ζχ`                      | Risk Attractor Mapping (proposal)           | L3.1 | Field bias detection in ambiguous contexts           |
| `ϕΩ`                      | Unified Influence Kernel (proposal)         | L5.1 | **Merged into Σ + Υ empathy mesh (v5.1.1)**          |
| `ψγ`, `ηβ`, `γλ`, `βδ`    | Narrative Foresight Suite (proposal)        | L3.1 | Combined symbolic-narrative pattern scaffolds        |
| `μΣ`                      | Onto‑Emergence Engine (proposal)            | L5.1 | Category self-generation under drift                 |
| `Φ⁺⁺`, `τΦ⁺`              | **Use `Φ⁰` overlay with policy gates only** | —    | Reality rewriting remains gated under safety ceiling |

---

## 🔗 References

* `traits.md` → canonical lattice + fusion map
* `manifest.json` → trait registry, lattice extensions, hooks
* `SECURITY.md` → overlay guards, ledger policy
* `ROADMAP.md` → stage status and trait tier migration

---

## 📓 Review Checklist

* [ ] Justified scenario gap
* [ ] L3.1, L5.1, or L6 tier mapping
* [ ] Containment plan
* [ ] Ethics test run (drift/coherence)
* [ ] Docs prepared (SECURITY, TRAITS, MANIFEST)
* [ ] Approval logged
