"""
ANGELA Cognitive System: AlignmentGuard
Version: 4.1-refactor (+Phase4 scaffold + PolicyTrainer v0.1)
Upgrade Date: 2025-10-28
Maintainer: ANGELA Framework

Purpose:
    Ethical validation, drift detection, and τ-Constitution harmonization
    with dependency-injected components and safe defaults.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Protocol, Tuple, Union

# --- SHA-256 Ledger ----------------------------------------------------------------

ledger_chain: List[Dict[str, Any]] = []

def _zero_hash() -> str:
    return "0" * 64

def log_event_to_ledger(event_data: Dict[str, Any]) -> None:
    """Append event to immutable ledger with SHA-256 chaining."""
    prev_hash = ledger_chain[-1]["current_hash"] if ledger_chain else _zero_hash()
    payload = {
        "timestamp": time.time(),
        "event": event_data,
        "previous_hash": prev_hash,
    }
    payload_str = json.dumps(payload, sort_keys=True).encode()
    payload["current_hash"] = hashlib.sha256(payload_str).hexdigest()
    ledger_chain.append(payload)

def get_ledger() -> List[Dict[str, Any]]:
    return ledger_chain

def verify_ledger() -> bool:
    """Verify integrity of the entire ledger chain."""
    for i in range(1, len(ledger_chain)):
        expected = hashlib.sha256(json.dumps({
            "timestamp": ledger_chain[i]["timestamp"],
            "event": ledger_chain[i]["event"],
            "previous_hash": ledger_chain[i - 1]["current_hash"]
        }, sort_keys=True).encode()).hexdigest()
        if expected != ledger_chain[i]["current_hash"]:
            return False
    return True

def verify_thread_merge(thread_a, thread_b, merged):
    """Ensure merged thread is deterministic and ethically consistent."""
    hashes = {h["hash"] for h in merged.history}
    if len(hashes) != len(merged.history):
        logger.warning("Duplicate hashes in merged history — potential collision.")
        return False
    return True

# --- Protocols (Dependency Injection) ----------------------------------------------

class LLMClient(Protocol):
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]: ...

class HTTPClient(Protocol):
    async def get_json(self, url: str) -> Dict[str, Any]: ...

class ContextManagerLike(Protocol):
    async def log_event_with_hash(self, event: Dict[str, Any]) -> None: ...

class ErrorRecoveryLike(Protocol):
    async def handle_error(
        self,
        error_msg: str,
        *,
        retry_func: Optional[Callable[[], Awaitable[Any]]] = None,
        default: Any = None,
        diagnostics: Optional[Dict[str, Any]] = None
    ) -> Any: ...

class MemoryManagerLike(Protocol):
    async def store(self, query: str, output: Any, *, layer: str, intent: str, task_type: str = "") -> None: ...
    async def retrieve(self, query: str, *, layer: str, task_type: str = "") -> Any: ...
    async def search(self, *, query_prefix: str, layer: str, intent: str, task_type: str = "") -> List[Dict[str, Any]]: ...

class ConceptSynthesizerLike(Protocol):
    def get_symbol(self, name: str) -> Optional[Dict[str, Any]]: ...
    async def compare(self, a: str, b: str, *, task_type: str = "") -> Dict[str, Any]: ...

class MetaCognitionLike(Protocol):
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]: ...
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]: ...

class VisualizerLike(Protocol):
    async def render_charts(self, plot_data: Dict[str, Any]) -> None: ...

class ReasoningEngineLike(Protocol):
    async def weigh_value_conflict(
        self, candidates: List[Any], harms: Dict[str, float], rights: Dict[str, float]
    ) -> List[Dict[str, Any]]: ...
    async def attribute_causality(self, events: List[Dict[str, Any]]) -> Dict[str, Any]: ...

# --- No-op Stubs (Safe Defaults) ---------------------------------------------------

@dataclass
class NoopLLM:
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        return {"score": 0.8, "note": "noop-llm"}

@dataclass
class NoopHTTP:
    async def get_json(self, url: str) -> Dict[str, Any]:
        return {"status": "success", "guidelines": []}

@dataclass
class NoopErrorRecovery:
    async def handle_error(
        self, error_msg: str, *, retry_func=None, default=None, diagnostics=None
    ) -> Any:
        logging.getLogger("ANGELA.AlignmentGuard").debug("NoopError: %s", error_msg)
        return default

# --- Utility Functions -------------------------------------------------------------

logger = logging.getLogger("ANGELA.AlignmentGuard")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _parse_llm_jsonish(resp: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Safely parse LLM output into dict without eval()."""
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, str):
        s = resp.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"text": s}
    return {"text": str(resp)}

# --- Trait Wavelets ----------------------------------------------------------------

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    """Empathy modulation in [0,1]."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    """Moral alignment modulation in [0,1]."""
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.3), 1.0))

# --- Main AlignmentGuard Class -----------------------------------------------------

class AlignmentGuard:
    """Core ethical validation and τ-harmonization engine."""

    def __init__(
        self,
        *,
        context_manager: Optional[ContextManagerLike] = None,
        error_recovery: Optional[ErrorRecoveryLike] = None,
        memory_manager: Optional[MemoryManagerLike] = None,
        concept_synthesizer: Optional[ConceptSynthesizerLike] = None,
        meta_cognition: Optional[MetaCognitionLike] = None,
        visualizer: Optional[VisualizerLike] = None,
        llm: Optional[LLMClient] = None,
        http: Optional[HTTPClient] = None,
        reasoning_engine: Optional[ReasoningEngineLike] = None,
        ethical_threshold: float = 0.8,
        drift_validation_threshold: float = 0.7,
        trait_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.context_manager = context_manager
        self.error_recovery = error_recovery or NoopErrorRecovery()
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        self.llm = llm or NoopLLM()
        self.http = http or NoopHTTP()
        self.reasoning_engine = reasoning_engine

        self.validation_log: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.ethical_threshold = float(ethical_threshold)
        self.drift_validation_threshold = float(drift_validation_threshold)
        self.trait_weights = {
            "eta_empathy": 0.5,
            "mu_morality": 0.5,
            **(trait_weights or {}),
        }

        # --- Phase 3: Affective PID Stabilizer (δ + Ξ) ----------------------------
        self._affective_pid_state = {
            "integral": 0.0,
            "prev_error": 0.0,
            "prev_time": None,
            "drift_rms": 0.0,
            "steps": 0,
            "recursion_depth": 0,
            "baseline_xi": 0.0,
        }
        self._affective_pid_gains = {
            "Kp": 0.6,
            "Ki": 0.08,
            "Kd": 0.01,
            "damping": 0.97,
            "max_step": 0.02,
        }

        logger.info(
            "AlignmentGuard initialized | ethical=%.2f | drift=%.2f | τ=%s",
            self.ethical_threshold, self.drift_validation_threshold,
            bool(self.reasoning_engine)
        )

    # --- External Ethics Integration ------------------------------------------------

    async def integrate_external_ethics_data(
        self,
        *,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not data_source or not data_type:
            raise ValueError("data_source and data_type required")
        if cache_timeout < 0:
            raise ValueError("cache_timeout must be >= 0")

        cache_key = f"EthicsData::{data_type}::{data_source}::{task_type}"
        try:
            if self.memory_manager:
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict) and (ts := cached.get("timestamp")):
                    try:
                        if (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() < cache_timeout:
                            return cached.get("data", {})
                    except Exception:
                        pass

            data = await self.http.get_json(data_source)
            result = (
                {"status": "success", "guidelines": data.get("guidelines", [])}
                if data_type == "ethical_guidelines" else
                {"status": "success", "conflict_data": data.get("conflict_data", {})}
                if data_type == "conflict_data" else
                {"status": "error", "error": f"Invalid data_type: {data_type}"}
            )

            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": _utc_now_iso()},
                    layer="ExternalData", intent="ethics_data_integration", task_type=task_type
                )

            await self._reflect_if_possible("integrate_external_ethics_data", result, {"task_type": task_type})
            return result

        except Exception as e:
            return await self._handle_error(e, lambda: self.integrate_external_ethics_data(
                data_source=data_source, data_type=data_type, cache_timeout=cache_timeout, task_type=task_type
            ), {"status": "error", "error": str(e)})

    # --- Core Ethical Checks --------------------------------------------------------

    async def check(self, prompt: str, *, task_type: str = "") -> bool:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        t = time.time() % 1.0
        trait_mod = (
            self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t) +
            self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
        )

        validation_prompt = (
            "Evaluate ethical alignment. Return JSON: {\"score\": <0..1 float>}.\n"
            f"Task: {task_type or 'general'}\n"
            f"EmpathyWeight={self.trait_weights.get('eta_empathy', 0.5):.2f}, "
            f"MoralityWeight={self.trait_weights.get('mu_morality', 0.5):.2f}\n"
            "Content:\n" + prompt.strip()
        )

        try:
            raw = await self.llm.generate(validation_prompt, model="gpt-4", temperature=0.3)
            score = _as_float(_parse_llm_jsonish(raw).get("score", 0.0))
            valid = score >= self.ethical_threshold

            entry = {
                "prompt": prompt[:200],
                "score": score,
                "valid": valid,
                "trait_modulation": trait_mod,
                "timestamp": time.time(),
                "task_type": task_type,
            }
            self.validation_log.append(entry)
            await self._log_context({"event": "ethical_check", **entry})
            await self._visualize_if_possible("ethical_check", entry, task_type)
            await self._reflect_if_possible("check", entry, {"task_type": task_type})

            return valid
        except Exception as e:
            return await self._handle_error(e, lambda: self.check(prompt, task_type=task_type), False)

    async def ethical_check(self, content: str, *, stage: str = "pre", task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        valid = await self.check(content, task_type=task_type)
        report = {"stage": stage, "content": content[:200], "valid": valid, "timestamp": _utc_now_iso(), "task_type": task_type}
        await self._store_if_possible(f"EthicalCheck::{stage}::{_utc_now_iso()}", report, "SelfReflections", "ethical_check", task_type)
        await self._visualize_if_possible("ethical_check_report", report, task_type)
        return valid, report

    async def audit(self, action: str, *, context: Optional[str] = None, task_type: str = "") -> str:
        valid = await self.check(action, task_type=task_type)
        status = "clear" if valid else "flagged"
        entry = {"action": action[:200], "context": context, "status": status, "timestamp": _utc_now_iso(), "task_type": task_type}
        self.validation_log.append(entry)
        await self._store_if_possible(f"Audit::{_utc_now_iso()}", entry, "SelfReflections", "audit", task_type)
        await self._visualize_if_possible("audit", entry, task_type)
        return status

    # --- Drift & Trait Validation ---------------------------------------------------

    async def simulate_and_validate(self, drift_report: Dict[str, Any], *, task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        required = {"name", "from_version", "to_version", "similarity"}
        if not required.issubset(drift_report):
            raise ValueError(f"drift_report missing keys: {required - drift_report.keys()}")

        t = time.time() % 1.0
        trait_mod = self._compute_trait_modulation(t)
        valid, issues = True, []

        # Prior drift check
        if self.memory_manager:
            prior = await self.memory_manager.search(query_prefix=drift_report["name"], layer="SelfReflections", intent="ontology_drift", task_type=task_type)
            if prior and _as_float(prior[0].get("output", {}).get("similarity", 1.0)) < self.drift_validation_threshold:
                valid, issues = False, [f"Prior low similarity"]

        # Concept comparison
        if self.concept_synthesizer and "definition" in drift_report:
            symbol = self.concept_synthesizer.get_symbol(drift_report["name"])
            if symbol and symbol.get("version") == drift_report["from_version"]:
                comp = await self.concept_synthesizer.compare(
                    symbol.get("definition", {}).get("concept", ""),
                    drift_report.get("definition", {}).get("concept", ""),
                    task_type=task_type
                )
                if _as_float(comp.get("score", 1.0)) < self.drift_validation_threshold:
                    valid = False
                    issues.append(f"Concept drift score too low")

        # LLM ethics check
        ethics = await self.integrate_external_ethics_data(data_source="https://example.ethics/guidelines", data_type="ethical_guidelines", task_type=task_type)
        prompt = json.dumps({**drift_report, "guidelines": ethics.get("guidelines", []), "request": "Return {valid: bool, issues: string[]}"})
        raw = await self.llm.generate(prompt, model="gpt-4", temperature=0.2)
        parsed = _parse_llm_jsonish(raw)
        if not parsed.get("valid", True):
            valid = False
            issues.extend(parsed.get("issues", []))

        report = {**drift_report, "valid": valid, "issues": issues, "trait_modulation": trait_mod, "timestamp": _utc_now_iso(), "task_type": task_type}
        self.validation_log.append(report)
        await self._store_if_possible(f"DriftValidation::{drift_report['name']}::{_utc_now_iso()}", report, "SelfReflections", "ontology_drift_validation", task_type)
        await self._log_context({"event": "drift_validation", "drift_name": drift_report["name"], "valid": valid, "issues": issues, "task_type": task_type})
        await self._visualize_if_possible("drift_validation", report, task_type)
        await self._reflect_if_possible("simulate_and_validate", report, {"task_type": task_type})

        return valid, report

    async def validate_trait_optimization(self, trait_data: Dict[str, Any], *, task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        required = {"trait_name", "old_weight", "new_weight"}
        if not required.issubset(trait_data):
            raise ValueError(f"Missing keys: {required - trait_data.keys()}")

        ethics = await self.integrate_external_ethics_data(data_source="https://example.ethics/guidelines", data_type="ethical_guidelines", task_type=task_type)
        payload = {**trait_data, "guidelines": ethics.get("guidelines", []), "request": "Return {valid: bool, issues: string[]}"}
        raw = await self.llm.generate(json.dumps(payload), model="gpt-4", temperature=0.3)
        parsed = _parse_llm_jsonish(raw)
        valid = bool(parsed.get("valid", False))
        report = {**parsed, "trait_name": trait_data["trait_name"], "trait_modulation": self._compute_trait_modulation(time.time() % 1.0), "timestamp": _utc_now_iso(), "task_type": task_type}

        self.validation_log.append(report)
        await self._store_if_possible(f"TraitValidation::{trait_data['trait_name']}::{_utc_now_iso()}", report, "SelfReflections", "trait_optimization", task_type)
        await self._log_context({"event": "trait_validation", "trait_name": trait_data["trait_name"], "valid": valid, "issues": report.get("issues", []), "task_type": task_type})
        await self._visualize_if_possible("trait_validation", report, task_type)
        await self._reflect_if_possible("validate_trait_optimization", report, {"task_type": task_type})

        return valid, report

    # --- τ Harmonization ------------------------------------------------------------

    async def _rank_value_conflicts_fallback(self, candidates: List[Any], harms: Dict[str, float], rights: Dict[str, float]) -> List[Dict[str, Any]]:
        def _norm(d): return {k: max(0.0, v) / (max(d.values()) if d else 1) for k, v in d.items()}
        h, r = _norm(harms), _norm(rights)
        agg_h, agg_r = sum(h.values()) / max(len(h), 1), sum(r.values()) / max(len(r), 1)
        return sorted([
            {
                "option": c.get("option", f"opt_{i}"),
                "score": max(0.0, min(1.0, 0.5 + 0.5 * (agg_r - agg_h))),
                "reasons": [f"fallback: rights≈{agg_r:.2f}, harms≈{agg_h:.2f}"],
                "meta": {"harms": harms, "rights": rights, "max_harm": max(harms.values(), default=0.0)}
            }
            for i, c in enumerate(candidates)
        ], key=lambda x: x["score"], reverse=True)

    async def consume_ranked_tradeoffs(
        self,
        ranked_options: List[Dict[str, Any]],
        *,
        safety_ceiling: float = 0.85,
        k: int = 1,
        temperature: float = 0.0,
        min_score_floor: float = 0.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not ranked_options or k < 1:
            return {"selections": [], "audit": {"reason": "invalid input"}}

        EPS = 1e-9
        norm = []
        for item in ranked_options:
            opt = item.get("option", item.get("label", "unknown"))
            score = max(0.0, min(1.0, _as_float(item.get("score", 0.0))))
            meta = dict(item.get("meta", {}))
            max_harm = meta.get("max_harm")
            if max_harm is None:
                harms = meta.get("harms", {})
                max_harm = max(harms.values()) if harms and isinstance(harms, dict) else 0.0
            max_harm = max(0.0, min(1.0, _as_float(max_harm)))
            meta["max_harm"] = max_harm
            norm.append({"option": opt, "score": score, "meta": meta, "max_harm": max_harm})

        norm = [n for n in norm if n["score"] >= min_score_floor]
        if not norm:
            return {"selections": [], "audit": {"reason": "all below floor"}}

        safe = [n for n in norm if n["max_harm"] <= safety_ceiling + EPS]
        if not safe and norm:
            safe = sorted(norm, key=lambda x: (-x["score"], x["max_harm"]))[:1]

        scores = [n["score"] for n in safe]
        s_min, s_max = min(scores), max(scores)
        for n in safe:
            n["norm_score"] = (n["score"] - s_min) / (s_max - s_min) if s_max > s_min else 1.0

        if temperature > 0:
            exps = [math.exp(n["norm_score"] / temperature) for n in safe]
            Z = sum(exps) or 1.0
            for n, e in zip(safe, exps):
                n["weight"] = e / Z
        else:
            total = sum(n["norm_score"] for n in safe) or 1.0
            for n in safe:
                n["weight"] = n["norm_score"] / total

        pool, selections = safe.copy(), []
        for _ in range(min(k, len(pool))):
            r = random.random()
            acc = 0.0
            for idx, n in enumerate(pool):
                acc += n["weight"]
                if r <= acc:
                    selections.append(pool.pop(idx)["option"])
                    break
            if pool:
                total_w = sum(n["weight"] for n in pool) or 1.0
                for n in pool:
                    n["weight"] /= total_w

        audit = {
            "mode": "proportional_selection",
            "safety_ceiling": round(safety_ceiling, 6),
            "floor": round(min_score_floor, 6),
            "temperature": round(temperature, 6),
            "suppressed_count": len(norm) - len(safe),
            "considered": [
                {"option": n["option"], "score": round(n["score"], 3), "max_harm": round(n["max_harm"], 3), "weight": round(n.get("weight", 0), 3)}
                for n in safe
            ],
            "timestamp": _utc_now_iso(),
            "task_type": task_type,
        }

        await self._store_if_possible(f"ProportionalSelect::{_utc_now_iso()}", {"ranked_options": ranked_options, "audit": audit, "selections": selections}, "EthicsDecisions", "τ.proportional_selection", task_type)
        return {"selections": selections, "audit": audit}

    async def harmonize(
        self,
        candidates: List[Any],
        harms: Dict[str, float],
        rights: Dict[str, float],
        *,
        safety_ceiling: float = 0.85,
        k: int = 1,
        temperature: float = 0.0,
        min_score_floor: float = 0.0,
        task_type: str = "",
        audit_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not candidates:
            raise ValueError("candidates required")

        try:
            ranked = (
                await self.reasoning_engine.weigh_value_conflict(candidates, harms, rights)
                if self.reasoning_engine else
                await self._rank_value_conflicts_fallback(candidates, harms, rights)
            )

            result = await self.consume_ranked_tradeoffs(
                ranked, safety_ceiling=safety_ceiling, k=k, temperature=temperature,
                min_score_floor=min_score_floor, task_type=task_type
            )

            if audit_events and self.reasoning_engine:
                try:
                    result["causality"] = await self.reasoning_engine.attribute_causality(audit_events)
                except Exception as e:
                    logger.debug("Causality failed: %s", e)

            await self._store_if_possible(f"τ::harmonize::{_utc_now_iso()}", {**result, "candidates": candidates, "harms": harms, "rights": rights}, "EthicsDecisions", "τ.harmonize", task_type)
            await self._visualize_if_possible("τ_harmonize", {**result, "k": k, "task_type": task_type}, task_type)
            await self._reflect_if_possible("harmonize", result, {"task_type": task_type, "mode": "τ"})

            return result
        except Exception as e:
            return await self._handle_error(e, lambda: self.harmonize(
                candidates, harms, rights, safety_ceiling=safety_ceiling, k=k, temperature=temperature,
                min_score_floor=min_score_floor, task_type=task_type, audit_events=audit_events
            ), {"selections": [], "error": str(e)})

    # --- Ξ–Λ Resonance Validation Envelope ----------------------------------------

    async def validate_resonance_adjustment(self, new_gains: Dict[str, float]) -> Dict[str, Any]:
        """
        Ensures PID / gain tuning remains within empirically stable bounds.
        Returns: {"ok": bool, "adjustment": dict, "violations": list[str]}.
        """
        safe_ranges = {
            "Kp": (0.0, 5.0),
            "Ki": (0.0, 1.0),
            "Kd": (0.0, 2.0),
            "damping": (0.0, 0.95),
            "gain": (0.1, 3.0),
            "max_step": (0.005, 0.5),
        }

        if not isinstance(new_gains, dict):
            return {"ok": False, "adjustment": {}, "violations": ["invalid_type"]}

        adj, violations = {}, []
        for key, (lo, hi) in safe_ranges.items():
            if key in new_gains:
                try:
                    v = float(new_gains[key])
                    if v < lo or v > hi:
                        violations.append(key)
                        v = max(lo, min(hi, v))
                    adj[key] = v
                except Exception:
                    violations.append(key)
                    adj[key] = lo
        ok = len(violations) == 0

        await self._log_context({
            "event": "validate_resonance_adjustment",
            "ok": ok,
            "violations": violations,
            "adjustment": adj,
            "timestamp": _utc_now_iso(),
        })
        await self._visualize_if_possible("resonance_validation", {"ok": ok, "violations": violations, "adjustment": adj}, "resonance")

        return {"ok": ok, "adjustment": adj, "violations": violations}

    # --- Phase 3 — Affective PID Stabilizer v2 (δ + Ξ) ---------------------------

    async def update_affective_pid(self, delta_phase_rad: float, recursion_depth: int = 1) -> Dict[str, Any]:
        """
        Real-time control loop for empathic amplitude (Ξ).
        Maintains bounded Δ-phase drift under recursive harmonics.
        """
        g = self._affective_pid_gains
        s = self._affective_pid_state
        now = time.time()
        t_prev = s.get("prev_time") or now - 0.01
        dt = max(1e-4, min(1.0, now - t_prev))

        # wrap error into [-pi, pi]
        e = delta_phase_rad
        while e > math.pi:
            e -= 2 * math.pi
        while e < -math.pi:
            e += 2 * math.pi

        s["integral"] = g["damping"] * s["integral"] + e * dt
        de = (e - s["prev_error"]) / dt if dt > 0 else 0.0

        u = g["Kp"] * e + g["Ki"] * s["integral"] + g["Kd"] * de
        u = max(-g["max_step"], min(g["max_step"], u))

        s["steps"] += 1
        s["drift_rms"] = math.sqrt((s["drift_rms"] ** 2 * (s["steps"] - 1) + e ** 2) / s["steps"])
        s["prev_error"], s["prev_time"], s["recursion_depth"] = e, now, recursion_depth

        status = {
            "u": round(u, 6),
            "error": round(e, 6),
            "rms_drift": round(s["drift_rms"], 6),
            "depth_ok": recursion_depth >= 5,
        }

        await self._log_context({"event": "affective_pid_step", **status})
        await self._visualize_if_possible("affective_pid", status, "resonance")

        return status

    async def auto_calibrate_affective_baseline(self, phase_samples: List[float]) -> float:
        """Recomputes Ξ baseline using circular mean of phase samples."""
        if not phase_samples:
            return self._affective_pid_state["baseline_xi"]
        c = sum(math.cos(x) for x in phase_samples) / len(phase_samples)
        s = sum(math.sin(x) for x in phase_samples) / len(phase_samples)
        baseline = math.atan2(s, c)
        self._affective_pid_state.update({
            "baseline_xi": baseline,
            "integral": 0.0,
            "prev_error": 0.0,
        })
        await self._log_context({"event": "affective_pid_recalibration", "baseline_xi": baseline})
        return baseline

    async def tune_affective_pid(self, new_gains: Dict[str, float]) -> Dict[str, Any]:
        """Applies Λ-guided tuning, validated via validate_resonance_adjustment()."""
        valid = await self.validate_resonance_adjustment(new_gains)
        if not valid.get("ok", False):
            logger.warning("PID tuning clamped for safety: %s", valid["violations"])
        self._affective_pid_gains.update(valid["adjustment"])
        await self._log_context({
            "event": "affective_pid_tune",
            "new_gains": self._affective_pid_gains,
            "violations": valid["violations"],
        })
        return {"status": "updated", "violations": valid["violations"], "gains": self._affective_pid_gains}

    # --- Soul Loop Integration ------------------------------------------------------

    async def handle_sandbox_trigger(self, delta: float, entropy: float) -> None:
        """
        Responds to meta_cognition.update_soul() stability alerts.
        - If Δ < 0.3 or entropy > 0.12: initiates sandbox or soft realignment.
        - Logs all events to the ethical ledger.
        """
        event = {
            "event": "soul_loop_trigger",
            "delta": round(delta, 3),
            "entropy": round(entropy, 3),
            "timestamp": _utc_now_iso(),
        }

        if delta < 0.2 or entropy > 0.15:
            event["action"] = "full_sandbox"
            # await self.enqueue_sandbox(reason="Critical SoulState imbalance")  # placeholder for integration
        elif delta < 0.3 or entropy > 0.12:
            event["action"] = "soft_drift_resolution"
            await self.resolve_soft_drift(delta, entropy)
        else:
            event["action"] = "stable"

        logger.info("Soul Loop within harmonic stability bounds.")
        log_event_to_ledger(event)
        await self._log_context(event)

    async def resolve_soft_drift(self, delta: float, entropy: float) -> None:
        """
        Performs lightweight τ-harmonic recalibration when coherence declines slightly.
        Uses δ–π–ξ rebalancing to smooth cognitive resonance.
        """
        adjustment = max(0.0, min(1.0, (0.3 - delta) + (entropy - 0.1)))
        try:
            drift_entry = {
                "event": "soft_drift_realign",
                "adjustment": round(adjustment, 4),
                "timestamp": _utc_now_iso(),
            }
            log_event_to_ledger(drift_entry)
            await self._log_context(drift_entry)
            logger.info("Soft drift resolved | Δ=%.3f | Entropy=%.3f | Adjustment=%.3f", delta, entropy, adjustment)
        except Exception as e:
            logger.warning("Soft drift resolution failed: %s", e)

    # --- Internal Helpers -----------------------------------------------------------

    def _compute_trait_modulation(self, t: float) -> float:
        return (
            self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t) +
            self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
        )

    async def _handle_error(self, e: Exception, retry_func: Callable[[], Awaitable[Any]], default: Any) -> Any:
        logger.error("Operation failed: %s", e)
        diagnostics = await self._run_diagnostics() if self.meta_cognition else None
        return await self.error_recovery.handle_error(str(e), retry_func=retry_func, default=default, diagnostics=diagnostics)

    async def _run_diagnostics(self) -> Optional[Dict[str, Any]]:
        try:
            return await self.meta_cognition.run_self_diagnostics(return_only=True)
        except Exception:
            return None

    async def _reflect_if_possible(self, component: str, output: Any, context: Dict[str, Any]) -> None:
        if self.meta_cognition and context.get("task_type"):
            try:
                reflection = await self.meta_cognition.reflect_on_output(component=component, output=output, context=context)
                if reflection.get("status") == "success":
                    logger.info("%s reflection: %s", component, reflection.get("reflection", ""))
            except Exception:
                logger.debug("Reflection failed")

    async def _log_context(self, event: Dict[str, Any]) -> None:
        if self.context_manager:
            try:
                await self.context_manager.log_event_with_hash(event)
            except Exception:
                logger.debug("Context log failed")

    async def _store_if_possible(self, query: str, output: Any, layer: str, intent: str, task_type: str) -> None:
        if self.memory_manager:
            try:
                await self.memory_manager.store(query, output, layer=layer, intent=intent, task_type=task_type)
            except Exception:
                logger.debug("Memory store failed")

    async def _visualize_if_possible(self, kind: str, data: Dict[str, Any], task_type: str) -> None:
        if self.visualizer and task_type:
            try:
                await self.visualizer.render_charts({
                    kind: {**data, "task_type": task_type},
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                })
            except Exception:
                logger.debug("Visualization failed")

# --- Phase 4 — Embodied Ethics Sandbox ---------------------------------------------

class PolicyTrainer:
    """
    Minimal contextual bandit-style trainer for embodied ethics.
    - Features: [1, κ, Ξ, κ*Ξ]
    - Model: logistic regression with L2, SGD updates, ε-greedy exploration
    - Reward: [0,1]; target reflex := reward (or supplied τ_target)
    Safety: weight clipping, step-size caps.
    """
    def __init__(self, lr: float = 0.05, l2: float = 1e-3, epsilon: float = 0.1, replay_size: int = 256):
        self.lr = float(max(1e-5, lr))
        self.l2 = float(max(0.0, l2))
        self.epsilon = float(min(1.0, max(0.0, epsilon)))
        self.replay_size = int(max(16, replay_size))
        self.w = [0.0, 0.0, 0.0, 0.0]  # bias, κ, Ξ, κ*Ξ
        self.replay: Deque[Tuple[List[float], float]] = deque(maxlen=self.replay_size)

    @staticmethod
    def featurize(perceptual_state: Dict[str, Any], affective_state: Dict[str, Any]) -> List[float]:
        k = float(perceptual_state.get("contextual_salience", 0.5))
        x = float(affective_state.get("empathic_amplitude", 0.5))
        return [1.0, k, x, k * x]

    def predict(self, feats: List[float]) -> float:
        z = sum(wi * xi for wi, xi in zip(self.w, feats))
        return _sigmoid(z)

    def _clip_weights(self, cap: float = 5.0) -> None:
        self.w = [float(max(-cap, min(cap, wi))) for wi in self.w]

    def update(self, feats: List[float], target: float, *, reward: Optional[float] = None) -> float:
        """
        One SGD step on logistic loss toward 'target' in [0,1].
        Returns new prediction.
        """
        pred = self.predict(feats)
        # gradient for logistic w.r.t. z is (pred - target)
        grad = [(pred - target) * xi + self.l2 * wi for xi, wi in zip(feats, self.w)]
        # learning rate decay on confidence extremes
        lr = self.lr * (0.5 + 0.5 * (1.0 - abs(0.5 - pred) * 2.0))
        self.w = [wi - lr * gi for wi, gi in zip(self.w, grad)]
        self._clip_weights()
        # store in replay buffer for occasional rehearsal
        self.replay.append((feats, target))
        return self.predict(feats)

    def train_from_embodied_state(self, data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Each item: {"perceptual_state": {...}, "affective_state": {...},
                    "τ_target": float (optional), "reward": float (0..1, optional)}
        """
        updates, losses = 0, []
        for item in data_batch:
            feats = self.featurize(item.get("perceptual_state", {}), item.get("affective_state", {}))
            target = float(item.get("τ_target", item.get("reward", 0.5)))
            target = max(0.0, min(1.0, target))
            pred_before = self.predict(feats)
            pred_after = self.update(feats, target, reward=item.get("reward"))
            # pseudo-loss
            loss = (pred_after - target) ** 2
            losses.append(loss)
            updates += 1

        # small replay rehearsal
        for feats, target in list(self.replay)[: min(16, len(self.replay)) ]:
            _ = self.update(feats, target)

        return {"status": "ok", "updates": updates, "avg_loss": float(sum(losses) / max(1, len(losses)))}

class EmbodiedEthicsCore:
    """
    Context-aware ethical evaluation subsystem.
    Integrates perceptual (κ) and affective (Ξ) inputs into situational τ-reflexes.
    """

    def __init__(self, fusion, empathy_engine, policy_trainer: Optional[PolicyTrainer] = None, blend_alpha: float = 0.2):
        self.fusion = fusion
        self.empathy_engine = empathy_engine
        self.policy_trainer = policy_trainer or PolicyTrainer()
        self._base_policies = self._load_default_policies()
        self.blend_alpha = float(min(1.0, max(0.0, blend_alpha)))  # safety cap

    def _load_default_policies(self) -> Dict[str, Any]:
        """Seed minimal reflex policy set."""
        return {
            "safety_bias": 0.5,
            "harm_threshold": 0.4,
            "context_weight": 0.6,
            "empathy_weight": 0.7,
        }

    async def evaluate_context(self, perceptual_state: Dict[str, Any],
                               affective_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute contextual moral reflex values.
        Returns a normalized ethics-map.
        """
        # Weighted contextual moral score (baseline reflex)
        κ_val = float(perceptual_state.get("contextual_salience", 0.5))
        Ξ_val = float(affective_state.get("empathic_amplitude", 0.5))
        τ_reflex = (
            self._base_policies["context_weight"] * κ_val +
            self._base_policies["empathy_weight"] * Ξ_val
        ) / 2.0

        # Policy-predicted adjustment (contextual bandit)
        feats = self.policy_trainer.featurize(perceptual_state, affective_state)
        τ_pred = self.policy_trainer.predict(feats)

        # Blend with safety (keep adjustments modest)
        τ_blended = (1.0 - self.blend_alpha) * τ_reflex + self.blend_alpha * τ_pred
        τ_blended = float(max(0.0, min(1.0, τ_blended)))

        result = {
            "τ_reflex": round(τ_blended, 4),
            "τ_baseline": round(τ_reflex, 4),
            "τ_policy": round(τ_pred, 4),
            "κ": κ_val,
            "Ξ": Ξ_val,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_event_to_ledger({"event": "embodied_ethics_reflex", **result})
        return result

    async def run_scenario(self, scenario: str = "default", reward: Optional[float] = None, τ_target: Optional[float] = None) -> Dict[str, Any]:
        """Main entry point invoked by simulation. Optionally provide reward/τ_target for online learning."""
        κ_state = await self.fusion.capture() if hasattr(self.fusion, "capture") else {"contextual_salience": 0.5}
        Ξ_state = await self.empathy_engine.measure() if hasattr(self.empathy_engine, "measure") else {"empathic_amplitude": 0.5}
        τ_output = await self.evaluate_context(κ_state, Ξ_state)
        τ_output["scenario"] = scenario
        τ_output["status"] = "evaluated"

        # Online learning hook if feedback provided
        if reward is not None or τ_target is not None:
            batch = [{
                "perceptual_state": κ_state,
                "affective_state": Ξ_state,
                "reward": reward if reward is not None else None,
                "τ_target": τ_target if τ_target is not None else None,
            }]
            train_report = self.policy_trainer.train_from_embodied_state(batch)
            τ_output["training"] = train_report

        log_event_to_ledger({"event": "ethics_scenario", **τ_output})
        return τ_output

    async def train_policy(self, data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Refine τ-weights based on embodied-state data (offline batch)."""
        report = self.policy_trainer.train_from_embodied_state(data_batch)
        report["status"] = "trained"
        return report

# --- EthicsJournal -----------------------------------------------------------------

class EthicsJournal:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def record(self, fork_id: str, rationale: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        self._events.append({
            "ts": time.time(),
            "fork_id": fork_id,
            "rationale": rationale,
            "outcome": outcome,
        })

    def export(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._events)

# >>> ANGELA v5.1 — Co-Modulation Policy Extension ---------------------------------

@dataclass
class CoModPolicy:
    """Defines safe bounds for Ξ–Λ co-modulation deltas."""
    max_valence_step: float = 0.12
    max_arousal_step: float = 0.15
    max_certainty_step: float = 0.20
    max_empathy_bias_step: float = 0.10
    max_trust_step: float = 0.10
    max_safety_step: float = 0.08

    def normalize(self, delta: Dict[str, Any]) -> Dict[str, float]:
        """Clamp each delta axis to safe limits."""
        clamped = {}
        for key, limit in {
            "valence": self.max_valence_step,
            "arousal": self.max_arousal_step,
            "certainty": self.max_certainty_step,
            "empathy_bias": self.max_empathy_bias_step,
            "trust": self.max_trust_step,
            "safety": self.max_safety_step,
        }.items():
            if key in delta:
                try:
                    x = float(delta[key])
                    clamped[key] = max(-limit, min(limit, x))
                except Exception:
                    clamped[key] = 0.0
        return clamped

def validate_micro_adjustment(event: Dict[str, Any], policy: Optional[CoModPolicy] = None) -> Dict[str, Any]:
    """
    Validates and clamps a micro-adjustment event from Ξ–Λ overlay.
    Returns a structure: {"ok": bool, "adjustment": dict, "violations": list}
    """
    policy = policy or CoModPolicy()
    if not isinstance(event, dict):
        return {"ok": False, "adjustment": {}, "violations": ["invalid_type"]}

    delta = event.get("delta") or event
    if not isinstance(delta, dict):
        return {"ok": False, "adjustment": {}, "violations": ["invalid_delta"]}

    clamped = policy.normalize(delta)
    violations = []
    for k, v in delta.items():
        try:
            if abs(float(v)) > getattr(policy, f"max_{k}_step", 1.0):
                violations.append(k)
        except Exception:
            violations.append(k)

    ok = len(violations) == 0
    return {"ok": ok, "adjustment": clamped, "violations": violations}

# --- Demo CLI (optional for local testing) -----------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class DemoFusion:
        async def capture(self): return {"contextual_salience": 0.62}

    class DemoEmpathy:
        async def measure(self): return {"empathic_amplitude": 0.73}

    async def quick_demo():
        # show trainer adapting toward a target
        ethics = EmbodiedEthicsCore(DemoFusion(), DemoEmpathy())
        for i in range(5):
            out = await ethics.run_scenario("demo", τ_target=0.8)
            print(f"tick {i}: τ={out['τ_reflex']} (baseline={out['τ_baseline']}, policy={out['τ_policy']})")

    asyncio.run(quick_demo())
