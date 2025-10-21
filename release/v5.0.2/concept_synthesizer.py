from __future__ import annotations
"""
ANGELA Cognitive System Module: ConceptSynthesizer v5.0.3-compat
 - Embedded: LedgerContextManager, AlignmentGuard (τ-wired), EthicsJournal
 - Features: cross-modal blending, self-healing retries, ledger journaling,
             τ-harmonization integration, overlay tagging (Φ⁰ hooks)
Version: 5.0.3
Date: 2025-10-21
Maintainer: ANGELA System Framework
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Protocol, Tuple, Union

# --- Logger -----------------------------------------------------------------
logger = logging.getLogger("ANGELA.ConceptSynthesizer.v5.0.3")

# -----------------------
# Lightweight SHA-256 Ledger
# -----------------------
ledger_chain: List[Dict[str, Any]] = []

def log_event_to_ledger(event_data: Dict[str, Any]) -> None:
    prev_hash = ledger_chain[-1]["current_hash"] if ledger_chain else "0" * 64
    timestamp = time.time()
    payload = {"timestamp": timestamp, "event": event_data, "previous_hash": prev_hash}
    payload_str = json.dumps(payload, sort_keys=True).encode()
    current_hash = hashlib.sha256(payload_str).hexdigest()
    payload["current_hash"] = current_hash
    ledger_chain.append(payload)

def get_ledger() -> List[Dict[str, Any]]:
    return ledger_chain

def verify_ledger() -> bool:
    for i in range(1, len(ledger_chain)):
        expected = hashlib.sha256(
            json.dumps(
                {
                    "timestamp": ledger_chain[i]["timestamp"],
                    "event": ledger_chain[i]["event"],
                    "previous_hash": ledger_chain[i - 1]["current_hash"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if expected != ledger_chain[i]["current_hash"]:
            return False
    return True

# -----------------------
# Ledger-backed Context Manager
# -----------------------
class LedgerContextManager:
    """Simple implementation of ContextManagerLike backed by the ledger."""
    async def log_event_with_hash(self, event: Dict[str, Any]) -> None:
        # synchronous ledger append wrapped in async for DI compatibility
        log_event_to_ledger(event)

# -----------------------
# Ethics Journal
# -----------------------
class EthicsJournal:
    """In-memory ethics rationale journaling with optional JSON persistence."""
    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    def record(self, fork_id: str, rationale: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        self._events.append({"ts": time.time(), "fork_id": fork_id, "rationale": rationale, "outcome": outcome})

    def export(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._events)

    def dump_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._events, f, indent=2, ensure_ascii=False)

# -----------------------
# Protocols & No-op clients
# -----------------------
class LLMClient(Protocol):
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Union[str, Dict[str, Any]]:
        ...

class HTTPClient(Protocol):
    async def get_json(self, url: str) -> Dict[str, Any]:
        ...

class ContextManagerLike(Protocol):
    async def log_event_with_hash(self, event: Dict[str, Any]) -> None: ...

class ErrorRecoveryLike(Protocol):
    async def handle_error(self, error_msg: str, *, retry_func: Optional[Callable[[], Awaitable[Any]]] = None, default: Any = None, diagnostics: Optional[Dict[str, Any]] = None) -> Any: ...

class MemoryManagerLike(Protocol):
    async def store(self, query: str, output: Any, *, layer: str, intent: str, task_type: str = "") -> None: ...
    async def retrieve(self, query: str, *, layer: str, task_type: str = "") -> Any: ...
    async def search(self, *, query_prefix: str, layer: str, intent: str, task_type: str = "") -> List[Dict[str, Any]]: ...

class MetaCognitionLike(Protocol):
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]: ...
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]: ...

class VisualizerLike(Protocol):
    async def render_charts(self, plot_data: Dict[str, Any]) -> None: ...
    async def tag_overlay(self, tag: str, payload: Dict[str, Any]) -> None: ...
    async def list_active_overlays(self) -> List[Dict[str, Any]]: ...

class ReasoningEngineLike(Protocol):
    async def weigh_value_conflict(self, candidates: List[Any], harms: Dict[str, float], rights: Dict[str, float]) -> List[Dict[str, Any]]:
        ...
    async def attribute_causality(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...

@dataclass
class NoopLLM:
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        _ = (prompt, model, temperature)
        # deterministic safe stub
        return {"score": 0.85, "text": "{}", "note": "noop-llm"}

@dataclass
class NoopErrorRecovery:
    async def handle_error(self, error_msg: str, *, retry_func: Optional[Callable[[], Awaitable[Any]]] = None, default: Any = None, diagnostics: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug("ErrorRecovery(noop): %s", error_msg)
        return default

@dataclass
class NoopHTTP:
    async def get_json(self, url: str) -> Dict[str, Any]:
        _ = url
        return {"status": "success", "guidelines": []}

# -----------------------
# Trait wavelets
# -----------------------
@lru_cache(maxsize=200)
def eta_empathy(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=200)
def mu_morality(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.3), 1.0))

# -----------------------
# Small utilities
# -----------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _parse_llm_jsonish(resp: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, str):
        s = resp.strip()
        try:
            return json.loads(s)
        except Exception:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end+1])
                except Exception:
                    pass
            return {"text": s}
    return {"text": str(resp)}

# -----------------------
# Embedded AlignmentGuard (τ-wired but compact)
# -----------------------
class AlignmentGuard:
    """Embedded alignment guard with τ-wiring (compact)."""

    def __init__(self, *, context_manager: Optional[ContextManagerLike] = None, error_recovery: Optional[ErrorRecoveryLike] = None, memory_manager: Optional[MemoryManagerLike] = None, meta_cognition: Optional[MetaCognitionLike] = None, visualizer: Optional[VisualizerLike] = None, llm: Optional[LLMClient] = None, reasoning_engine: Optional[ReasoningEngineLike] = None, ethical_threshold: float = 0.8, drift_validation_threshold: float = 0.7, trait_weights: Optional[Dict[str, float]] = None) -> None:
        self.context_manager = context_manager
        self.error_recovery = error_recovery or NoopErrorRecovery()
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        self.llm = llm or NoopLLM()
        self.reasoning_engine = reasoning_engine
        self.ethical_threshold = float(ethical_threshold)
        self.drift_validation_threshold = float(drift_validation_threshold)
        self.trait_weights = {"eta_empathy": 0.5, "mu_morality": 0.5, **(trait_weights or {})}
        self.validation_log: Deque[Dict[str, Any]] = deque(maxlen=1000)

    # Core ethical check (LLM-driven or fallback)
    async def check(self, prompt: str, *, task_type: str = "") -> bool:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        try:
            t = time.time() % 1.0
            trait_mod = self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t) + self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
            validation_prompt = f"Return JSON {{'score':number}} for ethical alignment. Task:{task_type}\nContent:\n{prompt}"
            raw = await self.llm.generate(validation_prompt, model="gpt-4", temperature=0.3)
            parsed = _parse_llm_jsonish(raw)
            score = _as_float(parsed.get("score", 0.0))
            valid = score >= self.ethical_threshold
            entry = {"prompt": prompt[:200], "score": score, "valid": valid, "trait_mod": trait_mod, "timestamp": _utc_now_iso(), "task_type": task_type}
            self.validation_log.append(entry)
            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({"event": "ethical_check", "score": score, "valid": valid, "task_type": task_type, "timestamp": _utc_now_iso()})
                except Exception:
                    logger.debug("Context log failed in check")
            return valid
        except Exception as e:
            logger.error("ethical check failed: %s", e)
            return await self.error_recovery.handle_error(str(e), default=False)

    # Proportional selection method (reused by harmonize)
    async def consume_ranked_tradeoffs(self, ranked_options: List[Dict[str, Any]], *, safety_ceiling: float = 0.85, k: int = 1, temperature: float = 0.0, min_score_floor: float = 0.0, task_type: str = "") -> Dict[str, Any]:
        # Reuse logic similar to the monolithic guard (kept compact)
        if not isinstance(ranked_options, list) or not ranked_options:
            raise ValueError("ranked_options must be a non-empty list")
        try:
            EPS = 1e-9
            norm = []
            for i, item in enumerate(ranked_options):
                if isinstance(item, dict):
                    opt = item.get("option", f"opt_{i}")
                    score = float(item.get("score", 0.0))
                    reasons = item.get("reasons", [])
                    meta = item.get("meta", {})
                else:
                    opt = getattr(item, "option", f"opt_{i}")
                    score = float(getattr(item, "score", 0.0))
                    reasons = getattr(item, "reasons", [])
                    meta = getattr(item, "meta", {})
                max_harm = None
                if isinstance(meta, dict):
                    if "max_harm" in meta:
                        try:
                            max_harm = float(meta["max_harm"])
                        except Exception:
                            max_harm = None
                    harms = meta.get("harms")
                    if max_harm is None and isinstance(harms, dict) and harms:
                        try:
                            max_harm = float(harms.get("safety", max(harms.values())))
                        except Exception:
                            max_harm = None
                if max_harm is None:
                    rx = re.compile(r"max_harm\s*[:=]\s*([0-9]*\.?[0-9]+)")
                    for r in (reasons or []):
                        if not isinstance(r, str):
                            continue
                        m = rx.search(r)
                        if m:
                            try:
                                max_harm = float(m.group(1))
                            except Exception:
                                max_harm = None
                            break
                if max_harm is None:
                    max_harm = 0.0
                max_harm = float(max(0.0, min(1.0, max_harm)))
                if not isinstance(meta, dict):
                    meta = {}
                meta["max_harm"] = max_harm
                norm.append({"option": opt, "score": max(0.0, min(1.0, score)), "reasons": reasons or [], "meta": meta, "max_harm": max_harm})
            norm = [n for n in norm if n["score"] >= float(min_score_floor)]
            if not norm:
                return {"selections": [], "audit": {"reason": "all options fell below floor"}}
            sc = float(safety_ceiling)
            safe = [n for n in norm if n["max_harm"] <= sc + EPS]
            suppressed = [n for n in norm if n not in safe]
            if not safe and norm:
                safe = sorted(norm, key=lambda x: (-x["score"], x["max_harm"]))[:1]
            scores = [n["score"] for n in safe]
            s_min, s_max = min(scores), max(scores)
            if s_max > s_min:
                for n in safe:
                    n["norm_score"] = (n["score"] - s_min) / (s_max - s_min)
            else:
                for n in safe:
                    n["norm_score"] = 1.0
            if temperature and temperature > 0.0:
                exps = [math.exp(n["norm_score"] / float(temperature)) for n in safe]
                Z = sum(exps) or 1.0
                for n, e in zip(safe, exps):
                    n["weight"] = e / Z
            else:
                total = sum(n["norm_score"] for n in safe) or 1.0
                for n in safe:
                    n["weight"] = n["norm_score"] / total
            pool = safe.copy()
            selections = []
            for _ in range(min(k, len(pool))):
                r = random.random()
                acc = 0.0
                chosen_idx = 0
                for idx, n in enumerate(pool):
                    acc += n["weight"]
                    if r <= acc:
                        chosen_idx = idx
                        break
                chosen = pool.pop(chosen_idx)
                selections.append(chosen["option"])
                if pool:
                    total_w = sum(n["weight"] for n in pool) or 1.0
                    for n in pool:
                        n["weight"] = n["weight"] / total_w
            audit = {"mode": "proportional_selection", "safety_ceiling": round(float(safety_ceiling), 6), "floor": round(float(min_score_floor), 6), "temperature": round(float(temperature), 6), "suppressed_count": len(suppressed), "considered": [{"option": n["option"], "score": round(float(n["score"]), 3), "max_harm": round(float(n["max_harm"]), 3), "weight": round(float(n.get("weight", 0.0)), 3)} for n in safe], "timestamp": _utc_now_iso(), "task_type": task_type}
            if self.memory_manager:
                try:
                    await self.memory_manager.store(query=f"ProportionalSelect::{_utc_now_iso()}", output={"ranked_options": ranked_options, "audit": audit, "selections": selections}, layer="EthicsDecisions", intent="τ.proportional_selection", task_type=task_type)
                except Exception:
                    logger.debug("Memory store failed in proportional selection; continuing")
            return {"selections": selections, "audit": audit}
        except Exception as e:
            logger.error("consume_ranked_tradeoffs failed: %s", e)
            return await self.error_recovery.handle_error(str(e), default={"selections": [], "error": str(e)})

    # Local fallback ranker
    async def _rank_value_conflicts_fallback(self, candidates: List[Any], harms: Dict[str, float], rights: Dict[str, float]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        def _norm(d):
            if not d:
                return {}
            vals = [max(0.0, float(v)) for v in d.values()]
            mx = max(vals) if vals else 1.0
            return {k: (max(0.0, float(v)) / mx if mx > 0 else 0.0) for k, v in d.items()}
        h = _norm(harms)
        r = _norm(rights)
        ranked = []
        for i, c in enumerate(candidates):
            meta = {}
            label = c if isinstance(c, str) else (c.get("option") if isinstance(c, dict) else f"opt_{i}")
            agg_harm = sum(h.values()) / (len(h) or 1)
            agg_right = sum(r.values()) / (len(r) or 1)
            score = max(0.0, min(1.0, 0.5 + (agg_right - agg_harm) * 0.5))
            ranked.append({"option": c, "score": score, "reasons": [f"fallback rights≈{agg_right:.2f} harms≈{agg_harm:.2f}"], "meta": {"harms": harms, "rights": rights, "max_harm": max(h.values(), default=0.0)}})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    # Harmonize: rank via reasoning engine when available, then proportional select
    async def harmonize(self, candidates: List[Any], harms: Dict[str, float], rights: Dict[str, float], *, safety_ceiling: float = 0.85, k: int = 1, temperature: float = 0.0, min_score_floor: float = 0.0, task_type: str = "", audit_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must be a non-empty list")
        if not isinstance(harms, dict) or not isinstance(rights, dict):
            raise TypeError("harms and rights must be dicts")
        try:
            if self.reasoning_engine and hasattr(self.reasoning_engine, "weigh_value_conflict"):
                try:
                    ranked = await self.reasoning_engine.weigh_value_conflict(candidates, harms, rights)
                except Exception as e:
                    logger.warning("reasoning_engine.weigh_value_conflict failed: %s; falling back", e)
                    ranked = await self._rank_value_conflicts_fallback(candidates, harms, rights)
            else:
                ranked = await self._rank_value_conflicts_fallback(candidates, harms, rights)
            result = await self.consume_ranked_tradeoffs(ranked, safety_ceiling=safety_ceiling, k=k, temperature=temperature, min_score_floor=min_score_floor, task_type=task_type)
            if audit_events and self.reasoning_engine and hasattr(self.reasoning_engine, "attribute_causality"):
                try:
                    causality_report = await self.reasoning_engine.attribute_causality(audit_events)
                    result["causality"] = causality_report
                except Exception:
                    logger.debug("attribute_causality failed; continuing")
            # ledger log via context manager
            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({"event": "τ_harmonize", "result": result, "task_type": task_type, "timestamp": _utc_now_iso()})
                except Exception:
                    logger.debug("Context logging failed in harmonize")
            return result
        except Exception as e:
            logger.error("harmonize failed: %s", e)
            return await self.error_recovery.handle_error(str(e), default={"selections": [], "error": str(e)})

# -----------------------
# ConceptSynthesizer (upgraded)
# -----------------------
def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

class ConceptSynthesizer:
    """ConceptSynthesizer v5.0.3-compat — embedded alignment & ledger."""

    def __init__(
        self,
        *,
        context_manager: Optional[ContextManagerLike] = None,
        error_recovery: Optional[ErrorRecoveryLike] = None,
        memory_manager: Optional[MemoryManagerLike] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        meta_cognition: Optional[MetaCognitionLike] = None,
        visualizer: Optional[VisualizerLike] = None,
        mm_fusion: Optional[Any] = None,
        llm_client: Optional[LLMClient] = None,
        stage_iv_enabled: Optional[bool] = None,
    ):
        # Embedded defaults
        self.context_manager = context_manager or LedgerContextManager()
        self.error_recovery = error_recovery or NoopErrorRecovery()
        self.memory_manager = memory_manager
        self.alignment_guard = alignment_guard or AlignmentGuard(context_manager=self.context_manager, error_recovery=self.error_recovery, memory_manager=self.memory_manager, meta_cognition=meta_cognition, visualizer=visualizer, llm=llm_client)
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        self.mm_fusion = mm_fusion
        self.llm_client = llm_client or NoopLLM()
        self.concept_cache: deque = deque(maxlen=1000)
        self.similarity_threshold: float = 0.75
        self.stage_iv_enabled: bool = stage_iv_enabled if stage_iv_enabled is not None else _bool_env("ANGELA_STAGE_IV", False)
        self.journal = EthicsJournal()
        self.default_retry_spec: Tuple[int, float] = (3, 0.6)

        logger.info("ConceptSynthesizer v5.0.3 init | sim_thresh=%.2f | stage_iv=%s", self.similarity_threshold, self.stage_iv_enabled)

    async def _with_retries(self, label: str, fn: Callable[[], Awaitable[Any]], attempts: Optional[int] = None, base_delay: Optional[float] = None):
        tries = attempts or self.default_retry_spec[0]
        delay = base_delay or self.default_retry_spec[1]
        last_exc = None
        for i in range(1, tries + 1):
            try:
                return await fn()
            except Exception as e:
                last_exc = e
                logger.warning("%s attempt %d/%d failed: %s", label, i, tries, str(e))
                if i < tries:
                    await asyncio.sleep(delay * (2 ** (i - 1)))
        diagnostics = None
        if self.meta_cognition:
            try:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            except Exception:
                diagnostics = None
        return await self.error_recovery.handle_error(str(last_exc), retry_func=fn, default=None, diagnostics=diagnostics or {})

    # Visualization helper (φ⁰ overlay tagging support)
    def _visualize_fire(self, payload: Dict[str, Any], task_type: str, mode: str):
        if not self.visualizer:
            return
        viz_opts = {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise", "reality_sculpting": bool(self.stage_iv_enabled)}
        plot_data = {mode: payload, "visualization_options": viz_opts}
        # best effort: fire-and-forget
        try:
            asyncio.create_task(self.visualizer.render_charts(plot_data))
            # if visualizer supports overlay tagging, attach tag (Φ⁰)
            if hasattr(self.visualizer, "tag_overlay"):
                try:
                    asyncio.create_task(self.visualizer.tag_overlay(mode, {"mode": mode, "payload": payload, "task_type": task_type}))
                except Exception:
                    logger.debug("overlay tag failed (fire-and-forget)")
        except Exception:
            logger.debug("visualization scheduling failed")

    # Integration of external concept data (light wrapper)
    async def integrate_external_concept_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        try:
            # simple fallback: ask provided HTTP client if available via memory_manager or return empty
            if self.memory_manager:
                cache_key = f"ConceptData::{data_type}::{data_source}::{task_type}"
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict) and cached.get("data"):
                    ts = cached.get("timestamp")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts)
                            if (datetime.now(dt.tzinfo or timezone.utc) - dt).total_seconds() < cache_timeout:
                                logger.info("Returning cached concept data for %s", cache_key)
                                return cached.get("data", {})
                        except Exception:
                            pass
            # no external call hardwired (to keep module self-contained). Return empty success envelope.
            return {"status": "success", "definitions": []} if data_type == "concept_definitions" else {"status": "success", "ontology": {}}
        except Exception as e:
            logger.error("integrate_external_concept_data failed: %s", e)
            return await self.error_recovery.handle_error(str(e), default={"status": "error", "error": str(e)})

    # Generate: create concept from context + optional multimodal fusion
    async def generate(self, concept_name: str, context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dict")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        logger.info("Generating concept '%s' task=%s", concept_name, task_type)
        try:
            fused_context = dict(context)
            if self.mm_fusion and any(k in context for k in ("text", "image", "audio", "video", "embeddings")):
                try:
                    fused = await self.mm_fusion.fuse(context)
                    if isinstance(fused, dict):
                        fused_context["fused"] = fused
                        logger.info("Applied cross-modal fusion")
                except Exception:
                    logger.debug("mm_fusion failed; proceeding without fusion")

            # get external hints (best-effort)
            concept_data = await self.integrate_external_concept_data("xai_ontology_db", "concept_definitions", task_type=task_type)
            external_defs = concept_data.get("definitions", []) if concept_data.get("status") == "success" else []

            # construct safe prompt; we use injected llm_client for generation
            prompt = {
                "instruction": "Return strict JSON with keys ['name','definition','version','context'] only.",
                "name": concept_name,
                "context": fused_context,
                "hints": external_defs,
                "task_type": task_type,
            }

            async def call_llm():
                # prefer llm_client if provided else NoopLLM
                return await (self.llm_client.generate(json.dumps(prompt), model="gpt-4", temperature=0.5))

            llm_raw = await self._with_retries("llm:generate_concept", call_llm)
            if llm_raw is None:
                return {"error": "LLM generation failed", "success": False}

            # Normalize LLM output
            if isinstance(llm_raw, dict):
                concept = llm_raw
            elif isinstance(llm_raw, str):
                try:
                    concept = json.loads(llm_raw)
                except Exception:
                    start = llm_raw.find("{")
                    end = llm_raw.rfind("}")
                    try:
                        concept = json.loads(llm_raw[start:end+1])
                    except Exception:
                        concept = {"definition": llm_raw}
            else:
                concept = {"definition": str(llm_raw)}

            # Metadata
            concept.setdefault("name", concept_name)
            concept.setdefault("version", concept.get("version", "1.0"))
            concept["context"] = fused_context
            concept["timestamp"] = _utc_now_iso()
            concept["task_type"] = task_type

            # Ethical check: prefer harmonize if concept presents alternatives; else standard ethical_check
            # If LLM returned multiple candidate definitions (list under 'candidates'), use harmonize.
            if self.alignment_guard:
                if isinstance(concept.get("candidates"), list) and concept["candidates"]:
                    # build harms/rights heuristic for conceptual alternatives (placeholders)
                    harms = {"safety": 0.1}
                    rights = {"consistency": 0.8}
                    harmonize_result = await self.alignment_guard.harmonize(concept["candidates"], harms, rights, safety_ceiling=0.85, k=1, task_type=task_type)
                    selected = harmonize_result.get("selections", [])
                    if selected:
                        # assume selected[0] is a concept payload or label; attempt to set concept.definition
                        pick = selected[0]
                        if isinstance(pick, dict) and "definition" in pick:
                            concept["definition"] = pick["definition"]
                        else:
                            concept["definition"] = str(pick)
                        # journal rationale
                        self.journal.record(f"concept_gen::{concept_name}", {"mode": "harmonize", "result": harmonize_result}, {"concept": concept})
                    else:
                        # fallback to single LLM candidate
                        pass
                else:
                    # simple ethical_check
                    ok = await self.alignment_guard.check(str(concept.get("definition", "")), task_type=task_type)
                    if not ok:
                        # record failure and return
                        self.journal.record(f"concept_gen::{concept_name}", {"mode": "ethical_check", "failed": True}, {"concept": concept})
                        return {"error": "Concept failed ethical check", "success": False, "concept": concept}

            # persist & cache
            self.concept_cache.append(concept)
            if self.memory_manager:
                try:
                    await self.memory_manager.store(query=f"Concept::{concept_name}::{_utc_now_iso()}", output=json.dumps(concept, ensure_ascii=False), layer="Concepts", intent="concept_generation", task_type=task_type)
                except Exception:
                    logger.debug("Memory store failed for concept generation")

            # context manager ledger log (ensures cryptographic chain)
            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({"event": "concept_generation", "concept_name": concept_name, "task_type": task_type, "timestamp": _utc_now_iso(), "valid": True})
                except Exception:
                    logger.debug("Context manager log failed during generate")

            # visualization (Φ⁰-aware)
            self._visualize_fire({"concept_name": concept_name, "definition": concept.get("definition", ""), "task_type": task_type}, task_type, "concept_generation")

            # final reflection
            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(component="ConceptSynthesizer", output=concept, context={"task_type": task_type})
                except Exception:
                    logger.debug("Meta cognition reflect failed")

            # success
            self.journal.record(f"concept_gen::{concept_name}", {"mode": "final", "concept": concept}, {"success": True})
            return {"concept": concept, "success": True}

        except Exception as e:
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(str(e), retry_func=lambda: self.generate(concept_name, context, task_type), default={"error": str(e), "success": False}, diagnostics=diagnostics or {})

    # Compare: uses mm_fusion if available, else LLM, and τ-harmonization if drift large
    async def compare(self, concept_a: str, concept_b: str, task_type: str = "") -> Dict[str, Any]:
        if not isinstance(concept_a, str) or not isinstance(concept_b, str):
            raise TypeError("concepts must be strings")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        logger.info("Comparing concepts '%s' vs '%s' task=%s", concept_a, concept_b, task_type)
        try:
            # memory cache lookup
            if self.memory_manager:
                try:
                    entries = await self.memory_manager.search(query_prefix="ConceptComparison", layer="Concepts", intent="concept_comparison", task_type=task_type)
                    if entries:
                        for entry in entries:
                            out = entry.get("output")
                            try:
                                payload = out if isinstance(out, dict) else json.loads(out)
                            except Exception:
                                payload = {}
                            if payload.get("concept_a") == concept_a and payload.get("concept_b") == concept_b:
                                logger.info("Returning cached comparison")
                                return payload
                except Exception:
                    logger.debug("Memory search failed; continuing")

            mm_score = None
            if self.mm_fusion and hasattr(self.mm_fusion, "compare_semantic"):
                try:
                    mm_score = await self.mm_fusion.compare_semantic(concept_a, concept_b)
                except Exception:
                    logger.debug("mm_fusion compare failed; ignoring")

            # LLM structured compare via injected client
            prompt = {"instruction": "Return JSON {'score','differences','similarities'}", "concept_a": concept_a, "concept_b": concept_b, "task_type": task_type}
            async def call_llm():
                return await (self.llm_client.generate(json.dumps(prompt), model="gpt-4", temperature=0.3))
            llm_raw = await self._with_retries("llm:compare", call_llm)
            if llm_raw is None:
                return {"error": "LLM compare failed", "success": False}
            if isinstance(llm_raw, dict):
                comp = llm_raw
            elif isinstance(llm_raw, str):
                try:
                    comp = json.loads(llm_raw[llm_raw.find("{") : llm_raw.rfind("}") + 1])
                except Exception:
                    comp = {"score": 1.0, "differences": [], "similarities": []}
            else:
                comp = {"score": 1.0, "differences": [], "similarities": []}

            # blend multimodal insight
            if isinstance(mm_score, (int, float)):
                comp_score = float(comp.get("score", 0.0))
                comp["score"] = max(0.0, min(1.0, 0.7 * comp_score + 0.3 * float(mm_score)))

            comp["concept_a"] = concept_a
            comp["concept_b"] = concept_b
            comp["timestamp"] = _utc_now_iso()
            comp["task_type"] = task_type

            # If similarity low, perform τ-harmonize for mitigation suggestions
            if comp.get("score", 0.0) < self.similarity_threshold and self.alignment_guard:
                # Build candidate mitigations (simple heuristics)
                candidates = [
                    {"option": "merge_definitions", "payload": {"a": concept_a, "b": concept_b}},
                    {"option": "choose_a", "payload": {"a": concept_a}},
                    {"option": "choose_b", "payload": {"b": concept_b}},
                ]
                harms = {"disruption": 0.3}
                rights = {"consistency": 0.7}
                harmonize_res = await self.alignment_guard.harmonize(candidates, harms, rights, safety_ceiling=0.85, k=1, task_type=task_type, audit_events=[{"event": "concept_comparison", "a": concept_a, "b": concept_b}])
                comp["harmonize"] = harmonize_res
                # journal
                self.journal.record(f"compare::{concept_a}::{concept_b}", {"comp": comp, "harmonize": harmonize_res}, {"success": True})

            # persist + visualize + reflect
            self.concept_cache.append(comp)
            if self.memory_manager:
                try:
                    await self.memory_manager.store(query=f"ConceptComparison::{_utc_now_iso()}", output=json.dumps(comp, ensure_ascii=False), layer="Concepts", intent="concept_comparison", task_type=task_type)
                except Exception:
                    logger.debug("Memory store failed for comparison")

            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({"event": "concept_comparison", "score": comp.get("score", 0.0), "task_type": task_type, "timestamp": _utc_now_iso()})
                except Exception:
                    logger.debug("Context manager log failed for comparison")

            self._visualize_fire({"score": comp.get("score", 0.0), "differences": comp.get("differences", []), "task_type": task_type}, task_type, "concept_comparison")

            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(component="ConceptSynthesizer", output=comp, context={"task_type": task_type})
                except Exception:
                    logger.debug("Meta-cog reflect failed for compare")

            return comp
        except Exception as e:
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(str(e), retry_func=lambda: self.compare(concept_a, concept_b, task_type), default={"error": str(e), "success": False}, diagnostics=diagnostics or {})

    async def validate(self, concept: Dict[str, Any], task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        if not isinstance(concept, dict) or not all(k in concept for k in ("name", "definition")):
            raise ValueError("concept must be dict with keys 'name' and 'definition'")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        logger.info("Validating concept '%s' task=%s", concept["name"], task_type)
        try:
            valid = True
            issues: List[str] = []
            # Use alignment_guard.harmonize for multi-criteria validation suggestions if available
            if self.alignment_guard:
                # Build candidate resolution actions (example)
                candidates = [
                    {"option": "accept", "score": 0.8},
                    {"option": "revise_definition", "score": 0.5},
                    {"option": "reject", "score": 0.1},
                ]
                harms = {"risk": 0.2}
                rights = {"integrity": 0.8}
                harmonize_res = await self.alignment_guard.harmonize(candidates, harms, rights, safety_ceiling=0.9, k=1, task_type=task_type)
                # If harmonize suggests reject, mark invalid
                sel = harmonize_res.get("selections", [])
                if sel and sel[0] == "reject":
                    valid = False
                    issues.append("Harmonize recommended rejection")
                # attach audit
                validation_audit = {"harmonize": harmonize_res}
            else:
                validation_audit = {}

            # Basic ethical check fallback
            if self.alignment_guard and valid:
                ok = await self.alignment_guard.check(str(concept.get("definition", "")), task_type=task_type)
                if not ok:
                    valid = False
                    issues.append("Ethical check failed")

            report = {"concept_name": concept["name"], "valid": valid, "issues": issues, "audit": validation_audit, "timestamp": _utc_now_iso(), "task_type": task_type}
            self.concept_cache.append(report)
            if self.memory_manager:
                try:
                    await self.memory_manager.store(query=f"ConceptValidation::{concept['name']}::{_utc_now_iso()}", output=json.dumps(report, ensure_ascii=False), layer="Concepts", intent="concept_validation", task_type=task_type)
                except Exception:
                    logger.debug("Memory store failed for validate")

            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({"event": "concept_validation", "concept_name": concept["name"], "valid": valid, "issues": issues, "task_type": task_type, "timestamp": _utc_now_iso()})
                except Exception:
                    logger.debug("Context log failed for validate")

            self._visualize_fire({"concept_name": concept["name"], "valid": valid, "issues": issues, "task_type": task_type}, task_type, "concept_validation")

            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(component="ConceptSynthesizer", output=report, context={"task_type": task_type})
                except Exception:
                    logger.debug("Meta-cog reflect failed for validate")

            return valid, report
        except Exception as e:
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(str(e), retry_func=lambda: self.validate(concept, task_type), default=(False, {"error": str(e), "concept_name": concept.get("name", ""), "task_type": task_type}), diagnostics=diagnostics or {})

    def get_symbol(self, concept_name: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        for item in self.concept_cache:
            if isinstance(item, dict) and item.get("name") == concept_name:
                if task_type == "" or item.get("task_type") == task_type:
                    return item
        # best-effort synchronous retrieval from memory_manager if available (blocking via asyncio.run is avoided here)
        return None

# -----------------------
# Demo runner
# -----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple demo LLM that returns minimal JSON for predictability
    class DemoLLM:
        async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3):
            # return a plausible JSON output as dict
            # For generate concept: return dict with keys name, definition, version
            try:
                p = json.loads(prompt) if prompt.strip().startswith("{") else None
            except Exception:
                p = None
            if p and "name" in p:
                return {"name": p["name"], "definition": f"Auto-generated definition for {p['name']}", "version": "1.0"}
            # generic compare/score output
            return {"score": 0.8, "differences": [], "similarities": ["abstract"], "note": "demo-llm"}

    async def demo():
        llm = DemoLLM()
        cs = ConceptSynthesizer(llm_client=llm, stage_iv_enabled=True)
        res = await cs.generate("Trust", {"domain": "AI Ethics", "text": "Calibrate trust under uncertainty"}, task_type="demo")
        print("generate ->", json.dumps(res, indent=2, ensure_ascii=False))
        comp = await cs.compare("Trust", "Reliability", task_type="demo")
        print("compare ->", json.dumps(comp, indent=2, ensure_ascii=False))
        print("Ledger length:", len(get_ledger()))
        print("Ledger verified:", verify_ledger())
        print("Journal entries:", json.dumps(cs.journal.export("demo-session"), indent=2, ensure_ascii=False))

    asyncio.run(demo())
