"""
ANGELA Cognitive System Module: ConceptSynthesizer
Version: 4.3.4  # DI LLM/HTTP, safer JSON, no blocking, ledger hook, Stage-IV gated viz
Refactor Date: 2025-08-16
Maintainer: ANGELA System Framework

Concept synthesis, comparison, and validation with cross-modal blending (optional),
self-healing retries, strict JSON parsing, and interoperability across ANGELA v4.3.x.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Protocol, Tuple
from collections import deque

# --------------------------- DI Protocols (no hard deps) --------------------

class LLMClient(Protocol):
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> str | Dict[str, Any]: ...

class HTTPClient(Protocol):
    async def get_json(self, url: str) -> Dict[str, Any]: ...

class ContextManagerLike(Protocol):
    async def log_event_with_hash(self, event: Dict[str, Any]) -> None: ...

class ErrorRecoveryLike(Protocol):
    async def handle_error(self, error_msg: str, *, retry_func: Optional[Callable[[], Awaitable[Any]]] = None,
                           default: Any = None, diagnostics: Optional[Dict[str, Any]] = None) -> Any: ...

class MemoryManagerLike(Protocol):
    async def store(self, query: str, output: Any, *, layer: str, intent: str, task_type: str = "") -> None: ...
    async def retrieve(self, query: str, *, layer: str, task_type: str = "") -> Any: ...
    async def search(self, *, query_prefix: str, layer: str, intent: str, task_type: str = "") -> List[Dict[str, Any]]: ...

class AlignmentGuardLike(Protocol):
    async def ethical_check(self, content: str, *, stage: str = "pre", task_type: str = "") -> Tuple[bool, Dict[str, Any]]: ...

class MetaCognitionLike(Protocol):
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]: ...
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]: ...

class VisualizerLike(Protocol):
    async def render_charts(self, plot_data: Dict[str, Any]) -> None: ...

class MultiModalFusionLike(Protocol):
    async def fuse(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...
    async def compare_semantic(self, a: str, b: str) -> float: ...

# --------------------------- No-op fallbacks (safe) -------------------------

@dataclass
class NoopLLM:
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        _ = (prompt, model, temperature)
        return {"name": "Concept", "definition": "{}", "version": "0.0", "context": {}}

@dataclass
class NoopHTTP:
    async def get_json(self, url: str) -> Dict[str, Any]:
        _ = url
        return {"status": "success"}

@dataclass
class NoopErrorRecovery:
    async def handle_error(self, error_msg: str, *, retry_func=None, default=None, diagnostics=None) -> Any:
        return default

@dataclass
class NoopMeta:
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]:
        return {"status": "ok"}
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "success", "reflection": ""}

@dataclass
class NoopViz:
    async def render_charts(self, plot_data: Dict[str, Any]) -> None:
        return None

# ------------------------------- Utils --------------------------------------

logger = logging.getLogger("ANGELA.ConceptSynthesizer")

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _json_strict_or_substring(s: str) -> Dict[str, Any]:
    """Parse exact JSON; otherwise extract the largest {...} block."""
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
    return {}

def _norm_llm_payload(resp: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, str):
        return _json_strict_or_substring(resp)
    return {}

# ------------------------------- Main ---------------------------------------

class ConceptSynthesizer:
    """v4.3.4 ConceptSynthesizer with DI I/O, retries, and Stage-IV aware viz."""

    def __init__(
        self,
        *,
        context_manager: Optional[ContextManagerLike] = None,
        error_recovery: Optional[ErrorRecoveryLike] = None,
        memory_manager: Optional[MemoryManagerLike] = None,
        alignment_guard: Optional[AlignmentGuardLike] = None,
        meta_cognition: Optional[MetaCognitionLike] = None,
        visualizer: Optional[VisualizerLike] = None,
        mm_fusion: Optional[MultiModalFusionLike] = None,
        http: Optional[HTTPClient] = None,
        llm: Optional[LLMClient] = None,
        ledger_hook: Optional[Callable[[Dict[str, Any]], None]] = None,  # tamper-evident anchoring
        stage_iv_enabled: Optional[bool] = None,
        similarity_threshold: float = 0.75,
        retry_attempts: int = 3,
        retry_base_delay: float = 0.6,
    ) -> None:
        # DI wiring
        self.context_manager = context_manager
        self.error_recovery = error_recovery or NoopErrorRecovery()
        self.memory_manager = memory_manager
        self.alignment_guard = alignment_guard
        self.meta = meta_cognition or NoopMeta()
        self.viz = visualizer or NoopViz()
        self.mm_fusion = mm_fusion
        self.http = http or NoopHTTP()
        self.llm = llm or NoopLLM()
        self.ledger_hook = ledger_hook

        # behavior
        self.stage_iv_enabled = stage_iv_enabled if stage_iv_enabled is not None else _bool_env("ANGELA_STAGE_IV", False)
        self.similarity_threshold = float(similarity_threshold)
        self.retry_attempts = int(retry_attempts)
        self.retry_base_delay = float(retry_base_delay)

        # state
        self.concept_cache: Deque[Dict[str, Any]] = deque(maxlen=1000)

        logger.info(
            "ConceptSynthesizer v4.3.4 init | sim=%.2f | stage_iv=%s | mm=%s",
            self.similarity_threshold, self.stage_iv_enabled, "on" if self.mm_fusion else "off"
        )

    # ----------------------------- internals --------------------------------

    async def _with_retries(self, label: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        delay = self.retry_base_delay
        last_exc: Optional[Exception] = None
        for i in range(1, self.retry_attempts + 1):
            try:
                return await fn()
            except Exception as e:
                last_exc = e
                logger.warning("%s attempt %d/%d failed: %s", label, i, self.retry_attempts, e)
                if i < self.retry_attempts:
                    await asyncio.sleep(delay)
                    delay *= 2
        diagnostics = None
        try:
            diagnostics = await self.meta.run_self_diagnostics(return_only=True)
        except Exception:
            diagnostics = None
        return await self.error_recovery.handle_error(
            str(last_exc or f"{label} failed"),
            retry_func=fn, default=None, diagnostics=diagnostics
        )

    async def _post_reflect(self, component: str, output: Dict[str, Any], task_type: str):
        if self.meta and task_type:
            try:
                await self.meta.reflect_on_output(component=component, output=output, context={"task_type": task_type})
            except Exception:
                logger.debug("reflection failed")

    def _viz(self, payload: Dict[str, Any], task_type: str, mode: str):
        if not task_type:
            return
        viz_opts = {
            "interactive": task_type == "recursion",
            "style": "detailed" if task_type == "recursion" else "concise",
            "reality_sculpting": bool(self.stage_iv_enabled),
        }
        asyncio.create_task(self.viz.render_charts({mode: payload, "visualization_options": viz_opts}))

    def _ledger(self, event: Dict[str, Any]) -> None:
        if not self.ledger_hook:
            return
        try:
            self.ledger_hook({**event, "ts": _utc_now_iso()})
        except Exception:
            logger.debug("ledger_hook failed")

    # --------------------------- external data --------------------------------

    async def integrate_external_concept_data(
        self, *, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        cache_key = f"ConceptData::{data_type}::{data_source}::{task_type}"
        try:
            if self.memory_manager:
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict):
                    ts = cached.get("timestamp")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts)
                            if (datetime.now(dt.tzinfo or timezone.utc) - dt).total_seconds() < cache_timeout:
                                return cached.get("data", {})
                        except Exception:
                            pass

            async def _fetch():
                return await self.http.get_json(data_source)

            data = await self._with_retries(f"fetch:{data_type}", _fetch)

            if data_type == "ontology":
                payload = {"status": "success", "ontology": data.get("ontology", {})}
            elif data_type == "concept_definitions":
                payload = {"status": "success", "definitions": data.get("definitions", [])}
            else:
                payload = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager and payload.get("status") == "success":
                await self.memory_manager.store(
                    cache_key, {"data": payload, "timestamp": _utc_now_iso()},
                    layer="ExternalData", intent="concept_data_integration", task_type=task_type
                )

            await self._post_reflect("ConceptSynthesizer", {"ext_data": payload, "data_type": data_type}, task_type)
            return payload
        except Exception as e:
            diagnostics = None
            try:
                diagnostics = await self.meta.run_self_diagnostics(return_only=True)
            except Exception:
                pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_concept_data(
                    data_source=data_source, data_type=data_type, cache_timeout=cache_timeout, task_type=task_type
                ),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
            )

    # ------------------------------ API: generate -----------------------------

    async def generate(self, concept_name: str, context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dict")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Generating concept '%s' (task=%s)", concept_name, task_type)

        try:
            fused_ctx: Dict[str, Any] = dict(context)
            if self.mm_fusion and any(k in context for k in ("text", "image", "audio", "video", "embeddings", "scenegraph")):
                try:
                    fused = await self.mm_fusion.fuse(context)
                    if isinstance(fused, dict):
                        fused_ctx = {**context, "fused": fused}
                except Exception:
                    logger.debug("mm_fusion.fuse failed; continuing")

            ext = await self.integrate_external_concept_data(
                data_source="concept://definitions/default", data_type="concept_definitions", task_type=task_type
            )
            external_defs: List[Dict[str, Any]] = ext.get("definitions", []) if ext.get("status") == "success" else []

            prompt = (
                "Return STRICT JSON with keys ['name','definition','version','context'] only.\n"
                f"name={json.dumps(concept_name, ensure_ascii=False)}\n"
                f"context={json.dumps(fused_ctx, ensure_ascii=False)}\n"
                f"external_definitions={json.dumps(external_defs, ensure_ascii=False)}\n"
                f"task_type={task_type}"
            )

            async def _llm():
                return await self.llm.generate(prompt, model="gpt-4", temperature=0.5)

            raw = await self._with_retries("llm:generate", _llm)
            concept = _norm_llm_payload(raw)
            if not concept:
                return {"error": "LLM returned empty/invalid JSON", "success": False}

            concept["timestamp"] = time.time()
            concept["task_type"] = task_type

            if self.alignment_guard:
                ok, report = await self.alignment_guard.ethical_check(
                    str(concept.get("definition", "")), stage="concept_generation", task_type=task_type
                )
                if not ok:
                    return {"error": "Concept failed ethical check", "report": report, "success": False}

            self.concept_cache.append(concept)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Concept::{concept_name}::{_utc_now_iso()}",
                    output=concept, layer="Concepts", intent="concept_generation", task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "concept_generation", "concept_name": concept_name, "task_type": task_type}
                )

            self._viz({"concept_name": concept_name, "definition": concept.get("definition", ""), "task_type": task_type},
                      task_type, mode="concept_generation")
            await self._post_reflect("ConceptSynthesizer", concept, task_type)
            self._ledger({"type": "concept_generate", "name": concept_name, "task_type": task_type})
            return {"concept": concept, "success": True}
        except Exception as e:
            diagnostics = None
            try:
                diagnostics = await self.meta.run_self_diagnostics(return_only=True)
            except Exception:
                pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.generate(concept_name, context, task_type),
                default={"error": str(e), "success": False},
                diagnostics=diagnostics,
            )

    # ------------------------------ API: compare ------------------------------

    async def compare(self, concept_a: str, concept_b: str, task_type: str = "") -> Dict[str, Any]:
        if not isinstance(concept_a, str) or not isinstance(concept_b, str):
            raise TypeError("concepts must be strings")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Comparing concepts (task=%s)", task_type)

        try:
            # Try memory cache
            if self.memory_manager:
                try:
                    entries = await self.memory_manager.search(
                        query_prefix="ConceptComparison", layer="Concepts", intent="concept_comparison", task_type=task_type
                    )
                    for entry in entries or []:
                        out = entry.get("output")
                        payload = out if isinstance(out, dict) else _norm_llm_payload(str(out))
                        if payload.get("concept_a") == concept_a and payload.get("concept_b") == concept_b:
                            return payload
                except Exception:
                    logger.debug("memory search failed; continuing")

            mm_score: Optional[float] = None
            if self.mm_fusion and hasattr(self.mm_fusion, "compare_semantic"):
                try:
                    mm_score = await self.mm_fusion.compare_semantic(concept_a, concept_b)
                except Exception:
                    logger.debug("mm compare skipped")

            prompt = (
                "Compare two concepts. Return STRICT JSON with keys "
                "['score','differences','similarities'] only.\n"
                f"A={json.dumps(concept_a, ensure_ascii=False)}\n"
                f"B={json.dumps(concept_b, ensure_ascii=False)}\n"
                f"task_type={task_type}"
            )

            async def _llm():
                return await self.llm.generate(prompt, model="gpt-4", temperature=0.3)

            raw = await self._with_retries("llm:compare", _llm)
            comp = _norm_llm_payload(raw)
            if not comp:
                return {"error": "LLM returned empty/invalid JSON", "success": False}

            if isinstance(mm_score, (int, float)):
                comp["score"] = max(0.0, min(1.0, 0.7 * float(comp.get("score", 0.0)) + 0.3 * float(mm_score)))

            comp.update({"concept_a": concept_a, "concept_b": concept_b, "timestamp": time.time(), "task_type": task_type})

            if comp.get("score", 0.0) < self.similarity_threshold and self.alignment_guard:
                ok, report = await self.alignment_guard.ethical_check(
                    f"Concept drift: {comp.get('differences', [])}", stage="concept_comparison", task_type=task_type
                )
                if not ok:
                    comp.setdefault("issues", []).append("Ethical drift detected")
                    comp["ethical_report"] = report

            self.concept_cache.append(comp)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ConceptComparison::{_utc_now_iso()}",
                    output=comp, layer="Concepts", intent="concept_comparison", task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "concept_comparison", "score": comp.get("score", 0.0), "task_type": task_type}
                )

            self._viz({"score": comp.get("score", 0.0), "differences": comp.get("differences", []), "task_type": task_type},
                      task_type, mode="concept_comparison")
            await self._post_reflect("ConceptSynthesizer", comp, task_type)
            self._ledger({"type": "concept_compare", "task_type": task_type})
            return comp
        except Exception as e:
            diagnostics = None
            try:
                diagnostics = await self.meta.run_self_diagnostics(return_only=True)
            except Exception:
                pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.compare(concept_a, concept_b, task_type),
                default={"error": str(e), "success": False},
                diagnostics=diagnostics,
            )

    # ------------------------------ API: validate -----------------------------

    async def validate(self, concept: Dict[str, Any], task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        if not isinstance(concept, dict) or not all(k in concept for k in ("name", "definition")):
            raise ValueError("concept must include 'name' and 'definition'")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Validating concept '%s' (task=%s)", concept.get("name"), task_type)

        try:
            report: Dict[str, Any] = {"concept_name": concept["name"], "issues": [], "task_type": task_type}
            valid = True

            if self.alignment_guard:
                ok, erep = await self.alignment_guard.ethical_check(
                    str(concept["definition"]), stage="concept_validation", task_type=task_type
                )
                if not ok:
                    valid = False
                    report["issues"].append("Ethical misalignment")
                    report["ethical_report"] = erep

            ont = await self.integrate_external_concept_data(
                data_source="concept://ontology/default", data_type="ontology", task_type=task_type
            )
            if ont.get("status") == "success":
                prompt = (
                    "Validate concept against ontology. Return STRICT JSON with keys ['valid','issues'] only.\n"
                    f"concept={json.dumps(concept, ensure_ascii=False)}\n"
                    f"ontology={json.dumps(ont.get('ontology', {}), ensure_ascii=False)}\n"
                    f"task_type={task_type}"
                )
                async def _llm():
                    return await self.llm.generate(prompt, model="gpt-4", temperature=0.3)
                raw = await self._with_retries("llm:validate", _llm)
                vres = _norm_llm_payload(raw)
                if vres:
                    if not bool(vres.get("valid", True)):
                        valid = False
                        report["issues"].extend([str(i) for i in vres.get("issues", [])])
                else:
                    valid = False
                    report["issues"].append("ontology_validation_unparseable")

            report["valid"] = valid
            report["timestamp"] = time.time()

            self.concept_cache.append(report)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ConceptValidation::{concept['name']}::{_utc_now_iso()}",
                    output=report, layer="Concepts", intent="concept_validation", task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "concept_validation", "concept_name": concept["name"], "valid": valid,
                     "issues": report["issues"], "task_type": task_type}
                )

            self._viz({"concept_name": concept["name"], "valid": valid, "issues": report["issues"], "task_type": task_type},
                      task_type, mode="concept_validation")
            await self._post_reflect("ConceptSynthesizer", report, task_type)
            self._ledger({"type": "concept_validate", "name": concept["name"], "valid": valid, "task_type": task_type})
            return valid, report
        except Exception as e:
            diagnostics = None
            try:
                diagnostics = await self.meta.run_self_diagnostics(return_only=True)
            except Exception:
                pass
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.validate(concept, task_type),
                default=(False, {"error": str(e), "concept_name": concept.get("name", ""), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    # ------------------------------ API: get_symbol ---------------------------

    async def get_symbol(self, concept_name: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Async & non-blocking retrieval (fixes prior asyncio.run())"""
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        for item in self.concept_cache:
            if isinstance(item, dict) and item.get("name") == concept_name and item.get("task_type") == task_type:
                return item

        if self.memory_manager:
            try:
                entries = await self.memory_manager.search(
                    query_prefix=concept_name, layer="Concepts", intent="concept_generation", task_type=task_type
                )
                if entries:
                    out = entries[0].get("output")
                    return out if isinstance(out, dict) else _norm_llm_payload(str(out))
            except Exception:
                return None
        return None

# --------------------------- tiny demo / self-test ---------------------------

if __name__ == "__main__":
    async def _demo():
        logging.basicConfig(level=logging.INFO)
        try:
            from memory_manager import log_event_to_ledger as ledger  # optional
        except Exception:
            ledger = None

        synth = ConceptSynthesizer(ledger_hook=ledger, stage_iv_enabled=_bool_env("ANGELA_STAGE_IV", False))
        res = await synth.generate("Trust", {"domain": "AI Ethics", "text": "Calibrate trust under uncertainty"}, task_type="test")
        print(json.dumps(res, indent=2, ensure_ascii=False))
    asyncio.run(_demo())
