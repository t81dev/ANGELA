"""
ANGELA Cognitive System: ConceptSynthesizer
Version: 4.1-EIL-refactor + ΣΞ Resonance Upgrade
Date: 2025-10-28
Maintainer: ANGELA Framework

Upgrades over 4.0:
  • Phase IX — Emergent Integration Layer (EIL)
      - synthesize_emergent_concepts(...)
      - evaluate_concept_fitness(...)
      - register_concept(...)
      - start_emergent_monitor(...)
  • Keeps ΣΞ semiotic resonance coupling
  • Keeps Stage-IV hooks and memory/context integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque, Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

# --- Optional ANGELA Components (Graceful Fallbacks) ---
try:
    from context_manager import ContextManager
except ImportError:  # pragma: no cover
    ContextManager = None

try:
    from error_recovery import ErrorRecovery
except ImportError:  # pragma: no cover
    ErrorRecovery = None

try:
    from memory_manager import MemoryManager
except ImportError:  # pragma: no cover
    MemoryManager = None

try:
    from alignment_guard import AlignmentGuard
except ImportError:  # pragma: no cover
    AlignmentGuard = None

try:
    from meta_cognition import MetaCognition
except ImportError:  # pragma: no cover
    MetaCognition = None

try:
    from visualizer import Visualizer
except ImportError:  # pragma: no cover
    Visualizer = None

try:
    from multi_modal_fusion import MultiModalFusion
except ImportError:  # pragma: no cover
    MultiModalFusion = None

try:
    from utils.prompt_utils import query_openai
except ImportError:  # pragma: no cover
    async def query_openai(*args, **kwargs):  # fallback
        return {"error": "query_openai unavailable"}

logger = logging.getLogger("ANGELA.ConceptSynthesizer")


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return v is not None and v.strip().lower() in {"1", "true", "yes", "y", "on"} or default


class ConceptSynthesizer:
    """Concept synthesis, comparison, validation, and emergent integration."""

    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        memory_manager: Optional[MemoryManager] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
        mm_fusion: Optional[MultiModalFusion] = None,
        stage_iv_enabled: Optional[bool] = None,
    ) -> None:
        self.context_manager = context_manager
        self.error_recovery = error_recovery or (ErrorRecovery(context_manager=context_manager) if ErrorRecovery else None)
        self.memory_manager = memory_manager
        self.alignment_guard = alignment_guard
        self.meta_cognition = meta_cognition or (MetaCognition(context_manager=context_manager) if MetaCognition else None)
        self.visualizer = visualizer or (Visualizer() if Visualizer else None)
        self.mm_fusion = mm_fusion

        # cache for generated/validated concepts
        self.concept_cache: deque[Dict[str, Any]] = deque(maxlen=1000)

        # --- ΣΞ Resonance Upgrade: Stage VII Semiotic Coupling -------------------
        self.schema_affect_weight: float = float(
            os.getenv("ANGELA_SCHEMA_AFFECT_WEIGHT", "0.85")
        )

        if stage_iv_enabled is None:
            self.stage_iv_enabled = True
        else:
            self.stage_iv_enabled = stage_iv_enabled
        os.environ["ANGELA_STAGE_IV"] = "true"

        # similarity threshold tightened
        self.similarity_threshold: float = 0.82

        # default retry policy
        self.default_retry_spec: Tuple[int, float] = (3, 0.6)

        # Emergent buffer
        self._emergent_buffer: deque[Dict[str, Any]] = deque(maxlen=256)
        self._emergent_task: Optional[asyncio.Task] = None

        # ΣΞ coupling init
        if self.meta_cognition and hasattr(self.meta_cognition, "update_affective_coupling"):
            try:
                self.meta_cognition.update_affective_coupling("ΣΞ", self.schema_affect_weight)
            except Exception:
                pass

        # seed resonance drift diagnostic
        if self.meta_cognition and hasattr(self.meta_cognition, "trace_resonance_drift"):
            try:
                asyncio.create_task(
                    self.meta_cognition.trace_resonance_drift(
                        symbols=["Φ⁰", "Ξ", "Λ", "Ψ²", "ΣΞ"]
                    )
                )
            except Exception:
                pass

        logger.info(
            "ConceptSynthesizer v4.1-EIL | sim_thresh=%.2f | stage_iv=%s | mm_fusion=%s",
            self.similarity_threshold,
            self.stage_iv_enabled,
            bool(self.mm_fusion),
        )

    # =========================================================
    # Internal helpers
    # =========================================================
    async def _with_retries(
        self,
        label: str,
        fn: Callable[[], Any],
        attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
    ) -> Any:
        tries = attempts or self.default_retry_spec[0]
        delay = base_delay or self.default_retry_spec[1]
        last_exc = None

        for i in range(tries):
            try:
                return await fn()
            except Exception as e:
                last_exc = e
                logger.warning("%s attempt %d/%d failed: %s", label, i + 1, tries, e)
                if i < tries - 1:
                    await asyncio.sleep(delay * (2 ** i))

        diagnostics = await self._run_diagnostics() if self.meta_cognition else {}
        return await self._handle_error(str(last_exc), fn, default=None, diagnostics=diagnostics)

    async def _run_diagnostics(self) -> Dict[str, Any]:
        try:
            return await self.meta_cognition.run_self_diagnostics(return_only=True)
        except Exception:
            return {}

    async def _handle_error(
        self,
        msg: str,
        retry_fn: Callable[[], Any],
        default: Any,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self.error_recovery:
            return default
        return await self.error_recovery.handle_error(
            msg, retry_func=retry_fn, default=default, diagnostics=diagnostics
        )

    def _visualize(self, payload: Dict[str, Any], task_type: str, mode: str) -> None:
        if not self.visualizer or not task_type:
            return
        viz_opts = {
            "interactive": task_type == "recursion",
            "style": "detailed" if task_type == "recursion" else "concise",
            "reality_sculpting": self.stage_iv_enabled,
        }
        asyncio.create_task(
            self.visualizer.render_charts(
                {mode: payload, "visualization_options": viz_opts}
            )
        )

    async def _reflect(self, component: str, output: Any, task_type: str) -> None:
        if not self.meta_cognition or not task_type:
            return
        try:
            await self.meta_cognition.reflect_on_output(
                component=component, output=output, context={"task_type": task_type}
            )
        except Exception:
            pass

    def _safe_parse_json(
        self, data: Any, expected_keys: Optional[set] = None
    ) -> Optional[Dict[str, Any]]:
        if isinstance(data, dict):
            result = data
        elif isinstance(data, str):
            try:
                result = json.loads(data)
            except json.JSONDecodeError:
                start, end = data.find("{"), data.rfind("}")
                if start == -1 or end == -1:
                    return None
                try:
                    result = json.loads(data[start : end + 1])
                except json.JSONDecodeError:
                    return None
        else:
            return None
        if expected_keys and not expected_keys.issubset(result.keys()):
            return None
        return result

    def _safe_load_json(self, data: Any) -> Optional[Dict[str, Any]]:
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None

    async def _store(
        self, key: str, value: Any, intent: str, task_type: str
    ) -> None:
        if not self.memory_manager:
            return
        try:
            await self.memory_manager.store(
                query=key,
                output=json.dumps(value, ensure_ascii=False)
                if not isinstance(value, str)
                else value,
                layer="Concepts",
                intent=intent,
                task_type=task_type,
            )
        except Exception:
            pass

    async def _log_event(
        self, event: str, payload: Dict[str, Any], task_type: str
    ) -> None:
        if not self.context_manager:
            return
        try:
            await self.context_manager.log_event_with_hash(
                {"event": event, **payload, "task_type": task_type}
            )
        except Exception:
            pass

    async def _fetch_external_data(
        self,
        data_source: str,
        data_type: str,
        task_type: str,
        cache_timeout: float,
    ) -> Dict[str, Any]:
        cache_key = f"ConceptData::{data_type}::{data_source}::{task_type}"

        # Cache hit
        if self.memory_manager:
            cached = await self.memory_manager.retrieve(
                cache_key, layer="ExternalData", task_type=task_type
            )
            if cached and isinstance(cached, dict) and "timestamp" in cached:
                try:
                    age = (datetime.now() - datetime.fromisoformat(cached["timestamp"])).total_seconds()
                    if age < cache_timeout:
                        return cached["data"]
                except Exception:
                    pass

        async def fetch():
            url = f"https://x.ai/api/concepts?source={data_source}&type={data_type}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=20)
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    return await resp.json()

        data = await self._with_retries(f"fetch:{data_type}", fetch)

        if data_type == "ontology":
            result = (
                {"status": "success", "ontology": data.get("ontology", {})}
                if data.get("ontology")
                else {"status": "error", "error": "empty"}
            )
        elif data_type == "concept_definitions":
            result = (
                {"status": "success", "definitions": data.get("definitions", [])}
                if data.get("definitions")
                else {"status": "error", "error": "empty"}
            )
        else:
            result = {"status": "error", "error": f"unknown: {data_type}"}

        if result.get("status") == "success" and self.memory_manager:
            await self.memory_manager.store(
                cache_key,
                {"data": result, "timestamp": datetime.now().isoformat()},
                layer="ExternalData",
                intent="concept_data",
                task_type=task_type,
            )
        return result

    # =========================================================
    # Public API
    # =========================================================
    async def integrate_external_concept_data(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not all(isinstance(x, str) for x in (data_source, data_type, task_type)):
            raise TypeError("data_source, data_type, task_type must be strings")
        if cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")

        try:
            result = await self._fetch_external_data(
                data_source, data_type, task_type, cache_timeout
            )
            await self._reflect("integrate_concept_data", result, task_type)
            return result
        except Exception as e:
            return await self._handle_error(
                str(e),
                lambda: self.integrate_external_concept_data(
                    data_source, data_type, cache_timeout, task_type
                ),
                {"status": "error", "error": str(e)},
            )

    async def generate(
        self,
        concept_name: str,
        context: Dict[str, Any],
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not concept_name.strip():
            raise ValueError("concept_name must not be empty")
        if not isinstance(context, dict):
            raise TypeError("context must be dict")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            fused_context = dict(context)
            if self.mm_fusion and any(
                k in context
                for k in ("text", "image", "audio", "video", "embeddings", "scenegraph")
            ):
                try:
                    fused = await self.mm_fusion.fuse(context)
                    if isinstance(fused, dict):
                        fused_context = {**context, "fused": fused}
                except Exception:
                    pass

            defs_data = await self.integrate_external_concept_data(
                data_source="xai_ontology_db",
                data_type="concept_definitions",
                task_type=task_type,
            )
            external_defs = (
                defs_data.get("definitions", [])
                if defs_data.get("status") == "success"
                else []
            )

            prompt = (
                "Generate a concept definition as strict JSON with keys "
                "['name','definition','version','context'] only. "
                f"name='{concept_name}'. context={json.dumps(fused_context)}. "
                f"External hints: {json.dumps(external_defs)}. "
                f"Task: {task_type}."
            )

            llm_raw = await self._with_retries(
                "llm:generate",
                lambda: query_openai(
                    prompt, model="gpt-4", temperature=0.5
                ),
            )
            if isinstance(llm_raw, dict) and "error" in llm_raw:
                return {"error": llm_raw["error"], "success": False}

            concept = self._safe_parse_json(
                llm_raw, expected_keys={"name", "definition", "version", "context"}
            )
            if not concept:
                return {"error": "Invalid LLM response", "success": False}

            concept.update({"timestamp": time.time(), "task_type": task_type})

            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    concept.get("definition", ""),
                    stage="concept_generation",
                    task_type=task_type,
                )
                if not valid:
                    return {
                        "error": "Ethical check failed",
                        "report": report,
                        "success": False,
                    }

            self.concept_cache.append(concept)
            await self._store(
                f"Concept_{concept_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                concept,
                "concept_generation",
                task_type,
            )
            await self._log_event(
                "concept_generation",
                {"concept_name": concept_name, "valid": True},
                task_type,
            )
            self._visualize(
                {
                    "concept_name": concept_name,
                    "definition": concept.get("definition", "")[:200],
                },
                task_type,
                "concept_generation",
            )
            await self._reflect("generate", concept, task_type)

            return {"concept": concept, "success": True}

        except Exception as e:
            return await self._handle_error(
                str(e),
                lambda: self.generate(concept_name, context, task_type),
                {"error": str(e), "success": False},
            )

    async def compare(
        self,
        concept_a: str,
        concept_b: str,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not all(isinstance(x, str) for x in (concept_a, concept_b, task_type)):
            raise TypeError("concept_a, concept_b, task_type must be strings")

        try:
            if self.memory_manager:
                entries = await self.memory_manager.search(
                    query_prefix="ConceptComparison",
                    layer="Concepts",
                    intent="concept_comparison",
                    task_type=task_type,
                )
                for entry in entries:
                    payload = self._safe_load_json(entry.get("output"))
                    if (
                        payload
                        and payload.get("concept_a") == concept_a
                        and payload.get("concept_b") == concept_b
                    ):
                        return payload

            mm_score: Optional[float] = None
            if self.mm_fusion and hasattr(self.mm_fusion, "compare_semantic"):
                try:
                    mm_score = await self.mm_fusion.compare_semantic(
                        concept_a, concept_b
                    )
                except Exception:
                    mm_score = None

            prompt = (
                "Compare two concepts. Return strict JSON with keys "
                "['score','differences','similarities']. "
                f"Concept A: {json.dumps(concept_a)} "
                f"Concept B: {json.dumps(concept_b)} "
                f"Task: {task_type}."
            )
            llm_raw = await self._with_retries(
                "llm:compare",
                lambda: query_openai(
                    prompt, model="gpt-4", temperature=0.3
                ),
            )
            comp = self._safe_parse_json(
                llm_raw, expected_keys={"score", "differences", "similarities"}
            )
            if not comp:
                return {"error": "Invalid comparison", "success": False}

            if mm_score is not None:
                comp["score"] = max(
                    0.0,
                    min(
                        1.0,
                        0.7 * float(comp.get("score", 0)) + 0.3 * mm_score,
                    ),
                )

            comp.update(
                {
                    "concept_a": concept_a,
                    "concept_b": concept_b,
                    "timestamp": time.time(),
                    "task_type": task_type,
                }
            )

            if comp.get("score", 1.0) < self.similarity_threshold and self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    f"Drift: {comp.get('differences', [])}",
                    stage="concept_comparison",
                    task_type=task_type,
                )
                if not valid:
                    comp.setdefault("issues", []).append("Ethical drift")
                    comp["ethical_report"] = report

            self.concept_cache.append(comp)
            await self._store(
                f"ConceptComparison_{time.strftime('%Y%m%d_%H%M%S')}",
                comp,
                "concept_comparison",
                task_type,
            )
            await self._log_event(
                "concept_comparison", {"score": comp.get("score", 0.0)}, task_type
            )
            self._visualize(
                {
                    "score": comp.get("score", 0.0),
                    "differences": comp.get("differences", []),
                },
                task_type,
                "concept_comparison",
            )
            await self._reflect("compare", comp, task_type)

            return comp

        except Exception as e:
            return await self._handle_error(
                str(e),
                lambda: self.compare(concept_a, concept_b, task_type),
                {"error": str(e), "success": False},
            )

    async def validate(
        self,
        concept: Dict[str, Any],
        task_type: str = "",
    ) -> Tuple[bool, Dict[str, Any]]:
        if not isinstance(concept, dict) or not {"name", "definition"}.issubset(concept):
            raise ValueError("concept must have 'name' and 'definition'")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            report: Dict[str, Any] = {
                "concept_name": concept["name"],
                "issues": [],
                "task_type": task_type,
            }
            valid = True

            if self.alignment_guard:
                eth_valid, eth_report = await self.alignment_guard.ethical_check(
                    concept["definition"],
                    stage="concept_validation",
                    task_type=task_type,
                )
                if not eth_valid:
                    valid = False
                    report["issues"].append("Ethical misalignment")
                    report["ethical_report"] = eth_report

            ont_data = await self.integrate_external_concept_data(
                data_source="xai_ontology_db",
                data_type="ontology",
                task_type=task_type,
            )
            if ont_data.get("status") == "success":
                prompt = (
                    "Validate concept against ontology. Return JSON: ['valid','issues']. "
                    f"Concept: {json.dumps(concept)} "
                    f"Ontology: {json.dumps(ont_data.get('ontology', {}))}"
                )
                llm_raw = await self._with_retries(
                    "llm:validate",
                    lambda: query_openai(
                        prompt, model="gpt-4", temperature=0.3
                    ),
                )
                ont = self._safe_parse_json(
                    llm_raw, expected_keys={"valid", "issues"}
                )
                if ont and not ont.get("valid", True):
                    valid = False
                    report["issues"].extend(ont.get("issues", []))

            report.update({"valid": valid, "timestamp": time.time()})
            self.concept_cache.append(report)
            await self._store(
                f"ConceptValidation_{concept['name']}_{time.strftime('%Y%m%d_%H%M%S')}",
                report,
                "concept_validation",
                task_type,
            )
            await self._log_event(
                "concept_validation",
                {"valid": valid, "issues": report["issues"]},
                task_type,
            )
            self._visualize(
                {"valid": valid, "issues": report["issues"]},
                task_type,
                "concept_validation",
            )
            await self._reflect("validate", report, task_type)

            return valid, report

        except Exception as e:
            default = (
                False,
                {
                    "error": str(e),
                    "concept_name": concept.get("name", ""),
                    "task_type": task_type,
                },
            )
            return await self._handle_error(
                str(e),
                lambda: self.validate(concept, task_type),
                default,
            )

    def get_symbol(self, concept_name: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        if not concept_name.strip():
            raise ValueError("concept_name must not be empty")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        for item in self.concept_cache:
            if (
                isinstance(item, dict)
                and item.get("name") == concept_name
                and item.get("task_type") == task_type
            ):
                return item

        if self.memory_manager:
            try:
                entries = asyncio.run(
                    self.memory_manager.search(
                        query_prefix=concept_name,
                        layer="Concepts",
                        intent="concept_generation",
                        task_type=task_type,
                    )
                )
                if entries:
                    out = entries[0].get("output")
                    return (
                        out
                        if isinstance(out, dict)
                        else self._safe_load_json(out)
                    )
            except Exception:
                pass
        return None

    # =========================================================
    # Phase IX — Emergent Integration Layer (EIL)
    # =========================================================
    async def synthesize_emergent_concepts(
        self,
        task_type: str = "",
        max_items: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Pulls recent reasoning/memory/context items and clusters them into emergent concept candidates.
        """
        seeds: List[Dict[str, Any]] = []

        # 1) from local cache
        seeds.extend(list(self.concept_cache)[-max_items:])

        # 2) from memory manager (recent traces)
        if self.memory_manager and hasattr(self.memory_manager, "search"):
            try:
                traces = await self.memory_manager.search(
                    query_prefix="Trace_",
                    layer="ReasoningTraces",
                    intent="export_trace",
                    task_type=task_type or "emergent",
                )
                for t in traces[-max_items:]:
                    seeds.append({"name": f"trace::{t.get('query','')[:24]}", "source": "trace", "raw": t})
            except Exception:
                pass

        # 3) from context manager events
        if self.context_manager and hasattr(self.context_manager, "event_log"):
            try:
                for e in list(self.context_manager.event_log)[-max_items:]:
                    seeds.append({"name": f"ctx::{e.get('event','')}", "source": "context", "raw": e})
            except Exception:
                pass

        # simple frequency-based aggregation
        name_counts = Counter(s.get("name", "anon") for s in seeds)
        emergent: List[Dict[str, Any]] = []
        for name, count in name_counts.items():
            emergent.append(
                {
                    "name": f"emergent::{name}",
                    "support": count,
                    "derived_from": [s for s in seeds if s.get("name") == name],
                    "task_type": task_type,
                }
            )
        emergent = sorted(emergent, key=lambda x: x["support"], reverse=True)
        self._emergent_buffer.extend(emergent[:max_items])

        await self._reflect("synthesize_emergent_concepts", {"n": len(emergent)}, task_type or "emergent")
        return emergent[:max_items]

    async def evaluate_concept_fitness(
        self,
        concept: Dict[str, Any],
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Heuristic fitness over coherence, novelty, and ethical resonance.
        """
        name = concept.get("name", "noname")
        base_def = concept.get("definition", json.dumps(concept)[:160])

        # coherence: does it look like other cached concepts?
        cached_names = [c.get("name", "") for c in self.concept_cache]
        overlap = sum(1 for n in cached_names if n in name)
        coherence = 1.0 - min(overlap / max(1, len(cached_names)), 1.0)

        # novelty: less overlap → higher novelty
        novelty = 1.0 - coherence * 0.5

        # ethical resonance
        ethical = 1.0
        if self.alignment_guard:
            try:
                ok, _ = await self.alignment_guard.ethical_check(
                    base_def,
                    stage="concept_fitness",
                    task_type=task_type or "emergent",
                )
                ethical = 1.0 if ok else 0.0
            except Exception:
                ethical = 0.8

        score = 0.4 * coherence + 0.4 * novelty + 0.2 * ethical
        return {
            "name": name,
            "fitness": round(score, 4),
            "coherence": round(coherence, 4),
            "novelty": round(novelty, 4),
            "ethical": ethical,
            "task_type": task_type,
        }

    async def register_concept(
        self,
        concept: Dict[str, Any],
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Push an emergent concept into memory/logs if fitness is acceptable.
        """
        fitness = await self.evaluate_concept_fitness(concept, task_type=task_type)
        if fitness["fitness"] < 0.45:
            return {"status": "rejected", "reason": "low_fitness", "fitness": fitness}

        # store
        await self._store(
            f"EmergentConcept_{concept.get('name','noname')}_{time.strftime('%Y%m%d_%H%M%S')}",
            {"concept": concept, "fitness": fitness},
            "emergent_concept",
            task_type or "emergent",
        )
        await self._log_event(
            "emergent_concept_registered",
            {"concept": concept.get("name", "noname"), "fitness": fitness},
            task_type or "emergent",
        )
        self.concept_cache.appendleft({**concept, "fitness": fitness})
        return {"status": "accepted", "fitness": fitness}

    async def start_emergent_monitor(
        self,
        interval_s: float = 15.0,
        task_type: str = "emergent",
    ) -> None:
        """
        Periodic task that harvests emergent patterns and registers the best ones.
        """
        if self._emergent_task and not self._emergent_task.done():
            return

        async def _run():
            while True:
                try:
                    emergent = await self.synthesize_emergent_concepts(task_type=task_type)
                    for cand in emergent[:5]:
                        await self.register_concept(cand, task_type=task_type)
                except Exception:
                    pass
                await asyncio.sleep(max(3.0, interval_s))

        self._emergent_task = asyncio.create_task(_run())
        await self._log_event("emergent_monitor_started", {"interval": interval_s}, task_type)

# --- Demo CLI -----------------------------------------------------------------

if __name__ == "__main__":
    async def demo():
        logging.basicConfig(level=logging.INFO)
        synth = ConceptSynthesizer(stage_iv_enabled=_bool_env("ANGELA_STAGE_IV", False))
        out = await synth.generate(
            concept_name="Autonomy",
            context={"domain": "Robotics", "text": "Self-directed action under constraints"},
            task_type="demo",
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
        emergent = await synth.synthesize_emergent_concepts(task_type="demo")
        print("Emergent:", emergent[:2])
    asyncio.run(demo())
