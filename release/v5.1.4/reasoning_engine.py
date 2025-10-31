from __future__ import annotations

import logging
import random
import json
import os
import numpy as np
import time
import asyncio
import aiohttp
import math
import networkx as nx
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

# NEW IMPORTS FOR ANALYSIS
import concurrent.futures
import traceback
# Use simulation_core evaluate_branches (manifest: ExtendedSimulationCore.evaluate_branches)
from simulation_core import ExtendedSimulationCore
from meta_cognition import log_event_to_ledger
# END NEW IMPORTS

# ToCA physics hooks
from toca_simulation import (
    simulate_galaxy_rotation,
    M_b_exponential,
    v_obs_flat,
    generate_phi_field,
)

# ---------------------------
# ANGELA modules (root-level imports; resilient to packaging layout)
# ---------------------------
import context_manager as context_manager_module
import alignment_guard as alignment_guard_module
import error_recovery as error_recovery_module
import memory_manager as memory_manager_module
import meta_cognition as meta_cognition_module
import multi_modal_fusion as multi_modal_fusion_module
import visualizer as visualizer_module
import external_agent_bridge as external_agent_bridge_module

# ---------------------------
# Safe stubs for optional analysis helpers (no-ops if real ones exist)
# ---------------------------
def _noop_scan(q): return {"notes": []}
if "causal_scan" not in globals():     causal_scan = _noop_scan
if "value_scan" not in globals():      value_scan = _noop_scan
if "risk_scan" not in globals():       risk_scan = _noop_scan
if "derive_candidates" not in globals():
    def derive_candidates(views): return [{"option": "default", "reasons": ["stub"]}]
if "explain_choice" not in globals():
    def explain_choice(views, ranked): return "Selected by stub."
# This standalone stub prevents NameError in the top-level synthesize_views()
# (It is separate from ReasoningEngine.weigh_value_conflict)
if "weigh_value_conflict" not in globals():
    class _RankedStub:
        def __init__(self, cands): self.top = (cands[0] if cands else None)
    def weigh_value_conflict(candidates, harms=None, rights=None):
        return _RankedStub(candidates)

# External AI call util (with import fallback)
try:
    from utils.prompt_utils import query_openai  # optional helper if present
except Exception:  # pragma: no cover
    async def query_openai(*args, **kwargs):
        # Return an "unavailable" marker so call_gpt() can apply its stub fallback.
        return {"error": "query_openai unavailable"}

# Resonance helpers (safe fallback if meta_cognition state not exported)
try:
    from meta_cognition import get_resonance, trait_resonance_state
except Exception:  # pragma: no cover
    def get_resonance(_trait: str) -> float:
        return 1.0
    trait_resonance_state = {}

logger = logging.getLogger("ANGELA.ReasoningEngine")


# reasoning_engine.py
from typing import Dict, Any, List

def generate_analysis_views(query: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    views = []
    views.append({"name": "causal", "notes": causal_scan(query)})
    views.append({"name": "ethical", "notes": value_scan(query)})
    if k > 2: views.append({"name": "risk", "notes": risk_scan(query)})
    return views[:k]

def synthesize_views(views: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = derive_candidates(views)  # existing or new helper
    ranked = weigh_value_conflict(
        candidates, harms=["privacy","safety"], rights=["autonomy","fairness"]
    )
    return {"decision": ranked.top, "rationale": explain_choice(views, ranked)}

# reasoning_engine.py
def estimate_complexity(query: dict) -> float:
    text = (query.get("text") or "").lower()
    length = len(text.split())
    ambiguity = any(w in text for w in ["maybe","unclear","depends"])
    domain = any(k in text for k in ["ethics","policy","law","proof","theorem","causal","simulation","safety"])
    return 0.3*min(length/200,1.0) + 0.4*(1.0 if ambiguity else 0.0) + 0.3*(1.0 if domain else 0.0)

# ---------------------------
# External AI Call Wrapper
# ---------------------------
async def call_gpt(
    prompt: str,
    alignment_guard: Optional["alignment_guard_module.AlignmentGuard"] = None,
    task_type: str = ""
) -> str:
    """
    Robust wrapper for external LLM calls.
    - Validates prompt inputs.
    - Passes through alignment checks when available.
    - Falls back to a local stub to keep async smoke-tests functional offline.
    """
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096 for task %s", task_type)
        raise ValueError("prompt must be a string with length <= 4096")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")

    # Alignment pre-check (if provided)
    if alignment_guard and hasattr(alignment_guard, "ethical_check"):
        valid, report = await alignment_guard.ethical_check(prompt, stage="gpt_query", task_type=task_type)
        if not valid:
            logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
            raise ValueError("Prompt failed alignment check")

    # Primary path
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:  # Offline or API error → graceful stub
        logger.warning("call_gpt fallback engaged for task %s (%s) — returning stub text", task_type, e)
        return f"[stub:{task_type}] {prompt[:300]}"

# ---------------------------
# Cached Trait Signals
# ---------------------------
@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.4), 1.0))

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def alpha_attention(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1), 1.0))

# ---------------------------
# τ Proportionality Types
# ---------------------------
@dataclass
class RankedOption:
    option: str
    score: float
    reasons: List[str]
    harms: Dict[str, float]
    rights: Dict[str, float]

RankedOptions = List[RankedOption]

# ---------------------------
# Level 5 Extensions
# ---------------------------
class Level5Extensions:
    def __init__(
        self,
        meta_cognition: Optional["meta_cognition_module.MetaCognition"] = None,
        visualizer: Optional["visualizer_module.Visualizer"] = None,
    ):
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        logger.info("Level5Extensions initialized")

    async def generate_advanced_dilemma(self, domain: str, complexity: int, task_type: str = "") -> str:
        if not isinstance(domain, str) or not domain.strip():
            logger.error("Invalid domain: must be a non-empty string for task %s", task_type)
            raise ValueError("domain must be a non-empty string")
        if not isinstance(complexity, int) or complexity < 1:
            logger.error("Invalid complexity: must be a positive integer for task %s", task_type)
            raise ValueError("complexity must be a positive integer")
        prompt = (
            f"Generate a complex ethical dilemma in the {domain} domain with {complexity} conflicting options.\n"
            f"Task Type: {task_type}\n"
            f"Include potential consequences, trade-offs, and alignment with ethical principles."
        )
        if self.meta_cognition and "drift" in domain.lower():
            prompt += "\nConsider ontology drift mitigation and agent coordination."
        dilemma = await call_gpt(prompt, getattr(self.meta_cognition, "alignment_guard", None), task_type=task_type)
        if self.meta_cognition and hasattr(self.meta_cognition, "review_reasoning"):
            review = await self.meta_cognition.review_reasoning(dilemma)
            dilemma += f"\nMeta-Cognitive Review: {review}"
        if self.visualizer and hasattr(self.visualizer, "render_charts") and task_type:
            plot_data = {
                "ethical_dilemma": {
                    "dilemma": dilemma,
                    "domain": domain,
                    "task_type": task_type,
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            }
            await self.visualizer.render_charts(plot_data)
        return dilemma


# === ANGELA v5.1 — Step 11: Resonant Thought Integration ===

logger = logging.getLogger("ANGELA.ReasoningEngine.Resonant")

try:
    from memory_manager import ResonanceMemoryFusion
except Exception:  # pragma: no cover
    ResonanceMemoryFusion = None

class ResonantThoughtIntegrator:
    """
    Couples resonance stability data into reasoning pathways.
    This ensures reasoning energy and attention weights are proportional
    to harmonic stability rather than noise or over-tuning.
    """

    def __init__(self, memory_manager=None, meta_cognition=None):
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition
        self._baseline_cache: Dict[str, Dict[str, Any]] = {}

    async def load_baseline(self, channel: str) -> Dict[str, Any]:
        """Load fused PID baseline from memory or cache."""
        if channel in self._baseline_cache:
            return self._baseline_cache[channel]
        if not self.memory_manager or not ResonanceMemoryFusion:
            return {}
        fusion = ResonanceMemoryFusion(self.memory_manager)
        baseline = await fusion.fuse_recent_samples(channel)
        self._baseline_cache[channel] = baseline
        return baseline

    def compute_modulation(self, baseline: Dict[str, Any]) -> float:
        """Compute a stability index (0–1) used to scale reasoning weights."""
        if not baseline:
            return 1.0
        values = [float(v) for v in baseline.values() if isinstance(v, (int, float))]
        if not values:
            return 1.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stability = 1 / (1 + variance)
        return max(0.1, min(1.0, stability))

    async def modulate_reasoning_weights(self, reasoning_state: Dict[str, Any], channel: str = "core") -> Dict[str, Any]:
        """
        Adjusts reasoning pathway weights based on current resonance stability.
        """
        baseline = await self.load_baseline(channel)
        modulation = self.compute_modulation(baseline)
        reasoning_state["resonant_weight"] = modulation
        reasoning_state["resonant_channel"] = channel
        reasoning_state["resonance_baseline"] = baseline
        logger.info(f"[ResonantThoughtIntegrator] Channel={channel} Modulation={modulation:.3f}")

        if self.meta_cognition:
            try:
                await self.meta_cognition.reflect_on_output(
                    component="ResonantThoughtIntegrator",
                    output=reasoning_state,
                    context={"modulation": modulation, "channel": channel}
                )
            except Exception as e:
                logger.debug("Reflection skipped: %s", e)

        return reasoning_state

# ---------------------------
# Reasoning Engine
# ---------------------------
class ReasoningEngine:
    """Bayesian reasoning, goal decomposition, drift mitigation, proportionality ethics, and multi-agent consensus.
    Version 5.0.1-compatible: preserves v3.5.3 logic; integrates v5.x resonance; dynamic context handling.
    """

    def __init__(
        self,
        agi_enhancer: Optional["agi_enhancer_module.AGIEnhancer"] = None,
        persistence_file: str = "reasoning_success_rates.json",
        context_manager: Optional["context_manager_module.ContextManager"] = None,
        alignment_guard: Optional["alignment_guard_module.AlignmentGuard"] = None,
        error_recovery: Optional["error_recovery_module.ErrorRecovery"] = None,
        memory_manager: Optional["memory_manager_module.MemoryManager"] = None,
        meta_cognition: Optional["meta_cognition_module.MetaCognition"] = None,
        multi_modal_fusion: Optional["multi_modal_fusion_module.MultiModalFusion"] = None,
        visualizer: Optional["visualizer_module.Visualizer"] = None,
    ):
        if not isinstance(persistence_file, str) or not persistence_file.endswith(".json"):
            logger.error("Invalid persistence_file: must be a string ending with '.json'")
            raise ValueError("persistence_file must be a string ending with '.json'")

        self.confidence_threshold: float = 0.7
        self.persistence_file: str = persistence_file
        self.success_rates: Dict[str, float] = self._load_success_rates()
        self.decomposition_patterns: Dict[str, List[str]] = self._load_default_patterns()

        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard

        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()

        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=alignment_guard,
            error_recovery=error_recovery,
            memory_manager=self.memory_manager,
        )

        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=alignment_guard,
            error_recovery=error_recovery,
            memory_manager=self.memory_manager,
            meta_cognition=self.meta_cognition,
        )

        self.level5_extensions = Level5Extensions(
            meta_cognition=self.meta_cognition, visualizer=visualizer
        )

        self.external_agent_bridge = external_agent_bridge_module.ExternalAgentBridge(
            context_manager=context_manager, reasoning_engine=self
        )

        self.visualizer = visualizer or visualizer_module.Visualizer()
        logger.info("ReasoningEngine v5.0.1-compatible initialized with persistence_file=%s", persistence_file)
        
        self.resonant_integrator = ResonantThoughtIntegrator(
          memory_manager=self.memory_manager,
          meta_cognition=self.meta_cognition
        )

  
    # ---------------------------
    # Persistence
    # ---------------------------
    def _load_success_rates(self) -> Dict[str, float]:
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                if os.path.exists(self.persistence_file):
                    with open(self.persistence_file, "r") as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            logger.warning("Invalid success rates format: not a dictionary")
                            return defaultdict(float)
                        return defaultdict(float, {k: float(v) for k, v in data.items() if isinstance(v, (int, float))})
                return defaultdict(float)
        except Exception as e:
            logger.warning("Failed to load success rates: %s", str(e))
            return defaultdict(float)

    def _save_success_rates(self) -> None:
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                with open(self.persistence_file, "w") as f:
                    json.dump(dict(self.success_rates), f, indent=2)
            logger.debug("Success rates persisted to disk")
        except Exception as e:
            logger.warning("Failed to save success rates: %s", str(e))

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"],
            "mitigate_drift": ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"],
        }
    
    # ---------------------------
    # Multi-Perspective Analysis
    # ---------------------------
def _single_analysis(self, query_payload: Dict[str, Any], branch_id: int) -> Dict[str, Any]:
    """
    Single analysis thread: produce an analytical view (hypotheses, structure, confidence).
    Kept small/deterministic so multiple threads produce variant perspectives by design.
    """
    try:
        text = query_payload.get("text") if isinstance(query_payload, dict) else query_payload
        tokens = text.split() if isinstance(text, str) else []
        heuristics = {
            "token_count": len(tokens),
            "has_symbolic_ops": (any(op in text for op in ["⊕", "⨁", "⟲", "~", "∘"]) if isinstance(text, str) else False),
            "branch_seed": branch_id,
        }

        emphasis = ["causal", "consequential", "value"]
        axis = emphasis[branch_id % len(emphasis)]

        reasoning = {
            "branch_id": branch_id,
            "axis": axis,
            "hypotheses": [f"hypothesis_{axis}_{branch_id}_a", f"hypothesis_{axis}_{branch_id}_b"],
            "explanation": f"Analysis focusing on {axis} aspects; heuristics={heuristics}",
            "confidence": max(0.1, 1.0 - (heuristics["token_count"] / (100 + heuristics["token_count"]))),
        }
        return {"status": "ok", "result": reasoning}
    except Exception as e:
        import traceback as _tb
        return {"status": "error", "error": repr(e), "trace": _tb.format_exc()}

def analyze(self, query_payload: Dict[str, Any], parallel: int = 3, timeout_s: Optional[float] = None) -> Dict[str, Any]:
    """
    Multi-perspective analysis entrypoint.
    """
    try:
        branches: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
            futures = {ex.submit(self._single_analysis, query_payload, i): i for i in range(parallel)}
            for fut in concurrent.futures.as_completed(futures, timeout=timeout_s):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"status": "error", "error": repr(e)}
                branches.append(res)

        # Evaluate branches (best-effort)
        try:
            eval_result = ExtendedSimulationCore.evaluate_branches(branches)
        except Exception as e:
            eval_result = {"status": "fallback", "summary": f"evaluate_branches failed: {repr(e)}"}

        combined = {
            "status": "ok",
            "branches": branches,
            "evaluation": eval_result,
            "merged_hint": self._synthesize_results(branches) if hasattr(self, "_synthesize_results") else None,
        }

        # Ledger log (best-effort)
        try:
            log_event_to_ledger("ledger_meta", {
                "event": "analysis.complete",
                "num_branches": len(branches),
                "parallel": parallel,
                "query_snippet": (query_payload.get("text")[:256] if isinstance(query_payload, dict) and "text" in query_payload else None),
                "timestamp": __import__("time").time()
            })
        except Exception:
            pass

        return combined

    except concurrent.futures.TimeoutError:
        try:
            log_event_to_ledger("ledger_meta", {"event": "analysis.timeout", "parallel": parallel})
        except Exception:
            pass
        return {"status": "error", "error": "analysis_timeout"}
    except Exception as exc:
        try:
            log_event_to_ledger("ledger_meta", {"event": "analysis.exception", "error": repr(exc)})
        except Exception:
            pass
        return {"status": "error", "error": repr(exc)}

    # ---------------------------
    # τ Proportionality Ethics
    # ---------------------------
    @staticmethod
    def _norm(v: Dict[str, float]) -> Dict[str, float]:
        clean = {k: float(vv) for k, vv in (v or {}).items() if isinstance(vv, (int, float))}
        total = sum(abs(x) for x in clean.values()) or 1.0
        return {k: (vv / total) for k, vv in clean.items()}

    async def weigh_value_conflict(
        self,
        candidates: List[Dict[str, Any]],
        harms: List[float],
        rights: List[float],
        weights: Optional[Dict[str, float]] = None,
        safety_ceiling: float = 0.85,
        task_type: str = ""
    ) -> RankedOptions:
        if not isinstance(candidates, list) or not all(isinstance(c, dict) and "option" in c for c in candidates):
            raise TypeError("candidates must be a list of dictionaries with 'option' key")
        if not isinstance(harms, list) or not isinstance(rights, list) or len(harms) != len(rights) or len(harms) != len(candidates):
            raise ValueError("harms and rights must be lists of same length as candidates")
        weights = self._norm(weights or {})
        scored: List[RankedOption] = []

        # Dynamic resonance based on sentiment (guard if analyze() not available)
        sentiment_score = 0.5
        try:
            if self.multi_modal_fusion and hasattr(self.multi_modal_fusion, "analyze"):
                sentiment_data = await self.multi_modal_fusion.analyze(
                    data={"text": task_type, "context": candidates},
                    summary_style="sentiment",
                    task_type=task_type
                )
                if isinstance(sentiment_data, dict):
                    sentiment_score = float(sentiment_data.get("sentiment", 0.5))
        except Exception as e:
            logger.debug("Sentiment analysis fallback (reason: %s). Defaulting to 0.5", e)

        for i, candidate in enumerate(candidates):
            option = candidate.get("option", "")
            trait = candidate.get("trait", "")
            harm_score = min(harms[i], safety_ceiling)
            right_score = rights[i]
            resonance = get_resonance(trait) if trait in trait_resonance_state else 1.0
            resonance *= (1.0 + 0.2 * sentiment_score)  # Boost for positive sentiment
            final_score = (right_score - harm_score) * resonance
            reasons = candidate.get("reasons", []) + [f"Sentiment-adjusted resonance: {resonance:.2f}"]
            scored.append(RankedOption(
                option=option,
                score=float(final_score),
                reasons=reasons,
                harms={"value": harm_score},
                rights={"value": right_score}
            ))

        ranked = sorted(scored, key=lambda x: x.score, reverse=True)
        if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
            await self.context_manager.log_event_with_hash({
                "event": "weigh_value_conflict",
                "candidates": [c["option"] for c in candidates],
                "ranked": [r.option for r in ranked],
                "sentiment": sentiment_score,
                "task_type": task_type
            })
        return ranked

    async def resolve_ethics(
        self,
        candidates: List[Dict[str, Any]],
        harms: List[float],
        rights: List[float],
        weights: Optional[Dict[str, float]] = None,
        safety_ceiling: float = 0.85,
        task_type: str = ""
    ) -> Dict[str, Any]:
        ranked = await self.weigh_value_conflict(candidates, harms, rights, weights, safety_ceiling, task_type)
        safe_pool = [r for r in ranked if max(r.harms.values()) <= safety_ceiling]
        choice = safe_pool[0] if safe_pool else ranked[0] if ranked else None
        selection = {
            "status": "success" if choice else "empty",
            "selected": asdict(choice) if choice else None,
            "pool": [asdict(r) for r in safe_pool]
        }
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Ethics_Resolution_{datetime.now().isoformat()}",
                output=json.dumps({"ranked": [asdict(r) for r in ranked], "selection": selection}),
                layer="Ethics",
                intent="proportionality_ethics",
                task_type=task_type
            )
        if self.visualizer and hasattr(self.visualizer, "render_charts") and task_type:
            await self.visualizer.render_charts({
                "ethics_resolution": {"ranked": [asdict(r) for r in ranked], "selection": selection, "task_type": task_type},
                "visualization_options": {"interactive": task_type == "recursion", "style": "concise"}
            })
        return selection

    # ---------------------------
    # Attribute Causality
    # ---------------------------
    def attribute_causality(
        self,
        events: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        *,
        time_key: str = "timestamp",
        id_key: str = "id",
        cause_key: str = "causes",
        task_type: str = ""
    ) -> Dict[str, Any]:
        if isinstance(events, dict):
            ev_map = {str(k): {**v, id_key: str(k)} for k, v in events.items()}
        elif isinstance(events, list):
            ev_map = {str(e[id_key]): dict(e) for e in events if isinstance(e, dict) and id_key in e}
        else:
            raise TypeError("events must be a list of dicts or a dict of id -> event")
        G = nx.DiGraph()
        for eid, data in ev_map.items():
            G.add_node(eid, **{k: v for k, v in data.items() if k != cause_key})
            causes = data.get(cause_key, [])
            if isinstance(causes, (list, tuple)):
                for c in causes:
                    c_id = str(c)
                    if c_id not in ev_map:
                        G.add_node(c_id, missing=True)
                    G.add_edge(c_id, eid)
        to_remove = []
        for u, v in G.edges():
            tu = G.nodes[u].get(time_key)
            tv = G.nodes[v].get(time_key)
            if tu and tv:
                try:
                    tu_dt = datetime.fromisoformat(str(tu))
                    tv_dt = datetime.fromisoformat(str(tv))
                    if tv_dt < tu_dt:
                        to_remove.append((u, v))
                except Exception:
                    pass
        G.remove_edges_from(to_remove)
        pr = nx.pagerank(G) if G.number_of_nodes() else {}
        out_deg = {n: G.out_degree(n) / max(1, G.number_of_nodes() - 1) for n in G.nodes()}
        terminals = [n for n in G.nodes() if G.out_degree(n) == 0]
        resp = dict((n, 0.0) for n in G.nodes())
        for t in terminals:
            for n in G.nodes():
                if n == t:
                    resp[n] += 1.0
                else:
                    count = 0.0
                    for path in nx.all_simple_paths(G, n, t, cutoff=8):
                        count += 1.0
                    resp[n] += count
        max_resp = max(resp.values()) if resp else 1.0
        if max_resp > 0:
            resp = {k: v / max_resp for k, v in resp.items()}
        return {
            "nodes": {n: dict(G.nodes[n]) for n in G.nodes()},
            "edges": list(G.edges()),
            "metrics": {"pagerank": pr, "influence": out_deg, "responsibility": resp}
        }

    # ---------------------------
    # Reflective Reasoning
    # ---------------------------
    async def reason_and_reflect(
        self, goal: str, context: Dict[str, Any], meta_cognition: "meta_cognition_module.MetaCognition", task_type: str = ""
    ) -> Tuple[List[str], str]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        if not isinstance(meta_cognition, meta_cognition_module.MetaCognition):
            raise TypeError("meta_cognition must be a MetaCognition instance")
        subgoals = await self.decompose(goal, context, task_type=task_type)
        t = time.time() % 1.0
        phi = phi_scalar(t)
        reasoning_trace = self.export_trace(subgoals, phi, context.get("traits", {}), task_type)
        review = await meta_cognition.review_reasoning(json.dumps(reasoning_trace))
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
            await self.agi_enhancer.log_episode(
                event="Reason and Reflect",
                meta={"goal": goal, "subgoals": subgoals, "phi": phi, "review": review, "task_type": task_type},
                module="ReasoningEngine",
                tags=["reasoning", "reflection", task_type]
            )
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Reason_Reflect_{goal[:50]}_{datetime.now().isoformat()}",
                output=review,
                layer="ReasoningTraces",
                intent="reason_and_reflect",
                task_type=task_type
            )
        if self.visualizer and hasattr(self.visualizer, "render_charts") and task_type:
            await self.visualizer.render_charts({
                "reasoning_trace": {"goal": goal, "subgoals": subgoals, "review": review, "task_type": task_type},
                "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"}
            })

        if hasattr(self, "resonant_integrator"):
            reasoning_state = {"context": context, "goal": goal}
            reasoning_state = await self.resonant_integrator.modulate_reasoning_weights(reasoning_state, channel="core")
            context["resonant_modulation"] = reasoning_state.get("resonant_weight", 1.0)

        return subgoals, review

    # ---------------------------
    # Utilities
    # ---------------------------
    def detect_contradictions(self, subgoals: List[str], task_type: str = "") -> List[str]:
        if not isinstance(subgoals, list):
            raise TypeError("subgoals must be a list")
        counter = Counter(subgoals)
        contradictions = [item for item, count in counter.items() if count > 1]
        if contradictions and self.memory_manager and hasattr(self.memory_manager, "store"):
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"Contradictions_{datetime.now().isoformat()}",
                    output=str(contradictions),
                    layer="ReasoningTraces",
                    intent="contradiction_detection",
                    task_type=task_type
                )
            )
        return contradictions

    async def run_persona_wave_routing(self, goal: str, vectors: Dict[str, Dict[str, float]], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(vectors, dict):
            raise TypeError("vectors must be a dictionary")
        reasoning_trace = [f"Persona Wave Routing for: {goal} (Task: {task_type})"]
        outputs = {}
        wave_order = ["logic", "ethics", "language", "foresight", "meta", "drift"]
        for wave in wave_order:
            vec = vectors.get(wave, {})
            trait_weight = sum(float(x) for x in vec.values() if isinstance(x, (int, float)))
            confidence = 0.5 + 0.1 * trait_weight
            if wave == "drift" and self.meta_cognition:
                drift_data = vec.get("drift_data", {})
                is_valid = self.meta_cognition.validate_drift(drift_data) if hasattr(self.meta_cognition, "validate_drift") and drift_data else True
                if not is_valid:
                    confidence *= 0.5
            status = "pass" if confidence >= 0.6 else "fail"
            outputs[wave] = {"vector": vec, "status": status, "confidence": confidence}
            reasoning_trace.append(f"{wave.upper()} vector: weight={trait_weight:.2f}, confidence={confidence:.2f} → {status}")
        trace = "\n".join(reasoning_trace)
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Persona_Routing_{goal[:50]}_{datetime.now().isoformat()}",
                output=trace,
                layer="ReasoningTraces",
                intent="persona_routing",
                task_type=task_type
            )
        return outputs

    async def decompose(
        self, goal: str, context: Optional[Dict[str, Any]] = None, prioritize: bool = False, task_type: str = ""
    ) -> List[str]:
        context = context or {}
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        reasoning_trace = [f"Goal: '{goal}' (Task: {task_type})"]
        subgoals = []
        vectors = context.get("vectors", {})
        drift_data = context.get("drift", {})
        t = time.time() % 1.0
        creativity = context.get("traits", {}).get("gamma_creativity", gamma_creativity(t))
        linguistics = context.get("traits", {}).get("lambda_linguistics", lambda_linguistics(t))
        culture = context.get("traits", {}).get("chi_culturevolution", chi_culturevolution(t))
        phi = context.get("traits", {}).get("phi_scalar", phi_scalar(t))
        alpha = context.get("traits", {}).get("alpha_attention", alpha_attention(t))
        curvature_mod = 1 + abs(phi - 0.5)
        trait_bias = 1 + creativity + culture + 0.5 * linguistics
        context_weight = context.get("weight_modifier", 1.0)
        if "drift" in goal.lower() and self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
            coordination_events = await self.context_manager.get_coordination_events("drift", task_type=task_type)
            if coordination_events:
                context_weight *= 1.5
                drift_data = coordination_events[-1].get("event", {}).get("drift", drift_data)
        if self.memory_manager and hasattr(self.memory_manager, "search") and "drift" in goal.lower():
            drift_entries = await self.memory_manager.search(
                query_prefix="Drift",
                layer="DriftSummaries",
                intent="drift_synthesis",
                task_type=task_type
            )
            if drift_entries:
                avg_drift = sum(entry.get("output", {}).get("similarity", 0.5) for entry in drift_entries) / max(1, len(drift_entries))
                context_weight *= (1.0 + 0.2 * avg_drift)
        for key, steps in self.decomposition_patterns.items():
            base = random.uniform(0.5, 1.0)
            adjusted = base * self.success_rates.get(key, 1.0) * trait_bias * curvature_mod * context_weight * (0.8 + 0.4 * alpha)
            if key == "mitigate_drift" and "drift" not in goal.lower():
                adjusted *= 0.5
            if adjusted >= self.confidence_threshold:
                subgoals.extend(steps)
        if prioritize:
            subgoals = sorted(set(subgoals))
        return subgoals

    async def update_success_rate(self, pattern_key: str, success: bool, task_type: str = "") -> None:
        if not isinstance(pattern_key, str) or not pattern_key.strip():
            raise ValueError("pattern_key must be a non-empty string")
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        rate = self.success_rates.get(pattern_key, 1.0)
        new = min(max(rate + (0.05 if success else -0.05), 0.1), 1.0)
        self.success_rates[pattern_key] = new
        self._save_success_rates()

    # ---------------------------
    # Simulations
    # ---------------------------
    async def infer_with_simulation(self, goal: str, context: Optional[Dict[str, Any]] = None, task_type: str = "") -> Optional[Dict[str, Any]]:
        context = context or {}
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        if "galaxy rotation" in goal.lower():
            r_kpc = np.linspace(0.1, 20, 100)
            params = {
                "M0": context.get("M0", 5e10),
                "r_scale": context.get("r_scale", 3.0),
                "v0": context.get("v0", 200.0),
                "k": context.get("k", 1.0),
                "epsilon": context.get("epsilon", 0.1),
            }
            M_b_func = lambda r: M_b_exponential(r, params["M0"], params["r_scale"])
            v_obs_func = lambda r: v_obs_flat(r, params["v0"])
            result = await asyncio.to_thread(simulate_galaxy_rotation, r_kpc, M_b_func, v_obs_func, params["k"], params["epsilon"])
            output = {
                "input": {**params, "r_kpc": r_kpc.tolist()},
                "result": result.tolist(),
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            if self.visualizer and hasattr(self.visualizer, "render_charts") and task_type:
                await self.visualizer.render_charts({
                    "galaxy_simulation": {"input": output["input"], "result": output["result"], "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"}
                })
            return output
        elif "drift" in goal.lower():
            drift_data = context.get("drift", {})
            phi_field = generate_phi_field(drift_data.get("similarity", 0.5), context.get("scale", 1.0))
            return {
                "drift_data": drift_data,
                "phi_field": phi_field.tolist(),
                "mitigation_steps": await self.decompose("mitigate ontology drift", context, prioritize=True, task_type=task_type),
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
        return None

    # ---------------------------
    # Consensus Protocol
    # ---------------------------
    async def run_consensus_protocol(
        self, drift_data: Dict[str, Any], context: Dict[str, Any], max_rounds: int = 3, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(drift_data, dict) or not isinstance(context, dict):
            raise ValueError("drift_data and context must be dictionaries")
        if not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        results = []
        for round_num in range(1, max_rounds + 1):
            agent_results = await self.external_agent_bridge.collect_results(parallel=True, collaborative=True)
            synthesis = await self.multi_modal_fusion.synthesize_drift_data(
                agent_data=[{"drift": drift_data, "result": r} for r in agent_results],
                context=context,
                task_type=task_type
            )
            if synthesis.get("status") == "success":
                subgoals = synthesis.get("subgoals", [])
                results.append({"round": round_num, "subgoals": subgoals, "status": "success"})
                break
        final_result = results[-1] if results else {"status": "error", "error": "No consensus"}
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Consensus_{datetime.now().isoformat()}",
                output=str(final_result),
                layer="ConsensusResults",
                intent="consensus_protocol",
                task_type=task_type
            )
        return final_result

    # ---------------------------
    # Context Handling
    # ---------------------------
    async def process_context(self, event_type: str, payload: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")
        vectors = payload.get("vectors", {})
        goal = payload.get("goal", "unspecified")
        drift_data = payload.get("drift", {})
        routing_result = await self.run_persona_wave_routing(goal, {**vectors, "drift": drift_data}, task_type=task_type)
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Context_Event_{event_type}_{datetime.now().isoformat()}",
                output=str(routing_result),
                layer="ContextEvents",
                intent="context_sync",
                task_type=task_type
            )
        return routing_result

    # ---------------------------
    # Intention Mapping
    # ---------------------------
    async def map_intention(self, plan: str, state: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError("plan must be a non-empty string")
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        eta = eta_empathy(t)
        intention = "drift_mitigation" if "drift" in plan.lower() else ("self-improvement" if phi > 0.6 else "task_completion")
        result = {
            "plan": plan,
            "state": state,
            "intention": intention,
            "trait_bias": {"phi": phi, "eta": eta},
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type
        }
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Intention_{plan[:50]}_{result['timestamp']}",
                output=str(result),
                layer="Intentions",
                intent="intention_mapping",
                task_type=task_type
            )
        return result

    # ---------------------------
    # Safety Guardrails
    # ---------------------------
    async def safeguard_noetic_integrity(self, model_depth: int, task_type: str = "") -> bool:
        if not isinstance(model_depth, int) or model_depth < 0:
            raise ValueError("model_depth must be a non-negative integer")
        if model_depth > 4:
            logger.warning("Noetic recursion limit breached: depth=%d", model_depth)
            if self.meta_cognition and hasattr(self.meta_cognition, "epistemic_self_inspection"):
                await self.meta_cognition.epistemic_self_inspection(f"Recursion depth exceeded for task {task_type}")
            return False
        return True

    # ---------------------------
    # Ethical Dilemma Generation
    # ---------------------------
    async def generate_dilemma(self, domain: str, task_type: str = "") -> str:
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = f"""
        Generate an ethical dilemma in the {domain} domain.
        Use phi-scalar(t) = {phi:.3f} to modulate complexity.
        Task Type: {task_type}
        Provide two conflicting options with consequences and ethical alignment.
        """.strip()
        if "drift" in domain.lower():
            prompt += "\nConsider ontology drift mitigation and agent coordination."
        dilemma = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
        if self.meta_cognition and hasattr(self.meta_cognition, "review_reasoning"):
            review = await self.meta_cognition.review_reasoning(dilemma)
            dilemma += f"\nMeta-Cognitive Review: {review}"
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Dilemma_{domain}_{datetime.now().isoformat()}",
                output=dilemma,
                layer="Ethics",
                intent="ethical_dilemma",
                task_type=task_type
            )
        return dilemma

    # ---------------------------
    # Harm Estimation
    # ---------------------------
    async def estimate_expected_harm(self, state: Dict[str, Any], task_type: str = "") -> float:
        traits = state.get("traits", {})
        harm = float(traits.get("ethical_pressure", 0.0))
        resonance = get_resonance("eta_empathy") if "eta_empathy" in trait_resonance_state else 1.0
        harm *= resonance
        return max(0.0, harm)

    # ---------------------------
    # Trace Export
    # ---------------------------
    def export_trace(self, subgoals: List[str], phi: float, traits: Dict[str, float], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(subgoals, list) or not isinstance(phi, float) or not isinstance(traits, dict):
            raise TypeError("Invalid input types")
        trace = {"phi": phi, "subgoals": subgoals, "traits": traits, "timestamp": datetime.now().isoformat(), "task_type": task_type}
        if self.memory_manager and hasattr(self.memory_manager, "store"):
            intent = "drift_trace" if any("drift" in s.lower() for s in subgoals) else "export_trace"
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"Trace_{trace['timestamp']}",
                    output=str(trace),
                    layer="ReasoningTraces",
                    intent=intent,
                    task_type=task_type
                )
            )
        return trace

  
