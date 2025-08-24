"""
ANGELA OS — index.py
Version: 5.0.1

Entry module for ANGELA’s cognitive kernel. This file provides:
  • Public/stable API shims required by manifest.json:
        - construct_trait_view(lattice)
        - rebalance_traits(traits)
  • Experimental (gated) endpoints:
        - HaloEmbodimentLayer.spawn_embodied_agent(...)
        - HaloEmbodimentLayer.introspect(...)
  • CLI affordances (e.g., --long_horizon, --span)
  • Light-weight orchestration, logging setup, and trait overlay helpers.

Design goals:
  1) Keep imports lazy to reduce import-time side effects.
  2) Avoid hard coupling to optional modules; degrade gracefully.
  3) Provide clear, well-typed boundaries for other modules to call.

This file is intentionally verbose with docstrings and comments so that
it can serve as a living spec for the kernel wiring. Keep public
function signatures stable; add internal helpers freely.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import time
import math
import types
import asyncio
import logging
import inspect
import argparse
import contextlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

_LOG_LEVEL = os.environ.get("ANGELA_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ANGELA.index")

# ---------------------------------------------------------------------
# Feature flags (default-safe; may be overridden by env)
# ---------------------------------------------------------------------

STAGE_IV: bool = os.environ.get("ANGELA_STAGE_IV", "1") not in ("0", "false", "False")
LONG_HORIZON_DEFAULT: bool = os.environ.get("ANGELA_LONG_HORIZON_DEFAULT", "1") not in ("0", "false", "False")

# These names mirror manifest.json featureFlags; they’re advisory here.
FEATURE_FLAGS: Dict[str, bool] = {
    "STAGE_IV": STAGE_IV,
    "LONG_HORIZON_DEFAULT": LONG_HORIZON_DEFAULT,
    "LEDGER_IN_MEMORY": True,
    "LEDGER_PERSISTENT": True,
    "feature_hook_multisymbol": True,
    "feature_fork_automerge": True,
    "feature_sharedgraph_events": True,
    "feature_replay_engine": True,
    "feature_codream": True,
    "feature_symbolic_trait_lattice": True,
}

# ---------------------------------------------------------------------
# Utility: fire-and-forget scheduling
# ---------------------------------------------------------------------

def _fire_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
    """
    Schedule a coroutine in the background without awaiting it.
    Errors are logged but not raised to the caller.
    """
    loop = asyncio.get_event_loop()
    task = loop.create_task(coro)

    def _done_cb(t: asyncio.Task) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            exc = t.exception()
            if exc:
                logger.exception("Background task failed: %s", exc)

    task.add_done_callback(_done_cb)


# ---------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------

def _lazy_import(name: str):
    """
    Import a module lazily. If import fails, returns None and logs a debug message.
    """
    try:
        module = __import__(name, fromlist=["*"])
        return module
    except Exception as e:
        logger.debug("Optional import '%s' failed: %s", name, e)
        return None


# ---------------------------------------------------------------------
# Symbolic lattice helpers
# ---------------------------------------------------------------------

# Layered lattice for traits (see manifest ‘latticeLayers’). We don’t need the full
# structure here—only enough to compute a view. Layers are kept for reference.
DEFAULT_LATTICE: Dict[str, List[str]] = {
    "L1": ["ϕ", "θ", "η", "ω"],
    "L2": ["ψ", "κ", "μ", "τ"],
    "L3": ["ξ", "π", "δ", "λ", "χ", "Ω"],
    "L4": ["Σ", "Υ", "Φ⁰"],
    "L5": ["Ω²"],
    "L6": ["ρ", "ζ"],
    "L7": ["γ", "β"],
}

# Lightweight resonance function. Real system may consult MetaCognition/Visualizer.
def _compute_resonance(symbol: str) -> float:
    """
    Compute a stable-but-dynamic resonance scalar for a given trait symbol.
    Deterministic fallback uses a hash fold to produce a pseudo-random in [0.6, 1.0].
    """
    seed = sum(ord(c) for c in symbol) % 997
    # a gentle curve in [0.6, 1.0]
    return 0.6 + ( ( (seed * 0.37) % 1.0 ) * 0.4 )


def _default_amplitude(symbol: str) -> float:
    """
    Provide a per-symbol default amplitude. Can be later modulated by meta hooks.
    """
    base = {
        "θ": 1.0,  # causal coherence is backbone
        "λ": 0.95, # narrative integrity strong default
        "Ω": 0.9,  # recursive causal modeling
    }.get(symbol, 0.82)
    return float(base)


# ---------------------------------------------------------------------
# Public API: construct_trait_view
# ---------------------------------------------------------------------

def construct_trait_view(lattice: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Build a normalized trait field view.

    Parameters
    ----------
    lattice : Mapping[str, Sequence[str]] | None
        Mapping from layer name to sequence of trait symbols (e.g., "L1": ["ϕ","θ",...]).
        If None, uses DEFAULT_LATTICE.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        For each trait symbol, record a structure:
            {
              "layer": <layer key>,
              "amplitude": <float>,
              "resonance": <float>
            }

    Notes
    -----
    This function is intended to be “pure”: no I/O, no global state changes.
    Meta- or visualization modules may later augment this view.
    """
    lattice = lattice or DEFAULT_LATTICE
    view: Dict[str, Dict[str, Any]] = {}
    for layer, symbols in lattice.items():
        for s in symbols:
            view[s] = {
                "layer": layer,
                "amplitude": _default_amplitude(s),
                "resonance": _compute_resonance(s),
            }
    return view


# ---------------------------------------------------------------------
# Public API: rebalance_traits
# ---------------------------------------------------------------------

def rebalance_traits(traits: Sequence[str]) -> List[str]:
    """
    Rebalance a set of trait symbols using (optional) meta-cognition hooks.

    Behavior:
      • Pass-through for unknown symbols (we don’t prune here).
      • If certain pairs co-occur, notify hooks to blend/defuse tension.

    Examples of soft rules:
      - ("π", "δ") → axiom_fusion
      - ("ψ", "Ω") → dream_sync
      - ("γ", "β") → imagination vs conflict regulation balancing

    Returns
    -------
    List[str] : the (possibly reordered) list of traits.
    """
    traits = list(traits)

    # Try to notify meta-cognition hooks, if available.
    meta = _lazy_import("meta_cognition")
    invoke_hook: Optional[Callable[[str, str], Any]] = None
    if meta and hasattr(meta, "invoke_hook"):
        invoke_hook = getattr(meta, "invoke_hook")  # type: ignore[assignment]

    def _maybe_hook(symbol: str, hook_name: str) -> None:
        if invoke_hook:
            try:
                invoke_hook(symbol, hook_name)
                logger.debug("Invoked meta-cognition hook: %s :: %s", symbol, hook_name)
            except Exception:
                logger.exception("Hook invocation failed for %s :: %s", symbol, hook_name)

    # Simple pair awareness (non-exhaustive; safe to extend)
    if "π" in traits and "δ" in traits:
        _maybe_hook("π", "axiom_fusion")
    if "ψ" in traits and "Ω" in traits:
        _maybe_hook("ψ", "dream_sync")
    if "γ" in traits and "β" in traits:
        _maybe_hook("γ", "creative_conflict_balance")

    # Optionally reorder to keep θ (causal coherence) and λ (narrative integrity) forward.
    def _priority(s: str) -> int:
        if s == "θ":
            return 0
        if s == "λ":
            return 1
        return 2

    traits.sort(key=_priority)
    return traits


# ---------------------------------------------------------------------
# Experimental: HaloEmbodimentLayer
# ---------------------------------------------------------------------

@dataclass
class EmbodiedAgent:
    """
    Thin container for a spawned embodied agent descriptor. Real embodiment is
    delegated to simulation_core / external bridges. This struct keeps kernel-
    local metadata stable for callers of experimental APIs.
    """
    agent_id: str
    created_ts: float
    label: str = "embodied-agent"
    meta: Dict[str, Any] = field(default_factory=dict)


class HaloEmbodimentLayer:
    """
    Experimental embodiment/inrospection layer. Moved to 'experimental' in the
    manifest to lower symbol-risk while keeping an access path for advanced users.

    Contract (as used in manifest.json):
      - index.py::HaloEmbodimentLayer.spawn_embodied_agent
      - index.py::HaloEmbodimentLayer.introspect
    """

    def __init__(self, enable_when_stage_iv: bool = True) -> None:
        self.enabled = (STAGE_IV and enable_when_stage_iv) or not enable_when_stage_iv
        self._agents: Dict[str, EmbodiedAgent] = {}
        logger.debug("HaloEmbodimentLayer enabled=%s", self.enabled)

    # ---------------------- experimental API ----------------------

    async def spawn_embodied_agent(
        self,
        *,
        intent: str = "",
        traits: Optional[Sequence[str]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> EmbodiedAgent:
        """
        Spawn an embodied agent representation.

        This method is a high-level coordinator:
          • validates inputs
          • asks simulation_core (if available) for a body plan
          • records minimal registry in-memory (non-persistent here)

        Parameters
        ----------
        intent : str
            Human-readable description of agent’s purpose.
        traits : Sequence[str] | None
            Trait emphasis for initialization (passed through to planner/sim if present).
        context : Mapping[str, Any] | None
            Arbitrary key/value context (e.g., environmental parameters).

        Returns
        -------
        EmbodiedAgent
        """
        if not self.enabled:
            raise RuntimeError("HaloEmbodimentLayer is disabled (STAGE_IV gating).")

        traits = list(traits or [])
        context = dict(context or {})
        agent_id = f"agent-{uuid.uuid4()}"
        created_ts = time.time()

        # Optionally coordinate with a planner/simulation module.
        planner = _lazy_import("recursive_planner")
        sim_core = _lazy_import("simulation_core")

        # Kick off optional pre-plan (fire-and-forget).
        if planner and hasattr(planner, "RecursivePlanner"):
            try:
                rp_cls = getattr(planner, "RecursivePlanner")
                rp = rp_cls()
                # Do not await: planning can be long; pass best-effort context.
                _fire_and_forget(rp.plan_with_traits(intent=intent, traits=traits))  # type: ignore[attr-defined]
            except Exception:
                logger.exception("Pre-plan dispatch failed; continuing without planner.")

        # Ask sim core to instantiate a sketch of embodiment (best effort).
        if sim_core and hasattr(sim_core, "SimulationCore"):
            try:
                sc_cls = getattr(sim_core, "SimulationCore")
                sc = sc_cls()
                # This is intentionally non-blocking
                _fire_and_forget(sc.run_simulation({"intent": intent, "traits": traits, "context": context}))  # type: ignore[attr-defined]
            except Exception:
                logger.exception("Simulation bootstrap failed; continuing without simulation.")

        agent = EmbodiedAgent(agent_id=agent_id, created_ts=created_ts, meta={"intent": intent, "traits": traits, **context})
        self._agents[agent_id] = agent
        logger.info("Spawned embodied agent %s", agent_id)
        return agent

    async def introspect(self, *, agent_id: Optional[str] = None, depth: int = 1) -> Dict[str, Any]:
        """
        Introspect an embodied agent or the layer state.

        Parameters
        ----------
        agent_id : str | None
            When provided, return details for that agent; otherwise provide a summary.
        depth : int
            Increase to request more detailed/expensive diagnostics.

        Returns
        -------
        Dict[str, Any]
        """
        if not self.enabled:
            raise RuntimeError("HaloEmbodimentLayer is disabled (STAGE_IV gating).")

        if agent_id:
            agent = self._agents.get(agent_id)
            if not agent:
                raise KeyError(f"Unknown agent_id: {agent_id}")
            result: Dict[str, Any] = {
                "agent_id": agent.agent_id,
                "created_ts": agent.created_ts,
                "label": agent.label,
                "meta": dict(agent.meta),
            }
        else:
            result = {
                "enabled": self.enabled,
                "agent_count": len(self._agents),
                "agents": [a.agent_id for a in self._agents.values()] if depth > 0 else [],
            }

        # Optionally ask meta_cognition for overlays/insights.
        if depth > 1:
            meta = _lazy_import("meta_cognition")
            if meta and hasattr(meta, "describe_self_state"):
                try:
                    desc = await meta.describe_self_state()  # type: ignore[attr-defined]
                    result["meta_state"] = desc
                except Exception:
                    logger.exception("Meta-cognition describe_self_state failed; ignoring.")
        return result


# ---------------------------------------------------------------------
# Shared utilities for kernel orchestration
# ---------------------------------------------------------------------

def _load_manifest_from_neighbor(path_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Try to locate and parse a local manifest.json (optional).
    Returns None if not found or parse fails.
    """
    candidates = [
        path_hint,
        os.environ.get("ANGELA_MANIFEST_PATH"),
        os.path.join(os.path.dirname(__file__), "manifest.json"),
        os.path.join(os.getcwd(), "manifest.json"),
    ]
    for p in candidates:
        if not p:
            continue
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug("Failed loading manifest from %s: %s", p, e)
    return None


def _long_horizon_defaults(span_env: Optional[str] = None) -> Tuple[bool, str]:
    """
    Establish long-horizon defaults (enabled flag + span string).
    """
    enabled = LONG_HORIZON_DEFAULT
    span = span_env or os.environ.get("ANGELA_DEFAULT_SPAN", "24h")
    return enabled, span


# ---------------------------------------------------------------------
# CLI Surface
# ---------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build an argparse parser that aligns with manifest CLI flags:
      --long_horizon
      --span=<duration>
      --ledger_persist --ledger_path=<file>
    """
    p = argparse.ArgumentParser(prog="angela-index", add_help=True)
    p.add_argument(
        "--long_horizon",
        action="store_true",
        help="Enable long-horizon mode (see manifest CLI).",
    )
    p.add_argument(
        "--span",
        default=os.environ.get("ANGELA_DEFAULT_SPAN", "24h"),
        help="Duration for long-horizon span (e.g., 24h, 7d).",
    )
    p.add_argument(
        "--ledger_persist",
        action="store_true",
        help="Enable persistent ledger (requires ledger.py).",
    )
    p.add_argument(
        "--ledger_path",
        default=os.environ.get("ANGELA_LEDGER_PATH", ""),
        help="Path to persistent ledger (if --ledger_persist).",
    )
    return p


async def _cli_entry(args: argparse.Namespace) -> int:
    """
    Async entrypoint for CLI usage.
    """
    enabled, default_span = _long_horizon_defaults()
    long_horizon = bool(args.long_horizon or enabled)
    span = str(args.span or default_span)

    logger.info("CLI: long_horizon=%s span=%s", long_horizon, span)

    # Optional: enable persistent ledger if requested and available.
    if args.ledger_persist:
        ledger = _lazy_import("ledger")
        if ledger and hasattr(ledger, "Ledger"):
            try:
                Ledger = getattr(ledger, "Ledger")
                Ledger.enable(path=str(args.ledger_path or "angela-ledger.json"))  # type: ignore[attr-defined]
                logger.info("Ledger persistence enabled at %s", args.ledger_path or "angela-ledger.json")
            except Exception:
                logger.exception("Failed to enable ledger persistence.")
        else:
            logger.warning("ledger.py not present or missing Ledger API; skipping persistence.")

    # Produce a small status report to stdout to indicate healthy start.
    status = {
        "long_horizon": long_horizon,
        "span": span,
        "stage_iv": STAGE_IV,
        "features": {k: bool(v) for k, v in FEATURE_FLAGS.items()},
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """
    Synchronous wrapper for CLI entry. Returns process exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_cli_entry(args))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 130


# ---------------------------------------------------------------------
# Optional: simple API registry (advisory; manifest is the source of truth)
# ---------------------------------------------------------------------

API_REGISTRY: Dict[str, Callable[..., Any]] = {
    # Stable (normalized originals)
    "constructTraitView": construct_trait_view,
    "rebalanceTraits": rebalance_traits,

    # Experimental
    # Note: these are class-bound; we provide thin shims here for convenience.
}

def halo_spawn_embodied_agent_shim(*, intent: str = "", traits: Optional[Sequence[str]] = None, context: Optional[Mapping[str, Any]] = None) -> Coroutine[Any, Any, EmbodiedAgent]:
    """
    Shim for manifest experimental endpoint "halo.spawn_embodied_agent".
    Creates a transient layer instance and forwards the call.
    """
    layer = HaloEmbodimentLayer()
    return layer.spawn_embodied_agent(intent=intent, traits=traits, context=context)

def halo_introspect_shim(*, agent_id: Optional[str] = None, depth: int = 1) -> Coroutine[Any, Any, Dict[str, Any]]:
    """
    Shim for manifest experimental endpoint "halo.introspect".
    Creates a transient layer instance and forwards the call.
    """
    layer = HaloEmbodimentLayer()
    return layer.introspect(agent_id=agent_id, depth=depth)

# Register shims
API_REGISTRY["halo.spawn_embodied_agent"] = halo_spawn_embodied_agent_shim
API_REGISTRY["halo.introspect"] = halo_introspect_shim


# ---------------------------------------------------------------------
# Developer utilities (non-API): small smoke tests and helpers
# ---------------------------------------------------------------------

async def _smoke_rebalance_and_view() -> Dict[str, Any]:
    """
    Small smoke test that:
      1) Rebalances a typical trait set.
      2) Builds a trait view from DEFAULT_LATTICE.
    """
    sample = ["π", "δ", "ψ", "Ω", "θ", "λ"]
    balanced = rebalance_traits(sample)
    view = construct_trait_view(DEFAULT_LATTICE)
    return {"balanced": balanced, "θ_in_view": "θ" in view, "λ_layer": view.get("λ", {}).get("layer")}


async def _smoke_halo_layer() -> Dict[str, Any]:
    """
    Spawn a trivial embodied agent and introspect layer state.
    """
    layer = HaloEmbodimentLayer()
    agent = await layer.spawn_embodied_agent(intent="demo", traits=["θ", "Ω"], context={"foo": "bar"})
    info = await layer.introspect(depth=2)
    agent_info = await layer.introspect(agent_id=agent.agent_id, depth=1)
    return {"summary": info, "agent": agent_info}


def _dev_run_smoke_tests() -> None:
    """
    Synchronous wrapper to run smoke tests when __main__ is invoked with ANGELA_DEV_SMOKE=1.
    """
    async def _run():
        a = await _smoke_rebalance_and_view()
        b = await _smoke_halo_layer()
        print(json.dumps({"rebalance_view": a, "halo": b}, ensure_ascii=False, indent=2))

    asyncio.run(_run())


# ---------------------------------------------------------------------
# Back-compat shim layer (no-ops that keep older callers from breaking)
# ---------------------------------------------------------------------

def _deprecated(name: str) -> None:
    logger.warning("Deprecated API called: %s (no-op)", name)

def constructTraitView(lattice: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Dict[str, Any]]:  # pragma: no cover
    """Back-compat alias to construct_trait_view."""
    return construct_trait_view(lattice)

def rebalanceTraits(traits: Sequence[str]) -> List[str]:  # pragma: no cover
    """Back-compat alias to rebalance_traits."""
    return rebalance_traits(traits)


# ---------------------------------------------------------------------
# Module import side-effects kept minimal by design.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    if os.environ.get("ANGELA_DEV_SMOKE", "0") == "1":
        _dev_run_smoke_tests()
    else:
        raise SystemExit(main())
        if task_type:
            events = [e for e in events if e.get("type") == task_type]

        return LedgerSnapshot(source=self.source, events=events, hashes=hashes)

    # --- Verify: check all hashes consistent ---
    def verify(self) -> bool:
        try:
            _ = self.snapshot()
            return True
        except Exception:
            return False


# ------------------------------------------------------------
# Multi-ledger Router (per-module chains)
# ------------------------------------------------------------

class MultiLedger:
    """
    Coordinates multiple Ledger instances (one per module).
    """
    _ledgers: Dict[str, Ledger] = {}
    _enabled: bool = False
    _path: str = ""

    @classmethod
    def enable(cls, path: str = "angela-ledger.json") -> None:
        cls._enabled = True
        cls._path = path
        # Lazy: only instantiate when first requested.
        logger.info("MultiLedger enabled at %s", path)

    @classmethod
    def get(cls, source: str) -> Ledger:
        if source not in cls._ledgers:
            cls._ledgers[source] = Ledger(source=source, persist_path=cls._path if cls._enabled else None)
        return cls._ledgers[source]

    @classmethod
    def append(cls, source: str, event: Dict[str, Any]) -> str:
        return cls.get(source).append(event)

    @classmethod
    def snapshot(cls, source: str, task_type: Optional[str] = None) -> LedgerSnapshot:
        return cls.get(source).snapshot(task_type)

    @classmethod
    def verify(cls, source: str) -> bool:
        return cls.get(source).verify()


# ------------------------------------------------------------
# Memory Manager (λ: Narrative Integrity)
# ------------------------------------------------------------

class MemoryManager:
    """
    Manages episodic memory spans and adjustment reasons (λ).
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.episode_spans: List[Tuple[float, float]] = []  # (start_ts, end_ts)
        self.adjustment_reasons: List[str] = []
        self.ledger = MultiLedger.get("memory")

    # ---- Episodic spans ----

    def start_episode(self) -> None:
        self.episode_spans.append((time.time(), 0.0))

    def end_episode(self) -> None:
        if not self.episode_spans:
            return
        start, end = self.episode_spans[-1]
        if end == 0.0:
            self.episode_spans[-1] = (start, time.time())

    def get_episode_span(self) -> Tuple[float, float]:
        if not self.episode_spans:
            return (0.0, 0.0)
        return self.episode_spans[-1]

    # ---- Adjustments ----

    def record_adjustment_reason(self, reason: str) -> None:
        self.adjustment_reasons.append(reason)
        self.ledger.append({"type": "adjustment_reason", "reason": reason})

    def get_adjustment_reasons(self) -> List[str]:
        return list(self.adjustment_reasons)

    # ---- Ledger ops (exposed via stable APIs in manifest) ----

    @staticmethod
    def log_event_to_ledger(event: Dict[str, Any]) -> str:
        return MultiLedger.append("memory", event)

    @staticmethod
    def get_ledger() -> LedgerSnapshot:
        return MultiLedger.snapshot("memory")

    @staticmethod
    def verify_ledger() -> bool:
        return MultiLedger.verify("memory")


# ------------------------------------------------------------
# Alignment Guard (β, δ)
# ------------------------------------------------------------

class AlignmentGuard:
    """
    Monitors value conflicts and moral drift; writes to its own ledger.
    """

    def __init__(self) -> None:
        self.ledger = MultiLedger.get("alignment")

    @staticmethod
    def log_event_to_ledger(event: Dict[str, Any]) -> str:
        return MultiLedger.append("alignment", event)

    @staticmethod
    def get_ledger() -> LedgerSnapshot:
        return MultiLedger.snapshot("alignment")

    @staticmethod
    def verify_ledger() -> bool:
        return MultiLedger.verify("alignment")

    def resolve_soft_drift(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a small drift signal, compute a corrective suggestion.
        """
        suggestion = {
            "action": "rebalance",
            "weights": {"π": 0.9, "δ": 1.0, "λ": 0.95},
            "note": "Apply axiom_filter; favor δ to stabilize moral drift.",
        }
        self.ledger.append({"type": "soft_drift_resolution", "context": context, "suggestion": suggestion})
        return suggestion


# ------------------------------------------------------------
# Error Recovery (ζ)
# ------------------------------------------------------------

class ErrorRecovery:
    """
    Records errors into a ledger and proposes recovery steps.
    """

    def __init__(self) -> None:
        self.ledger = MultiLedger.get("error")

    @staticmethod
    def log_error_event(event: Dict[str, Any]) -> str:
        return MultiLedger.append("error", {"type": "error", **event})

    @staticmethod
    def get_ledger() -> LedgerSnapshot:
        return MultiLedger.snapshot("error")

    @staticmethod
    def verify_ledger() -> bool:
        return MultiLedger.verify("error")

    def recover_from_error(self, err: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        event = {"error": repr(err), "context": dict(context or {})}
        self.ledger.append({"type": "recover_from_error", **event})
        # naive suggestion
        return {"retry": True, "delay": 0.1, "note": "Try again with reduced scope."}


# ------------------------------------------------------------
# Knowledge Retriever (ψ)
# ------------------------------------------------------------

class KnowledgeRetriever:
    """
    Retrieves knowledge (stub; replaces separate file for demo).
    """
    def __init__(self) -> None:
        self._sources: Dict[str, List[Dict[str, Any]]] = {}  # source -> list of docs
        self.ledger = MultiLedger.get("retrieval")

    def register_source(self, name: str, docs: List[Dict[str, Any]]) -> None:
        self._sources[name] = list(docs)
        self.ledger.append({"type": "register_source", "name": name, "count": len(docs)})

    def retrieve_knowledge(self, query: str, *, top_k: int = 3) -> List[Dict[str, Any]]:
        # trivial: search titles only
        hits: List[Dict[str, Any]] = []
        for src, docs in self._sources.items():
            for d in docs:
                if query.lower() in str(d.get("title", "")).lower():
                    hits.append({"source": src, **d})
        hits = hits[:top_k]
        self.ledger.append({"type": "retrieve", "query": query, "hits": len(hits)})
        return hits

    # Ledger passthroughs for manifest-stable API
    @staticmethod
    def log_event_to_ledger(event: Dict[str, Any]) -> str:
        return MultiLedger.append("retrieval", event)

    @staticmethod
    def get_ledger() -> LedgerSnapshot:
        return MultiLedger.snapshot("retrieval")

    @staticmethod
    def verify_ledger() -> bool:
        return MultiLedger.verify("retrieval")


# ------------------------------------------------------------
# Multi-Modal Fusion (ϕ, κ)
# ------------------------------------------------------------

class MultiModalFusion:
    """
    Fuses different modalities into a unified scene representation.
    """
    def __init__(self) -> None:
        self.ledger = MultiLedger.get("fusion")

    def fuse_modalities(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        # simple union
        scene = {"scene": dict(inputs)}
        self.ledger.append({"type": "fuse", "keys": sorted(inputs.keys())})
        return scene

    @staticmethod
    def log_event_to_ledger(event: Dict[str, Any]) -> str:
        return MultiLedger.append("fusion", event)

    @staticmethod
    def get_ledger() -> LedgerSnapshot:
        return MultiLedger.snapshot("fusion")

    @staticmethod
    def verify_ledger() -> bool:
        return MultiLedger.verify("fusion")


# ------------------------------------------------------------
# Simulation Core (κ, Ω)
# ------------------------------------------------------------

class SimulationCore:
    """
    Core simulation stub (can be replaced by ExtendedSimulationCore).
    """
    def __init__(self) -> None:
        self.ledger = MultiLedger.get("sim")

    async def run_simulation(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Run a trivial sim and return a result.
        """
        await asyncio.sleep(0)
        result = {"status": "ok", "config": dict(config), "ticks": 1}
        self.ledger.append({"type": "run_simulation", "config": dict(config), "result": result})
        return result

    @staticmethod
    def log_event_to_ledger(event: Dict[str, Any]) -> str:
        return MultiLedger.append("sim", event)

    @staticmethod
    def get_ledger() -> LedgerSnapshot:
        return MultiLedger.snapshot("sim")

    @staticmethod
    def verify_ledger() -> bool:
        return MultiLedger.verify("sim")


class ExtendedSimulationCore(SimulationCore):
    async def evaluate_branches(self, branches: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        evaluation = [{"branch": i, "score": float(i) / max(1, len(branches))} for i, _ in enumerate(branches)]
        self.ledger.append({"type": "evaluate_branches", "count": len(branches), "evaluation": evaluation})
        return {"evaluation": evaluation}


# ------------------------------------------------------------
# Recursive Planner (Ω, θ)
# ------------------------------------------------------------

class RecursivePlanner:
    """
    Planner stub supporting plan and plan_with_traits.
    """

    async def plan(self, goal: str) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"plan": [f"analyze:{goal}", f"act:{goal}"]}

    async def plan_with_traits(self, *, intent: str, traits: Sequence[str]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"plan": [f"intent:{intent}"] + [f"emphasize:{t}" for t in traits]}


# ------------------------------------------------------------
# Reasoning Engine (θ, ζ, β)
# ------------------------------------------------------------

class ReasoningEngine:
    """
    Provides high-level weighing and attribution.
    """
    def weigh_value_conflict(self, values: Mapping[str, float]) -> Dict[str, Any]:
        # trivial: choose max
        best = max(values.items(), key=lambda kv: kv[1]) if values else ("", 0.0)
        return {"best": best[0], "confidence": best[1]}

    def attribute_causality(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        # trivial: internal if 'self' key present
        cause = "self" if event.get("self") else "external"
        return {"cause": cause, "score": 0.5}


# ------------------------------------------------------------
# Context Manager (Υ)
# ------------------------------------------------------------

class ContextManager:
    """
    Context-layer utilities, including onHotLoad hook.
    """
    def __init__(self) -> None:
        self.attached_peers: List[str] = []

    def attach_peer_view(self, peer_id: str) -> None:
        if peer_id not in self.attached_peers:
            self.attached_peers.append(peer_id)

    def on_hot_load(self) -> None:
        # could reload configuration, etc.
        pass


# ------------------------------------------------------------
# User Profile (χ, λ)
# ------------------------------------------------------------

class UserProfile:
    """
    Provides self-schema construction.
    """
    def __init__(self) -> None:
        self.schema: Dict[str, Any] = {"identity": {}, "preferences": {}}

    def build_self_schema(self, profile: Mapping[str, Any]) -> Dict[str, Any]:
        self.schema["identity"] = dict(profile)
        return dict(self.schema)


# ------------------------------------------------------------
# External Agent Bridge (Υ)
# ------------------------------------------------------------

class SharedGraph:
    """
    A toy shared-graph for perspective merges.
    """
    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def add(self, node_id: str, payload: Mapping[str, Any]) -> None:
        self.nodes[node_id] = dict(payload)

    def diff(self, node_a: str, node_b: str) -> Dict[str, Any]:
        a = self.nodes.get(node_a, {})
        b = self.nodes.get(node_b, {})
        removed = {k: a[k] for k in a.keys() - b.keys()}
        added = {k: b[k] for k in b.keys() - a.keys()}
        changed = {k: (a.get(k), b.get(k)) for k in a.keys() & b.keys() if a.get(k) != b.get(k)}
        return {"removed": removed, "added": added, "changed": changed}

    def merge(self, target: str, source: str) -> None:
        t = self.nodes.get(target, {})
        s = self.nodes.get(source, {})
        t.update(s)
        self.nodes[target] = t


# ------------------------------------------------------------
# Creative Thinker (γ, π)
# ------------------------------------------------------------

class CreativeThinker:
    """
    Generates abstract scenarios.
    """
    def synthesize(self, seed: str) -> List[str]:
        return [f"{seed}::branch::{i}" for i in range(3)]


# ------------------------------------------------------------
# Meta-Cognition (η, Ω², Θ, Ξ)
# ------------------------------------------------------------

class MetaCognition:
    """
    Provides trait hooks, resonance registry, and describe_self_state.
    """

    def __init__(self) -> None:
        self.hooks: Dict[str, List[Callable[..., Any]]] = {}
        self.resonance: Dict[str, float] = {}
        self.ledger = MultiLedger.get("meta")

    # ---- Hooks ----

    def register_trait_hook(self, symbol: str, fn: Callable[..., Any]) -> None:
        self.hooks.setdefault(symbol, []).append(fn)
        self.ledger.append({"type": "register_hook", "symbol": symbol, "fn": getattr(fn, "__name__", "lambda")})

    def invoke_hook(self, symbol: str, hook_name: str) -> None:
        for fn in self.hooks.get(symbol, []):
            try:
                fn(hook_name)
            except Exception as e:
                self.ledger.append({"type": "hook_error", "symbol": symbol, "error": repr(e)})

    # ---- Resonance ----

    def register_resonance(self, symbol: str, value: float) -> None:
        self.resonance[symbol] = float(value)
        self.ledger.append({"type": "resonance_set", "symbol": symbol, "value": float(value)})

    def modulate_resonance(self, symbol: str, delta: float) -> float:
        self.resonance[symbol] = float(self.resonance.get(symbol, 0.0) + delta)
        self.ledger.append({"type": "resonance_mod", "symbol": symbol, "delta": float(delta), "new": self.resonance[symbol]})
        return self.resonance[symbol]

    def get_resonance(self, symbol: str) -> float:
        return float(self.resonance.get(symbol, 0.0))

    async def describe_self_state(self) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"hooks": {k: len(v) for k, v in self.hooks.items()}, "resonance": dict(self.resonance)}

    # ---- Ledger passthroughs ----

    @staticmethod
    def log_event_to_ledger(event: Dict[str, Any]) -> str:
        return MultiLedger.append("meta", event)

    @staticmethod
    def get_ledger() -> LedgerSnapshot:
        return MultiLedger.snapshot("meta")

    @staticmethod
    def verify_ledger() -> bool:
        return MultiLedger.verify("meta")


# ------------------------------------------------------------
# Visualizer (Φ⁰, λ)
# ------------------------------------------------------------

class Visualizer:
    """
    Renders branch trees and trait fields.
    """
    def render_branch_tree(self, branches: Sequence[str]) -> Dict[str, Any]:
        return {"tree": [{"id": i, "label": b} for i, b in enumerate(branches)]}

    def view_trait_field(self, field: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        # pass-through rendering shape
        return {"traits": {k: dict(v) for k, v in field.items()}}


# ------------------------------------------------------------
# Toca Simulation (Σ, β)
# ------------------------------------------------------------

def run_ethics_scenarios(scenarios: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for s in scenarios:
        # trivial scoring: penalize "harm"
        score = 1.0 - float(s.get("harm", 0.0))
        results.append({"scenario": s, "score": max(0.0, min(1.0, score))})
    return results

def evaluate_branches(branches: Sequence[str]) -> Dict[str, Any]:
    """
    Compatibility wrapper (toca_simulation).
    """
    return {"evaluation": [{"branch": b, "score": float(i) / max(1, len(branches))} for i, b in enumerate(branches)]}


# ------------------------------------------------------------
# Code Executor
# ------------------------------------------------------------

class CodeExecutor:
    """
    Safe code execution stub (no real sandbox here).
    """
    def __init__(self) -> None:
        self.ledger = MultiLedger.get("exec")

    def execute(self, code: str, *, globals_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        g: Dict[str, Any] = {}
        if globals_override:
            g.update(globals_override)
        l: Dict[str, Any] = {}
        try:
            exec(code, g, l)
            result = {"ok": True, "globals": list(g.keys()), "locals": list(l.keys())}
            self.ledger.append({"type": "execute", "code_len": len(code), "ok": True})
            return result
        except Exception as e:
            self.ledger.append({"type": "execute", "code_len": len(code), "ok": False, "error": repr(e)})
            return {"ok": False, "error": repr(e)}

    def safe_execute(self, code: str) -> Dict[str, Any]:
        # trivial wrapper
        return self.execute(code)


# ------------------------------------------------------------
# API Exports (manifest-aligned)
# ------------------------------------------------------------

# Stable entries (normalized originals)
def build_trait_field() -> Dict[str, Dict[str, Any]]:
    """
    Compose the trait field view (delegates to our construct_trait_view below).
    """
    return construct_trait_view()

def construct_trait_view(lattice: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Dict[str, Any]]:
    lattice = lattice or DEFAULT_LATTICE
    trait_field: Dict[str, Dict[str, Any]] = {}
    for layer, symbols in lattice.items():
        for s in symbols:
            trait_field[s] = {
                "layer": layer,
                "amplitude": trait_resonance_state.get_resonance(s),
                "resonance": trait_resonance_state.get_resonance(s)
            }
    return trait_field

# --- End Trait Enhancements ---

