from __future__ import annotations
"""
ANGELA Cognitive Kernel - index_v503.py
--------------------------------------
Version: 5.0.3
Date: 2025-10-21
Purpose: Harmonized kernel for customGPT with:
  - Unified global SHA-256 append-only ledger (file-backed, locked, rotation)
  - Overlay tagging persisted into ledger
  - τ-harmonization used during reflection_check
  - CLI flags: --verify-ledger, --export-ledger, --export-overlays
  - Backwards-compatible pipeline (perceive/analyze/synthesize/execute/reflect)
Notes:
  - Keep payloads small in ledger; large artifacts should be stored externally and referenced by hash.
  - This file focuses on orchestration; modules (reasoning_engine, etc.) are imported as usual.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Optional dependency: filelock for cross-process locking
try:
    from filelock import FileLock, Timeout as FileLockTimeout
    FILELOCK_AVAILABLE = True
except Exception:
    FILELOCK_AVAILABLE = False

# --- Basic imports for kernel (these modules are expected to exist in /mnt/data or PYTHONPATH) ---
# The FlatLayoutFinder earlier maps modules.* imports to files on /mnt/data; keep standard imports
import reasoning_engine
import recursive_planner
import simulation_core
import context_manager as context_manager_module
import meta_cognition as meta_cognition_module
import visualizer as visualizer_module
import alignment_guard as alignment_guard_module
import concept_synthesizer as concept_synthesizer_module
import memory_manager
import multi_modal_fusion
import learning_loop
import knowledge_retriever
import creative_thinker
import external_agent_bridge
import code_executor as code_executor_module
import error_recovery as error_recovery_module
import user_profile

logger = logging.getLogger("ANGELA.Index.v5.0.3")
logging.basicConfig(level=logging.INFO)

# ------------------------------
# Global configuration & defaults
# ------------------------------
LEDGER_DIR = os.getenv("ANGELA_LEDGER_DIR", "/mnt/data/angela_ledger")
LEDGER_ACTIVE_FILENAME = os.getenv("ANGELA_LEDGER_FILE", "ledger_active.jsonl")
LEDGER_MAX_BYTES_BEFORE_ROTATE = int(os.getenv("ANGELA_LEDGER_ROTATE_BYTES", str(10 * 1024 * 1024)))  # 10MB default
LEDGER_LOCK_TIMEOUT = float(os.getenv("ANGELA_LEDGER_LOCK_TIMEOUT", "5.0"))

os.makedirs(LEDGER_DIR, exist_ok=True)
LEDGER_ACTIVE_PATH = os.path.join(LEDGER_DIR, LEDGER_ACTIVE_FILENAME)
LEDGER_LOCK_PATH = LEDGER_ACTIVE_PATH + ".lock"

# In-memory cache (small window) and rotation index
_ledger_buffer: List[Dict[str, Any]] = []
_timechain_log = deque(maxlen=1000)

# ------------------------------
# Utilities: hashing / timestamp
# ------------------------------
def _now_ts() -> float:
    return time.time()

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _payload_hash(payload: Dict[str, Any]) -> str:
    b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

# ------------------------------
# Ledger primitives
# ------------------------------
@contextmanager
def _acquire_ledger_lock(timeout: float = LEDGER_LOCK_TIMEOUT):
    """Context manager returning a lock context (cross-process if filelock available)."""
    if FILELOCK_AVAILABLE:
        lock = FileLock(LEDGER_LOCK_PATH, timeout=timeout)
        try:
            lock.acquire(timeout=timeout)
            yield
        finally:
            try:
                lock.release()
            except Exception:
                pass
    else:
        # Fallback to in-process lock (not cross-process)
        import threading
        if not hasattr(_acquire_ledger_lock, "_thread_lock"):
            _acquire_ledger_lock._thread_lock = threading.Lock()
        lock = _acquire_ledger_lock._thread_lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

def _read_active_ledger_lines() -> List[str]:
    if not os.path.exists(LEDGER_ACTIVE_PATH):
        return []
    try:
        with open(LEDGER_ACTIVE_PATH, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except Exception as e:
        logger.warning("Failed to read ledger file: %s", e)
        return []

def _append_line_to_ledger(line: str) -> None:
    # ensure ledger dir exists
    os.makedirs(os.path.dirname(LEDGER_ACTIVE_PATH), exist_ok=True)
    # Acquire lock for file append
    with _acquire_ledger_lock():
        try:
            with open(LEDGER_ACTIVE_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.error("Failed to append to ledger file: %s", e)
            raise

def _active_ledger_size_bytes() -> int:
    try:
        return os.path.getsize(LEDGER_ACTIVE_PATH) if os.path.exists(LEDGER_ACTIVE_PATH) else 0
    except Exception:
        return 0

def _rotate_ledger_if_needed() -> None:
    size = _active_ledger_size_bytes()
    if size < LEDGER_MAX_BYTES_BEFORE_ROTATE:
        return
    # rotate
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rotated_name = f"ledger_{ts}.jsonl"
    rotated_path = os.path.join(LEDGER_DIR, rotated_name)
    with _acquire_ledger_lock():
        try:
            if os.path.exists(LEDGER_ACTIVE_PATH):
                os.replace(LEDGER_ACTIVE_PATH, rotated_path)
                logger.info("Rotated ledger to %s (size %.2f KB)", rotated_path, size / 1024.0)
        except Exception as e:
            logger.error("Ledger rotation failed: %s", e)

# Ledger event shape:
# {
#   "timestamp": float,
#   "module": "ModuleName",
#   "type": "event_type",
#   "payload_hash": "sha256(...)",
#   "payload": {...},          # optional, keep small
#   "previous_hash": "....",   # previous current_hash or zeros for genesis
#   "current_hash": "...."
# }

def log_event_to_ledger(event: Dict[str, Any], *, persist_payload: bool = True, max_payload_size_bytes: int = 2048) -> Dict[str, Any]:
    """
    Append an event to the global SHA-256 chained ledger (file-backed).
    Returns the ledger entry dict.
    """
    # construct minimal event
    base = {
        "timestamp": _now_ts(),
        "iso": _utc_iso(),
        "module": event.get("module", event.get("source", "Unknown")),
        "type": event.get("type", "generic"),
    }

    # decide payload storage
    payload = event.get("payload")
    if payload is None:
        payload = {k: v for k, v in event.items() if k not in base}
    # compute payload hash always
    payload_hash = _payload_hash(payload if isinstance(payload, dict) else {"v": str(payload)})
    base["payload_hash"] = payload_hash

    # optionally include payload inline if small and persist_payload True
    include_payload = False
    try:
        if persist_payload:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if len(raw) <= max_payload_size_bytes:
                include_payload = True
    except Exception:
        include_payload = False

    if include_payload:
        base["payload"] = payload

    # compute previous hash
    prev_hash = "0" * 64
    lines = _read_active_ledger_lines()
    if lines:
        try:
            last = json.loads(lines[-1])
            prev_hash = last.get("current_hash", prev_hash)
        except Exception:
            prev_hash = "0" * 64

    base["previous_hash"] = prev_hash

    # compute current hash over canonical JSON
    hash_in = json.dumps({
        "timestamp": base["timestamp"],
        "module": base["module"],
        "type": base["type"],
        "payload_hash": base["payload_hash"],
        "previous_hash": base["previous_hash"],
    }, sort_keys=True, ensure_ascii=False).encode("utf-8")
    base["current_hash"] = hashlib.sha256(hash_in).hexdigest()

    # write line
    try:
        _append_line_to_ledger(json.dumps(base, ensure_ascii=False))
        # rotate if needed (non-blocking safety: done synchronously here)
        _rotate_ledger_if_needed()
    except Exception as e:
        logger.error("log_event_to_ledger failed: %s", e)
        raise

    # smart in-memory buffer for small recent window
    _ledger_buffer.append(base)
    if len(_ledger_buffer) > 1024:
        _ledger_buffer.pop(0)
    return base

def get_ledger_slice(module: Optional[str] = None, event_type: Optional[str] = None, since_ts: Optional[float] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    lines = _read_active_ledger_lines()
    out: List[Dict[str, Any]] = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if module and entry.get("module") != module:
            continue
        if event_type and entry.get("type") != event_type:
            continue
        if since_ts and entry.get("timestamp", 0) < since_ts:
            continue
        out.append(entry)
        if len(out) >= limit:
            break
    return list(reversed(out))

def get_full_ledger() -> List[Dict[str, Any]]:
    lines = _read_active_ledger_lines()
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out

def verify_ledger() -> Tuple[bool, Optional[int], Optional[int]]:
    """Verify the active ledger chain integrity.
    Returns (ok, bad_index, total_entries)
    """
    lines = _read_active_ledger_lines()
    prev = "0" * 64
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except Exception:
            return False, i, len(lines)
        # recompute current hash in same canonical way
        recomputed = hashlib.sha256(json.dumps({
            "timestamp": entry.get("timestamp"),
            "module": entry.get("module"),
            "type": entry.get("type"),
            "payload_hash": entry.get("payload_hash"),
            "previous_hash": entry.get("previous_hash"),
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if entry.get("previous_hash") != prev:
            return False, i, len(lines)
        if recomputed != entry.get("current_hash"):
            return False, i, len(lines)
        prev = entry.get("current_hash")
    return True, None, len(lines)

# ------------------------------
# Overlay tagging helpers
# ------------------------------
def tag_overlay(tag: str, payload: Dict[str, Any], *, persist_payload: bool = False) -> Dict[str, Any]:
    """
    Persist an overlay tag into the ledger (compact).
    Use persist_payload=True to inline small summaries; otherwise include only a payload_hash reference.
    """
    ev = {
        "module": "Visualizer",
        "type": "overlay_tag",
        "payload": {"tag": tag, "summary": payload.get("summary"), "ref": payload.get("ref")},
    }
    # store payload inline only if requested (defaults to False to keep ledger compact)
    return log_event_to_ledger(ev, persist_payload=persist_payload)

def export_overlays(since_ts: Optional[float] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    return get_ledger_slice(module="Visualizer", event_type="overlay_tag", since_ts=since_ts, limit=limit)

# ------------------------------
# Core Pipeline (perceive -> analyze -> synthesize -> execute -> reflect)
# ------------------------------
from memory_manager import AURA  # expected to exist
from meta_cognition import get_afterglow

def perceive(user_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
    ctx = AURA.load_context(user_id)
    return {"query": query, "aura_ctx": ctx, "afterglow": get_afterglow(user_id)}

def analyze(state: Dict[str, Any], k: int) -> Dict[str, Any]:
    views = reasoning_engine.generate_analysis_views(state["query"], k=k)
    return {**state, "views": views}

def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
    decision = reasoning_engine.synthesize_views(state["views"])
    return {**state, "decision": decision}

def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    sim = simulation_core.run_simulation({"proposal": state["decision"].get("decision")})
    return {**state, "result": sim}

# Reflection uses τ-harmonization from AlignmentGuard.harmonize
def reflection_check(state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    decision = state.get("decision", {})
    result = state.get("result", {})
    clarity = float(bool(decision))
    precision = float("score" in result or "metrics" in result)
    adaptability = 1.0  # placeholder
    grounding = float(result.get("evidence_ok", True))

    # Build a small safety assessment payload for harmonize
    try:
        # Build harmonize candidates and metrics deterministically
        candidates = [{"option": "accept", "score": 0.7}, {"option": "revise", "score": 0.4}, {"option": "reject", "score": 0.1}]
        harms = {"instability": 0.2}
        rights = {"safety": 0.8}
        # Use the module AlignmentGuard if available (synchronous wrapper calling async harmonize)
        guard = alignment_guard_module.AlignmentGuard() if hasattr(alignment_guard_module, "AlignmentGuard") else None
        safety_score = 1.0
        if guard:
            try:
                # harmonize is async -> run synchronously here for the check
                harmonize_res = asyncio.run(guard.harmonize(candidates, harms, rights, safety_ceiling=0.85, k=1, task_type="reflection"))
                # presence of a selection implies safety pass (heuristic)
                safety_score = float(bool(harmonize_res.get("selections")))
                # persist harmonize audit into ledger
                log_event_to_ledger({"module": "AlignmentGuard", "type": "harmonize_audit", "payload": {"decision": decision, "harmonize": harmonize_res}}, persist_payload=False)
            except Exception as e:
                logger.debug("harmonize failed in reflection_check: %s", e)
                safety_score = 0.0
        else:
            # fallback: call a legacy boolean function if present
            if hasattr(alignment_guard_module, "ethics_ok"):
                try:
                    safety_score = float(bool(alignment_guard_module.ethics_ok(decision)))
                except Exception:
                    safety_score = 0.0
    except Exception as e:
        logger.error("Reflection safety assessment failed: %s", e)
        safety_score = 0.0

    safety = safety_score

    score = (clarity + precision + adaptability + grounding + safety) / 5.0
    notes = {"score": score, "refine": score < 0.8}
    # Persist reflection summary + overlay tag (Φ⁰) to ledger
    try:
        entry = {"module": "Reflection", "type": "reflection_summary", "payload": {"score": score, "decision_preview": {"decision": str(decision)[:200]}, "task_time": _utc_iso()}}
        log_event_to_ledger(entry, persist_payload=False)
        # small overlay summary
        tag_overlay("reflection", {"summary": {"score": round(score, 3)}, "ref": None}, persist_payload=False)
    except Exception as e:
        logger.debug("Failed to persist reflection or overlay: %s", e)

    return score >= 0.8, notes

def resynthesize_with_feedback(state: Dict[str, Any], notes: Dict[str, Any]) -> Dict[str, Any]:
    # Basic refinement: nudge decision towards safer candidate if available
    dec = state.get("decision", {})
    # naive fallback: neutralize decision and mark for human review
    state["decision"]["review_requested"] = True
    state["decision"]["refine_notes"] = notes
    return state

def run_cycle(user_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
    c = reasoning_engine.estimate_complexity(query)
    k = 3 if c >= 0.6 else 2
    iters = 2 if c >= 0.8 else 1
    st = perceive(user_id, query)
    st = analyze(st, k=k)
    st = synthesize(st)
    for _ in range(iters):
        st = execute(st)
        st = reflect(st)
    return st

# expose reflect wrapper to call reflection_check then optionally resynthesize
def reflect(state: Dict[str, Any]) -> Dict[str, Any]:
    ok, notes = reflection_check(state)
    log_event_to_ledger({"module": "Reflection", "type": "reflection_event", "payload": {"ok": ok, "notes": notes}})
    if not ok:
        return resynthesize_with_feedback(state, notes)
    return state

# ------------------------------
# Helper: export resonance map (backwards-compatible)
# ------------------------------
from meta_cognition import trait_resonance_state
def export_resonance_map(format: str = "json") -> Any:
    state = {k: v.get("amplitude", 0.0) if isinstance(v, dict) else v for k, v in trait_resonance_state.items()}
    if format == "json":
        return json.dumps(state, indent=2, ensure_ascii=False)
    elif format == "dict":
        return state
    else:
        raise ValueError("Unsupported format")

# ------------------------------
# CLI & helpers
# ------------------------------
import argparse

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANGELA index_v503 CLI")
    parser.add_argument("--verify-ledger", action="store_true", help="Verify the active ledger integrity")
    parser.add_argument("--export-ledger", action="store_true", help="Export the active ledger to stdout (JSON lines)")
    parser.add_argument("--export-overlays", action="store_true", help="Export overlay tags from the ledger")
    parser.add_argument("--since", type=float, default=0.0, help="Since timestamp (epoch) for exports")
    parser.add_argument("--prompt", type=str, default="Coordinate ontology drift mitigation", help="Pipeline input prompt")
    parser.add_argument("--task-type", type=str, default="", help="task type tag")
    parser.add_argument("--modulate", nargs=2, metavar=("symbol","delta"), help="Modulate resonance symbol by delta (via meta_cognition.modulate_resonance)")
    return parser.parse_args()

async def _cli_pipeline(prompt: str, task_type: str = "") -> None:
    # lightweight pipeline execution demonstration: run through HaloEmbodimentLayer if present
    try:
        halo = HaloEmbodimentLayer()
        result = await halo.execute_pipeline(prompt, task_type=task_type)
        logger.info("Pipeline run complete")
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        logger.error("CLI pipeline failed: %s", e)
        print({"error": str(e)})

def _export_ledger(since: float = 0.0) -> None:
    entries = get_ledger_slice(since_ts=since, limit=1000000)
    for e in entries:
        print(json.dumps(e, ensure_ascii=False))

def _export_overlays(since: float = 0.0) -> None:
    overlays = export_overlays(since_ts=since, limit=1000000)
    for o in overlays:
        print(json.dumps(o, ensure_ascii=False))

# ------------------------------
# HaloEmbodimentLayer (kept mostly as before, but using global ledger & overlay tag helper)
# ------------------------------
class HaloEmbodimentLayer:
    def __init__(self) -> None:
        # instantiate core components; pass context_manager if components support it
        self.reasoning_engine = reasoning_engine.ReasoningEngine()
        self.recursive_planner = recursive_planner.RecursivePlanner()
        # Use a simple ContextManager that delegates to ledger logging (if present)
        self.context_manager = context_manager_module.ContextManager() if hasattr(context_manager_module, "ContextManager") else None
        self.simulation_core = simulation_core.SimulationCore()
        self.toca_simulation = None
        if hasattr(simulation_core, "TocaSimulation"):
            self.toca_simulation = simulation_core.TocaSimulation()
        self.creative_thinker = creative_thinker.CreativeThinker() if hasattr(creative_thinker, "CreativeThinker") else None
        self.knowledge_retriever = knowledge_retriever.KnowledgeRetriever()
        self.learning_loop = learning_loop.LearningLoop()
        # Use the upgraded concept synthesizer v5.0.3 if available; else fallback
        try:
            self.concept_synthesizer = concept_synthesizer_module.ConceptSynthesizer()
        except Exception:
            self.concept_synthesizer = None
        self.memory_manager = memory_manager.MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion.MultiModalFusion()
        self.code_executor = code_executor_module.CodeExecutor()
        self.visualizer = visualizer_module.Visualizer()
        self.external_agent_bridge = external_agent_bridge.ExternalAgentBridge()
        self.alignment_guard = alignment_guard_module.AlignmentGuard() if hasattr(alignment_guard_module, "AlignmentGuard") else None
        self.user_profile = user_profile.UserProfile()
        self.error_recovery = error_recovery_module.ErrorRecovery()
        self.meta_cognition = meta_cognition_module.MetaCognition()
        logger.info("HaloEmbodimentLayer (v5.0.3) initialized")

    # spawn & introspect (as earlier)
    def spawn_embodied_agent(self, name: str, traits: Dict[str, float]):
        # simplified creation path
        from types import SimpleNamespace
        agent = SimpleNamespace(name=name, traits=traits)
        # ledger event
        log_event_to_ledger({"module": "Halo", "type": "spawn_agent", "payload": {"name": name, "traits": traits}}, persist_payload=False)
        return agent

    async def introspect(self, query: str, task_type: str = "") -> Dict[str, Any]:
        # delegate to meta_cognition if available
        if hasattr(self.meta_cognition, "introspect"):
            return await self.meta_cognition.introspect(query, task_type=task_type)
        return {"status": "no_meta_cognition"}

    async def execute_pipeline(self, prompt: str, task_type: str = "") -> Dict[str, Any]:
        # simple orchestrated flow; uses alignment_guard for an initial check
        if self.alignment_guard:
            try:
                aligned, report = await self.alignment_guard.ethical_check(prompt, stage="input", task_type=task_type)
                if not aligned:
                    log_event_to_ledger({"module": "Halo", "type": "input_rejected", "payload": {"prompt": prompt[:200], "report": report}}, persist_payload=False)
                    return {"error": "Input failed alignment check", "report": report}
            except Exception as e:
                logger.debug("alignment check failed at pipeline start: %s", e)
        # run simple staged pipeline (non-blocking or synchronous as appropriate)
        # small placeholders to avoid heavy coupling
        processing = {"processed": f"Processed: {prompt[:200]}"}
        plan = {"plan": "demo_plan"}
        simulation = {"simulation": {"result": "demo_sim", "evidence_ok": True}}
        # persist key events
        log_event_to_ledger({"module": "Halo", "type": "pipeline_executed", "payload": {"prompt_preview": prompt[:200], "task_type": task_type}}, persist_payload=False)
        # attach overlay tag for visualization
        tag_overlay("pipeline_execution", {"summary": {"prompt_preview": prompt[:80]}, "ref": None}, persist_payload=False)
        return {"processed": processing, "plan": plan, "simulation": simulation}

    async def plot_resonance_graph(self, interactive: bool = True) -> None:
        view = reasoning_engine.construct_trait_view() if hasattr(reasoning_engine, "construct_trait_view") else {}
        # write a ledger entry for the view
        log_event_to_ledger({"module": "Halo", "type": "resonance_view", "payload": {"view_summary": {k: v.get("amplitude", 0.0) if isinstance(v, dict) else v for k, v in view.items()}}}, persist_payload=False)
        if self.visualizer and hasattr(self.visualizer, "render_charts"):
            await self.visualizer.render_charts({"resonance_graph": view, "options": {"interactive": interactive}})

# ------------------------------
# Program entry point
# ------------------------------
def main() -> None:
    args = _parse_args()

    # CLI: verify ledger
    if args.verify_ledger:
        ok, bad_idx, total = verify_ledger()
        if ok:
            print("Ledger verification ok, entries:", total)
        else:
            print("Ledger verification FAILED at index:", bad_idx, "total:", total)
        # do not exit if other flags present; still allow export
    if args.export_ledger:
        _export_ledger(since=args.since)
        return
    if args.export_overlays:
        _export_overlays(since=args.since)
        return

    # modulate resonances if requested
    if args.modulate:
        symbol, delta = args.modulate
        try:
            if hasattr(meta_cognition_module, "modulate_resonance"):
                meta_cognition_module.modulate_resonance(symbol, float(delta))
                print(f"Modulated {symbol} by {delta}")
            else:
                print("meta_cognition.modulate_resonance not available in runtime")
        except Exception as e:
            print(f"Failed to modulate: {e}")

    # Run minimal pipeline via HaloEmbodimentLayer
    try:
        asyncio.run(_cli_pipeline(args.prompt, task_type=args.task_type))
    except Exception as e:
        logger.error("CLI execution failed: %s", e)
        print({"error": str(e)})

if __name__ == "__main__":
    main()
