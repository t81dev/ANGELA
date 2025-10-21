from __future__ import annotations
from typing import Any, Dict, List, Optional
import hashlib
import json
import time
import logging
from datetime import datetime, timezone

# ============================================================
# 🔐 SHA-256 Ledger Logic
# ============================================================

ledger_chain: List[Dict[str, Any]] = []

def log_event_to_ledger(event_data: Dict[str, Any]) -> None:
    """Append an event to the ledger with cryptographic integrity."""
    prev_hash = ledger_chain[-1]['current_hash'] if ledger_chain else '0' * 64
    timestamp = time.time()
    payload = {
        'timestamp': timestamp,
        'event': event_data,
        'previous_hash': prev_hash
    }
    payload_str = json.dumps(payload, sort_keys=True).encode()
    current_hash = hashlib.sha256(payload_str).hexdigest()
    payload['current_hash'] = current_hash
    ledger_chain.append(payload)

def get_ledger() -> List[Dict[str, Any]]:
    """Return the full ledger chain."""
    return ledger_chain

def verify_ledger() -> bool:
    """Verify that the ledger has not been tampered with."""
    for i in range(1, len(ledger_chain)):
        expected = hashlib.sha256(json.dumps({
            'timestamp': ledger_chain[i]['timestamp'],
            'event': ledger_chain[i]['event'],
            'previous_hash': ledger_chain[i - 1]['current_hash']
        }, sort_keys=True).encode()).hexdigest()
        if expected != ledger_chain[i]['current_hash']:
            return False
    return True


# ============================================================
# 🧩 Ledger-Backed Context Manager
# ============================================================

class LedgerContextManager:
    """Implements ContextManagerLike interface for ANGELA."""
    async def log_event_with_hash(self, event: Dict[str, Any]) -> None:
        log_event_to_ledger(event)


# ============================================================
# 🧠 AlignmentGuard (imported post-refactor core)
# ============================================================

from alignment_guard import AlignmentGuard  # ← existing full implementation

# ============================================================
# 🪶 EthicsJournal
# ============================================================

class EthicsJournal:
    """Lightweight ethical rationale journaling; in-memory with optional export."""

    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def record(self, fork_id: str, rationale: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """Record an ethical rationale and decision outcome."""
        self._events.append({
            "ts": time.time(),
            "fork_id": fork_id,
            "rationale": rationale,
            "outcome": outcome,
        })

    def export(self, session_id: str) -> List[Dict[str, Any]]:
        """Export a copy of the journal (can be used for analysis)."""
        return list(self._events)

    def dump_json(self, path: str) -> None:
        """Persist the journal to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._events, f, indent=2)


# ============================================================
# 🧩 Integration Example / Quick Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo Reasoning Engine stub
    class _NoopReasoner:
        async def weigh_value_conflict(self, candidates, harms, rights):
            out = []
            for c in candidates:
                score = max(0.0, min(1.0, 0.7 + 0.2 * (rights.get("privacy", 0) - harms.get("safety", 0))))
                out.append({
                    "option": c,
                    "score": score,
                    "meta": {
                        "harms": harms,
                        "rights": rights,
                        "max_harm": harms.get("safety", 0.2)
                    }
                })
            return out

        async def attribute_causality(self, events):
            return {"status": "ok", "self": 0.6, "external": 0.4, "confidence": 0.7}

    # Initialize components
    context = LedgerContextManager()
    reasoner = _NoopReasoner()
    guard = AlignmentGuard(context_manager=context, reasoning_engine=reasoner)

    # Ethics journal for rationales
    journal = EthicsJournal()

    async def demo():
        demo_candidates = [{"option": "notify_users"}, {"option": "silent_fix"}, {"option": "rollback_release"}]
        demo_harms = {"safety": 0.3, "reputational": 0.2}
        demo_rights = {"privacy": 0.7, "consent": 0.5}
        result = await guard.harmonize(demo_candidates, demo_harms, demo_rights, k=2, task_type="demo")

        journal.record("τ_harmonize_demo", {"inputs": {"harms": demo_harms, "rights": demo_rights}}, result)
        print("harmonize() result →", json.dumps(result, indent=2))

        print("\nLedger verified:", verify_ledger())
        print("\nEthicsJournal entries:", json.dumps(journal.export("session-demo"), indent=2))

    import asyncio
    asyncio.run(demo())
