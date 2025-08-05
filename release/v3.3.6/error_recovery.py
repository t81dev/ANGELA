"""
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the ErrorRecovery class for handling errors and recovering in the ANGELA v3.5 architecture.
"""

import time
import logging
import hashlib
import re
from datetime import datetime
from typing import Callable, Any, Optional, Dict, List
from collections import deque
from functools import lru_cache

from index import iota_intuition, nu_narrative, psi_resilience, phi_prioritization
from toca_simulation import run_simulation
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer
from modules.context_manager import ContextManager

logger = logging.getLogger("ANGELA.ErrorRecovery")

def hash_failure(event: Dict[str, Any]) -> str:
    """Compute a SHA-256 hash of a failure event."""
    raw = f"{event['timestamp']}{event['error']}{event.get('resolved', False)}"
    return hashlib.sha256(raw.encode()).hexdigest()

class ErrorRecovery:
    """A class for handling errors and recovering in the ANGELA v3.5 architecture.

    Attributes:
        failure_log (deque): Log of failure events with timestamps and error messages.
        omega (dict): System-wide state with timeline, traits, symbolic_log, and timechain.
        error_index (dict): Index mapping error messages to timeline entries.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for retrying code-based operations.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for fallback suggestions.
        context_manager (ContextManager): Optional context manager for contextual recovery.
    """

    def __init__(self, alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None,
                 concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 context_manager: Optional[ContextManager] = None):
        self.failure_log = deque(maxlen=1000)
        self.omega = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000)
        }
        self.error_index = {}
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        logger.info("ErrorRecovery initialized")

    def handle_error(self, error_message: str, retry_func: Optional[Callable[[], Any]] = None,
                     retries: int = 3, backoff_factor: float = 2.0) -> Any:
        """Handle an error with retries and fallback suggestions.

        Args:
            error_message: Description of the error.
            retry_func: Optional function to retry.
            retries: Number of retry attempts (default: 3).
            backoff_factor: Exponential backoff factor (default: 2.0).

        Returns:
            Result of retry_func if successful, otherwise a fallback suggestion.

        Raises:
            TypeError: If error_message is not a string or retry_func is not callable.
            ValueError: If retries or backoff_factor is invalid or error_message fails alignment check.
        """
        if not isinstance(error_message, str):
            logger.error("Invalid error_message type: must be a string.")
            raise TypeError("error_message must be a string")
        if retry_func is not None and not callable(retry_func):
            logger.error("Invalid retry_func: must be callable or None.")
            raise TypeError("retry_func must be callable or None")
        if not isinstance(retries, int) or retries < 0:
            logger.error("Invalid retries: must be a non-negative integer.")
            raise ValueError("retries must be a non-negative integer")
        if not isinstance(backoff_factor, (int, float)) or backoff_factor <= 0:
            logger.error("Invalid backoff_factor: must be a positive number.")
            raise ValueError("backoff_factor must be a positive number")
        if self.alignment_guard and not self.alignment_guard.check(error_message):
            logger.warning("Error message failed alignment check.")
            raise ValueError("Error message failed alignment check")

        logger.error("Error encountered: %s", error_message)
        self._log_failure(error_message)

        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "error_handled", "error": error_message})

        try:
            resilience = psi_resilience()
            max_attempts = max(1, int(retries * resilience))

            for attempt in range(1, max_attempts + 1):
                if retry_func:
                    wait_time = backoff_factor ** (attempt - 1)
                    logger.info("Retry attempt %d/%d (waiting %.2fs)...", attempt, max_attempts, wait_time)
                    time.sleep(wait_time)
                    try:
                        if self.code_executor and callable(retry_func):
                            result = self.code_executor.execute(retry_func.__code__, language="python")
                            if result["success"]:
                                logger.info("Recovery successful on retry attempt %d.", attempt)
                                return result["output"]
                        else:
                            result = retry_func()
                            logger.info("Recovery successful on retry attempt %d.", attempt)
                            return result
                    except Exception as e:
                        logger.warning("Retry attempt %d failed: %s", attempt, str(e))
                        self._log_failure(str(e))

            fallback = self._suggest_fallback(error_message)
            self._link_timechain_failure(error_message)
            logger.error("Recovery attempts failed. Providing fallback suggestion: %s", fallback)
            return fallback
        except Exception as e:
            logger.error("Error handling failed: %s", str(e))
            raise

    def _log_failure(self, error_message: str) -> None:
        """Log a failure event with timestamp."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        self.failure_log.append(entry)
        self.omega["timeline"].append(entry)
        self.error_index[error_message] = entry
        logger.debug("Failure logged: %s", entry)

    def _suggest_fallback(self, error_message: str) -> str:
        """Suggest a fallback strategy for an error."""
        try:
            t = time.time()
            intuition = iota_intuition()
            narrative = nu_narrative()
            phi_focus = phi_prioritization(t)

            sim_result = self._cached_run_simulation(f"Fallback planning for: {error_message}") or "no simulation data"
            logger.debug("Simulated fallback insights: %s | φ-priority=%.2f", sim_result, phi_focus)

            if self.concept_synthesizer:
                synthesis_result = self.concept_synthesizer.synthesize(error_message, style="fallback")
                if synthesis_result["valid"]:
                    return synthesis_result["concept"]

            if re.search(r"timeout|timed out", error_message, re.IGNORECASE):
                return f"{narrative}: The operation timed out. Try a streamlined variant or increase limits."
            elif re.search(r"unauthorized|permission", error_message, re.IGNORECASE):
                return f"{narrative}: Check credentials or reauthenticate."
            elif phi_focus > 0.5:
                return f"{narrative}: High φ-priority suggests focused root-cause diagnostics."
            elif intuition > 0.5:
                return f"{narrative}: Intuition suggests exploring alternate module pathways."
            else:
                return f"{narrative}: Consider modifying input parameters or simplifying task complexity."
        except Exception as e:
            logger.error("Fallback suggestion failed: %s", str(e))
            return f"Error generating fallback: {str(e)}"

    def _link_timechain_failure(self, error_message: str) -> None:
        """Link a failure to the timechain with a hash."""
        failure_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "resolved": False
        }
        prev_hash = self.omega["timechain"][-1]["hash"] if self.omega["timechain"] else ""
        entry_hash = hash_failure(failure_entry)
        self.omega["timechain"].append({"event": failure_entry, "hash": entry_hash, "prev": prev_hash})
        logger.debug("Timechain updated with failure: %s", entry_hash)

    def trace_failure_origin(self, error_message: str) -> Optional[Dict[str, Any]]:
        """Trace the origin of a failure in the Ω timeline."""
        if not isinstance(error_message, str):
            logger.error("Invalid error_message type: must be a string.")
            raise TypeError("error_message must be a string")
        
        if error_message in self.error_index:
            event = self.error_index[error_message]
            logger.info("Failure trace found in Ω: %s", event)
            return event
        logger.info("No causal trace found in Ω timeline.")
        return None

    def detect_symbolic_drift(self, recent: int = 5) -> bool:
        """Detect symbolic drift in recent symbolic log entries."""
        if not isinstance(recent, int) or recent <= 0:
            logger.error("Invalid recent: must be a positive integer.")
            raise ValueError("recent must be a positive integer")
        
        recent_symbols = list(self.omega["symbolic_log"])[-recent:]
        if len(set(recent_symbols)) < recent / 2:
            logger.warning("Symbolic drift detected: repeated or unstable symbolic states.")
            return True
        return False

    def analyze_failures(self) -> Dict[str, int]:
        """Analyze failure logs for recurring error patterns."""
        logger.info("Analyzing failure logs...")
        error_types = {}
        for entry in self.failure_log:
            key = entry["error"].split(":")[0].strip()
            error_types[key] = error_types.get(key, 0) + 1
        for error, count in error_types.items():
            if count > 3:
                logger.warning("Pattern detected: '%s' recurring %d times.", error, count)
        return error_types

    @lru_cache(maxsize=100)
    def _cached_run_simulation(self, input_str: str) -> str:
        """Cached wrapper for run_simulation."""
        return run_simulation(input_str)
