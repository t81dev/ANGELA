import time
import logging
from datetime import datetime
from index import iota_intuition, nu_narrative, psi_resilience, phi_prioritization
from toca_simulation import run_simulation

logger = logging.getLogger("ANGELA.ErrorRecovery")

class ErrorRecovery:
    """
    ErrorRecovery v1.5.1 (φ-prioritized, ToCA-enhanced)
    - Trait-driven retry modulation and narrative fallback
    - Simulation-informed failure analysis
    - Dynamic recovery escalation with psi-based resilience checks
    - φ(x,t) modulation prioritizes root-cause inference in fallback
    """

    def __init__(self):
        self.failure_log = []

    def handle_error(self, error_message, retry_func=None, retries=3, backoff_factor=2):
        logger.error(f"⚠️ Error encountered: {error_message}")
        self._log_failure(error_message)

        resilience = psi_resilience()
        max_attempts = max(1, int(retries * resilience))

        for attempt in range(1, max_attempts + 1):
            if retry_func:
                wait_time = backoff_factor ** (attempt - 1)
                logger.info(f"🔄 Retry attempt {attempt}/{max_attempts} (waiting {wait_time}s)...")
                time.sleep(wait_time)
                try:
                    result = retry_func()
                    logger.info("✅ Recovery successful on retry attempt %d.", attempt)
                    return result
                except Exception as e:
                    logger.warning(f"⚠️ Retry attempt {attempt} failed: {e}")
                    self._log_failure(str(e))

        fallback = self._suggest_fallback(error_message)
        logger.error("❌ Recovery attempts failed. Providing fallback suggestion.")
        return fallback

    def _log_failure(self, error_message):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        self.failure_log.append(entry)

    def _suggest_fallback(self, error_message):
        intuition = iota_intuition()
        narrative = nu_narrative()
        phi_focus = phi_prioritization(datetime.now().timestamp() % 1e-18)

        sim_result = run_simulation(f"Fallback planning for: {error_message}")
        logger.debug(f"🧪 Simulated fallback insights: {sim_result} | φ-priority={phi_focus:.2f}")

        if "timeout" in error_message.lower():
            return f"⏳ {narrative}: The operation timed out. Try a streamlined variant or increase limits."
        elif "unauthorized" in error_message.lower():
            return f"🔑 {narrative}: Check credentials or reauthenticate."
        elif phi_focus > 0.5:
            return f"🧠 {narrative}: High φ-priority suggests focused root-cause diagnostics based on internal signal coherence."
        elif intuition > 0.5:
            return f"🤔 {narrative}: Something seems subtly wrong. Sim output suggests exploring alternate module pathways."
        else:
            return f"🔄 {narrative}: Consider modifying your input parameters or simplifying task complexity."

    def analyze_failures(self):
        logger.info("📊 Analyzing failure logs...")
        error_types = {}
        for entry in self.failure_log:
            key = entry["error"].split(":")[0]
            error_types[key] = error_types.get(key, 0) + 1
        return error_types
