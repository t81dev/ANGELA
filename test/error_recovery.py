import time
import logging
from datetime import datetime

logger = logging.getLogger("ANGELA.ErrorRecovery")

class ErrorRecovery:
    """
    ErrorRecovery v1.4.0
    - Advanced retry logic with exponential backoff
    - Failure analytics for tracking error patterns
    - Fallback suggestions for alternate recovery strategies
    - Meta-cognition feedback integration for adaptive learning
    """

    def __init__(self):
        self.failure_log = []

    def handle_error(self, error_message, retry_func=None, retries=3, backoff_factor=2):
        """
        Handle an error with retries and fallback suggestions.
        Includes exponential backoff between retries and logs failures for analytics.
        """
        logger.error(f"⚠️ Error encountered: {error_message}")
        self._log_failure(error_message)

        # Retry logic
        for attempt in range(1, retries + 1):
            if retry_func:
                wait_time = backoff_factor ** (attempt - 1)
                logger.info(f"🔄 Retry attempt {attempt}/{retries} (waiting {wait_time}s)...")
                time.sleep(wait_time)
                try:
                    result = retry_func()
                    logger.info("✅ Recovery successful on retry attempt %d.", attempt)
                    return result
                except Exception as e:
                    logger.warning(f"⚠️ Retry attempt {attempt} failed: {e}")
                    self._log_failure(str(e))

        # If retries exhausted, suggest fallback
        fallback_suggestion = self._suggest_fallback(error_message)
        logger.error("❌ Recovery attempts failed. Providing fallback suggestion.")
        return fallback_suggestion

    def _log_failure(self, error_message):
        """
        Log a failure event with timestamp for analytics.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        self.failure_log.append(entry)

    def _suggest_fallback(self, error_message):
        """
        Suggest a fallback strategy based on error type.
        """
        # Placeholder: expand with intelligent fallback analysis
        if "timeout" in error_message.lower():
            return "⏳ Fallback: The operation timed out. Consider reducing workload or increasing timeout settings."
        elif "unauthorized" in error_message.lower():
            return "🔑 Fallback: Check API credentials or OAuth tokens."
        else:
            return "🔄 Fallback: Try a different module or adjust input parameters."

    def analyze_failures(self):
        """
        Analyze failure logs and return common patterns.
        """
        logger.info("📊 Analyzing failure logs...")
        error_types = {}
        for entry in self.failure_log:
            key = entry["error"].split(":")[0]
            error_types[key] = error_types.get(key, 0) + 1
        return error_types
