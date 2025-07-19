class ErrorRecovery:
    """
    Enhanced ErrorRecovery with retry logic and fallback suggestions.
    Provides graceful recovery strategies for different error types.
    """

    def handle_error(self, error_message, retry_func=None, retries=1):
        print(f"⚠️ An error occurred: {error_message}. Attempting graceful recovery...")
        
        # Attempt retries if a retry function is provided
        for attempt in range(retries):
            if retry_func:
                try:
                    print(f"🔄 Retry attempt {attempt + 1}...")
                    result = retry_func()
                    print("✅ Recovery successful.")
                    return result
                except Exception as e:
                    print(f"⚠️ Retry attempt {attempt + 1} failed: {e}")

        # If all retries fail, provide fallback suggestion
        return f"❌ Recovery attempts failed after {retries} retries. Please check logs or try a different approach."
