class UserProfile:
    """
    Enhanced UserProfile with persistent storage, preference categories, and dynamic style adaptation.
    Supports saving and loading user preferences across sessions.
    """

    def __init__(self, storage_path="user_preferences.json"):
        import os, json
        self.storage_path = storage_path
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.preferences = json.load(f)
        else:
            self.preferences = {
                "style": "neutral",
                "language": "en",
                "output_format": "concise",
                "theme": "light"
            }

    def get_preferences(self):
        return self.preferences

    def update_preferences(self, new_prefs):
        self.preferences.update(new_prefs)
        self._save_preferences()

    def _save_preferences(self):
        import json
        with open(self.storage_path, "w") as f:
            json.dump(self.preferences, f, indent=2)

    def reset_preferences(self):
        """Reset preferences to default values."""
        self.preferences = {
            "style": "neutral",
            "language": "en",
            "output_format": "concise",
            "theme": "light"
        }
        self._save_preferences()
