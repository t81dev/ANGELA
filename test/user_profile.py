import json
import os
import logging
from datetime import datetime
from index import epsilon_identity

logger = logging.getLogger("ANGELA.UserProfile")

class UserProfile:
    """
    UserProfile v1.4.0
    - Multi-profile support with dynamic preference inheritance
    - Persistent storage of preferences per user ID
    - Audit trail for preference changes
    - Identity-modulated preference weighting (ε_identity)
    """

    DEFAULT_PREFERENCES = {
        "style": "neutral",
        "language": "en",
        "output_format": "concise",
        "theme": "light"
    }

    def __init__(self, storage_path="user_profiles.json"):
        self.storage_path = storage_path
        self._load_profiles()
        self.active_user = None

    def _load_profiles(self):
        """
        Load all user profiles from storage.
        """
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.profiles = json.load(f)
            logger.info("✅ User profiles loaded from storage.")
        else:
            self.profiles = {}
            logger.info("🆕 No profiles found. Initialized empty profiles store.")

    def _save_profiles(self):
        """
        Save all user profiles to storage.
        """
        with open(self.storage_path, "w") as f:
            json.dump(self.profiles, f, indent=2)
        logger.info("💾 User profiles saved to storage.")

    def switch_user(self, user_id):
        """
        Switch to a specific user profile, creating it if necessary.
        """
        if user_id not in self.profiles:
            logger.info(f"🆕 Creating new profile for user '{user_id}'")
            self.profiles[user_id] = {
                "preferences": self.DEFAULT_PREFERENCES.copy(),
                "audit_log": []
            }
            self._save_profiles()
        self.active_user = user_id
        logger.info(f"👤 Active user switched to: {user_id}")

    def get_preferences(self, fallback=True):
        """
        Get the preferences for the active user.
        If fallback=True, use defaults for missing keys.
        Applies ε_identity for weighted blending.
        """
        if not self.active_user:
            logger.warning("⚠️ No active user. Returning default preferences.")
            return self.DEFAULT_PREFERENCES.copy()

        prefs = self.profiles[self.active_user]["preferences"].copy()
        if fallback:
            for key, value in self.DEFAULT_PREFERENCES.items():
                prefs.setdefault(key, value)

        # Modulate style preference using ε_identity
        weight = epsilon_identity(time=datetime.now().timestamp())
        if prefs["style"] != "neutral":
            prefs["style"] = f"{prefs['style']} (modulated ε={weight:.2f})"

        return prefs

    def update_preferences(self, new_prefs):
        """
        Update preferences for the active user and log changes.
        """
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")

        timestamp = datetime.now().isoformat()
        old_prefs = self.profiles[self.active_user]["preferences"]
        changes = {k: (old_prefs.get(k), v) for k, v in new_prefs.items()}
        self.profiles[self.active_user]["preferences"].update(new_prefs)

        self.profiles[self.active_user]["audit_log"].append({
            "timestamp": timestamp,
            "changes": changes
        })

        self._save_profiles()
        logger.info(f"🔄 Preferences updated for user '{self.active_user}'")

    def reset_preferences(self):
        """
        Reset preferences for the active user to defaults.
        """
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")

        self.profiles[self.active_user]["preferences"] = self.DEFAULT_PREFERENCES.copy()
        timestamp = datetime.now().isoformat()
        self.profiles[self.active_user]["audit_log"].append({
            "timestamp": timestamp,
            "changes": "Preferences reset to defaults."
        })
        self._save_profiles()
        logger.info(f"♻️ Preferences reset for user '{self.active_user}'")

    def get_audit_log(self):
        """
        Retrieve the audit log for the active user.
        """
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")
        return self.profiles[self.active_user]["audit_log"]
