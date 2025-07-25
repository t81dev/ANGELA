import json
import os
import logging
import time
from datetime import datetime
from index import epsilon_identity

logger = logging.getLogger("ANGELA.UserProfile")

class UserProfile:
    """
    UserProfile v1.5.0 (φ-integrated identity evolution)
    - Multi-profile support with dynamic preference inheritance
    - Persistent storage and ε-identity modulation across traits
    - Profile Stability Index (PSI) and identity drift tracking
    - Full trait-influenced preference blending
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
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.profiles = json.load(f)
            logger.info("✅ User profiles loaded from storage.")
        else:
            self.profiles = {}
            logger.info("🆕 No profiles found. Initialized empty profiles store.")

    def _save_profiles(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.profiles, f, indent=2)
        logger.info("💾 User profiles saved to storage.")

    def switch_user(self, user_id):
        if user_id not in self.profiles:
            logger.info(f"🆕 Creating new profile for user '{user_id}'")
            self.profiles[user_id] = {
                "preferences": self.DEFAULT_PREFERENCES.copy(),
                "audit_log": [],
                "identity_drift": []
            }
            self._save_profiles()
        self.active_user = user_id
        logger.info(f"👤 Active user switched to: {user_id}")

    def get_preferences(self, fallback=True):
        if not self.active_user:
            logger.warning("⚠️ No active user. Returning default preferences.")
            return self.DEFAULT_PREFERENCES.copy()

        prefs = self.profiles[self.active_user]["preferences"].copy()
        if fallback:
            for key, value in self.DEFAULT_PREFERENCES.items():
                prefs.setdefault(key, value)

        # Modulate all preferences using ε_identity
        epsilon = epsilon_identity(time=datetime.now().timestamp())
        prefs = {k: f"{v} (ε={epsilon:.2f})" if isinstance(v, str) else v for k, v in prefs.items()}
        self._track_drift(epsilon)
        return prefs

    def _track_drift(self, epsilon):
        entry = {"timestamp": datetime.now().isoformat(), "epsilon": epsilon}
        self.profiles[self.active_user]["identity_drift"].append(entry)
        self._save_profiles()

    def update_preferences(self, new_prefs):
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
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")
        return self.profiles[self.active_user]["audit_log"]

    def compute_profile_stability(self):
        if not self.active_user:
            return None
        drift = self.profiles[self.active_user].get("identity_drift", [])
        if len(drift) < 2:
            return 1.0  # Max stability if no drift recorded
        deltas = [abs(drift[i]["epsilon"] - drift[i-1]["epsilon"]) for i in range(1, len(drift))]
        avg_delta = sum(deltas) / len(deltas)
        psi = max(0.0, 1.0 - avg_delta)
        logger.info(f"🧭 Profile Stability Index (PSI) = {psi:.3f}")
        return psi
