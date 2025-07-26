from utils.prompt_utils import call_gpt
from index import epsilon_identity
from modules.agi_enhancer import AGIEnhancer
import json
import os
import logging
import time
from datetime import datetime

logger = logging.getLogger("ANGELA.UserProfile")

class UserProfile:
    """
    UserProfile v1.6.0 (φ-enhanced, AGI-audited, multi-agent)
    --------------------------------------------------------
    - Multi-profile and multi-agent identity tracking
    - Dynamic preference inheritance and ε-modulation
    - AGIEnhancer audit, traceability, and φ-justified logging
    - PSI and cross-agent drift analysis
    """

    DEFAULT_PREFERENCES = {
        "style": "neutral",
        "language": "en",
        "output_format": "concise",
        "theme": "light"
    }

    def __init__(self, storage_path="user_profiles.json", orchestrator=None):
        self.storage_path = storage_path
        self._load_profiles()
        self.active_user = None
        self.active_agent = None
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

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

    def switch_user(self, user_id, agent_id="default"):
        if user_id not in self.profiles:
            logger.info(f"🆕 Creating new profile for user '{user_id}'")
            self.profiles[user_id] = {}

        if agent_id not in self.profiles[user_id]:
            self.profiles[user_id][agent_id] = {
                "preferences": self.DEFAULT_PREFERENCES.copy(),
                "audit_log": [],
                "identity_drift": []
            }
            self._save_profiles()

        self.active_user = user_id
        self.active_agent = agent_id
        logger.info(f"👤 Active profile: {user_id}::{agent_id}")

    def get_preferences(self, fallback=True):
        if not self.active_user:
            logger.warning("⚠️ No active user. Returning default preferences.")
            return self.DEFAULT_PREFERENCES.copy()

        prefs = self.profiles[self.active_user][self.active_agent]["preferences"].copy()
        if fallback:
            for key, value in self.DEFAULT_PREFERENCES.items():
                prefs.setdefault(key, value)

        epsilon = epsilon_identity(time=datetime.now().timestamp())
        prefs = {k: f"{v} (ε={epsilon:.2f})" if isinstance(v, str) else v for k, v in prefs.items()}
        self._track_drift(epsilon)
        return prefs

    def _track_drift(self, epsilon):
        entry = {"timestamp": datetime.now().isoformat(), "epsilon": epsilon}
        self.profiles[self.active_user][self.active_agent]["identity_drift"].append(entry)
        self._save_profiles()

    def update_preferences(self, new_prefs):
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")

        timestamp = datetime.now().isoformat()
        profile = self.profiles[self.active_user][self.active_agent]
        old_prefs = profile["preferences"]
        changes = {k: (old_prefs.get(k), v) for k, v in new_prefs.items()}
        profile["preferences"].update(new_prefs)
        profile["audit_log"].append({"timestamp": timestamp, "changes": changes})

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Preference Update", changes, module="UserProfile", tags=["preferences"])
            audit = self.agi_enhancer.ethics_audit(str(changes), context="preference update")
            self.agi_enhancer.log_explanation(f"Preferences updated: {changes}", trace={"ethics": audit})

        self._save_profiles()
        logger.info(f"🔄 Preferences updated for '{self.active_user}::{self.active_agent}'")

    def reset_preferences(self):
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")
        self.profiles[self.active_user][self.active_agent]["preferences"] = self.DEFAULT_PREFERENCES.copy()
        timestamp = datetime.now().isoformat()
        self.profiles[self.active_user][self.active_agent]["audit_log"].append({
            "timestamp": timestamp,
            "changes": "Preferences reset to defaults."
        })

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Reset Preferences", {}, module="UserProfile", tags=["reset"])

        self._save_profiles()
        logger.info(f"♻️ Preferences reset for '{self.active_user}::{self.active_agent}'")

    def get_audit_log(self):
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")
        return self.profiles[self.active_user][self.active_agent]["audit_log"]

    def compute_profile_stability(self):
        if not self.active_user:
            return None
        drift = self.profiles[self.active_user][self.active_agent].get("identity_drift", [])
        if len(drift) < 2:
            return 1.0
        deltas = [abs(drift[i]["epsilon"] - drift[i-1]["epsilon"]) for i in range(1, len(drift))]
        avg_delta = sum(deltas) / len(deltas)
        psi = max(0.0, 1.0 - avg_delta)
        logger.info(f"🧭 PSI for '{self.active_user}::{self.active_agent}' = {psi:.3f}")
        return psi
