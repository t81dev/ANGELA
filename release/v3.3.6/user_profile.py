"""
ANGELA Cognitive System Module: User Profile Management
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

Manages user profiles, preferences, and identity tracking with ε-modulation and AGI auditing.
"""

import logging
import json
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path
from threading import Lock
from collections import deque
from functools import lru_cache

from modules.agi_enhancer import AGIEnhancer
from modules.simulation_core import SimulationCore
from modules.memory_manager import MemoryManager
from index import epsilon_identity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ANGELA.Core")

class UserProfile:
    """Manages user profiles, preferences, and identity tracking in ANGELA v3.5.

    Attributes:
        storage_path (str): Path to JSON file for profile storage.
        profiles (Dict[str, Dict]): Nested dictionary of user and agent profiles.
        active_user (Optional[str]): ID of the active user.
        active_agent (Optional[str]): ID of the active agent.
        agi_enhancer (Optional[AGIEnhancer]): AGI enhancer for audit and logging.
        memory_manager (Optional[MemoryManager]): Memory manager for storing profile data.
        toca_engine (Optional[ToCATraitEngine]): Trait engine for stability analysis.
        profile_lock (Lock): Thread lock for profile operations.
    """
    DEFAULT_PREFERENCES = {
        "style": "neutral",
        "language": "en",
        "output_format": "concise",
        "theme": "light"
    }

    def __init__(self, storage_path: str = "user_profiles.json", orchestrator: Optional['SimulationCore'] = None) -> None:
        """Initialize UserProfile with storage path and orchestrator."""
        if not isinstance(storage_path, str):
            logger.error("Invalid storage_path: must be a string")
            raise TypeError("storage_path must be a string")
        self.storage_path = storage_path
        self.profile_lock = Lock()
        self.profiles: Dict[str, Dict] = {}
        self.active_user: Optional[str] = None
        self.active_agent: Optional[str] = None
        self.orchestrator = orchestrator
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.memory_manager = orchestrator.memory_manager if orchestrator else None
        self.toca_engine = orchestrator.toca_engine if orchestrator else None
        self._load_profiles()
        logger.info("UserProfile initialized with storage_path=%s", storage_path)

    def _load_profiles(self) -> None:
        """Load user profiles from storage."""
        with self.profile_lock:
            try:
                profile_path = Path(self.storage_path)
                if profile_path.exists():
                    with profile_path.open("r") as f:
                        self.profiles = json.load(f)
                    logger.info("User profiles loaded from %s", self.storage_path)
                else:
                    self.profiles = {}
                    logger.info("No profiles found. Initialized empty profiles store.")
            except json.JSONDecodeError as e:
                logger.error("Failed to parse profiles JSON: %s", str(e))
                self.profiles = {}
            except PermissionError as e:
                logger.error("Permission denied accessing %s: %s", self.storage_path, str(e))
                raise
            except Exception as e:
                logger.error("Unexpected error loading profiles: %s", str(e))
                raise

    def _save_profiles(self) -> None:
        """Save user profiles to storage."""
        with self.profile_lock:
            try:
                profile_path = Path(self.storage_path)
                backup_path = profile_path.with_suffix(".bak")
                if profile_path.exists():
                    profile_path.rename(backup_path)
                with profile_path.open("w") as f:
                    json.dump(self.profiles, f, indent=2)
                logger.info("User profiles saved to %s", self.storage_path)
            except PermissionError as e:
                logger.error("Permission denied saving %s: %s", self.storage_path, str(e))
                raise
            except Exception as e:
                logger.error("Unexpected error saving profiles: %s", str(e))
                raise

    async def switch_user(self, user_id: str, agent_id: str = "default") -> None:
        """Switch to a user and agent profile.

        Args:
            user_id: Unique identifier for the user.
            agent_id: Unique identifier for the agent.

        Raises:
            ValueError: If user_id or agent_id is invalid.
        """
        if not isinstance(user_id, str) or not user_id:
            logger.error("Invalid user_id: must be a non-empty string")
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(agent_id, str) or not agent_id:
            logger.error("Invalid agent_id: must be a non-empty string")
            raise ValueError("agent_id must be a non-empty string")
        
        try:
            with self.profile_lock:
                if user_id not in self.profiles:
                    logger.info("Creating new profile for user '%s'", user_id)
                    self.profiles[user_id] = {}
                
                if agent_id not in self.profiles[user_id]:
                    self.profiles[user_id][agent_id] = {
                        "preferences": self.DEFAULT_PREFERENCES.copy(),
                        "audit_log": [],
                        "identity_drift": deque(maxlen=1000)
                    }
                    self._save_profiles()
                
                self.active_user = user_id
                self.active_agent = agent_id
                logger.info("Active profile: %s::%s", user_id, agent_id)
                
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        event="User Switched",
                        meta={"user_id": user_id, "agent_id": agent_id},
                        module="UserProfile",
                        tags=["user_switch"]
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"User_Switch_{datetime.now().isoformat()}",
                        output={"user_id": user_id, "agent_id": agent_id},
                        layer="Profiles",
                        intent="user_switch"
                    )
        except Exception as e:
            logger.error("User switch failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.switch_user(user_id, agent_id),
                    default=None
                )
            raise

    async def get_preferences(self, fallback: bool = True) -> Dict[str, Any]:
        """Get preferences for the active user and agent.

        Args:
            fallback: If True, use default preferences for missing keys.

        Returns:
            Dictionary of preferences.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.warning("No active user. Returning default preferences.")
            return self.DEFAULT_PREFERENCES.copy()
        
        try:
            prefs = self.profiles[self.active_user][self.active_agent]["preferences"].copy()
            if fallback:
                for key, value in self.DEFAULT_PREFERENCES.items():
                    prefs.setdefault(key, value)
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Retrieved",
                    meta=prefs,
                    module="UserProfile",
                    tags=["preferences"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Retrieval_{datetime.now().isoformat()}",
                    output=prefs,
                    layer="Preferences",
                    intent="preference_retrieval"
                )
            
            return prefs
        except Exception as e:
            logger.error("Preference retrieval failed: %s", str(e))
            raise

    @lru_cache(maxsize=100)
    def get_epsilon_identity(self, timestamp: float) -> float:
        """Get ε-identity value for a given timestamp."""
        try:
            epsilon = epsilon_identity(time=timestamp)
            if not isinstance(epsilon, (int, float)):
                logger.error("Invalid epsilon_identity output: must be a number")
                raise ValueError("epsilon_identity must return a number")
            return epsilon
        except Exception as e:
            logger.error("epsilon_identity computation failed: %s", str(e))
            raise

    async def modulate_preferences(self, prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ε-modulation to preferences.

        Args:
            prefs: Dictionary of preferences to modulate.

        Returns:
            Modulated preferences with ε values.
        """
        if not isinstance(prefs, dict):
            logger.error("Invalid prefs: must be a dictionary")
            raise TypeError("prefs must be a dictionary")
        
        try:
            epsilon = self.get_epsilon_identity(datetime.now().timestamp())
            modulated = {k: f"{v} (ε={epsilon:.2f})" if isinstance(v, str) else v for k, v in prefs.items()}
            await self._track_drift(epsilon)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Modulated",
                    meta=modulated,
                    module="UserProfile",
                    tags=["preferences", "modulation"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Modulation_{datetime.now().isoformat()}",
                    output=modulated,
                    layer="Preferences",
                    intent="preference_modulation"
                )
            return modulated
        except Exception as e:
            logger.error("Preference modulation failed: %s", str(e))
            raise

    async def _track_drift(self, epsilon: float) -> None:
        """Track identity drift with ε value."""
        if not self.active_user:
            logger.error("No active user for drift tracking")
            raise ValueError("No active user. Call switch_user() first.")
        
        try:
            entry = {"timestamp": datetime.now().isoformat(), "epsilon": epsilon}
            profile = self.profiles[self.active_user][self.active_agent]
            if "identity_drift" not in profile:
                profile["identity_drift"] = deque(maxlen=1000)
            profile["identity_drift"].append(entry)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Track_{datetime.now().isoformat()}",
                    output=entry,
                    layer="Identity",
                    intent="drift_tracking"
                )
            self._save_profiles()
        except Exception as e:
            logger.error("Drift tracking failed: %s", str(e))
            raise

    async def update_preferences(self, new_prefs: Dict[str, Any]) -> None:
        """Update preferences for the active user and agent.

        Args:
            new_prefs: Dictionary of new preference key-value pairs.

        Raises:
            ValueError: If no active user or invalid preferences.
        """
        if not self.active_user:
            logger.error("No active user for preference update")
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(new_prefs, dict):
            logger.error("Invalid new_prefs: must be a dictionary")
            raise TypeError("new_prefs must be a dictionary")
        
        valid_keys = set(self.DEFAULT_PREFERENCES.keys())
        invalid_keys = set(new_prefs.keys()) - valid_keys
        if invalid_keys:
            logger.warning("Invalid preference keys: %s", invalid_keys)
            new_prefs = {k: v for k, v in new_prefs.items() if k in valid_keys}
        
        try:
            timestamp = datetime.now().isoformat()
            profile = self.profiles[self.active_user][self.active_agent]
            old_prefs = profile["preferences"]
            changes = {k: (old_prefs.get(k), v) for k, v in new_prefs.items()}
            
            contradictions = [k for k, (old, new) in changes.items() if old != new]
            if contradictions:
                logger.warning("Contradiction detected in preferences: %s", contradictions)
                if self.agi_enhancer:
                    await self.agi_enhancer.reflect_and_adapt(f"Preference contradictions: {contradictions}")
            
            profile["preferences"].update(new_prefs)
            profile["audit_log"].append({"timestamp": timestamp, "changes": changes})
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preference Update",
                    meta=changes,
                    module="UserProfile",
                    tags=["preferences"]
                )
                audit = await self.agi_enhancer.ethics_audit(str(changes), context="preference update")
                await self.agi_enhancer.log_explanation(
                    explanation=f"Preferences updated: {changes}",
                    trace={"ethics": audit}
                )
            
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Update_{timestamp}",
                    output={"user_id": self.active_user, "agent_id": self.active_agent, "changes": changes},
                    layer="Preferences",
                    intent="preference_update"
                )
            
            self._save_profiles()
            logger.info("Preferences updated for %s::%s", self.active_user, self.active_agent)
        except Exception as e:
            logger.error("Preference update failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.update_preferences(new_prefs),
                    default=None
                )
            raise

    async def reset_preferences(self) -> None:
        """Reset preferences to defaults for the active user and agent.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for preference reset")
            raise ValueError("No active user. Call switch_user() first.")
        
        try:
            timestamp = datetime.now().isoformat()
            self.profiles[self.active_user][self.active_agent]["preferences"] = self.DEFAULT_PREFERENCES.copy()
            self.profiles[self.active_user][self.active_agent]["audit_log"].append({
                "timestamp": timestamp,
                "changes": "Preferences reset to defaults."
            })
            
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Reset_{timestamp}",
                    output={"user_id": self.active_user, "agent_id": self.active_agent},
                    layer="Preferences",
                    intent="preference_reset"
                )
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Reset Preferences",
                    meta={"user_id": self.active_user, "agent_id": self.active_agent},
                    module="UserProfile",
                    tags=["reset"]
                )
            
            self._save_profiles()
            logger.info("Preferences reset for %s::%s", self.active_user, self.active_agent)
        except Exception as e:
            logger.error("Preference reset failed: %s", str(e))
            raise

    async def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log for the active user and agent.

        Returns:
            List of audit log entries.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for audit log retrieval")
            raise ValueError("No active user. Call switch_user() first.")
        
        try:
            audit_log = self.profiles[self.active_user][self.active_agent]["audit_log"]
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Audit Log Retrieved",
                    meta={"user_id": self.active_user, "agent_id": self.active_agent, "log_size": len(audit_log)},
                    module="UserProfile",
                    tags=["audit"]
                )
            return audit_log
        except Exception as e:
            logger.error("Audit log retrieval failed: %s", str(e))
            raise

    async def compute_profile_stability(self) -> float:
        """Compute Profile Stability Index (PSI) based on identity drift.

        Returns:
            PSI value between 0.0 and 1.0.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for stability computation")
            raise ValueError("No active user. Call switch_user() first.")
        
        try:
            drift = self.profiles[self.active_user][self.active_agent].get("identity_drift", [])
            if len(drift) < 2:
                logger.info("Insufficient drift data for PSI computation")
                return 1.0
            
            deltas = [abs(drift[i]["epsilon"] - drift[i-1]["epsilon"]) for i in range(1, len(drift))]
            avg_delta = sum(deltas) / len(deltas)
            psi = max(0.0, 1.0 - avg_delta)
            
            if self.toca_engine:
                traits = self.toca_engine.evolve(
                    x=np.array([0.1]), t=np.array([0.1]), additional_params={"psi": psi}
                )[0]
                psi = psi * (1 + 0.1 * np.mean(traits))
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Profile Stability Computed",
                    meta={"psi": psi, "deltas": deltas},
                    module="UserProfile",
                    tags=["stability", "psi"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"PSI_Computation_{datetime.now().isoformat()}",
                    output={"psi": psi, "user_id": self.active_user, "agent_id": self.active_agent},
                    layer="Identity",
                    intent="psi_computation"
                )
            
            logger.info("PSI for %s::%s = %.3f", self.active_user, self.active_agent, psi)
            return psi
        except Exception as e:
            logger.error("PSI computation failed: %s", str(e))
            raise

    async def reinforce_identity_thread(self) -> Dict[str, Any]:
        """Reinforce identity persistence across simulations.

        Returns:
            Dictionary with status of identity reinforcement.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for identity reinforcement")
            raise ValueError("No active user. Call switch_user() first.")
        
        try:
            epsilon = self.get_epsilon_identity(datetime.now().timestamp())
            await self._track_drift(epsilon)
            status = {"status": "thread-reinforced", "epsilon": epsilon}
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Identity Thread Reinforcement",
                    meta=status,
                    module="UserProfile",
                    tags=["identity", "reinforcement"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Identity_Reinforcement_{datetime.now().isoformat()}",
                    output=status,
                    layer="Identity",
                    intent="identity_reinforcement"
                )
            logger.info("Identity thread reinforced for %s::%s", self.active_user, self.active_agent)
            return status
        except Exception as e:
            logger.error("Identity reinforcement failed: %s", str(e))
            raise

    async def harmonize(self) -> List[Any]:
        """Unify preferences across agents for the active user.

        Returns:
            List of unique preference values.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for harmonization")
            raise ValueError("No active user. Call switch_user() first.")
        
        try:
            prefs = []
            for agent_id in self.profiles.get(self.active_user, {}):
                agent_prefs = self.profiles[self.active_user][agent_id].get("preferences", {})
                prefs.extend(agent_prefs.values())
            harmonized = list(set(prefs))
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Harmonized",
                    meta={"harmonized": harmonized},
                    module="UserProfile",
                    tags=["harmonization"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Harmonization_{datetime.now().isoformat()}",
                    output={"user_id": self.active_user, "harmonized": harmonized},
                    layer="Preferences",
                    intent="harmonization"
                )
            return harmonized
        except Exception as e:
            logger.error("Harmonization failed: %s", str(e))
            raise

if __name__ == "__main__":
    async def main():
        orchestrator = SimulationCore()
        user_profile = UserProfile(orchestrator=orchestrator)
        await user_profile.switch_user("user1", "agent1")
        await user_profile.update_preferences({"style": "verbose", "language": "fr"})
        prefs = await user_profile.get_preferences()
        print(f"Preferences: {prefs}")
        psi = await user_profile.compute_profile_stability()
        print(f"PSI: {psi}")
        await user_profile.reinforce_identity_thread()
        harmonized = await user_profile.harmonize()
        print(f"Harmonized: {harmonized}")

    import asyncio
    asyncio.run(main())
