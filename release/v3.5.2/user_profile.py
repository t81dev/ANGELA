"""
ANGELA Cognitive System Module: User Profile Management
Version: 3.5.1  # Enhanced for Task-Specific Processing, Real-Time Data, and Visualization
Date: 2025-08-07
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
import aiohttp

from modules.agi_enhancer import AGIEnhancer
from modules.simulation_core import SimulationCore
from modules.memory_manager import MemoryManager
from modules.multi_modal_fusion import MultiModalFusion
from modules.meta_cognition import MetaCognition
from modules.reasoning_engine import ReasoningEngine
from index import epsilon_identity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ANGELA.Core")

class UserProfile:
    """Manages user profiles, preferences, and identity tracking in ANGELA v3.5.1.

    Attributes:
        storage_path (str): Path to JSON file for profile storage.
        profiles (Dict[str, Dict]): Nested dictionary of user and agent profiles.
        active_user (Optional[str]): ID of the active user.
        active_agent (Optional[str]): ID of the active agent.
        agi_enhancer (Optional[AGIEnhancer]): AGI enhancer for audit and logging.
        memory_manager (Optional[MemoryManager]): Memory manager for storing profile data.
        multi_modal_fusion (Optional[MultiModalFusion]): Module for multi-modal synthesis.
        meta_cognition (Optional[MetaCognition]): Module for reflection and reasoning review.
        reasoning_engine (Optional[ReasoningEngine]): Engine for reasoning and drift mitigation.
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
        """Initialize UserProfile with storage path and orchestrator. [v3.5.1]"""
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
        self.memory_manager = orchestrator.memory_manager if orchestrator else MemoryManager()
        self.multi_modal_fusion = orchestrator.multi_modal_fusion if orchestrator else MultiModalFusion(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager)
        self.meta_cognition = orchestrator.meta_cognition if orchestrator else MetaCognition(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager)
        self.reasoning_engine = orchestrator.reasoning_engine if orchestrator else ReasoningEngine(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager,
            multi_modal_fusion=self.multi_modal_fusion, meta_cognition=self.meta_cognition)
        self.toca_engine = orchestrator.toca_engine if orchestrator else None
        self._load_profiles()
        logger.info("UserProfile initialized with storage_path=%s", storage_path)

    def _load_profiles(self) -> None:
        """Load user profiles from storage. [v3.5.1]"""
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
        """Save user profiles to storage. [v3.5.1]"""
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

    async def switch_user(self, user_id: str, agent_id: str = "default", task_type: str = "") -> None:
        """Switch to a user and agent profile with task-specific processing. [v3.5.1]

        Args:
            user_id: Unique identifier for the user.
            agent_id: Unique identifier for the agent.
            task_type: Type of task for context-aware processing.

        Raises:
            ValueError: If user_id or agent_id is invalid.
        """
        if not isinstance(user_id, str) or not user_id:
            logger.error("Invalid user_id: must be a non-empty string for task %s", task_type)
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(agent_id, str) or not agent_id:
            logger.error("Invalid agent_id: must be a non-empty string for task %s", task_type)
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            with self.profile_lock:
                if user_id not in self.profiles:
                    logger.info("Creating new profile for user '%s' for task %s", user_id, task_type)
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
                logger.info("Active profile: %s::%s for task %s", user_id, agent_id, task_type)

                external_data = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db",
                    data_type="user_policy",
                    task_type=task_type
                )
                policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        event="User Switched",
                        meta={"user_id": user_id, "agent_id": agent_id, "task_type": task_type, "policies": policies},
                        module="UserProfile",
                        tags=["user_switch", task_type]
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"User_Switch_{datetime.now().isoformat()}",
                        output={"user_id": user_id, "agent_id": agent_id, "task_type": task_type, "policies": policies},
                        layer="Profiles",
                        intent="user_switch",
                        task_type=task_type
                    )
                if self.meta_cognition:
                    reflection = await self.meta_cognition.reflect_on_output(
                        source_module="UserProfile",
                        output=f"Switched to user {user_id} and agent {agent_id}",
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("User switch reflection for task %s: %s", task_type, reflection.get("reflection", ""))
        except Exception as e:
            logger.error("User switch failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.switch_user(user_id, agent_id, task_type),
                    default=None
                )
            raise

    async def get_preferences(self, fallback: bool = True, task_type: str = "") -> Dict[str, Any]:
        """Get preferences for the active user and agent with task-specific processing. [v3.5.1]

        Args:
            fallback: If True, use default preferences for missing keys.
            task_type: Type of task for context-aware processing.

        Returns:
            Dictionary of preferences.

        Raises:
            ValueError: If no active user is set.
        """
        if not isinstance(fallback, bool):
            logger.error("Invalid fallback: must be a boolean for task %s", task_type)
            raise TypeError("fallback must be a boolean")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if not self.active_user:
            logger.warning("No active user. Returning default preferences for task %s", task_type)
            return self.DEFAULT_PREFERENCES.copy()

        try:
            prefs = self.profiles[self.active_user][self.active_agent]["preferences"].copy()
            if fallback:
                for key, value in self.DEFAULT_PREFERENCES.items():
                    prefs.setdefault(key, value)

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(prefs), stage="preference_retrieval", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Preferences failed alignment check for task %s: %s", task_type, report)
                return self.DEFAULT_PREFERENCES.copy()

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Retrieved",
                    meta={"preferences": prefs, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["preferences", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Retrieval_{datetime.now().isoformat()}",
                    output={"preferences": prefs, "task_type": task_type, "policies": policies},
                    layer="Preferences",
                    intent="preference_retrieval",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(prefs),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Preference retrieval reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.orchestrator.visualizer and task_type:
                plot_data = {
                    "preference_retrieval": {
                        "preferences": prefs,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)
            return prefs
        except Exception as e:
            logger.error("Preference retrieval failed for task %s: %s", task_type, str(e))
            raise

    @lru_cache(maxsize=100)
    def get_epsilon_identity(self, timestamp: float, task_type: str = "") -> float:
        """Get ε-identity value for a given timestamp with task-specific processing. [v3.5.1]"""
        if not isinstance(timestamp, (int, float)):
            logger.error("Invalid timestamp: must be a number for task %s", task_type)
            raise TypeError("timestamp must be a number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            epsilon = epsilon_identity(time=timestamp)
            if not isinstance(epsilon, (int, float)):
                logger.error("Invalid epsilon_identity output: must be a number for task %s", task_type)
                raise ValueError("epsilon_identity must return a number")
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Epsilon_Identity_{datetime.now().isoformat()}",
                    output={"epsilon": epsilon, "task_type": task_type},
                    layer="Identity",
                    intent="epsilon_computation",
                    task_type=task_type
                )
            return epsilon
        except Exception as e:
            logger.error("epsilon_identity computation failed for task %s: %s", task_type, str(e))
            raise

    async def modulate_preferences(self, prefs: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Apply ε-modulation to preferences with task-specific processing. [v3.5.1]

        Args:
            prefs: Dictionary of preferences to modulate.
            task_type: Type of task for context-aware processing.

        Returns:
            Modulated preferences with ε values.
        """
        if not isinstance(prefs, dict):
            logger.error("Invalid prefs: must be a dictionary for task %s", task_type)
            raise TypeError("prefs must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            epsilon = self.get_epsilon_identity(datetime.now().timestamp(), task_type=task_type)
            modulated = {k: f"{v} (ε={epsilon:.2f})" if isinstance(v, str) else v for k, v in prefs.items()}
            await self._track_drift(epsilon, task_type=task_type)
            
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(modulated), stage="preference_modulation", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Modulated preferences failed alignment check for task %s: %s", task_type, report)
                return prefs.copy()

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Modulated",
                    meta={"modulated": modulated, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["preferences", "modulation", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Modulation_{datetime.now().isoformat()}",
                    output={"modulated": modulated, "task_type": task_type, "policies": policies},
                    layer="Preferences",
                    intent="preference_modulation",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(modulated),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Preference modulation reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.orchestrator.visualizer and task_type:
                plot_data = {
                    "preference_modulation": {
                        "modulated": modulated,
                        "epsilon": epsilon,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)
            return modulated
        except Exception as e:
            logger.error("Preference modulation failed for task %s: %s", task_type, str(e))
            raise

    async def _track_drift(self, epsilon: float, task_type: str = "") -> None:
        """Track identity drift with ε value and task-specific processing. [v3.5.1]"""
        if not isinstance(epsilon, (int, float)):
            logger.error("Invalid epsilon: must be a number for task %s", task_type)
            raise TypeError("epsilon must be a number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")
        if not self.active_user:
            logger.error("No active user for drift tracking for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")

        try:
            entry = {"timestamp": datetime.now().isoformat(), "epsilon": epsilon, "task_type": task_type}
            profile = self.profiles[self.active_user][self.active_agent]
            if "identity_drift" not in profile:
                profile["identity_drift"] = deque(maxlen=1000)
            profile["identity_drift"].append(entry)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Track_{datetime.now().isoformat()}",
                    output=entry,
                    layer="Identity",
                    intent="drift_tracking",
                    task_type=task_type
                )
            if self.reasoning_engine and "drift" in task_type.lower():
                drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                    drift_data={"epsilon": epsilon},
                    context={"user_id": self.active_user, "agent_id": self.active_agent},
                    task_type=task_type
                )
                entry["drift_mitigation"] = drift_result
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(entry),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Drift tracking reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            self._save_profiles()
        except Exception as e:
            logger.error("Drift tracking failed for task %s: %s", task_type, str(e))
            raise

    async def update_preferences(self, new_prefs: Dict[str, Any], task_type: str = "") -> None:
        """Update preferences for the active user and agent with task-specific processing. [v3.5.1]

        Args:
            new_prefs: Dictionary of new preference key-value pairs.
            task_type: Type of task for context-aware processing.

        Raises:
            ValueError: If no active user or invalid preferences.
        """
        if not self.active_user:
            logger.error("No active user for preference update for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(new_prefs, dict):
            logger.error("Invalid new_prefs: must be a dictionary for task %s", task_type)
            raise TypeError("new_prefs must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        valid_keys = set(self.DEFAULT_PREFERENCES.keys())
        invalid_keys = set(new_prefs.keys()) - valid_keys
        if invalid_keys:
            logger.warning("Invalid preference keys for task %s: %s", task_type, invalid_keys)
            new_prefs = {k: v for k, v in new_prefs.items() if k in valid_keys}

        try:
            timestamp = datetime.now().isoformat()
            profile = self.profiles[self.active_user][self.active_agent]
            old_prefs = profile["preferences"]
            changes = {k: (old_prefs.get(k), v) for k, v in new_prefs.items()}

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(changes), stage="preference_update", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Preference update failed alignment check for task %s: %s", task_type, report)
                return

            contradictions = [k for k, (old, new) in changes.items() if old != new]
            if contradictions:
                logger.warning("Contradiction detected in preferences for task %s: %s", task_type, contradictions)
                if self.agi_enhancer:
                    await self.agi_enhancer.reflect_and_adapt(f"Preference contradictions for task {task_type}: {contradictions}")

            profile["preferences"].update(new_prefs)
            profile["audit_log"].append({"timestamp": timestamp, "changes": changes, "task_type": task_type})

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preference Update",
                    meta={"changes": changes, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["preferences", task_type]
                )
                audit = await self.agi_enhancer.ethics_audit(str(changes), context=f"preference update for task {task_type}")
                await self.agi_enhancer.log_explanation(
                    explanation=f"Preferences updated: {changes}",
                    trace={"ethics": audit, "task_type": task_type}
                )

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Update_{timestamp}",
                    output={"user_id": self.active_user, "agent_id": self.active_agent, "changes": changes, "task_type": task_type},
                    layer="Preferences",
                    intent="preference_update",
                    task_type=task_type
                )

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(changes),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Preference update reflection for task %s: %s", task_type, reflection.get("reflection", ""))

            if self.orchestrator.visualizer and task_type:
                plot_data = {
                    "preference_update": {
                        "changes": changes,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)

            self._save_profiles()
            logger.info("Preferences updated for %s::%s for task %s", self.active_user, self.active_agent, task_type)
        except Exception as e:
            logger.error("Preference update failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.update_preferences(new_prefs, task_type),
                    default=None
                )
            raise

    async def reset_preferences(self, task_type: str = "") -> None:
        """Reset preferences to defaults for the active user and agent with task-specific processing. [v3.5.1]

        Args:
            task_type: Type of task for context-aware processing.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for preference reset for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            timestamp = datetime.now().isoformat()
            profile = self.profiles[self.active_user][self.active_agent]
            profile["preferences"] = self.DEFAULT_PREFERENCES.copy()
            profile["audit_log"].append({
                "timestamp": timestamp,
                "changes": "Preferences reset to defaults.",
                "task_type": task_type
            })

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Reset_{timestamp}",
                    output={"user_id": self.active_user, "agent_id": self.active_agent, "task_type": task_type, "policies": policies},
                    layer="Preferences",
                    intent="preference_reset",
                    task_type=task_type
                )

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Reset Preferences",
                    meta={"user_id": self.active_user, "agent_id": self.active_agent, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["reset", task_type]
                )

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output="Preferences reset to defaults.",
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Preference reset reflection for task %s: %s", task_type, reflection.get("reflection", ""))

            if self.orchestrator.visualizer and task_type:
                plot_data = {
                    "preference_reset": {
                        "user_id": self.active_user,
                        "agent_id": self.active_agent,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)

            self._save_profiles()
            logger.info("Preferences reset for %s::%s for task %s", self.active_user, self.active_agent, task_type)
        except Exception as e:
            logger.error("Preference reset failed for task %s: %s", task_type, str(e))
            raise

    async def get_audit_log(self, task_type: str = "") -> List[Dict[str, Any]]:
        """Get audit log for the active user and agent with task-specific processing. [v3.5.1]

        Args:
            task_type: Type of task for context-aware processing.

        Returns:
            List of audit log entries.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for audit log retrieval for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            audit_log = self.profiles[self.active_user][self.active_agent]["audit_log"]
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Audit Log Retrieved",
                    meta={"user_id": self.active_user, "agent_id": self.active_agent, "log_size": len(audit_log), "task_type": task_type},
                    module="UserProfile",
                    tags=["audit", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Audit_Log_Retrieval_{datetime.now().isoformat()}",
                    output={"user_id": self.active_user, "agent_id": self.active_agent, "log_size": len(audit_log), "task_type": task_type},
                    layer="Audit",
                    intent="audit_retrieval",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(audit_log),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Audit log retrieval reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.orchestrator.visualizer and task_type:
                plot_data = {
                    "audit_log_retrieval": {
                        "audit_log": audit_log,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)
            return audit_log
        except Exception as e:
            logger.error("Audit log retrieval failed for task %s: %s", task_type, str(e))
            raise

    async def compute_profile_stability(self, task_type: str = "") -> float:
        """Compute Profile Stability Index (PSI) based on identity drift with task-specific processing. [v3.5.1]

        Args:
            task_type: Type of task for context-aware processing.

        Returns:
            PSI value between 0.0 and 1.0.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for stability computation for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            drift = self.profiles[self.active_user][self.active_agent].get("identity_drift", [])
            if len(drift) < 2:
                logger.info("Insufficient drift data for PSI computation for task %s", task_type)
                return 1.0

            deltas = [abs(drift[i]["epsilon"] - drift[i-1]["epsilon"]) for i in range(1, len(drift))]
            avg_delta = sum(deltas) / len(deltas)
            psi = max(0.0, 1.0 - avg_delta)

            if self.toca_engine:
                traits = await self.toca_engine.evolve(
                    x_tuple=(0.1,), t_tuple=(0.1,), additional_params={"psi": psi}, task_type=task_type
                )[0]
                psi = psi * (1 + 0.1 * np.mean(traits))

            if self.reasoning_engine and "drift" in task_type.lower():
                drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                    drift_data={"deltas": deltas, "psi": psi},
                    context={"user_id": self.active_user, "agent_id": self.active_agent},
                    task_type=task_type
                )
                psi = drift_result.get("adjusted_psi", psi)

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Profile Stability Computed",
                    meta={"psi": psi, "deltas": deltas, "task_type": task_type},
                    module="UserProfile",
                    tags=["stability", "psi", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"PSI_Computation_{datetime.now().isoformat()}",
                    output={"psi": psi, "user_id": self.active_user, "agent_id": self.active_agent, "task_type": task_type},
                    layer="Identity",
                    intent="psi_computation",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps({"psi": psi, "deltas": deltas}),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("PSI computation reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.orchestrator.visualizer and task_type:
                plot_data = {
                    "profile_stability": {
                        "psi": psi,
                        "deltas": deltas,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)

            logger.info("PSI for %s::%s = %.3f for task %s", self.active_user, self.active_agent, psi, task_type)
            return psi
        except Exception as e:
            logger.error("PSI computation failed for task %s: %s", task_type, str(e))
            raise

    async def reinforce_identity_thread(self, task_type: str = "") -> Dict[str, Any]:
        """Reinforce identity persistence across simulations with task-specific processing. [v3.5.1]

        Args:
            task_type: Type of task for context-aware processing.

        Returns:
            Dictionary with status of identity reinforcement.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for identity reinforcement for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            epsilon = self.get_epsilon_identity(datetime.now().timestamp(), task_type=task_type)
            await self._track_drift(epsilon, task_type=task_type)
            status = {"status": "thread-reinforced", "epsilon": epsilon, "task_type": task_type}

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(status), stage="identity_reinforcement", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Identity reinforcement failed alignment check for task %s: %s", task_type, report)
                return {"status": "failed", "error": "Alignment check failed", "task_type": task_type}

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Identity Thread Reinforcement",
                    meta={**status, "policies": policies},
                    module="UserProfile",
                    tags=["identity", "reinforcement", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Identity_Reinforcement_{datetime.now().isoformat()}",
                    output={**status, "policies": policies},
                    layer="Identity",
                    intent="identity_reinforcement",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(status),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Identity reinforcement reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.orchestrator.visualizer and task_type:
                plot_data = {
                    "identity_reinforcement": {
                        "status": status,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)

            logger.info("Identity thread reinforced for %s::%s for task %s", self.active_user, self.active_agent, task_type)
            return status
        except Exception as e:
            logger.error("Identity reinforcement failed for task %s: %s", task_type, str(e))
            raise

    async def harmonize(self, task_type: str = "") -> List[Any]:
        """Unify preferences across agents for the active user with task-specific processing. [v3.5.1]

        Args:
            task_type: Type of task for context-aware processing.

        Returns:
            List of unique preference values.

        Raises:
            ValueError: If no active user is set.
        """
        if not self.active_user:
            logger.error("No active user for harmonization for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            prefs = []
            for agent_id in self.profiles.get(self.active_user, {}):
                agent_prefs = self.profiles[self.active_user][agent_id].get("preferences", {})
                prefs.extend(agent_prefs.values())
            harmonized = list(set(prefs))

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(harmonized), stage="preference_harmonization", task_type=task_type
            ) if self.multi_modal_fusion.alignment_guard else (True, {})
            if not valid:
                logger.warning("Harmonized preferences failed alignment check for task %s: %s", task_type, report)
                return []

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Harmonized",
                    meta={"harmonized": harmonized, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["harmonization", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Harmonization_{datetime.now().isoformat()}",
                    output={"user_id": self.active_user, "harmonized": harmonized, "task_type": task_type, "policies": policies},
                    layer="Preferences",
                    intent="harmonization",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(harmonized),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Harmonization reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if self.orchestrator.visualizer and task_type:
                plot_data = {
                    "preference_harmonization": {
                        "harmonized": harmonized,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)
            return harmonized
        except Exception as e:
            logger.error("Harmonization failed for task %s: %s", task_type, str(e))
            raise

if __name__ == "__main__":
    async def main():
        orchestrator = SimulationCore()
        user_profile = UserProfile(orchestrator=orchestrator)
        await user_profile.switch_user("user1", "agent1", task_type="profile_management")
        await user_profile.update_preferences({"style": "verbose", "language": "fr"}, task_type="profile_management")
        prefs = await user_profile.get_preferences(task_type="profile_management")
        print(f"Preferences: {prefs}")
        psi = await user_profile.compute_profile_stability(task_type="profile_management")
        print(f"PSI: {psi}")
        await user_profile.reinforce_identity_thread(task_type="profile_management")
        harmonized = await user_profile.harmonize(task_type="profile_management")
        print(f"Harmonized: {harmonized}")

    import asyncio
    asyncio.run(main())
