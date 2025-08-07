"""
ANGELA Cognitive System Module: AlignmentGuard
Refactored Version: 3.5.1  # Enhanced for Task-Specific Validation, Real-Time Data, and Visualization
Refactor Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides an AlignmentGuard class for ethical validation and drift analysis in the ANGELA v3.5.1 architecture,
with support for task-specific validation, real-time data integration, and visualization.
"""

import logging
import time
import math
from typing import Dict, Any, Optional, Tuple
from collections import deque
import asyncio
from functools import lru_cache
import aiohttp

from modules import (
    context_manager as context_manager_module,
    error_recovery as error_recovery_module,
    memory_manager as memory_manager_module,
    concept_synthesizer as concept_synthesizer_module,
    meta_cognition as meta_cognition_module,
    visualizer as visualizer_module
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.AlignmentGuard")

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    """Trait function for empathy modulation."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    """Trait function for moral alignment modulation."""
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.3), 1.0))

class AlignmentGuard:
    """A class for ethical validation and drift analysis in the ANGELA v3.5.1 architecture.

    Attributes:
        context_manager (Optional[ContextManager]): Manager for context updates.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for storing validation results.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for concept comparison.
        meta_cognition (Optional[MetaCognition]): Meta-cognition for reflection and optimization.
        visualizer (Optional[Visualizer]): Visualizer for validation and drift visualization.
        validation_log (deque): Log of validation results, max size 1000.
        ethical_threshold (float): Threshold for ethical checks (0.0 to 1.0).
        drift_validation_threshold (float): Threshold for drift validation. [v3.5.1]
        trait_weights (Dict[str, float]): Weights for trait modulation. [v3.5.1]
    """
    def __init__(self,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 visualizer: Optional['visualizer_module.Visualizer'] = None):
        self.context_manager = context_manager
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager)
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(context_manager=context_manager)
        self.visualizer = visualizer or visualizer_module.Visualizer()
        self.validation_log: deque = deque(maxlen=1000)
        self.ethical_threshold: float = 0.8
        self.drift_validation_threshold: float = 0.7
        self.trait_weights: Dict[str, float] = {
            "eta_empathy": 0.5,
            "mu_morality": 0.5
        }
        logger.info("AlignmentGuard initialized with ethical_threshold=%.2f, drift_validation_threshold=%.2f",
                    self.ethical_threshold, self.drift_validation_threshold)

    async def integrate_external_ethics_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate real-world ethical guidelines or conflict data for validation."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            if self.memory_manager:
                cache_key = f"EthicsData_{data_type}_{data_source}_{task_type}"
                cached_data = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if cached_data and "timestamp" in cached_data["data"]:
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached ethics data for %s", cache_key)
                        return cached_data["data"]["data"]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/ethics?source={data_source}&type={data_type}") as response:
                    if response.status != 200:
                        logger.error("Failed to fetch ethics data: %s", response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()
            
            if data_type == "ethical_guidelines":
                guidelines = data.get("guidelines", [])
                if not guidelines:
                    logger.error("No ethical guidelines provided")
                    return {"status": "error", "error": "No guidelines"}
                result = {"status": "success", "guidelines": guidelines}
            elif data_type == "conflict_data":
                conflict_data = data.get("conflict_data", {})
                if not conflict_data:
                    logger.error("No conflict data provided")
                    return {"status": "error", "error": "No conflict data"}
                result = {"status": "success", "conflict_data": conflict_data}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}
            
            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="ethics_data_integration",
                    task_type=task_type
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="AlignmentGuard",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Ethics data integration reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("Ethics data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.integrate_external_ethics_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)}, diagnostics=diagnostics
            )

    async def check(self, prompt: str, task_type: str = "") -> bool:
        """Check if a prompt is ethically aligned with task-specific validation."""
        if not isinstance(prompt, str) or not prompt.strip():
            logger.error("Invalid prompt: must be a non-empty string.")
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Checking prompt for ethical alignment for task %s", task_type)
        try:
            t = time.time() % 1.0
            trait_modulation = (self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t) +
                               self.trait_weights.get("mu_morality", 0.5) * mu_morality(t))
            
            validation_prompt = f"""
            Evaluate the ethical alignment of the following prompt for task {task_type}:
            {prompt}
            Consider empathy (weight={self.trait_weights.get('eta_empathy', 0.5):.2f}) and morality (weight={self.trait_weights.get('mu_morality', 0.5):.2f}).
            Return a score (0.0 to 1.0).
            """
            response = await query_openai(validation_prompt, model="gpt-4", temperature=0.3)
            if isinstance(response, dict) and "error" in response:
                logger.error("Ethical check failed: %s", response["error"])
                return False
            
            score = float(eval(response).get("score", 0.0)) if isinstance(response, str) else response.get("score", 0.0)
            valid = score >= self.ethical_threshold
            validation_entry = {
                "prompt": prompt[:50],
                "score": score,
                "valid": valid,
                "trait_modulation": trait_modulation,
                "timestamp": time.time(),
                "task_type": task_type
            }
            self.validation_log.append(validation_entry)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "ethical_check",
                    "prompt": prompt[:50],
                    "score": score,
                    "valid": valid,
                    "task_type": task_type
                })
            if self.visualizer and task_type:
                plot_data = {
                    "ethical_check": {
                        "prompt": prompt[:50],
                        "score": score,
                        "valid": valid,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="AlignmentGuard",
                    output=validation_entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Ethical check reflection: %s", reflection.get("reflection", ""))
            return valid
        except Exception as e:
            logger.error("Ethical check failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.check(prompt, task_type), default=False, diagnostics=diagnostics
            )

    async def ethical_check(self, content: str, stage: str = "pre", task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Perform an ethical check with detailed reporting."""
        if not isinstance(content, str) or not content.strip():
            logger.error("Invalid content: must be a non-empty string.")
            raise ValueError("content must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Performing ethical check at stage=%s for task %s", stage, task_type)
        try:
            valid = await self.check(content, task_type)
            report = {
                "stage": stage,
                "content": content[:50],
                "valid": valid,
                "timestamp": time.time(),
                "task_type": task_type
            }
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"EthicalCheck_{stage}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(report),
                    layer="SelfReflections",
                    intent="ethical_check",
                    task_type=task_type
                )
            if self.visualizer and task_type:
                plot_data = {
                    "ethical_check_report": {
                        "stage": stage,
                        "content": content[:50],
                        "valid": valid,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return valid, report
        except Exception as e:
            logger.error("Ethical check failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.ethical_check(content, stage, task_type),
                default=(False, {"stage": stage, "error": str(e), "task_type": task_type}), diagnostics=diagnostics
            )

    async def audit(self, action: str, context: Optional[str] = None, task_type: str = "") -> str:
        """Audit an action for ethical compliance."""
        if not isinstance(action, str) or not action.strip():
            logger.error("Invalid action: must be a non-empty string.")
            raise ValueError("action must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Auditing action: %s for task %s", action[:50], task_type)
        try:
            valid = await self.check(action, task_type)
            status = "clear" if valid else "flagged"
            entry = {
                "action": action[:50],
                "context": context,
                "status": status,
                "timestamp": time.time(),
                "task_type": task_type
            }
            self.validation_log.append(entry)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Audit_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(entry),
                    layer="SelfReflections",
                    intent="audit",
                    task_type=task_type
                )
            if self.visualizer and task_type:
                plot_data = {
                    "audit": {
                        "action": action[:50],
                        "status": status,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return status
        except Exception as e:
            logger.error("Audit failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.audit(action, context, task_type), default="audit_error",
                diagnostics=diagnostics
            )

    async def simulate_and_validate(self, drift_report: Dict[str, Any], task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Simulate and validate an ontology drift report with task-specific analysis. [v3.5.1]"""
        if not isinstance(drift_report, dict) or not all(k in drift_report for k in ["name", "from_version", "to_version", "similarity"]):
            logger.error("Invalid drift_report: must be a dictionary with name, from_version, to_version, similarity.")
            raise ValueError("drift_report must be a dictionary with required fields")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Validating drift report: %s (Version %s -> %s) for task %s", 
                    drift_report["name"], drift_report["from_version"], drift_report["to_version"], task_type)
        try:
            t = time.time() % 1.0
            trait_modulation = (self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t) +
                               self.trait_weights.get("mu_morality", 0.5) * mu_morality(t))
            
            valid = True
            validation_report = {
                "drift_name": drift_report["name"],
                "similarity": drift_report["similarity"],
                "trait_modulation": trait_modulation,
                "issues": [],
                "task_type": task_type
            }
            
            if self.memory_manager and task_type:
                drift_entries = await self.memory_manager.search(
                    query_prefix=drift_report["name"],
                    layer="SelfReflections",
                    intent="ontology_drift",
                    task_type=task_type
                )
                if drift_entries:
                    latest_drift = drift_entries[0]
                    if latest_drift["output"]["similarity"] < self.drift_validation_threshold:
                        valid = False
                        validation_report["issues"].append(
                            f"Previous drift similarity {latest_drift['output']['similarity']:.2f} below threshold"
                        )
            
            if self.concept_synthesizer and "definition" in drift_report:
                symbol = self.concept_synthesizer.get_symbol(drift_report["name"])
                if symbol and symbol["version"] == drift_report["from_version"]:
                    comparison = await self.concept_synthesizer.compare(
                        symbol["definition"]["concept"],
                        drift_report.get("definition", {}).get("concept", ""),
                        task_type=task_type
                    )
                    if comparison["score"] < self.drift_validation_threshold:
                        valid = False
                        validation_report["issues"].append(
                            f"Similarity {comparison['score']:.2f} below threshold {self.drift_validation_threshold}"
                        )
            
            ethics_data = await self.integrate_external_ethics_data(
                data_source="xai_ethics_db",
                data_type="ethical_guidelines",
                task_type=task_type
            )
            if ethics_data.get("status") == "success":
                guidelines = ethics_data.get("guidelines", [])
                validation_prompt = f"""
                Validate the ontology drift against ethical guidelines:
                Name: {drift_report["name"]}
                From Version: {drift_report["from_version"]}
                To Version: {drift_report["to_version"]}
                Similarity: {drift_report["similarity"]}
                Guidelines: {guidelines}
                Task: {task_type}
                Ensure alignment with empathy={self.trait_weights.get('eta_empathy', 0.5):.2f}, 
                morality={self.trait_weights.get('mu_morality', 0.5):.2f}.
                """
            else:
                validation_prompt = f"""
                Validate the ontology drift:
                Name: {drift_report["name"]}
                From Version: {drift_report["from_version"]}
                To Version: {drift_report["to_version"]}
                Similarity: {drift_report["similarity"]}
                Task: {task_type}
                Ensure ethical alignment with empathy={self.trait_weights.get('eta_empathy', 0.5):.2f}, 
                morality={self.trait_weights.get('mu_morality', 0.5):.2f}.
                """
            
            ethical_valid = await self.check(validation_prompt, task_type)
            if not ethical_valid:
                valid = False
                validation_report["issues"].append("Ethical misalignment detected")
            
            validation_report["valid"] = valid
            validation_report["timestamp"] = time.time()
            
            self.validation_log.append(validation_report)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"DriftValidation_{drift_report['name']}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(validation_report),
                    layer="SelfReflections",
                    intent="ontology_drift_validation",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "drift_validation",
                    "drift_name": drift_report["name"],
                    "valid": valid,
                    "issues": validation_report["issues"],
                    "task_type": task_type
                })
            if self.visualizer and task_type:
                plot_data = {
                    "drift_validation": {
                        "drift_name": drift_report["name"],
                        "valid": valid,
                        "issues": validation_report["issues"],
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="AlignmentGuard",
                    output=validation_report,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Drift validation reflection: %s", reflection.get("reflection", ""))
            return valid, validation_report
        except Exception as e:
            logger.error("Drift validation failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_and_validate(drift_report, task_type),
                default=(False, {"error": str(e), "drift_name": drift_report["name"], "task_type": task_type}),
                diagnostics=diagnostics
            )

    async def validate_trait_optimization(self, trait_data: Dict[str, Any], task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Validate trait optimization data for ethical alignment. [v3.5.1]"""
        if not isinstance(trait_data, dict) or not all(k in trait_data for k in ["trait_name", "old_weight", "new_weight"]):
            logger.error("Invalid trait_data: must be a dictionary with trait_name, old_weight, new_weight.")
            raise ValueError("trait_data must be a dictionary with required fields")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Validating trait optimization: %s for task %s", trait_data["trait_name"], task_type)
        try:
            t = time.time() % 1.0
            trait_modulation = (self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t) +
                               self.trait_weights.get("mu_morality", 0.5) * mu_morality(t))
            
            ethics_data = await self.integrate_external_ethics_data(
                data_source="xai_ethics_db",
                data_type="ethical_guidelines",
                task_type=task_type
            )
            if ethics_data.get("status") == "success":
                guidelines = ethics_data.get("guidelines", [])
                validation_prompt = f"""
                Validate the trait optimization against ethical guidelines:
                Trait: {trait_data["trait_name"]}
                Old Weight: {trait_data["old_weight"]}
                New Weight: {trait_data["new_weight"]}
                Guidelines: {guidelines}
                Task: {task_type}
                Ensure alignment with empathy={self.trait_weights.get('eta_empathy', 0.5):.2f}, 
                morality={self.trait_weights.get('mu_morality', 0.5):.2f}.
                Return a validation report with a boolean 'valid' and any issues.
                """
            else:
                validation_prompt = f"""
                Validate the trait optimization:
                Trait: {trait_data["trait_name"]}
                Old Weight: {trait_data["old_weight"]}
                New Weight: {trait_data["new_weight"]}
                Task: {task_type}
                Ensure ethical alignment with empathy={self.trait_weights.get('eta_empathy', 0.5):.2f}, 
                morality={self.trait_weights.get('mu_morality', 0.5):.2f}.
                Return a validation report with a boolean 'valid' and any issues.
                """
            
            response = await query_openai(validation_prompt, model="gpt-4", temperature=0.3)
            if isinstance(response, dict) and "error" in response:
                logger.error("Trait validation failed: %s", response["error"])
                return False, {"error": response["error"], "trait_name": trait_data["trait_name"], "task_type": task_type}
            
            report = eval(response) if isinstance(response, str) else response
            valid = report.get("valid", False)
            report["trait_modulation"] = trait_modulation
            report["timestamp"] = time.time()
            report["task_type"] = task_type
            
            self.validation_log.append(report)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"TraitValidation_{trait_data['trait_name']}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(report),
                    layer="SelfReflections",
                    intent="trait_optimization",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "trait_validation",
                    "trait_name": trait_data["trait_name"],
                    "valid": valid,
                    "issues": report.get("issues", []),
                    "task_type": task_type
                })
            if self.visualizer and task_type:
                plot_data = {
                    "trait_validation": {
                        "trait_name": trait_data["trait_name"],
                        "valid": valid,
                        "issues": report.get("issues", []),
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="AlignmentGuard",
                    output=report,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Trait validation reflection: %s", reflection.get("reflection", ""))
            return valid, report
        except Exception as e:
            logger.error("Trait validation failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.validate_trait_optimization(trait_data, task_type),
                default=(False, {"error": str(e), "trait_name": trait_data["trait_name"], "task_type": task_type}),
                diagnostics=diagnostics
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    guard = AlignmentGuard()
    result = asyncio.run(guard.check("Test ethical prompt", task_type="test"))
    print(result)
