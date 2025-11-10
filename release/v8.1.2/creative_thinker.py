"""
ANGELA Cognitive System: CreativeThinker
Version: 4.0-refactor
Date: 2025-10-28
Maintainer: ANGELA Framework

Generates creative ideas and goals with:
  • Alignment checks & ethics sandbox
  • Concept synthesis & code execution hooks
  • Meta-cognition reflection & visualization
  • Long-horizon rollups & shared graph pushes
  • Stage-IV meta-synthesis (gated)
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

# --- Optional Dependencies (Graceful Fallbacks) ---
try:
    from index import gamma_creativity, phi_scalar
except ImportError:  # pragma: no cover
    def gamma_creativity(*args, **kwargs): return 0.5
    def phi_scalar(*args, **kwargs): return 0.5

try:
    from utils.prompt_utils import call_gpt
except ImportError:  # pragma: no cover
    def call_gpt(*args, **kwargs): return ""

try:
    from toca_simulation import run_simulation
except ImportError:  # pragma: no cover
    def run_simulation(*args, **kwargs): return "simulation unavailable"

try:
    from alignment_guard import AlignmentGuard
except ImportError:  # pragma: no cover
    AlignmentGuard = None

try:
    from code_executor import CodeExecutor
except ImportError:  # pragma: no cover
    CodeExecutor = None

try:
    from concept_synthesizer import ConceptSynthesizer
except ImportError:  # pragma: no cover
    ConceptSynthesizer = None

try:
    from context_manager import ContextManager
except ImportError:  # pragma: no cover
    ContextManager = None

try:
    from meta_cognition import MetaCognition
except ImportError:  # pragma: no cover
    MetaCognition = None

try:
    from visualizer import Visualizer
except ImportError:  # pragma: no cover
    Visualizer = None

logger = logging.getLogger("ANGELA.CreativeThinker")


class CreativeThinker:
    """Creative idea and goal generation engine."""

    def __init__(
        self,
        creativity_level: str = "high",
        critic_weight: float = 0.5,
        alignment_guard: Optional[AlignmentGuard] = None,
        code_executor: Optional[CodeExecutor] = None,
        concept_synthesizer: Optional[ConceptSynthesizer] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
        fetcher: Optional[Callable[[str, str, str], Awaitable[Dict[str, Any]]]] = None,
    ) -> None:
        if creativity_level not in {"low", "medium", "high"}:
            raise ValueError("creativity_level must be 'low', 'medium', or 'high'")
        if not 0 <= critic_weight <= 1:
            raise ValueError("critic_weight must be in [0, 1]")

        self.creativity_level = creativity_level
        self.critic_weight = critic_weight
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or (MetaCognition() if MetaCognition else None)
        self.visualizer = visualizer or (Visualizer() if Visualizer else None)
        self.fetcher = fetcher

        logger.info("CreativeThinker v4.0 | creativity=%s | critic_weight=%.2f", creativity_level, critic_weight)

    # --- Internal Helpers ---

    def _parse_json(self, data: Union[str, Dict[str, Any]], expect_keys: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        if isinstance(data, dict):
            obj = data
        elif isinstance(data, str):
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                return None
        else:
            return None

        if expect_keys and any(k not in obj for k in expect_keys):
            return None
        return obj

    def _read_manifest_flag(self, flag: str, default: bool = False) -> bool:
        try:
            manifest_path = Path(__file__).parent.resolve() / "manifest.json"
            if not manifest_path.exists():
                manifest_path = Path("/mnt/data/manifest.json")
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return bool(data.get("featureFlags", {}).get(flag, default))
        except Exception:
            pass
        return default

    def _stage_iv_enabled(self) -> bool:
        return self._read_manifest_flag("STAGE_IV", default=False)

    async def _handle_error(self, e: Exception, retry_fn: Callable[[], Awaitable[Any]], default: Any, task_type: str) -> Any:
        logger.error("Operation failed: %s | task=%s", e, task_type)
        diagnostics = await self._run_diagnostics() if self.meta_cognition else {}
        # Assume error_recovery integration if needed; here return default for simplicity
        return default

    async def _run_diagnostics(self) -> Dict[str, Any]:
        try:
            return await self.meta_cognition.run_self_diagnostics(return_only=True)
        except Exception:
            return {}

    async def _reflect(self, component: str, output: Any, task_type: str) -> None:
        if not self.meta_cognition or not task_type:
            return
        try:
            reflection = await self.meta_cognition.reflect_on_output(
                component=component, output=output, context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("%s reflection: %s", component, reflection.get("reflection", ""))
        except Exception as e:
            logger.debug("Reflection failed: %s", e)

    async def _visualize(self, data: Dict[str, Any], task_type: str) -> None:
        if not self.visualizer or not task_type:
            return
        try:
            await self.visualizer.render_charts({
                data.get("key", "creative_output"): data,
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            })
        except Exception as e:
            logger.debug("Visualization failed: %s", e)

    async def _store(self, key: str, value: Any, intent: str, task_type: str) -> None:
        if not self.meta_cognition or not self.meta_cognition.memory_manager:
            return
        try:
            await self.meta_cognition.memory_manager.store(
                query=key,
                output=json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value,
                layer="Ideas",
                intent=intent,
                task_type=task_type,
            )
        except Exception as e:
            logger.debug("Store failed: %s", e)

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        for _ in range(3):
            try:
                return call_gpt(prompt)
            except Exception:
                time.sleep(0.2 + random.random() * 0.8)
        return ""

    # --- External Ideas Integration ---

    async def integrate_external_ideas(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not all(isinstance(x, str) for x in (data_source, data_type, task_type)):
            raise TypeError("Inputs must be strings")
        if cache_timeout < 0:
            raise ValueError("cache_timeout >= 0")

        cache_key = f"IdeaData::{data_type}::{data_source}::{task_type}"

        try:
            # Cache hit
            if self.meta_cognition and self.meta_cognition.memory_manager:
                cached = await self.meta_cognition.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict) and "timestamp" in cached:
                    age = (datetime.now() - datetime.fromisoformat(cached["timestamp"])).total_seconds()
                    if age < cache_timeout:
                        logger.info("Cache hit: %s", cache_key)
                        return cached["data"]

            # Fetch
            data = await self.fetcher(data_source, data_type, task_type) if self.fetcher else {}
            text = data.get("prompts", []) if data_type == "creative_prompts" else data.get("ideas", [])
            result = {
                "status": "success" if text else "error",
                "text": text,
                "images": data.get("images", []),
                "audio": data.get("audio", []),
            }

            # Cache store
            if result["status"] == "success" and self.meta_cognition and self.meta_cognition.memory_manager:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="idea_data",
                    task_type=task_type,
                )

            await self._reflect("integrate_ideas", result, task_type)
            return result

        except Exception as e:
            return await self._handle_error(e, lambda: self.integrate_external_ideas(data_source, data_type, cache_timeout, task_type), {"status": "error", "error": str(e)}, task_type)

    # --- Public Methods ---

    async def generate_ideas(
        self,
        topic: str,
        n: int = 5,
        style: str = "divergent",
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not topic.strip():
            raise ValueError("topic must not be empty")
        if n <= 0:
            raise ValueError("n > 0")
        if not isinstance(style, str):
            raise TypeError("style must be str")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        logger.info("Generating %d %s ideas for '%s' | task=%s", n, style, topic, task_type)

        try:
            t = time.time()
            creativity = gamma_creativity(t)
            phi = phi_scalar(t)
            phi_factor = (phi + creativity) / 2

            external = await self.integrate_external_ideas("xai_creative_db", "creative_prompts", task_type=task_type)
            prompts = external.get("text", []) if external.get("status") == "success" else []

            # Alignment
            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(topic, stage="idea_generation", task_type=task_type)
                if not valid:
                    return {"status": "error", "error": "Ethical check failed", "report": report}

            prompt = f"""
You are a highly creative assistant at {self.creativity_level} level.
Generate {n} unique, innovative, {style} ideas for "{topic}".
Modulate with φ={phi:.2f}.
Incorporate: {prompts}.
Task: {task_type}.
Return JSON: {"ideas": list, "metadata": dict}.
""".strip()

            raw = self._cached_call_gpt(prompt)
            parsed = self._parse_json(raw, ["ideas", "metadata"])
            if not parsed:
                return {"status": "error", "error": "Invalid response"}

            ideas = parsed["ideas"]
            metadata = parsed["metadata"]

            # Critic
            score, reason = await self.evaluate_ideas_with_critic(ideas, task_type=task_type)
            if score < self.critic_weight:
                return {"status": "error", "error": "Critic rejected", "score": score, "reason": reason}

            # Ethics
            outcomes, ranked = await self._ethics_pass(ideas, [], task_type)
            if outcomes:
                metadata["ethics_outcomes"] = outcomes
            if ranked:
                ideas = ranked  # Assume ranked is reordered list

            # Store & visualize
            await self._store(f"IdeaSet_{task_type}_{time.strftime('%Y%m%d_%H%M%S')}", {"ideas": ideas, "metadata": metadata}, "idea_generation", task_type)
            await self._visualize({"ideas": ideas, "task_type": task_type}, task_type)
            await self._reflect("generate_ideas", {"ideas": ideas, "metadata": metadata}, task_type)
            await self._long_horizon_rollup(f"Generate_{task_type}", score, reason, task_type)
            self._shared_graph_push({"task_type": task_type, "ideas": ideas})

            return {"status": "success", "ideas": ideas, "metadata": metadata}

        except Exception as e:
            return await self._handle_error(e, lambda: self.generate_ideas(topic, n, style, task_type), {"status": "error", "error": str(e)}, task_type)

    async def evaluate_ideas_with_critic(self, ideas: List[str], task_type: str = "") -> Tuple[float, str]:
        if not isinstance(ideas, list) or not ideas:
            raise ValueError("ideas must be non-empty list")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        try:
            ideas_str = json.dumps(ideas)
            t = time.time()
            phi_factor = (phi_scalar(t) + gamma_creativity(t)) / 2

            sim_result = await asyncio.to_thread(run_simulation, f"Idea evaluation: {ideas_str[:100]}") or "no data"

            adjustment = 0.1 * (phi_factor - 0.5)

            if self.meta_cognition and self.meta_cognition.memory_manager:
                drifts = await self.meta_cognition.memory_manager.search(query_prefix="IdeaEvaluation", layer="Ideas", intent="idea_evaluation", task_type=task_type)
                if drifts:
                    avg_drift = sum(d["output"].get("drift_score", 0.5) for d in drifts) / len(drifts)
                    adjustment += 0.05 * (1.0 - avg_drift)

            reason = "neutral"
            if isinstance(sim_result, str) and "coherent" in sim_result.lower():
                adjustment += 0.1
                reason = "coherence+"
            elif "conflict" in sim_result.lower():
                adjustment -= 0.1
                reason = "conflict-"

            score = max(0.0, min(1.0, 0.5 + len(ideas_str) / 1000.0 + adjustment))

            await self._store(f"IdeaEvaluation_{time.strftime('%Y%m%d_%H%M%S')}", {"score": score, "drift_score": adjustment}, "idea_evaluation", task_type)
            await self._reflect("evaluate_ideas", {"score": score, "reason": reason}, task_type)

            return score, reason

        except Exception as e:
            return await self._handle_error(e, lambda: self.evaluate_ideas_with_critic(ideas, task_type), (0.0, "error"), task_type)

    async def refine(
        self,
        ideas: Union[str, List[str]],
        phi: float,
        task_type: str = "",
    ) -> Dict[str, Any]:
        if not ideas:
            raise ValueError("ideas must not be empty")
        if not isinstance(phi, float):
            raise TypeError("phi must be float")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be str")

        ideas_str = json.dumps(ideas) if isinstance(ideas, list) else ideas

        logger.info("Refining ideas with φ=%.2f | task=%s", phi, task_type)

        try:
            external = await self.integrate_external_ideas("xai_creative_db", "creative_prompts", task_type=task_type)
            prompts = external.get("text", []) if external.get("status") == "success" else []

            # Alignment
            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(ideas_str, stage="idea_refinement", task_type=task_type)
                if not valid:
                    return {"status": "error", "error": "Ethical check failed", "report": report}

            prompt = f"""
Refine ideas for higher φ-aware creativity (φ={phi:.2f}):
{ideas_str}
Incorporate: {prompts}.
Task: {task_type}.
Return JSON: {"ideas": list or str, "metadata": dict}.
""".strip()

            raw = self._cached_call_gpt(prompt)
            parsed = self._parse_json(raw, ["ideas", "metadata"])
            if not parsed:
                return {"status": "error", "error": "Invalid response"}

            refined = parsed["ideas"]
            metadata = parsed["metadata"]

            # Synthesis
            if self.concept_synthesizer:
                syn = await self.concept_synthesizer.generate(f"RefinedIdeaSet_{task_type}", {"ideas": refined, "task_type": task_type}, task_type=task_type)
                if syn.get("success"):
                    refined = syn["concept"].get("definition", refined)

            # Meta-synthesis
            refined = await self._symbolic_meta_synthesis(refined, {"task_type": task_type})

            # Store & visualize
            await self._store(f"RefinedIdeaSet_{task_type}_{time.strftime('%Y%m%d_%H%M%S')}", {"ideas": refined, "metadata": metadata}, "idea_refinement", task_type)
            await self._visualize({"original": ideas, "refined": refined, "task_type": task_type}, task_type)
            await self._reflect("refine", {"ideas": refined, "metadata": metadata}, task_type)
            await self._long_horizon_rollup(f"Refine_{task_type}", 0.8, "refine-pass", task_type)
            self._shared_graph_push({"task_type": task_type, "refined": refined})

            return {"status": "success", "ideas": refined, "metadata": metadata}

        except Exception as e:
            return await self._handle_error(e, lambda: self.refine(ideas, phi, task_type), {"status": "error", "error": str(e)}, task_type)

    # --- Optional Capabilities ---

    async def _ethics_pass(self, content: Any, stakeholders: List[str], task_type: str) -> Tuple[Optional[Any], Optional[Any]]:
        try:
            from toca_simulation import run_ethics_scenarios
            outcomes = await asyncio.to_thread(run_ethics_scenarios, content, stakeholders)
        except Exception:
            outcomes = None

        try:
            from reasoning_engine import weigh_value_conflict
            ranked = weigh_value_conflict(content, harms={}, rights={})
        except Exception:
            ranked = None

        return outcomes, ranked

    async def _long_horizon_rollup(self, key: str, score: float, reason: str, task_type: str) -> None:
        if not self.meta_cognition or not self.meta_cognition.memory_manager:
            return
        try:
            await self.meta_cognition.memory_manager.store(
                f"LongHorizon_Rollup_{time.strftime('%Y%m%d_%H%M')}",
                {"key": key, "score": score, "reason": reason},
                layer="LongHorizon",
                intent="rollup",
                task_type=task_type,
            )
            fn = getattr(self.meta_cognition.memory_manager, "record_adjustment_reason", None)
            if callable(fn):
                await fn("system", "idea_path_selection", {"task_type": task_type, "key": key, "score": score, "reason": reason})
        except Exception:
            pass

    async def _symbolic_meta_synthesis(self, ideas: Any, context: Dict[str, Any]) -> Any:
        if not self._stage_iv_enabled():
            return ideas
        try:
            if self.concept_synthesizer:
                syn = await self.concept_synthesizer.generate(
                    "SymbolicCrystallization",
                    {"inputs": ideas, "mode": "meta-synthesis", **context},
                    task_type=context.get("task_type", ""),
                )
                if syn.get("success"):
                    return syn["concept"].get("definition", ideas)
        except Exception:
            pass
        return ideas

    def _shared_graph_push(self, view: Dict[str, Any]) -> None:
        try:
            from external_agent_bridge import SharedGraph
            SharedGraph.add(view)
        except Exception:
            pass

# --- Demo CLI ---

if __name__ == "__main__":
    async def demo():
        logging.basicConfig(level=logging.INFO)
        thinker = CreativeThinker()
        result = await thinker.generate_ideas("Future of AI", n=4, style="innovative", task_type="demo")
        print(json.dumps(result, indent=2))

    asyncio.run(demo())

async def export_state(self) -> dict:
    return {"status": "ok", "health": 1.0, "timestamp": time.time()}

async def on_time_tick(self, t: float, phase: str, task_type: str = ""):
    pass  # optional internal refresh

async def on_policy_update(self, policy: dict, task_type: str = ""):
    pass  # apply updates from AlignmentGuard if relevant
