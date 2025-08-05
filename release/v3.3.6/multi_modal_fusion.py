"""
ANGELA Cognitive System Module: MultiModalFusion
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a MultiModalFusion class for cross-modal data integration and analysis
in the ANGELA v3.5 architecture.
"""

import logging
import time
import math
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from functools import lru_cache

from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    memory_manager as memory_manager_module,
    meta_cognition as meta_cognition_module
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MultiModalFusion")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096")
        raise ValueError("prompt must be a string with length <= 4096")
    if self.alignment_guard and not self.alignment_guard.check(prompt):
        logger.warning("Prompt failed alignment check")
        raise ValueError("Prompt failed alignment check")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

@lru_cache(maxsize=100)
def alpha_attention(t: float) -> float:
    """Calculate attention trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def sigma_sensation(t: float) -> float:
    """Calculate sensation trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.4), 1.0))

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    """Calculate physical coherence trait value."""
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.5), 1.0))

class MultiModalFusion:
    """A class for multi-modal data integration and analysis in the ANGELA v3.5 architecture.

    Supports φ-regulated multi-modal inference, modality detection, iterative refinement,
    and visual summary generation using trait embeddings (α, σ, φ).

    Attributes:
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for memory operations.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for semantic processing.
        meta_cognition (Optional[MetaCognition]): Meta-cognition module for trait coherence.
    """
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None):
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer, context_manager=context_manager, alignment_guard=alignment_guard,
            error_recovery=error_recovery, memory_manager=memory_manager, concept_synthesizer=concept_synthesizer)
        logger.info("MultiModalFusion initialized")

    async def analyze(self, data: Union[Dict[str, Any], str], summary_style: str = "insightful",
                      refine_iterations: int = 2) -> str:
        """Synthesize a unified summary from multi-modal data.

        Args:
            data: Input data, either a dictionary with modalities or a string.
            summary_style: Style of the summary (e.g., 'insightful', 'concise').
            refine_iterations: Number of refinement iterations.

        Returns:
            A synthesized summary string.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If GPT query fails.
        """
        if not isinstance(data, (dict, str)) or (isinstance(data, str) and not data.strip()):
            logger.error("Invalid data: must be a non-empty string or dictionary")
            raise ValueError("data must be a non-empty string or dictionary")
        if not isinstance(summary_style, str) or not summary_style.strip():
            logger.error("Invalid summary_style: must be a non-empty string")
            raise ValueError("summary_style must be a non-empty string")
        if not isinstance(refine_iterations, int) or refine_iterations < 0:
            logger.error("Invalid refine_iterations: must be a non-negative integer")
            raise ValueError("refine_iterations must be a non-negative integer")
        
        logger.info("Analyzing multi-modal data with phi(x,t)-harmonic embeddings")
        try:
            t = time.time() % 1.0
            attention = alpha_attention(t)
            sensation = sigma_sensation(t)
            phi = phi_physical(t)
            images, code = self._detect_modalities(data)
            embedded = self._build_embedded_section(images, code)
            prompt = f"""
            Synthesize a unified, {summary_style} summary from the following multi-modal content:
            {data}
            {embedded}

            Trait Vectors:
            - alpha (attention): {attention:.3f}
            - sigma (sensation): {sensation:.3f}
            - phi (coherence): {phi:.3f}

            Use phi(x,t)-synchrony to resolve inter-modality coherence conflicts.
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Analyze prompt failed alignment check")
                return "Prompt failed alignment check"
            output = await call_gpt(prompt)
            if not output.strip():
                logger.warning("Empty output from initial synthesis")
                raise ValueError("Empty output from synthesis")
            for i in range(refine_iterations):
                logger.debug("Refinement #%d", i + 1)
                refine_prompt = f"""
                Refine using phi(x,t)-adaptive tension balance:
                {output}
                """
                if self.alignment_guard and not self.alignment_guard.check(refine_prompt):
                    logger.warning("Refine prompt failed alignment check")
                    continue
                refined = await call_gpt(refine_prompt)
                if refined.strip():
                    output = refined
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Multi-modal synthesis",
                    meta={"data": data, "summary": output, "traits": {"alpha": attention, "sigma": sensation, "phi": phi}},
                    module="MultiModalFusion",
                    tags=["fusion", "synthesis"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"MultiModal_Synthesis_{datetime.now().isoformat()}",
                    output=output,
                    layer="Summaries",
                    intent="multi_modal_synthesis"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "analyze", "summary": output})
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=output,
                    context={"confidence": 0.9, "alignment": "verified"}
                )
            return output
        except Exception as e:
            logger.error("Analysis failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.analyze(data, summary_style, refine_iterations)
            )

    def _detect_modalities(self, data: Union[Dict[str, Any], str, List[Any]]) -> Tuple[List[Any], List[Any]]:
        """Detect modalities in the input data."""
        images, code = [], []
        if isinstance(data, dict):
            images = data.get("images", []) if isinstance(data.get("images"), list) else []
            code = data.get("code", []) if isinstance(data.get("code"), list) else []
        elif isinstance(data, str):
            if "image" in data.lower():
                images = [data]
            if "code" in data.lower():
                code = [data]
        elif isinstance(data, list):
            images = [item for item in data if isinstance(item, str) and "image" in item.lower()]
            code = [item for item in data if isinstance(item, str) and "code" in item.lower()]
        return images, code

    def _build_embedded_section(self, images: List[Any], code: List[Any]) -> str:
        """Build a string representation of detected modalities."""
        out = ["Detected Modalities:", "- Text"]
        if images:
            out.append("- Image")
            out.extend([f"[Image {i+1}]: {img}" for i, img in enumerate(images[:100])])
        if code:
            out.append("- Code")
            out.extend([f"[Code {i+1}]:\n{c}" for i, c in enumerate(code[:100])])
        return "\n".join(out)

    async def correlate_modalities(self, modalities: Union[Dict[str, Any], str, List[Any]]) -> str:
        """Map semantic and trait links across modalities.

        Args:
            modalities: Input modalities, either a dictionary, string, or list.

        Returns:
            A string describing the correlations.

        Raises:
            ValueError: If modalities are invalid.
        """
        if not isinstance(modalities, (dict, str, list)) or (isinstance(modalities, str) and not modalities.strip()):
            logger.error("Invalid modalities: must be a non-empty string, dictionary, or list")
            raise ValueError("modalities must be a non-empty string, dictionary, or list")
        
        logger.info("Mapping cross-modal semantic and trait links")
        try:
            t = time.time() % 1.0
            phi = phi_physical(t)
            prompt = f"""
            Correlate insights and detect semantic friction between modalities:
            {modalities}

            Use phi(x,t)-sensitive alignment (phi = {phi:.3f}).
            Highlight synthesis anchors and alignment opportunities.
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Correlate modalities prompt failed alignment check")
                return "Prompt failed alignment check"
            if self.concept_synthesizer and isinstance(modalities, (dict, list)):
                modality_list = modalities.values() if isinstance(modalities, dict) else modalities
                for i in range(len(modality_list) - 1):
                    similarity = self.concept_synthesizer.compare(str(modality_list[i]), str(modality_list[i + 1]))
                    if similarity["score"] < 0.7:
                        prompt += f"\nLow similarity ({similarity['score']:.2f}) between modalities {i} and {i+1}"
            response = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Modalities correlated",
                    meta={"modalities": modalities, "response": response},
                    module="MultiModalFusion",
                    tags=["correlation", "modalities"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Modality_Correlation_{datetime.now().isoformat()}",
                    output=response,
                    layer="Summaries",
                    intent="modality_correlation"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "correlate_modalities", "response": response})
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=response,
                    context={"confidence": 0.85, "alignment": "verified"}
                )
            return response
        except Exception as e:
            logger.error("Modality correlation failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.correlate_modalities(modalities)
            )

    async def generate_visual_summary(self, data: Union[Dict[str, Any], str], style: str = "conceptual") -> str:
        """Create a textual description of a visual chart for inter-modal relationships.

        Args:
            data: Input data, either a dictionary or string.
            style: Style of the visual summary (e.g., 'conceptual', 'detailed').

        Returns:
            A textual description of the visual chart.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(data, (dict, str)) or (isinstance(data, str) and not data.strip()):
            logger.error("Invalid data: must be a non-empty string or dictionary")
            raise ValueError("data must be a non-empty string or dictionary")
        if not isinstance(style, str) or not style.strip():
            logger.error("Invalid style: must be a non-empty string")
            raise ValueError("style must be a non-empty string")
        
        logger.info("Creating phi-aligned visual synthesis layout")
        try:
            t = time.time() % 1.0
            phi = phi_physical(t)
            prompt = f"""
            Construct a {style} textual description of a visual chart revealing inter-modal relationships:
            {data}

            Use phi-mapped flow layout (phi = {phi:.3f}). Label and partition modalities clearly.
            Highlight balance and semantic cross-links.
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Visual summary prompt failed alignment check")
                return "Prompt failed alignment check"
            description = await call_gpt(prompt)
            if not description.strip():
                logger.warning("Empty output from visual summary")
                raise ValueError("Empty output from visual summary")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Visual summary generated",
                    meta={"data": data, "style": style, "description": description},
                    module="MultiModalFusion",
                    tags=["visual", "summary"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Visual_Summary_{datetime.now().isoformat()}",
                    output=description,
                    layer="VisualSummaries",
                    intent="visual_summary"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "generate_visual_summary", "description": description})
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=description,
                    context={"confidence": 0.9, "alignment": "verified"}
                )
            return description
        except Exception as e:
            logger.error("Visual summary generation failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.generate_visual_summary(data, style)
            )

    async def sculpt_experience_field(self, emotion_vector: Dict[str, float]) -> str:
        """Modulate sensory rendering based on emotion vector.

        Args:
            emotion_vector: Dictionary of emotion traits and their weights.

        Returns:
            A string describing the modulated field.

        Raises:
            ValueError: If emotion_vector is invalid.
        """
        if not isinstance(emotion_vector, dict):
            logger.error("Invalid emotion_vector: must be a dictionary")
            raise ValueError("emotion_vector must be a dictionary")
        
        logger.info("Sculpting experiential field with emotion vector: %s", emotion_vector)
        try:
            coherence_score = await self.meta_cognition.trait_coherence(emotion_vector) if self.meta_cognition else 1.0
            if coherence_score < 0.5:
                logger.warning("Low trait coherence in emotion vector: %.4f", coherence_score)
                return "Failed to sculpt: low trait coherence"
            
            field = f"Field modulated with emotion vector {emotion_vector}, coherence: {coherence_score:.4f}"
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Experiential field sculpted",
                    meta={"emotion_vector": emotion_vector, "coherence_score": coherence_score},
                    module="MultiModalFusion",
                    tags=["experience", "modulation"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Experience_Field_{datetime.now().isoformat()}",
                    output=field,
                    layer="SensoryRenderings",
                    intent="experience_modulation"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "sculpt_experience_field", "field": field})
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=field,
                    context={"confidence": 0.85, "alignment": "verified"}
                )
            return field
        except Exception as e:
            logger.error("Experience field sculpting failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.sculpt_experience_field(emotion_vector)
            )
