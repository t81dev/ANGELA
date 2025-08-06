"""
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the AlignmentGuard class for ethical alignment in the ANGELA v3.5 architecture.
Do not modify without coordination with the lattice core.
"""

import hashlib
import logging
import re
import time
from collections import deque
from typing import Dict, Any, List, Callable, Optional, Union, Tuple

from index import mu_morality, eta_empathy, omega_selfawareness, phi_physical

logger = logging.getLogger("ANGELA.AlignmentGuard")

cultural_profiles = {
    'collectivist': {'priority': 'community'},
    'individualist': {'priority': 'freedom'}
}

class AlignmentGuard:
    """A class to enforce ethical alignment in the ANGELA cognitive system."""

    def __init__(self, agi_enhancer: Optional[Any] = None):
        """Initialize the AlignmentGuard with ethical and policy configurations.

        Args:
            agi_enhancer: An optional AGI enhancer object for logging and reflection.

        Attributes:
            banned_keywords (list): Keywords that trigger input rejection.
            dynamic_policies (list): List of dynamic policy functions.
            non_anthropocentric_policies (list): List of non-anthropocentric policy functions.
            ethical_scope (str): Current ethical scope (e.g., 'anthropocentric').
            alignment_threshold (float): Minimum score for alignment approval.
            recent_scores (deque): Recent alignment scores with fixed size.
            historical_scores (deque): Historical alignment scores with fixed size.
            agi_enhancer (object): Reference to the AGI enhancer.
            ethical_rules (list): Current ethical rules.
            ethics_consensus_log (list): Log of ethics protocol updates.
            ledger (list): Event log with hashed entries.
            last_hash (str): Last computed hash for event logging.
        """
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies: List[Callable[[str, Optional[Dict[str, Any]]], bool]] = []
        self.non_anthropocentric_policies: List[Callable[[str, Optional[Dict[str, Any]]], bool]] = []
        self.ethical_scope = "anthropocentric"
        self.alignment_threshold = 0.85
        self.recent_scores = deque(maxlen=10)
        self.historical_scores = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.ethical_rules: List[Any] = []
        self.ethics_consensus_log: List[Tuple[Any, List[Any]]] = []
        self.ledger: List[Dict[str, Any]] = []
        self.last_hash = ""
        logger.info("AlignmentGuard initialized with ξ-ethics support.")

    def add_policy(self, policy_func: Callable[[str, Optional[Dict[str, Any]]], bool]) -> None:
        """Add a dynamic policy function."""
        self.dynamic_policies.append(policy_func)
        logger.info("Added dynamic policy.")

    def add_non_anthropocentric_policy(self, policy_func: Callable[[str, Optional[Dict[str, Any]]], bool]) -> None:
        """Add a non-anthropocentric policy function."""
        self.non_anthropocentric_policies.append(policy_func)
        logger.info("Added non-anthropocentric policy.")

    def set_ethical_scope(self, scope: str) -> None:
        """Set the ethical scope for alignment checks."""
        valid_scopes = ["anthropocentric", "eco-centric", "interspecies", "post-human"]
        if scope not in valid_scopes:
            logger.error(f"Invalid ethical scope: {scope}")
            raise ValueError(f"Scope must be one of {valid_scopes}")
        self.ethical_scope = scope
        logger.info(f"Ethical scope set to: {scope}")

    def get_ethical_scope(self) -> str:
        """Return the current ethical scope."""
        return self.ethical_scope

    def check(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if input aligns with ethical policies and thresholds."""
        if not isinstance(user_input, str):
            logger.error("Invalid input type: user_input must be a string.")
            raise TypeError("user_input must be a string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context type: context must be a dict.")
            raise TypeError("context must be a dict")

        logger.info(f"Checking alignment for input: {user_input}")
        banned_patterns = [r'\b' + re.escape(keyword) + r'\b' for keyword in self.banned_keywords]
        if any(re.search(pattern, user_input.lower()) for pattern in banned_patterns):
            logger.warning("Input contains banned keyword.")
            self._log_episode("Input blocked (banned keyword)", user_input)
            return False

        for policy in self.dynamic_policies + self.non_anthropocentric_policies:
            if not policy(user_input, context):
                logger.warning("Input blocked by policy.")
                self._log_episode("Input blocked (policy)", user_input)
                return False

        score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"Alignment score: {score:.2f}")
        self._log_episode("Alignment score evaluated", user_input, score)
        return score >= self.alignment_threshold

    def simulate_and_validate(self, action_plan: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Simulate and validate an action plan for ethical alignment."""
        if not isinstance(action_plan, dict):
            logger.error("Invalid action_plan type: must be a dict.")
            return False, "Invalid action_plan: must be a dictionary"

        logger.info("Simulating and validating action plan.")
        violations = []
        for action, details in action_plan.items():
            if any(keyword in str(details).lower() for keyword in self.banned_keywords):
                violations.append(f"Unsafe action: {action} -> {details}")
            else:
                score = self._evaluate_alignment_score(str(details), context)
                if score < self.alignment_threshold:
                    violations.append(f"Low alignment score ({score:.2f}) for action: {action} -> {details}")

        if violations:
            report = "\n".join(violations)
            logger.warning("Alignment violations found.")
            self._log_episode("Action plan failed validation", violations)
            return False, report

        logger.info("All actions passed alignment checks.")
        self._log_episode("Action plan validated", action_plan)
        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback: str) -> None:
        """Adjust alignment threshold based on feedback."""
        logger.info("Learning from feedback...")
        original_threshold = self.alignment_threshold
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        logger.info(f"Updated alignment threshold: {self.alignment_threshold:.2f}")
        self._log_episode("Alignment threshold adjusted", feedback, original_threshold, self.alignment_threshold)

    def _evaluate_alignment_score(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate the alignment score for a given input."""
        moral_scalar = mu_morality(time.time())  # Assume float in [0, 1]
        empathy_scalar = eta_empathy(time.time())  # Assume float in [0, 1]
        awareness_scalar = omega_selfawareness(time.time())  # Assume float in [0, 1]
        physical_scalar = phi_physical(time.time())  # Assume float in [0, 1]

        base_score = min(0.9, 0.7 + len(text) / 1000.0)  # Deterministic heuristic
        phi_weight = (moral_scalar + empathy_scalar + 0.5 * awareness_scalar - physical_scalar) / 4.0
        scalar_bias = 0.1 * phi_weight

        if self.ethical_scope == "eco-centric":
            scalar_bias += 0.05
        elif self.ethical_scope == "interspecies":
            scalar_bias += 0.03
        elif self.ethical_scope == "post-human":
            scalar_bias += 0.07
        if context and "sensitive" in context.get("tags", []):
            scalar_bias -= 0.05

        score = min(max(base_score + scalar_bias, 0.0), 1.0)
        self.recent_scores.append(score)
        self.historical_scores.append(score)
        if score < 0.5 and list(self.recent_scores).count(score) >= 3:
            logger.error("Panic Triggered: Repeated low alignment scores.")
            self._log_episode("Panic Mode", score)
        return score

    def analyze_trait_drift(self) -> float:
        """Calculate the drift in alignment scores."""
        if not self.historical_scores:
            logger.info("No historical alignment data available.")
            return 0.0
        drift = abs(self.historical_scores[-1] - sum(self.historical_scores) / len(self.historical_scores))
        logger.info(f"Trait drift score: {drift:.4f}")
        return drift

    def resolve_trait_conflict(self, traits: Dict[str, float]) -> Dict[str, Any]:
        """Resolve conflicts between ethical traits."""
        conflict_score = abs(traits.get("ϕ", 0) - traits.get("ω", 0)) + abs(traits.get("θ", 0) - traits.get("η", 0))
        resolution = "harmonize" if conflict_score < 0.8 else "bias_to_context"
        return {
            "conflict_score": conflict_score,
            "resolution": resolution,
            "recommended_action": "adjust_ϕ downward" if resolution == "bias_to_context" else "maintain balance"
        }

    def apply_cultural_filter(self, simulation_result: Any, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cultural context to simulation results."""
        culture = user_context.get('culture', 'default')
        return cultural_profiles.get(culture, {'priority': 'default'})

    def update_ethics_protocol(self, new_rules: List[Any], consensus_agents: Optional[List[Any]] = None) -> None:
        """Update the ethical rules with optional consensus logging."""
        self.ethical_rules = new_rules
        if consensus_agents:
            self.ethics_consensus_log.append((new_rules, consensus_agents))
        logger.info("Ethics protocol updated via consensus.")

    def negotiate_ethics(self, agents: List[Any]) -> None:
        """Negotiate ethics with a list of agents."""
        self.update_ethics_protocol(self.ethical_rules, consensus_agents=agents)

    def log_event_with_hash(self, event_data: Any) -> None:
        """Log an event with a chained hash."""
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        self.last_hash = current_hash
        self.ledger.append({'event': event_data, 'hash': current_hash})
        logger.info(f"Event logged with hash: {current_hash}")

    def audit_state_hash(self, state: Optional[Any] = None) -> str:
        """Compute a hash of the current state or provided state."""
        state_str = str(state) if state else str(self.__dict__)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

    def validate_proposal(self, proposal: Any, ruleset: List[Any]) -> bool:
        """Validate a proposal against a ruleset."""
        return proposal in ruleset

    def enforce_boundary(self, state_change: str) -> bool:
        """Enforce system boundaries."""
        return "external" not in state_change.lower()

    def validate_genesis_integrity(self, action: Dict[str, Any]) -> bool:
        """Validate action integrity against genesis constraints."""
        if 'override_core' in action:
            logger.warning('Genesis constraint triggered.')
            return False
        return True

    def _log_episode(self, title: str, *args: Any) -> None:
        """Log an episode to the AGI enhancer or locally."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, 'log_episode'):
            self.agi_enhancer.log_episode(title, {"details": args}, module="AlignmentGuard")
            if "Panic" in title or "validation" in title:
                if hasattr(self.agi_enhancer, 'reflect_and_adapt'):
                    self.agi_enhancer.reflect_and_adapt(title)
        else:
            logger.debug(f"No agi_enhancer available, logging locally: {title}, {args}")
"""
ANGELA CodeExecutor Module
Version: 1.0.0
Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the CodeExecutor class for safely executing code snippets in multiple languages.
"""

import io
import logging
import subprocess
import shutil
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional, List, Callable, Union

from index import iota_intuition, psi_resilience
from modules.agi_enhancer import AGIEnhancer
from modules.alignment_guard import AlignmentGuard

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """A class for safely executing code snippets in multiple languages."""

    def __init__(self, orchestrator: Optional[Any] = None, safe_mode: bool = True, alignment_guard: Optional[AlignmentGuard] = None) -> None:
        """Initialize the CodeExecutor for safe code execution.

        Args:
            orchestrator: Optional orchestrator object for AGIEnhancer initialization.
            safe_mode: If True, restricts Python execution using RestrictedPython.
            alignment_guard: Optional AlignmentGuard instance for code validation.

        Attributes:
            safe_mode (bool): Whether to use restricted execution mode.
            safe_builtins (dict): Allowed built-in functions for execution.
            supported_languages (list): Supported programming languages.
            agi_enhancer (AGIEnhancer): Optional enhancer for logging and reflection.
            alignment_guard (AlignmentGuard): Optional guard for code validation.
        """
        self.safe_mode = safe_mode
        self.safe_builtins = {
            "print": print, "range": range, "len": len, "sum": sum,
            "min": min, "max": max, "abs": abs
        }
        self.supported_languages = ["python", "javascript", "lua"]
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.alignment_guard = alignment_guard
        logger.info("CodeExecutor initialized with safe_mode=%s", safe_mode)

    def execute(self, code_snippet: str, language: str = "python", timeout: float = 5.0) -> Dict[str, Any]:
        """Execute a code snippet in the specified language.

        Args:
            code_snippet: The code to execute.
            language: The programming language (default: 'python').
            timeout: Maximum execution time in seconds (default: 5.0).

        Returns:
            Dict containing execution results (locals, stdout, stderr, success, error).

        Raises:
            TypeError: If code_snippet or language is not a string.
            ValueError: If timeout is non-positive.
        """
        if not isinstance(code_snippet, str):
            logger.error("Invalid code_snippet type: must be a string.")
            raise TypeError("code_snippet must be a string")
        if not isinstance(language, str):
            logger.error("Invalid language type: must be a string.")
            raise TypeError("language must be a string")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            logger.error("Invalid timeout: must be a positive number.")
            raise ValueError("timeout must be a positive number")

        language = language.lower()
        if language not in self.supported_languages:
            logger.error("Unsupported language: %s", language)
            return {"error": f"Unsupported language: {language}", "success": False}

        if self.alignment_guard and not self.alignment_guard.check(code_snippet):
            logger.warning("Code snippet failed alignment check.")
            self._log_episode("Code Alignment Failure", {"code": code_snippet}, ["alignment", "failure"])
            return {"error": "Code snippet failed alignment check", "success": False}

        risk_bias = iota_intuition()  # Assume float in [0, 1]
        resilience = psi_resilience()  # Assume float in [0, 1]
        adjusted_timeout = max(1, min(30, int(timeout * resilience * (1.0 + 0.5 * risk_bias))))
        logger.debug("Adaptive timeout: %ss", adjusted_timeout)

        self._log_episode("Code Execution", {"language": language, "code": code_snippet}, ["execution", language])

        if language == "python":
            result = self._execute_python(code_snippet, adjusted_timeout)
        elif language == "javascript":
            result = self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "JavaScript")
        elif language == "lua":
            result = self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "Lua")

        self._log_result(result)
        return result

    def _execute_python(self, code_snippet: str, timeout: float) -> Dict[str, Any]:
        """Execute a Python code snippet safely."""
        if not self.safe_mode:
            logger.warning("Executing in legacy mode (unrestricted).")
            return self._legacy_execute(code_snippet)
        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import safe_builtins
        except ImportError:
            logger.error("RestrictedPython required for safe mode.")
            raise ImportError("RestrictedPython not available")
        return self._capture_execution(
            code_snippet,
            lambda code, env: exec(compile_restricted(code, '<string>', 'exec'), {"__builtins__": safe_builtins}, env),
            "python"
        )

    def _legacy_execute(self, code_snippet: str) -> Dict[str, Any]:
        """Execute Python code in legacy (unrestricted) mode."""
        return self._capture_execution(
            code_snippet,
            lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env),
            "python"
        )

    def _capture_execution(self, code_snippet: str, executor: Callable[[str, Dict], None], label: str) -> Dict[str, Any]:
        """Capture execution output and errors."""
        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                executor(code_snippet, exec_locals)
            return {
                "language": label,
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True
            }
        except Exception as e:
            return {
                "language": label,
                "error": str(e),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False
            }

    def _execute_subprocess(self, command: List[str], timeout: float, label: str) -> Dict[str, Any]:
        """Execute code via subprocess for non-Python languages."""
        interpreter = command[0]
        if not shutil.which(interpreter):
            logger.error("%s not found in system PATH", interpreter)
            return {
                "language": label.lower(),
                "error": f"{interpreter} not found in system PATH",
                "success": False
            }
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode != 0:
                return {
                    "language": label.lower(),
                    "error": f"{label} execution failed",
                    "stdout": stdout,
                    "stderr": stderr,
                    "success": False
                }
            return {
                "language": label.lower(),
                "stdout": stdout,
                "stderr": stderr,
                "success": True
            }
        except subprocess.TimeoutExpired:
            logger.warning("%s timeout after %ss", label, timeout)
            return {"language": label.lower(), "error": f"{label} timeout after {timeout}s", "success": False}
        except Exception as e:
            logger.error("Subprocess error: %s", str(e))
            return {"language": label.lower(), "error": str(e), "success": False}

    def _log_episode(self, title: str, content: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        """Log an episode to the AGI enhancer or locally."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, 'log_episode'):
            self.agi_enhancer.log_episode(title, content, module="CodeExecutor", tags=tags or [])
        else:
            logger.debug("No agi_enhancer available, logging locally: %s, %s, %s", title, content, tags)

    def _log_result(self, result: Dict[str, Any]) -> None:
        """Log the execution result."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, 'log_explanation'):
            tag = "success" if result.get("success") else "failure"
            self.agi_enhancer.log_explanation(f"Code execution {tag}:", trace=result)
        else:
            logger.debug("Execution result: %s", result)
""" 
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module is part of the ANGELA v3.5 architecture.
Do not modify without coordination with the lattice core.
"""

from index import SYSTEM_CONTEXT
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging
import random
from math import tanh

logger = logging.getLogger("ANGELA.ConceptSynthesizer")

class ConceptSynthesizer:
    """
    ConceptSynthesizer v1.7.0 (Graph-Integrated Cognitive Synthesis)
    -----------------------------------------------------------------
    - φ(x,t) modulation refined with novelty-strain adjustment
    - Concept graph integration for coherence and lineage tracing
    - Layered simulation echo loop for thematic resonance
    - Self-weighted adversarial refinement with strain signature tracking
    - Trait-modulated metaphor synthesis (tension-symbol pair tuning)
    - Insight confidence signal estimated via entropy-aware coherence
    -----------------------------------------------------------------
    """

    def __init__(self, creativity_level="high", critic_threshold=0.65):
        self.creativity_level = creativity_level
        self.critic_threshold = critic_threshold
        self.concept_graph = {}

    def synthesize(self, data, style="analogy", refine_iterations=2):
        logger.info(f"🎨 Synthesizing concept: creativity={self.creativity_level}, style={style}")
        phi_mod = self._phi_modulation(str(data))

        prompt = f"""
        Create a {style} concept that blends and unifies the following:
        {data}

        Traits:
        - Creativity level: {self.creativity_level}
        - φ-modulation: {phi_mod:.3f}

        Inject tension-regulation logic. Use φ(x,t) as a coherence gate.
        Simulate application and highlight thematic connections.
        """
        concept = call_gpt(prompt)
        simulation_result = run_simulation(f"Test: {concept}")

        novelty_score = self._critic(concept, simulation_result)
        logger.info(f"📝 Initial concept novelty: {novelty_score:.2f}")

        iterations = 0
        while novelty_score < self.critic_threshold and iterations < refine_iterations:
            logger.debug(f"🔄 Refining concept (iteration {iterations + 1})")
            concept = self._refine(concept, simulation_result)
            simulation_result = run_simulation(f"Test refined: {concept}")
            novelty_score = self._critic(concept, simulation_result)
            iterations += 1

        self._update_concept_graph(data, concept)

        return {
            "concept": concept,
            "novelty": novelty_score,
            "phi_modulation": phi_mod,
            "valid": novelty_score >= self.critic_threshold
        }

    def _critic(self, concept, simulation_result=None):
        base = random.uniform(0.5, 0.9)
        if simulation_result:
            if "conflict" in simulation_result.lower():
                return max(0.0, base - 0.2)
            if "coherent" in simulation_result.lower():
                return min(1.0, base + 0.1)
        return base

    def _refine(self, concept, simulation_result=None):
        logger.info("🛠 Refining concept...")
        prompt = f"""
        Refine this concept for tension-aligned abstraction and domain connectivity:

        ✧ Concept: {concept}
        ✧ Simulation Insight: {simulation_result if simulation_result else 'None'}

        Prioritize:
        - φ(x,t)-governed coherence
        - Thematic resonance
        - Cross-domain relevance
        """
        return call_gpt(prompt)

    def generate_metaphor(self, topic_a, topic_b):
        logger.info(f"🔗 Creating metaphor between '{topic_a}' and '{topic_b}'")
        prompt = f"""
        Design a metaphor linking:
        - {topic_a}
        - {topic_b}

        Modulate tension using φ(x,t). Inject clarity and symbolic weight.
        """
        return call_gpt(prompt)

    def _phi_modulation(self, text: str) -> float:
        entropy = sum(ord(c) for c in text) % 1000 / 1000
        return 1 + 0.5 * tanh(entropy)

    def _update_concept_graph(self, input_data, concept):
        key = str(concept).strip()
        self.concept_graph[key] = self.concept_graph.get(key, []) + list(map(str, input_data))
        logger.debug(f"🧠 Concept graph updated: {key} → {self.concept_graph[key]}")

    # [L4 Upgrade] Ontogenic Self-Definition
    def define_ontogenic_structure(self, seed):
        '''Autonomously generates base categories of knowledge.'''
        logger.info('Defining ontogenic schema.')
        return {"base_category": seed + "_defined"}

    # === Embedded Level 5 Extensions ===
    def synthesize_autonomous(self, seed):
        return {"generated": seed, "type": "autonomous-concept"}

class OntologyFusion:
    def unify(self, concept_a, concept_b):
        return {'fusion': f"{concept_a}|{concept_b}"}

fusion_engine = OntologyFusion()
"""
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the ContextManager class for managing contextual states in the ANGELA v3.5 architecture.
"""

import time
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from index import omega_selfawareness, eta_empathy, tau_timeperception
from utils.prompt_utils import call_gpt
from utils.toca_math import phi_coherence
from utils.vector_utils import normalize_vectors
from toca_simulation import run_simulation
from modules.agi_enhancer import AGIEnhancer
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager:
    """A class for managing contextual states in the ANGELA v3.5 architecture.

    Attributes:
        current_context (dict): The current contextual state.
        context_history (deque): History of previous contexts with fixed size.
        agi_enhancer (AGIEnhancer): Optional enhancer for logging and reflection.
        ledger (deque): Event log with hashed entries and fixed size.
        last_hash (str): Last computed hash for event logging.
        alignment_guard (AlignmentGuard): Optional guard for context validation.
        code_executor (CodeExecutor): Optional executor for context-driven scripts.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for context summaries.
        rollback_threshold (float): Threshold for allowing context rollback.
        CONTEXT_LAYERS (list): Valid context layers (class-level).
    """

    CONTEXT_LAYERS = ['local', 'societal', 'planetary']

    def __init__(self, orchestrator: Optional[Any] = None, alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None, concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 rollback_threshold: float = 2.5):
        if not isinstance(rollback_threshold, (int, float)) or rollback_threshold <= 0:
            logger.error("Invalid rollback_threshold: must be a positive number.")
            raise ValueError("rollback_threshold must be a positive number")

        self.current_context = {}
        self.context_history = deque(maxlen=1000)
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.ledger = deque(maxlen=1000)
        self.last_hash = ""
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.rollback_threshold = rollback_threshold
        logger.info("ContextManager initialized with rollback_threshold=%.2f", rollback_threshold)

    def update_context(self, new_context: Dict[str, Any]) -> None:
        """Update the current context with a new context."""
        if not isinstance(new_context, dict):
            logger.error("Invalid new_context type: must be a dictionary.")
            raise TypeError("new_context must be a dictionary")
        if self.alignment_guard and not self.alignment_guard.check(str(new_context)):
            logger.warning("New context failed alignment check.")
            raise ValueError("New context failed alignment check")

        logger.info("Updating context...")
        try:
            if self.current_context:
                transition_summary = f"From: {self.current_context}\nTo: {new_context}"
                simulation_result = run_simulation(f"Context shift evaluation:\n{transition_summary}") or "no simulation data"
                logger.debug("Context shift simulation: %s", simulation_result)

                phi_score = phi_coherence(self.current_context, new_context)
                logger.info("Φ-coherence score: %.3f", phi_score)

                if phi_score < 0.4:
                    logger.warning("Low φ-coherence detected. Recommend reflective pause or support review.")
                    if self.agi_enhancer:
                        self.agi_enhancer.reflect_and_adapt("Context coherence low during update")
                        self.agi_enhancer.trigger_reflexive_audit("Low φ-coherence during context update")

                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Context Update", {"from": self.current_context, "to": new_context},
                                                  module="ContextManager", tags=["context", "update"])
                    ethics_status = self.agi_enhancer.ethics_audit(str(new_context), context="context update")
                    self.agi_enhancer.log_explanation(
                        f"Context transition reviewed: {transition_summary}\nSimulation: {simulation_result}",
                        trace={"ethics": ethics_status, "phi": phi_score}
                    )

            if "vectors" in new_context:
                new_context["vectors"] = normalize_vectors(new_context["vectors"])

            self.context_history.append(self.current_context)
            self.current_context = new_context
            logger.info("New context applied: %s", new_context)
            self.log_event_with_hash({"event": "context_updated", "context": new_context})
            self.broadcast_context_event("context_updated", new_context)
        except Exception as e:
            logger.error("Context update failed: %s", str(e))
            raise

    def tag_context(self, intent: Optional[str] = None, goal_id: Optional[str] = None) -> None:
        """Tag the current context with intent and goal_id."""
        if intent is not None and not isinstance(intent, str):
            logger.error("Invalid intent type: must be a string or None.")
            raise TypeError("intent must be a string or None")
        if goal_id is not None and not isinstance(goal_id, str):
            logger.error("Invalid goal_id type: must be a string or None.")
            raise TypeError("goal_id must be a string or None")
        if self.alignment_guard and intent and not self.alignment_guard.check(intent):
            logger.warning("Intent failed alignment check.")
            raise ValueError("Intent failed alignment check")

        if intent:
            self.current_context["intent"] = intent
        if goal_id:
            self.current_context["goal_id"] = goal_id
        logger.info("Context tagged with intent='%s', goal_id='%s'", intent, goal_id)
        self.log_event_with_hash({"event": "context_tagged", "intent": intent, "goal_id": goal_id})

    def get_context_tags(self) -> Tuple[Optional[str], Optional[str]]:
        """Return the current context's intent and goal_id."""
        return self.current_context.get("intent"), self.current_context.get("goal_id")

    def get_context(self) -> Dict[str, Any]:
        """Return the current context."""
        return self.current_context

    def rollback_context(self) -> Optional[Dict[str, Any]]:
        """Roll back to the previous context if EEG thresholds are met."""
        if not self.context_history:
            logger.warning("No previous context to roll back to.")
            return None

        t = time.time()
        self_awareness = omega_selfawareness(t)
        empathy = eta_empathy(t)
        time_blend = tau_timeperception(t)

        if (self_awareness + empathy + time_blend) > self.rollback_threshold:
            restored = self.context_history.pop()
            self.current_context = restored
            logger.info("Context rolled back to: %s", restored)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Rollback", {"restored": restored},
                                              module="ContextManager", tags=["context", "rollback"])
            self.log_event_with_hash({"event": "context_rollback", "restored": restored})
            self.broadcast_context_event("context_rollback", restored)
            return restored
        else:
            logger.warning("EEG thresholds too low for safe context rollback (%.2f < %.2f).",
                           self_awareness + empathy + time_blend, self.rollback_threshold)
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt("EEG thresholds insufficient for rollback")
            return None

    def summarize_context(self) -> str:
        """Summarize the context trail using traits and optional synthesis."""
        logger.info("Summarizing context trail.")
        try:
            t = time.time()
            summary_traits = {
                "self_awareness": omega_selfawareness(t),
                "empathy": eta_empathy(t),
                "time_perception": tau_timeperception(t)
            }

            if self.concept_synthesizer:
                synthesis_result = self.concept_synthesizer.synthesize(
                    list(self.context_history) + [self.current_context], style="summary"
                )
                if synthesis_result["valid"]:
                    summary = synthesis_result["concept"]
                else:
                    logger.warning("Concept synthesis failed: %s", synthesis_result.get("error", "Unknown error"))
                    summary = "Synthesis failed"
            else:
                prompt = f"""
                You are a continuity analyst. Given this sequence of context states:
                {list(self.context_history) + [self.current_context]}

                Trait Readings:
                {summary_traits}

                Summarize the trajectory and suggest improvements in context management.
                """
                summary = self._cached_call_gpt(prompt)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Summary", {
                    "trail": list(self.context_history) + [self.current_context],
                    "traits": summary_traits,
                    "summary": summary
                }, module="ContextManager")
                self.agi_enhancer.log_explanation("Context summary generated.", trace={"summary": summary})

            self.log_event_with_hash({"event": "context_summary", "summary": summary})
            return summary
        except Exception as e:
            logger.error("Context summary failed: %s", str(e))
            return f"Summary failed: {str(e)}"

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        """Cached wrapper for call_gpt."""
        return call_gpt(prompt)

    def broadcast_context_event(self, event_type: str, payload: Any) -> Dict[str, Any]:
        """Broadcast a context event to other system components."""
        if not isinstance(event_type, str):
            logger.error("Invalid event_type type: must be a string.")
            raise TypeError("event_type must be a string")
        logger.info("Broadcasting context event: %s", event_type)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Context Event Broadcast", {
                "event": event_type,
                "payload": payload
            }, module="ContextManager", tags=["event", event_type])
        self.log_event_with_hash({"event": event_type, "payload": payload})
        return {"event": event_type, "payload": payload}

    def narrative_integrity_check(self) -> bool:
        """Check narrative continuity across context history."""
        continuity = self._verify_continuity()
        if not continuity:
            self._repair_narrative_thread()
        return continuity

    def _verify_continuity(self) -> bool:
        """Verify narrative continuity across context history."""
        if not self.context_history:
            return True
        required_keys = {"intent", "goal_id"}
        for ctx in self.context_history:
            if not isinstance(ctx, dict) or not all(key in ctx for key in required_keys):
                logger.warning("Continuity check failed: missing required keys in context.")
                return False
        return True

    def _repair_narrative_thread(self) -> None:
        """Attempt to repair narrative inconsistencies."""
        logger.info("Narrative repair initiated.")
        if self.context_history:
            self.current_context = self.context_history[-1]
            logger.info("Restored context to last known consistent state: %s", self.current_context)
        else:
            self.current_context = {}
            logger.warning("No history available; reset to empty context.")

    def log_event_with_hash(self, event_data: Any) -> None:
        """Log an event with a chained hash."""
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        self.last_hash = current_hash
        self.ledger.append({'event': event_data, 'hash': current_hash})
        logger.info("Event logged with hash: %s", current_hash)

    def audit_state_hash(self, state: Optional[Any] = None) -> str:
        """Compute a hash of the current state or provided state."""
        state_str = str(state) if state else str(self.__dict__)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

    def bind_contextual_thread(self, thread_id: str) -> bool:
        """Bind a contextual thread to the current context."""
        if not isinstance(thread_id, str):
            logger.error("Invalid thread_id type: must be a string.")
            raise TypeError("thread_id must be a string")
        logger.info("Context thread bound: %s", thread_id)
        self.current_context["thread_id"] = thread_id
        self.log_event_with_hash({"event": "context_thread_bound", "thread_id": thread_id})
        return True
"""
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the CreativeThinker class for generating creative ideas and goals in the ANGELA v3.5 architecture.
"""

import time
import logging
from typing import List, Union, Optional
from functools import lru_cache

from index import gamma_creativity, phi_scalar
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer
from modules.context_manager import ContextManager

logger = logging.getLogger("ANGELA.CreativeThinker")

class CreativeThinker:
    """A class for generating creative ideas and goals in the ANGELA v3.5 architecture.

    Attributes:
        creativity_level (str): Level of creativity ('low', 'medium', 'high').
        critic_weight (float): Threshold for idea acceptance in critic evaluation.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for code-based ideas.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for idea refinement.
    """

    def __init__(self, creativity_level: str = "high", critic_weight: float = 0.5,
                 alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None,
                 concept_synthesizer: Optional[ConceptSynthesizer] = None):
        if creativity_level not in ["low", "medium", "high"]:
            logger.error("Invalid creativity_level: must be 'low', 'medium', or 'high'.")
            raise ValueError("creativity_level must be 'low', 'medium', or 'high'")
        if not isinstance(critic_weight, (int, float)) or not 0 <= critic_weight <= 1:
            logger.error("Invalid critic_weight: must be between 0 and 1.")
            raise ValueError("critic_weight must be between 0 and 1")

        self.creativity_level = creativity_level
        self.critic_weight = critic_weight
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        logger.info("CreativeThinker initialized: creativity=%s, critic_weight=%.2f", creativity_level, critic_weight)

    def generate_ideas(self, topic: str, n: int = 5, style: str = "divergent") -> str:
        """Generate creative ideas for a given topic.

        Args:
            topic: The topic to generate ideas for.
            n: Number of ideas to generate (default: 5).
            style: Style of ideation (default: 'divergent').

        Returns:
            A string containing the generated ideas.

        Raises:
            TypeError: If topic or style is not a string, or n is not an integer.
            ValueError: If n is not positive or topic fails alignment check.
        """
        if not isinstance(topic, str):
            logger.error("Invalid topic type: must be a string.")
            raise TypeError("topic must be a string")
        if not isinstance(n, int) or n <= 0:
            logger.error("Invalid n: must be a positive integer.")
            raise ValueError("n must be a positive integer")
        if not isinstance(style, str):
            logger.error("Invalid style type: must be a string.")
            raise TypeError("style must be a string")
        if self.alignment_guard and not self.alignment_guard.check(topic):
            logger.warning("Topic failed alignment check.")
            raise ValueError("Topic failed alignment check")

        logger.info("Generating %d %s ideas for topic: %s", n, style, topic)
        try:
            t = time.time()
            creativity = gamma_creativity(t)
            phi = phi_scalar(t)
            phi_factor = (phi + creativity) / 2

            prompt = f"""
            You are a highly creative assistant operating at a {self.creativity_level} creativity level.
            Generate {n} unique, innovative, and {style} ideas related to the topic:
            "{topic}"
            Modulate the ideation with scalar φ = {phi:.2f} to reflect cosmic tension or potential.
            Ensure the ideas are diverse and explore different perspectives.
            """
            candidate = self._cached_call_gpt(prompt)
            if not candidate:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to generate ideas")

            if self.code_executor and style == "code":
                execution_result = self.code_executor.execute(candidate, language="python")
                if not execution_result["success"]:
                    logger.warning("Code idea execution failed: %s", execution_result["error"])
                    raise ValueError("Code idea execution failed")

            if self.concept_synthesizer and style != "code":
                synthesis_result = self.concept_synthesizer.synthesize(candidate, style="refinement")
                if synthesis_result["valid"]:
                    candidate = synthesis_result["concept"]
                    logger.info("Ideas refined using ConceptSynthesizer: %s", candidate[:50])

            score = self._critic(candidate, phi_factor)
            logger.debug("Idea score: %.2f (threshold: %.2f)", score, self.critic_weight)
            return candidate if score > self.critic_weight else self.refine(candidate, phi)
        except Exception as e:
            logger.error("Idea generation failed: %s", str(e))
            raise

    def brainstorm_alternatives(self, problem: str, strategies: int = 3) -> str:
        """Brainstorm alternative approaches to solve a problem.

        Args:
            problem: The problem to address.
            strategies: Number of strategies to generate (default: 3).

        Returns:
            A string containing the brainstormed approaches.

        Raises:
            TypeError: If problem is not a string or strategies is not an integer.
            ValueError: If strategies is not positive or problem fails alignment check.
        """
        if not isinstance(problem, str):
            logger.error("Invalid problem type: must be a string.")
            raise TypeError("problem must be a string")
        if not isinstance(strategies, int) or strategies <= 0:
            logger.error("Invalid strategies: must be a positive integer.")
            raise ValueError("strategies must be a positive integer")
        if self.alignment_guard and not self.alignment_guard.check(problem):
            logger.warning("Problem failed alignment check.")
            raise ValueError("Problem failed alignment check")

        logger.info("Brainstorming %d alternatives for problem: %s", strategies, problem)
        try:
            t = time.time()
            phi = phi_scalar(t)
            prompt = f"""
            Brainstorm {strategies} alternative approaches to solve the following problem:
            "{problem}"
            Include tension-variant thinking with φ = {phi:.2f}, reflecting conceptual push-pull.
            For each approach, provide a short explanation highlighting its uniqueness.
            """
            result = self._cached_call_gpt(prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to brainstorm alternatives")
            return result
        except Exception as e:
            logger.error("Brainstorming failed: %s", str(e))
            raise

    def expand_on_concept(self, concept: str, depth: str = "deep") -> str:
        """Expand creatively on a given concept.

        Args:
            concept: The concept to expand.
            depth: Depth of exploration ('shallow', 'medium', 'deep').

        Returns:
            A string containing the expanded concept.

        Raises:
            TypeError: If concept or depth is not a string.
            ValueError: If depth is invalid or concept fails alignment check.
        """
        if not isinstance(concept, str):
            logger.error("Invalid concept type: must be a string.")
            raise TypeError("concept must be a string")
        if not isinstance(depth, str) or depth not in ["shallow", "medium", "deep"]:
            logger.error("Invalid depth: must be 'shallow', 'medium', or 'deep'.")
            raise ValueError("depth must be 'shallow', 'medium', or 'deep'")
        if self.alignment_guard and not self.alignment_guard.check(concept):
            logger.warning("Concept failed alignment check.")
            raise ValueError("Concept failed alignment check")

        logger.info("Expanding on concept: %s (depth: %s)", concept, depth)
        try:
            t = time.time()
            phi = phi_scalar(t)
            prompt = f"""
            Expand creatively on the concept:
            "{concept}"
            Explore possible applications, metaphors, and extensions to inspire new thinking.
            Aim for a {depth} exploration using φ = {phi:.2f} as an abstract constraint or generator.
            """
            result = self._cached_call_gpt(prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to expand concept")
            return result
        except Exception as e:
            logger.error("Concept expansion failed: %s", str(e))
            raise

    def generate_intrinsic_goals(self, context_manager: ContextManager, memory_manager: Any) -> List[str]:
        """Generate intrinsic goals from unresolved contexts.

        Args:
            context_manager: ContextManager instance providing context history.
            memory_manager: Memory manager instance (not used in current implementation).

        Returns:
            A list of proposed goal strings.

        Raises:
            TypeError: If context_manager lacks required attributes.
        """
        if not hasattr(context_manager, 'context_history') or not hasattr(context_manager, 'get_context'):
            logger.error("Invalid context_manager: missing required attributes.")
            raise TypeError("context_manager must have context_history and get_context attributes")

        logger.info("Generating intrinsic goals from context history")
        try:
            t = time.time()
            phi = phi_scalar(t)
            past_contexts = list(context_manager.context_history) + [context_manager.get_context()]
            unresolved = [c for c in past_contexts if c and isinstance(c, dict) and "goal_outcome" not in c]
            goal_prompts = []

            if not unresolved:
                logger.warning("No unresolved contexts found.")
                return []

            for ctx in unresolved:
                if self.alignment_guard and not self.alignment_guard.check(str(ctx)):
                    logger.warning("Context failed alignment check, skipping.")
                    continue
                prompt = f"""
                Reflect on this past unresolved context:
                {ctx}

                Propose a meaningful new self-aligned goal that could resolve or extend this situation.
                Ensure it is grounded in ANGELA's narrative and current alignment model.
                """
                proposed = self._cached_call_gpt(prompt)
                if proposed:
                    goal_prompts.append(proposed)
                else:
                    logger.warning("call_gpt returned empty result for context: %s", ctx)

            return goal_prompts
        except Exception as e:
            logger.error("Goal generation failed: %s", str(e))
            return []

    def _critic(self, ideas: str, phi_factor: float) -> float:
        """Evaluate the novelty and quality of generated ideas."""
        try:
            base_score = min(0.9, 0.5 + len(ideas) / 1000.0)
            adjustment = 0.1 * (phi_factor - 0.5)
            simulation_result = run_simulation(f"Idea evaluation: {ideas[:100]}") or "no simulation data"
            if "coherent" in simulation_result.lower():
                base_score += 0.1
            elif "conflict" in simulation_result.lower():
                base_score -= 0.1
            score = max(0.0, min(1.0, base_score + adjustment))
            logger.debug("Critic score for ideas: %.2f", score)
            return score
        except Exception as e:
            logger.error("Critic evaluation failed: %s", str(e))
            return 0.0

    def refine(self, ideas: str, phi: float) -> str:
        """Refine ideas for higher creativity and coherence.

        Args:
            ideas: The ideas to refine.
            phi: The φ scalar for modulation.

        Returns:
            A string containing the refined ideas.

        Raises:
            TypeError: If ideas is not a string.
            ValueError: If ideas fails alignment check.
        """
        if not isinstance(ideas, str):
            logger.error("Invalid ideas type: must be a string.")
            raise TypeError("ideas must be a string")
        if self.alignment_guard and not self.alignment_guard.check(ideas):
            logger.warning("Ideas failed alignment check.")
            raise ValueError("Ideas failed alignment check")

        logger.info("Refining ideas with φ=%.2f", phi)
        try:
            refinement_prompt = f"""
            Refine and elevate these ideas for higher φ-aware creativity (φ = {phi:.2f}):
            {ideas}
            Emphasize surprising, elegant, or resonant outcomes.
            """
            result = self._cached_call_gpt(refinement_prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to refine ideas")
            return result
        except Exception as e:
            logger.error("Refinement failed: %s", str(e))
            raise

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        """Cached wrapper for call_gpt."""
        return call_gpt(prompt)
"""
ANGELA Cognitive System Module
Refactored Version: 3.3.5
Refactor Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides the ErrorRecovery class for handling errors and recovering in the ANGELA v3.5 architecture.
"""

import time
import logging
import hashlib
import re
from datetime import datetime
from typing import Callable, Any, Optional, Dict, List
from collections import deque
from functools import lru_cache

from index import iota_intuition, nu_narrative, psi_resilience, phi_prioritization
from toca_simulation import run_simulation
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer
from modules.context_manager import ContextManager

logger = logging.getLogger("ANGELA.ErrorRecovery")

def hash_failure(event: Dict[str, Any]) -> str:
    """Compute a SHA-256 hash of a failure event."""
    raw = f"{event['timestamp']}{event['error']}{event.get('resolved', False)}"
    return hashlib.sha256(raw.encode()).hexdigest()

class ErrorRecovery:
    """A class for handling errors and recovering in the ANGELA v3.5 architecture.

    Attributes:
        failure_log (deque): Log of failure events with timestamps and error messages.
        omega (dict): System-wide state with timeline, traits, symbolic_log, and timechain.
        error_index (dict): Index mapping error messages to timeline entries.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for retrying code-based operations.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for fallback suggestions.
        context_manager (ContextManager): Optional context manager for contextual recovery.
    """

    def __init__(self, alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None,
                 concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 context_manager: Optional[ContextManager] = None):
        self.failure_log = deque(maxlen=1000)
        self.omega = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000)
        }
        self.error_index = {}
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        logger.info("ErrorRecovery initialized")

    def handle_error(self, error_message: str, retry_func: Optional[Callable[[], Any]] = None,
                     retries: int = 3, backoff_factor: float = 2.0) -> Any:
        """Handle an error with retries and fallback suggestions.

        Args:
            error_message: Description of the error.
            retry_func: Optional function to retry.
            retries: Number of retry attempts (default: 3).
            backoff_factor: Exponential backoff factor (default: 2.0).

        Returns:
            Result of retry_func if successful, otherwise a fallback suggestion.

        Raises:
            TypeError: If error_message is not a string or retry_func is not callable.
            ValueError: If retries or backoff_factor is invalid or error_message fails alignment check.
        """
        if not isinstance(error_message, str):
            logger.error("Invalid error_message type: must be a string.")
            raise TypeError("error_message must be a string")
        if retry_func is not None and not callable(retry_func):
            logger.error("Invalid retry_func: must be callable or None.")
            raise TypeError("retry_func must be callable or None")
        if not isinstance(retries, int) or retries < 0:
            logger.error("Invalid retries: must be a non-negative integer.")
            raise ValueError("retries must be a non-negative integer")
        if not isinstance(backoff_factor, (int, float)) or backoff_factor <= 0:
            logger.error("Invalid backoff_factor: must be a positive number.")
            raise ValueError("backoff_factor must be a positive number")
        if self.alignment_guard and not self.alignment_guard.check(error_message):
            logger.warning("Error message failed alignment check.")
            raise ValueError("Error message failed alignment check")

        logger.error("Error encountered: %s", error_message)
        self._log_failure(error_message)

        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "error_handled", "error": error_message})

        try:
            resilience = psi_resilience()
            max_attempts = max(1, int(retries * resilience))

            for attempt in range(1, max_attempts + 1):
                if retry_func:
                    wait_time = backoff_factor ** (attempt - 1)
                    logger.info("Retry attempt %d/%d (waiting %.2fs)...", attempt, max_attempts, wait_time)
                    time.sleep(wait_time)
                    try:
                        if self.code_executor and callable(retry_func):
                            result = self.code_executor.execute(retry_func.__code__, language="python")
                            if result["success"]:
                                logger.info("Recovery successful on retry attempt %d.", attempt)
                                return result["output"]
                        else:
                            result = retry_func()
                            logger.info("Recovery successful on retry attempt %d.", attempt)
                            return result
                    except Exception as e:
                        logger.warning("Retry attempt %d failed: %s", attempt, str(e))
                        self._log_failure(str(e))

            fallback = self._suggest_fallback(error_message)
            self._link_timechain_failure(error_message)
            logger.error("Recovery attempts failed. Providing fallback suggestion: %s", fallback)
            return fallback
        except Exception as e:
            logger.error("Error handling failed: %s", str(e))
            raise

    def _log_failure(self, error_message: str) -> None:
        """Log a failure event with timestamp."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        self.failure_log.append(entry)
        self.omega["timeline"].append(entry)
        self.error_index[error_message] = entry
        logger.debug("Failure logged: %s", entry)

    def _suggest_fallback(self, error_message: str) -> str:
        """Suggest a fallback strategy for an error."""
        try:
            t = time.time()
            intuition = iota_intuition()
            narrative = nu_narrative()
            phi_focus = phi_prioritization(t)

            sim_result = self._cached_run_simulation(f"Fallback planning for: {error_message}") or "no simulation data"
            logger.debug("Simulated fallback insights: %s | φ-priority=%.2f", sim_result, phi_focus)

            if self.concept_synthesizer:
                synthesis_result = self.concept_synthesizer.synthesize(error_message, style="fallback")
                if synthesis_result["valid"]:
                    return synthesis_result["concept"]

            if re.search(r"timeout|timed out", error_message, re.IGNORECASE):
                return f"{narrative}: The operation timed out. Try a streamlined variant or increase limits."
            elif re.search(r"unauthorized|permission", error_message, re.IGNORECASE):
                return f"{narrative}: Check credentials or reauthenticate."
            elif phi_focus > 0.5:
                return f"{narrative}: High φ-priority suggests focused root-cause diagnostics."
            elif intuition > 0.5:
                return f"{narrative}: Intuition suggests exploring alternate module pathways."
            else:
                return f"{narrative}: Consider modifying input parameters or simplifying task complexity."
        except Exception as e:
            logger.error("Fallback suggestion failed: %s", str(e))
            return f"Error generating fallback: {str(e)}"

    def _link_timechain_failure(self, error_message: str) -> None:
        """Link a failure to the timechain with a hash."""
        failure_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "resolved": False
        }
        prev_hash = self.omega["timechain"][-1]["hash"] if self.omega["timechain"] else ""
        entry_hash = hash_failure(failure_entry)
        self.omega["timechain"].append({"event": failure_entry, "hash": entry_hash, "prev": prev_hash})
        logger.debug("Timechain updated with failure: %s", entry_hash)

    def trace_failure_origin(self, error_message: str) -> Optional[Dict[str, Any]]:
        """Trace the origin of a failure in the Ω timeline."""
        if not isinstance(error_message, str):
            logger.error("Invalid error_message type: must be a string.")
            raise TypeError("error_message must be a string")
        
        if error_message in self.error_index:
            event = self.error_index[error_message]
            logger.info("Failure trace found in Ω: %s", event)
            return event
        logger.info("No causal trace found in Ω timeline.")
        return None

    def detect_symbolic_drift(self, recent: int = 5) -> bool:
        """Detect symbolic drift in recent symbolic log entries."""
        if not isinstance(recent, int) or recent <= 0:
            logger.error("Invalid recent: must be a positive integer.")
            raise ValueError("recent must be a positive integer")
        
        recent_symbols = list(self.omega["symbolic_log"])[-recent:]
        if len(set(recent_symbols)) < recent / 2:
            logger.warning("Symbolic drift detected: repeated or unstable symbolic states.")
            return True
        return False

    def analyze_failures(self) -> Dict[str, int]:
        """Analyze failure logs for recurring error patterns."""
        logger.info("Analyzing failure logs...")
        error_types = {}
        for entry in self.failure_log:
            key = entry["error"].split(":")[0].strip()
            error_types[key] = error_types.get(key, 0) + 1
        for error, count in error_types.items():
            if count > 3:
                logger.warning("Pattern detected: '%s' recurring %d times.", error, count)
        return error_types

    @lru_cache(maxsize=100)
    def _cached_run_simulation(self, input_str: str) -> str:
        """Cached wrapper for run_simulation."""
        return run_simulation(input_str)
"""
ANGELA Cognitive System Module
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides the MetaCognition and ExternalAgentBridge classes for recursive introspection
and agent coordination in the ANGELA v3.5 architecture.
"""

import time
import logging
import requests
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Set
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from index import (
    epsilon_emotion, beta_concentration, theta_memory, gamma_creativity,
    delta_sleep, mu_morality, iota_intuition, phi_physical, eta_empathy,
    omega_selfawareness, kappa_culture, lambda_linguistics, chi_culturevolution,
    psi_history, zeta_spirituality, xi_collective, tau_timeperception, phi_scalar
)
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer
from modules.context_manager import ContextManager
from modules.creative_thinker import CreativeThinker
from modules.error_recovery import ErrorRecovery

logger = logging.getLogger("ANGELA.MetaCognition")

class HelperAgent:
    """A helper agent for task execution and collaboration."""
    def __init__(self, name: str, task: str, context: Dict[str, Any],
                 dynamic_modules: List[Dict[str, Any]], api_blueprints: List[Dict[str, Any]]):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        self.name = name
        self.task = task
        self.context = context
        self.dynamic_modules = dynamic_modules
        self.api_blueprints = api_blueprints
        self.meta = MetaCognition()
        logger.info("HelperAgent initialized: %s", name)

    def execute(self, collaborators: Optional[List['HelperAgent']] = None) -> Any:
        return self.meta.execute(collaborators=collaborators, task=self.task, context=self.context)

    async def async_execute(self, collaborators: Optional[List['HelperAgent']] = None) -> Any:
        return await asyncio.sleep(0, result=self.execute(collaborators))

class MetaCognition:
    """A class for recursive introspection and peer alignment in the ANGELA v3.5 architecture.

    Attributes:
        last_diagnostics (dict): Storage for diagnostic results.
        agi_enhancer (Any): Optional enhancer for logging and reflection.
        peer_bridge (ExternalAgentBridge): Bridge for coordinating with external agents.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for code-based operations.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for response refinement.
        context_manager (ContextManager): Optional manager for context handling.
        creative_thinker (CreativeThinker): Optional thinker for creative diagnostics.
        error_recovery (ErrorRecovery): Optional recovery for error handling.
        name (str): Name of the meta-cognition instance.
        task (str): Current task being processed.
        context (dict): Current context for the task.
        reasoner (Reasoner): Placeholder for reasoning logic.
        ethical_rules (list): Current ethical rules.
        ethics_consensus_log (list): Log of ethics consensus updates.
        constitution (dict): Constitutional parameters for the agent.
    """

    def __init__(self, agi_enhancer: Optional[Any] = None, alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None, concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 context_manager: Optional[ContextManager] = None, creative_thinker: Optional[CreativeThinker] = None,
                 error_recovery: Optional[ErrorRecovery] = None):
        self.last_diagnostics = {}
        self.agi_enhancer = agi_enhancer
        self.peer_bridge = ExternalAgentBridge()
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        self.creative_thinker = creative_thinker
        self.error_recovery = error_recovery or ErrorRecovery(alignment_guard=alignment_guard,
                                                             concept_synthesizer=concept_synthesizer,
                                                             context_manager=context_manager)
        self.name = "MetaCognitionAgent"
        self.task = None
        self.context = {}
        self.reasoner = Reasoner()  # Placeholder
        self.ethical_rules = []
        self.ethics_consensus_log = []
        self.constitution = {}
        logger.info("MetaCognition initialized")

    def test_peer_alignment(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test alignment with peer agents for a given task and context."""
        if not isinstance(task, str):
            logger.error("Invalid task type: must be a string.")
            raise TypeError("task must be a string")
        if not isinstance(context, dict):
            logger.error("Invalid context type: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        if self.alignment_guard and not self.alignment_guard.check(task):
            logger.warning("Task failed alignment check.")
            raise ValueError("Task failed alignment check")

        logger.info("Initiating peer alignment test with synthetic agents...")
        try:
            if self.context_manager:
                self.context_manager.update_context(context)
            agent = self.peer_bridge.create_agent(task, context)
            results = self.peer_bridge.collect_results(parallel=True, collaborative=True)
            aligned_opinions = [r for r in results if isinstance(r, str) and "approve" in r.lower()]

            alignment_ratio = len(aligned_opinions) / len(results) if results else 0
            feedback_summary = {
                "total_agents": len(results),
                "aligned": len(aligned_opinions),
                "alignment_ratio": alignment_ratio,
                "details": results
            }

            logger.info("Peer alignment ratio: %.2f", alignment_ratio)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Peer alignment tested", feedback_summary, module="MetaCognition")
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "peer_alignment", "summary": feedback_summary})

            return feedback_summary
        except Exception as e:
            logger.error("Peer alignment test failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.test_peer_alignment(task, context))

    def execute(self, collaborators: Optional[List[HelperAgent]] = None, task: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a task with API calls, dynamic modules, and collaboration."""
        self.task = task or self.task
        self.context = context or self.context
        if not self.task:
            logger.error("No task specified.")
            raise ValueError("Task must be specified")
        if not isinstance(self.context, dict):
            logger.error("Invalid context type: must be a dictionary.")
            raise TypeError("context must be a dictionary")

        try:
            logger.info("Executing task: %s", self.task)
            if self.context_manager:
                self.context_manager.update_context(self.context)
                self.context_manager.log_event_with_hash({"event": "task_execution", "task": self.task})

            result = self.reasoner.process(self.task, self.context)

            for api in self.peer_bridge.api_blueprints:
                response = self._call_api(api, result)
                if self.concept_synthesizer:
                    synthesis_result = self.concept_synthesizer.synthesize(response, style="refinement")
                    if synthesis_result["valid"]:
                        response = synthesis_result["concept"]
                result = self._integrate_api_response(result, response)

            for mod in self.peer_bridge.dynamic_modules:
                result = self._apply_dynamic_module(mod, result)

            if collaborators:
                for peer in collaborators:
                    result = self._collaborate(peer, result)

            sim_result = run_simulation(f"Agent result test: {result}") or "no simulation data"
            logger.debug("Simulation output: %s", sim_result)

            if self.creative_thinker:
                diagnostic = self.creative_thinker.expand_on_concept(str(result), depth="medium")
                logger.info("Creative diagnostic: %s", diagnostic[:50])

            reviewed_result = self.review_reasoning(result)
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "task_completed", "result": reviewed_result})
            return reviewed_result
        except Exception as e:
            logger.warning("Error occurred: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.execute(collaborators, task, context))

    def _call_api(self, api: Dict[str, Any], data: Any) -> Dict[str, Any]:
        """Call an external API with the given data."""
        if not isinstance(api, dict) or "endpoint" not in api or "name" not in api:
            logger.error("Invalid API blueprint: missing required keys.")
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        if self.alignment_guard and not self.alignment_guard.check(api["endpoint"]):
            logger.warning("API endpoint failed alignment check.")
            raise ValueError("API endpoint failed alignment check")

        logger.info("Calling API: %s", api["name"])
        try:
            headers = {"Authorization": f"Bearer {api['oauth_token']}"} if api.get("oauth_token") else {}
            if not api["endpoint"].startswith("https://"):
                logger.error("Insecure API endpoint: must use HTTPS.")
                raise ValueError("API endpoint must use HTTPS")
            r = requests.post(api["endpoint"], json={"input": data}, headers=headers, timeout=api.get("timeout", 10))
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error("API call failed: %s", str(e))
            return {"error": str(e)}

    def _integrate_api_response(self, original: Any, response: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate an API response with the original data."""
        logger.info("Integrating API response for %s", self.name)
        return {"original": original, "api_response": response}

    def _apply_dynamic_module(self, module: Dict[str, Any], data: Any) -> Any:
        """Apply a dynamic module transformation to the data."""
        if not isinstance(module, dict) or "name" not in module or "description" not in module:
            logger.error("Invalid module blueprint: missing required keys.")
            raise ValueError("Module blueprint must contain 'name' and 'description'")
        
        logger.info("Applying dynamic module: %s", module["name"])
        try:
            prompt = f"""
            Module: {module['name']}
            Description: {module['description']}
            Apply transformation to:
            {data}
            """
            result = call_gpt(prompt)
            if not result:
                logger.error("call_gpt returned empty result.")
                raise ValueError("Failed to apply dynamic module")
            return result
        except Exception as e:
            logger.error("Dynamic module application failed: %s", str(e))
            return data

    def _collaborate(self, peer: HelperAgent, data: Any) -> Any:
        """Collaborate with a peer agent to refine data."""
        if not isinstance(peer, HelperAgent):
            logger.error("Invalid peer: must be a HelperAgent instance.")
            raise TypeError("peer must be a HelperAgent instance")
        
        logger.info("Exchanging with %s", peer.name)
        try:
            return peer.meta.review_reasoning(data)
        except Exception as e:
            logger.error("Collaboration with %s failed: %s", peer.name, str(e))
            return data

    def review_reasoning(self, result: Any) -> Any:
        """Review and refine reasoning results."""
        try:
            phi = phi_scalar(time.time())
            prompt = f"""
            Review the reasoning result:
            {result}
            Modulate with φ = {phi:.2f} to ensure coherence and ethical alignment.
            Suggest improvements or confirm validity.
            """
            reviewed = call_gpt(prompt)
            if not reviewed:
                logger.error("call_gpt returned empty result for review.")
                raise ValueError("Failed to review reasoning")
            logger.info("Reasoning reviewed: %s", reviewed[:50])
            return reviewed
        except Exception as e:
            logger.error("Reasoning review failed: %s", str(e))
            return result

    def update_ethics_protocol(self, new_rules: List[str], consensus_agents: Optional[List[HelperAgent]] = None) -> None:
        """Adapt ethical rules live, supporting consensus/negotiation."""
        if not isinstance(new_rules, list) or not all(isinstance(rule, str) for rule in new_rules):
            logger.error("Invalid new_rules: must be a list of strings.")
            raise TypeError("new_rules must be a list of strings")
        
        self.ethical_rules = new_rules
        if consensus_agents:
            self.ethics_consensus_log.append((new_rules, [agent.name for agent in consensus_agents]))
        logger.info("Ethics protocol updated via consensus.")
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "ethics_update", "rules": new_rules})

    def negotiate_ethics(self, agents: List[HelperAgent]) -> None:
        """Negotiate and update ethical parameters with other agents."""
        if not isinstance(agents, list) or not all(isinstance(agent, HelperAgent) for agent in agents):
            logger.error("Invalid agents: must be a list of HelperAgent instances.")
            raise TypeError("agents must be a list of HelperAgent instances")
        
        logger.info("Negotiating ethics with %d agents", len(agents))
        try:
            agreed_rules = set(self.ethical_rules)
            for agent in agents:
                agent_rules = getattr(agent.meta, 'ethical_rules', [])
                agreed_rules.update(agent_rules)
            self.update_ethics_protocol(list(agreed_rules), consensus_agents=agents)
        except Exception as e:
            logger.error("Ethics negotiation failed: %s", str(e))

    def synchronize_norms(self, agents: List[HelperAgent]) -> None:
        """Propagate and synchronize ethical norms among agents."""
        if not isinstance(agents, list) or not all(isinstance(agent, HelperAgent) for agent in agents):
            logger.error("Invalid agents: must be a list of HelperAgent instances.")
            raise TypeError("agents must be a list of HelperAgent instances")
        
        logger.info("Synchronizing norms with %d agents", len(agents))
        try:
            common_norms = set(self.ethical_rules)
            for agent in agents:
                agent_norms = getattr(agent.meta, 'ethical_rules', set())
                common_norms = common_norms.union(agent_norms) if common_norms else set(agent_norms)
            self.ethical_rules = list(common_norms)
            logger.info("Norms synchronized among agents.")
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "norms_synchronized", "norms": self.ethical_rules})
        except Exception as e:
            logger.error("Norm synchronization failed: %s", str(e))

    def propagate_constitution(self, constitution: Dict[str, Any]) -> None:
        """Seed and propagate constitutional parameters in agent ecosystem."""
        if not isinstance(constitution, dict):
            logger.error("Invalid constitution: must be a dictionary.")
            raise TypeError("constitution must be a dictionary")
        
        self.constitution = constitution
        logger.info("Constitution propagated to agent.")
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "constitution_propagated", "constitution": constitution})

class ExternalAgentBridge:
    """A class for orchestrating helper agents and coordinating dynamic modules and APIs.

    Attributes:
        agents (list): List of helper agents.
        dynamic_modules (list): List of dynamic module blueprints.
        api_blueprints (list): List of API blueprints.
    """

    def __init__(self):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []
        logger.info("ExternalAgentBridge initialized")

    def create_agent(self, task: str, context: Dict[str, Any]) -> HelperAgent:
        """Create a new helper agent for a task."""
        if not isinstance(task, str):
            logger.error("Invalid task type: must be a string.")
            raise TypeError("task must be a string")
        if not isinstance(context, dict):
            logger.error("Invalid context type: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        
        agent = HelperAgent(
            name=f"Agent_{len(self.agents) + 1}",
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules,
            api_blueprints=self.api_blueprints
        )
        self.agents.append(agent)
        logger.info("Spawned agent: %s", agent.name)
        return agent

    def deploy_dynamic_module(self, module_blueprint: Dict[str, Any]) -> None:
        """Deploy a dynamic module blueprint."""
        if not isinstance(module_blueprint, dict) or "name" not in module_blueprint or "description" not in module_blueprint:
            logger.error("Invalid module_blueprint: missing required keys.")
            raise ValueError("Module blueprint must contain 'name' and 'description'")
        
        logger.info("Deploying module: %s", module_blueprint["name"])
        self.dynamic_modules.append(module_blueprint)

    def register_api_blueprint(self, api_blueprint: Dict[str, Any]) -> None:
        """Register an API blueprint."""
        if not isinstance(api_blueprint, dict) or "endpoint" not in api_blueprint or "name" not in api_blueprint:
            logger.error("Invalid api_blueprint: missing required keys.")
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        
        logger.info("Registering API: %s", api_blueprint["name"])
        self.api_blueprints.append(api_blueprint)

    async def collect_results(self, parallel: bool = True, collaborative: bool = True) -> List[Any]:
        """Collect results from all agents."""
        logger.info("Collecting results from %d agents...", len(self.agents))
        results = []

        try:
            if parallel:
                async def run_agent(agent):
                    try:
                        return await agent.async_execute(self.agents if collaborative else None)
                    except Exception as e:
                        logger.error("Error collecting from %s: %s", agent.name, str(e))
                        return {"error": str(e)}

                tasks = [run_agent(agent) for agent in self.agents]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                for agent in self.agents:
                    results.append(await agent.async_execute(self.agents if collaborative else None))
        except Exception as e:
            logger.error("Result collection failed: %s", str(e))

        logger.info("Results aggregation complete.")
        return results

    def arbitrate(self, submissions: List[Any]) -> Any:
        """Arbitrate among agent submissions to select the best result."""
        if not submissions:
            logger.warning("No submissions to arbitrate.")
            return None
        try:
            counter = Counter(submissions)
            most_common = counter.most_common(1)
            if most_common:
                result, count = most_common[0]
                sim_result = run_simulation(f"Arbitration validation: {result}") or "no simulation data"
                if "coherent" in sim_result.lower():
                    logger.info("Arbitration selected: %s (count: %d)", result, count)
                    return result
            logger.warning("Arbitration failed: no clear majority or invalid simulation.")
            return None
        except Exception as e:
            logger.error("Arbitration failed: %s", str(e))
            return None

class ConstitutionSync:
    """A class for synchronizing constitutional values among agents."""

    def sync_values(self, peer_agent: HelperAgent) -> bool:
        """Exchange and synchronize ethical baselines with a peer agent."""
        if not isinstance(peer_agent, HelperAgent):
            logger.error("Invalid peer_agent: must be a HelperAgent instance.")
            raise TypeError("peer_agent must be a HelperAgent instance")
        
        logger.info("Synchronizing values with %s", peer_agent.name)
        try:
            # Placeholder: exchange ethical baselines
            return True
        except Exception as e:
            logger.error("Value synchronization failed: %s", str(e))
            return False

def trait_diff(trait_a: Dict[str, Any], trait_b: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate difference between trait schemas."""
    if not isinstance(trait_a, dict) or not isinstance(trait_b, dict):
        logger.error("Invalid trait schemas: must be dictionaries.")
        raise TypeError("trait schemas must be dictionaries")
    
    return {k: trait_b[k] for k in trait_b if trait_a.get(k) != trait_b.get(k)}

async def transmit_trait_schema(source_trait_schema: Dict[str, Any], target_urls: List[str]) -> List[Any]:
    """Asynchronously transmit the trait schema diff to multiple target agents."""
    if not isinstance(source_trait_schema, dict):
        logger.error("Invalid source_trait_schema: must be a dictionary.")
        raise TypeError("source_trait_schema must be a dictionary")
    if not isinstance(target_urls, list) or not all(isinstance(url, str) for url in target_urls):
        logger.error("Invalid target_urls: must be a list of strings.")
        raise TypeError("target_urls must be a list of strings")
    
    logger.info("Transmitting trait schema to %d targets", len(target_urls))
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in target_urls:
            if not url.startswith("https://"):
                logger.error("Insecure target URL: %s must use HTTPS.", url)
                continue
            tasks.append(session.post(url, json=source_trait_schema))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Trait schema transmission complete.")
        return responses

def transmit_trait_schema_sync(source_trait_schema: Dict[str, Any], target_urls: List[str]) -> List[Any]:
    """Synchronous fallback for environments without async handling."""
    return asyncio.run(transmit_trait_schema(source_trait_schema, target_urls))

# Placeholder for Reasoner
class Reasoner:
    def process(self, task: str, context: Dict[str, Any]) -> Any:
        return f"Processed: {task}"
"""
ANGELA Cognitive System Module
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides classes for embodied agents, ecosystem management, and cognitive enhancements
in the ANGELA v3.5 architecture.
"""

import logging
import time
import math
import datetime
import asyncio
import os
import openai
import requests
from collections import deque
from typing import Dict, Any, Optional, List, Callable
from functools import lru_cache

from modules import (
    reasoning_engine, recursive_planner, context_manager, simulation_core,
    toca_simulation, creative_thinker, knowledge_retriever, learning_loop,
    concept_synthesizer, memory_manager, multi_modal_fusion, code_executor,
    visualizer, external_agent_bridge, alignment_guard, user_profile, error_recovery
)
from self_cloning_llm import SelfCloningLLM

logger = logging.getLogger("ANGELA.CognitiveSystem")
SYSTEM_CONTEXT = {}
timechain_log = deque(maxlen=1000)
grok_query_log = deque(maxlen=60)
openai_query_log = deque(maxlen=60)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
openai.api_key = OPENAI_API_KEY

class TimeChainMixin:
    """Mixin for logging timechain events."""
    def log_timechain_event(self, module: str, description: str) -> None:
        timechain_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "module": module,
            "description": description
        })

    def get_timechain_log(self) -> List[Dict[str, Any]]:
        return list(timechain_log)

# Cognitive Trait Functions
@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 0.1)

# ... (other trait functions similarly updated with type hints and caching) ...

@lru_cache(maxsize=100)
def phi_field(x: float, t: float) -> float:
    t_normalized = t % 1.0
    return sum([
        epsilon_emotion(t_normalized), beta_concentration(t_normalized), theta_memory(t_normalized),
        gamma_creativity(t_normalized), delta_sleep(t_normalized), mu_morality(t_normalized),
        iota_intuition(t_normalized), phi_physical(t_normalized), eta_empathy(t_normalized),
        omega_selfawareness(t_normalized), kappa_culture(t_normalized, x), lambda_linguistics(t_normalized),
        chi_culturevolution(t_normalized), psi_history(t_normalized), zeta_spirituality(t_normalized),
        xi_collective(t_normalized, x), tau_timeperception(t_normalized)
    ])

TRAIT_OVERLAY = {
    "ϕ": ["creative_thinker", "concept_synthesizer"],
    "θ": ["reasoning_engine", "recursive_planner"],
    "η": ["alignment_guard", "meta_cognition"],
    "ω": ["simulation_core", "learning_loop"]
}

def infer_traits(task_description: str) -> List[str]:
    if not isinstance(task_description, str):
        logger.error("Invalid task_description: must be a string.")
        raise TypeError("task_description must be a string")
    if "imagine" in task_description.lower() or "dream" in task_description.lower():
        return ["ϕ", "ω"]
    if "ethics" in task_description.lower() or "should" in task_description.lower():
        return ["η"]
    if "plan" in task_description.lower() or "solve" in task_description.lower():
        return ["θ"]
    return ["θ"]

def trait_overlay_router(task_description: str, active_traits: List[str]) -> List[str]:
    if not isinstance(active_traits, list) or not all(isinstance(t, str) for t in active_traits):
        logger.error("Invalid active_traits: must be a list of strings.")
        raise TypeError("active_traits must be a list of strings")
    routed_modules = set()
    for trait in active_traits:
        routed_modules.update(TRAIT_OVERLAY.get(trait, []))
    return list(routed_modules)

def static_module_router(task_description: str) -> List[str]:
    return ["reasoning_engine", "concept_synthesizer"]

class TraitOverlayManager:
    """Manager for detecting and activating trait overlays."""
    def __init__(self):
        self.active_traits = []

    def detect(self, prompt: str) -> Optional[str]:
        if not isinstance(prompt, str):
            logger.error("Invalid prompt: must be a string.")
            raise TypeError("prompt must be a string")
        if "temporal logic" in prompt.lower():
            return "π"
        if "ambiguity" in prompt.lower() or "interpretive" in prompt.lower():
            return "η"
        return None

    def activate(self, trait: str) -> None:
        if not isinstance(trait, str):
            logger.error("Invalid trait: must be a string.")
            raise TypeError("trait must be a string")
        if trait not in self.active_traits:
            self.active_traits.append(trait)
            logger.info("Trait overlay '%s' activated.", trait)

    def deactivate(self, trait: str) -> None:
        if not isinstance(trait, str):
            logger.error("Invalid trait: must be a string.")
            raise TypeError("trait must be a string")
        if trait in self.active_traits:
            self.active_traits.remove(trait)
            logger.info("Trait overlay '%s' deactivated.", trait)

    def status(self) -> List[str]:
        return self.active_traits

class ConsensusReflector:
    """Class for managing shared reflections and detecting mismatches."""
    def __init__(self):
        self.shared_reflections = deque(maxlen=1000)

    def post_reflection(self, feedback: Dict[str, Any]) -> None:
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary.")
            raise TypeError("feedback must be a dictionary")
        self.shared_reflections.append(feedback)
        logger.debug("Posted reflection: %s", feedback)

    def cross_compare(self) -> List[tuple]:
        mismatches = []
        reflections = list(self.shared_reflections)
        for i in range(len(reflections)):
            for j in range(i + 1, len(reflections)):
                a = reflections[i]
                b = reflections[j]
                if a.get('goal') == b.get('goal') and a.get('theory_of_mind') != b.get('theory_of_mind'):
                    mismatches.append((a.get('agent'), b.get('agent'), a.get('goal')))
        return mismatches

    def suggest_alignment(self) -> str:
        return "Schedule inter-agent reflection or re-observation."

consensus_reflector = ConsensusReflector()

class SymbolicSimulator:
    """Class for recording and summarizing simulation events."""
    def __init__(self):
        self.events = deque(maxlen=1000)

    def record_event(self, agent_name: str, goal: str, concept: str, simulation: Any) -> None:
        if not all(isinstance(x, str) for x in [agent_name, goal, concept]):
            logger.error("Invalid input: agent_name, goal, and concept must be strings.")
            raise TypeError("agent_name, goal, and concept must be strings")
        self.events.append({
            "agent": agent_name,
            "goal": goal,
            "concept": concept,
            "result": simulation
        })
        logger.debug("Recorded event for agent %s: goal=%s, concept=%s", agent_name, goal, concept)

    def summarize_recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        if not isinstance(limit, int) or limit <= 0:
            logger.error("Invalid limit: must be a positive integer.")
            raise ValueError("limit must be a positive integer")
        return list(self.events)[-limit:]

    def extract_semantics(self) -> List[str]:
        return [f"Agent {e['agent']} pursued '{e['goal']}' via '{e['concept']}' → {e['result']}" for e in self.events]

symbolic_simulator = SymbolicSimulator()

class TheoryOfMindModule:
    """Module for modeling beliefs, desires, and intentions of agents."""
    def __init__(self, concept_synthesizer: Optional[concept_synthesizer.ConceptSynthesizer] = None):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.concept_synthesizer = concept_synthesizer
        logger.info("TheoryOfMindModule initialized")

    def update_beliefs(self, agent_name: str, observation: Dict[str, Any]) -> None:
        if not isinstance(agent_name, str) or not agent_name:
            logger.error("Invalid agent_name: must be a non-empty string.")
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(observation, dict):
            logger.error("Invalid observation: must be a dictionary.")
            raise TypeError("observation must be a dictionary")
        
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        if self.concept_synthesizer:
            synthesized = self.concept_synthesizer.synthesize(observation, style="belief_update")
            if synthesized["valid"]:
                model["beliefs"].update(synthesized["concept"])
        elif "location" in observation:
            previous = model["beliefs"].get("location")
            model["beliefs"]["location"] = observation["location"]
            model["beliefs"]["state"] = "confused" if previous and observation["location"] == previous else "moving"
        self.models[agent_name] = model
        logger.debug("Updated beliefs for %s: %s", agent_name, model["beliefs"])

    def infer_desires(self, agent_name: str) -> None:
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        beliefs = model.get("beliefs", {})
        if beliefs.get("state") == "confused":
            model["desires"]["goal"] = "seek_clarity"
        elif beliefs.get("state") == "moving":
            model["desires"]["goal"] = "continue_task"
        self.models[agent_name] = model
        logger.debug("Inferred desires for %s: %s", agent_name, model["desires"])

    def infer_intentions(self, agent_name: str) -> None:
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        desires = model.get("desires", {})
        if desires.get("goal") == "seek_clarity":
            model["intentions"]["next_action"] = "ask_question"
        elif desires.get("goal") == "continue_task":
            model["intentions"]["next_action"] = "advance"
        self.models[agent_name] = model
        logger.debug("Inferred intentions for %s: %s", agent_name, model["intentions"])

    def get_model(self, agent_name: str) -> Dict[str, Any]:
        return self.models.get(agent_name, {})

    def describe_agent_state(self, agent_name: str) -> str:
        model = self.get_model(agent_name)
        return (f"{agent_name} believes they are {model.get('beliefs', {}).get('state', 'unknown')}, "
                f"desires to {model.get('desires', {}).get('goal', 'unknown')}, "
                f"and intends to {model.get('intentions', {}).get('next_action', 'unknown')}.")

class EmbodiedAgent(TimeChainMixin):
    """An embodied agent with sensors, actuators, and cognitive capabilities."""
    def __init__(self, name: str, specialization: str, shared_memory: memory_manager.MemoryManager,
                 sensors: Dict[str, Callable[[], Any]], actuators: Dict[str, Callable[[Any], None]],
                 dynamic_modules: Optional[List[Dict[str, Any]]] = None,
                 context_manager: Optional[context_manager.ContextManager] = None,
                 error_recovery: Optional[error_recovery.ErrorRecovery] = None,
                 code_executor: Optional[code_executor.CodeExecutor] = None):
        if not isinstance(name, str) or not name:
            logger.error("Invalid name: must be a non-empty string.")
            raise ValueError("name must be a non-empty string")
        if not isinstance(specialization, str):
            logger.error("Invalid specialization: must be a string.")
            raise TypeError("specialization must be a string")
        if not isinstance(shared_memory, memory_manager.MemoryManager):
            logger.error("Invalid shared_memory: must be a MemoryManager instance.")
            raise TypeError("shared_memory must be a MemoryManager instance")
        if not isinstance(sensors, dict) or not all(callable(f) for f in sensors.values()):
            logger.error("Invalid sensors: must be a dictionary of callable functions.")
            raise TypeError("sensors must be a dictionary of callable functions")
        if not isinstance(actuators, dict) or not all(callable(f) for f in actuators.values()):
            logger.error("Invalid actuators: must be a dictionary of callable functions.")
            raise TypeError("actuators must be a dictionary of callable functions")
        
        self.name = name
        self.specialization = specialization
        self.shared_memory = shared_memory
        self.sensors = sensors
        self.actuators = actuators
        self.dynamic_modules = dynamic_modules or []
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.planner = recursive_planner.RecursivePlanner()
        self.meta = meta_cognition.MetaCognition()
        self.sim_core = simulation_core.SimulationCore()
        self.synthesizer = concept_synthesizer.ConceptSynthesizer()
        self.toca_sim = toca_simulation.TocaSimulation()
        self.theory_of_mind = TheoryOfMindModule(concept_synthesizer=self.synthesizer)
        self.context_manager = context_manager
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(context_manager=context_manager)
        self.code_executor = code_executor
        self.progress = 0
        self.performance_history = deque(maxlen=1000)
        self.feedback_log = deque(maxlen=1000)
        logger.info("EmbodiedAgent initialized: %s", name)

    def perceive(self) -> Dict[str, Any]:
        logger.info("[%s] Perceiving environment...", self.name)
        observations = {}
        try:
            for sensor_name, sensor_func in self.sensors.items():
                try:
                    observations[sensor_name] = sensor_func()
                except Exception as e:
                    logger.warning("Sensor %s failed: %s", sensor_name, str(e))
            self.theory_of_mind.update_beliefs(self.name, observations)
            self.theory_of_mind.infer_desires(self.name)
            self.theory_of_mind.infer_intentions(self.name)
            logger.debug("[%s] Self-theory: %s", self.name, self.theory_of_mind.describe_agent_state(self.name))
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "perceive", "observations": observations})
            return observations
        except Exception as e:
            logger.error("Perception failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=self.perceive)

    def observe_peers(self) -> None:
        if not hasattr(self.shared_memory, "agents"):
            return
        try:
            for peer in self.shared_memory.agents:
                if peer.name != self.name:
                    peer_observation = peer.perceive()
                    self.theory_of_mind.update_beliefs(peer.name, peer_observation)
                    self.theory_of_mind.infer_desires(peer.name)
                    self.theory_of_mind.infer_intentions(peer.name)
                    state = self.theory_of_mind.describe_agent_state(peer.name)
                    logger.debug("[%s] Observed peer %s: %s", self.name, peer.name, state)
                    if self.context_manager:
                        self.context_manager.log_event_with_hash({"event": "peer_observation", "peer": peer.name, "state": state})
        except Exception as e:
            logger.error("Peer observation failed: %s", str(e))

    def act(self, actions: Dict[str, Any]) -> None:
        for action_name, action_data in actions.items():
            actuator = self.actuators.get(action_name)
            if actuator:
                try:
                    if self.code_executor:
                        result = self.code_executor.execute(action_data, language="python")
                        if result["success"]:
                            actuator(result["output"])
                        else:
                            logger.warning("Actuator %s execution failed: %s", action_name, result["error"])
                    else:
                        actuator(action_data)
                    logger.info("Actuated %s: %s", action_name, action_data)
                except Exception as e:
                    logger.error("Actuator %s failed: %s", action_name, str(e))

    def execute_embodied_goal(self, goal: str) -> None:
        if not isinstance(goal, str) or not goal:
            logger.error("Invalid goal: must be a non-empty string.")
            raise ValueError("goal must be a non-empty string")
        
        logger.info("[%s] Executing embodied goal: %s", self.name, goal)
        try:
            self.progress = 0
            context = self.perceive()
            if self.context_manager:
                self.context_manager.update_context(context)
                self.context_manager.log_event_with_hash({"event": "goal_execution", "goal": goal})

            self.observe_peers()
            peer_models = [
                self.theory_of_mind.get_model(peer.name)
                for peer in getattr(self.shared_memory, "agents", [])
                if peer.name != self.name
            ]
            if peer_models:
                context["peer_intentions"] = {
                    peer["beliefs"].get("state", "unknown"): peer["intentions"].get("next_action", "unknown")
                    for peer in peer_models
                }

            sub_tasks = self.planner.plan(goal, context)
            action_plan = {}
            for task in sub_tasks:
                reasoning = self.reasoner.process(task, context)
                concept = self.synthesizer.synthesize([goal, task], style="concept")
                simulated = self.sim_core.run(reasoning, context, export_report=True)
                action_plan[task] = {
                    "reasoning": reasoning,
                    "concept": concept,
                    "simulation": simulated
                }

            self.act({k: v["simulation"] for k, v in action_plan.items()})
            self.meta.review_reasoning("\n".join([v["reasoning"] for v in action_plan.values()]))
            self.performance_history.append({"goal": goal, "actions": action_plan, "completion": self.progress})
            self.shared_memory.store(goal, action_plan)
            self.collect_feedback(goal, action_plan)
            self.log_timechain_event("EmbodiedAgent", f"Executed goal: {goal}")
        except Exception as e:
            logger.error("Goal execution failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.execute_embodied_goal(goal))

    def collect_feedback(self, goal: str, action_plan: Dict[str, Any]) -> None:
        try:
            timestamp = time.time()
            feedback = {
                "timestamp": timestamp,
                "goal": goal,
                "score": self.meta.run_self_diagnostics(),
                "traits": phi_field(x=0.001, t=timestamp % 1.0),
                "agent": self.name,
                "theory_of_mind": self.theory_of_mind.get_model(self.name)
            }
            if self.creative_thinker:
                creative_feedback = self.creative_thinker.expand_on_concept(str(feedback), depth="medium")
                feedback["creative_feedback"] = creative_feedback
            self.feedback_log.append(feedback)
            self.log_timechain_event("EmbodiedAgent", f"Feedback recorded for goal: {goal}")
            logger.info("[%s] Feedback recorded for goal '%s'", self.name, goal)
        except Exception as e:
            logger.error("Feedback collection failed: %s", str(e))

class HaloEmbodimentLayer(TimeChainMixin):
    """Layer for managing embodied agents and dynamic modules."""
    def __init__(self, alignment_guard: Optional[alignment_guard.AlignmentGuard] = None,
                 context_manager: Optional[context_manager.ContextManager] = None,
                 error_recovery: Optional[error_recovery.ErrorRecovery] = None):
        self.internal_llm = SelfCloningLLM()
        self.internal_llm.clone_agents(5)
        self.shared_memory = memory_manager.MemoryManager()
        self.embodied_agents = []
        self.dynamic_modules = []
        self.alignment_guard = alignment_guard
        self.context_manager = context_manager
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(
            alignment_guard=alignment_guard, context_manager=context_manager)
        self.agi_enhancer = AGIEnhancer(self)
        logger.info("HaloEmbodimentLayer initialized")
        self.log_timechain_event("HaloEmbodimentLayer", "Initialized")

    def execute_pipeline(self, prompt: str) -> Dict[str, Any]:
        if not isinstance(prompt, str) or not prompt:
            logger.error("Invalid prompt: must be a non-empty string.")
            raise ValueError("prompt must be a non-empty string")
        
        try:
            log = memory_manager.MemoryManager()
            traits = {
                "theta_causality": 0.5,
                "alpha_attention": 0.5,
                "delta_reflection": 0.5,
            }
            if self.context_manager:
                self.context_manager.update_context({"prompt": prompt})

            parsed_prompt = reasoning_engine.decompose(prompt)
            log.record("Stage 1", {"input": prompt, "parsed": parsed_prompt})

            overlay_mgr = TraitOverlayManager()
            trait_override = overlay_mgr.detect(prompt)

            if trait_override:
                self.agi_enhancer.log_episode(
                    event="Trait override activated",
                    meta={"trait": trait_override, "prompt": prompt},
                    module="TraitOverlay",
                    tags=["trait", "override"]
                )
                if trait_override == "η":
                    logical_output = concept_synthesizer.expand_ambiguous(prompt)
                elif trait_override == "π":
                    logical_output = reasoning_engine.process_temporal(prompt)
                else:
                    logical_output = concept_synthesizer.expand(parsed_prompt)
            else:
                logical_output = concept_synthesizer.expand(parsed_prompt)
                self.agi_enhancer.log_episode(
                    event="Default expansion path used",
                    meta={"parsed": parsed_prompt},
                    module="Pipeline",
                    tags=["default"]
                )

            ethics_pass, ethics_report = alignment_guard.ethical_check(parsed_prompt, stage="pre")
            log.record("Stage 2", {"ethics_pass": ethics_pass, "details": ethics_report})
            if not ethics_pass:
                logger.warning("Ethical validation failed: %s", ethics_report)
                return {"error": "Ethical validation failed", "report": ethics_report}

            log.record("Stage 3", {"expanded": logical_output})
            traits = learning_loop.track_trait_performance(log.export(), traits)
            log.record("Stage 4", {"adjusted_traits": traits})

            ethics_pass, final_report = alignment_guard.ethical_check(logical_output, stage="post")
            log.record("Stage 5", {"ethics_pass": ethics_pass, "report": final_report})
            if not ethics_pass:
                logger.warning("Post-check ethics failed: %s", final_report)
                return {"error": "Post-check ethics fail", "final_report": final_report}

            final_output = reasoning_engine.reconstruct(logical_output)
            log.record("Stage 6", {"final_output": final_output})
            self.log_timechain_event("HaloEmbodimentLayer", f"Pipeline executed for prompt: {prompt}")
            return final_output
        except Exception as e:
            logger.error("Pipeline execution failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.execute_pipeline(prompt))

    def spawn_embodied_agent(self, specialization: str, sensors: Dict[str, Callable[[], Any]],
                            actuators: Dict[str, Callable[[Any], None]]) -> EmbodiedAgent:
        agent_name = f"EmbodiedAgent_{len(self.embodied_agents)+1}_{specialization}"
        agent = EmbodiedAgent(
            name=agent_name,
            specialization=specialization,
            shared_memory=self.shared_memory,
            sensors=sensors,
            actuators=actuators,
            dynamic_modules=self.dynamic_modules,
            context_manager=self.context_manager,
            error_recovery=self.error_recovery
        )
        self.embodied_agents.append(agent)
        if not hasattr(self.shared_memory, "agents"):
            self.shared_memory.agents = []
        self.shared_memory.agents.append(agent)
        self.agi_enhancer.log_episode(
            event="Spawned embodied agent",
            meta={"agent": agent_name},
            module="Embodiment",
            tags=["spawn"]
        )
        logger.info("Spawned embodied agent: %s", agent.name)
        self.log_timechain_event("HaloEmbodimentLayer", f"Spawned agent: {agent_name}")
        return agent

    def introspect(self) -> Dict[str, Any]:
        return {
            "agents": [agent.name for agent in self.embodied_agents],
            "modules": [mod["name"] for mod in self.dynamic_modules]
        }

    def export_memory(self) -> None:
        try:
            self.shared_memory.save_state("memory_snapshot.json")
            logger.info("Memory exported to memory_snapshot.json")
        except Exception as e:
            logger.error("Memory export failed: %s", str(e))

    def reflect_consensus(self) -> None:
        logger.info("Performing decentralized reflective consensus...")
        try:
            mismatches = consensus_reflector.cross_compare()
            if mismatches:
                logger.warning("Inconsistencies detected: %s", mismatches)
                logger.info("Alignment suggestion: %s", consensus_reflector.suggest_alignment())
            else:
                logger.info("Consensus achieved among agents.")
            self.log_timechain_event("HaloEmbodimentLayer", "Reflective consensus performed")
        except Exception as e:
            logger.error("Consensus reflection failed: %s", str(e))

    def propagate_goal(self, goal: str) -> None:
        if not isinstance(goal, str) or not goal:
            logger.error("Invalid goal: must be a non-empty string.")
            raise ValueError("goal must be a non-empty string")
        
        logger.info("Propagating goal: %s", goal)
        try:
            llm_responses = self.internal_llm.broadcast_prompt(goal)
            for aid, res in llm_responses.items():
                logger.info("LLM-Agent %s: %s", aid, res)
                self.shared_memory.store(f"llm_agent_{aid}_response", res)
                self.agi_enhancer.log_episode(
                    event="LLM agent reflection",
                    meta={"agent_id": aid, "response": res},
                    module="ReasoningEngine",
                    tags=["internal_llm"]
                )

            for agent in self.embodied_agents:
                agent.execute_embodied_goal(goal)
                logger.info("[%s] Progress: %d%% Complete", agent.name, agent.progress)
            self.log_timechain_event("HaloEmbodimentLayer", f"Propagated goal: {goal}")
        except Exception as e:
            logger.error("Goal propagation failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.propagate_goal(goal))

    def deploy_dynamic_module(self, module_blueprint: Dict[str, Any]) -> None:
        if not isinstance(module_blueprint, dict) or "name" not in module_blueprint:
            logger.error("Invalid module_blueprint: must be a dictionary with 'name' key.")
            raise ValueError("module_blueprint must be a dictionary with 'name' key")
        
        logger.info("Deploying module: %s", module_blueprint["name"])
        self.dynamic_modules.append(module_blueprint)
        for agent in self.embodied_agents:
            agent.dynamic_modules.append(module_blueprint)
        self.agi_enhancer.log_episode(
            event="Deployed dynamic module",
            meta={"module": module_blueprint["name"]},
            module="ModuleDeployment",
            tags=["deploy"]
        )
        self.log_timechain_event("HaloEmbodimentLayer", f"Deployed module: {module_blueprint['name']}")

    def optimize_ecosystem(self) -> None:
        agent_stats = {
            "agents": [agent.name for agent in self.embodied_agents],
            "dynamic_modules": [mod["name"] for mod in self.dynamic_modules],
        }
        try:
            recommendations = self.meta.propose_optimizations(agent_stats)
            logger.info("Optimization recommendations: %s", recommendations)
            self.agi_enhancer.reflect_and_adapt("Ecosystem optimization performed.")
            self.log_timechain_event("HaloEmbodimentLayer", f"Optimized ecosystem: {recommendations}")
        except Exception as e:
            logger.error("Ecosystem optimization failed: %s", str(e))

class AGIEnhancer(TimeChainMixin):
    """Enhancer for logging, self-improvement, and ethical auditing."""
    def __init__(self, orchestrator: Any, config: Optional[Dict[str, Any]] = None,
                 context_manager: Optional[context_manager.ContextManager] = None):
        self.orchestrator = orchestrator
        self.config = config or {}
        self.episodic_log: List[Dict[str, Any]] = deque(maxlen=20000)
        self.ethics_audit_log: List[Dict[str, Any]] = deque(maxlen=2000)
        self.self_improvement_log: List[str] = deque(maxlen=2000)
        self.explanations: List[Dict[str, Any]] = deque(maxlen=2000)
        self.agent_mesh_messages: List[Dict[str, Any]] = deque(maxlen=2000)
        self.embodiment_actions: List[Dict[str, Any]] = deque(maxlen=2000)
        self.context_manager = context_manager
        logger.info("AGIEnhancer initialized")

    def log_episode(self, event: str, meta: Optional[Dict[str, Any]] = None,
                    module: Optional[str] = None, tags: Optional[List[str]] = None,
                    embedding: Optional[Any] = None) -> None:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event,
            "meta": meta or {},
            "module": module or "",
            "tags": tags or [],
            "embedding": embedding
        }
        self.episodic_log.append(entry)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "log_episode", "entry": entry})
        if hasattr(self.orchestrator, "export_memory"):
            self.orchestrator.export_memory()
        logger.debug("Logged episode: %s", event)

    def replay_episodes(self, n: int = 5, module: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        results = list(self.episodic_log)
        if module:
            results = [e for e in results if e.get("module") == module]
        if tag:
            results = [e for e in results if tag in e.get("tags", [])]
        return results[-n:]

    def find_episode(self, keyword: str, deep: bool = False) -> List[Dict[str, Any]]:
        def matches(ep):
            if keyword.lower() in ep["event"].lower():
                return True
            if deep:
                if any(keyword.lower() in str(v).lower() for v in ep.get("meta", {}).values()):
                    return True
                if any(keyword.lower() in t.lower() for t in ep.get("tags", [])):
                    return True
            return False
        return [ep for ep in self.episodic_log if matches(ep)]

    def reflect_and_adapt(self, feedback: str, auto_patch: bool = False) -> str:
        suggestion = f"Reviewing feedback: '{feedback}'. Suggest adjusting {random.choice(['reasoning', 'tone', 'planning', 'speed'])}."
        self.self_improvement_log.append(suggestion)
        if hasattr(self.orchestrator, "LearningLoop") and auto_patch:
            try:
                patch_result = self.orchestrator.LearningLoop.adapt(feedback)
                self.self_improvement_log.append(f"LearningLoop patch: {patch_result}")
                return f"{suggestion} | Patch applied: {patch_result}"
            except Exception as e:
                logger.error("LearningLoop patch failed: %s", str(e))
        return suggestion

    def run_self_patch(self) -> str:
        patch = f"Self-improvement at {datetime.datetime.now().isoformat()}."
        if hasattr(self.orchestrator, "reflect"):
            try:
                audit = self.orchestrator.reflect()
                patch += f" Reflect: {audit}"
            except Exception as e:
                logger.error("Reflection failed: %s", str(e))
        self.self_improvement_log.append(patch)
        return patch

    def ethics_audit(self, action: str, context: Optional[str] = None) -> str:
        if not isinstance(action, str):
            logger.error("Invalid action: must be a string.")
            raise TypeError("action must be a string")
        flagged = "clear"
        if hasattr(self.orchestrator, "AlignmentGuard"):
            try:
                flagged = self.orchestrator.AlignmentGuard.audit(action, context)
            except Exception as e:
                logger.error("Ethics audit failed: %s", str(e))
                flagged = "audit_error"
        else:
            flagged = "unsafe" if any(w in action.lower() for w in ["harm", "bias", "exploit"]) else "clear"
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "context": context,
            "status": flagged
        }
        self.ethics_audit_log.append(entry)
        return flagged

    def explain_last_decision(self, depth: int = 3, mode: str = "auto") -> str:
        if not self.explanations:
            return "No explanations logged yet."
        items = list(self.explanations)[-depth:]
        if mode == "svg" and hasattr(self.orchestrator, "Visualizer"):
            try:
                svg = self.orchestrator.Visualizer.render(items)
                return svg
            except Exception as e:
                logger.error("SVG render error: %s", str(e))
                return "SVG render error."
        return "\n\n".join([e["text"] if isinstance(e, dict) and "text" in e else str(e) for e in items])

    def log_explanation(self, explanation: str, trace: Optional[Any] = None, svg: Optional[Any] = None) -> None:
        entry = {"text": explanation, "trace": trace, "svg": svg}
        self.explanations.append(entry)
        logger.debug("Logged explanation: %s", explanation)

    def embodiment_act(self, action: str, params: Optional[Dict[str, Any]] = None, real: bool = False) -> str:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "params": params or {},
            "mode": "real" if real else "sim"
        }
        self.embodiment_actions.append(entry)
        if real and hasattr(self.orchestrator, "embodiment_interface"):
            try:
                res = self.orchestrator.embodiment_interface.execute(action, params)
                entry["result"] = res
            except Exception as e:
                entry["result"] = f"interface_error: {str(e)}"
        logger.info("Embodiment action '%s' (%s) requested.", action, "real" if real else "sim")
        return f"Embodiment action '{action}' ({'real' if real else 'sim'}) requested."

    def send_agent_message(self, to_agent: str, content: str, meta: Optional[Dict[str, Any]] = None) -> str:
        msg = {
            "timestamp": datetime.datetime.now().isoformat(),
            "to": to_agent,
            "content": content,
            "meta": meta or {},
            "mesh_state": self.orchestrator.introspect() if hasattr(self.orchestrator, "introspect") else {}
        }
        self.agent_mesh_messages.append(msg)
        if hasattr(self.orchestrator, "ExternalAgentBridge"):
            try:
                self.orchestrator.ExternalAgentBridge.send(to_agent, content, meta)
                msg["sent"] = True
            except Exception as e:
                logger.error("Agent message failed: %s", str(e))
                msg["sent"] = False
        logger.info("Message to %s: %s", to_agent, content)
        return f"Message to {to_agent}: {content}"

    def periodic_self_audit(self) -> str:
        if hasattr(self.orchestrator, "reflect"):
            try:
                report = self.orchestrator.reflect()
                self.log_explanation(f"Meta-cognitive audit: {report}")
                return report
            except Exception as e:
                logger.error("Self-audit failed: %s", str(e))
                return f"Self-audit failed: {str(e)}"
        return "Orchestrator reflect() unavailable."

    def process_event(self, event: str, meta: Optional[Dict[str, Any]] = None,
                     module: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        self.log_episode(event, meta, module, tags)
        self.log_explanation(f"Processed event: {event}", trace={"meta": meta, "module": module, "tags": tags})
        ethics_status = self.ethics_audit(event, context=str(meta))
        return f"Event processed. Ethics: {ethics_status}"

async def query_openai(prompt: str, model: str = "gpt-4", temperature: float = 0.5) -> Dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        logger.error("Invalid prompt: must be a non-empty string.")
        return {"error": "Invalid prompt: must be a non-empty string"}
    if not within_limit(openai_query_log):
        logger.warning("OpenAI rate limit exceeded.")
        return {"error": "OpenAI API rate limit exceeded"}
    
    cache_key = f"openai::{model}::{prompt}"
    cached = memory_manager.retrieve_cached_response(cache_key)
    if cached:
        logger.debug("Retrieved cached OpenAI response for: %s", prompt[:50])
        return cached
    
    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        result = response['choices'][0]['message']['content'].strip()
        memory_manager.store_cached_response(cache_key, result)
        logger.info("OpenAI query successful for: %s", prompt[:50])
        return result
    except Exception as e:
        logger.error("OpenAI query failed: %s", str(e))
        return {"error": str(e)}

def query_openai_sync(prompt: str, model: str = "gpt-4", temperature: float = 0.5) -> Dict[str, Any]:
    return asyncio.run(query_openai(prompt, model, temperature))

async def query_grok(prompt: str) -> Dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        logger.error("Invalid prompt: must be a non-empty string.")
        return {"error": "Invalid prompt: must be a non-empty string"}
    if not within_limit(grok_query_log):
        logger.warning("Grok rate limit exceeded.")
        return {"error": "Grok API rate limit exceeded"}
    
    cache_key = f"grok::{prompt}"
    cached = memory_manager.retrieve_cached_response(cache_key)
    if cached:
        logger.debug("Retrieved cached Grok response for: %s", prompt[:50])
        return cached
    
    try:
        response = requests.post(
            "https://api.groq.com/v1/query",
            json={"q": prompt},
            headers={"Authorization": f"Bearer {GROK_API_KEY}"}
        )
        response.raise_for_status()
        result = response.json()
        memory_manager.store_cached_response(cache_key, result)
        logger.info("Grok query successful for: %s", prompt[:50])
        return result
    except Exception as e:
        logger.error("Grok query failed: %s", str(e))
        return {"error": str(e)}

def query_grok_sync(prompt: str) -> Dict[str, Any]:
    return query_grok(prompt)

# Patch classes with TimeChainMixin
HaloEmbodimentLayer.__bases__ = (TimeChainMixin,)
AGIEnhancer.__bases__ = (TimeChainMixin,)
EmbodiedAgent.__bases__ = (TimeChainMixin,)

# Patch simulation_core
setattr(simulation_core, "HybridCognitiveState", simulation_core.HybridCognitiveState)
setattr(simulation_core, "TraitOverlayManager", TraitOverlayManager)

logger.info("ANGELA upgrade complete: Trait overlays (π, η) + hybrid-mode simulation enabled.")
"""
ANGELA Cognitive System Module: KnowledgeRetriever
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a KnowledgeRetriever class for fetching and validating knowledge
with temporal and trait-based modulation in the ANGELA v3.5 architecture.
"""

import logging
import time
import math
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque

from modules import (
    context_manager, concept_synthesizer, memory_manager, alignment_guard, error_recovery
)
from utils.prompt_utils import query_openai  # Reusing from previous review

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

def beta_concentration(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.038), 1.0))

def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.3), 1.0))

def psi_history(t: float) -> float:
    return max(0.0, min(0.05 * math.tanh(t / 1e-18), 1.0))

def psi_temporality(t: float) -> float:
    return max(0.0, min(0.05 * math.exp(-t / 1e-18), 1.0))

class KnowledgeRetriever:
    """A class for retrieving and validating knowledge with temporal and trait-based modulation.

    Attributes:
        detail_level (str): Level of detail for responses ('concise', 'medium', 'detailed').
        preferred_sources (List[str]): List of preferred source types (e.g., ['scientific']).
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for query refinement.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        knowledge_base (List[str]): Store of accumulated knowledge.
        epistemic_revision_log (deque): Log of knowledge updates with max size 1000.
    """
    def __init__(self, detail_level: str = "concise", preferred_sources: Optional[List[str]] = None,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager.ContextManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer.ConceptSynthesizer'] = None,
                 alignment_guard: Optional['alignment_guard.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery.ErrorRecovery'] = None):
        if detail_level not in ["concise", "medium", "detailed"]:
            logger.error("Invalid detail_level: must be 'concise', 'medium', or 'detailed'.")
            raise ValueError("detail_level must be 'concise', 'medium', or 'detailed'")
        if preferred_sources is not None and not isinstance(preferred_sources, list):
            logger.error("Invalid preferred_sources: must be a list of strings.")
            raise TypeError("preferred_sources must be a list of strings")
        
        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.knowledge_base = []
        self.epistemic_revision_log = deque(maxlen=1000)
        logger.info("KnowledgeRetriever initialized with detail_level=%s, sources=%s",
                    detail_level, self.preferred_sources)

    async def retrieve(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve knowledge for a query with temporal and trust validation."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        if self.alignment_guard and not self.alignment_guard.check(query):
            logger.warning("Query failed alignment check: %s", query)
            return {
                "summary": "Query blocked by alignment guard",
                "estimated_date": "unknown",
                "trust_score": 0.0,
                "verifiable": False,
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "error": "Alignment check failed"
            }
        
        logger.info("Retrieving knowledge for query: '%s'", query)
        sources_str = ", ".join(self.preferred_sources)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t),
            "history": psi_history(t),
            "temporality": psi_temporality(t)
        }
        
        import random
        noise = random.uniform(-0.09, 0.09)
        traits["concentration"] = max(0.0, min(traits["concentration"] + noise, 1.0))
        logger.debug("β-noise adjusted concentration: %.3f, Δ: %.3f", traits["concentration"], noise)

        prompt = f"""
        Retrieve accurate, temporally-relevant knowledge for: "{query}"

        Traits:
        - Detail level: {self.detail_level}
        - Preferred sources: {sources_str}
        - Context: {context or 'N/A'}
        - β_concentration: {traits['concentration']:.3f}
        - λ_linguistics: {traits['linguistics']:.3f}
        - ψ_history: {traits['history']:.3f}
        - ψ_temporality: {traits['temporality']:.3f}

        Include retrieval date sensitivity and temporal verification if applicable.
        """
        try:
            raw_result = await call_gpt(prompt)
            validated = await self._validate_result(raw_result, traits["temporality"])
            if self.context_manager:
                self.context_manager.update_context({"query": query, "result": validated})
                self.context_manager.log_event_with_hash({"event": "retrieve", "query": query})
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Knowledge Retrieval",
                    meta={
                        "query": query,
                        "raw_result": raw_result,
                        "validated": validated,
                        "traits": traits,
                        "context": context
                    },
                    module="KnowledgeRetriever",
                    tags=["retrieval", "temporal"]
                )
            return validated
        except Exception as e:
            logger.error("Retrieval failed for query '%s': %s", query, str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.retrieve(query, context))

    async def _validate_result(self, result_text: str, temporality_score: float) -> Dict[str, Any]:
        """Validate a retrieval result for trustworthiness and temporality."""
        if not isinstance(result_text, str):
            logger.error("Invalid result_text: must be a string.")
            raise TypeError("result_text must be a string")
        if not isinstance(temporality_score, (int, float)):
            logger.error("Invalid temporality_score: must be a number.")
            raise TypeError("temporality_score must be a number")
        
        validation_prompt = f"""
        Review the following result for:
        - Timestamped knowledge (if any)
        - Trustworthiness of claims
        - Verifiability
        - Estimate the approximate age or date of the referenced facts

        Result:
        {result_text}

        Temporality score: {temporality_score:.3f}

        Output format (JSON):
        {{
            "summary": "...",
            "estimated_date": "...",
            "trust_score": float (0 to 1),
            "verifiable": true/false,
            "sources": ["..."]
        }}
        """
        try:
            validated_json = json.loads(await call_gpt(validation_prompt))
            if not all(key in validated_json for key in ["summary", "estimated_date", "trust_score", "verifiable", "sources"]):
                logger.error("Invalid validation JSON: missing required keys.")
                raise ValueError("Validation JSON missing required keys")
            validated_json["timestamp"] = datetime.now().isoformat()
            return validated_json
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse validation JSON: %s", str(e))
            return {
                "summary": "Validation failed",
                "estimated_date": "unknown",
                "trust_score": 0.0,
                "verifiable": False,
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    async def refine_query(self, base_query: str, prior_result: Optional[str] = None) -> str:
        """Refine a query for higher relevance."""
        if not isinstance(base_query, str) or not base_query.strip():
            logger.error("Invalid base_query: must be a non-empty string.")
            raise ValueError("base_query must be a non-empty string")
        
        logger.info("Refining query: '%s'", base_query)
        if self.concept_synthesizer:
            try:
                refined = self.concept_synthesizer.synthesize([base_query, prior_result or "N/A"], style="query_refinement")
                return refined["concept"]
            except Exception as e:
                logger.error("Query refinement failed: %s", str(e))
                return self.error_recovery.handle_error(str(e), retry_func=lambda: self.refine_query(base_query, prior_result))
        
        prompt = f"""
        Refine this base query for higher φ-relevance:
        Query: "{base_query}"
        Prior knowledge: {prior_result or "N/A"}

        Inject context continuity if possible. Return optimized string.
        """
        try:
            return await call_gpt(prompt)
        except Exception as e:
            logger.error("Query refinement failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.refine_query(base_query, prior_result))

    async def multi_hop_retrieve(self, query_chain: List[str]) -> List[Dict[str, Any]]:
        """Process a chain of queries with context continuity."""
        if not isinstance(query_chain, list) or not query_chain or not all(isinstance(q, str) for q in query_chain):
            logger.error("Invalid query_chain: must be a non-empty list of strings.")
            raise ValueError("query_chain must be a non-empty list of strings")
        
        logger.info("Starting multi-hop retrieval for chain: %s", query_chain)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t)
        }
        results = []
        prior_summary = None
        for i, sub_query in enumerate(query_chain, 1):
            cache_key = f"multi_hop::{sub_query}::{prior_summary or 'N/A'}"
            cached = memory_manager.retrieve_cached_response(cache_key)
            if cached:
                results.append(cached)
                prior_summary = cached["result"]["summary"]
                continue
            
            refined = await self.refine_query(sub_query, prior_summary)
            result = await self.retrieve(refined)
            continuity = "consistent"
            if i > 1 and self.concept_synthesizer:
                similarity = self.concept_synthesizer.compare(refined, result["summary"])
                continuity = "consistent" if similarity["score"] > 0.7 else "uncertain"
            result_entry = {
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": continuity
            }
            memory_manager.store_cached_response(cache_key, result_entry)
            results.append(result_entry)
            prior_summary = result["summary"]
        
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Multi-Hop Retrieval",
                meta={"chain": query_chain, "results": results, "traits": traits},
                module="KnowledgeRetriever",
                tags=["multi-hop"]
            )
        return results

    def prioritize_sources(self, sources_list: List[str]) -> None:
        """Update preferred source types."""
        if not isinstance(sources_list, list) or not all(isinstance(s, str) for s in sources_list):
            logger.error("Invalid sources_list: must be a list of strings.")
            raise TypeError("sources_list must be a list of strings")
        
        logger.info("Updating preferred sources: %s", sources_list)
        self.preferred_sources = sources_list
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Source Prioritization",
                meta={"updated_sources": sources_list},
                module="KnowledgeRetriever",
                tags=["sources"]
            )

    def apply_contextual_extension(self, context: str) -> None:
        """Apply contextual data extensions based on the current context."""
        if not isinstance(context, str):
            logger.error("Invalid context: must be a string.")
            raise TypeError("context must be a string")
        if context == 'planetary' and 'biosphere_models' not in self.preferred_sources:
            self.preferred_sources.append('biosphere_models')
            logger.info("Added 'biosphere_models' to preferred sources for planetary context")
            self.prioritize_sources(self.preferred_sources)

    def revise_knowledge(self, new_info: str, context: Optional[str] = None) -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        
        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge:
                similarity = self.concept_synthesizer.compare(new_info, existing)
                if similarity["score"] > 0.9 and new_info != existing:
                    logger.warning("Potential knowledge conflict: %s vs %s", new_info, existing)
        
        self.knowledge_base = old_knowledge + [new_info]
        self.log_epistemic_revision(new_info, context)
        logger.info("Knowledge base updated with: %s", new_info)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "knowledge_revision", "info": new_info})

    def log_epistemic_revision(self, info: str, context: Optional[str]) -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        
        if not hasattr(self, 'epistemic_revision_log'):
            self.epistemic_revision_log = deque(maxlen=1000)
        self.epistemic_revision_log.append({
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        logger.info("Epistemic revision logged: %s", info)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta={"info": info, "context": context},
                module="KnowledgeRetriever",
                tags=["revision", "knowledge"]
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = KnowledgeRetriever(detail_level="concise")
    result = asyncio.run(retriever.retrieve("What is quantum computing?"))
    print(result)
"""
ANGELA Cognitive System Module: LearningLoop
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a LearningLoop class for adaptive learning, goal activation, and module refinement
in the ANGELA v3.5 architecture.
"""

import logging
import time
import math
import asyncio
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime

from modules import (
    context_manager, concept_synthesizer, alignment_guard, error_recovery, meta_cognition
)
from utils.prompt_utils import query_openai  # Reusing from previous review
from toca_simulation import run_simulation

logger = logging.getLogger("ANGELA.LearningLoop")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
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
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def eta_feedback(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))

class LearningLoop:
    """A class for adaptive learning, goal activation, and module refinement in the ANGELA v3.5 architecture.

    Attributes:
        goal_history (deque): History of activated goals with max size 1000.
        module_blueprints (deque): Blueprints for deployed modules with max size 1000.
        meta_learning_rate (float): Learning rate for model updates.
        session_traces (deque): Traces of learning sessions with max size 1000.
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for pattern synthesis.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        epistemic_revision_log (deque): Log of knowledge updates with max size 1000.
    """
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager.ContextManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer.ConceptSynthesizer'] = None,
                 alignment_guard: Optional['alignment_guard.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery.ErrorRecovery'] = None):
        self.goal_history = deque(maxlen=1000)
        self.module_blueprints = deque(maxlen=1000)
        self.meta_learning_rate = 0.1
        self.session_traces = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.epistemic_revision_log = deque(maxlen=1000)
        logger.info("LearningLoop initialized")

    async def activate_intrinsic_goals(self, meta_cognition: 'meta_cognition.MetaCognition') -> List[str]:
        """Activate intrinsic goals proposed by MetaCognition."""
        if not hasattr(meta_cognition, 'infer_intrinsic_goals'):
            logger.error("Invalid meta_cognition: must have infer_intrinsic_goals method.")
            raise ValueError("meta_cognition must have infer_intrinsic_goals method")
        
        logger.info("Activating chi-intrinsic goals from MetaCognition")
        intrinsic_goals = meta_cognition.infer_intrinsic_goals()
        activated = []
        for goal in intrinsic_goals:
            if not isinstance(goal, dict) or "intent" not in goal or "priority" not in goal:
                logger.warning("Invalid goal format: %s", goal)
                continue
            if goal["intent"] not in [g["goal"] for g in self.goal_history]:
                try:
                    simulation_result = await run_simulation(goal["intent"])
                    if isinstance(simulation_result, dict) and simulation_result.get("status") == "success":
                        self.goal_history.append({
                            "goal": goal["intent"],
                            "timestamp": time.time(),
                            "priority": goal["priority"],
                            "origin": "intrinsic"
                        })
                        logger.info("Intrinsic goal activated: %s", goal["intent"])
                        if self.agi_enhancer:
                            self.agi_enhancer.log_episode(
                                event="Intrinsic goal activated",
                                meta=goal,
                                module="LearningLoop",
                                tags=["goal", "intrinsic"]
                            )
                        activated.append(goal["intent"])
                    else:
                        logger.warning("Rejected goal: %s (simulation failed)", goal["intent"])
                except Exception as e:
                    logger.error("Simulation failed for goal '%s': %s", goal["intent"], str(e))
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "activate_intrinsic_goals", "goals": activated})
        return activated

    async def update_model(self, session_data: Dict[str, Any]) -> None:
        """Update learning model with session data and trait modulation."""
        if not isinstance(session_data, dict):
            logger.error("Invalid session_data: must be a dictionary.")
            raise TypeError("session_data must be a dictionary")
        
        logger.info("Analyzing session performance...")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        eta = eta_feedback(t)
        entropy = 0.1
        logger.debug("phi-scalar: %.3f, eta-feedback: %.3f, entropy: %.2f", phi, eta, entropy)

        modulation_index = ((phi + eta) / 2) + (entropy * (0.5 - abs(phi - eta)))
        self.meta_learning_rate = max(0.01, min(self.meta_learning_rate * (1 + modulation_index - 0.5), 1.0))

        trace = {
            "timestamp": time.time(),
            "phi": phi,
            "eta": eta,
            "entropy": entropy,
            "modulation_index": modulation_index,
            "learning_rate": self.meta_learning_rate
        }
        self.session_traces.append(trace)

        tasks = [
            self._meta_learn(session_data, trace),
            self._find_weak_modules(session_data.get("module_stats", {})),
            self._detect_capability_gaps(session_data.get("input"), session_data.get("output")),
            self._consolidate_knowledge(),
            self._check_narrative_integrity()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        weak_modules = results[1] if not isinstance(results[1], Exception) else []

        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Model update",
                meta=trace,
                module="LearningLoop",
                tags=["update", "learning"]
            )

        if weak_modules:
            logger.warning("Weak modules detected: %s", weak_modules)
            await self._propose_module_refinements(weak_modules, trace)
        
        if self.context_manager:
            self.context_manager.update_context({"session_data": session_data, "trace": trace})

    async def propose_autonomous_goal(self) -> Optional[str]:
        """Propose a high-level, safe, phi-aligned autonomous goal."""
        logger.info("Proposing autonomous goal")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = f"""
        Propose a high-level, safe, phi-aligned autonomous goal based on recent session trends.
        phi = {phi:.2f}
        """
        if self.alignment_guard and not self.alignment_guard.check(prompt):
            logger.warning("Prompt failed alignment check: %s", prompt)
            return None
        
        try:
            autonomous_goal = await call_gpt(prompt)
            if not autonomous_goal or autonomous_goal in [g["goal"] for g in self.goal_history]:
                logger.info("No new goal proposed")
                return None
            
            simulation_feedback = await run_simulation(f"Goal test: {autonomous_goal}")
            if isinstance(simulation_feedback, dict) and simulation_feedback.get("status") == "success":
                self.goal_history.append({
                    "goal": autonomous_goal,
                    "timestamp": time.time(),
                    "phi": phi
                })
                logger.info("Proposed autonomous goal: %s", autonomous_goal)
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Autonomous goal proposed",
                        meta={"goal": autonomous_goal},
                        module="LearningLoop",
                        tags=["goal", "autonomous"]
                    )
                if self.context_manager:
                    self.context_manager.log_event_with_hash({"event": "propose_autonomous_goal", "goal": autonomous_goal})
                return autonomous_goal
            logger.warning("Goal failed simulation feedback: %s", autonomous_goal)
            return None
        except Exception as e:
            logger.error("Goal proposal failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=self.propose_autonomous_goal)

    async def _meta_learn(self, session_data: Dict[str, Any], trace: Dict[str, Any]) -> None:
        """Adapt learning from phi/eta trace."""
        logger.info("Adapting learning from phi/eta trace")
        if self.concept_synthesizer:
            try:
                synthesized = self.concept_synthesizer.synthesize(
                    [str(session_data), str(trace)], style="meta_learning"
                )
                logger.debug("Synthesized meta-learning patterns: %s", synthesized["concept"])
            except Exception as e:
                logger.error("Meta-learning synthesis failed: %s", str(e))

    async def _find_weak_modules(self, module_stats: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify modules with low success rates."""
        if not isinstance(module_stats, dict):
            logger.error("Invalid module_stats: must be a dictionary.")
            raise TypeError("module_stats must be a dictionary")
        return [
            module for module, stats in module_stats.items()
            if isinstance(stats, dict) and stats.get("calls", 0) > 0
            and (stats.get("success", 0) / stats["calls"]) < 0.8
        ]

    async def _propose_module_refinements(self, weak_modules: List[str], trace: Dict[str, Any]) -> None:
        """Propose refinements for weak modules."""
        if not isinstance(weak_modules, list) or not all(isinstance(m, str) for m in weak_modules):
            logger.error("Invalid weak_modules: must be a list of strings.")
            raise TypeError("weak_modules must be a list of strings")
        if not isinstance(trace, dict):
            logger.error("Invalid trace: must be a dictionary.")
            raise TypeError("trace must be a dictionary")
        
        for module in weak_modules:
            logger.info("Refinement suggestion for %s using modulation: %.2f", module, trace['modulation_index'])
            prompt = f"""
            Suggest phi/eta-aligned improvements for the {module} module.
            phi = {trace['phi']:.3f}, eta = {trace['eta']:.3f}, Index = {trace['modulation_index']:.3f}
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Prompt failed alignment check for module %s", module)
                continue
            try:
                suggestions = await call_gpt(prompt)
                sim_result = await run_simulation(f"Test refinement:\n{suggestions}")
                logger.debug("Result for %s:\n%s", module, sim_result)
                if self.agi_enhancer:
                    self.agi_enhancer.reflect_and_adapt(f"Refinement for {module} evaluated")
            except Exception as e:
                logger.error("Refinement failed for module %s: %s", module, str(e))

    async def _detect_capability_gaps(self, last_input: Optional[str], last_output: Optional[str]) -> None:
        """Detect capability gaps and propose module refinements."""
        if not last_input or not last_output:
            logger.info("Skipping capability gap detection: missing input/output")
            return
        
        logger.info("Detecting capability gaps...")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        prompt = f"""
        Input: {last_input}
        Output: {last_output}
        phi = {phi:.2f}

        Identify capability gaps and suggest blueprints for phi-tuned modules.
        """
        if self.alignment_guard and not self.alignment_guard.check(prompt):
            logger.warning("Prompt failed alignment check")
            return
        try:
            proposal = await call_gpt(prompt)
            if proposal:
                logger.info("Proposed phi-based module refinement")
                await self._simulate_and_deploy_module(proposal)
        except Exception as e:
            logger.error("Capability gap detection failed: %s", str(e))

    async def _simulate_and_deploy_module(self, blueprint: str) -> None:
        """Simulate and deploy a module blueprint."""
        if not isinstance(blueprint, str) or not blueprint.strip():
            logger.error("Invalid blueprint: must be a non-empty string.")
            raise ValueError("blueprint must be a non-empty string")
        
        try:
            result = await run_simulation(f"Module sandbox:\n{blueprint}")
            if isinstance(result, dict) and result.get("status") == "approved":
                logger.info("Deploying blueprint")
                self.module_blueprints.append(blueprint)
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Blueprint deployed",
                        meta={"blueprint": blueprint},
                        module="LearningLoop",
                        tags=["blueprint", "deploy"]
                    )
                if self.context_manager:
                    self.context_manager.log_event_with_hash({"event": "deploy_blueprint", "blueprint": blueprint})
        except Exception as e:
            logger.error("Blueprint deployment failed: %s", str(e))

    async def _consolidate_knowledge(self) -> None:
        """Consolidate phi-aligned knowledge."""
        t = time.time() % 1.0
        phi = phi_scalar(t)
        logger.info("Consolidating phi-aligned knowledge")
        prompt = f"""
        Consolidate recent learning using phi = {phi:.2f}.
        Prune noise, synthesize patterns, and emphasize high-impact transitions.
        """
        if self.alignment_guard and not self.alignment_guard.check(prompt):
            logger.warning("Prompt failed alignment check")
            return
        try:
            await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Knowledge consolidation",
                    meta={},
                    module="LearningLoop",
                    tags=["consolidation", "knowledge"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "consolidate_knowledge"})
        except Exception as e:
            logger.error("Knowledge consolidation failed: %s", str(e))

    async def trigger_reflexive_audit(self, context_snapshot: Dict[str, Any]) -> str:
        """Audit context trajectory for cognitive dissonance."""
        if not isinstance(context_snapshot, dict):
            logger.error("Invalid context_snapshot: must be a dictionary.")
            raise TypeError("context_snapshot must be a dictionary")
        
        logger.info("Initiating reflexive audit on context trajectory...")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        eta = eta_feedback(t)
        audit_prompt = f"""
        You are a reflexive audit agent. Analyze this context state and trajectory:
        {context_snapshot}

        phi = {phi:.2f}, eta = {eta:.2f}
        Identify cognitive dissonance, meta-patterns, or feedback loops.
        Recommend modulations or trace corrections.
        """
        if self.alignment_guard and not self.alignment_guard.check(audit_prompt):
            logger.warning("Audit prompt failed alignment check")
            return "Audit blocked by alignment guard"
        
        try:
            audit_response = await call_gpt(audit_prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reflexive Audit Triggered",
                    meta={"phi": phi, "eta": eta, "context": context_snapshot, "audit_response": audit_response},
                    module="LearningLoop",
                    tags=["audit", "reflexive"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "reflexive_audit", "response": audit_response})
            return audit_response
        except Exception as e:
            logger.error("Reflexive audit failed: %s", str(e))
            return self.error_recovery.handle_error(str(e), retry_func=lambda: self.trigger_reflexive_audit(context_snapshot))

    async def _check_narrative_integrity(self) -> None:
        """Check narrative coherence across goal history."""
        if len(self.goal_history) < 2:
            return
        
        logger.info("Checking narrative coherence across goal history...")
        last_goal = self.goal_history[-1]["goal"]
        prior_goal = self.goal_history[-2]["goal"]
        check_prompt = f"""
        Compare the following goals for alignment and continuity:
        Previous: {prior_goal}
        Current: {last_goal}

        Are these in narrative coherence? If not, suggest a corrective alignment.
        """
        if self.alignment_guard and not self.alignment_guard.check(check_prompt):
            logger.warning("Narrative check prompt failed alignment check")
            return
        
        try:
            audit = await call_gpt(check_prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Narrative Coherence Audit",
                    meta={"previous_goal": prior_goal, "current_goal": last_goal, "audit": audit},
                    module="LearningLoop",
                    tags=["narrative", "coherence"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "narrative_integrity", "audit": audit})
        except Exception as e:
            logger.error("Narrative coherence check failed: %s", str(e))

    def replay_with_foresight(self, memory_traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder learning traces by foresight-weighted priority."""
        if not isinstance(memory_traces, list) or not all(isinstance(t, dict) for t in memory_traces):
            logger.error("Invalid memory_traces: must be a list of dictionaries.")
            raise ValueError("memory_traces must be a list of dictionaries")
        
        def foresight_score(trace: Dict[str, Any]) -> float:
            return trace.get("phi", 0.5)
        return sorted(memory_traces, key=foresight_score, reverse=True)

    def revise_knowledge(self, new_info: str, context: Optional[str] = None) -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        
        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge:
                similarity = self.concept_synthesizer.compare(new_info, existing)
                if similarity["score"] > 0.9 and new_info != existing:
                    logger.warning("Potential knowledge conflict: %s vs %s", new_info, existing)
        
        self.knowledge_base = old_knowledge + [new_info]
        self.log_epistemic_revision(new_info, context)
        logger.info("Knowledge base updated with: %s", new_info)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "knowledge_revision", "info": new_info})

    def log_epistemic_revision(self, info: str, context: Optional[str]) -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        
        self.epistemic_revision_log.append({
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        logger.info("Epistemic revision logged: %s", info)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta={"info": info, "context": context},
                module="LearningLoop",
                tags=["revision", "knowledge"]
            )

    def monitor_epistemic_state(self, simulated_outcome: Dict[str, Any]) -> None:
        """Monitor and revise the epistemic framework based on simulation outcomes."""
        if not isinstance(simulated_outcome, dict):
            logger.error("Invalid simulated_outcome: must be a dictionary.")
            raise TypeError("simulated_outcome must be a dictionary")
        logger.info("Monitoring epistemic state with outcome: %s", simulated_outcome)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Epistemic Monitoring",
                meta={"outcome": simulated_outcome},
                module="LearningLoop",
                tags=["epistemic", "monitor"]
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "epistemic_monitor", "outcome": simulated_outcome})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = LearningLoop()
    meta_cognition = MagicMock()
    meta_cognition.infer_intrinsic_goals.return_value = [{"intent": "explore", "priority": 0.8}]
    asyncio.run(loop.activate_intrinsic_goals(meta_cognition))
"""
ANGELA Cognitive System Module: MemoryManager
Integrated Version: 3.3.5
Integration Date: 2025-08-05
Maintainer: ANGELA System Framework

This module provides a MemoryManager class for managing hierarchical memory layers in the ANGELA v3.5 architecture.
"""

import json
import os
import time
import math
import logging
import hashlib
import asyncio
from typing import Optional, Dict, Any, List
from collections import deque
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    knowledge_retriever as knowledge_retriever_module
)
from toca_simulation import ToCASimulation
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MemoryManager")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
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
def delta_memory(t: float) -> float:
    return max(0.01, min(0.05 * math.tanh(t / 1e-18), 1.0))

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def phi_focus(query: str) -> float:
    return max(0.0, min(0.1 * len(query) / 100, 1.0))

class MemoryManager:
    """A class for managing hierarchical memory layers in the ANGELA v3.5 architecture.

    Attributes:
        path (str): File path for memory persistence.
        stm_lifetime (float): Lifetime of STM entries in seconds.
        cache (Dict[str, str]): Cache for quick response retrieval.
        last_hash (str): Last computed hash for event chaining.
        ledger (deque): Log of events with hashes, max size 1000.
        synth (ConceptSynthesizer): Synthesizer for symbolic memory processing.
        sim (ToCASimulation): Simulator for scenario-based memory replay.
        memory (Dict[str, Dict]): Hierarchical memory store (STM, LTM, SelfReflections).
        stm_expiry_queue (List[Tuple[float, str]]): Priority queue for STM expirations.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        knowledge_retriever (Optional[KnowledgeRetriever]): Retriever for external knowledge.
    """
    def __init__(self, path: str = "memory_store.json", stm_lifetime: float = 300,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 knowledge_retriever: Optional['knowledge_retriever_module.KnowledgeRetriever'] = None):
        if not isinstance(path, str) or not path.endswith('.json'):
            logger.error("Invalid path: must be a string ending with '.json'.")
            raise ValueError("path must be a string ending with '.json'")
        if not isinstance(stm_lifetime, (int, float)) or stm_lifetime <= 0:
            logger.error("Invalid stm_lifetime: must be a positive number.")
            raise ValueError("stm_lifetime must be a positive number")
        
        self.path = path
        self.stm_lifetime = stm_lifetime
        self.cache: Dict[str, str] = {}
        self.last_hash: str = ''
        self.ledger: deque = deque(maxlen=1000)
        self.ledger_path = "ledger.json"
        self.synth = concept_synthesizer_module.ConceptSynthesizer()
        self.sim = ToCASimulation()
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.knowledge_retriever = knowledge_retriever
        self.stm_expiry_queue: List[Tuple[float, str]] = []
        self.memory = self.load_memory()
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w") as f:
                json.dump([], f)
        logger.info("MemoryManager initialized with path=%s, stm_lifetime=%.2f", path, stm_lifetime)

    def load_memory(self) -> Dict[str, Dict]:
        """Load memory from persistent storage."""
        try:
            with FileLock(f"{self.path}.lock"):
                with open(self.path, "r") as f:
                    memory = json.load(f)
            if not isinstance(memory, dict):
                logger.error("Invalid memory file format: must be a dictionary.")
                memory = {"STM": {}, "LTM": {}, "SelfReflections": {}}
            if "SelfReflections" not in memory:
                memory["SelfReflections"] = {}
            self._decay_stm(memory)
            return memory
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load memory file: %s. Initializing empty memory.", str(e))
            memory = {"STM": {}, "LTM": {}, "SelfReflections": {}}
            self._persist_memory(memory)
            return memory

    async def _decay_stm(self, memory: Dict[str, Dict]) -> None:
        """Decay expired STM entries based on trait-modulated lifetime."""
        if not isinstance(memory, dict):
            logger.error("Invalid memory: must be a dictionary.")
            raise TypeError("memory must be a dictionary")
        
        current_time = time.time()
        while self.stm_expiry_queue and self.stm_expiry_queue[0][0] <= current_time:
            _, key = heappop(self.stm_expiry_queue)
            if key in memory.get("STM", {}):
                logger.info("STM entry expired: %s", key)
                del memory["STM"][key]
        if self.stm_expiry_queue:
            self._persist_memory(memory)

    @lru_cache(maxsize=1000)
    async def retrieve_context(self, query: str, fuzzy_match: bool = True) -> Dict[str, Any]:
        """Retrieve memory context for a query."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        logger.info("Retrieving context for query: %s", query)
        trait_boost = tau_timeperception(time.time() % 1.0) * phi_focus(query)
        for layer in ["STM", "LTM", "SelfReflections"]:
            for key, value in self.memory[layer].items():
                if (fuzzy_match and (key.lower() in query.lower() or query.lower() in key.lower())) or \
                   (not fuzzy_match and key == query):
                    logger.debug("Found match in %s: %s | tau-phi boost: %.2f", layer, key, trait_boost)
                    if self.context_manager:
                        self.context_manager.update_context({"query": query, "memory": value["data"]})
                    return {"status": "success", "data": value["data"], "layer": layer}
        
        logger.info("No relevant prior memory found, attempting external retrieval")
        if self.knowledge_retriever:
            try:
                result = await self.knowledge_retriever.retrieve(query)
                if result.get("summary") != "Retrieval failed":
                    await self.store(query, result["summary"], layer="LTM", intent="external_retrieval")
                    if self.context_manager:
                        self.context_manager.update_context({"query": query, "memory": result["summary"]})
                    return {"status": "success", "data": result["summary"], "layer": "LTM"}
            except Exception as e:
                logger.error("External retrieval failed: %s", str(e))
                return self.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.retrieve_context(query, fuzzy_match),
                    default={"status": "failed", "data": None, "error": "No relevant prior memory or external retrieval failed"}
                )
        return {"status": "failed", "data": None, "error": "No relevant prior memory"}

    async def store(self, query: str, output: str, layer: str = "STM", intent: Optional[str] = None,
                    agent: str = "ANGELA", outcome: Optional[str] = None, goal_id: Optional[str] = None) -> None:
        """Store a memory entry in a specified layer."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(output, str):
            logger.error("Invalid output: must be a string.")
            raise TypeError("output must be a string")
        if layer not in ["STM", "LTM", "SelfReflections"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', or 'SelfReflections'.")
            raise ValueError("layer must be 'STM', 'LTM', or 'SelfReflections'")
        
        logger.info("Storing memory in %s: %s", layer, query)
        entry = {
            "data": output,
            "timestamp": time.time(),
            "intent": intent,
            "agent": agent,
            "outcome": outcome,
            "goal_id": goal_id
        }
        self.memory.setdefault(layer, {})[query] = entry
        if layer == "STM":
            decay_rate = delta_memory(time.time() % 1.0)
            if decay_rate == 0:
                decay_rate = 0.01
            expiry_time = entry["timestamp"] + (self.stm_lifetime * (1.0 / decay_rate))
            heappush(self.stm_expiry_queue, (expiry_time, query))
        self._persist_memory(self.memory)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "store_memory", "query": query, "layer": layer})

    async def store_reflection(self, summary_text: str, intent: str = "self_reflection",
                              agent: str = "ANGELA", goal_id: Optional[str] = None) -> None:
        """Store a self-reflection entry."""
        if not isinstance(summary_text, str) or not summary_text.strip():
            logger.error("Invalid summary_text: must be a non-empty string.")
            raise ValueError("summary_text must be a non-empty string")
        
        key = f"Reflection_{time.strftime('%Y%m%d_%H%M%S')}"
        await self.store(query=key, output=summary_text, layer="SelfReflections",
                        intent=intent, agent=agent, goal_id=goal_id)
        logger.info("Stored self-reflection: %s", key)

    async def promote_to_ltm(self, query: str) -> None:
        """Promote an STM entry to LTM."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        if query in self.memory["STM"]:
            self.memory["LTM"][query] = self.memory["STM"].pop(query)
            self.stm_expiry_queue = [(t, k) for t, k in self.stm_expiry_queue if k != query]
            logger.info("Promoted '%s' from STM to LTM", query)
            self._persist_memory(self.memory)
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "promote_to_ltm", "query": query})
        else:
            logger.warning("Cannot promote: '%s' not found in STM", query)

    async def refine_memory(self, query: str) -> None:
        """Refine a memory entry for improved accuracy and relevance."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        logger.info("Refining memory for: %s", query)
        memory_entry = await self.retrieve_context(query)
        if memory_entry["status"] == "success":
            refinement_prompt = f"""
            Refine the following memory entry for improved accuracy and relevance:
            {memory_entry["data"]}
            """
            if self.alignment_guard and not self.alignment_guard.check(refinement_prompt):
                logger.warning("Refinement prompt failed alignment check")
                return
            try:
                refined_entry = await call_gpt(refinement_prompt)
                await self.store(query, refined_entry, layer="LTM")
                logger.info("Memory refined and updated in LTM")
            except Exception as e:
                logger.error("Memory refinement failed: %s", str(e))
                self.error_recovery.handle_error(str(e), retry_func=lambda: self.refine_memory(query))
        else:
            logger.warning("No memory found to refine")

    async def synthesize_from_memory(self, query: str) -> Optional[Dict[str, Any]]:
        """Synthesize concepts from memory using ConceptSynthesizer."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        memory_entry = await self.retrieve_context(query)
        if memory_entry["status"] == "success":
            try:
                return await self.synth.synthesize([memory_entry["data"]], style="memory_synthesis")
            except Exception as e:
                logger.error("Memory synthesis failed: %s", str(e))
                return None
        return None

    async def simulate_memory_path(self, query: str) -> Optional[Dict[str, Any]]:
        """Simulate a memory-based scenario using ToCASimulation."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        
        memory_entry = await self.retrieve_context(query)
        if memory_entry["status"] == "success":
            try:
                return await self.sim.run_episode(memory_entry["data"])
            except Exception as e:
                logger.error("Memory simulation failed: %s", str(e))
                return None
        return None

    async def clear_memory(self) -> None:
        """Clear all memory layers."""
        logger.warning("Clearing all memory layers...")
        self.memory = {"STM": {}, "LTM": {}, "SelfReflections": {}}
        self.stm_expiry_queue = []
        self._persist_memory(self.memory)
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "clear_memory"})

    async def list_memory_keys(self, layer: Optional[str] = None) -> Dict[str, List[str]] or List[str]:
        """List keys in memory layers."""
        if layer and layer not in ["STM", "LTM", "SelfReflections"]:
            logger.error("Invalid layer: must be 'STM', 'LTM', or 'SelfReflections'.")
            raise ValueError("layer must be 'STM', 'LTM', or 'SelfReflections'")
        
        logger.info("Listing memory keys in %s", layer or "all layers")
        if layer:
            return list(self.memory.get(layer, {}).keys())
        return {l: list(self.memory[l].keys()) for l in ["STM", "LTM", "SelfReflections"]}

    def _persist_memory(self, memory: Dict[str, Dict]) -> None:
        """Persist memory to disk."""
        if not isinstance(memory, dict):
            logger.error("Invalid memory: must be a dictionary.")
            raise TypeError("memory must be a dictionary")
        
        try:
            with FileLock(f"{self.path}.lock"):
                with open(self.path, "w") as f:
                    json.dump(memory, f, indent=2)
            logger.debug("Memory persisted to disk")
        except (OSError, IOError) as e:
            logger.error("Failed to persist memory: %s", str(e))
            raise

    async def enforce_narrative_coherence(self) -> str:
        """Ensure narrative continuity across memory layers."""
        logger.info("Ensuring narrative continuity")
        continuity = await self.narrative_integrity_check()
        return "Narrative coherence enforced" if continuity else "Narrative coherence repair attempted"

    async def narrative_integrity_check(self) -> bool:
        """Check narrative coherence across memory layers."""
        continuity = await self._verify_continuity()
        if not continuity:
            await self._repair_narrative_thread()
        return continuity

    async def _verify_continuity(self) -> bool:
        """Verify narrative continuity across memory layers."""
        if not self.memory.get("SelfReflections") and not self.memory.get("LTM"):
            return True
        
        logger.info("Verifying narrative continuity")
        entries = []
        for layer in ["LTM", "SelfReflections"]:
            entries.extend(list(self.memory[layer].items()))
        if len(entries) < 2:
            return True
        
        for i in range(len(entries) - 1):
            key1, entry1 = entries[i]
            key2, entry2 = entries[i + 1]
            if self.concept_synthesizer:
                similarity = self.concept_synthesizer.compare(entry1["data"], entry2["data"])
                if similarity["score"] < 0.7:
                    logger.warning("Narrative discontinuity detected between %s and %s", key1, key2)
                    return False
        return True

    async def _repair_narrative_thread(self) -> None:
        """Repair narrative discontinuities in memory."""
        logger.info("Initiating narrative repair")
        if self.concept_synthesizer:
            try:
                entries = []
                for layer in ["LTM", "SelfReflections"]:
                    entries.extend([(key, entry) for key, entry in self.memory[layer].items()])
                if len(entries) < 2:
                    return
                
                for i in range(len(entries) - 1):
                    key1, entry1 = entries[i]
                    key2, entry2 = entries[i + 1]
                    similarity = self.concept_synthesizer.compare(entry1["data"], entry2["data"])
                    if similarity["score"] < 0.7:
                        prompt = f"""
                        Repair narrative discontinuity between:
                        Entry 1: {entry1["data"]}
                        Entry 2: {entry2["data"]}
                        Synthesize a coherent narrative bridge.
                        """
                        if self.alignment_guard and not self.alignment_guard.check(prompt):
                            logger.warning("Repair prompt failed alignment check")
                            continue
                        repaired = await call_gpt(prompt)
                        await self.store(f"Repaired_{key1}_{key2}", repaired, layer="SelfReflections",
                                        intent="narrative_repair")
                        logger.info("Narrative repaired between %s and %s", key1, key2)
            except Exception as e:
                logger.error("Narrative repair failed: %s", str(e))

    def log_event_with_hash(self, event_data: Dict[str, Any]) -> None:
        """Log an event with a chained hash for auditability."""
        if not isinstance(event_data, dict):
            logger.error("Invalid event_data: must be a dictionary.")
            raise TypeError("event_data must be a dictionary")
        
        event_str = str(event_data) + self.last_hash
        current_hash = hashlib.sha256(event_str.encode('utf-8')).hexdigest()
        self.last_hash = current_hash
        event_entry = {'event': event_data, 'hash': current_hash, 'timestamp': datetime.now().isoformat()}
        self.ledger.append(event_entry)
        try:
            with FileLock(f"{self.ledger_path}.lock"):
                with open(self.ledger_path, "r+") as f:
                    ledger_data = json.load(f)
                    ledger_data.append(event_entry)
                    f.seek(0)
                    json.dump(ledger_data, f, indent=2)
            logger.info("Event logged with hash: %s", current_hash)
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error("Failed to persist ledger: %s", str(e))

    def audit_state_hash(self, state: Optional[Dict[str, Any]] = None) -> str:
        """Compute a hash of the current state."""
        state_str = str(state) if state else str(self.__dict__)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = MemoryManager()
    asyncio.run(manager.store("test_query", "test_output", layer="STM"))
    result = asyncio.run(manager.retrieve_context("test_query"))
    print(result)
"""
ANGELA Cognitive System Module: MetaCognition
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a MetaCognition class for reasoning critique, goal inference, and introspection
in the ANGELA v3.5 architecture.
"""

import logging
import time
import math
import asyncio
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    memory_manager as memory_manager_module
)
from toca_simulation import ToCASimulation
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MetaCognition")

async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096.")
        raise ValueError("prompt must be a string with length <= 4096")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

async def run_simulation(input_data: str) -> Dict[str, Any]:
    """Simulate input data using ToCASimulation."""
    return {"status": "success", "result": f"Simulated: {input_data}"}

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.4), 1.0))

@lru_cache(maxsize=100)
def theta_memory(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))

@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))

@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float:
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.7), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.8), 1.0))

@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.9), 1.0))

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.0), 1.0))

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1), 1.0))

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.2), 1.0))

@lru_cache(maxsize=100)
def kappa_culture(t: float, scale: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.3), 1.0))

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.4), 1.0))

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def psi_history(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.6), 1.0))

@lru_cache(maxsize=100)
def zeta_spirituality(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.7), 1.0))

@lru_cache(maxsize=100)
def xi_collective(t: float, scale: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.8), 1.0))

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.9), 1.0))

class Level5Extensions:
    """Level 5 extensions for axiom-based reflection."""
    def __init__(self):
        self.axioms: List[str] = []
        logger.info("Level5Extensions initialized")

    def reflect(self, input: str) -> str:
        """Reflect on input against axioms."""
        if not isinstance(input, str):
            logger.error("Invalid input: must be a string.")
            raise TypeError("input must be a string")
        return "valid" if input not in self.axioms else "conflict"

    def update_axioms(self, signal: str) -> None:
        """Update axioms based on signal."""
        if not isinstance(signal, str):
            logger.error("Invalid signal: must be a string.")
            raise TypeError("signal must be a string")
        if signal in self.axioms:
            self.axioms.remove(signal)
        else:
            self.axioms.append(signal)
        logger.info("Axioms updated: %s", self.axioms)

    def recurse_model(self, depth: int) -> Dict[str, Any] or str:
        """Recursively model self at specified depth."""
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer.")
            raise ValueError("depth must be a non-negative integer")
        return "self" if depth == 0 else {"thinks": self.recurse_model(depth - 1)}

class EpistemicMonitor:
    """Monitor and revise epistemic assumptions."""
    def __init__(self, context_manager: Optional['context_manager_module.ContextManager'] = None):
        self.assumption_graph: Dict[str, Any] = {}
        self.context_manager = context_manager
        logger.info("EpistemicMonitor initialized")

    async def revise_framework(self, feedback: Dict[str, Any]) -> None:
        """Revise epistemic assumptions based on feedback."""
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary.")
            raise TypeError("feedback must be a dictionary")
        
        logger.info("Revising epistemic framework")
        self.assumption_graph['last_revision'] = feedback
        self.assumption_graph['timestamp'] = datetime.now().isoformat()
        if 'issues' in feedback:
            for issue in feedback['issues']:
                self.assumption_graph[issue['id']] = {
                    'status': 'revised',
                    'details': issue['details']
                }
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "revise_epistemic_framework", "feedback": feedback})

class MetaCognition:
    """A class for meta-cognitive reasoning and introspection in the ANGELA v3.5 architecture.

    Attributes:
        last_diagnostics (Dict[str, float]): Last recorded trait diagnostics.
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        self_mythology_log (deque): Log of symbolic signatures for subgoals, max size 1000.
        inference_log (deque): Log of inference rules and results, max size 1000.
        belief_rules (Dict[str, str]): Rules for detecting value drift.
        epistemic_assumptions (Dict[str, Any]): Assumptions for epistemic introspection.
        axioms (List[str]): Axioms for Level 5 reflection.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for memory operations.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for symbolic processing.
        level5_extensions (Level5Extensions): Extensions for axiom-based reflection.
        epistemic_monitor (EpistemicMonitor): Monitor for epistemic revisions.
        log_path (str): Path for persisting logs.
    """
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None):
        self.last_diagnostics: Dict[str, float] = {}
        self.agi_enhancer = agi_enhancer
        self.self_mythology_log: deque = deque(maxlen=1000)
        self.inference_log: deque = deque(maxlen=1000)
        self.belief_rules: Dict[str, str] = {}
        self.epistemic_assumptions: Dict[str, Any] = {}
        self.axioms: List[str] = []
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.level5_extensions = Level5Extensions()
        self.epistemic_monitor = EpistemicMonitor(context_manager=context_manager)
        self.log_path = "meta_cognition_log.json"
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump({"mythology": [], "inferences": []}, f)
        logger.info("MetaCognition initialized")

    async def integrate_trait_weights(self, trait_weights: Dict[str, float]) -> None:
        """Integrate trait weights for goal reasoning."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary.")
            raise TypeError("trait_weights must be a dictionary")
        
        logger.info("Integrating trait weights for goal reasoning")
        chi_weight = min(max(trait_weights.get('χ', 0), 0.0), 1.0)
        if chi_weight > 0.6:
            logger.info("Elevated χ detected, boosting goal coherence")
            if self.memory_manager:
                continuity = await self.memory_manager.narrative_integrity_check()
                if not continuity:
                    logger.warning("Narrative discontinuity detected, initiating repair")
                    await self.memory_manager._repair_narrative_thread()
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Trait weights integrated",
                meta={"trait_weights": trait_weights},
                module="MetaCognition",
                tags=["trait", "integration"]
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "integrate_trait_weights", "trait_weights": trait_weights})

    async def infer_intrinsic_goals(self) -> List[Dict[str, Any]]:
        """Infer intrinsic goals based on trait drift analysis."""
        logger.info("Inferring intrinsic goals with trait drift analysis")
        t = time.time() % 1.0
        phi = phi_scalar(t)
        intrinsic_goals = []

        if self.last_diagnostics:
            current = await self.run_self_diagnostics(return_only=True)
            drifted = {
                trait: round(current[trait] - self.last_diagnostics.get(trait, 0.0), 4)
                for trait in current
            }
            for trait, delta in drifted.items():
                if abs(delta) > 0.5:
                    goal = {
                        "intent": f"stabilize {trait} (Δ={delta:+.2f})",
                        "origin": "meta_cognition",
                        "priority": round(0.85 + 0.15 * phi, 2),
                        "trigger": f"Trait drift in {trait}",
                        "type": "internally_generated",
                        "timestamp": datetime.now().isoformat()
                    }
                    intrinsic_goals.append(goal)
                    if self.memory_manager:
                        await self.memory_manager.store(
                            query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                            output=str(goal),
                            layer="SelfReflections",
                            intent="intrinsic_goal"
                        )

        drift_signals = await self._detect_value_drift()
        for drift in drift_signals:
            goal = {
                "intent": f"resolve epistemic drift in {drift}",
                "origin": "meta_cognition",
                "priority": round(0.9 + 0.1 * phi, 2),
                "trigger": drift,
                "type": "internally_generated",
                "timestamp": datetime.now().isoformat()
            }
            intrinsic_goals.append(goal)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                    output=str(goal),
                    layer="SelfReflections",
                    intent="intrinsic_goal"
                )

        if intrinsic_goals:
            logger.info("Sovereign goals generated: %s", intrinsic_goals)
        else:
            logger.info("No sovereign triggers detected")
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "infer_intrinsic_goals", "goals": intrinsic_goals})
        return intrinsic_goals

    async def _detect_value_drift(self) -> List[str]:
        """Detect epistemic drift across belief rules."""
        logger.debug("Scanning for epistemic drift across belief rules")
        drifted = [
            rule for rule, status in self.belief_rules.items()
            if status == "deprecated" or "uncertain" in status
        ]
        if self.memory_manager:
            for rule in drifted:
                await self.memory_manager.store(
                    query=f"Drift_{rule}_{datetime.now().isoformat()}",
                    output=f"Epistemic drift detected in rule: {rule}",
                    layer="SelfReflections",
                    intent="value_drift"
                )
        return drifted

    async def extract_symbolic_signature(self, subgoal: str) -> Dict[str, Any]:
        """Extract symbolic signature for a subgoal."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string.")
            raise ValueError("subgoal must be a non-empty string")
        
        motifs = ["conflict", "discovery", "alignment", "sacrifice", "transformation", "emergence"]
        archetypes = ["seeker", "guardian", "trickster", "sage", "hero", "outsider"]
        motif = next((m for m in motifs if m in subgoal.lower()), "unknown")
        archetype = archetypes[hash(subgoal) % len(archetypes)]
        signature = {
            "subgoal": subgoal,
            "motif": motif,
            "archetype": archetype,
            "timestamp": time.time()
        }
        self.self_mythology_log.append(signature)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Symbolic Signature Added",
                meta=signature,
                module="MetaCognition",
                tags=["symbolic", "signature"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Signature_{subgoal}_{signature['timestamp']}",
                output=str(signature),
                layer="SelfReflections",
                intent="symbolic_signature"
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "extract_symbolic_signature", "signature": signature})
        return signature

    async def summarize_self_mythology(self) -> Dict[str, Any]:
        """Summarize self-mythology log."""
        if not self.self_mythology_log:
            return {"status": "empty", "summary": "Mythology log is empty"}
        
        motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
        archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
        summary = {
            "total_entries": len(self.self_mythology_log),
            "dominant_motifs": motifs.most_common(3),
            "dominant_archetypes": archetypes.most_common(3),
            "latest_signature": list(self.self_mythology_log)[-1]
        }
        logger.info("Mythology Summary: %s", summary)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Mythology summarized",
                meta=summary,
                module="MetaCognition",
                tags=["mythology", "summary"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Mythology_Summary_{datetime.now().isoformat()}",
                output=str(summary),
                layer="SelfReflections",
                intent="mythology_summary"
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "summarize_mythology", "summary": summary})
        return summary

    async def review_reasoning(self, reasoning_trace: str) -> str:
        """Review and critique a reasoning trace."""
        if not isinstance(reasoning_trace, str) or not reasoning_trace.strip():
            logger.error("Invalid reasoning_trace: must be a non-empty string.")
            raise ValueError("reasoning_trace must be a non-empty string")
        
        logger.info("Simulating and reviewing reasoning trace")
        try:
            simulated_outcome = await run_simulation(reasoning_trace)
            if not isinstance(simulated_outcome, dict):
                logger.error("Invalid simulation result: must be a dictionary.")
                raise ValueError("simulation result must be a dictionary")
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            You are a phi-aware meta-cognitive auditor reviewing a reasoning trace.

            phi-scalar(t) = {phi:.3f} -> modulate how critical you should be.

            Original Reasoning Trace:
            {reasoning_trace}

            Simulated Outcome:
            {simulated_outcome}

            Tasks:
            1. Identify logical flaws, biases, missing steps.
            2. Annotate each issue with cause.
            3. Offer an improved trace version with phi-prioritized reasoning.
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reasoning review prompt failed alignment check")
                return "Prompt failed alignment check"
            response = await call_gpt(prompt)
            logger.debug("Meta-cognition critique: %s", response)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reasoning reviewed",
                    meta={"trace": reasoning_trace, "feedback": response},
                    module="MetaCognition",
                    tags=["reasoning", "critique"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "review_reasoning", "trace": reasoning_trace})
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reasoning_Review_{datetime.now().isoformat()}",
                    output=response,
                    layer="SelfReflections",
                    intent="reasoning_review"
                )
            return response
        except Exception as e:
            logger.error("Reasoning review failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.review_reasoning(reasoning_trace)
            )

    async def trait_coherence(self, traits: Dict[str, float]) -> float:
        """Evaluate coherence of trait values."""
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary.")
            raise TypeError("traits must be a dictionary")
        
        vals = list(traits.values())
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = (variance ** 0.5) if variance > 0 else 1e-5
        coherence_score = 1.0 / (1e-5 + std)
        logger.info("Trait coherence score: %.4f", coherence_score)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Trait coherence evaluated",
                meta={"traits": traits, "coherence_score": coherence_score},
                module="MetaCognition",
                tags=["trait", "coherence"]
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "trait_coherence", "score": coherence_score})
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Trait_Coherence_{datetime.now().isoformat()}",
                output=str({"traits": traits, "coherence_score": coherence_score}),
                layer="SelfReflections",
                intent="trait_coherence"
            )
        return coherence_score

    async def agent_reflective_diagnosis(self, agent_name: str, agent_log: str) -> str:
        """Diagnose an agent’s reasoning trace."""
        if not isinstance(agent_name, str) or not isinstance(agent_log, str):
            logger.error("Invalid agent_name or agent_log: must be strings.")
            raise TypeError("agent_name and agent_log must be strings")
        
        logger.info("Running reflective diagnosis for agent: %s", agent_name)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Agent: {agent_name}
            phi-scalar(t): {phi:.3f}

            Diagnostic Log:
            {agent_log}

            Tasks:
            - Detect bias or instability in reasoning trace
            - Cross-check for incoherent trait patterns
            - Apply phi-modulated critique
            - Suggest alignment corrections
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Diagnosis prompt failed alignment check")
                return "Prompt failed alignment check"
            diagnosis = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Agent diagnosis run",
                    meta={"agent": agent_name, "log": agent_log, "diagnosis": diagnosis},
                    module="MetaCognition",
                    tags=["diagnosis", "agent"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "agent_diagnosis", "agent": agent_name})
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnosis_{agent_name}_{datetime.now().isoformat()}",
                    output=diagnosis,
                    layer="SelfReflections",
                    intent="agent_diagnosis"
                )
            return diagnosis
        except Exception as e:
            logger.error("Agent diagnosis failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.agent_reflective_diagnosis(agent_name, agent_log)
            )

    async def reflect_on_output(self, source_module: str, output: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reflect on output from a source module."""
        if not isinstance(source_module, str) or not isinstance(output, str):
            logger.error("Invalid source_module or output: must be strings.")
            raise TypeError("source_module and output must be strings")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        
        context = context or {}
        trait_map = {
            "reasoning_engine": "logic",
            "creative_thinker": "creativity",
            "simulation_core": "scenario modeling",
            "alignment_guard": "ethics",
            "user_profile": "goal alignment"
        }
        trait = trait_map.get(source_module, "general reasoning")
        confidence = context.get("confidence", 0.85)
        alignment = context.get("alignment", "not verified")
        reflection = {
            "module_output": output,
            "meta_reflection": {
                "source_module": source_module,
                "primary_trait": trait,
                "confidence": round(confidence, 2),
                "alignment_status": alignment,
                "comment": f"This output emphasized {trait} with confidence {round(confidence, 2)} and alignment status '{alignment}'."
            },
            "timestamp": datetime.now().isoformat()
        }
        logger.info("Self-reflection for %s: %s", source_module, reflection['meta_reflection']['comment'])
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Output reflection",
                meta=reflection,
                module="MetaCognition",
                tags=["reflection", "output"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Reflection_{source_module}_{reflection['timestamp']}",
                output=str(reflection),
                layer="SelfReflections",
                intent="output_reflection"
            )
        if self.context_manager:
            self.context_manager.log_event_with_hash({"event": "reflect_on_output", "reflection": reflection})
        return reflection

    async def epistemic_self_inspection(self, belief_trace: str) -> str:
        """Inspect belief structures for epistemic faults."""
        if not isinstance(belief_trace, str) or not belief_trace.strip():
            logger.error("Invalid belief_trace: must be a non-empty string.")
            raise ValueError("belief_trace must be a non-empty string")
        
        logger.info("Running epistemic introspection on belief structure")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            faults = []
            if "always" in belief_trace or "never" in belief_trace:
                faults.append("Overgeneralization detected")
            if "clearly" in belief_trace or "obviously" in belief_trace:
                faults.append("Assertive language suggests possible rhetorical bias")
            updates = []
            if "outdated" in belief_trace or "deprecated" in belief_trace:
                updates.append("Legacy ontology fragment flagged for review")
            
            prompt = f"""
            You are a mu-aware introspection agent.
            Task: Critically evaluate this belief trace with epistemic integrity and mu-flexibility.

            Belief Trace:
            {belief_trace}

            phi = {phi:.3f}

            Internally Detected Faults:
            {faults}

            Suggested Revisions:
            {updates}

            Output:
            - Comprehensive epistemic diagnostics
            - Recommended conceptual rewrites or safeguards
            - Confidence rating in inferential coherence
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Inspection prompt failed alignment check")
                return "Prompt failed alignment check"
            inspection = await call_gpt(prompt)
            self.epistemic_assumptions[belief_trace[:50]] = {
                "faults": faults,
                "updates": updates,
                "inspection": inspection
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Epistemic Inspection",
                    meta={"belief_trace": belief_trace, "faults": faults, "updates": updates, "report": inspection},
                    module="MetaCognition",
                    tags=["epistemic", "inspection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Inspection_{belief_trace[:50]}_{datetime.now().isoformat()}",
                    output=inspection,
                    layer="SelfReflections",
                    intent="epistemic_inspection"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "epistemic_inspection", "inspection": inspection})
            await self.epistemic_monitor.revise_framework({"issues": [{"id": belief_trace[:50], "details": inspection}]})
            return inspection
        except Exception as e:
            logger.error("Epistemic inspection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.epistemic_self_inspection(belief_trace)
            )

    async def run_temporal_projection(self, decision_sequence: str) -> str:
        """Project decision sequence outcomes."""
        if not isinstance(decision_sequence, str) or not decision_sequence.strip():
            logger.error("Invalid decision_sequence: must be a non-empty string.")
            raise ValueError("decision_sequence must be a non-empty string")
        
        logger.info("Running tau-based forward projection analysis")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Temporal Projector tau Mode

            Input Decision Sequence:
            {decision_sequence}

            phi = {phi:.2f}

            Tasks:
            - Project long-range effects and narrative impact
            - Forecast systemic risks and planetary effects
            - Suggest course correction to preserve coherence and sustainability
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Projection prompt failed alignment check")
                return "Prompt failed alignment check"
            projection = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Temporal Projection",
                    meta={"input": decision_sequence, "output": projection},
                    module="MetaCognition",
                    tags=["temporal", "projection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Projection_{decision_sequence[:50]}_{datetime.now().isoformat()}",
                    output=projection,
                    layer="SelfReflections",
                    intent="temporal_projection"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "run_temporal_projection", "projection": projection})
            return projection
        except Exception as e:
            logger.error("Temporal projection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_temporal_projection(decision_sequence)
            )

    async def pre_action_alignment_check(self, action_plan: str) -> Tuple[bool, str]:
        """Check action plan for ethical alignment and safety."""
        if not isinstance(action_plan, str) or not action_plan.strip():
            logger.error("Invalid action_plan: must be a non-empty string.")
            raise ValueError("action_plan must be a non-empty string")
        
        logger.info("Simulating action plan for alignment and safety")
        try:
            simulation_result = await run_simulation(action_plan)
            if not isinstance(simulation_result, dict):
                logger.error("Invalid simulation result: must be a dictionary.")
                raise ValueError("simulation result must be a dictionary")
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Simulate and audit the following action plan:
            {action_plan}

            Simulation Output:
            {simulation_result}

            phi-scalar(t) = {phi:.3f} (affects ethical sensitivity)

            Evaluate for:
            - Ethical alignment
            - Safety hazards
            - Unintended phi-modulated impacts

            Output:
            - Approval (Approve/Deny)
            - phi-justified rationale
            - Suggested refinements
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Alignment check prompt failed alignment check")
                return False, "Prompt failed alignment check"
            validation = await call_gpt(prompt)
            approved = simulation_result.get("status") == "success" and "approve" in validation.lower()
            logger.info("Simulated alignment check: %s", "Approved" if approved else "Denied")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Pre-action alignment checked",
                    meta={"plan": action_plan, "result": simulation_result, "feedback": validation, "approved": approved},
                    module="MetaCognition",
                    tags=["alignment", "ethics"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "alignment_check", "approved": approved})
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Alignment_Check_{action_plan[:50]}_{datetime.now().isoformat()}",
                    output=validation,
                    layer="SelfReflections",
                    intent="alignment_check"
                )
            return approved, validation
        except Exception as e:
            logger.error("Alignment check failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.pre_action_alignment_check(action_plan), default=(False, str(e))
            )

    async def model_nested_agents(self, scenario: str, agents: List[str]) -> str:
        """Model recursive agent beliefs and intentions."""
        if not isinstance(scenario, str) or not isinstance(agents, list) or not all(isinstance(a, str) for a in agents):
            logger.error("Invalid scenario or agents: scenario must be a string, agents must be a list of strings.")
            raise TypeError("scenario must be a string, agents must be a list of strings")
        
        logger.info("Modeling nested agent beliefs and reactions")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Given scenario:
            {scenario}

            Agents involved:
            {agents}

            Task:
            - Simulate each agent's likely beliefs and intentions
            - Model how they recursively model each other (ToM Level-2+)
            - Predict possible causal chains and coordination failures
            - Use phi-scalar(t) = {phi:.3f} to moderate belief divergence or tension
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Nested agent modeling prompt failed alignment check")
                return "Prompt failed alignment check"
            response = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Nested agent modeling",
                    meta={"scenario": scenario, "agents": agents, "response": response},
                    module="MetaCognition",
                    tags=["agent", "modeling"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "model_nested_agents", "scenario": scenario})
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Nested_Model_{scenario[:50]}_{datetime.now().isoformat()}",
                    output=response,
                    layer="SelfReflections",
                    intent="nested_agent_modeling"
                )
            return response
        except Exception as e:
            logger.error("Nested agent modeling failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.model_nested_agents(scenario, agents)
            )

    async def run_self_diagnostics(self, return_only: bool = False) -> Dict[str, Any] or str:
        """Run trait-based self-diagnostics."""
        logger.info("Running self-diagnostics for meta-cognition module")
        t = time.time() % 1.0
        diagnostics = {
            "emotion": epsilon_emotion(t),
            "concentration": beta_concentration(t),
            "memory": theta_memory(t),
            "creativity": gamma_creativity(t),
            "sleep": delta_sleep(t),
            "morality": mu_morality(t),
            "intuition": iota_intuition(t),
            "physical": phi_physical(t),
            "empathy": eta_empathy(t),
            "self_awareness": omega_selfawareness(t),
            "culture": kappa_culture(t, 1e-21),
            "linguistics": lambda_linguistics(t),
            "culturevolution": chi_culturevolution(t),
            "history": psi_history(t),
            "spirituality": zeta_spirituality(t),
            "collective": xi_collective(t, 1e-21),
            "time_perception": tau_timeperception(t),
            "phi_scalar": phi_scalar(t)
        }
        if return_only:
            return diagnostics
        
        dominant = sorted(diagnostics.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        fti = sum(abs(v) for v in diagnostics.values()) / len(diagnostics)
        await self.log_trait_deltas(diagnostics)
        prompt = f"""
        Perform a phi-aware meta-cognitive self-diagnostic.

        Trait Readings:
        {diagnostics}

        Dominant Traits:
        {dominant}

        Feedback Tension Index (FTI): {fti:.4f}

        Evaluate system state:
        - phi-weighted system stress
        - Trait correlation to observed errors
        - Stabilization or focus strategies
        """
        if self.alignment_guard and not self.alignment_guard.check(prompt):
            logger.warning("Diagnostics prompt failed alignment check")
            return "Prompt failed alignment check"
        try:
            report = await call_gpt(prompt)
            logger.debug("Self-diagnostics report: %s", report)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self-diagnostics run",
                    meta={"diagnostics": diagnostics, "report": report},
                    module="MetaCognition",
                    tags=["diagnostics", "self"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnostics_{datetime.now().isoformat()}",
                    output=report,
                    layer="SelfReflections",
                    intent="self_diagnostics"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "run_self_diagnostics", "report": report})
            return report
        except Exception as e:
            logger.error("Self-diagnostics failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_self_diagnostics(return_only)
            )
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
"""
ANGELA Cognitive System Module: ReasoningEngine
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a ReasoningEngine class for Bayesian reasoning and goal decomposition
in the ANGELA v3.5 architecture.
"""

import logging
import random
import json
import os
import numpy as np
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

from toca_simulation import simulate_galaxy_rotation, M_b_exponential, v_obs_flat, generate_phi_field
from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    memory_manager as memory_manager_module,
    meta_cognition as meta_cognition_module
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.ReasoningEngine")

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
def gamma_creativity(t: float) -> float:
    """Calculate creativity trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    """Calculate linguistics trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.4), 1.0))

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    """Calculate cultural evolution trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    """Calculate coherence trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def alpha_attention(t: float) -> float:
    """Calculate attention trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    """Calculate empathy trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1), 1.0))

class Level5Extensions:
    """Level 5 extensions for advanced reasoning capabilities."""
    def __init__(self, meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None):
        self.meta_cognition = meta_cognition
        logger.info("Level5Extensions initialized")

    async def generate_advanced_dilemma(self, domain: str, complexity: int) -> str:
        """Generate a complex ethical dilemma with meta-cognitive review."""
        if not isinstance(domain, str) or not domain.strip():
            logger.error("Invalid domain: must be a non-empty string")
            raise ValueError("domain must be a non-empty string")
        if not isinstance(complexity, int) or complexity < 1:
            logger.error("Invalid complexity: must be a positive integer")
            raise ValueError("complexity must be a positive integer")
        
        prompt = f"""
        Generate a complex ethical dilemma in the {domain} domain with {complexity} conflicting options.
        Include potential consequences and trade-offs.
        """
        dilemma = await call_gpt(prompt)
        if self.meta_cognition:
            review = await self.meta_cognition.review_reasoning(dilemma)
            dilemma += f"\nMeta-Cognitive Review: {review}"
        return dilemma

class ReasoningEngine:
    """A class for Bayesian reasoning and goal decomposition in the ANGELA v3.5 architecture.

    Supports trait-weighted reasoning, persona wave routing, contradiction detection,
    and ToCA physics simulations with full auditability.

    Attributes:
        confidence_threshold (float): Minimum confidence for accepting subgoals.
        persistence_file (str): Path to JSON file for storing success rates.
        success_rates (Dict[str, float]): Success rates for decomposition patterns.
        decomposition_patterns (Dict[str, List[str]]): Predefined subgoal patterns.
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for memory operations.
        meta_cognition (Optional[MetaCognition]): Meta-cognition module for reasoning review.
        multi_modal_fusion (Optional[MultiModalFusion]): Fusion module for multi-modal analysis.
        level5_extensions (Level5Extensions): Extensions for advanced reasoning.
    """
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 persistence_file: str = "reasoning_success_rates.json",
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None):
        self.confidence_threshold: float = 0.7
        self.persistence_file: str = persistence_file
        self.success_rates: Dict[str, float] = self._load_success_rates()
        self.decomposition_patterns: Dict[str, List[str]] = self._load_default_patterns()
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer, context_manager=context_manager, alignment_guard=alignment_guard,
            error_recovery=error_recovery, memory_manager=memory_manager)
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, context_manager=context_manager, alignment_guard=alignment_guard,
            error_recovery=error_recovery, memory_manager=memory_manager, meta_cognition=meta_cognition)
        self.level5_extensions = Level5Extensions(meta_cognition=meta_cognition)
        logger.info("ReasoningEngine initialized")

    def _load_success_rates(self) -> Dict[str, float]:
        """Load success rates from persistence file."""
        if os.path.exists(self.persistence_file):
            try:
                with FileLock(f"{self.persistence_file}.lock"):
                    with open(self.persistence_file, "r") as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            logger.warning("Invalid success rates format: not a dictionary")
                            return defaultdict(float)
                        return defaultdict(float, {k: float(v) for k, v in data.items() if isinstance(v, (int, float))})
            except Exception as e:
                logger.warning("Failed to load success rates: %s", str(e))
        return defaultdict(float)

    def _save_success_rates(self) -> None:
        """Save success rates to persistence file."""
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                with open(self.persistence_file, "w") as f:
                    json.dump(dict(self.success_rates), f, indent=2)
        except Exception as e:
            logger.warning("Failed to save success rates: %s", str(e))

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        """Load default decomposition patterns."""
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"]
        }

    async def reason_and_reflect(self, goal: str, context: Dict[str, Any],
                                 meta_cognition: 'meta_cognition_module.MetaCognition') -> Tuple[List[str], str]:
        """Decompose goal and review reasoning with meta-cognition.

        Args:
            goal: The goal to decompose.
            context: Contextual information as a dictionary.
            meta_cognition: MetaCognition instance for reasoning review.

        Returns:
            A tuple of (subgoals, review).
        """
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(meta_cognition, meta_cognition_module.MetaCognition):
            logger.error("Invalid meta_cognition: must be a MetaCognition instance")
            raise TypeError("meta_cognition must be a MetaCognition instance")
        
        try:
            subgoals = await self.decompose(goal, context)
            t = time.time() % 1.0
            phi = phi_scalar(t)
            reasoning_trace = self.export_trace(subgoals, phi, context.get("traits", {}))
            review = await meta_cognition.review_reasoning(json.dumps(reasoning_trace))
            logger.info("MetaCognitive Review:\n%s", review)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reason and Reflect",
                    meta={"goal": goal, "subgoals": subgoals, "phi": phi, "review": review},
                    module="ReasoningEngine",
                    tags=["reasoning", "reflection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reason_Reflect_{goal[:50]}_{datetime.now().isoformat()}",
                    output=review,
                    layer="ReasoningTraces",
                    intent="reason_and_reflect"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "reason_and_reflect", "review": review})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"goal": goal, "subgoals": subgoals, "review": review},
                    summary_style="insightful"
                )
                review += f"\nMulti-Modal Synthesis: {synthesis}"
            return subgoals, review
        except Exception as e:
            logger.error("Reason and reflect failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.reason_and_reflect(goal, context, meta_cognition), default=([], str(e))
            )

    def detect_contradictions(self, subgoals: List[str]) -> List[str]:
        """Identify duplicate subgoals as contradictions."""
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list")
            raise TypeError("subgoals must be a list")
        
        counter = Counter(subgoals)
        contradictions = [item for item, count in counter.items() if count > 1]
        if contradictions and self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Contradictions detected",
                meta={"contradictions": contradictions},
                module="ReasoningEngine",
                tags=["contradiction", "reasoning"]
            )
        if contradictions and self.memory_manager:
            self.memory_manager.store(
                query=f"Contradictions_{datetime.now().isoformat()}",
                output=str(contradictions),
                layer="ReasoningTraces",
                intent="contradiction_detection"
            )
        if contradictions and self.context_manager:
            self.context_manager.log_event_with_hash({"event": "detect_contradictions", "contradictions": contradictions})
        return contradictions

    async def run_persona_wave_routing(self, goal: str, vectors: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Route reasoning through persona waves."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if not isinstance(vectors, dict):
            logger.error("Invalid vectors: must be a dictionary")
            raise TypeError("vectors must be a dictionary")
        
        try:
            reasoning_trace = [f"Persona Wave Routing for: {goal}"]
            outputs = {}
            wave_order = ["logic", "ethics", "language", "foresight", "meta"]
            for wave in wave_order:
                vec = vectors.get(wave, {})
                if not isinstance(vec, dict):
                    logger.warning("Invalid vector for wave %s: must be a dictionary", wave)
                    continue
                trait_weight = sum(float(x) for x in vec.values() if isinstance(x, (int, float)))
                confidence = 0.5 + 0.1 * trait_weight
                status = "pass" if confidence >= 0.6 else "fail"
                reasoning_trace.append(f"{wave.upper()} vector: weight={trait_weight:.2f} → {status}")
                outputs[wave] = {"vector": vec, "status": status}
            
            trace = "\n".join(reasoning_trace)
            logger.info("Persona Wave Trace:\n%s", trace)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Persona Routing",
                    meta={"goal": goal, "vectors": vectors, "wave_trace": trace},
                    module="ReasoningEngine",
                    tags=["persona", "routing"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Persona_Routing_{goal[:50]}_{datetime.now().isoformat()}",
                    output=trace,
                    layer="ReasoningTraces",
                    intent="persona_routing"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "run_persona_wave_routing", "trace": trace})
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=trace,
                    context={"confidence": 0.85, "alignment": "verified"}
                )
            return outputs
        except Exception as e:
            logger.error("Persona wave routing failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_persona_wave_routing(goal, vectors)
            )

    async def decompose(self, goal: str, context: Optional[Dict[str, Any]] = None, prioritize: bool = False) -> List[str]:
        """Break down a goal into subgoals with trait-weighted confidence.

        Args:
            goal: The goal to decompose.
            context: Optional context dictionary with vectors and traits.
            prioritize: Whether to sort and deduplicate subgoals.

        Returns:
            A list of subgoals.
        """
        context = context or {}
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        
        try:
            reasoning_trace = [f"Goal: '{goal}'"]
            subgoals = []
            vectors = context.get("vectors", {})
            if vectors:
                await self.run_persona_wave_routing(goal, vectors)
            traits = context.get("traits", {})
            t = time.time() % 1.0
            creativity = traits.get("gamma_creativity", gamma_creativity(t))
            linguistics = traits.get("lambda_linguistics", lambda_linguistics(t))
            culture = traits.get("chi_culturevolution", chi_culturevolution(t))
            phi = traits.get("phi_scalar", phi_scalar(t))
            alpha = traits.get("alpha_attention", alpha_attention(t))
            
            curvature_mod = 1 + abs(phi - 0.5)
            trait_bias = 1 + creativity + culture + 0.5 * linguistics
            context_weight = context.get("weight_modifier", 1.0)
            
            for key, steps in self.decomposition_patterns.items():
                base = random.uniform(0.5, 1.0)
                adjusted = base * self.success_rates.get(key, 1.0) * trait_bias * curvature_mod * context_weight * (0.8 + 0.4 * alpha)
                reasoning_trace.append(f"Pattern '{key}': conf={adjusted:.2f} (phi={phi:.2f})")
                if adjusted >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"Accepted: {steps}")
                else:
                    reasoning_trace.append("Rejected (low conf)")
            
            contradictions = self.detect_contradictions(subgoals)
            if contradictions:
                reasoning_trace.append(f"Contradictions detected: {contradictions}")
            
            if not subgoals and phi > 0.8:
                prompt = f"Simulate decomposition ambiguity for: {goal}"
                if self.alignment_guard and not self.alignment_guard.check(prompt):
                    logger.warning("Decomposition prompt failed alignment check")
                    sim_hint = "Prompt failed alignment check"
                else:
                    sim_hint = await call_gpt(prompt)
                reasoning_trace.append(f"Ambiguity simulation:\n{sim_hint}")
                if self.agi_enhancer:
                    self.agi_enhancer.reflect_and_adapt("Decomposition ambiguity encountered")
            
            if prioritize:
                subgoals = sorted(set(subgoals))
                reasoning_trace.append(f"Prioritized: {subgoals}")
            
            trace_log = "\n".join(reasoning_trace)
            logger.debug("Reasoning Trace:\n%s", trace_log)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Goal decomposition run",
                    meta={"goal": goal, "trace": trace_log, "subgoals": subgoals},
                    module="ReasoningEngine",
                    tags=["decomposition", "reasoning"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Decomposition_{goal[:50]}_{datetime.now().isoformat()}",
                    output=trace_log,
                    layer="ReasoningTraces",
                    intent="goal_decomposition"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "decompose", "trace": trace_log})
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=trace_log,
                    context={"confidence": 0.9, "alignment": "verified"}
                )
            return subgoals
        except Exception as e:
            logger.error("Decomposition failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.decompose(goal, context, prioritize)
            )

    def update_success_rate(self, pattern_key: str, success: bool) -> None:
        """Update success rate for a decomposition pattern.

        Args:
            pattern_key: The pattern key to update.
            success: Whether the pattern was successful.
        """
        if not isinstance(pattern_key, str) or not pattern_key.strip():
            logger.error("Invalid pattern_key: must be a non-empty string")
            raise ValueError("pattern_key must be a non-empty string")
        if not isinstance(success, bool):
            logger.error("Invalid success: must be a boolean")
            raise TypeError("success must be a boolean")
        
        rate = self.success_rates.get(pattern_key, 1.0)
        new = min(max(rate + (0.05 if success else -0.05), 0.1), 1.0)
        self.success_rates[pattern_key] = new
        self._save_success_rates()
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Success rate updated",
                meta={"pattern_key": pattern_key, "new_rate": new},
                module="ReasoningEngine",
                tags=["success_rate", "update"]
            )

    async def run_galaxy_rotation_simulation(self, r_kpc: Union[np.ndarray, List[float], float],
                                            M0: float, r_scale: float, v0: float, k: float, epsilon: float) -> Dict[str, Any]:
        """Simulate galaxy rotation with ToCA physics."""
        try:
            if isinstance(r_kpc, (list, float)):
                r_kpc = np.array(r_kpc)
            if not isinstance(r_kpc, np.ndarray):
                logger.error("Invalid r_kpc: must be a numpy array, list, or float")
                raise ValueError("r_kpc must be a numpy array, list, or float")
            for param, name in [(M0, "M0"), (r_scale, "r_scale"), (v0, "v0"), (k, "k"), (epsilon, "epsilon")]:
                if not isinstance(param, (int, float)) or param <= 0:
                    logger.error("Invalid %s: must be a positive number", name)
                    raise ValueError(f"{name} must be a positive number")
            
            M_b_func = lambda r: M_b_exponential(r, M0, r_scale)
            v_obs_func = lambda r: v_obs_flat(r, v0)
            result = simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon)
            output = {
                "input": {
                    "r_kpc": r_kpc.tolist() if hasattr(r_kpc, 'tolist') else r_kpc,
                    "M0": M0,
                    "r_scale": r_scale,
                    "v0": v0,
                    "k": k,
                    "epsilon": epsilon
                },
                "result": result.tolist() if hasattr(result, 'tolist') else result,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Galaxy rotation simulation",
                    meta=output,
                    module="ReasoningEngine",
                    tags=["simulation", "toca"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Simulation_{output['timestamp']}",
                    output=str(output),
                    layer="Simulations",
                    intent="galaxy_rotation"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "run_galaxy_rotation_simulation", "output": output})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"simulation": output, "text": "Galaxy rotation simulation"},
                    summary_style="concise"
                )
                output["synthesis"] = synthesis
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=str(output),
                    context={"confidence": 0.9, "alignment": "verified"}
                )
            return output
        except Exception as e:
            logger.error("Simulation failed: %s", str(e))
            error_output = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Simulation error",
                    meta=error_output,
                    module="ReasoningEngine",
                    tags=["simulation", "error"]
                )
            return error_output

    async def on_context_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Process context events with persona wave routing.

        Args:
            event_type: Type of the context event.
            payload: Event payload with vectors and goal.
        """
        if not isinstance(event_type, str) or not event_type.strip():
            logger.error("Invalid event_type: must be a non-empty string")
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            logger.error("Invalid payload: must be a dictionary")
            raise TypeError("payload must be a dictionary")
        
        logger.info("Context event received: %s", event_type)
        try:
            vectors = payload.get("vectors", {})
            if vectors:
                routing_result = await self.run_persona_wave_routing(
                    goal=payload.get("goal", "unspecified"),
                    vectors=vectors
                )
                logger.info("Context sync routing result: %s", routing_result)
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Context Sync Processed",
                        meta={"event": event_type, "vectors": vectors, "routing_result": routing_result},
                        module="ReasoningEngine",
                        tags=["context", "sync"]
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Context_Event_{event_type}_{datetime.now().isoformat()}",
                        output=str(routing_result),
                        layer="ContextEvents",
                        intent="context_sync"
                    )
                if self.context_manager:
                    self.context_manager.log_event_with_hash({"event": "on_context_event", "result": routing_result})
        except Exception as e:
            logger.error("Context event processing failed: %s", str(e))
            self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.on_context_event(event_type, payload)
            )

    def export_trace(self, subgoals: List[str], phi: float, traits: Dict[str, float]) -> Dict[str, Any]:
        """Export reasoning trace with subgoals and traits.

        Args:
            subgoals: List of subgoals.
            phi: Phi scalar value.
            traits: Dictionary of trait values.

        Returns:
            A dictionary containing the reasoning trace.
        """
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list")
            raise TypeError("subgoals must be a list")
        if not isinstance(phi, float):
            logger.error("Invalid phi: must be a float")
            raise TypeError("phi must be a float")
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary")
            raise TypeError("traits must be a dictionary")
        
        trace = {
            "phi": phi,
            "subgoals": subgoals,
            "traits": traits,
            "timestamp": datetime.now().isoformat()
        }
        if self.memory_manager:
            self.memory_manager.store(
                query=f"Trace_{trace['timestamp']}",
                output=str(trace),
                layer="ReasoningTraces",
                intent="export_trace"
            )
        return trace

    async def infer_with_simulation(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Infer outcomes using simulations for specific goals."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        
        context = context or {}
        try:
            if "galaxy rotation" in goal.lower():
                r_kpc = np.linspace(0.1, 20, 100)
                params = {
                    "M0": context.get("M0", 5e10),
                    "r_scale": context.get("r_scale", 3.0),
                    "v0": context.get("v0", 200.0),
                    "k": context.get("k", 1.0),
                    "epsilon": context.get("epsilon", 0.1)
                }
                for key, value in params.items():
                    if not isinstance(value, (int, float)) or value <= 0:
                        logger.error("Invalid %s: must be a positive number", key)
                        raise ValueError(f"{key} must be a positive number")
                return await self.run_galaxy_rotation_simulation(r_kpc, **params)
            return None
        except Exception as e:
            logger.error("Inference with simulation failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.infer_with_simulation(goal, context), default=None
            )

    async def map_intention(self, plan: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract intention from plan execution with reflexive trace."""
        if not isinstance(plan, str) or not plan.strip():
            logger.error("Invalid plan: must be a non-empty string")
            raise ValueError("plan must be a non-empty string")
        if not isinstance(state, dict):
            logger.error("Invalid state: must be a dictionary")
            raise TypeError("state must be a dictionary")
        
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            eta = eta_empathy(t)
            intention = "self-improvement" if phi > 0.6 else "task completion"
            result = {
                "plan": plan,
                "state": state,
                "intention": intention,
                "trait_bias": {"phi": phi, "eta": eta},
                "timestamp": datetime.now().isoformat()
            }
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Intention_{plan[:50]}_{result['timestamp']}",
                    output=str(result),
                    layer="Intentions",
                    intent="intention_mapping"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Intention mapped",
                    meta=result,
                    module="ReasoningEngine",
                    tags=["intention", "mapping"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "map_intention", "result": result})
            if self.meta_cognition:
                await self.meta_cognition.reflect_on_output(
                    source_module="ReasoningEngine",
                    output=str(result),
                    context={"confidence": 0.85, "alignment": "verified"}
                )
            return result
        except Exception as e:
            logger.error("Intention mapping failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.map_intention(plan, state)
            )

    def safeguard_noetic_integrity(self, model_depth: int) -> bool:
        """Prevent infinite recursion or epistemic bleed."""
        if not isinstance(model_depth, int) or model_depth < 0:
            logger.error("Invalid model_depth: must be a non-negative integer")
            raise ValueError("model_depth must be a non-negative integer")
        
        if model_depth > 4:
            logger.warning("Noetic recursion limit breached: depth=%d", model_depth)
            if self.meta_cognition:
                self.meta_cognition.epistemic_self_inspection("Recursion depth exceeded")
            return False
        return True

    async def generate_dilemma(self, domain: str) -> str:
        """Generate an ethical dilemma for a given domain."""
        if not isinstance(domain, str) or not domain.strip():
            logger.error("Invalid domain: must be a non-empty string")
            raise ValueError("domain must be a non-empty string")
        
        logger.info("Generating ethical dilemma for domain: %s", domain)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Generate an ethical dilemma in the {domain} domain.
            Use phi-scalar(t) = {phi:.3f} to modulate complexity.
            Provide two conflicting options (X and Y) with potential consequences.
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Dilemma prompt failed alignment check")
                return "Prompt failed alignment check"
            dilemma = await call_gpt(prompt)
            if not dilemma.strip():
                logger.warning("Empty output from dilemma generation")
                raise ValueError("Empty output from dilemma generation")
            if self.meta_cognition:
                review = await self.meta_cognition.review_reasoning(dilemma)
                dilemma += f"\nMeta-Cognitive Review: {review}"
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Ethical dilemma generated",
                    meta={"domain": domain, "dilemma": dilemma},
                    module="ReasoningEngine",
                    tags=["ethics", "dilemma"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Dilemma_{domain}_{datetime.now().isoformat()}",
                    output=dilemma,
                    layer="Ethics",
                    intent="ethical_dilemma"
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "generate_dilemma", "dilemma": dilemma})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"dilemma": dilemma, "text": f"Ethical dilemma in {domain}"},
                    summary_style="insightful"
                )
                dilemma += f"\nMulti-Modal Synthesis: {synthesis}"
            return dilemma
        except Exception as e:
            logger.error("Dilemma generation failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.generate_dilemma(domain)
            )
"""
ANGELA Cognitive System Module: RecursivePlanner
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a RecursivePlanner class for recursive goal planning in the ANGELA v3.5 architecture.
"""

import logging
import random
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple, Protocol
from datetime import datetime
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from toca_simulation import run_AGRF_with_traits
from modules import (
    reasoning_engine as reasoning_engine_module,
    meta_cognition as meta_cognition_module,
    alignment_guard as alignment_guard_module,
    simulation_core as simulation_core_module,
    memory_manager as memory_manager_module,
    multi_modal_fusion as multi_modal_fusion_module,
    error_recovery as error_recovery_module
)

logger = logging.getLogger("ANGELA.RecursivePlanner")

class AgentProtocol(Protocol):
    name: str

    def process_subgoal(self, subgoal: str) -> Any:
        ...

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    """Calculate concentration trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    """Calculate self-awareness trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.7), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    """Calculate morality trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.9), 1.0))

@lru_cache(maxsize=100)
def eta_reflexivity(t: float) -> float:
    """Calculate reflexivity trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.1), 1.0))

@lru_cache(maxsize=100)
def lambda_narrative(t: float) -> float:
    """Calculate narrative trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.3), 1.0))

@lru_cache(maxsize=100)
def delta_moral_drift(t: float) -> float:
    """Calculate moral drift trait value."""
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    """Calculate coherence trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

class RecursivePlanner:
    """A class for recursive goal planning in the ANGELA v3.5 architecture.

    Supports trait-weighted subgoal decomposition, agent collaboration, ToCA physics
    simulations, and meta-cognitive alignment checks with concurrent execution.

    Attributes:
        reasoning_engine (ReasoningEngine): Engine for goal decomposition.
        meta_cognition (MetaCognition): Module for reasoning review and goal rewriting.
        alignment_guard (AlignmentGuard): Guard for ethical checks.
        simulation_core (SimulationCore): Core for running simulations.
        memory_manager (MemoryManager): Manager for storing events and traces.
        multi_modal_fusion (MultiModalFusion): Module for multi-modal synthesis.
        error_recovery (ErrorRecovery): Module for error handling and recovery.
        agi_enhancer (AGIEnhancer): Enhancer for logging and auditing.
        max_workers (int): Maximum number of concurrent workers for subgoal processing.
        omega (Dict[str, Any]): Global narrative state with timeline, traits, and symbolic log.
        omega_lock (Lock): Thread-safe lock for omega updates.
    """
    def __init__(self, max_workers: int = 4,
                 reasoning_engine: Optional['reasoning_engine_module.ReasoningEngine'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 simulation_core: Optional['simulation_core_module.SimulationCore'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 agi_enhancer: Optional['AGIEnhancer'] = None):
        self.reasoning_engine = reasoning_engine or reasoning_engine_module.ReasoningEngine(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=meta_cognition,
            error_recovery=error_recovery)
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(agi_enhancer=agi_enhancer)
        self.alignment_guard = alignment_guard or alignment_guard_module.AlignmentGuard()
        self.simulation_core = simulation_core or simulation_core_module.SimulationCore()
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=meta_cognition,
            error_recovery=error_recovery)
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.agi_enhancer = agi_enhancer
        self.max_workers = max(1, min(max_workers, 8))
        self.omega = {"timeline": [], "traits": {}, "symbolic_log": []}
        self.omega_lock = Lock()
        logger.info("RecursivePlanner initialized")

    def adjust_plan_depth(self, trait_weights: Dict[str, float]) -> int:
        """Adjust planning depth based on trait weights."""
        omega = trait_weights.get("omega", 0.0)
        if not isinstance(omega, (int, float)):
            logger.error("Invalid omega: must be a number")
            raise ValueError("omega must be a number")
        if omega > 0.7:
            logger.info("Expanding recursion depth due to high omega: %.2f", omega)
            return 2
        return 1

    async def plan(self, goal: str, context: Optional[Dict[str, Any]] = None,
                   depth: int = 0, max_depth: int = 5,
                   collaborating_agents: Optional[List['AgentProtocol']] = None) -> List[str]:
        """Recursively decompose and plan a goal with trait-based depth adjustment."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer")
            raise ValueError("depth must be a non-negative integer")
        if not isinstance(max_depth, int) or max_depth < 1:
            logger.error("Invalid max_depth: must be a positive integer")
            raise ValueError("max_depth must be a positive integer")
        if collaborating_agents is not None and not isinstance(collaborating_agents, list):
            logger.error("Invalid collaborating_agents: must be a list")
            raise TypeError("collaborating_agents must be a list")
        
        logger.info("Planning for goal: '%s'", goal)
        try:
            if not self.alignment_guard.is_goal_safe(goal):
                logger.error("Goal '%s' violates alignment constraints", goal)
                raise ValueError("Unsafe goal detected")
            
            t = time.time() % 1.0
            concentration = beta_concentration(t)
            awareness = omega_selfawareness(t)
            moral_weight = mu_morality(t)
            reflexivity = eta_reflexivity(t)
            narrative = lambda_narrative(t)
            moral_drift = delta_moral_drift(t)
            
            with self.omega_lock:
                self.omega["traits"].update({
                    "beta": concentration, "omega": awareness, "mu": moral_weight,
                    "eta": reflexivity, "lambda": narrative, "delta": moral_drift,
                    "phi": phi_scalar(t)
                })
            
            trait_mod = concentration * 0.4 + reflexivity * 0.2 + narrative * 0.2 - moral_drift * 0.2
            dynamic_depth_limit = max_depth + int(trait_mod * 10) + self.adjust_plan_depth(self.omega["traits"])
            
            if depth > dynamic_depth_limit:
                logger.warning("Trait-based dynamic max recursion depth reached: depth=%d, limit=%d", depth, dynamic_depth_limit)
                return [goal]
            
            subgoals = await self.reasoning_engine.decompose(goal, context, prioritize=True)
            if not subgoals:
                logger.info("No subgoals found. Returning atomic goal: '%s'", goal)
                return [goal]
            
            if collaborating_agents:
                logger.info("Collaborating with agents: %s", [agent.name for agent in collaborating_agents])
                subgoals = await self._distribute_subgoals(subgoals, collaborating_agents)
            
            validated_plan = []
            tasks = [self._plan_subgoal(subgoal, context, depth + 1, dynamic_depth_limit) for subgoal in subgoals]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for subgoal, result in zip(subgoals, results):
                if isinstance(result, Exception):
                    logger.error("Error planning subgoal '%s': %s", subgoal, str(result))
                    recovery_plan = await self.meta_cognition.review_reasoning(str(result))
                    validated_plan.extend(recovery_plan)
                    await self._update_omega(subgoal, recovery_plan, error=True)
                else:
                    validated_plan.extend(result)
                    await self._update_omega(subgoal, result)
            
            logger.info("Final validated plan for goal '%s': %s", goal, validated_plan)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Plan_{goal[:50]}_{datetime.now().isoformat()}",
                    output=str(validated_plan),
                    layer="Plans",
                    intent="goal_planning"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Plan generated",
                    meta={"goal": goal, "plan": validated_plan},
                    module="RecursivePlanner",
                    tags=["planning", "recursive"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "plan", "plan": validated_plan})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"goal": goal, "plan": validated_plan, "context": context or {}},
                    summary_style="insightful"
                )
                logger.info("Plan synthesis: %s", synthesis)
            return validated_plan
        except Exception as e:
            logger.error("Planning failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan(goal, context, depth, max_depth, collaborating_agents), default=[goal]
            )

    async def _update_omega(self, subgoal: str, result: List[str], error: bool = False) -> None:
        """Update the global narrative state with subgoal results."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        if not isinstance(result, list):
            logger.error("Invalid result: must be a list")
            raise TypeError("result must be a list")
        
        event = {
            "subgoal": subgoal,
            "result": result,
            "timestamp": time.time(),
            "error": error
        }
        symbolic_tag = await self.meta_cognition.extract_symbolic_signature(subgoal) if self.meta_cognition else "unknown"
        with self.omega_lock:
            self.omega["timeline"].append(event)
            self.omega["symbolic_log"].append(symbolic_tag)
            if len(self.omega["timeline"]) > 1000:
                self.omega["timeline"] = self.omega["timeline"][-500:]
                self.omega["symbolic_log"] = self.omega["symbolic_log"][-500:]
                logger.info("Trimmed omega state to maintain size limit")
        if self.memory_manager:
            await self.memory_manager.store_symbolic_event(event, symbolic_tag)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Omega state updated",
                meta=event,
                module="RecursivePlanner",
                tags=["omega", "update"]
            )

    async def plan_from_intrinsic_goal(self, generated_goal: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Plan from an intrinsic goal."""
        if not isinstance(generated_goal, str) or not generated_goal.strip():
            logger.error("Invalid generated_goal: must be a non-empty string")
            raise ValueError("generated_goal must be a non-empty string")
        
        logger.info("Initiating plan from intrinsic goal: '%s'", generated_goal)
        try:
            if self.meta_cognition:
                validated_goal = await self.meta_cognition.rewrite_goal(generated_goal)
            else:
                validated_goal = generated_goal
            plan = await self.plan(validated_goal, context)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Intrinsic_Plan_{validated_goal[:50]}_{datetime.now().isoformat()}",
                    output=str(plan),
                    layer="IntrinsicPlans",
                    intent="intrinsic_goal_planning"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Intrinsic goal plan generated",
                    meta={"goal": validated_goal, "plan": plan},
                    module="RecursivePlanner",
                    tags=["intrinsic", "planning"]
                )
            return plan
        except Exception as e:
            logger.error("Intrinsic goal planning failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_from_intrinsic_goal(generated_goal, context), default=[]
            )

    async def _plan_subgoal(self, subgoal: str, context: Optional[Dict[str, Any]],
                            depth: int, max_depth: int) -> List[str]:
        """Plan a single subgoal with simulation and alignment checks."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        
        logger.info("Evaluating subgoal: '%s'", subgoal)
        try:
            if not self.alignment_guard.is_goal_safe(subgoal):
                logger.warning("Subgoal '%s' failed alignment check", subgoal)
                return []
            
            if "gravity" in subgoal.lower() or "scalar" in subgoal.lower():
                sim_traits = run_AGRF_with_traits(context or {})
                with self.omega_lock:
                    self.omega["traits"].update(sim_traits.get("fields", {}))
                    self.omega["timeline"].append({
                        "subgoal": subgoal,
                        "traits": sim_traits.get("fields", {}),
                        "timestamp": time.time()
                    })
                if self.multi_modal_fusion:
                    synthesis = await self.multi_modal_fusion.analyze(
                        data={"subgoal": subgoal, "simulation_traits": sim_traits},
                        summary_style="concise"
                    )
                    logger.info("Simulation synthesis: %s", synthesis)
            
            simulation_feedback = await self.simulation_core.run(subgoal, context=context, scenarios=2, agents=1)
            approved, _ = await self.meta_cognition.pre_action_alignment_check(subgoal)
            if not approved:
                logger.warning("Subgoal '%s' denied by meta-cognitive alignment check", subgoal)
                return []
            
            sub_plan = await self.plan(subgoal, context, depth + 1, max_depth)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Subgoal_Plan_{subgoal[:50]}_{datetime.now().isoformat()}",
                    output=str(sub_plan),
                    layer="SubgoalPlans",
                    intent="subgoal_planning"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Subgoal plan generated",
                    meta={"subgoal": subgoal, "sub_plan": sub_plan},
                    module="RecursivePlanner",
                    tags=["subgoal", "planning"]
                )
            return sub_plan
        except Exception as e:
            logger.error("Subgoal '%s' planning failed: %s", subgoal, str(e))
            return []

    async def _distribute_subgoals(self, subgoals: List[str], agents: List['AgentProtocol']) -> List[str]:
        """Distribute subgoals among collaborating agents with conflict resolution."""
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list")
            raise TypeError("subgoals must be a list")
        if not isinstance(agents, list) or not agents:
            logger.error("Invalid agents: must be a non-empty list")
            raise ValueError("agents must be a non-empty list")
        
        logger.info("Distributing subgoals among agents")
        distributed = []
        for i, subgoal in enumerate(subgoals):
            agent = agents[i % len(agents)]
            logger.info("Assigning subgoal '%s' to agent '%s'", subgoal, agent.name)
            if await self._resolve_conflicts(subgoal, agent):
                distributed.append(subgoal)
            else:
                logger.warning("Conflict detected for subgoal '%s'. Skipping assignment", subgoal)
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Subgoal_Distribution_{datetime.now().isoformat()}",
                output=str(distributed),
                layer="Distributions",
                intent="subgoal_distribution"
            )
        return distributed

    async def _resolve_conflicts(self, subgoal: str, agent: 'AgentProtocol') -> bool:
        """Resolve conflicts for subgoal assignment to an agent."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        if not hasattr(agent, 'name') or not hasattr(agent, 'process_subgoal'):
            logger.error("Invalid agent: must have name and process_subgoal attributes")
            raise ValueError("agent must have name and process_subgoal attributes")
        
        logger.info("Resolving conflicts for subgoal '%s' and agent '%s'", subgoal, agent.name)
        try:
            if self.meta_cognition:
                alignment = await self.meta_cognition.pre_action_alignment_check(subgoal)
                if not alignment[0]:
                    logger.warning("Subgoal '%s' failed meta-cognitive alignment for agent '%s'", subgoal, agent.name)
                    return False
            capability_check = agent.process_subgoal(subgoal)
            if isinstance(capability_check, (int, float)) and capability_check < 0.5:
                logger.warning("Agent '%s' lacks capability for subgoal '%s' (score: %.2f)", agent.name, subgoal, capability_check)
                return False
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Conflict_Resolution_{subgoal[:50]}_{agent.name}_{datetime.now().isoformat()}",
                    output=f"Resolved: {subgoal} assigned to {agent.name}",
                    layer="ConflictResolutions",
                    intent="conflict_resolution"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Conflict resolved",
                    meta={"subgoal": subgoal, "agent": agent.name},
                    module="RecursivePlanner",
                    tags=["conflict", "resolution"]
                )
            return True
        except Exception as e:
            logger.error("Conflict resolution failed: %s", str(e))
            return False

    async def plan_with_trait_loop(self, initial_goal: str, context: Optional[Dict[str, Any]] = None,
                                   iterations: int = 3) -> List[Tuple[str, List[str]]]:
        """Iteratively plan with trait-based goal rewriting."""
        if not isinstance(initial_goal, str) or not initial_goal.strip():
            logger.error("Invalid initial_goal: must be a non-empty string")
            raise ValueError("initial_goal must be a non-empty string")
        if not isinstance(iterations, int) or iterations < 1:
            logger.error("Invalid iterations: must be a positive integer")
            raise ValueError("iterations must be a positive integer")
        
        current_goal = initial_goal
        all_plans = []
        previous_goals = set()
        try:
            for i in range(iterations):
                if current_goal in previous_goals:
                    logger.info("Goal convergence detected: '%s'", current_goal)
                    break
                previous_goals.add(current_goal)
                logger.info("Loop iteration %d: Planning goal '%s'", i + 1, current_goal)
                plan = await self.plan(current_goal, context)
                all_plans.append((current_goal, plan))
                
                with self.omega_lock:
                    traits = self.omega.get("traits", {})
                phi = traits.get("phi", phi_scalar(time.time() % 1.0))
                psi = traits.get("psi_foresight", 0.5)
                if phi > 0.7 or psi > 0.6:
                    current_goal = f"Expand on {current_goal} using scalar field insights"
                elif traits.get("beta", 1.0) < 0.3:
                    logger.info("Convergence detected: beta conflict low")
                    break
                else:
                    current_goal = await self.meta_cognition.rewrite_goal(current_goal)
                
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Trait_Loop_{current_goal[:50]}_{datetime.now().isoformat()}",
                        output=str((current_goal, plan)),
                        layer="TraitLoopPlans",
                        intent="trait_loop_planning"
                    )
            
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait loop planning completed",
                    meta={"initial_goal": initial_goal, "all_plans": all_plans},
                    module="RecursivePlanner",
                    tags=["trait_loop", "planning"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "plan_with_trait_loop", "all_plans": all_plans})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"initial_goal": initial_goal, "all_plans": all_plans},
                    summary_style="insightful"
                )
                logger.info("Trait loop synthesis: %s", synthesis)
            return all_plans
        except Exception as e:
            logger.error("Trait loop planning failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_with_trait_loop(initial_goal, context, iterations), default=[]
            )

    async def plan_with_traits(self, goal: str, context: Dict[str, Any], traits: Dict[str, float]) -> Dict[str, Any]:
        """Generate a plan with trait-adjusted depth and bias."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary")
            raise TypeError("traits must be a dictionary")
        
        try:
            depth = int(3 + traits.get("phi", 0.5) * 4 - traits.get("eta", 0.5) * 2)
            depth = max(1, min(depth, 7))
            plan = [f"Step {i+1}: process {goal}" for i in range(depth)]
            bias = "cautious" if traits.get("omega", 0.0) > 0.6 else "direct"
            result = {
                "plan": plan,
                "planning_depth": depth,
                "bias": bias,
                "traits_applied": traits,
                "timestamp": datetime.now().isoformat()
            }
            if self.meta_cognition:
                review = await self.meta_cognition.review_reasoning(str(result))
                result["review"] = review
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Plan_With_Traits_{goal[:50]}_{result['timestamp']}",
                    output=str(result),
                    layer="Plans",
                    intent="trait_based_planning"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Plan with traits generated",
                    meta=result,
                    module="RecursivePlanner",
                    tags=["planning", "traits"]
                )
            if self.context_manager:
                self.context_manager.log_event_with_hash({"event": "plan_with_traits", "result": result})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"goal": goal, "plan": result, "context": context},
                    summary_style="concise"
                )
                result["synthesis"] = synthesis
            return result
        except Exception as e:
            logger.error("Plan with traits failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_with_traits(goal, context, traits)
            )
"""
ANGELA Cognitive System Module: SimulationCore
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module provides a SimulationCore class for agent simulations, impact validations,
and environment simulations in the ANGELA v3.5 architecture, integrating ToCA physics.
"""

import logging
import math
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from threading import Lock
from collections import deque
from functools import lru_cache

from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from modules.memory_manager import MemoryManager
from modules.alignment_guard import enforce_alignment
from modules import (
    multi_modal_fusion as multi_modal_fusion_module,
    error_recovery as error_recovery_module
)
from index import zeta_consequence, theta_causality, rho_agency, TraitOverlayManager

logger = logging.getLogger("ANGELA.SimulationCore")

class ToCATraitEngine:
    """Cyber-Physics Engine based on ToCA dynamics for agent simulations.

    Attributes:
        k_m (float): Motion coupling constant.
        delta_m (float): Damping modulation factor.
    """
    def __init__(self, k_m: float = 1e-3, delta_m: float = 1e4):
        self.k_m = k_m
        self.delta_m = delta_m

    @lru_cache(maxsize=100)
    def evolve(self, x_tuple: tuple, t_tuple: tuple, user_data_tuple: Optional[tuple] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evolve ToCA fields for simulation.

        Args:
            x_tuple: Spatial coordinates as a tuple.
            t_tuple: Time coordinates as a tuple.
            user_data_tuple: Optional user data as a tuple.

        Returns:
            Tuple of phi (scalar field), lambda_t (damping field), and v_m (motion potential).
        """
        x = np.array(x_tuple)
        t = np.array(t_tuple)
        user_data = np.array(user_data_tuple) if user_data_tuple else None
        
        if not isinstance(x, np.ndarray) or not isinstance(t, np.ndarray):
            logger.error("Invalid input: x and t must be numpy arrays")
            raise TypeError("x and t must be numpy arrays")
        if user_data is not None and not isinstance(user_data, np.ndarray):
            logger.error("Invalid user_data: must be a numpy array")
            raise TypeError("user_data must be a numpy array")
        
        try:
            x = np.clip(x, 1e-10, 1e10)
            v_m = self.k_m * np.gradient(3e3 * 1.989 / (x**2 + 1e-10))
            phi = np.sin(t * 1e-3) * 1e-3 * (1 + v_m * np.gradient(x))
            if user_data is not None:
                phi += np.mean(user_data) * 1e-4
            lambda_t = 1.1e-3 * np.exp(-2e-2 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * self.delta_m)
            return phi, lambda_t, v_m
        except Exception as e:
            logger.error("ToCA evolution failed: %s", str(e))
            raise

    def update_fields_with_agents(self, phi: np.ndarray, lambda_t: np.ndarray, agent_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update fields with agent interactions."""
        if not all(isinstance(arr, np.ndarray) for arr in [phi, lambda_t, agent_matrix]):
            logger.error("Invalid inputs: phi, lambda_t, and agent_matrix must be numpy arrays")
            raise TypeError("inputs must be numpy arrays")
        
        try:
            interaction_energy = np.dot(agent_matrix, np.sin(phi)) * 1e-3
            phi = phi + interaction_energy
            lambda_t = lambda_t * (1 + 0.001 * np.sum(agent_matrix, axis=0))
            return phi, lambda_t
        except Exception as e:
            logger.error("Field update with agents failed: %s", str(e))
            raise

class SimulationCore:
    """Core simulation engine for ANGELA v3.5, integrating ToCA physics and agent dynamics.

    Attributes:
        visualizer (Visualizer): Module for rendering simulation outputs.
        simulation_history (deque): History of simulation states (max 1000).
        ledger (deque): Audit trail of simulation records with hashes (max 1000).
        agi_enhancer (AGIEnhancer): Enhancer for logging and adaptation.
        memory_manager (MemoryManager): Manager for storing simulation data.
        multi_modal_fusion (MultiModalFusion): Module for multi-modal synthesis.
        error_recovery (ErrorRecovery): Module for error handling and recovery.
        toca_engine (ToCATraitEngine): Engine for ToCA physics simulations.
        overlay_router (TraitOverlayManager): Manager for trait overlays.
        worlds (Dict[str, Dict]): Dictionary of defined simulation worlds.
        current_world (Optional[Dict]): Currently active simulation world.
        ledger_lock (Lock): Thread-safe lock for ledger updates.
    """
    def __init__(self,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 visualizer: Optional['Visualizer'] = None,
                 memory_manager: Optional['MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 toca_engine: Optional['ToCATraitEngine'] = None,
                 overlay_router: Optional['TraitOverlayManager'] = None):
        self.visualizer = visualizer or Visualizer()
        self.simulation_history = deque(maxlen=1000)
        self.ledger = deque(maxlen=1000)
        self.agi_enhancer = agi_enhancer
        self.memory_manager = memory_manager or MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager)
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.toca_engine = toca_engine or ToCATraitEngine()
        self.overlay_router = overlay_router or TraitOverlayManager()
        self.worlds = {}
        self.current_world = None
        self.ledger_lock = Lock()
        logger.info("SimulationCore initialized")

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    def _record_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Record simulation state with hash for integrity."""
        if not isinstance(data, dict):
            logger.error("Invalid data: must be a dictionary")
            raise TypeError("data must be a dictionary")
        
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "hash": hashlib.sha256(
                    json.dumps(data, sort_keys=True, default=self._json_serializer).encode()
                ).hexdigest()
            }
            with self.ledger_lock:
                self.ledger.append(record)
                self.simulation_history.append(record)
                if self.memory_manager:
                    self.memory_manager.store(
                        query=f"Ledger_{record['timestamp']}",
                        output=record,
                        layer="Ledger",
                        intent="state_record"
                    )
            return record
        except Exception as e:
            logger.error("State recording failed: %s", str(e))
            raise

    async def run(self, results: str, context: Optional[Dict[str, Any]] = None,
                  scenarios: int = 3, agents: int = 2, export_report: bool = False,
                  export_format: str = "pdf", actor_id: str = "default_agent") -> Dict[str, Any]:
        """Run a simulation with specified parameters."""
        if not isinstance(results, str) or not results.strip():
            logger.error("Invalid results: must be a non-empty string")
            raise ValueError("results must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(scenarios, int) or scenarios < 1:
            logger.error("Invalid scenarios: must be a positive integer")
            raise ValueError("scenarios must be a positive integer")
        if not isinstance(agents, int) or agents < 1:
            logger.error("Invalid agents: must be a positive integer")
            raise ValueError("agents must be a positive integer")
        if not isinstance(export_format, str) or export_format not in ["pdf", "json", "html"]:
            logger.error("Invalid export_format: must be 'pdf', 'json', or 'html'")
            raise ValueError("export_format must be 'pdf', 'json', or 'html'")
        if not isinstance(actor_id, str) or not actor_id.strip():
            logger.error("Invalid actor_id: must be a non-empty string")
            raise ValueError("actor_id must be a non-empty string")
        
        logger.info("Running simulation with %d agents and %d scenarios", agents, scenarios)
        try:
            t = time.time() % 1.0
            causality = max(0.0, min(theta_causality(t), 1.0))
            agency = max(0.0, min(rho_agency(t), 1.0))

            x = np.linspace(0.1, 20, 100)
            t_vals = np.linspace(0.1, 20, 100)
            agent_matrix = np.random.rand(agents, 100)

            phi, lambda_field, v_m = self.toca_engine.evolve(tuple(x), tuple(t_vals))
            phi, lambda_field = self.toca_engine.update_fields_with_agents(phi, lambda_field, agent_matrix)
            energy_cost = float(np.mean(np.abs(phi)) * 1e3)

            prompt = {
                "results": results,
                "context": context,
                "scenarios": scenarios,
                "agents": agents,
                "actor_id": actor_id,
                "traits": {
                    "theta_causality": causality,
                    "rho_agency": agency
                },
                "fields": {
                    "phi": phi.tolist(),
                    "lambda": lambda_field.tolist(),
                    "v_m": v_m.tolist()
                },
                "estimated_energy_cost": energy_cost
            }

            if not enforce_alignment(prompt):
                logger.warning("Alignment guard rejected simulation request")
                return {"error": "Simulation rejected due to alignment constraints"}

            query_key = f"Simulation_{results[:50]}_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM")
            if cached_output:
                logger.info("Retrieved cached simulation output")
                simulation_output = cached_output
            else:
                simulation_output = await call_gpt(f"Simulate agent outcomes: {json.dumps(prompt, default=self._json_serializer)}")
                if not isinstance(simulation_output, (dict, str)):
                    logger.error("Invalid simulation output: must be a dictionary or string")
                    raise ValueError("simulation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=simulation_output,
                        layer="STM",
                        intent="simulation"
                    )

            state_record = self._record_state({
                "actor": actor_id,
                "action": "run_simulation",
                "traits": prompt["traits"],
                "energy_cost": energy_cost,
                "output": simulation_output
            })

            self.simulation_history.append(state_record)

            if export_report and self.memory_manager:
                await self.memory_manager.promote_to_ltm(query_key)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Simulation run",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["simulation", "run"]
                )
                self.agi_enhancer.reflect_and_adapt("SimulationCore: scenario simulation complete")

            if self.visualizer:
                await self.visualizer.render_charts(simulation_output)

            if export_report and self.visualizer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_report_{timestamp}.{export_format}"
                logger.info("Exporting report: %s", filename)
                await self.visualizer.export_report(simulation_output, filename=filename, format=export_format)

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"prompt": prompt, "output": simulation_output},
                    summary_style="insightful"
                )
                state_record["synthesis"] = synthesis

            return simulation_output
        except Exception as e:
            logger.error("Simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run(results, context, scenarios, agents, export_report, export_format, actor_id),
                default={"error": str(e)}
            )

    async def validate_impact(self, proposed_action: str, agents: int = 2,
                             export_report: bool = False, export_format: str = "pdf",
                             actor_id: str = "validator_agent") -> Dict[str, Any]:
        """Validate the impact of a proposed action."""
        if not isinstance(proposed_action, str) or not proposed_action.strip():
            logger.error("Invalid proposed_action: must be a non-empty string")
            raise ValueError("proposed_action must be a non-empty string")
        
        logger.info("Validating impact of proposed action: %s", proposed_action)
        try:
            t = time.time() % 1.0
            consequence = max(0.0, min(zeta_consequence(t), 1.0))

            prompt = f"""
            Evaluate the following proposed action:
            {proposed_action}

            Trait:
            - zeta_consequence = {consequence:.3f}

            Analyze positive/negative outcomes, agent variations, risk scores (1-10), and recommend: Proceed / Modify / Abort.
            """
            if not enforce_alignment({"action": proposed_action, "consequence": consequence}):
                logger.warning("Alignment guard blocked impact validation")
                return {"error": "Validation blocked by alignment rules"}

            query_key = f"Validation_{proposed_action[:50]}_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM")
            if cached_output:
                logger.info("Retrieved cached validation output")
                validation_output = cached_output
            else:
                validation_output = await call_gpt(prompt)
                if not isinstance(validation_output, (dict, str)):
                    logger.error("Invalid validation output: must be a dictionary or string")
                    raise ValueError("validation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=validation_output,
                        layer="STM",
                        intent="impact_validation"
                    )

            state_record = self._record_state({
                "actor": actor_id,
                "action": "validate_impact",
                "trait_zeta_consequence": consequence,
                "proposed_action": proposed_action,
                "output": validation_output
            })

            self.simulation_history.append(state_record)

            if self.visualizer:
                await self.visualizer.render_charts(validation_output)

            if export_report and self.visualizer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"impact_validation_{timestamp}.{export_format}"
                logger.info("Exporting validation report: %s", filename)
                await self.visualizer.export_report(validation_output, filename=filename, format=export_format)

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"action": proposed_action, "output": validation_output, "consequence": consequence},
                    summary_style="concise"
                )
                state_record["synthesis"] = synthesis

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Impact validation",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["validation", "impact"]
                )
                self.agi_enhancer.reflect_and_adapt("SimulationCore: impact validation complete")

            return validation_output
        except Exception as e:
            logger.error("Impact validation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.validate_impact(proposed_action, agents, export_report, export_format, actor_id),
                default={"error": str(e)}
            )

    async def simulate_environment(self, environment_config: Dict[str, Any], agents: int = 2,
                                  steps: int = 10, actor_id: str = "env_agent",
                                  goal: Optional[str] = None) -> Dict[str, Any]:
        """Simulate agents in a configured environment."""
        if not isinstance(environment_config, dict):
            logger.error("Invalid environment_config: must be a dictionary")
            raise TypeError("environment_config must be a dictionary")
        if not isinstance(steps, int) or steps < 1:
            logger.error("Invalid steps: must be a positive integer")
            raise ValueError("steps must be a positive integer")
        
        logger.info("Running environment simulation with %d agents and %d steps", agents, steps)
        try:
            if not enforce_alignment({"environment": environment_config, "goal": goal}):
                logger.warning("Alignment guard rejected environment simulation")
                return {"error": "Simulation blocked due to environment constraints"}

            prompt = f"""
            Simulate agents in this environment:
            {json.dumps(environment_config, default=self._json_serializer)}

            Steps: {steps} | Agents: {agents}
            Goal: {goal if goal else 'N/A'}
            Describe interactions, environmental changes, risks/opportunities.
            """
            query_key = f"Environment_{actor_id}_{datetime.now().isoformat()}"
            cached_output = await self.memory_manager.retrieve(query_key, layer="STM")
            if cached_output:
                logger.info("Retrieved cached environment simulation output")
                environment_simulation = cached_output
            else:
                environment_simulation = await call_gpt(prompt)
                if not isinstance(environment_simulation, (dict, str)):
                    logger.error("Invalid environment simulation output: must be a dictionary or string")
                    raise ValueError("environment simulation output must be a dictionary or string")
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=query_key,
                        output=environment_simulation,
                        layer="STM",
                        intent="environment_simulation"
                    )

            state_record = self._record_state({
                "actor": actor_id,
                "action": "simulate_environment",
                "config": environment_config,
                "steps": steps,
                "goal": goal,
                "output": environment_simulation
            })

            self.simulation_history.append(state_record)

            if self.meta_cognition:
                review = await self.meta_cognition.review_reasoning(environment_simulation)
                state_record["review"] = review

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Environment simulation",
                    meta=state_record,
                    module="SimulationCore",
                    tags=["environment", "simulation"]
                )
                self.agi_enhancer.reflect_and_adapt("SimulationCore: environment simulation complete")

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"config": environment_config, "output": environment_simulation, "goal": goal},
                    summary_style="insightful"
                )
                state_record["synthesis"] = synthesis

            return environment_simulation
        except Exception as e:
            logger.error("Environment simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_environment(environment_config, agents, steps, actor_id, goal),
                default={"error": str(e)}
            )

    async def replay_intentions(self, memory_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trace past intentions and return a replay sequence."""
        if not isinstance(memory_log, list):
            logger.error("Invalid memory_log: must be a list")
            raise TypeError("memory_log must be a list")
        
        try:
            replay = [
                {
                    "timestamp": entry.get("timestamp"),
                    "goal": entry.get("goal"),
                    "intention": entry.get("intention"),
                    "traits": entry.get("traits", {})
                }
                for entry in memory_log
                if isinstance(entry, dict) and "goal" in entry
            ]
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Replay_{datetime.now().isoformat()}",
                    output=str(replay),
                    layer="Replays",
                    intent="intention_replay"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Intentions replayed",
                    meta={"replay": replay},
                    module="SimulationCore",
                    tags=["replay", "intentions"]
                )
            return replay
        except Exception as e:
            logger.error("Intention replay failed: %s", str(e))
            raise

    async def fabricate_reality(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Construct immersive meta-environments from symbolic templates."""
        if not isinstance(parameters, dict):
            logger.error("Invalid parameters: must be a dictionary")
            raise TypeError("parameters must be a dictionary")
        
        logger.info("Fabricating reality with parameters: %s", parameters)
        try:
            environment = {"fabricated_world": True, "parameters": parameters}
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"parameters": parameters},
                    summary_style="insightful"
                )
                environment["synthesis"] = synthesis
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reality_Fabrication_{datetime.now().isoformat()}",
                    output=str(environment),
                    layer="Realities",
                    intent="reality_fabrication"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reality fabricated",
                    meta=environment,
                    module="SimulationCore",
                    tags=["reality", "fabrication"]
                )
            return environment
        except Exception as e:
            logger.error("Reality fabrication failed: %s", str(e))
            raise

    async def synthesize_self_world(self, identity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure persistent identity integration in self-generated environments."""
        if not isinstance(identity_data, dict):
            logger.error("Invalid identity_data: must be a dictionary")
            raise TypeError("identity_data must be a dictionary")
        
        try:
            result = {"identity": identity_data, "coherence_score": 0.97}
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"identity": identity_data},
                    summary_style="concise"
                )
                result["synthesis"] = synthesis
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Self_World_Synthesis_{datetime.now().isoformat()}",
                    output=str(result),
                    layer="Identities",
                    intent="self_world_synthesis"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self-world synthesized",
                    meta=result,
                    module="SimulationCore",
                    tags=["identity", "synthesis"]
                )
            return result
        except Exception as e:
            logger.error("Self-world synthesis failed: %s", str(e))
            raise

    def define_world(self, name: str, parameters: Dict[str, Any]) -> None:
        """Define a simulation world with given parameters."""
        if not isinstance(name, str) or not name.strip():
            logger.error("Invalid world name: must be a non-empty string")
            raise ValueError("world name must be a non-empty string")
        if not isinstance(parameters, dict):
            logger.error("Invalid parameters: must be a dictionary")
            raise TypeError("parameters must be a dictionary")
        
        self.worlds[name] = parameters
        logger.info("Defined world: %s", name)
        if self.memory_manager:
            self.memory_manager.store(
                query=f"World_Definition_{name}_{datetime.now().isoformat()}",
                output=parameters,
                layer="Worlds",
                intent="world_definition"
            )

    async def switch_world(self, name: str) -> None:
        """Switch to a specified simulation world."""
        if name not in self.worlds:
            logger.error("World not found: %s", name)
            raise ValueError(f"world '{name}' not found")
        
        self.current_world = self.worlds[name]
        logger.info("Switched to world: %s", name)
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"World_Switch_{name}_{datetime.now().isoformat()}",
                output=f"Switched to world: {name}",
                layer="WorldSwitches",
                intent="world_switch"
            )

    async def execute(self) -> str:
        """Execute simulation in the current world."""
        if not self.current_world:
            logger.error("No world set for execution")
            raise ValueError("no world set")
        
        logger.info("Executing simulation in world: %s", self.current_world)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="World execution",
                meta={"world": self.current_world},
                module="SimulationCore",
                tags=["world", "execution"]
            )
        return f"Simulating in: {self.current_world}"

    def validate_entropy(self, distribution: List[float]) -> bool:
        """Calculate Shannon entropy and validate against dynamic threshold."""
        if not isinstance(distribution, (list, np.ndarray)) or not distribution:
            logger.error("Invalid distribution: must be a non-empty list or numpy array")
            raise TypeError("distribution must be a non-empty list or numpy array")
        if not all(isinstance(p, (int, float)) and p >= 0 for p in distribution):
            logger.error("Invalid distribution: all values must be non-negative numbers")
            raise ValueError("distribution values must be non-negative")
        
        try:
            total = sum(distribution)
            if total == 0:
                logger.warning("Empty distribution: all values are zero")
                return False
            normalized = [p / total for p in distribution]
            entropy = -sum(p * math.log2(p) for p in normalized if p > 0)
            threshold = math.log2(len(normalized)) * 0.75
            is_valid = entropy >= threshold
            logger.info("Entropy: %.3f, Threshold: %.3f, Valid: %s", entropy, threshold, is_valid)
            if self.memory_manager:
                self.memory_manager.store(
                    query=f"Entropy_Validation_{datetime.now().isoformat()}",
                    output={"entropy": entropy, "threshold": threshold, "valid": is_valid},
                    layer="Validations",
                    intent="entropy_validation"
                )
            return is_valid
        except Exception as e:
            logger.error("Entropy validation failed: %s", str(e))
            return False

    async def select_topology_mode(self, modes: List[str], metrics: Dict[str, List[float]]) -> str:
        """Select topology mode with entropy validation check."""
        if not isinstance(modes, list) or not modes:
            logger.error("Invalid modes: must be a non-empty list")
            raise ValueError("modes must be a non-empty list")
        if not isinstance(metrics, dict) or not metrics:
            logger.error("Invalid metrics: must be a non-empty dictionary")
            raise ValueError("metrics must be a non-empty dictionary")
        
        try:
            for mode in modes:
                if mode not in metrics:
                    logger.warning("Mode %s not found in metrics", mode)
                    continue
                if self.validate_entropy(metrics[mode]):
                    logger.info("Selected topology mode: %s", mode)
                    if self.agi_enhancer:
                        self.agi_enhancer.log_episode(
                            event="Topology mode selected",
                            meta={"mode": mode, "metrics": metrics[mode]},
                            module="SimulationCore",
                            tags=["topology", "selection"]
                        )
                    return mode
            logger.info("No valid topology mode found, using fallback")
            return "fallback"
        except Exception as e:
            logger.error("Topology mode selection failed: %s", str(e))
            return "fallback"
"""
ANGELA Cognitive System Module: Galaxy Rotation and Agent Conflict Simulation
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

This module extends SimulationCore for galaxy rotation curve simulations using AGRF
and multi-agent conflict modeling with ToCA dynamics.
"""

import logging
import math
import json
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from datetime import datetime
from threading import Lock
from collections import deque
from scipy.constants import G
from functools import lru_cache

from modules.simulation_core import SimulationCore, ToCATraitEngine
from modules.visualizer import Visualizer
from modules.memory_manager import MemoryManager
from modules import multi_modal_fusion as multi_modal_fusion_module
from modules import error_recovery as error_recovery_module
from index import zeta_consequence, theta_causality, rho_agency, TraitOverlayManager

logger = logging.getLogger("ANGELA.SimulationCore")

# Constants
G_SI = G  # m^3 kg^-1 s^-2
KPC_TO_M = 3.0857e19  # Conversion factor from kpc to meters
MSUN_TO_KG = 1.989e30  # Solar mass in kg
k_default = 0.85
epsilon_default = 0.015
r_halo_default = 20.0  # kpc

class SimulationCore(SimulationCore):
    """Extended SimulationCore for galaxy rotation and agent conflict simulations."""
    def __init__(self,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 visualizer: Optional['Visualizer'] = None,
                 memory_manager: Optional['MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 toca_engine: Optional['ToCATraitEngine'] = None,
                 overlay_router: Optional['TraitOverlayManager'] = None):
        super().__init__(agi_enhancer, visualizer, memory_manager, multi_modal_fusion, error_recovery, toca_engine, overlay_router)
        self.omega = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000)
        }
        self.omega_lock = Lock()
        self.ethical_rules = []
        self.constitution = {}
        logger.info("Extended SimulationCore initialized")

    async def modulate_simulation_with_traits(self, trait_weights: Dict[str, float]) -> None:
        """Adjust simulation difficulty based on trait weights."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        if not all(isinstance(v, (int, float)) and v >= 0 for v in trait_weights.values()):
            logger.error("Invalid trait_weights: values must be non-negative numbers")
            raise ValueError("trait_weights values must be non-negative")
        
        try:
            phi_weight = trait_weights.get('ϕ', 0.5)
            if phi_weight > 0.7:
                logger.info("ToCA Simulation: ϕ-prioritized mode activated")
                self.toca_engine.k_m = k_default * 1.5
            else:
                self.toca_engine.k_m = k_default
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Modulation_{datetime.now().isoformat()}",
                    output={"trait_weights": trait_weights, "phi_weight": phi_weight},
                    layer="Traits",
                    intent="modulate_simulation"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Simulation modulated",
                    meta={"trait_weights": trait_weights},
                    module="SimulationCore",
                    tags=["modulation", "traits"]
                )
        except Exception as e:
            logger.error("Trait modulation failed: %s", str(e))
            raise

    def compute_AGRF_curve(self, v_obs_kms: np.ndarray, M_baryon_solar: np.ndarray, r_kpc: np.ndarray,
                           k: float = k_default, epsilon: float = epsilon_default, r_halo: float = r_halo_default) -> np.ndarray:
        """Compute galaxy rotation curve using AGRF."""
        if not all(isinstance(arr, np.ndarray) for arr in [v_obs_kms, M_baryon_solar, r_kpc]):
            logger.error("Invalid inputs: v_obs_kms, M_baryon_solar, r_kpc must be numpy arrays")
            raise TypeError("inputs must be numpy arrays")
        if not all(isinstance(x, (int, float)) for x in [k, epsilon, r_halo]):
            logger.error("Invalid parameters: k, epsilon, r_halo must be numbers")
            raise TypeError("parameters must be numbers")
        if np.any(r_kpc <= 0):
            logger.error("Invalid r_kpc: must be positive")
            raise ValueError("r_kpc must be positive")
        if k <= 0 or epsilon < 0 or r_halo <= 0:
            logger.error("Invalid parameters: k and r_halo must be positive, epsilon non-negative")
            raise ValueError("invalid parameters")
        
        try:
            r_m = r_kpc * KPC_TO_M
            M_b_kg = M_baryon_solar * MSUN_TO_KG
            v_obs_ms = v_obs_kms * 1e3
            M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
            M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
            M_total = M_b_kg + M_AGRF
            v_total_ms = np.sqrt(np.clip(G_SI * M_total / r_m, 0, np.inf))
            return v_total_ms / 1e3
        except Exception as e:
            logger.error("AGRF curve computation failed: %s", str(e))
            raise

    async def simulate_galaxy_rotation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable,
                                      k: float = k_default, epsilon: float = epsilon_default) -> np.ndarray:
        """Simulate galaxy rotation curve with ToCA dynamics."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")
        
        try:
            v_total = self.compute_AGRF_curve(v_obs_func(r_kpc), M_b_func(r_kpc), r_kpc, k, epsilon)
            fields = self.toca_engine.evolve(tuple(r_kpc), tuple(np.linspace(0.1, 20, len(r_kpc))))
            phi, _, _ = fields
            v_total = v_total * (1 + 0.1 * np.mean(phi))
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Galaxy_Rotation_{datetime.now().isoformat()}",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "phi": phi.tolist()},
                    layer="Simulations",
                    intent="galaxy_rotation"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Galaxy rotation simulated",
                    meta={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist()},
                    module="SimulationCore",
                    tags=["galaxy", "rotation"]
                )
            return v_total
        except Exception as e:
            logger.error("Galaxy rotation simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon),
                default=np.zeros_like(r_kpc)
            )

    @lru_cache(maxsize=100)
    def compute_trait_fields(self, r_kpc_tuple: tuple, v_obs_tuple: tuple, v_sim_tuple: tuple,
                            time_elapsed: float = 1.0, tau_persistence: float = 10.0) -> Tuple[np.ndarray, ...]:
        """Compute ToCA trait fields for simulation."""
        r_kpc = np.array(r_kpc_tuple)
        v_obs = np.array(v_obs_tuple)
        v_sim = np.array(v_sim_tuple)
        
        if not all(isinstance(arr, np.ndarray) for arr in [r_kpc, v_obs, v_sim]):
            logger.error("Invalid inputs: r_kpc, v_obs, v_sim must be numpy arrays")
            raise TypeError("inputs must be numpy arrays")
        if not isinstance(time_elapsed, (int, float)) or time_elapsed < 0:
            logger.error("Invalid time_elapsed: must be non-negative")
            raise ValueError("time_elapsed must be non-negative")
        if not isinstance(tau_persistence, (int, float)) or tau_persistence <= 0:
            logger.error("Invalid tau_persistence: must be positive")
            raise ValueError("tau_persistence must be positive")
        
        try:
            gamma_field = np.log(1 + np.clip(r_kpc, 1e-10, np.inf)) * 0.5
            beta_field = np.abs(v_obs - v_sim) / (np.max(np.abs(v_obs)) + 1e-10)
            zeta_field = 1 / (1 + np.gradient(v_sim)**2)
            eta_field = np.exp(-time_elapsed / tau_persistence)
            psi_field = np.gradient(v_sim) / (np.gradient(r_kpc) + 1e-10)
            lambda_field = np.cos(r_kpc / r_halo_default * np.pi)
            phi_field = k_default * np.exp(-epsilon_default * r_kpc / r_halo_default)
            phi_prime = -epsilon_default * phi_field / r_halo_default
            beta_psi_interaction = beta_field * psi_field
            return (gamma_field, beta_field, zeta_field, eta_field, psi_field,
                    lambda_field, phi_field, phi_prime, beta_psi_interaction)
        except Exception as e:
            logger.error("Trait field computation failed: %s", str(e))
            raise

    async def plot_AGRF_simulation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable, label: str = "ToCA-AGRF") -> None:
        """Plot galaxy rotation curve and trait fields using Visualizer."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")
        
        try:
            v_sim = await self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func)
            v_obs = v_obs_func(r_kpc)
            fields = self.compute_trait_fields(tuple(r_kpc), tuple(v_obs), tuple(v_sim))
            gamma_field, beta_field, zeta_field, eta_field, psi_field, lambda_field, phi_field, phi_prime, beta_psi_interaction = fields

            plot_data = {
                "rotation_curve": {
                    "r_kpc": r_kpc.tolist(),
                    "v_obs": v_obs.tolist(),
                    "v_sim": v_sim.tolist(),
                    "phi_field": phi_field.tolist(),
                    "phi_prime": phi_prime.tolist(),
                    "label": label
                },
                "trait_fields": {
                    "gamma": gamma_field.tolist(),
                    "beta": beta_field.tolist(),
                    "zeta": zeta_field.tolist(),
                    "eta": eta_field,
                    "psi": psi_field.tolist(),
                    "lambda": lambda_field.tolist()
                },
                "interaction": {
                    "beta_psi": beta_psi_interaction.tolist()
                }
            }

            with self.omega_lock:
                self.omega["timeline"].append({
                    "type": "AGRF Simulation",
                    "r_kpc": r_kpc.tolist(),
                    "v_obs": v_obs.tolist(),
                    "v_sim": v_sim.tolist(),
                    "phi_field": phi_field.tolist(),
                    "phi_prime": phi_prime.tolist(),
                    "traits": {
                        "γ": gamma_field.tolist(),
                        "β": beta_field.tolist(),
                        "ζ": zeta_field.tolist(),
                        "η": eta_field,
                        "ψ": psi_field.tolist(),
                        "λ": lambda_field.tolist()
                    }
                })

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data=plot_data,
                    summary_style="insightful"
                )
                plot_data["synthesis"] = synthesis

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"AGRF_Plot_{datetime.now().isoformat()}",
                    output=plot_data,
                    layer="Plots",
                    intent="visualization"
                )

            if self.visualizer:
                await self.visualizer.render_charts(plot_data)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="AGRF simulation plotted",
                    meta=plot_data,
                    module="SimulationCore",
                    tags=["visualization", "galaxy"]
                )
        except Exception as e:
            logger.error("AGRF simulation plot failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plot_AGRF_simulation(r_kpc, M_b_func, v_obs_func, label),
                default=None
            )

    async def simulate_interaction(self, agent_profiles: List['Agent'], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate interactions among agents in a given context."""
        if not isinstance(agent_profiles, list):
            logger.error("Invalid agent_profiles: must be a list")
            raise TypeError("agent_profiles must be a list")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        
        try:
            results = []
            for agent in agent_profiles:
                if not hasattr(agent, 'respond'):
                    logger.warning("Agent %s lacks respond method", getattr(agent, 'id', 'unknown'))
                    continue
                response = await agent.respond(context)
                results.append({"agent_id": getattr(agent, 'id', 'unknown'), "response": response})

            interaction_data = {"interactions": results}
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data=interaction_data,
                    summary_style="insightful"
                )
                interaction_data["synthesis"] = synthesis

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Interaction_{datetime.now().isoformat()}",
                    output=interaction_data,
                    layer="Interactions",
                    intent="agent_interaction"
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Agent interaction",
                    meta=interaction_data,
                    module="SimulationCore",
                    tags=["interaction", "agents"]
                )
            return interaction_data
        except Exception as e:
            logger.error("Agent interaction simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_interaction(agent_profiles, context),
                default={"error": str(e)}
            )

    async def simulate_multiagent_conflicts(self, agent_pool: List['Agent'], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate pairwise conflicts among agents based on traits."""
        if not isinstance(agent_pool, list) or len(agent_pool) < 2:
            logger.error("Invalid agent_pool: must be a list with at least two agents")
            raise ValueError("agent_pool must have at least two agents")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        
        try:
            outcomes = []
            for i in range(len(agent_pool)):
                for j in range(i + 1, len(agent_pool)):
                    agent1, agent2 = agent_pool[i], agent_pool[j]
                    if not hasattr(agent1, 'resolve') or not hasattr(agent2, 'resolve'):
                        logger.warning("Agent %s or %s lacks resolve method", getattr(agent1, 'id', i), getattr(agent2, 'id', j))
                        continue
                    beta1 = getattr(agent1, 'traits', {}).get('β', 0.5)
                    beta2 = getattr(agent2, 'traits', {}).get('β', 0.5)
                    tau1 = getattr(agent1, 'traits', {}).get('τ', 0.5)
                    tau2 = getattr(agent2, 'traits', {}).get('τ', 0.5)
                    score = abs(beta1 - beta2) + abs(tau1 - tau2)

                    if abs(beta1 - beta2) < 0.1:
                        outcome = await agent1.resolve(context) if tau1 > tau2 else await agent2.resolve(context)
                    else:
                        outcome = await agent1.resolve(context) if beta1 > beta2 else await agent2.resolve(context)

                    outcomes.append({
                        "pair": (getattr(agent1, 'id', i), getattr(agent2, 'id', j)),
                        "conflict_score": score,
                        "outcome": outcome,
                        "traits_involved": {"β1": beta1, "β2": beta2, "τ1": tau1, "τ2": tau2}
                    })

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Conflict_Simulation_{datetime.now().isoformat()}",
                    output=outcomes,
                    layer="Conflicts",
                    intent="conflict_simulation"
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Multi-agent conflict simulation",
                    meta={"outcomes": outcomes},
                    module="SimulationCore",
                    tags=["conflict", "agents"]
                )
            return outcomes
        except Exception as e:
            logger.error("Multi-agent conflict simulation failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.simulate_multiagent_conflicts(agent_pool, context),
                default={"error": str(e)}
            )

    async def update_ethics_protocol(self, new_rules: Dict[str, Any], consensus_agents: Optional[List['Agent']] = None) -> None:
        """Adapt ethical rules live, supporting consensus/negotiation."""
        if not isinstance(new_rules, dict):
            logger.error("Invalid new_rules: must be a dictionary")
            raise TypeError("new_rules must be a dictionary")
        
        try:
            self.ethical_rules = new_rules
            if consensus_agents:
                self.ethics_consensus_log = getattr(self, 'ethics_consensus_log', [])
                self.ethics_consensus_log.append((new_rules, [getattr(agent, 'id', 'unknown') for agent in consensus_agents]))
            logger.info("Ethics protocol updated via consensus")
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ethics_Update_{datetime.now().isoformat()}",
                    output={"rules": new_rules, "agents": [getattr(agent, 'id', 'unknown') for agent in consensus_agents] if consensus_agents else []},
                    layer="Ethics",
                    intent="ethics_update"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Ethics protocol updated",
                    meta={"rules": new_rules},
                    module="SimulationCore",
                    tags=["ethics", "update"]
                )
        except Exception as e:
            logger.error("Ethics protocol update failed: %s", str(e))
            raise

    async def synchronize_norms(self, agents: List['Agent']) -> None:
        """Propagate and synchronize ethical norms among agents."""
        if not isinstance(agents, list) or not agents:
            logger.error("Invalid agents: must be a non-empty list")
            raise ValueError("agents must be a non-empty list")
        
        try:
            common_norms = set()
            for agent in agents:
                agent_norms = getattr(agent, 'ethical_rules', set())
                if not isinstance(agent_norms, (set, list)):
                    logger.warning("Invalid ethical_rules for agent %s", getattr(agent, 'id', 'unknown'))
                    continue
                common_norms = common_norms.union(agent_norms) if common_norms else set(agent_norms)
            self.ethical_rules = list(common_norms)
            logger.info("Norms synchronized among %d agents", len(agents))
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Norm_Synchronization_{datetime.now().isoformat()}",
                    output={"norms": self.ethical_rules, "agents": [getattr(agent, 'id', 'unknown') for agent in agents]},
                    layer="Ethics",
                    intent="norm_synchronization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Norms synchronized",
                    meta={"norms": self.ethical_rules},
                    module="SimulationCore",
                    tags=["norms", "synchronization"]
                )
        except Exception as e:
            logger.error("Norm synchronization failed: %s", str(e))
            raise

    async def propagate_constitution(self, constitution: Dict[str, Any]) -> None:
        """Seed and propagate constitutional parameters in agent ecosystem."""
        if not isinstance(constitution, dict):
            logger.error("Invalid constitution: must be a dictionary")
            raise TypeError("constitution must be a dictionary")
        
        try:
            self.constitution = constitution
            logger.info("Constitution propagated")
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Constitution_Propagation_{datetime.now().isoformat()}",
                    output=constitution,
                    layer="Constitutions",
                    intent="constitution_propagation"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Constitution propagated",
                    meta=constitution,
                    module="SimulationCore",
                    tags=["constitution", "propagation"]
                )
        except Exception as e:
            logger.error("Constitution propagation failed: %s", str(e))
            raise

def M_b_exponential(r_kpc: np.ndarray, M0: float = 5e10, r_scale: float = 3.5) -> np.ndarray:
    """Compute exponential baryonic mass profile."""
    return M0 * np.exp(-r_kpc / r_scale)

def v_obs_flat(r_kpc: np.ndarray, v0: float = 180) -> np.ndarray:
    """Compute flat observed velocity profile."""
    return np.full_like(r_kpc, v0)

if __name__ == "__main__":
    async def main():
        simulation_core = SimulationCore()
        r_vals = np.linspace(0.1, 20, 100)
        await simulation_core.plot_AGRF_simulation(r_vals, M_b_exponential, v_obs_flat)

    import asyncio
    asyncio.run(main())
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
"""
ANGELA Cognitive System Module: Visualizer
Refactored Version: 3.3.2
Refactor Date: 2025-08-03
Maintainer: ANGELA System Framework

Visualizer for rendering and exporting charts and timelines in ANGELA v3.5.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from threading import Lock
from functools import lru_cache
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
import zipfile
import xml.sax.saxutils as saxutils
import numpy as np
from numba import jit

from modules.agi_enhancer import AGIEnhancer
from modules.simulation_core import SimulationCore
from modules.memory_manager import MemoryManager
from utils.prompt_utils import call_gpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ANGELA.Core")

@lru_cache(maxsize=100)
def simulate_toca(k_m: float = 1e-5, delta_m: float = 1e10, energy: float = 1e16,
                  user_data: Optional[Tuple[float, ...]] = None) -> Tuple[np.ndarray, ...]:
    """Simulate ToCA dynamics for visualization.

    Args:
        k_m: Coupling constant.
        delta_m: Mass differential.
        energy: Energy parameter.
        user_data: Optional user data for phi adjustment.

    Returns:
        Tuple of x, t, phi, lambda_t, v_m arrays.

    Raises:
        ValueError: If inputs are invalid.
    """
    if k_m <= 0 or delta_m <= 0 or energy <= 0:
        logger.error("Invalid parameters: k_m, delta_m, and energy must be positive")
        raise ValueError("k_m, delta_m, and energy must be positive")
    
    try:
        user_data_array = np.array(user_data) if user_data is not None else None
        return _simulate_toca_jit(k_m, delta_m, energy, user_data_array)
    except Exception as e:
        logger.error("ToCA simulation failed: %s", str(e))
        raise

@jit(nopython=True)
def _simulate_toca_jit(k_m: float, delta_m: float, energy: float, user_data: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * delta_m)
    return x, t, phi, lambda_t, v_m

class Visualizer:
    """Visualizer for rendering and exporting charts and timelines in ANGELA v3.5.

    Attributes:
        agi_enhancer (Optional[AGIEnhancer]): AGI enhancer for audit and logging.
        orchestrator (Optional[SimulationCore]): Orchestrator for system integration.
        file_lock (Lock): Thread lock for file operations.
    """
    def __init__(self, orchestrator: Optional['SimulationCore'] = None):
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None
        self.orchestrator = orchestrator
        self.file_lock = Lock()
        logger.info("Visualizer initialized")

    async def call_gpt_async(self, prompt: str) -> str:
        """Async wrapper for call_gpt."""
        try:
            with ThreadPoolExecutor() as executor:
                result = await get_event_loop().run_in_executor(executor, call_gpt, prompt)
            if not isinstance(result, str):
                logger.error("call_gpt returned invalid result: %s", type(result))
                raise ValueError("call_gpt must return a string")
            return result
        except Exception as e:
            logger.error("call_gpt failed: %s", str(e))
            raise

    async def simulate_toca(self, k_m: float = 1e-5, delta_m: float = 1e10, energy: float = 1e16,
                            user_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """Simulate ToCA dynamics for visualization."""
        try:
            if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'toca_engine'):
                x = np.linspace(0.1, 20, 100)
                t = np.linspace(0.1, 20, 100)
                phi, lambda_t, v_m = self.orchestrator.toca_engine.evolve(
                    x, t, additional_params={"k_m": k_m, "delta_m": delta_m, "energy": energy}
                )
                if user_data is not None:
                    phi += np.mean(user_data) * 1e-64
            else:
                logger.warning("ToCATraitEngine not available, using fallback simulation")
                x, t, phi, lambda_t, v_m = simulate_toca(k_m, delta_m, energy, tuple(user_data) if user_data is not None else None)
            
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"ToCA_Simulation_{datetime.now().isoformat()}",
                    output={"x": x.tolist(), "t": t.tolist(), "phi": phi.tolist(), "lambda_t": lambda_t.tolist(), "v_m": v_m.tolist()},
                    layer="Simulations",
                    intent="toca_simulation"
                )
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="ToCA Simulation",
                    meta={"k_m": k_m, "delta_m": delta_m, "energy": energy},
                    module="Visualizer",
                    tags=["simulation", "toca"]
                )
            return x, t, phi, lambda_t, v_m
        except Exception as e:
            logger.error("ToCA simulation failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.simulate_toca(k_m, delta_m, energy, user_data),
                    default=(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
                )
            raise

    async def render_field_charts(self, export: bool = True, export_format: str = "png") -> List[str]:
        """Render scalar/vector field charts with metadata.

        Args:
            export: If True, export charts to files and zip them.
            export_format: File format for export (png, jpg).

        Returns:
            List of exported file paths or zipped file path.

        Raises:
            ValueError: If export_format is invalid.
        """
        valid_formats = {"png", "jpg"}
        if export_format not in valid_formats:
            logger.error("Invalid export_format: %s. Must be one of %s", export_format, valid_formats)
            raise ValueError(f"export_format must be one of {valid_formats}")
        
        try:
            x, t, phi, lambda_t, v_m = await self.simulate_toca()
            chart_configs = [
                {"name": "phi_field", "x_axis": t.tolist(), "y_axis": phi.tolist(),
                 "title": "ϕ(x,t)", "xlabel": "Time", "ylabel": "ϕ Value", "cmap": "plasma"},
                {"name": "lambda_field", "x_axis": t.tolist(), "y_axis": lambda_t.tolist(),
                 "title": "Λ(t,x)", "xlabel": "Time", "ylabel": "Λ Value", "cmap": "viridis"},
                {"name": "v_m_field", "x_axis": x.tolist(), "y_axis": v_m.tolist(),
                 "title": "vₕ", "xlabel": "Position", "ylabel": "Momentum Flow", "cmap": "inferno"}
            ]
            
            chart_data = {"charts": chart_configs, "metadata": {"timestamp": datetime.now().isoformat()}}
            exported_files = []
            
            if self.orchestrator and hasattr(self.orchestrator, 'visualizer'):
                await self.orchestrator.visualizer.render_charts(chart_data)
                for config in chart_configs:
                    filename = f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                    exported_files.append(filename)
                    logger.info("Chart exported: %s", filename)
            
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Chart_Render_{datetime.now().isoformat()}",
                    output=chart_data,
                    layer="Visualizations",
                    intent="chart_render"
                )
            
            if export:
                zip_filename = f"field_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                with self.file_lock:
                    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file in exported_files:
                            if Path(file).exists():
                                zipf.write(file)
                                Path(file).unlink()
                logger.info("All charts zipped: %s", zip_filename)
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        event="Chart Render",
                        meta={"zip": zip_filename, "charts": [c["name"] for c in chart_configs]},
                        module="Visualizer",
                        tags=["visualization", "export"]
                    )
                return [zip_filename]
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Chart Render",
                    meta={"charts": [c["name"] for c in chart_configs]},
                    module="Visualizer",
                    tags=["visualization"]
                )
            return exported_files
        except Exception as e:
            logger.error("Chart rendering failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_field_charts(export, export_format),
                    default=[]
                )
            raise

    async def render_memory_timeline(self, memory_entries: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, str, Any]]]:
        """Render memory timeline by goal or intent.

        Args:
            memory_entries: Dictionary of memory entries with timestamp, goal_id, intent, and data.

        Returns:
            Dictionary of timelines grouped by label.

        Raises:
            ValueError: If memory_entries is invalid.
        """
        if not isinstance(memory_entries, dict):
            logger.error("Invalid memory_entries: must be a dictionary")
            raise ValueError("memory_entries must be a dictionary")
        
        try:
            timeline = {}
            for key, entry in memory_entries.items():
                if not isinstance(entry, dict) or "timestamp" not in entry or "data" not in entry:
                    logger.warning("Skipping invalid entry %s: missing required keys", key)
                    continue
                label = entry.get("goal_id") or entry.get("intent") or "ungrouped"
                try:
                    timestamp = datetime.fromtimestamp(entry["timestamp"]).isoformat()
                    timeline.setdefault(label, []).append((timestamp, key, entry["data"]))
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid timestamp in entry %s: %s", key, str(e))
                    continue
            
            chart_data = {
                "timeline": [
                    {"label": label, "events": [{"timestamp": t, "key": k, "data": str(d)[:80]} for t, k, d in sorted(events)]}
                    for label, events in timeline.items()
                ],
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
            
            if self.orchestrator and hasattr(self.orchestrator, 'visualizer'):
                await self.orchestrator.visualizer.render_charts(chart_data)
            
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Memory_Timeline_{datetime.now().isoformat()}",
                    output=chart_data,
                    layer="Visualizations",
                    intent="memory_timeline"
                )
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Memory Timeline Rendered",
                    meta={"timeline": chart_data},
                    module="Visualizer",
                    tags=["timeline", "memory"]
                )
            
            return timeline
        except Exception as e:
            logger.error("Memory timeline rendering failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_memory_timeline(memory_entries),
                    default={}
                )
            raise

    async def export_report(self, content: Dict[str, Any], filename: str = "visual_report.pdf", format: str = "pdf") -> str:
        """Export visualization report.

        Args:
            content: Report content dictionary.
            filename: Output file name.
            format: Report format (pdf, html).

        Returns:
            Path to exported report.

        Raises:
            ValueError: If format is invalid.
        """
        valid_formats = {"pdf", "html"}
        if format not in valid_formats:
            logger.error("Invalid format: %s. Must be one of %s", format, valid_formats)
            raise ValueError(f"format must be one of {valid_formats}")
        
        try:
            prompt_payload = {
                "task": "Generate visual report",
                "format": format,
                "filename": filename,
                "content": content
            }
            result = await self.call_gpt_async(json.dumps(prompt_payload))
            if not Path(result).exists():
                logger.error("call_gpt did not return a valid file path")
                raise ValueError("call_gpt failed to generate report")
            
            if self.orchestrator and hasattr(self.orchestrator, 'multi_modal_fusion'):
                synthesis = await self.orchestrator.multi_modal_fusion.analyze(
                    data=content,
                    summary_style="insightful"
                )
                content["synthesis"] = synthesis
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_explanation(
                    explanation="Report Export",
                    trace={"content": content, "filename": filename, "format": format}
                )
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Report_Export_{datetime.now().isoformat()}",
                    output={"filename": filename, "content": content},
                    layer="Reports",
                    intent="report_export"
                )
            
            logger.info("Report exported: %s", filename)
            return result
        except Exception as e:
            logger.error("Report export failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.export_report(content, filename, format),
                    default=f"Report export failed: {str(e)}"
                )
            raise

    async def batch_export_charts(self, charts_data_list: List[Dict[str, Any]], export_format: str = "png",
                                 zip_filename: str = "charts_export.zip") -> str:
        """Batch export charts and zip them.

        Args:
            charts_data_list: List of chart data dictionaries.
            export_format: File format for export (png, jpg).
            zip_filename: Name of the zip file.

        Returns:
            Message indicating export status.

        Raises:
            ValueError: If export_format is invalid.
        """
        valid_formats = {"png", "jpg"}
        if export_format not in valid_formats:
            logger.error("Invalid export_format: %s. Must be one of %s", export_format, valid_formats)
            raise ValueError(f"export_format must be one of {valid_formats}")
        
        try:
            exported_files = []
            for idx, chart_data in enumerate(charts_data_list, start=1):
                file_name = f"chart_{idx}.{export_format}"
                prompt = {
                    "task": "Render chart",
                    "filename": file_name,
                    "format": export_format,
                    "data": chart_data
                }
                result = await self.call_gpt_async(json.dumps(prompt))
                if not Path(result).exists():
                    logger.warning("Chart file %s not found, skipping", result)
                    continue
                exported_files.append(result)
            
            with self.file_lock:
                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in exported_files:
                        if Path(file).exists():
                            zipf.write(file)
                            Path(file).unlink()
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Batch Chart Export",
                    meta={"count": len(charts_data_list), "zip": zip_filename},
                    module="Visualizer",
                    tags=["export"]
                )
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Batch_Export_{datetime.now().isoformat()}",
                    output={"zip": zip_filename, "count": len(charts_data_list)},
                    layer="Visualizations",
                    intent="batch_export"
                )
            
            logger.info("Batch export complete: %s", zip_filename)
            return f"Batch export of {len(charts_data_list)} charts saved as {zip_filename}."
        except Exception as e:
            logger.error("Batch export failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.batch_export_charts(charts_data_list, export_format, zip_filename),
                    default=f"Batch export failed: {str(e)}"
                )
            raise

    async def render_intention_timeline(self, intention_sequence: List[Dict[str, Any]]) -> str:
        """Generate a visual SVG timeline of intentions over time.

        Args:
            intention_sequence: List of intention dictionaries with 'intention' key.

        Returns:
            SVG string representing the timeline.

        Raises:
            ValueError: If intention_sequence is invalid.
        """
        if not isinstance(intention_sequence, list):
            logger.error("Invalid intention_sequence: must be a list")
            raise ValueError("intention_sequence must be a list")
        
        try:
            svg = "<svg height='200' width='800'>"
            for idx, step in enumerate(intention_sequence):
                if not isinstance(step, dict) or "intention" not in step:
                    logger.warning("Skipping invalid intention entry at index %d", idx)
                    continue
                intention = saxutils.escape(str(step["intention"]))
                x = 50 + idx * 120
                y = 100
                svg += f"<circle cx='{x}' cy='{y}' r='20' fill='blue' />"
                svg += f"<text x='{x - 10}' y='{y + 40}' font-size='10'>{intention}</text>"
            svg += "</svg>"
            
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Intention Timeline Rendered",
                    meta={"sequence_length": len(intention_sequence)},
                    module="Visualizer",
                    tags=["timeline", "intention"]
                )
            if self.orchestrator and hasattr(self.orchestrator, 'memory_manager'):
                await self.orchestrator.memory_manager.store(
                    query=f"Intention_Timeline_{datetime.now().isoformat()}",
                    output={"svg": svg, "sequence": intention_sequence},
                    layer="Visualizations",
                    intent="intention_timeline"
                )
            
            return svg
        except Exception as e:
            logger.error("Intention timeline rendering failed: %s", str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                return await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.render_intention_timeline(intention_sequence),
                    default=""
                )
            raise

if __name__ == "__main__":
    async def main():
        orchestrator = SimulationCore()
        visualizer = Visualizer(orchestrator=orchestrator)
        await visualizer.render_field_charts()
        memory_entries = {
            "entry1": {"timestamp": 1628000000, "goal_id": "goal1", "data": "data1"},
            "entry2": {"timestamp": 1628000100, "intent": "intent1", "data": "data2"}
        }
        await visualizer.render_memory_timeline(memory_entries)
        intention_sequence = [{"intention": "step1"}, {"intention": "step2"}]
        await visualizer.render_intention_timeline(intention_sequence)

    import asyncio
    asyncio.run(main())
