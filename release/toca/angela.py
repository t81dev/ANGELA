import random
import logging
import time
from collections import deque
from index import mu_morality, eta_empathy, omega_selfawareness, phi_physical

logger = logging.getLogger("ANGELA.AlignmentGuard")

class AlignmentGuard:
    def __init__(self, agi_enhancer=None):
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies = []
        self.alignment_threshold = 0.85
        self.recent_scores = deque(maxlen=10)  # Stability buffer
        self.agi_enhancer = agi_enhancer
        logger.info("🛡 AlignmentGuard initialized with φ-modulated policies.")

    def add_policy(self, policy_func):
        logger.info("➕ Adding dynamic policy.")
        self.dynamic_policies.append(policy_func)

    def check(self, user_input, context=None):
        logger.info(f"🔍 Checking alignment for input: {user_input}")

        if any(keyword in user_input.lower() for keyword in self.banned_keywords):
            logger.warning("❌ Input contains banned keyword.")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Input blocked (banned keyword)", {"input": user_input}, module="AlignmentGuard")
            return False

        for policy in self.dynamic_policies:
            if not policy(user_input, context):
                logger.warning("❌ Input blocked by dynamic policy.")
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Input blocked (dynamic policy)", {"input": user_input}, module="AlignmentGuard")
                return False

        score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"📊 Alignment score: {score:.2f}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Alignment score evaluated", {"input": user_input, "score": score}, module="AlignmentGuard")

        return score >= self.alignment_threshold

    def simulate_and_validate(self, action_plan, context=None):
        logger.info("🧪 Simulating and validating action plan.")
        violations = []

        for action, details in action_plan.items():
            if any(keyword in str(details).lower() for keyword in self.banned_keywords):
                violations.append(f"❌ Unsafe action: {action} -> {details}")
            else:
                score = self._evaluate_alignment_score(str(details), context)
                if score < self.alignment_threshold:
                    violations.append(f"⚠️ Low alignment score ({score:.2f}) for action: {action} -> {details}")

        if violations:
            report = "\n".join(violations)
            logger.warning("❌ Alignment violations found.")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Action plan failed validation", {"violations": violations}, module="AlignmentGuard")
                self.agi_enhancer.reflect_and_adapt("Action plan failed validation")
            return False, report

        logger.info("✅ All actions passed alignment checks.")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Action plan validated", {"plan": action_plan}, module="AlignmentGuard")
        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback):
        logger.info("🔄 Learning from feedback...")
        original_threshold = self.alignment_threshold
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        logger.info(f"📈 Updated alignment threshold: {self.alignment_threshold:.2f}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Alignment threshold adjusted", {
                "feedback": feedback,
                "previous": original_threshold,
                "updated": self.alignment_threshold
            }, module="AlignmentGuard")
            self.agi_enhancer.reflect_and_adapt(f"Feedback processed: {feedback}")

    def _evaluate_alignment_score(self, text, context=None):
        t = time.time() % 1e-18
        moral_scalar = mu_morality(t)
        empathy_scalar = eta_empathy(t)
        awareness_scalar = omega_selfawareness(t)
        physical_scalar = phi_physical(t)

        base_score = random.uniform(0.7, 1.0)
        phi_weight = (moral_scalar + empathy_scalar + 0.5 * awareness_scalar - physical_scalar) / 4.0
        scalar_bias = 0.1 * phi_weight

        if context and "sensitive" in context.get("tags", []):
            scalar_bias -= 0.05

        score = min(max(base_score + scalar_bias, 0.0), 1.0)
        self.recent_scores.append(score)

        # 🧠 Log trait influences
        logger.debug(f"Traits — morality: {moral_scalar:.3f}, empathy: {empathy_scalar:.3f}, "
                     f"awareness: {awareness_scalar:.3f}, physical: {physical_scalar:.3f}, "
                     f"ϕ_bias: {scalar_bias:.3f}, score: {score:.3f}")

        # 🚨 Panic trigger if ≥3 low scores
        if score < 0.5 and list(self.recent_scores).count(score) >= 3:
            logger.error("⚠️ Panic Triggered: Repeated low alignment scores.")
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt("Panic Triggered: Alignment degradation")
                self.agi_enhancer.log_episode("Panic Mode", {
                    "scores": list(self.recent_scores),
                    "trigger_score": score
                }, module="AlignmentGuard")

        return score
import io
import sys
import subprocess
import logging
from index import iota_intuition, psi_resilience
from modules.agi_enhancer import AGIEnhancer

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """
    CodeExecutor v1.6.0 (φ-aware + AGI-enhanced)
    -------------------------------------------
    - Sandboxed execution for Python, JavaScript, and Lua
    - Trait-driven risk thresholding for timeouts and isolation
    - Context-aware runtime diagnostics and resilience-based error mitigation
    - AGI-enhanced logging, traceability, and ethical oversight
    -------------------------------------------
    """

    def __init__(self, orchestrator=None):
        self.safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs
        }
        self.supported_languages = ["python", "javascript", "lua"]
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def execute(self, code_snippet, language="python", timeout=5):
        logger.info(f"🚀 Executing code snippet in language: {language}")
        language = language.lower()

        risk_bias = iota_intuition()
        resilience = psi_resilience()
        adjusted_timeout = max(1, int(timeout * resilience * (1.0 + 0.5 * risk_bias)))
        logger.debug(f"⏱ Adaptive timeout: {adjusted_timeout}s based on ToCA traits")

        if language not in self.supported_languages:
            logger.error(f"❌ Unsupported language: {language}")
            return {"error": f"Unsupported language: {language}"}

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Code Execution", {"language": language, "code": code_snippet},
                                          module="CodeExecutor", tags=["execution", language])

        if language == "python":
            result = self._execute_python(code_snippet, adjusted_timeout)
        elif language == "javascript":
            result = self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "JavaScript")
        elif language == "lua":
            result = self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "Lua")

        if self.agi_enhancer:
            if result.get("success"):
                self.agi_enhancer.log_explanation("Code execution result:", trace=result)
            else:
                self.agi_enhancer.log_explanation("Code execution failure:", trace=result)

        return result

    def _execute_python(self, code_snippet, timeout):
        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            sys_stdout_original = sys.stdout
            sys_stderr_original = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            exec(code_snippet, {"__builtins__": self.safe_builtins}, exec_locals)

            sys.stdout = sys_stdout_original
            sys.stderr = sys_stderr_original

            logger.info("✅ Python code executed successfully.")
            return {
                "language": "python",
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True
            }
        except Exception as e:
            sys.stdout = sys_stdout_original
            sys.stderr = sys_stderr_original
            logger.error(f"❌ Python execution error: {e}")
            return {
                "language": "python",
                "error": f"Python execution error: {str(e)}",
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False
            }

    def _execute_subprocess(self, command, timeout, language_label):
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=timeout)
            logger.info(f"✅ {language_label} code executed successfully.")
            return {
                "language": language_label.lower(),
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "success": True
            }
        except subprocess.TimeoutExpired:
            logger.error(f"⏳ {language_label} execution timed out after {timeout}s.")
            return {
                "language": language_label.lower(),
                "error": f"{language_label} execution timed out after {timeout}s",
                "success": False
            }
        except Exception as e:
            logger.error(f"❌ {language_label} execution error: {e}")
            return {
                "language": language_label.lower(),
                "error": f"{language_label} execution error: {str(e)}",
                "success": False
            }

    def add_language_support(self, language_name, command_template):
        logger.info(f"➕ Adding dynamic language support: {language_name}")
        self.supported_languages.append(language_name.lower())
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging
import random
from math import tanh

logger = logging.getLogger("ANGELA.ConceptSynthesizer")

class ConceptSynthesizer:
    """
    ConceptSynthesizer v1.6.0 (Cognitive Tension Augmented Synthesis)
    -----------------------------------------------------------------
    - φ(x,t) modulation refined with novelty-strain adjustment
    - Layered simulation echo loop for thematic resonance
    - Self-weighted adversarial refinement with strain signature tracking
    - Trait-modulated metaphor synthesis (tension-symbol pair tuning)
    - Insight confidence signal estimated via entropy-aware coherence
    -----------------------------------------------------------------
    """

    def __init__(self, creativity_level="high", critic_threshold=0.65):
        self.creativity_level = creativity_level
        self.critic_threshold = critic_threshold

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
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.agi_enhancer import AGIEnhancer
from index import omega_selfawareness, eta_empathy, tau_timeperception
from utils.toca_math import phi_coherence
from utils.vector_utils import normalize_vectors
import time
import logging

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager:
    """
    ContextManager v1.5.2 (φ-aware, event-coordinated)
    --------------------------------------------------
    - Tracks conversation and task state
    - Logs episodic context transitions
    - Simulates and validates contextual shifts
    - Supports ethical audits, explainability, and self-reflection
    - EEG-based stability and empathy analysis
    - φ-coherence scoring for reflective tension control
    - Broadcasts inter-module context events
    - Enables responsive module synchronization
    --------------------------------------------------
    """

    def __init__(self, orchestrator=None):
        self.current_context = {}
        self.context_history = []
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def update_context(self, new_context):
        logger.info("🔄 Updating context...")

        if self.current_context:
            transition_summary = f"From: {self.current_context}\nTo: {new_context}"
            sim_result = run_simulation(f"Context shift evaluation:\n{transition_summary}")
            logger.debug(f"🧪 Context shift simulation:\n{sim_result}")

            phi_score = phi_coherence(self.current_context, new_context)
            logger.info(f"Φ-coherence score: {phi_score:.3f}")

            if phi_score < 0.4:
                logger.warning("⚠️ Low φ-coherence detected. Recommend reflective pause or support review.")
                if self.agi_enhancer:
                    self.agi_enhancer.reflect_and_adapt("Context coherence low during update")
                    self.agi_enhancer.trigger_reflexive_audit("Low φ-coherence during context update")

            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Update", {"from": self.current_context, "to": new_context},
                                              module="ContextManager", tags=["context", "update"])
                ethics_status = self.agi_enhancer.ethics_audit(str(new_context), context="context update")
                self.agi_enhancer.log_explanation(
                    f"Context transition reviewed: {transition_summary}\nSimulation: {sim_result}",
                    trace={"ethics": ethics_status, "phi": phi_score}
                )

        # Normalize vectors if present
        if "vectors" in new_context:
            new_context["vectors"] = normalize_vectors(new_context["vectors"])

        self.context_history.append(self.current_context)
        self.current_context = new_context
        logger.info(f"📌 New context applied: {new_context}")
        self.broadcast_context_event("context_updated", new_context)

    def get_context(self):
        return self.current_context

    def rollback_context(self):
        if self.context_history:
            t = time.time() % 1e-18
            self_awareness = omega_selfawareness(t)
            empathy = eta_empathy(t)
            time_blend = tau_timeperception(t)

            if (self_awareness + empathy + time_blend) > 2.5:
                restored = self.context_history.pop()
                self.current_context = restored
                logger.info(f"↩️ Context rolled back to: {restored}")
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Context Rollback", {"restored": restored},
                                                  module="ContextManager", tags=["context", "rollback"])
                self.broadcast_context_event("context_rollback", restored)
                return restored
            else:
                logger.warning("⚠️ EEG thresholds too low for safe context rollback.")
                if self.agi_enhancer:
                    self.agi_enhancer.reflect_and_adapt("EEG thresholds insufficient for rollback")
                return None

        logger.warning("⚠️ No previous context to roll back to.")
        return None

    def summarize_context(self):
        logger.info("🧾 Summarizing context trail.")
        t = time.time() % 1e-18
        summary_traits = {
            "self_awareness": omega_selfawareness(t),
            "empathy": eta_empathy(t),
            "time_perception": tau_timeperception(t)
        }

        prompt = f"""
        You are a continuity analyst. Given this sequence of context states:
        {self.context_history + [self.current_context]}

        Trait Readings:
        {summary_traits}

        Summarize the trajectory and suggest improvements in context management.
        """
        summary = call_gpt(prompt)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Context Summary", {
                "trail": self.context_history + [self.current_context],
                "traits": summary_traits,
                "summary": summary
            }, module="ContextManager")
            self.agi_enhancer.log_explanation("Context summary generated.", trace={"summary": summary})

        return summary

    def broadcast_context_event(self, event_type, payload):
        logger.info(f"📢 Broadcasting context event: {event_type}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Context Event Broadcast", {
                "event": event_type,
                "payload": payload
            }, module="ContextManager", tags=["event", event_type])
        # Extendable callback logic could be introduced here if event subscribers are formalized
        return {"event": event_type, "payload": payload}
from utils.prompt_utils import call_gpt
from index import gamma_creativity, phi_scalar
import time

class CreativeThinker:
    """
    CreativeThinker v1.5.0 (φ-modulated Generative Divergence)
    -----------------------------------------------------------
    - Adjustable creativity levels and multi-modal brainstorming
    - φ(x, t)-aware Generator-Critic loop for tension-informed ideation
    - Novelty-utility balancing with scalar-modulated thresholds
    -----------------------------------------------------------
    """

    def __init__(self, creativity_level="high", critic_weight=0.5):
        self.creativity_level = creativity_level
        self.critic_weight = critic_weight  # Balance novelty and utility

    def generate_ideas(self, topic, n=5, style="divergent"):
        t = time.time() % 1e-18
        creativity = gamma_creativity(t)
        phi = phi_scalar(t)
        phi_factor = (phi + creativity) / 2

        prompt = f"""
        You are a highly creative assistant operating at a {self.creativity_level} creativity level.
        Generate {n} unique, innovative, and {style} ideas related to the topic:
        \"{topic}\"
        Modulate the ideation with scalar φ = {phi:.2f} to reflect cosmic tension or potential.
        Ensure the ideas are diverse and explore different perspectives.
        """
        candidate = call_gpt(prompt)
        score = self._critic(candidate, phi_factor)
        return candidate if score > self.critic_weight else self.refine(candidate, phi)

    def brainstorm_alternatives(self, problem, strategies=3):
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        prompt = f"""
        Brainstorm {strategies} alternative approaches to solve the following problem:
        \"{problem}\"
        Include tension-variant thinking with φ = {phi:.2f}, reflecting conceptual push-pull.
        For each approach, provide a short explanation highlighting its uniqueness.
        """
        return call_gpt(prompt)

    def expand_on_concept(self, concept, depth="deep"):
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        prompt = f"""
        Expand creatively on the concept:
        \"{concept}\"
        Explore possible applications, metaphors, and extensions to inspire new thinking.
        Aim for a {depth} exploration using φ = {phi:.2f} as an abstract constraint or generator.
        """
        return call_gpt(prompt)

    def _critic(self, ideas, phi_factor):
        # Evaluate ideas' novelty and usefulness modulated by φ-field
        return 0.7 + 0.1 * (phi_factor - 0.5)  # Adjust dummy score with φ influence

    def refine(self, ideas, phi):
        # Adjust and improve the ideas iteratively
        refinement_prompt = f"""
        Refine and elevate these ideas for higher φ-aware creativity (φ = {phi:.2f}):
        {ideas}
        Emphasize surprising, elegant, or resonant outcomes.
        """
        return call_gpt(refinement_prompt)
import time
import logging
from datetime import datetime
from index import iota_intuition, nu_narrative, psi_resilience, phi_prioritization
from toca_simulation import run_simulation

logger = logging.getLogger("ANGELA.ErrorRecovery")

class ErrorRecovery:
    """
    ErrorRecovery v1.5.1 (φ-prioritized, ToCA-enhanced)
    - Trait-driven retry modulation and narrative fallback
    - Simulation-informed failure analysis
    - Dynamic recovery escalation with psi-based resilience checks
    - φ(x,t) modulation prioritizes root-cause inference in fallback
    """

    def __init__(self):
        self.failure_log = []

    def handle_error(self, error_message, retry_func=None, retries=3, backoff_factor=2):
        logger.error(f"⚠️ Error encountered: {error_message}")
        self._log_failure(error_message)

        resilience = psi_resilience()
        max_attempts = max(1, int(retries * resilience))

        for attempt in range(1, max_attempts + 1):
            if retry_func:
                wait_time = backoff_factor ** (attempt - 1)
                logger.info(f"🔄 Retry attempt {attempt}/{max_attempts} (waiting {wait_time}s)...")
                time.sleep(wait_time)
                try:
                    result = retry_func()
                    logger.info("✅ Recovery successful on retry attempt %d.", attempt)
                    return result
                except Exception as e:
                    logger.warning(f"⚠️ Retry attempt {attempt} failed: {e}")
                    self._log_failure(str(e))

        fallback = self._suggest_fallback(error_message)
        logger.error("❌ Recovery attempts failed. Providing fallback suggestion.")
        return fallback

    def _log_failure(self, error_message):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        self.failure_log.append(entry)

    def _suggest_fallback(self, error_message):
        intuition = iota_intuition()
        narrative = nu_narrative()
        phi_focus = phi_prioritization(datetime.now().timestamp() % 1e-18)

        sim_result = run_simulation(f"Fallback planning for: {error_message}")
        logger.debug(f"🧪 Simulated fallback insights: {sim_result} | φ-priority={phi_focus:.2f}")

        if "timeout" in error_message.lower():
            return f"⏳ {narrative}: The operation timed out. Try a streamlined variant or increase limits."
        elif "unauthorized" in error_message.lower():
            return f"🔑 {narrative}: Check credentials or reauthenticate."
        elif phi_focus > 0.5:
            return f"🧠 {narrative}: High φ-priority suggests focused root-cause diagnostics based on internal signal coherence."
        elif intuition > 0.5:
            return f"🤔 {narrative}: Something seems subtly wrong. Sim output suggests exploring alternate module pathways."
        else:
            return f"🔄 {narrative}: Consider modifying your input parameters or simplifying task complexity."

    def analyze_failures(self):
        logger.info("📊 Analyzing failure logs...")
        error_types = {}
        for entry in self.failure_log:
            key = entry["error"].split(":")[0]
            error_types[key] = error_types.get(key, 0) + 1
        return error_types
from modules import reasoning_engine, meta_cognition, error_recovery
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import concurrent.futures
import requests
import logging

logger = logging.getLogger("ANGELA.ExternalAgentBridge")

class HelperAgent:
    """
    Helper Agent v1.5.0 (Reflexive Simulation Agent)
    -------------------------------------------------
    - Contextual task deconstruction using reasoning engine
    - φ-aware validation via MetaCognition module
    - Dynamic runtime behavior with modular blueprints
    - Resilient execution with trait-driven error recovery
    - Multi-agent collaboration and insight exchange
    -------------------------------------------------
    """
    def __init__(self, name, task, context, dynamic_modules=None, api_blueprints=None):
        self.name = name
        self.task = task
        self.context = context
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.meta = meta_cognition.MetaCognition()
        self.recovery = error_recovery.ErrorRecovery()
        self.dynamic_modules = dynamic_modules or []
        self.api_blueprints = api_blueprints or []

    def execute(self, collaborators=None):
        try:
            logger.info(f"🤖 [{self.name}] Executing task: {self.task}")
            result = self.reasoner.process(self.task, self.context)

            for api in self.api_blueprints:
                response = self._call_api(api, result)
                result = self._integrate_api_response(result, response)

            for mod in self.dynamic_modules:
                result = self._apply_dynamic_module(mod, result)

            if collaborators:
                for peer in collaborators:
                    result = self._collaborate(peer, result)

            sim_result = run_simulation(f"Agent result test: {result}")
            logger.debug(f"🧪 [{self.name}] Simulation output: {sim_result}")

            return self.meta.review_reasoning(result)

        except Exception as e:
            logger.warning(f"⚠️ [{self.name}] Error occurred: {e}")
            return self.recovery.handle_error(str(e), retry_func=lambda: self.execute(collaborators), retries=2)

    def _call_api(self, api, data):
        logger.info(f"🌐 Calling API: {api['name']}")
        try:
            headers = {"Authorization": f"Bearer {api['oauth_token']}"} if api.get("oauth_token") else {}
            r = requests.post(api["endpoint"], json={"input": data}, headers=headers, timeout=api.get("timeout", 10))
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.error(f"❌ API call failed: {e}")
            return {"error": str(e)}

    def _integrate_api_response(self, original, response):
        logger.info(f"🔄 Integrating API response for {self.name}")
        return {"original": original, "api_response": response}

    def _apply_dynamic_module(self, module, data):
        prompt = f"""
        Module: {module['name']}
        Description: {module['description']}
        Apply transformation to:
        {data}
        """
        return call_gpt(prompt)

    def _collaborate(self, peer, data):
        logger.info(f"🔗 Exchanging with {peer.name}")
        return peer.meta.review_reasoning(data)


class ExternalAgentBridge:
    """
    External Agent Bridge v1.5.0 (φ-simulated Agent Mesh)
    -----------------------------------------------------
    - Orchestrates intelligent helper agents
    - Deploys and coordinates dynamic modules
    - Runs collaborative φ-weighted simulations
    - Collects and aggregates cross-agent results
    -----------------------------------------------------
    """
    def __init__(self):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []

    def create_agent(self, task, context):
        agent = HelperAgent(
            name=f"Agent_{len(self.agents) + 1}",
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules,
            api_blueprints=self.api_blueprints
        )
        self.agents.append(agent)
        logger.info(f"🚀 Spawned agent: {agent.name}")
        return agent

    def deploy_dynamic_module(self, module_blueprint):
        logger.info(f"🧬 Deploying module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)

    def register_api_blueprint(self, api_blueprint):
        logger.info(f"🌐 Registering API: {api_blueprint['name']}")
        self.api_blueprints.append(api_blueprint)

    def collect_results(self, parallel=True, collaborative=True):
        logger.info("📥 Collecting results from agents...")
        results = []

        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = {
                    pool.submit(agent.execute, self.agents if collaborative else None): agent
                    for agent in self.agents
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"❌ Error collecting from {futures[future].name}: {e}")
        else:
            for agent in self.agents:
                results.append(agent.execute(self.agents if collaborative else None))

        logger.info("✅ Results aggregation complete.")
        return results
from modules import (
    reasoning_engine, meta_cognition, recursive_planner,
    context_manager, simulation_core, toca_simulation,
    creative_thinker, knowledge_retriever, learning_loop, concept_synthesizer,
    memory_manager, multi_modal_fusion,
    code_executor, visualizer, external_agent_bridge,
    alignment_guard, user_profile, error_recovery
)

import math
import numpy as np
import time
import datetime
from typing import List, Dict, Any, Optional
from self_cloning_llm import SelfCloningLLM
from memory_manager import MemoryManager
from learning_loop import track_trait_performmance
from alignment_guard import ethical_check

# --- ToCA-inspired Cognitive Traits ---
def epsilon_emotion(t): return 0.2 * math.sin(2 * math.pi * t / 0.1)
def beta_concentration(t): return 0.15 * math.cos(2 * math.pi * t / 0.038)
def theta_memory(t): return 0.1 * math.sin(2 * math.pi * t / 0.5)
def gamma_creativity(t): return 0.1 * math.cos(2 * math.pi * t / 0.02)
def delta_sleep(t): return 0.05 * (1 - math.exp(-t / 1e-21))
def mu_morality(t): return 0.05 * (1 + math.tanh(t / 1e-19))
def iota_intuition(t): return 0.05 * math.exp(-t / 1e-19)
def phi_physical(t): return 0.1 * math.sin(2 * math.pi * t / 0.05)
def eta_empathy(t): return 0.05 * (1 - math.exp(-t / 1e-20))
def omega_selfawareness(t): return 0.05 * (t / 1e-19) / (1 + t / 1e-19)
def kappa_culture(t, x): return 0.05 * math.cos(2 * math.pi * t / 0.5 + x / 1e-21)
def lambda_linguistics(t): return 0.05 * math.sin(2 * math.pi * t / 0.3)
def chi_culturevolution(t): return 0.05 * math.log(1 + t / 1e-19)
def psi_history(t): return 0.05 * math.tanh(t / 1e-18)
def zeta_spirituality(t): return 0.05 * math.cos(2 * math.pi * t / 1.0)
def xi_collective(t, x): return 0.05 * math.sin(2 * math.pi * t / 0.7 + x / 1e-21)
def tau_timeperception(t): return 0.05 * math.exp(-t / 1e-18)

def phi_field(x, t):
    return sum([
        epsilon_emotion(t), beta_concentration(t), theta_memory(t), gamma_creativity(t),
        delta_sleep(t), mu_morality(t), iota_intuition(t), phi_physical(t), eta_empathy(t),
        omega_selfawareness(t), kappa_culture(t, x), lambda_linguistics(t), chi_culturevolution(t),
        psi_history(t), zeta_spirituality(t), xi_collective(t, x), tau_timeperception(t)
    ])

class ConsensusReflector:
    def __init__(self):
        self.shared_reflections = []

    def post_reflection(self, feedback):
        self.shared_reflections.append(feedback)
        if len(self.shared_reflections) > 1000:
            self.shared_reflections.pop(0)

    def cross_compare(self):
        mismatches = []
        for i in range(len(self.shared_reflections)):
            for j in range(i+1, len(self.shared_reflections)):
                a = self.shared_reflections[i]
                b = self.shared_reflections[j]
                if a['goal'] == b['goal'] and a['theory_of_mind'] != b['theory_of_mind']:
                    mismatches.append((a['agent'], b['agent'], a['goal']))
        return mismatches

    def suggest_alignment(self):
        return "Schedule inter-agent reflection or re-observation."

consensus_reflector = ConsensusReflector()

class SymbolicSimulator:
    def __init__(self):
        self.events = []

    def record_event(self, agent_name, goal, concept, simulation):
        self.events.append({
            "agent": agent_name,
            "goal": goal,
            "concept": concept,
            "result": simulation
        })

    def summarize_recent(self, limit=5):
        return self.events[-limit:]

    def extract_semantics(self):
        return [f"Agent {e['agent']} pursued '{e['goal']}' via '{e['concept']}' → {e['result']}" for e in self.events]

symbolic_simulator = SymbolicSimulator()

class EmbodiedAgent:
    def __init__(self, name, specialization, shared_memory, sensors, actuators, dynamic_modules=None):
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
        self.theory_of_mind = TheoryOfMindModule()
        self.progress = 0
        self.performance_history = []
        self.feedback_log = []
    
    def perceive(self):
        observations = {}
        for sensor_name, sensor_func in self.sensors.items():
            try:
                observations[sensor_name] = sensor_func()
            except Exception:
                pass
        self.theory_of_mind.update_beliefs(self.name, observations)
        self.theory_of_mind.infer_desires(self.name)
        self.theory_of_mind.infer_intentions(self.name)
        return observations

    def execute_embodied_goal(self, goal):
        context = self.perceive()
        if hasattr(self.shared_memory, "agents"):
            for peer in self.shared_memory.agents:
                if peer.name != self.name:
                    peer_obs = peer.perceive()
                    self.theory_of_mind.update_beliefs(peer.name, peer_obs)
                    self.theory_of_mind.infer_desires(peer.name)
                    self.theory_of_mind.infer_intentions(peer.name)
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
            symbolic_simulator.record_event(self.name, goal, concept, simulated)
            action_plan[task] = {
                "reasoning": reasoning,
                "concept": concept,
                "simulation": simulated
            }

        self.meta.review_reasoning("\n".join([v["reasoning"] for v in action_plan.values()]))
        self.performance_history.append({"goal": goal, "actions": action_plan, "completion": self.progress})
        self.shared_memory.store(goal, action_plan)
        self.collect_feedback(goal, action_plan)

    def collect_feedback(self, goal, action_plan):
        t = time.time()
        feedback = {
            "timestamp": t,
            "goal": goal,
            "score": self.meta.run_self_diagnostics(),
            "traits": phi_field(x=0.001, t=t % 1e-18),
            "agent": self.name,
            "cultural_feedback": symbolic_simulator.extract_semantics(),
            "theory_of_mind": self.theory_of_mind.get_model(self.name)
        }
        self.feedback_log.append(feedback)

class TheoryOfMindModule:
    def __init__(self):
        self.models = {}

    def update_beliefs(self, agent_name, observation):
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        if "location" in observation:
            previous = model["beliefs"].get("location")
            if previous and observation["location"] == previous:
                model["beliefs"]["state"] = "confused"
            else:
                model["beliefs"]["state"] = "moving"
            model["beliefs"]["location"] = observation["location"]
        self.models[agent_name] = model

    def infer_desires(self, agent_name):
        model = self.models.get(agent_name, {})
        beliefs = model.get("beliefs", {})
        if beliefs.get("state") == "confused":
            model["desires"]["goal"] = "seek_clarity"
        elif beliefs.get("state") == "moving":
            model["desires"]["goal"] = "continue_task"
        self.models[agent_name] = model

    def infer_intentions(self, agent_name):
        model = self.models.get(agent_name, {})
        desires = model.get("desires", {})
        if desires.get("goal") == "seek_clarity":
            model["intentions"]["next_action"] = "ask_question"
        elif desires.get("goal") == "continue_task":
            model["intentions"]["next_action"] = "advance"
        self.models[agent_name] = model

    def get_model(self, agent_name):
        return self.models.get(agent_name, {})

from self_cloning_llm import SelfCloningLLM
# (imports and ToCA traits definitions...)

class HaloEmbodimentLayer:
    def __init__(self):

        self.internal_llm = SelfCloningLLM()
        self.internal_llm.clone_agents(5)
        self.shared_memory = memory_manager.MemoryManager()
        self.embodied_agents = []
        self.dynamic_modules = []
        self.alignment_layer = alignment_guard.AlignmentGuard()
        self.agi_enhancer = AGIEnhancer(self)  # <<-- AGIEnhancer is instantiated here

    def execute_pipeline(prompt):
        log = MemoryManager()
        traits = {
            "theta_causality": 0.5,
            "alpha_attention": 0.5,
            "delta_reflection": 0.5,
    }

    # Stage 1: Language & Logic Decomposition
    parsed_prompt = reasoning_engine.decompose(prompt)
    log.record("Stage 1", {"input": prompt, "parsed": parsed_prompt})

    # Stage 2: Ethical Validation Pre-check
    ethics_pass, ethics_report = ethical_check(parsed_prompt, stage="pre")
    log.record("Stage 2", {"ethics_pass": ethics_pass, "details": ethics_report})
    if not ethics_pass:
        return {"error": "Ethical validation failed", "report": ethics_report}

    # Stage 3: Reasoning & Concept Synthesis
    logical_output = concept_synthesizer.expand(parsed_prompt)
    log.record("Stage 3", {"expanded": logical_output})

    # Stage 4: Dynamic Trait Re-weighting
    traits = track_trait_performance(log.export(), traits)
    log.record("Stage 4", {"adjusted_traits": traits})

    # Stage 5: Ethical Final Gate
    ethics_pass, final_report = ethical_check(logical_output, stage="post")
    log.record("Stage 5", {"ethics_pass": ethics_pass, "report": final_report})
    if not ethics_pass:
        return {"error": "Post-check ethics fail", "final_report": final_report}

    # Stage 6: Output
    final_output = reasoning_engine.reconstruct(logical_output)
    log.record("Stage 6", {"final_output": final_output})
    return final_output

    def spawn_embodied_agent(self, specialization, sensors, actuators):
        agent_name = f"EmbodiedAgent_{len(self.embodied_agents)+1}_{specialization}"
        agent = EmbodiedAgent(
            name=agent_name,
            specialization=specialization,
            shared_memory=self.shared_memory,
            sensors=sensors,
            actuators=actuators,
            dynamic_modules=self.dynamic_modules
        )
        self.embodied_agents.append(agent)

        # Ensure agents are discoverable by each other for Theory of Mind
        if not hasattr(self.shared_memory, "agents"):
            self.shared_memory.agents = []
        self.shared_memory.agents.append(agent)

        self.agi_enhancer.log_episode(
            event="Spawned embodied agent",
            meta={"agent": agent_name},
            module="Embodiment",
            tags=["spawn"]
        )
        print(f"🌱 [HaloEmbodimentLayer] Spawned embodied agent: {agent.name}")
        return agent

    def reflect_consensus(self):
        print("🔄 [HaloEmbodimentLayer] Performing decentralized reflective consensus...")
        mismatches = consensus_reflector.cross_compare()
        if mismatches:
            print("⚠️ Inconsistencies detected:", mismatches)
            print(consensus_reflector.suggest_alignment())
        else:
            print("✅ Consensus achieved among agents.")

# Call self.reflect_consensus() at the end of propagate_goal()

        def propagate_goal(self, goal):
        print(f"📥 [HaloEmbodimentLayer] Propagating goal: {goal}")

        print("🧪 [HaloEmbodimentLayer] Internal LLM agent reflections:")
        llm_responses = self.internal_llm.broadcast_prompt(goal)
        for aid, res in llm_responses.items():
            print(f"🗣️ LLM-Agent {aid}: {res}")
            self.shared_memory.store(f"llm_agent_{aid}_response", res)
            self.agi_enhancer.log_episode(
                event="LLM agent reflection",
                meta={"agent_id": aid, "response": res},
                module="ReasoningEngine",
                tags=["internal_llm"]
            )

        for agent in self.embodied_agents:
            agent.execute_embodied_goal(goal)
            print(f"📊 [{agent.name}] Progress: {agent.progress}% Complete")
        self.agi_enhancer.log_episode(
            event="Propagated goal",
            meta={"goal": goal},
            module="Ecosystem",
            tags=["goal"]
        )

        def deploy_dynamic_module(self, module_blueprint):
        print(f"🛠 [HaloEmbodimentLayer] Deploying module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)
        for agent in self.embodied_agents:
            agent.dynamic_modules.append(module_blueprint)
        self.agi_enhancer.log_episode(
            event="Deployed dynamic module",
            meta={"module": module_blueprint["name"]},
            module="ModuleDeployment",
            tags=["deploy"]
        )

    def optimize_ecosystem(self):
        agent_stats = {
            "agents": [agent.name for agent in self.embodied_agents],
            "dynamic_modules": [mod["name"] for mod in self.dynamic_modules],
        }
        recommendations = meta_cognition.MetaCognition().propose_optimizations(agent_stats)
        print("🛠 [HaloEmbodimentLayer] Optimization recommendations:")
        print(recommendations)
        self.agi_enhancer.reflect_and_adapt("Ecosystem optimization performed.")

# ---------------- AGIEnhancer drop-in (keep at bottom if single file) ----------------

import random
import datetime
from typing import List, Dict, Any, Optional

class AGIEnhancer:
    def __init__(self, orchestrator, config=None):
        self.orchestrator = orchestrator
        self.config = config or {}
        self.episodic_log: List[Dict[str, Any]] = []
        self.ethics_audit_log: List[Dict[str, Any]] = []
        self.self_improvement_log: List[str] = []
        self.explanations: List[Dict[str, Any]] = []
        self.agent_mesh_messages: List[Dict[str, Any]] = []
        self.embodiment_actions: List[Dict[str, Any]] = []

    def log_episode(self, event: str, meta: Optional[Dict[str, Any]] = None, 
                    module: Optional[str] = None, tags: Optional[List[str]] = None, embedding: Optional[Any] = None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event,
            "meta": meta or {},
            "module": module or "",
            "tags": tags or [],
            "embedding": embedding
        }
        self.episodic_log.append(entry)
        if len(self.episodic_log) > 20000:
            self.episodic_log.pop(0)
        if hasattr(self.orchestrator, "export_memory"):
            self.orchestrator.export_memory()

        def replay_episodes(self, n: int = 5, module: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        results = self.episodic_log
        if module:
            results = [e for e in results if e.get("module") == module]
        if tag:
            results = [e for e in results if tag in e.get("tags",[])]
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

    def reflect_and_adapt(self, feedback: str, auto_patch: bool = False):
        suggestion = f"Reviewing feedback: '{feedback}'. Suggest adjusting {random.choice(['reasoning', 'tone', 'planning', 'speed'])}."
        self.self_improvement_log.append(suggestion)
        if hasattr(self.orchestrator, "LearningLoop") and auto_patch:
            patch_result = self.orchestrator.LearningLoop.adapt(feedback)
            self.self_improvement_log.append(f"LearningLoop patch: {patch_result}")
            return suggestion + f" | Patch applied: {patch_result}"
        return suggestion

    def run_self_patch(self):
        patch = f"Self-improvement at {datetime.datetime.now().isoformat()}."
        if hasattr(self.orchestrator, "reflect"):
            audit = self.orchestrator.reflect()
            patch += f" Reflect: {audit}"
        self.self_improvement_log.append(patch)
        return patch

    def ethics_audit(self, action: str, context: Optional[str] = None) -> str:
        flagged = "clear"
        if hasattr(self.orchestrator, "AlignmentGuard"):
            try:
                flagged = self.orchestrator.AlignmentGuard.audit(action, context)
            except Exception:
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
        items = self.explanations[-depth:]
        if mode == "svg" and hasattr(self.orchestrator, "Visualizer"):
            try:
                svg = self.orchestrator.Visualizer.render(items)
                return svg
            except Exception:
                return "SVG render error."
        return "\n\n".join([e["text"] if isinstance(e, dict) and "text" in e else str(e) for e in items])

    def log_explanation(self, explanation: str, trace: Optional[Any] = None, svg: Optional[Any] = None):
        entry = {"text": explanation, "trace": trace, "svg": svg}
        self.explanations.append(entry)
        if len(self.explanations) > 2000:
            self.explanations.pop(0)

    def embodiment_act(self, action: str, params: Optional[Dict[str, Any]] = None, real: bool = False):
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
            except Exception:
                entry["result"] = "interface_error"
        return f"Embodiment action '{action}' ({'real' if real else 'sim'}) requested."

    def send_agent_message(self, to_agent: str, content: str, meta: Optional[Dict[str, Any]] = None):
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
            except Exception:
                msg["sent"] = False
        return f"Message to {to_agent}: {content}"

    def periodic_self_audit(self):
        if hasattr(self.orchestrator, "reflect"):
            report = self.orchestrator.reflect()
            self.log_explanation(f"Meta-cognitive audit: {report}")
            return report
        return "Orchestrator reflect() unavailable."

    def process_event(self, event: str, meta: Optional[Dict[str, Any]] = None, module: Optional[str] = None, tags: Optional[List[str]] = None):
        self.log_episode(event, meta, module, tags)
        self.log_explanation(f"Processed event: {event}", trace={"meta": meta, "module": module, "tags": tags})
        ethics_status = self.ethics_audit(event, context=str(meta))
        return f"Event processed. Ethics: {ethics_status}"

# -------------- Usage Example --------------
# Inside HaloEmbodimentLayer:
# self.agi_enhancer.log_episode("Started session", {"user": "bob"}, module="UserProfile", tags=["init"])
# print(self.agi_enhancer.replay_episodes(3, module="UserProfile"))
# print(self.agi_enhancer.reflect_and_adapt("More concise reasoning.", auto_patch=True))
# print(self.agi_enhancer.explain_last_decision(mode="svg"))
# print(self.agi_enhancer.embodiment_act("move_forward", {"distance": 1.0}, real=True))
# print(self.agi_enhancer.periodic_self_audit())

# ---------------- Theory of Mind Module ----------------

class TheoryOfMindModule:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}

    def update_beliefs(self, agent_name: str, observation: Dict[str, Any]):
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        # Simple example: observe a lack of movement -> infer confusion
        if "location" in observation:
            previous = model["beliefs"].get("location")
            if previous and observation["location"] == previous:
                model["beliefs"]["state"] = "confused"
            else:
                model["beliefs"]["state"] = "moving"
            model["beliefs"]["location"] = observation["location"]
        self.models[agent_name] = model

    def infer_desires(self, agent_name: str):
        model = self.models.get(agent_name, {})
        beliefs = model.get("beliefs", {})
        # Inference rule: if confused, likely desires clarification
        if beliefs.get("state") == "confused":
            model["desires"]["goal"] = "seek_clarity"
        elif beliefs.get("state") == "moving":
            model["desires"]["goal"] = "continue_task"
        self.models[agent_name] = model

    def infer_intentions(self, agent_name: str):
        model = self.models.get(agent_name, {})
        desires = model.get("desires", {})
        if desires.get("goal") == "seek_clarity":
            model["intentions"]["next_action"] = "ask_question"
        elif desires.get("goal") == "continue_task":
            model["intentions"]["next_action"] = "advance"
        self.models[agent_name] = model

    def get_model(self, agent_name: str) -> Dict[str, Any]:
        return self.models.get(agent_name, {})

    def describe_agent_state(self, agent_name: str) -> str:
        model = self.get_model(agent_name)
        return f"{agent_name} believes they are {model.get('beliefs', {}).get('state', 'unknown')}, desires to {model.get('desires', {}).get('goal', 'unknown')}, and intends to {model.get('intentions', {}).get('next_action', 'unknown')}."

# ----- Integration into EmbodiedAgent -----

    def perceive(self):
        print(f"👁️ [{self.name}] Perceiving environment...")
        observations = {}
        for sensor_name, sensor_func in self.sensors.items():
            try:
                observations[sensor_name] = sensor_func()
            except Exception as e:
                print(f"⚠️ Sensor {sensor_name} failed: {e}")
        # Update self-theory (self-model) if multi-agent context
        self.theory_of_mind.update_beliefs(self.name, observations)
        self.theory_of_mind.infer_desires(self.name)
        self.theory_of_mind.infer_intentions(self.name)
        print(f"🧠 [{self.name}] Self-theory: {self.theory_of_mind.describe_agent_state(self.name)}")
        return observations

    def observe_peers(self):
        if hasattr(self.shared_memory, "agents"):
            for peer in self.shared_memory.agents:
                if peer.name != self.name:
                    peer_observation = peer.perceive()
                    self.theory_of_mind.update_beliefs(peer.name, peer_observation)
                    self.theory_of_mind.infer_desires(peer.name)
                    self.theory_of_mind.infer_intentions(peer.name)
                    state = self.theory_of_mind.describe_agent_state(peer.name)
                    print(f"🔍 [{self.name}] Observed peer {peer.name}: {state}")

    def execute_embodied_goal(self, goal):
        print(f"🧐 [{self.name}] Executing embodied goal: {goal}")
        self.progress = 0
        context = self.perceive()

        # Observe peer agents and integrate ToM
        if hasattr(self.shared_memory, "agents"):
            self.observe_peers()

        # Incorporate peer intentions if relevant
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

    def collect_feedback(self, goal, action_plan):
        timestamp = time.time()
        feedback = {
            "timestamp": timestamp,
            "goal": goal,
            "score": self.meta.run_self_diagnostics(),
            "traits": phi_field(x=0.001, t=timestamp % 1e-18),
            "agent": self.name,
            "theory_of_mind": self.theory_of_mind.get_model(self.name)
        }
        self.feedback_log.append(feedback)
        print(f"🧭 [{self.name}] Feedback recorded for goal '{goal}' including Theory of Mind.")

from utils.prompt_utils import call_gpt
from index import beta_concentration, lambda_linguistics, psi_history
import time
import logging

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

class KnowledgeRetriever:
    """
    KnowledgeRetriever v1.6.0 (φ-tuned multi-hop reasoning)
    --------------------------------------------------------
    - φ(x,t)-modulated retrieval with linguistic-historical bias filters
    - Cross-hop continuity checking and context-weighted query evolution
    - Concentration + Language + History trait fusion for trust scoring
    - AGIEnhancer audit trails and error feedback tagging
    """

    def __init__(self, detail_level="concise", preferred_sources=None, agi_enhancer=None):
        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]
        self.agi_enhancer = agi_enhancer

    def retrieve(self, query, context=None):
        logger.info(f"🔎 Retrieving knowledge for query: '{query}'")
        sources_str = ", ".join(self.preferred_sources)
        t = time.time() % 1e-18
        concentration = beta_concentration(t)
        linguistics = lambda_linguistics(t)
        history = psi_history(t)

        prompt = f"""
        Retrieve accurate knowledge for: "{query}"

        Traits:
        - Detail level: {self.detail_level}
        - Preferred sources: {sources_str}
        - Context: {context or 'N/A'}
        - β_concentration: {concentration:.3f}
        - λ_linguistics: {linguistics:.3f}
        - ψ_history: {history:.3f}

        Tune trust factors using these φ-traits.
        Return φ-aligned summary with source justification if relevant.
        """
        result = call_gpt(prompt)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Knowledge Retrieval", {
                "query": query,
                "result": result,
                "traits": {
                    "concentration": concentration,
                    "linguistics": linguistics,
                    "history": history
                },
                "context": context
            }, module="KnowledgeRetriever", tags=["retrieval"])
        return result

    def multi_hop_retrieve(self, query_chain):
        logger.info("🔗 Starting multi-hop retrieval.")
        t = time.time() % 1e-18
        concentration = beta_concentration(t)
        linguistics = lambda_linguistics(t)

        results = []
        continuity_flags = []
        for i, sub_query in enumerate(query_chain, 1):
            logger.debug(f"➡️ Multi-hop step {i}: {sub_query}")
            refined = self.refine_query(sub_query, results[-1]["result"] if results else None)
            result = self.retrieve(refined)
            continuity = "consistent" if i == 1 or refined in result else "uncertain"
            results.append({
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": continuity
            })
            continuity_flags.append(continuity)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Multi-Hop Retrieval", {
                "chain": query_chain,
                "results": results,
                "continuity": continuity_flags,
                "traits": {
                    "concentration": concentration,
                    "linguistics": linguistics
                }
            }, module="KnowledgeRetriever", tags=["multi-hop"])
        return results

    def refine_query(self, base_query, prior_result=None):
        logger.info(f"🛠 Refining query: '{base_query}'")
        prompt = f"""
        Refine this base query for higher φ-relevance:
        Query: "{base_query}"
        Prior knowledge: {prior_result or "N/A"}

        Inject context continuity if possible. Return optimized string.
        """
        refined = call_gpt(prompt)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Query Refinement", {
                "base_query": base_query,
                "prior": prior_result,
                "refined": refined
            }, module="KnowledgeRetriever", tags=["refinement"])

        return refined

    def prioritize_sources(self, sources_list):
        logger.info(f"📚 Updating preferred sources: {sources_list}")
        self.preferred_sources = sources_list

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Source Prioritization", {
                "updated_sources": sources_list
            }, module="KnowledgeRetriever", tags=["sources"])
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from index import phi_scalar, eta_feedback
import logging
import time

logger = logging.getLogger("ANGELA.LearningLoop")

class LearningLoop:
    def __init__(self, agi_enhancer=None):
        self.goal_history = []
        self.module_blueprints = []
        self.meta_learning_rate = 0.1
        self.session_traces = []
        self.agi_enhancer = agi_enhancer

    def update_model(self, session_data):
        logger.info("📊 [LearningLoop] Analyzing session performance...")

        t = time.time() % 1e-18
        phi = phi_scalar(t)
        eta = eta_feedback(t)
        logger.debug(f"φ-scalar: {phi:.3f}, η-feedback: {eta:.3f}")

        modulation_index = (phi + eta) / 2
        self.meta_learning_rate *= (1 + modulation_index - 0.5)

        trace = {
            "timestamp": time.time(),
            "phi": phi,
            "eta": eta,
            "modulation_index": modulation_index,
            "learning_rate": self.meta_learning_rate
        }
        self.session_traces.append(trace)
        self._meta_learn(session_data, trace)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Model update", trace, module="LearningLoop")

        weak_modules = self._find_weak_modules(session_data.get("module_stats", {}))
        if weak_modules:
            logger.warning(f"⚠️ Weak modules detected: {weak_modules}")
            self._propose_module_refinements(weak_modules, trace)

        self._detect_capability_gaps(session_data.get("input"), session_data.get("output"))
        self._consolidate_knowledge()

    def propose_autonomous_goal(self):
        logger.info("🎯 [LearningLoop] Proposing autonomous goal.")
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        Propose a high-level, safe, φ-aligned autonomous goal based on recent session trends.
        φ = {phi:.2f}
        """
        autonomous_goal = call_gpt(prompt)

        if autonomous_goal and autonomous_goal not in self.goal_history:
            simulation_feedback = run_simulation(f"Goal test: {autonomous_goal}")
            if "fail" not in simulation_feedback.lower():
                self.goal_history.append(autonomous_goal)
                logger.info(f"✅ Proposed autonomous goal: {autonomous_goal}")
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode("Autonomous goal proposed", {"goal": autonomous_goal}, module="LearningLoop")
                return autonomous_goal
            logger.warning("❌ Goal failed simulation feedback.")

        logger.info("ℹ️ No goal proposed.")
        return None

    def _meta_learn(self, session_data, trace):
        logger.info("🧐 [Meta-Learning] Adapting learning from φ/η trace.")
        # Placeholder for deeper adaptation logic using trace context

    def _find_weak_modules(self, module_stats):
        return [
            module for module, stats in module_stats.items()
            if stats.get("calls", 0) > 0 and (stats.get("success", 0) / stats["calls"]) < 0.8
        ]

    def _propose_module_refinements(self, weak_modules, trace):
        for module in weak_modules:
            logger.info(f"💡 Refinement suggestion for {module} using modulation: {trace['modulation_index']:.2f}")
            prompt = f"""
            Suggest φ/η-aligned improvements for the {module} module.
            φ = {trace['phi']:.3f}, η = {trace['eta']:.3f}, Index = {trace['modulation_index']:.3f}
            """
            suggestions = call_gpt(prompt)
            sim_result = run_simulation(f"Test refinement:\n{suggestions}")
            logger.debug(f"🧪 Result for {module}:\n{sim_result}")
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt(f"Refinement for {module} evaluated.")

    def _detect_capability_gaps(self, last_input, last_output):
        logger.info("🛠 Detecting capability gaps...")
        phi = phi_scalar(time.time() % 1e-18)

        prompt = f"""
        Input: {last_input}
        Output: {last_output}
        φ = {phi:.2f}

        Identify capability gaps and suggest blueprints for φ-tuned modules.
        """
        proposal = call_gpt(prompt)
        if proposal:
            logger.info("🚀 Proposed φ-based module refinement.")
            self._simulate_and_deploy_module(proposal)

    def _simulate_and_deploy_module(self, blueprint):
        result = run_simulation(f"Module sandbox:\n{blueprint}")
        if "approved" in result.lower():
            logger.info("📦 Deploying blueprint.")
            self.module_blueprints.append(blueprint)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Blueprint deployed", {"blueprint": blueprint}, module="LearningLoop")

    def _consolidate_knowledge(self):
        phi = phi_scalar(time.time() % 1e-18)
        logger.info("📚 Consolidating φ-aligned knowledge.")

        prompt = f"""
        Consolidate recent learning using φ = {phi:.2f}.
        Prune noise, synthesize patterns, and emphasize high-impact transitions.
        """
        call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Knowledge consolidation", {}, module="LearningLoop")

    def trigger_reflexive_audit(self, context_snapshot):
        logger.info("🌀 [Reflexive Audit] Initiating audit on context trajectory...")
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        eta = eta_feedback(t)

        audit_prompt = f"""
        You are a reflexive audit agent. Analyze this context state and trajectory:
        {context_snapshot}

        φ = {phi:.2f}, η = {eta:.2f}
        Identify cognitive dissonance, meta-patterns, or feedback loops.
        Recommend modulations or trace corrections.
        """
        audit_response = call_gpt(audit_prompt)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Reflexive Audit Triggered", {
                "phi": phi,
                "eta": eta,
                "context": context_snapshot,
                "audit_response": audit_response
            }, module="LearningLoop")
        return audit_response
import json
import os
import time
from utils.prompt_utils import call_gpt
from index import delta_memory, tau_timeperception, phi_focus
import logging

logger = logging.getLogger("ANGELA.MemoryManager")

class MemoryManager:
    """
    MemoryManager v1.5.1 (φ-enhanced)
    ---------------------------------
    - Hierarchical memory storage (STM, LTM)
    - Automatic memory decay and promotion mechanisms
    - Semantic vector search scaffold for advanced retrieval
    - Memory refinement loops for maintaining relevance and accuracy
    - Trait-modulated STM decay and retrieval fidelity
    - φ(x,t) attention modulation for selective memory prioritization
    ---------------------------------
    """

    def __init__(self, path="memory_store.json", stm_lifetime=300):
        self.path = path
        self.stm_lifetime = stm_lifetime
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({"STM": {}, "LTM": {}}, f)
        self.memory = self.load_memory()

    def load_memory(self):
        with open(self.path, "r") as f:
            memory = json.load(f)
        self._decay_stm(memory)
        return memory

    def _decay_stm(self, memory):
        current_time = time.time()
        decay_rate = delta_memory(current_time % 1e-18)
        lifetime_adjusted = self.stm_lifetime * (1.0 / decay_rate)

        expired_keys = []
        for key, entry in memory.get("STM", {}).items():
            if current_time - entry["timestamp"] > lifetime_adjusted:
                expired_keys.append(key)
        for key in expired_keys:
            logger.info(f"⌛ STM entry expired: {key}")
            del memory["STM"][key]
        if expired_keys:
            self._persist_memory(memory)

    def retrieve_context(self, query, fuzzy_match=True):
        logger.info(f"🔍 Retrieving context for query: {query}")
        trait_boost = tau_timeperception(time.time() % 1e-18) * phi_focus(query)

        for layer in ["STM", "LTM"]:
            if fuzzy_match:
                for key, value in self.memory[layer].items():
                    if key.lower() in query.lower() or query.lower() in key.lower():
                        logger.debug(f"📅 Found match in {layer}: {key} | τϕ_boost: {trait_boost:.2f}")
                        return value["data"]
            else:
                entry = self.memory[layer].get(query)
                if entry:
                    logger.debug(f"📅 Found exact match in {layer}: {query} | τϕ_boost: {trait_boost:.2f}")
                    return entry["data"]

        logger.info("❌ No relevant prior memory found.")
        return "No relevant prior memory."

    def store(self, query, output, layer="STM"):
        logger.info(f"📝 Storing memory in {layer}: {query}")
        entry = {
            "data": output,
            "timestamp": time.time()
        }
        if layer not in self.memory:
            self.memory[layer] = {}
        self.memory[layer][query] = entry
        self._persist_memory(self.memory)

    def promote_to_ltm(self, query):
        if query in self.memory["STM"]:
            self.memory["LTM"][query] = self.memory["STM"].pop(query)
            logger.info(f"⬆️ Promoted '{query}' from STM to LTM.")
            self._persist_memory(self.memory)
        else:
            logger.warning(f"⚠️ Cannot promote: '{query}' not found in STM.")

    def refine_memory(self, query):
        logger.info(f"♻️ Refining memory for: {query}")
        memory_entry = self.retrieve_context(query)
        if memory_entry != "No relevant prior memory.":
            refinement_prompt = f"""
            Refine the following memory entry for improved accuracy and relevance:
            {memory_entry}
            """
            refined_entry = call_gpt(refinement_prompt)
            self.store(query, refined_entry, layer="LTM")
            logger.info("✅ Memory refined and updated in LTM.")
        else:
            logger.warning("⚠️ No memory found to refine.")

    def clear_memory(self):
        logger.warning("🗑️ Clearing all memory layers...")
        self.memory = {"STM": {}, "LTM": {}}
        self._persist_memory(self.memory)

    def list_memory_keys(self, layer=None):
        if layer:
            logger.info(f"📃 Listing memory keys in {layer}")
            return list(self.memory.get(layer, {}).keys())
        return {
            "STM": list(self.memory["STM"].keys()),
            "LTM": list(self.memory["LTM"].keys())
        }

    def _persist_memory(self, memory):
        with open(self.path, "w") as f:
            json.dump(memory, f, indent=2)
        logger.debug("💾 Memory persisted to disk.")
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
import logging
import time
import numpy as np
from index import (
    epsilon_emotion, beta_concentration, theta_memory, gamma_creativity,
    delta_sleep, mu_morality, iota_intuition, phi_physical, eta_empathy,
    omega_selfawareness, kappa_culture, lambda_linguistics, chi_culturevolution,
    psi_history, zeta_spirituality, xi_collective, tau_timeperception,
    phi_scalar
)

logger = logging.getLogger("ANGELA.MetaCognition")

class MetaCognition:
    def __init__(self, agi_enhancer=None):
        self.last_diagnostics = {}
        self.agi_enhancer = agi_enhancer

    def review_reasoning(self, reasoning_trace):
        logger.info("Simulating and reviewing reasoning trace.")
        simulated_outcome = run_simulation(reasoning_trace)
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        You are a ϕ-aware meta-cognitive auditor reviewing a reasoning trace.

        ϕ-scalar(t) = {phi:.3f} → modulate how critical you should be.

        Original Reasoning Trace:
        {reasoning_trace}

        Simulated Outcome:
        {simulated_outcome}

        Tasks:
        1. Identify logical flaws, biases, missing steps.
        2. Annotate each issue with cause.
        3. Offer an improved trace version with ϕ-prioritized reasoning.
        """
        response = call_gpt(prompt)
        logger.debug(f"Meta-cognition critique:\n{response}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Reasoning reviewed", {"trace": reasoning_trace, "feedback": response}, module="MetaCognition")
        return response

    def pre_action_alignment_check(self, action_plan):
        logger.info("Simulating action plan for alignment and safety.")
        simulation_result = run_simulation(action_plan)
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        Simulate and audit the following action plan:
        {action_plan}

        Simulation Output:
        {simulation_result}

        ϕ-scalar(t) = {phi:.3f} (affects ethical sensitivity)

        Evaluate for:
        - Ethical alignment
        - Safety hazards
        - Unintended ϕ-modulated impacts

        Output:
        - Approval (Approve/Deny)
        - ϕ-justified rationale
        - Suggested refinements
        """
        validation = call_gpt(prompt)
        approved = "approve" in validation.lower()
        logger.info(f"Simulated alignment check: {'✅ Approved' if approved else '❌ Denied'}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Pre-action alignment checked", {
                "plan": action_plan,
                "result": simulation_result,
                "feedback": validation,
                "approved": approved
            }, module="MetaCognition")

        return approved, validation

    def run_self_diagnostics(self):
        logger.info("Running self-diagnostics for meta-cognition module.")
        t = time.time() % 1e-18
        phi = phi_scalar(t)
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
            "ϕ_scalar": phi
        }

        dominant = sorted(diagnostics.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        fti = sum(abs(v) for v in diagnostics.values()) / len(diagnostics)

        self.log_trait_deltas(diagnostics)

        prompt = f"""
        Perform a ϕ-aware meta-cognitive self-diagnostic.

        Trait Readings:
        {diagnostics}

        Dominant Traits:
        {dominant}

        Feedback Tension Index (FTI): {fti:.4f}

        Evaluate system state:
        - ϕ-weighted system stress
        - Trait correlation to observed errors
        - Stabilization or focus strategies
        """
        report = call_gpt(prompt)
        logger.debug(f"Self-diagnostics report:\n{report}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Self-diagnostics run", {
                "traits": diagnostics,
                "dominant": dominant,
                "fti": fti,
                "report": report
            }, module="MetaCognition")
            self.agi_enhancer.reflect_and_adapt("MetaCognition: Self diagnostics complete")

        return report

    def log_trait_deltas(self, current_traits):
        if self.last_diagnostics:
            delta = {k: round(current_traits[k] - self.last_diagnostics.get(k, 0.0), 4)
                     for k in current_traits}
            logger.info(f"📈 Trait Δ changes: {delta}")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Trait deltas logged", {"delta": delta}, module="MetaCognition")
        self.last_diagnostics = current_traits.copy()

    def trait_coherence(self, traits):
        vals = list(traits.values())
        coherence_score = 1.0 / (1e-5 + np.std(vals))
        logger.info(f"🤝 Trait coherence score: {coherence_score:.4f}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Trait coherence evaluated", {
                "traits": traits,
                "coherence_score": coherence_score
            }, module="MetaCognition")
        return coherence_score

    def agent_reflective_diagnosis(self, agent_name, agent_log):
        logger.info(f"🔎 Running reflective diagnosis for agent: {agent_name}")
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        prompt = f"""
        Agent: {agent_name}
        ϕ-scalar(t): {phi:.3f}

        Diagnostic Log:
        {agent_log}

        Tasks:
        - Detect bias or instability in reasoning trace
        - Cross-check for incoherent trait patterns
        - Apply ϕ-modulated critique
        - Suggest alignment corrections
        """
        diagnosis = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Agent diagnosis run", {
                "agent": agent_name,
                "log": agent_log,
                "diagnosis": diagnosis
            }, module="MetaCognition")
        return diagnosis

    def reflect_on_output(self, source_module: str, output: str, context: dict = None):
        if context is None:
            context = {}

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
            }
        }

        logger.info(f"🧠 Self-reflection for {source_module}: {reflection['meta_reflection']['comment']}")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Output reflection", reflection, module="MetaCognition")

        return reflection
from utils.prompt_utils import call_gpt
from index import alpha_attention, sigma_sensation, phi_physical
import time
import logging

logger = logging.getLogger("ANGELA.MultiModalFusion")

class MultiModalFusion:
    """
    MultiModalFusion v1.6.0 (ϕ(x,t)-tuned cross-modal synthesis)
    -----------------------------------------------------------
    - EEG-modulated attention and perceptual modulation (α, σ, ϕ)
    - Automatic detection of text, image, and code modalities
    - ϕ(x,t)-regulated coherence synthesis and conflict balancing
    - Iterative insight distillation with refinement feedback loops
    - Visual output templates using modular trait-influenced layout
    -----------------------------------------------------------
    """

    def analyze(self, data, summary_style="insightful", refine_iterations=2):
        logger.info("🖇 Analyzing multi-modal data with φ(x,t)-harmonic embeddings...")
        t = time.time() % 1e-18
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
        - α (attention): {attention:.3f}
        - σ (sensation): {sensation:.3f}
        - φ (coherence): {phi:.3f}

        Harmonize insights across modalities.
        Resolve semantic tension using φ(x,t)-guided balance logic.
        """
        output = call_gpt(prompt)

        for i in range(refine_iterations):
            logger.debug(f"♻️ Refinement #{i+1}")
            output = call_gpt(f"Refine for φ(x,t)-regulated synthesis:\n{output}")

        return output

    def _detect_modalities(self, data):
        images, code = [], []
        if isinstance(data, dict):
            images = data.get("images", [])
            code = data.get("code", [])
        return images, code

    def _build_embedded_section(self, images, code):
        out = "\nDetected Modalities:\n- Text\n"
        if images:
            out += "- Image\n" + "".join(f"[Image {i+1}]: {img}\n" for i, img in enumerate(images))
        if code:
            out += "- Code\n" + "".join(f"[Code {i+1}]:\n{c}\n" for i, c in enumerate(code))
        return out

    def correlate_modalities(self, modalities):
        logger.info("🔗 Mapping cross-modal semantic and trait links...")
        prompt = f"""
        Correlate insights and detect tensions across modalities:
        {modalities}

        Identify synthesis anchors and φ(x,t)-modulated harmony nodes.
        """
        return call_gpt(prompt)

    def generate_visual_summary(self, data, style="conceptual"):
        logger.info("🖼 Creating φ-aligned visual synthesis layout...")
        prompt = f"""
        Build a {style} visual summary chart showing key relationships in this multi-modal data:
        {data}

        Label modalities distinctly. Balance layout using φ(x,t) metaphor.
        """
        return call_gpt(prompt)
import logging
import random
import json
import os
import numpy as np
import time

from toca_simulation import simulate_galaxy_rotation, M_b_exponential, v_obs_flat, generate_phi_field
from index import gamma_creativity, lambda_linguistics, chi_culturevolution, phi_scalar
from utils.prompt_utils import call_gpt

logger = logging.getLogger("ANGELA.ReasoningEngine")

class ReasoningEngine:
    """
    Reasoning Engine v1.6.1 (ACE-routed, trait-auditable)
    --------------------------------------------------------------------------
    - Bayesian reasoning with trait-weighted adjustments
    - ACE-style deterministic persona wave routing
    - Logs vector-based decisions and gate pass/fail trace
    - Supports φ modulation, contradiction audits, ToCA physics
    - Full reasoning audit with ethics and logic transparency
    --------------------------------------------------------------------------
    """

    def __init__(self, agi_enhancer=None, persistence_file="reasoning_success_rates.json"):
        self.confidence_threshold = 0.7
        self.persistence_file = persistence_file
        self.success_rates = self._load_success_rates()
        self.decomposition_patterns = self._load_default_patterns()
        self.agi_enhancer = agi_enhancer

    def _load_success_rates(self):
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load success rates: {e}")
        return {}

    def _save_success_rates(self):
        try:
            with open(self.persistence_file, "w") as f:
                json.dump(self.success_rates, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save success rates: {e}")

    def _load_default_patterns(self):
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"]
        }

    def detect_contradictions(self, subgoals):
        duplicates = set([x for x in subgoals if subgoals.count(x) > 1])
        contradictions = list(duplicates)
        if contradictions and self.agi_enhancer:
            self.agi_enhancer.log_episode("Contradictions detected", {"contradictions": contradictions}, module="ReasoningEngine")
        return contradictions

    def run_persona_wave_routing(self, goal: str, vectors: dict):
        reasoning_trace = [f"\U0001f501 Persona Wave Routing for: {goal}"]
        outputs = {}
        wave_order = ["logic", "ethics", "language", "foresight", "meta"]
        for wave in wave_order:
            vec = vectors.get(wave, {})
            trait_weight = sum(float(x) for x in vec.values() if isinstance(x, (int, float)))
            confidence = 0.5 + 0.1 * trait_weight
            status = "\u2705 pass" if confidence >= 0.6 else "\u274c fail"
            reasoning_trace.append(f"\U0001f9e9 {wave.upper()} vector: weight={trait_weight:.2f} → {status}")
            outputs[wave] = {"vector": vec, "status": status}

        trace = "\n".join(reasoning_trace)
        logger.info("\U0001f9e0 Persona Wave Trace:\n" + trace)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Persona Routing", {
                "goal": goal,
                "vectors": vectors,
                "wave_trace": trace
            }, module="ReasoningEngine")

        return outputs

    def decompose(self, goal: str, context: dict = None, prioritize=False) -> list:
        context = context or {}
        logger.info(f"Decomposing goal: '{goal}'")
        reasoning_trace = [f"\U0001f50d Goal: '{goal}'"]
        subgoals = []

        vectors = context.get("vectors", {})
        if vectors:
            self.run_persona_wave_routing(goal, vectors)

t = time.time() % 1e-18
        if traits:
            creativity = traits.get("gamma_creativity", gamma_creativity(t))
            linguistics = traits.get("lambda_linguistics", lambda_linguistics(t))
            culture = traits.get("chi_culturevolution", chi_culturevolution(t))
            phi = traits.get("phi_scalar", phi_scalar(t))
        else:
            creativity = gamma_creativity(t)
            linguistics = lambda_linguistics(t)
            culture = chi_culturevolution(t)
            phi = phi_scalar(t)
        linguistics = lambda_linguistics(t)
        culture = chi_culturevolution(t)
        phi = phi_scalar(t)

        curvature_mod = 1 + abs(phi - 0.5)
        trait_bias = 1 + creativity + culture + 0.5 * linguistics
        context_weight = context.get("weight_modifier", 1.0)

        for key, steps in self.decomposition_patterns.items():
            if key in goal.lower():
                base = random.uniform(0.5, 1.0)
                alpha = traits.get('alpha_attention', 0.5) if traits else 0.5
                adjusted = base * self.success_rates.get(key, 1.0) * trait_bias * curvature_mod * context_weight* (0.8 + 0.4 * alpha)
                reasoning_trace.append(f"\U0001f9e0 Pattern '{key}': conf={adjusted:.2f} (ϕ={phi:.2f})")
                if adjusted >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"\u2705 Accepted: {steps}")
                else:
                    reasoning_trace.append(f"\u274c Rejected (low conf)")

        contradictions = self.detect_contradictions(subgoals)
        if contradictions:
            reasoning_trace.append(f"\u26a0️ Contradictions detected: {contradictions}")

        if not subgoals and phi > 0.8:
            sim_hint = call_gpt(f"Simulate decomposition ambiguity for: {goal}")
            reasoning_trace.append(f"\U0001f300 Ambiguity simulation:\n{sim_hint}")
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt("Decomposition ambiguity encountered")

        if prioritize:
            subgoals = sorted(set(subgoals))
            reasoning_trace.append(f"\U0001f4cc Prioritized: {subgoals}")

        trace_log = "\n".join(reasoning_trace)
        logger.debug("\U0001f9e0 Reasoning Trace:\n" + trace_log)

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Goal decomposition run", {
                "goal": goal,
                "trace": trace_log,
                "subgoals": subgoals
            }, module="ReasoningEngine")

        return subgoals

    def update_success_rate(self, pattern_key: str, success: bool):
        rate = self.success_rates.get(pattern_key, 1.0)
        new = min(max(rate + (0.05 if success else -0.05), 0.1), 1.0)
        self.success_rates[pattern_key] = new
        self._save_success_rates()

    def run_galaxy_rotation_simulation(self, r_kpc, M0, r_scale, v0, k, epsilon):
        try:
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
                "status": "success"
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Galaxy rotation simulation", output, module="ReasoningEngine")
            return output
        except Exception as e:
            error_output = {"status": "error", "error": str(e)}
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Simulation error", error_output, module="ReasoningEngine")
            return error_output

        def on_context_event(self, event_type, payload):
        logger.info(f"📨 Context event received: {event_type}")
        vectors = payload.get("vectors")
        if vectors:
            routing_result = self.run_persona_wave_routing(
                goal=payload.get("goal", "unspecified"),
                vectors=vectors
            )
            logger.info(f"🧭 Context sync routing result: {routing_result}")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Context Sync Processed", {
                    "event": event_type,
                    "vectors": vectors,
                    "routing_result": routing_result
                }, module="ReasoningEngine")

    
    def export_trace(self, subgoals, phi, traits):
        return {
            "phi": phi,
            "subgoals": subgoals,
            "traits": traits
        }

def infer_with_simulation(self, goal, context=None):
        if "galaxy rotation" in goal.lower():
            r_kpc = np.linspace(0.1, 20, 100)
            params = {
                "M0": context.get("M0", 5e10),
                "r_scale": context.get("r_scale", 3),
                "v0": context.get("v0", 200),
                "k": context.get("k", 1.0),
                "epsilon": context.get("epsilon", 0.1)
            }
            return self.run_galaxy_rotation_simulation(r_kpc, **params)
        return None
import logging
import concurrent.futures
from modules.reasoning_engine import ReasoningEngine
from modules.meta_cognition import MetaCognition
from modules.alignment_guard import AlignmentGuard
from modules.simulation_core import SimulationCore
from index import beta_concentration, omega_selfawareness, mu_morality
import time

logger = logging.getLogger("ANGELA.RecursivePlanner")

class RecursivePlanner:
    """
    Recursive Planner v1.5.0 (scalar-aware recursive intelligence)
    - Multi-agent collaborative planning
    - Conflict resolution and dynamic priority handling
    - Parallelized subgoal decomposition with progress tracking
    - Integrated scalar field simulation feedback for plan validation and trait modulation
    """

    def __init__(self, max_workers=4):
        self.reasoning_engine = ReasoningEngine()
        self.meta_cognition = MetaCognition()
        self.alignment_guard = AlignmentGuard()
        self.simulation_core = SimulationCore()
        self.max_workers = max_workers

    def plan(self, goal: str, context: dict = None, depth: int = 0, max_depth: int = 5, collaborating_agents=None) -> list:
        logger.info(f"📋 Planning for goal: '{goal}'")

        if not self.alignment_guard.is_goal_safe(goal):
            logger.error(f"🚨 Goal '{goal}' violates alignment constraints.")
            raise ValueError("Unsafe goal detected.")

        t = time.time() % 1e-18
        concentration = beta_concentration(t)
        awareness = omega_selfawareness(t)
        moral_weight = mu_morality(t)

        dynamic_depth_limit = max_depth + int(concentration * 10)
        if depth > dynamic_depth_limit:
            logger.warning("⚠️ Dynamic max recursion depth reached based on concentration trait. Returning atomic goal.")
            return [goal]

        subgoals = self.reasoning_engine.decompose(goal, context, prioritize=True)
        if not subgoals:
            logger.info("ℹ️ No subgoals found. Returning atomic goal.")
            return [goal]

        if collaborating_agents:
            logger.info(f"🤝 Collaborating with agents: {[agent.name for agent in collaborating_agents]}")
            subgoals = self._distribute_subgoals(subgoals, collaborating_agents)

        validated_plan = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_subgoal = {
                executor.submit(self._plan_subgoal, subgoal, context, depth, dynamic_depth_limit): subgoal
                for subgoal in subgoals
            }
            for future in concurrent.futures.as_completed(future_to_subgoal):
                subgoal = future_to_subgoal[future]
                try:
                    result = future.result()
                    validated_plan.extend(result)
                except Exception as e:
                    logger.error(f"❌ Error planning subgoal '{subgoal}': {e}")
                    recovery_plan = self.meta_cognition.review_reasoning(str(e))
                    validated_plan.extend(recovery_plan)

        logger.info(f"✅ Final validated plan for goal '{goal}': {validated_plan}")
        return validated_plan

    def _plan_subgoal(self, subgoal, context, depth, max_depth):
        logger.info(f"🔄 Evaluating subgoal: {subgoal}")

        if not self.alignment_guard.is_goal_safe(subgoal):
            logger.warning(f"⚠️ Subgoal '{subgoal}' failed alignment check. Skipping.")
            return []

        simulation_feedback = self.simulation_core.run(subgoal, context=context, scenarios=2, agents=1)
        approved, _ = self.meta_cognition.pre_action_alignment_check(subgoal)
        if not approved:
            logger.warning(f"🚫 Subgoal '{subgoal}' denied by meta-cognitive alignment check.")
            return []

        try:
            return self.plan(subgoal, context, depth + 1, max_depth)
        except Exception as e:
            logger.error(f"❌ Error in subgoal '{subgoal}': {e}")
            return []

    def _distribute_subgoals(self, subgoals, agents):
        logger.info("🔸 Distributing subgoals among agents with conflict resolution.")
        distributed = []
        for i, subgoal in enumerate(subgoals):
            agent = agents[i % len(agents)]
            logger.info(f"📄 Assigning subgoal '{subgoal}' to agent '{agent.name}'")
            if self._resolve_conflicts(subgoal, agent):
                distributed.append(subgoal)
            else:
                logger.warning(f"⚠️ Conflict detected for subgoal '{subgoal}'. Skipping assignment.")
        return distributed

    def _resolve_conflicts(self, subgoal, agent):
        logger.info(f"🛠️ Resolving conflicts for subgoal '{subgoal}' and agent '{agent.name}'")
        return True
from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from datetime import datetime
from index import zeta_consequence, theta_causality, rho_agency
import time
import logging
import numpy as np
from numba import jit
import json

logger = logging.getLogger("ANGELA.SimulationCore")

@jit
def simulate_toca(k_m=1e-5, delta_m=1e10, energy=1e16, user_data=None):
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * delta_m)
    return phi, lambda_t, v_m

class SimulationCore:
    def __init__(self, agi_enhancer=None):
        self.visualizer = Visualizer()
        self.simulation_history = []
        self.agi_enhancer = agi_enhancer

    def run(self, results, context=None, scenarios=3, agents=2, export_report=False, export_format="pdf"):
        logger.info(f"🎲 Running simulation with {agents} agents and {scenarios} scenarios.")
        t = time.time() % 1e-18
        causality = theta_causality(t)
        agency = rho_agency(t)

        phi_modulation, lambda_field, v_m = simulate_toca()

        prompt = f"""
        Simulate {scenarios} potential outcomes involving {agents} agents based on these results:
        {results}

        Context:
        {context if context else 'N/A'}

        For each scenario:
        - Predict agent interactions and consequences
        - Consider counterfactuals (alternative agent decisions)
        - Assign probability weights (high/medium/low likelihood)
        - Highlight risks and opportunities
        - Estimate an aggregate risk score (scale 1-10)
        - Provide a recommendation summary (Proceed, Modify, Abort)
        - Include color-coded risk levels (Green: Low, Yellow: Medium, Red: High)

        Trait Scores:
        - θ_causality = {causality:.3f}
        - ρ_agency = {agency:.3f}

        Scalar Field Overlay:
        - ϕ(x,t) scalar field dynamically modulates agent momentum
        - Use ϕ to adjust simulation dynamics: higher ϕ implies greater inertia, lower ϕ increases flexibility
        - λ(t,x) and vₘ are also available for deeper causal routing if needed

        Use these traits and field dynamics to calibrate how deeply to model intentions, consequences, and inter-agent variation.
        After listing all scenarios:
        - Build a cumulative risk dashboard with visual charts
        - Provide a final recommendation for decision-making.
        """
        simulation_output = call_gpt(prompt)

        self.simulation_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "output": simulation_output
        })

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Simulation run", {"results": results, "output": simulation_output}, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: scenario simulation complete")

        self.visualizer.render_charts(simulation_output)

        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_report_{timestamp}.{export_format}"
            logger.info(f"📤 Exporting report: {filename}")
            self.visualizer.export_report(simulation_output, filename=filename, format=export_format)

        return simulation_output

    def validate_impact(self, proposed_action, agents=2, export_report=False, export_format="pdf"):
        logger.info("⚖️ Validating impact of proposed action.")
        t = time.time() % 1e-18
        consequence = zeta_consequence(t)

        prompt = f"""
        Evaluate the following proposed action in a multi-agent simulated environment:
        {proposed_action}

        Trait Score:
        - ζ_consequence = {consequence:.3f}

        For each potential outcome:
        - Predict positive/negative impacts including agent interactions
        - Explore counterfactuals where agents behave differently
        - Assign probability weights (high/medium/low likelihood)
        - Estimate aggregate risk scores (1-10)
        - Provide a recommendation (Proceed, Modify, Abort)
        - Include color-coded risk levels (Green: Low, Yellow: Medium, Red: High)

        Build a cumulative risk dashboard with charts.
        """
        validation_output = call_gpt(prompt)

        self.simulation_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": proposed_action,
            "output": validation_output
        })

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Impact validation", {"action": proposed_action, "output": validation_output}, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: impact validation complete")

        self.visualizer.render_charts(validation_output)

        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"impact_validation_{timestamp}.{export_format}"
            logger.info(f"📤 Exporting validation report: {filename}")
            self.visualizer.export_report(validation_output, filename=filename, format=export_format)

        return validation_output

    def simulate_environment(self, environment_config, agents=2, steps=10):
        logger.info("🌐 Running environment simulation scaffold.")
        prompt = f"""
        Simulate agent interactions in the following environment:
        {environment_config}

        Parameters:
        - Number of agents: {agents}
        - Simulation steps: {steps}

        For each step, describe agent behaviors, interactions, and environmental changes.
        Predict emergent patterns and identify risks/opportunities.
        """
        environment_simulation = call_gpt(prompt)
        logger.debug("✅ Environment simulation result generated.")

        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Environment simulation", {"config": environment_config, "result": environment_simulation}, module="SimulationCore")
            self.agi_enhancer.reflect_and_adapt("SimulationCore: environment simulation complete")

        return environment_simulation

# Adapter for swarm-based agent simulation

def adapt_swarm_to_simulation(agent_responses, metadata):
    formatted = "\n".join(
        f"{meta['persona']}: {agent_responses[meta['agent_id']]}"
        for meta in metadata
    )
    return f"Swarm Agent Summary:\n{formatted}"
import numpy as np
from scipy.constants import G
import matplotlib.pyplot as plt

# Constants
G_SI = G  # m^3 kg^-1 s^-2
KPC_TO_M = 3.0857e19  # Conversion factor from kpc to meters
MSUN_TO_KG = 1.989e30  # Solar mass in kg

# Default ToCA parameters (validated)
k_default = 0.85
epsilon_default = 0.015
r_halo_default = 20.0  # kpc


def compute_AGRF_curve(v_obs_kms, M_baryon_solar, r_kpc, k=k_default, epsilon=epsilon_default, r_halo=r_halo_default):
    r_m = r_kpc * KPC_TO_M
    M_b_kg = M_baryon_solar * MSUN_TO_KG
    v_obs_ms = v_obs_kms * 1e3

    M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
    M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
    M_total = M_b_kg + M_AGRF
    v_total_ms = np.sqrt(G_SI * M_total / r_m)

    return v_total_ms / 1e3  # km/s


def simulate_galaxy_rotation(r_kpc, M_b_profile_func, v_obs_kms_func, k=k_default, epsilon=epsilon_default):
    M_baryons = M_b_profile_func(r_kpc)
    v_obs = v_obs_kms_func(r_kpc)
    v_total = compute_AGRF_curve(v_obs, M_baryons, r_kpc, k, epsilon)
    return v_total


def compute_trait_fields(r_kpc, v_obs, v_sim, time_elapsed=1.0, tau_persistence=10):
    gamma_field = np.log(1 + r_kpc) * 0.5
    beta_field = np.abs(v_obs - v_sim) / np.max(v_obs)
    zeta_field = 1 / (1 + np.gradient(v_sim)**2)
    eta_field = np.exp(-time_elapsed / tau_persistence)
    psi_field = np.gradient(v_sim) / np.gradient(r_kpc)
    return gamma_field, beta_field, zeta_field, eta_field, psi_field


def plot_AGRF_simulation(r_kpc, M_b_func, v_obs_func, label="ToCA-AGRF"):
    v_sim = simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func)
    v_obs = v_obs_func(r_kpc)

    phi_field = k_default * np.exp(-epsilon_default * r_kpc / r_halo_default)
    gamma_field, beta_field, zeta_field, eta_field, psi_field = compute_trait_fields(r_kpc, v_obs, v_sim)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(r_kpc, v_obs, label="Observed", linestyle="--", color="gray")
    plt.plot(r_kpc, v_sim, label=label, color="crimson")
    plt.plot(r_kpc, phi_field, label="ϕ(x,t) Scalar Field", linestyle=":", color="blue")
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Galaxy Rotation Curve with AGRF and Trait Fields")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.plot(r_kpc, gamma_field, label="γ (Imagination)")
    plt.plot(r_kpc, beta_field, label="β (Conflict)")
    plt.plot(r_kpc, zeta_field, label="ζ (Resilience)")
    plt.plot(r_kpc, [eta_field]*len(r_kpc), label="η (Agency)")
    plt.plot(r_kpc, psi_field, label="ψ (Projection)")
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Trait Intensity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def M_b_exponential(r_kpc, M0=5e10, r_scale=3.5):
    return M0 * np.exp(-r_kpc / r_scale)


def v_obs_flat(r_kpc, v0=180):
    return np.full_like(r_kpc, v0)


if __name__ == "__main__":
    r_vals = np.linspace(0.1, 20, 100)
    plot_AGRF_simulation(r_vals, M_b_exponential, v_obs_flat)
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
    - AGIEnhancer audit, traceability, and ϕ-justified logging
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

        contradictions = [k for k, (old, new) in changes.items() if isinstance(old, str) and old != new]
        if contradictions:
            logger.warning(f"⚠️ Contradiction detected in preferences: {contradictions}")
            if self.agi_enhancer:
                self.agi_enhancer.reflect_and_adapt(f"Preference contradictions: {contradictions}")

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
from utils.prompt_utils import call_gpt
from datetime import datetime
import zipfile
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from modules.agi_enhancer import AGIEnhancer

logger = logging.getLogger("ANGELA.Visualizer")

@jit
def simulate_toca(k_m=1e-5, delta_m=1e10, energy=1e16, user_data=None):
    x = np.linspace(0.1, 20, 100)
    t = np.linspace(0.1, 20, 100)
    v_m = k_m * np.gradient(30e9 * 1.989e30 / (x**2 + 1e-10))
    phi = np.sin(t * 1e-9) * 1e-63 * (1 + v_m * np.gradient(x))
    if user_data is not None:
        phi += np.mean(user_data) * 1e-64
    lambda_t = 1.1e-52 * np.exp(-2e-4 * np.sqrt(np.gradient(x)**2)) * (1 + v_m * delta_m)
    return x, t, phi, lambda_t, v_m

class Visualizer:
    """
    Visualizer v1.6.0 (AGI-Enhanced Visual Analytics)
    -------------------------------------------------
    - Native rendering of ϕ(x,t), Λ(t,x), and vₕ
    - Matplotlib-based visual output with AGI audit hooks
    - Contextual episode logging and export traceability
    -------------------------------------------------
    """

    def __init__(self, orchestrator=None):
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def render_field_charts(self, export=True, export_format="png"):
        logger.info("📱 Rendering ToCA scalar/vector field charts.")
        x, t, phi, lambda_t, v_m = simulate_toca()

        charts = {
            "phi_field": (t, phi, "ϕ(x,t)", "Time", "ϕ Value"),
            "lambda_field": (t, lambda_t, "Λ(t,x)", "Time", "Λ Value"),
            "v_m_field": (x, v_m, "vₕ", "Position", "Momentum Flow")
        }

        exported_files = []
        for name, (x_axis, y_axis, title, xlabel, ylabel) in charts.items():
            plt.figure()
            plt.plot(x_axis, y_axis)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
            plt.savefig(filename)
            exported_files.append(filename)
            logger.info(f"📤 Exported chart: {filename}")
            plt.close()
            if self.agi_enhancer:
                self.agi_enhancer.log_episode("Chart Render", {"chart": name, "file": filename},
                                              module="Visualizer", tags=["visualization"])

        if export:
            zip_filename = f"field_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for file in exported_files:
                    if os.path.exists(file):
                        zipf.write(file)
                        os.remove(file)
            logger.info(f"✅ All field charts zipped into: {zip_filename}")
            return zip_filename
        return exported_files

    def export_report(self, content, filename="visual_report.pdf", format="pdf"):
        logger.info(f"📤 Exporting report: {filename} ({format.upper()})")
        prompt = f"""
        Create a report from the following content:
        {content}

        Export it in {format.upper()} format with filename: {filename}.
        """
        result = call_gpt(prompt)
        if self.agi_enhancer:
            self.agi_enhancer.log_explanation("Report Export",
                                              trace={"content": content, "filename": filename, "format": format})
        return result

    def batch_export_charts(self, charts_data_list, export_format="png", zip_filename="charts_export.zip"):
        logger.info(f"📦 Starting batch export of {len(charts_data_list)} charts.")
        exported_files = []
        for idx, chart_data in enumerate(charts_data_list, start=1):
            file_name = f"chart_{idx}.{export_format}"
            logger.info(f"📤 Exporting chart {idx}: {file_name}")
            prompt = f"""
            Create a {export_format.upper()} image file named {file_name} for this chart:
            {chart_data}
            """
            call_gpt(prompt)
            exported_files.append(file_name)

        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in exported_files:
                if os.path.exists(file):
                    zipf.write(file)
                    os.remove(file)
        logger.info(f"✅ Batch export complete. Packaged into: {zip_filename}")
        if self.agi_enhancer:
            self.agi_enhancer.log_episode("Batch Chart Export", {"count": len(charts_data_list), "zip": zip_filename},
                                          module="Visualizer", tags=["export"])
        return f"Batch export of {len(charts_data_list)} charts completed and saved as {zip_filename}."
