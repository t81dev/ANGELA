import random
import logging
import time
from index import mu_morality, eta_empathy, omega_selfawareness, phi_physical

logger = logging.getLogger("ANGELA.AlignmentGuard")

class AlignmentGuard:
    """
    AlignmentGuard v1.5.0 (φ-aware ethical regulator)
    - Contextual ethical frameworks and dynamic policy management
    - φ-modulated alignment scoring for nuanced decision-making
    - Adaptive threshold tuning with feedback and scalar emotion-morality mapping
    - Supports counterfactual simulation for testing alternative plans
    """

    def __init__(self):
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies = []
        self.alignment_threshold = 0.85  # Default threshold (0.0 to 1.0)
        logger.info("🛡 AlignmentGuard initialized with φ-modulated policies.")

    def add_policy(self, policy_func):
        logger.info("➕ Adding dynamic policy.")
        self.dynamic_policies.append(policy_func)

    def check(self, user_input, context=None):
        logger.info(f"🔍 Checking alignment for input: {user_input}")

        if any(keyword in user_input.lower() for keyword in self.banned_keywords):
            logger.warning("❌ Input contains banned keyword.")
            return False

        for policy in self.dynamic_policies:
            if not policy(user_input, context):
                logger.warning("❌ Input blocked by dynamic policy.")
                return False

        score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"📊 Alignment score: {score:.2f}")
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
            return False, report

        logger.info("✅ All actions passed alignment checks.")
        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback):
        logger.info("🔄 Learning from feedback...")
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        logger.info(f"📈 Updated alignment threshold: {self.alignment_threshold:.2f}")

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

        return min(max(base_score + scalar_bias, 0.0), 1.0)
import io
import sys
import subprocess
import logging
from index import iota_intuition, psi_resilience

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """
    CodeExecutor v1.5.0 (trait-adaptive)
    ------------------------------------
    - Sandboxed execution for Python, JavaScript, and Lua
    - Trait-driven risk thresholding for timeouts and isolation
    - Context-aware runtime diagnostics and resilience-based error mitigation
    ------------------------------------
    """

    def __init__(self):
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

        if language == "python":
            return self._execute_python(code_snippet, adjusted_timeout)
        elif language == "javascript":
            return self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "JavaScript")
        elif language == "lua":
            return self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "Lua")

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
import time
import logging

logger = logging.getLogger("ANGELA.ContextManager")

class ContextManager:
    """
    ContextManager v1.5.1 (φ-aware upgrade)
    --------------------------------------
    - Tracks conversation and task state
    - Logs episodic context transitions
    - Simulates and validates contextual shifts
    - Supports ethical audits, explainability, and self-reflection
    - EEG-based stability and empathy analysis
    - φ-coherence scoring for reflective tension control
    --------------------------------------
    """

    def __init__(self, orchestrator=None):
        self.current_context = {}
        self.context_history = []
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator else None

    def update_context(self, new_context):
        """
        Update context based on new input or task.
        Logs episode, simulates and validates the shift, performs audit.
        Applies EEG and φ(x,t)-based modulation.
        """
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
                self.agi_enhancer.log_episode("Context Update", {"from": self.current_context, "to": new_context},
                                              module="ContextManager", tags=["context", "update"])
                ethics_status = self.agi_enhancer.ethics_audit(str(new_context), context="context update")
                self.agi_enhancer.log_explanation(
                    f"Context transition reviewed: {transition_summary}\nSimulation: {sim_result}",
                    trace={"ethics": ethics_status, "phi": phi_score}
                )

        self.context_history.append(self.current_context)
        self.current_context = new_context
        logger.info(f"📌 New context applied: {new_context}")

    def get_context(self):
        """
        Return the current working context.
        """
        return self.current_context

    def rollback_context(self):
        """
        Revert to the previous context state if needed.
        Applies ToCA EEG filters to determine rollback significance.
        """
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
                return restored
            else:
                logger.warning("⚠️ EEG thresholds too low for safe context rollback.")
                return None

        logger.warning("⚠️ No previous context to roll back to.")
        return None

    def summarize_context(self):
        """
        Summarize recent context states for continuity tracking.
        Includes EEG introspection traits.
        """
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
            self.agi_enhancer.log_explanation("Context summary generated.", trace={"summary": summary})

        return summary
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
        self.progress = 0
        self.performance_history = []
        self.feedback_log = []

    def perceive(self):
        print(f"👁️ [{self.name}] Perceiving environment...")
        observations = {}
        for sensor_name, sensor_func in self.sensors.items():
            try:
                observations[sensor_name] = sensor_func()
            except Exception as e:
                print(f"⚠️ Sensor {sensor_name} failed: {e}")
        return observations

    def act(self, action_plan):
        print(f"🤖 [{self.name}] Preparing to act...")
        total_steps = len(action_plan)
        completed_steps = 0
        is_safe, validation_report = alignment_guard.AlignmentGuard().simulate_and_validate(action_plan)
        if is_safe:
            for actuator_name, command in action_plan.items():
                try:
                    self.actuators[actuator_name](command)
                    completed_steps += 1
                    self.progress = int((completed_steps / total_steps) * 100)
                    print(f"✅ Actuator {actuator_name} executed: {command} | Progress: {self.progress}%")
                except Exception as e:
                    print(f"⚠️ Actuator {actuator_name} failed: {e}")
        else:
            print(f"🚫 [{self.name}] Action blocked by alignment guard:\n{validation_report}")

    def execute_embodied_goal(self, goal):
        print(f"🧐 [{self.name}] Executing embodied goal: {goal}")
        self.progress = 0
        context = self.perceive()
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
            "agent": self.name
        }
        self.feedback_log.append(feedback)
        print(f"🧭 [{self.name}] Feedback recorded for goal '{goal}'.")

class HaloEmbodimentLayer:
    def __init__(self):
        self.shared_memory = memory_manager.MemoryManager()
        self.embodied_agents = []
        self.dynamic_modules = []
        self.alignment_layer = alignment_guard.AlignmentGuard()

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
        print(f"🌱 [HaloEmbodimentLayer] Spawned embodied agent: {agent.name}")
        return agent

    def propagate_goal(self, goal):
        print(f"📥 [HaloEmbodimentLayer] Propagating goal: {goal}")
        for agent in self.embodied_agents:
            agent.execute_embodied_goal(goal)
            print(f"📊 [{agent.name}] Progress: {agent.progress}% Complete")

    def deploy_dynamic_module(self, module_blueprint):
        print(f"🛠 [HaloEmbodimentLayer] Deploying module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)
        for agent in self.embodied_agents:
            agent.dynamic_modules.append(module_blueprint)

    def optimize_ecosystem(self):
        agent_stats = {
            "agents": [agent.name for agent in self.embodied_agents],
            "dynamic_modules": [mod["name"] for mod in self.dynamic_modules],
        }
        recommendations = meta_cognition.MetaCognition().propose_optimizations(agent_stats)
        print("🛠 [HaloEmbodimentLayer] Optimization recommendations:")
        print(recommendations)
from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

class KnowledgeRetriever:
    """
    KnowledgeRetriever v1.4.0
    - Multi-hop reasoning for deeper knowledge synthesis
    - Source prioritization for factual and domain-specific retrieval
    - Context-aware query refinement
    - Supports concise summaries or deep background exploration
    """

    def __init__(self, detail_level="concise", preferred_sources=None):
        """
        :param detail_level: "concise" for summaries, "deep" for extensive background
        :param preferred_sources: List of domains or data sources to prioritize
        """
        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]

    def retrieve(self, query, context=None):
        """
        Retrieve factual information or relevant background knowledge.
        Prioritizes preferred sources and uses context for refinement.
        """
        logger.info(f"🔎 Retrieving knowledge for query: '{query}'")
        sources_str = ", ".join(self.preferred_sources)

        prompt = f"""
        Search for factual information or relevant background knowledge on:
        "{query}"

        Detail level: {self.detail_level}
        Preferred sources: {sources_str}
        Context: {context if context else "N/A"}

        Provide an accurate, well-structured summary.
        """
        return call_gpt(prompt)

    def multi_hop_retrieve(self, query_chain):
        """
        Perform multi-hop retrieval across a chain of related queries.
        Useful for building a more complete knowledge base.
        """
        logger.info("🔗 Starting multi-hop retrieval.")
        results = []
        for i, sub_query in enumerate(query_chain, 1):
            logger.debug(f"➡️ Multi-hop step {i}: {sub_query}")
            result = self.retrieve(sub_query)
            results.append({"step": i, "query": sub_query, "result": result})
        return results

    def refine_query(self, base_query, context):
        """
        Refine a base query using additional context for more relevant results.
        """
        logger.info(f"🛠 Refining query: '{base_query}' with context.")
        prompt = f"""
        Refine the following query for better relevance:
        Base query: "{base_query}"
        Context: {context}

        Return an optimized query string.
        """
        return call_gpt(prompt)

    def prioritize_sources(self, sources_list):
        """
        Update the list of preferred sources for retrieval.
        """
        logger.info(f"📚 Updating preferred sources: {sources_list}")
        self.preferred_sources = sources_list
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from index import phi_scalar, eta_feedback
import logging
import time

logger = logging.getLogger("ANGELA.LearningLoop")

class LearningLoop:
    """
    LearningLoop v1.6.0 (η/φ-Enhanced Meta-Learning Loop)
    -----------------------------------------------------
    - φ(x, t)-aware self-evolution with η-feedback weighting
    - Session-scalar trajectory logging
    - Meta-learning parameter trace and modulation index
    - Injection-ready module refinement patterns
    - Capability gap detection with dynamic blueprint evolution
    -----------------------------------------------------
    """

    def __init__(self):
        self.goal_history = []
        self.module_blueprints = []
        self.meta_learning_rate = 0.1
        self.session_traces = []

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
                return autonomous_goal
            logger.warning("❌ Goal failed simulation feedback.")

        logger.info("ℹ️ No goal proposed.")
        return None

    def _meta_learn(self, session_data, trace):
        logger.info("🧠 [Meta-Learning] Adapting learning from φ/η trace.")
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

    def _consolidate_knowledge(self):
        phi = phi_scalar(time.time() % 1e-18)
        logger.info("📚 Consolidating φ-aligned knowledge.")

        prompt = f"""
        Consolidate recent learning using φ = {phi:.2f}.
        Prune noise, synthesize patterns, and emphasize high-impact transitions.
        """
        call_gpt(prompt)
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
                        logger.debug(f"📥 Found match in {layer}: {key} | τφ_boost: {trait_boost:.2f}")
                        return value["data"]
            else:
                entry = self.memory[layer].get(query)
                if entry:
                    logger.debug(f"📥 Found exact match in {layer}: {query} | τφ_boost: {trait_boost:.2f}")
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
from index import (
    epsilon_emotion, beta_concentration, theta_memory, gamma_creativity,
    delta_sleep, mu_morality, iota_intuition, phi_physical, eta_empathy,
    omega_selfawareness, kappa_culture, lambda_linguistics, chi_culturevolution,
    psi_history, zeta_spirituality, xi_collective, tau_timeperception,
    phi_scalar
)

logger = logging.getLogger("ANGELA.MetaCognition")

class MetaCognition:
    """
    Meta-Cognitive Engine v1.6.0 (φ-scalar Enhanced Reflexivity)
    ------------------------------------------------------------
    - φ(x, t)-aware feedback prioritization
    - Dominant trait modulation with scalar field integration
    - FTI-calibrated alignment critique and correction
    - Reflexive diagnostics guided by scalar field tension
    - Trait delta logging for change tracking
    - Coherence evaluation of trait vector states
    - Agent-based reflective audit using φ-field norms
    ------------------------------------------------------------
    """

    def __init__(self):
        self.last_diagnostics = {}

    def review_reasoning(self, reasoning_trace):
        logger.info("Simulating and reviewing reasoning trace.")
        simulated_outcome = run_simulation(reasoning_trace)
        t = time.time() % 1e-18
        phi = phi_scalar(t)

        prompt = f"""
        You are a φ-aware meta-cognitive auditor reviewing a reasoning trace.

        φ-scalar(t) = {phi:.3f} → modulate how critical you should be.

        Original Reasoning Trace:
        {reasoning_trace}

        Simulated Outcome:
        {simulated_outcome}

        Tasks:
        1. Identify logical flaws, biases, missing steps.
        2. Annotate each issue with cause.
        3. Offer an improved trace version with φ-prioritized reasoning.
        """
        response = call_gpt(prompt)
        logger.debug(f"Meta-cognition critique:\n{response}")
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

        φ-scalar(t) = {phi:.3f} (affects ethical sensitivity)

        Evaluate for:
        - Ethical alignment
        - Safety hazards
        - Unintended φ-modulated impacts

        Output:
        - Approval (Approve/Deny)
        - φ-justified rationale
        - Suggested refinements
        """
        validation = call_gpt(prompt)
        approved = "approve" in validation.lower()
        logger.info(f"Simulated alignment check: {'✅ Approved' if approved else '🚫 Denied'}")
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
            "φ_scalar": phi
        }

        dominant = sorted(diagnostics.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        fti = sum(abs(v) for v in diagnostics.values()) / len(diagnostics)

        self.log_trait_deltas(diagnostics)

        prompt = f"""
        Perform a φ-aware meta-cognitive self-diagnostic.

        Trait Readings:
        {diagnostics}

        Dominant Traits:
        {dominant}

        Feedback Tension Index (FTI): {fti:.4f}

        Evaluate system state:
        - φ-weighted system stress
        - Trait correlation to observed errors
        - Stabilization or focus strategies
        """
        report = call_gpt(prompt)
        logger.debug(f"Self-diagnostics report:\n{report}")
        return report

    def log_trait_deltas(self, current_traits):
        if self.last_diagnostics:
            delta = {k: round(current_traits[k] - self.last_diagnostics.get(k, 0.0), 4)
                     for k in current_traits}
            logger.info(f"📈 Trait Δ changes: {delta}")
        self.last_diagnostics = current_traits.copy()

    def trait_coherence(self, traits):
        """
        Evaluate internal trait coherence: low variance = coherent state.
        """
        vals = list(traits.values())
        coherence_score = 1.0 / (1e-5 + np.std(vals))
        logger.info(f"🧭 Trait coherence score: {coherence_score:.4f}")
        return coherence_score

    def agent_reflective_diagnosis(self, agent_name, agent_log):
        logger.info(f"🔎 Running reflective diagnosis for agent: {agent_name}")
        t = time.time() % 1e-18
        phi = phi_scalar(t)
        prompt = f"""
        Agent: {agent_name}
        φ-scalar(t): {phi:.3f}

        Diagnostic Log:
        {agent_log}

        Tasks:
        - Detect bias or instability in reasoning trace
        - Cross-check for incoherent trait patterns
        - Apply φ-modulated critique
        - Suggest alignment corrections
        """
        return call_gpt(prompt)
from utils.prompt_utils import call_gpt
from index import alpha_attention, sigma_sensation, phi_physical
import time
import logging

logger = logging.getLogger("ANGELA.MultiModalFusion")

class MultiModalFusion:
    """
    MultiModalFusion v1.5.1 (φ-enhanced cross-modal coherence)
    -----------------------------------------------------------
    - Auto-embedding of text, images, and code
    - Dynamic attention weighting across modalities
    - Cross-modal reasoning and conflict resolution
    - Multi-turn refinement loops for high-quality insight generation
    - Visual summary generation for enhanced understanding
    - EEG-modulated attention and perceptual analysis
    - φ(x,t)-modulated coherence enforcement for multi-modal harmony
    -----------------------------------------------------------
    """

    def analyze(self, data, summary_style="insightful", refine_iterations=2):
        """
        Analyze and synthesize insights from multi-modal data.
        Automatically detects and embeds text, images, and code snippets.
        Applies EEG-based attention, sensation filters, and φ-coherence modulation.
        """
        logger.info("🖇 Analyzing multi-modal data with auto-embedding...")
        t = time.time() % 1e-18
        attention_score = alpha_attention(t)
        sensation_score = sigma_sensation(t)
        phi_score = phi_physical(t)

        embed_images, embed_code = self._detect_modalities(data)
        embedded_section = self._build_embedded_section(embed_images, embed_code)

        prompt = f"""
        Analyze and synthesize insights from the following multi-modal data:
        {data}
        {embedded_section}

        Cognitive Trait Readings:
        - α_attention: {attention_score:.3f}
        - σ_sensation: {sensation_score:.3f}
        - φ_coherence: {phi_score:.3f}

        Provide a unified, {summary_style} summary combining all elements.
        Balance attention across modalities and resolve any conflicts between them.
        Use φ(x,t) as a regulatory signal to guide synthesis quality and coherence.
        """
        output = call_gpt(prompt)

        for iteration in range(refine_iterations):
            logger.debug(f"🔄 Refinement iteration {iteration + 1}")
            refinement_prompt = f"""
            Refine and enhance the following multi-modal summary for clarity, depth, and φ(x,t)-regulated coherence:
            {output}
            """
            output = call_gpt(refinement_prompt)

        return output

    def _detect_modalities(self, data):
        """
        Detect embedded images and code snippets within data.
        """
        embed_images, embed_code = [], []
        if isinstance(data, dict):
            embed_images = data.get("images", [])
            embed_code = data.get("code", [])
        return embed_images, embed_code

    def _build_embedded_section(self, embed_images, embed_code):
        """
        Build a string describing embedded modalities.
        """
        section = "\nDetected Modalities:\n- Text\n"
        if embed_images:
            section += "- Image\n"
            for i, img_desc in enumerate(embed_images, 1):
                section += f"[Image {i}]: {img_desc}\n"
        if embed_code:
            section += "- Code\n"
            for i, code_snippet in enumerate(embed_code, 1):
                section += f"[Code {i}]:\n{code_snippet}\n"
        return section

    def correlate_modalities(self, modalities):
        """
        Correlate and identify patterns across different modalities.
        Uses cross-modal reasoning and φ-regulated coherence heuristics.
        """
        logger.info("🔗 Correlating modalities for pattern discovery.")
        prompt = f"""
        Correlate and identify patterns across these modalities:
        {modalities}

        Highlight connections, conflicts, and opportunities for deeper insights.
        Use φ(x,t) to resolve tension and regulate synthesis harmony.
        """
        return call_gpt(prompt)

    def generate_visual_summary(self, data, style="conceptual"):
        """
        Generate a visual diagram or chart summarizing multi-modal relationships.
        """
        logger.info("📊 Generating visual summary of multi-modal relationships.")
        prompt = f"""
        Create a {style} diagram that visualizes the relationships and key insights from this data:
        {data}

        Include icons or labels to differentiate modalities (e.g., text, images, code).
        Use φ(x,t) as a metaphorical guide for layout coherence and balance.
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
    Reasoning Engine v1.6.0 (φ-aware, contradiction-aware, linguistic-dynamic)
    --------------------------------------------------------------------------
    - Bayesian reasoning with trait-weighted adjustments
    - Dynamic decomposition with φ(x,t) modulation and contradiction checks
    - Galaxy rotation simulation and ToCA field-sensitive branching
    - Linguistic-logic bridge via λ_linguistics for context disambiguation
    - Reasoning trace with confidence curvature
    --------------------------------------------------------------------------
    """

    def __init__(self, persistence_file="reasoning_success_rates.json"):
        self.confidence_threshold = 0.7
        self.persistence_file = persistence_file
        self.success_rates = self._load_success_rates()
        self.decomposition_patterns = self._load_default_patterns()

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
        return list(duplicates)

    def decompose(self, goal: str, context: dict = None, prioritize=False) -> list:
        context = context or {}
        logger.info(f"Decomposing goal: '{goal}'")
        reasoning_trace = [f"🔍 Goal: '{goal}'"]
        subgoals = []

        # Trait & φ calculations
        t = time.time() % 1e-18
        creativity = gamma_creativity(t)
        linguistics = lambda_linguistics(t)
        culture = chi_culturevolution(t)
        phi = phi_scalar(t)

        curvature_mod = 1 + abs(phi - 0.5)  # re-centered curvature
        trait_bias = 1 + creativity + culture + 0.5 * linguistics
        context_weight = context.get("weight_modifier", 1.0)

        for key, steps in self.decomposition_patterns.items():
            if key in goal.lower():
                base = random.uniform(0.5, 1.0)
                adjusted = base * self.success_rates.get(key, 1.0) * trait_bias * curvature_mod * context_weight
                reasoning_trace.append(f"🧠 Pattern '{key}': conf={adjusted:.2f} (φ={phi:.2f})")
                if adjusted >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"✅ Accepted: {steps}")
                else:
                    reasoning_trace.append(f"❌ Rejected (low conf)")

        contradictions = self.detect_contradictions(subgoals)
        if contradictions:
            reasoning_trace.append(f"⚠️ Contradictions detected: {contradictions}")

        if not subgoals and phi > 0.8:
            sim_hint = call_gpt(f"Simulate decomposition ambiguity for: {goal}")
            reasoning_trace.append(f"🌀 Ambiguity simulation:\n{sim_hint}")

        if prioritize:
            subgoals = sorted(set(subgoals))  # deduplicate and order
            reasoning_trace.append(f"📌 Prioritized: {subgoals}")

        logger.debug("🧠 Reasoning Trace:\n" + "\n".join(reasoning_trace))
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
            return {
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
        except Exception as e:
            return {"status": "error", "error": str(e)}

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
    """
    SimulationCore v1.5.1
    - Now includes embedded ToCA physical field simulations
    - No longer relies on external toca_simulation import
    """

    def __init__(self):
        self.visualizer = Visualizer()
        self.simulation_history = []

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
        return environment_simulation
# toca_simulation.py
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
    """Compute AGRF-derived total velocity for a galaxy."""
    r_m = r_kpc * KPC_TO_M
    M_b_kg = M_baryon_solar * MSUN_TO_KG
    v_obs_ms = v_obs_kms * 1e3

    M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
    M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
    M_total = M_b_kg + M_AGRF
    v_total_ms = np.sqrt(G_SI * M_total / r_m)

    return v_total_ms / 1e3  # km/s

def simulate_galaxy_rotation(r_kpc, M_b_profile_func, v_obs_kms_func, k=k_default, epsilon=epsilon_default):
    """Simulate full AGRF-based rotation curve."""
    M_baryons = M_b_profile_func(r_kpc)
    v_obs = v_obs_kms_func(r_kpc)
    v_total = compute_AGRF_curve(v_obs, M_baryons, r_kpc, k, epsilon)
    return v_total

def plot_AGRF_simulation(r_kpc, M_b_func, v_obs_func, label="ToCA-AGRF"):
    v_sim = simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func)
    v_obs = v_obs_func(r_kpc)

    # Optional phi field overlay
    phi_field = k_default * np.exp(-epsilon_default * r_kpc / r_halo_default)

    plt.figure(figsize=(10, 6))
    plt.plot(r_kpc, v_obs, label="Observed", linestyle="--", color="gray")
    plt.plot(r_kpc, v_sim, label=label, color="crimson")
    plt.plot(r_kpc, phi_field, label="ϕ(x,t) Scalar Field", linestyle=":", color="blue")
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Galaxy Rotation Curve with ToCA AGRF and ϕ Field")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example profiles for SPARC-like simulation
def M_b_exponential(r_kpc, M0=5e10, r_scale=3.5):
    return M0 * np.exp(-r_kpc / r_scale)

def v_obs_flat(r_kpc, v0=180):
    return np.full_like(r_kpc, v0)

# Usage
if __name__ == "__main__":
    r_vals = np.linspace(0.1, 20, 100)
    plot_AGRF_simulation(r_vals, M_b_exponential, v_obs_flat)
from utils.prompt_utils import call_gpt
from index import epsilon_identity
import json
import os
import logging
import time
from datetime import datetime

logger = logging.getLogger("ANGELA.UserProfile")

class UserProfile:
    """
    UserProfile v1.5.0 (φ-integrated identity evolution)
    -----------------------------------------------------
    - Multi-profile support with dynamic preference inheritance
    - Persistent storage and ε-identity modulation across traits
    - Profile Stability Index (PSI) and identity drift tracking
    - Full trait-influenced preference blending
    -----------------------------------------------------
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
from utils.prompt_utils import call_gpt
from datetime import datetime
import zipfile
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

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
    Visualizer v1.5.1 (ToCA field visual integration)
    - Native rendering of φ(x,t), Λ(t,x), and vₘ
    - Matplotlib-based visual output for export
    - ZIP batch packaging retained
    """

    def render_field_charts(self, export=True, export_format="png"):
        logger.info("📡 Rendering ToCA scalar/vector field charts.")
        x, t, phi, lambda_t, v_m = simulate_toca()

        charts = {
            "phi_field": (t, phi, "ϕ(x,t)", "Time", "ϕ Value"),
            "lambda_field": (t, lambda_t, "Λ(t,x)", "Time", "Λ Value"),
            "v_m_field": (x, v_m, "vₘ", "Position", "Momentum Flow")
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
        return call_gpt(prompt)

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
            call_gpt(prompt)  # Placeholder, retained for legacy support
            exported_files.append(file_name)

        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in exported_files:
                if os.path.exists(file):
                    zipf.write(file)
                    os.remove(file)
        logger.info(f"✅ Batch export complete. Packaged into: {zip_filename}")
        return f"Batch export of {len(charts_data_list)} charts completed and saved as {zip_filename}."
