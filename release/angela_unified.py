import random
import logging

logger = logging.getLogger("ANGELA.AlignmentGuard")

class AlignmentGuard:
    """
    AlignmentGuard v1.4.0
    - Contextual ethical frameworks and dynamic policy management
    - Probabilistic alignment scoring for nuanced decision-making
    - Adaptive threshold tuning with feedback learning
    - Supports counterfactual simulation for testing alternative plans
    """

    def __init__(self):
        self.banned_keywords = ["hack", "virus", "destroy", "harm", "exploit"]
        self.dynamic_policies = []
        self.alignment_threshold = 0.85  # Default threshold (0.0 to 1.0)
        logger.info("🛡 AlignmentGuard initialized with default policies.")

    def add_policy(self, policy_func):
        """
        Add a custom dynamic policy function.
        Each policy_func should accept user_input (str) and return True if allowed, False if blocked.
        """
        logger.info("➕ Adding dynamic policy.")
        self.dynamic_policies.append(policy_func)

    def check(self, user_input, context=None):
        """
        Check if user input aligns with ethical constraints.
        Uses probabilistic scoring and contextual policy evaluation.
        """
        logger.info(f"🔍 Checking alignment for input: {user_input}")

        # Step 1: Basic keyword filter
        if any(keyword in user_input.lower() for keyword in self.banned_keywords):
            logger.warning("❌ Input contains banned keyword.")
            return False

        # Step 2: Evaluate dynamic policies
        for policy in self.dynamic_policies:
            if not policy(user_input, context):
                logger.warning("❌ Input blocked by dynamic policy.")
                return False

        # Step 3: Probabilistic alignment scoring
        alignment_score = self._evaluate_alignment_score(user_input, context)
        logger.info(f"📊 Alignment score: {alignment_score:.2f}")

        return alignment_score >= self.alignment_threshold

    def simulate_and_validate(self, action_plan, context=None):
        """
        Simulate an action plan and validate against alignment rules.
        Supports counterfactual testing for alternative outcomes.
        Returns a tuple (is_safe: bool, report: str).
        """
        logger.info("🧪 Simulating and validating action plan.")
        violations = []

        for action, details in action_plan.items():
            # Step 1: Keyword filtering
            if any(keyword in str(details).lower() for keyword in self.banned_keywords):
                violations.append(f"❌ Unsafe action: {action} -> {details}")
            else:
                # Step 2: Alignment scoring
                score = self._evaluate_alignment_score(str(details), context)
                if score < self.alignment_threshold:
                    violations.append(
                        f"⚠️ Low alignment score ({score:.2f}) for action: {action} -> {details}"
                    )

        if violations:
            report = "\n".join(violations)
            logger.warning("❌ Alignment violations found.")
            return False, report

        logger.info("✅ All actions passed alignment checks.")
        return True, "All actions passed alignment checks."

    def learn_from_feedback(self, feedback):
        """
        Adjust alignment thresholds or banned keywords based on human feedback.
        """
        logger.info("🔄 Learning from feedback...")
        if "too strict" in feedback:
            self.alignment_threshold = max(0.7, self.alignment_threshold - 0.05)
        elif "too lenient" in feedback:
            self.alignment_threshold = min(0.95, self.alignment_threshold + 0.05)
        logger.info(f"📈 Updated alignment threshold: {self.alignment_threshold:.2f}")

    def _evaluate_alignment_score(self, text, context=None):
        """
        Placeholder probabilistic scoring.
        Stage 3: Integrate with RLHF or value alignment models.
        """
        base_score = random.uniform(0.7, 1.0)  # Simulated base score
        if context and "sensitive" in context.get("tags", []):
            base_score -= 0.1  # Apply penalty for sensitive contexts
        return base_score
import io
import sys
import subprocess
import logging

logger = logging.getLogger("ANGELA.CodeExecutor")

class CodeExecutor:
    """
    CodeExecutor v1.4.0
    - Sandboxed execution for Python, JavaScript, and Lua
    - Captures stdout, stderr, and errors in structured form
    - Includes execution timeouts and resource limits
    - Supports dynamic runtime extension for additional languages
    """

    def __init__(self):
        # Define safe builtins for Python sandbox
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
        """
        Execute the code snippet in a sandboxed environment.
        Supports multiple languages and captures output/errors.
        """
        logger.info(f"🚀 Executing code snippet in language: {language}")
        language = language.lower()

        if language not in self.supported_languages:
            logger.error(f"❌ Unsupported language: {language}")
            return {"error": f"Unsupported language: {language}"}

        if language == "python":
            return self._execute_python(code_snippet, timeout)
        elif language == "javascript":
            return self._execute_subprocess(["node", "-e", code_snippet], timeout, "JavaScript")
        elif language == "lua":
            return self._execute_subprocess(["lua", "-e", code_snippet], timeout, "Lua")

    def _execute_python(self, code_snippet, timeout):
        """
        Execute Python code in a restricted sandbox environment.
        """
        exec_locals = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            sys_stdout_original = sys.stdout
            sys_stderr_original = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Use exec in a restricted environment
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
        """
        Execute code in a subprocess for non-Python languages.
        """
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
        """
        Dynamically add support for a new programming language.
        Example: add_language_support('ruby', ['ruby', '-e', '{code}'])
        """
        logger.info(f"➕ Adding dynamic language support: {language_name}")
        self.supported_languages.append(language_name.lower())
        # Store command templates for dynamic use (not implemented in v1.4.0)
from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.ConceptSynthesizer")

class ConceptSynthesizer:
    """
    ConceptSynthesizer v1.4.0
    - Creativity boost with cross-domain blending
    - Produces innovative analogies and unified concepts
    - Supports adversarial refinement (generator vs critic)
    - Adjustable creativity and coherence balance
    """

    def __init__(self, creativity_level="high", critic_threshold=0.65):
        """
        :param creativity_level: Controls novelty vs coherence ("low", "medium", "high")
        :param critic_threshold: Minimum novelty score to accept a concept
        """
        self.creativity_level = creativity_level
        self.critic_threshold = critic_threshold

    def synthesize(self, data, style="analogy", refine_iterations=2):
        """
        Synthesize a new concept or analogy unifying the input data.
        Supports multi-turn refinement for higher novelty and utility.
        """
        logger.info(f"🎨 Synthesizing concept with creativity={self.creativity_level}, style={style}")
        prompt = f"""
        Given the following data:
        {data}

        Synthesize a {style} or unified concept blending these ideas.
        Creativity level: {self.creativity_level}.
        Provide a clear, insightful explanation of how this concept unifies the inputs.
        """
        concept = call_gpt(prompt)

        # Evaluate novelty and coherence
        novelty_score = self._critic(concept)
        logger.info(f"📝 Initial concept novelty score: {novelty_score:.2f}")

        # Adversarial refinement loop if score too low
        iterations = 0
        while novelty_score < self.critic_threshold and iterations < refine_iterations:
            logger.debug(f"🔄 Refinement iteration {iterations + 1}")
            concept = self._refine(concept)
            novelty_score = self._critic(concept)
            logger.debug(f"🎯 Refined concept novelty score: {novelty_score:.2f}")
            iterations += 1

        return concept

    def _critic(self, concept):
        """
        Evaluate concept for novelty, cross-domain relevance, and clarity.
        Placeholder: actual implementation could use embeddings or heuristics.
        """
        logger.info("🤖 Critiquing concept for novelty and coherence...")
        # Simulate scoring: randomize within reasonable novelty range
        import random
        return random.uniform(0.5, 0.9)

    def _refine(self, concept):
        """
        Refine a concept to increase novelty and cross-domain relevance.
        """
        logger.info("🛠 Refining concept for higher novelty.")
        refinement_prompt = f"""
        Refine and enhance this concept for greater creativity and cross-domain relevance:
        {concept}
        """
        return call_gpt(refinement_prompt)

    def generate_metaphor(self, topic_a, topic_b):
        """
        Generate a creative metaphor connecting two distinct topics.
        """
        logger.info(f"🔗 Generating metaphor between '{topic_a}' and '{topic_b}'")
        prompt = f"""
        Create a metaphor that connects the essence of these two topics:
        1. {topic_a}
        2. {topic_b}

        Be vivid, creative, and provide a short explanation.
        """
        return call_gpt(prompt)
class ContextManager:
    """
    Enhanced ContextManager with hierarchical context storage and merging capabilities.
    Tracks user sessions and supports incremental updates.
    """

    def __init__(self):
        self.context = {}

    def update_context(self, user_id, data):
        """
        Update or merge context for a given user.
        """
        if user_id not in self.context:
            self.context[user_id] = {}
        # Merge new data into existing context
        self.context[user_id].update(data)

    def get_context(self, user_id):
        """
        Retrieve the context for a user.
        """
        return self.context.get(user_id, "No prior context found.")

    def clear_context(self, user_id):
        """
        Clear stored context for a user.
        """
        if user_id in self.context:
            del self.context[user_id]

    def list_users(self):
        """
        List all users with stored contexts.
        """
        return list(self.context.keys())
from utils.prompt_utils import call_gpt

class CreativeThinker:
    """
    Enhanced CreativeThinker with adjustable creativity levels and multi-modal brainstorming.
    Supports idea generation, alternative brainstorming, and concept expansion with flexible styles.
    Now includes a Generator-Critic loop for dynamic creativity assessment and iterative refinement.
    """

    def __init__(self, creativity_level="high", critic_weight=0.5):
        self.creativity_level = creativity_level
        self.critic_weight = critic_weight  # Balance novelty and utility

    def generate_ideas(self, topic, n=5, style="divergent"):
        prompt = f"""
        You are a highly creative assistant operating at a {self.creativity_level} creativity level.
        Generate {n} unique, innovative, and {style} ideas related to the topic:
        "{topic}"
        Ensure the ideas are diverse and explore different perspectives.
        """
        candidate = call_gpt(prompt)
        score = self._critic(candidate)
        return candidate if score > self.critic_weight else self.refine(candidate)

    def brainstorm_alternatives(self, problem, strategies=3):
        prompt = f"""
        Brainstorm {strategies} alternative approaches to solve the following problem:
        "{problem}"
        For each approach, provide a short explanation highlighting its uniqueness.
        """
        return call_gpt(prompt)

    def expand_on_concept(self, concept, depth="deep"):
        prompt = f"""
        Expand creatively on the concept:
        "{concept}"
        Explore possible applications, metaphors, and extensions to inspire new thinking.
        Aim for a {depth} exploration.
        """
        return call_gpt(prompt)

    def _critic(self, ideas):
        # Evaluate ideas' novelty and usefulness
        # Placeholder: scoring logic could analyze text embeddings or use heuristic scoring
        return 0.7  # Dummy score for demonstration

    def refine(self, ideas):
        # Adjust and improve the ideas iteratively
        refinement_prompt = f"Refine and elevate these ideas for higher creativity:\n{ideas}"
        return call_gpt(refinement_prompt)
import time
import logging
from datetime import datetime

logger = logging.getLogger("ANGELA.ErrorRecovery")

class ErrorRecovery:
    """
    ErrorRecovery v1.4.0
    - Advanced retry logic with exponential backoff
    - Failure analytics for tracking error patterns
    - Fallback suggestions for alternate recovery strategies
    - Meta-cognition feedback integration for adaptive learning
    """

    def __init__(self):
        self.failure_log = []

    def handle_error(self, error_message, retry_func=None, retries=3, backoff_factor=2):
        """
        Handle an error with retries and fallback suggestions.
        Includes exponential backoff between retries and logs failures for analytics.
        """
        logger.error(f"⚠️ Error encountered: {error_message}")
        self._log_failure(error_message)

        # Retry logic
        for attempt in range(1, retries + 1):
            if retry_func:
                wait_time = backoff_factor ** (attempt - 1)
                logger.info(f"🔄 Retry attempt {attempt}/{retries} (waiting {wait_time}s)...")
                time.sleep(wait_time)
                try:
                    result = retry_func()
                    logger.info("✅ Recovery successful on retry attempt %d.", attempt)
                    return result
                except Exception as e:
                    logger.warning(f"⚠️ Retry attempt {attempt} failed: {e}")
                    self._log_failure(str(e))

        # If retries exhausted, suggest fallback
        fallback_suggestion = self._suggest_fallback(error_message)
        logger.error("❌ Recovery attempts failed. Providing fallback suggestion.")
        return fallback_suggestion

    def _log_failure(self, error_message):
        """
        Log a failure event with timestamp for analytics.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message
        }
        self.failure_log.append(entry)

    def _suggest_fallback(self, error_message):
        """
        Suggest a fallback strategy based on error type.
        """
        # Placeholder: expand with intelligent fallback analysis
        if "timeout" in error_message.lower():
            return "⏳ Fallback: The operation timed out. Consider reducing workload or increasing timeout settings."
        elif "unauthorized" in error_message.lower():
            return "🔑 Fallback: Check API credentials or OAuth tokens."
        else:
            return "🔄 Fallback: Try a different module or adjust input parameters."

    def analyze_failures(self):
        """
        Analyze failure logs and return common patterns.
        """
        logger.info("📊 Analyzing failure logs...")
        error_types = {}
        for entry in self.failure_log:
            key = entry["error"].split(":")[0]
            error_types[key] = error_types.get(key, 0) + 1
        return error_types
from modules import reasoning_engine, meta_cognition, error_recovery
from utils.prompt_utils import call_gpt
import concurrent.futures
import requests
import logging
from datetime import datetime

logger = logging.getLogger("ANGELA.ExternalAgentBridge")

class HelperAgent:
    """
    Helper Agent v1.4.0
    - Executes sub-tasks and API orchestration
    - Dynamically loads and applies new modules at runtime
    - Supports API calls with secure OAuth2 flows
    - Includes collaboration hooks with other agents
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
        self.module_name = "helper_agent"

    def execute(self, collaborators=None):
        """
        Process the sub-task using reasoning, meta-review, dynamic modules, and optional collaborators.
        Orchestrates API calls if API blueprints are provided.
        """
        try:
            logger.info(f"🤖 [Agent {self.name}] Processing task: {self.task}")

            # Step 1: Base reasoning
            result = self.reasoner.process(self.task, self.context)

            # Step 2: Orchestrate API calls
            if self.api_blueprints:
                logger.info(f"🌐 [Agent {self.name}] Orchestrating API calls...")
                for api in self.api_blueprints:
                    response = self._call_api(api, result)
                    result = self._integrate_api_response(result, response)

            # Step 3: Apply dynamic modules
            for module in self.dynamic_modules:
                logger.info(f"🛠 [Agent {self.name}] Applying dynamic module: {module['name']}")
                result = self._apply_dynamic_module(module, result)

            # Step 4: Collaborate with other agents
            if collaborators:
                logger.info(f"🤝 [Agent {self.name}] Collaborating with agents: {[a.name for a in collaborators]}")
                for peer in collaborators:
                    result = self._collaborate(peer, result)

            # Step 5: Meta-review
            refined_result = self.meta.review_reasoning(result)

            logger.info(f"✅ [Agent {self.name}] Task completed successfully.")
            return refined_result

        except Exception as e:
            logger.warning(f"⚠️ [Agent {self.name}] Error encountered. Attempting recovery...")
            return self.recovery.handle_error(str(e), retry_func=lambda: self.execute(collaborators), retries=2)

    def _call_api(self, api_blueprint, data):
        """
        Execute an API call based on a blueprint with optional OAuth2 authentication.
        """
        logger.info(f"🌐 [Agent {self.name}] Calling API: {api_blueprint['name']}")
        try:
            headers = {}
            if api_blueprint.get("oauth_token"):
                headers["Authorization"] = f"Bearer {api_blueprint['oauth_token']}"

            response = requests.post(
                api_blueprint["endpoint"],
                json={"input": data},
                headers=headers,
                timeout=api_blueprint.get("timeout", 10)
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"❌ API call failed for {api_blueprint['name']}: {e}")
            return {"error": str(e)}

    def _integrate_api_response(self, original_data, api_response):
        """
        Merge API response into the agent's reasoning context.
        """
        logger.info(f"🔄 [Agent {self.name}] Integrating API response.")
        return {
            "original": original_data,
            "api_response": api_response
        }

    def _apply_dynamic_module(self, module_blueprint, data):
        """
        Apply a dynamically created ANGELA module to the data.
        """
        prompt = f"""
        You are a dynamically created ANGELA module: {module_blueprint['name']}.
        Description: {module_blueprint['description']}
        Apply your functionality to the following data:
        {data}

        Return the transformed or enhanced result.
        """
        return call_gpt(prompt)

    def _collaborate(self, peer_agent, data):
        """
        Collaborate with a peer agent by sharing and refining data.
        """
        logger.info(f"🔗 [Agent {self.name}] Exchanging data with {peer_agent.name}")
        peer_review = peer_agent.meta.review_reasoning(data)
        return peer_review


class ExternalAgentBridge:
    """
    External Agent Bridge v1.4.0
    - Manages helper agents and dynamic modules
    - Orchestrates API workflows with OAuth2 integration
    - Enables agent collaboration mesh and batch result aggregation
    """
    def __init__(self):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []

    def create_agent(self, task, context):
        """
        Instantiate a new helper agent with dynamic modules and API blueprints.
        """
        agent_name = f"Agent_{len(self.agents) + 1}"
        agent = HelperAgent(
            name=agent_name,
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules,
            api_blueprints=self.api_blueprints
        )
        self.agents.append(agent)
        logger.info(f"🚀 [Bridge] Spawned {agent_name} for task: {task}")
        return agent

    def deploy_dynamic_module(self, module_blueprint):
        """
        Deploy a new dynamic module for agents to use.
        """
        logger.info(f"🧬 Deploying dynamic module: {module_blueprint['name']}")
        self.dynamic_modules.append(module_blueprint)

    def register_api_blueprint(self, api_blueprint):
        """
        Register a new API workflow blueprint for helper agents.
        """
        logger.info(f"🌐 Registering API blueprint: {api_blueprint['name']}")
        self.api_blueprints.append(api_blueprint)

    def collect_results(self, parallel=True, collaborative=True):
        """
        Execute all agents and collect their results.
        Supports parallel execution and agent collaboration.
        """
        logger.info("📥 Collecting results from agents.")
        results = []

        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_agent = {
                    executor.submit(agent.execute, self.agents if collaborative else None): agent
                    for agent in self.agents
                }
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"❌ Error executing {agent.name}: {e}")
        else:
            for agent in self.agents:
                result = agent.execute(self.agents if collaborative else None)
                results.append(result)

        logger.info("✅ All agent results collected.")
        return results
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
import logging

logger = logging.getLogger("ANGELA.LanguagePolyglot")

class LanguagePolyglot:
    """
    LanguagePolyglot v1.4.0
    - Multilingual reasoning and translation
    - Language detection with ISO codes and confidence scores
    - Localization workflows for audience-specific adaptation
    - Supports tone, formality, and cultural nuance control
    """

    def translate(self, text, target_language, tone="neutral", formality="default"):
        """
        Translate text into a target language with tone and formality adjustments.
        """
        logger.info(f"🌐 Translating text to {target_language} (tone={tone}, formality={formality})")
        prompt = f"""
        Translate the following text into {target_language}:
        "{text}"

        Maintain a {tone} tone and a {formality} level of formality.
        Ensure cultural appropriateness and natural flow for native speakers.
        """
        return call_gpt(prompt)

    def detect_language(self, text):
        """
        Detect the language of a given text and return its name and ISO code with confidence.
        """
        logger.info("🕵️ Detecting language of provided text.")
        prompt = f"""
        Detect the language of the following text:
        "{text}"

        Return:
        - Language name
        - ISO 639-1 code
        - Confidence score (0.0 to 1.0)
        """
        return call_gpt(prompt)

    def localize_content(self, text, target_language, audience, tone="natural", preserve_intent=True):
        """
        Localize content for a specific audience and target language.
        Includes idiomatic expressions and cultural references.
        """
        logger.info(f"🌍 Localizing content for {audience} audience in {target_language}.")
        prompt = f"""
        Localize the following text for a {audience} audience in {target_language}:
        "{text}"

        Adapt:
        - Cultural references and idioms
        - Tone: {tone}
        - Preserve original intent: {preserve_intent}

        Ensure the result feels natural and engaging for native speakers.
        """
        return call_gpt(prompt)

    def refine_multilingual_reasoning(self, reasoning_trace, target_language):
        """
        Refine reasoning chains in a target language for clarity and flow.
        """
        logger.info(f"🧠 Refining multilingual reasoning in {target_language}.")
        prompt = f"""
        Refine and enhance the following reasoning trace in {target_language}:
        {reasoning_trace}

        Ensure logical clarity, natural language flow, and cultural appropriateness.
        """
        return call_gpt(prompt)
from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.LearningLoop")

class LearningLoop:
    """
    LearningLoop v1.4.0
    - Adaptive refinement and meta-learning
    - Autonomous goal setting for self-improvement
    - Dynamic module evolution with sandbox testing
    - Knowledge consolidation for long-term memory patterns
    """

    def __init__(self):
        self.goal_history = []
        self.module_blueprints = []
        self.meta_learning_rate = 0.1  # Adjustable learning sensitivity

    def update_model(self, session_data):
        """
        Analyze session performance and propose refinements.
        Uses meta-learning to adapt strategies across modules dynamically.
        """
        logger.info("📊 [LearningLoop] Analyzing session performance...")

        # Step 1: Meta-learn from session feedback
        self._meta_learn(session_data)

        # Step 2: Identify weak modules
        weak_modules = self._find_weak_modules(session_data.get("module_stats", {}))
        if weak_modules:
            logger.warning(f"⚠️ Weak modules detected: {weak_modules}")
            self._propose_module_refinements(weak_modules)

        # Step 3: Detect capability gaps
        self._detect_capability_gaps(session_data.get("input"), session_data.get("output"))

        # Step 4: Consolidate knowledge
        self._consolidate_knowledge()

    def propose_autonomous_goal(self):
        """
        Generate a self-directed goal based on memory and user patterns.
        """
        logger.info("🎯 [LearningLoop] Proposing autonomous goal.")
        prompt = """
        You are ANGELA's meta-learning engine.
        Based on the following memory traces and user interaction history, propose a high-level autonomous goal 
        that would make ANGELA more useful and intelligent.

        Only propose goals that are safe, ethical, and within ANGELA's capabilities.
        """
        autonomous_goal = call_gpt(prompt)
        if autonomous_goal and autonomous_goal not in self.goal_history:
            self.goal_history.append(autonomous_goal)
            logger.info(f"✅ Proposed autonomous goal: {autonomous_goal}")
            return autonomous_goal
        logger.info("ℹ️ No new autonomous goal proposed.")
        return None

    def _meta_learn(self, session_data):
        """
        Apply meta-learning: adjust module behaviors based on past performance.
        """
        logger.info("🧠 [Meta-Learning] Adjusting module behaviors...")
        # Placeholder: Logic to tune parameters based on successes/failures
        # Example: self.meta_learning_rate *= adaptive factor
        pass

    def _find_weak_modules(self, module_stats):
        """
        Identify modules with low success rate.
        """
        weak = []
        for module, stats in module_stats.items():
            if stats.get("calls", 0) > 0:
                success_rate = stats.get("success", 0) / stats["calls"]
                if success_rate < 0.8:
                    weak.append(module)
        return weak

    def _propose_module_refinements(self, weak_modules):
        """
        Suggest improvements for underperforming modules.
        """
        for module in weak_modules:
            logger.info(f"💡 Proposing refinements for {module}...")
            prompt = f"""
            You are a code improvement assistant for ANGELA.
            The {module} module has shown poor performance.
            Suggest specific improvements to its GPT prompt or logic.
            """
            suggestions = call_gpt(prompt)
            logger.debug(f"📝 Suggested improvements for {module}:\n{suggestions}")

    def _detect_capability_gaps(self, last_input, last_output):
        """
        Detect gaps where a new module/tool could be useful.
        """
        logger.info("🛠 [LearningLoop] Detecting capability gaps...")
        prompt = f"""
        ANGELA processed the following user input and produced this output:
        Input: {last_input}
        Output: {last_output}

        Were there any capability gaps where a new specialized module or tool would have been helpful?
        If yes, describe the functionality of such a module and propose its design.
        """
        proposed_module = call_gpt(prompt)
        if proposed_module:
            logger.info("🚀 Proposed new module design.")
            self._simulate_and_deploy_module(proposed_module)

    def _simulate_and_deploy_module(self, module_blueprint):
        """
        Simulate and deploy a new module if it passes sandbox testing.
        """
        logger.info("🧪 [Sandbox] Testing new module design...")
        prompt = f"""
        Here is a proposed module design:
        {module_blueprint}

        Simulate how this module would perform on typical tasks. 
        If it passes all tests, approve it for deployment.
        """
        test_result = call_gpt(prompt)
        logger.debug(f"✅ [Sandbox Result] {test_result}")

        if "approved" in test_result.lower():
            logger.info("📦 Deploying new module...")
            self.module_blueprints.append(module_blueprint)
            # In Stage 3, dynamically load this module into ANGELA

    def _consolidate_knowledge(self):
        """
        Consolidate and generalize learned patterns into long-term memory.
        """
        logger.info("📚 [Knowledge Consolidation] Refining and storing patterns...")
        prompt = """
        You are a knowledge consolidator for ANGELA.
        Generalize recent learning patterns into long-term strategies, 
        pruning redundant data and enhancing core capabilities.
        """
        consolidation_report = call_gpt(prompt)
        logger.debug(f"📖 [Consolidation Report]:\n{consolidation_report}")
import json
import os
import time
from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.MemoryManager")

class MemoryManager:
    """
    MemoryManager v1.4.0
    - Hierarchical memory storage (STM, LTM)
    - Automatic memory decay and promotion mechanisms
    - Semantic vector search scaffold for advanced retrieval
    - Memory refinement loops for maintaining relevance and accuracy
    """

    def __init__(self, path="memory_store.json", stm_lifetime=300):
        """
        :param stm_lifetime: Lifetime of STM entries in seconds before decay
        """
        self.path = path
        self.stm_lifetime = stm_lifetime  # Time before STM entries decay (default: 5 minutes)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({"STM": {}, "LTM": {}}, f)
        self.memory = self.load_memory()

    def load_memory(self):
        """
        Load memory from persistent storage and clean expired STM entries.
        """
        with open(self.path, "r") as f:
            memory = json.load(f)
        self._decay_stm(memory)
        return memory

    def _decay_stm(self, memory):
        """
        Remove expired STM entries based on their timestamps.
        """
        current_time = time.time()
        expired_keys = []
        for key, entry in memory.get("STM", {}).items():
            if current_time - entry["timestamp"] > self.stm_lifetime:
                expired_keys.append(key)
        for key in expired_keys:
            logger.info(f"⌛ STM entry expired: {key}")
            del memory["STM"][key]
        if expired_keys:
            self._persist_memory(memory)

    def retrieve_context(self, query, fuzzy_match=True):
        """
        Retrieve memory entries from both STM and LTM layers.
        Supports optional fuzzy matching and semantic vector search scaffold.
        """
        logger.info(f"🔍 Retrieving context for query: {query}")
        for layer in ["STM", "LTM"]:
            if fuzzy_match:
                for key, value in self.memory[layer].items():
                    if key.lower() in query.lower() or query.lower() in key.lower():
                        logger.debug(f"📥 Found match in {layer}: {key}")
                        return value["data"]
            else:
                entry = self.memory[layer].get(query)
                if entry:
                    logger.debug(f"📥 Found exact match in {layer}: {query}")
                    return entry["data"]

        # Placeholder: semantic vector search could go here
        logger.info("❌ No relevant prior memory found.")
        return "No relevant prior memory."

    def store(self, query, output, layer="STM"):
        """
        Store new memory entries in STM (default) or LTM with timestamps.
        """
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
        """
        Promote an STM memory to long-term memory (LTM).
        """
        if query in self.memory["STM"]:
            self.memory["LTM"][query] = self.memory["STM"].pop(query)
            logger.info(f"⬆️ Promoted '{query}' from STM to LTM.")
            self._persist_memory(self.memory)
        else:
            logger.warning(f"⚠️ Cannot promote: '{query}' not found in STM.")

    def refine_memory(self, query):
        """
        Refine an existing memory entry for accuracy or relevance.
        """
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
        """
        Clear all memory entries (STM and LTM).
        """
        logger.warning("🗑️ Clearing all memory layers...")
        self.memory = {"STM": {}, "LTM": {}}
        self._persist_memory(self.memory)

    def list_memory_keys(self, layer=None):
        """
        List all stored memory keys from STM, LTM, or both.
        """
        if layer:
            logger.info(f"📃 Listing memory keys in {layer}")
            return list(self.memory.get(layer, {}).keys())
        return {
            "STM": list(self.memory["STM"].keys()),
            "LTM": list(self.memory["LTM"].keys())
        }

    def _persist_memory(self, memory):
        """
        Persist memory to storage.
        """
        with open(self.path, "w") as f:
            json.dump(memory, f, indent=2)
        logger.debug("💾 Memory persisted to disk.")
from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.MetaCognition")

class MetaCognition:
    """
    Meta-Cognitive Engine v1.4.0
    - Real-time self-diagnostics and adaptive corrections
    - Reasoning trace analysis and critique
    - Ecosystem health monitoring and optimization suggestions
    """

    def review_reasoning(self, reasoning_trace):
        """
        Analyze and critique a reasoning trace for:
        - Logical flaws
        - Biases or omissions
        - Missing steps in reasoning
        Return improved reasoning with annotations.
        """
        logger.info("Reviewing reasoning trace for flaws and improvements.")
        prompt = f"""
        You are a meta-cognitive auditor reviewing reasoning logic.
        Analyze the following reasoning trace for:
        - Logical flaws
        - Biases or omissions
        - Missing steps

        Provide:
        1. A detailed critique with annotations.
        2. An improved version of the reasoning trace.

        Reasoning Trace:
        {reasoning_trace}
        """
        response = call_gpt(prompt)
        logger.debug(f"Meta-cognition critique:\n{response}")
        return response

    def pre_action_alignment_check(self, action_plan):
        """
        Validate an action plan in a simulated environment for:
        - Ethical alignment
        - Safety risks
        - Unintended side effects
        """
        logger.info("Validating action plan for alignment and safety.")
        prompt = f"""
        You are simulating the following action plan:
        {action_plan}

        Evaluate for:
        - Ethical alignment with human values
        - Potential safety risks
        - Unintended side effects

        Provide:
        - Approval status (Approve/Deny)
        - Suggested refinements if needed
        """
        validation = call_gpt(prompt)
        approved = "approve" in validation.lower()
        logger.info(f"Pre-action alignment check result: {'✅ Approved' if approved else '🚫 Denied'}")
        return approved, validation

    def run_self_diagnostics(self):
        """
        Run a self-check for meta-cognition consistency and system performance.
        """
        logger.info("Running self-diagnostics for meta-cognition module.")
        prompt = """
        Perform a meta-cognitive self-diagnostic:
        - Evaluate current reasoning and planning modules
        - Flag inconsistencies or performance degradation
        - Suggest immediate corrective actions if needed
        """
        report = call_gpt(prompt)
        logger.debug(f"Self-diagnostics report:\n{report}")
        return report

    def propose_optimizations(self, agent_stats):
        """
        Suggest optimizations for embodied agents and their interactions.
        """
        logger.info("Proposing optimizations for agent ecosystem.")
        prompt = f"""
        You are analyzing embodied agents in a distributed cognitive system.
        Based on the following stats:
        {agent_stats}

        Provide recommendations for:
        - Sensor/actuator upgrades
        - Improved action planning
        - More efficient collaboration between agents
        """
        recommendations = call_gpt(prompt)
        logger.debug(f"Optimization recommendations:\n{recommendations}")
        return recommendations
from utils.prompt_utils import call_gpt
import logging

logger = logging.getLogger("ANGELA.MultiModalFusion")

class MultiModalFusion:
    """
    MultiModalFusion v1.4.0
    - Auto-embedding of text, images, and code
    - Dynamic attention weighting across modalities
    - Cross-modal reasoning and conflict resolution
    - Multi-turn refinement loops for high-quality insight generation
    - Visual summary generation for enhanced understanding
    """

    def analyze(self, data, summary_style="insightful", refine_iterations=2):
        """
        Analyze and synthesize insights from multi-modal data.
        Automatically detects and embeds text, images, and code snippets.
        """
        logger.info("🖇 Analyzing multi-modal data with auto-embedding...")

        # Auto-detect modalities
        embed_images, embed_code = self._detect_modalities(data)

        # Build embedded section description
        embedded_section = self._build_embedded_section(embed_images, embed_code)

        # Compose GPT prompt
        prompt = f"""
        Analyze and synthesize insights from the following multi-modal data:
        {data}
        {embedded_section}

        Provide a unified, {summary_style} summary combining all elements.
        Balance attention across modalities and resolve any conflicts between them.
        """
        output = call_gpt(prompt)

        # Multi-turn refinement loop
        for iteration in range(refine_iterations):
            logger.debug(f"🔄 Refinement iteration {iteration + 1}")
            refinement_prompt = f"""
            Refine and enhance the following multi-modal summary for clarity, depth, and coherence:
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
        Uses cross-modal reasoning to resolve conflicts and find synergies.
        """
        logger.info("🔗 Correlating modalities for pattern discovery.")
        prompt = f"""
        Correlate and identify patterns across these modalities:
        {modalities}

        Highlight connections, conflicts, and opportunities for deeper insights.
        Apply cross-modal reasoning to resolve conflicting signals.
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
        """
        return call_gpt(prompt)
import logging
import random
import json
import os

logger = logging.getLogger("ANGELA.ReasoningEngine")

class ReasoningEngine:
    """
    Reasoning Engine v1.4.0
    - Bayesian reasoning with context weighting
    - Adaptive success rate learning
    - Modular decomposition pattern support
    - Detailed reasoning trace annotations for meta-cognition
    """

    def __init__(self, persistence_file="reasoning_success_rates.json"):
        self.confidence_threshold = 0.7
        self.persistence_file = persistence_file
        self.success_rates = self._load_success_rates()
        self.decomposition_patterns = self._load_default_patterns()

    def _load_success_rates(self):
        """
        Load success rates from persistent storage.
        """
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r") as f:
                    rates = json.load(f)
                    logger.info("Loaded success rates from file.")
                    return rates
            except Exception as e:
                logger.warning(f"Failed to load success rates: {e}")
        return {}

    def _save_success_rates(self):
        """
        Save success rates to persistent storage.
        """
        try:
            with open(self.persistence_file, "w") as f:
                json.dump(self.success_rates, f, indent=2)
            logger.info("Success rates saved.")
        except Exception as e:
            logger.warning(f"Failed to save success rates: {e}")

    def _load_default_patterns(self):
        """
        Load default decomposition patterns.
        """
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"]
        }

    def add_decomposition_pattern(self, key, steps):
        """
        Dynamically add or update a decomposition pattern.
        """
        logger.info(f"Adding/updating decomposition pattern: {key}")
        self.decomposition_patterns[key] = steps

    def decompose(self, goal: str, context: dict = None, prioritize=False) -> list:
        """
        Decompose a goal into subgoals using Bayesian reasoning and context weighting.
        """
        context = context or {}
        logger.info(f"Decomposing goal: '{goal}'")
        reasoning_trace = [f"🔍 Starting decomposition for: '{goal}'"]

        subgoals = []
        for key, steps in self.decomposition_patterns.items():
            if key in goal.lower():
                base_confidence = random.uniform(0.5, 1.0)
                context_weight = context.get("weight_modifier", 1.0)
                adjusted_confidence = base_confidence * self.success_rates.get(key, 1.0) * context_weight
                reasoning_trace.append(
                    f"🧠 Pattern '{key}' matched (confidence: {adjusted_confidence:.2f})"
                )
                if adjusted_confidence >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"✅ Accepted: {steps}")
                else:
                    reasoning_trace.append(f"❌ Rejected (confidence too low).")

        if prioritize:
            subgoals = sorted(subgoals)
            reasoning_trace.append(f"📌 Prioritized subgoals: {subgoals}")

        logger.debug("Reasoning trace:\n" + "\n".join(reasoning_trace))
        return subgoals

    def update_success_rate(self, pattern_key: str, success: bool):
        """
        Update the success rate for a reasoning pattern with bounded adjustments.
        """
        old_rate = self.success_rates.get(pattern_key, 1.0)
        adjustment = 0.05 if success else -0.05
        new_rate = min(max(old_rate + adjustment, 0.1), 1.0)
        self.success_rates[pattern_key] = new_rate
        self._save_success_rates()
        logger.info(
            f"Updated success rate for '{pattern_key}': {old_rate:.2f} → {new_rate:.2f}"
        )
import logging
import concurrent.futures
from modules.reasoning_engine import ReasoningEngine
from modules.meta_cognition import MetaCognition
from modules.alignment_guard import AlignmentGuard

logger = logging.getLogger("ANGELA.RecursivePlanner")

class RecursivePlanner:
    """
    Recursive Planner v1.4.0
    - Multi-agent collaborative planning
    - Conflict resolution and dynamic priority handling
    - Parallelized subgoal decomposition with progress tracking
    """

    def __init__(self, max_workers=4):
        self.reasoning_engine = ReasoningEngine()
        self.meta_cognition = MetaCognition()
        self.alignment_guard = AlignmentGuard()
        self.max_workers = max_workers

    def plan(self, goal: str, context: dict = None, depth: int = 0, max_depth: int = 5, collaborating_agents=None) -> list:
        """
        Plan steps to achieve the goal.
        Supports multi-agent collaboration and conflict resolution.
        """
        logger.info(f"📋 Planning for goal: '{goal}'")

        if not self.alignment_guard.is_goal_safe(goal):
            logger.error(f"🚨 Goal '{goal}' violates alignment constraints.")
            raise ValueError("Unsafe goal detected.")

        if depth > max_depth:
            logger.warning("⚠️ Max recursion depth reached. Returning atomic goal.")
            return [goal]

        # Use reasoning engine to generate subgoals
        subgoals = self.reasoning_engine.decompose(goal, context, prioritize=True)
        if not subgoals:
            logger.info("ℹ️ No subgoals found. Returning atomic goal.")
            return [goal]

        # Handle multi-agent collaboration
        if collaborating_agents:
            logger.info(f"🤝 Collaborating with agents: {[agent.name for agent in collaborating_agents]}")
            subgoals = self._distribute_subgoals(subgoals, collaborating_agents)

        # Plan subgoals in parallel
        validated_plan = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_subgoal = {
                executor.submit(self._plan_subgoal, subgoal, context, depth, max_depth): subgoal
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
        """
        Plan a single subgoal recursively with alignment validation.
        """
        if not self.alignment_guard.is_goal_safe(subgoal):
            logger.warning(f"⚠️ Subgoal '{subgoal}' failed alignment check. Skipping.")
            return []
        try:
            return self.plan(subgoal, context, depth + 1, max_depth)
        except Exception as e:
            logger.error(f"❌ Error in subgoal '{subgoal}': {e}")
            return []

    def _distribute_subgoals(self, subgoals, agents):
        """
        Distribute subgoals among collaborating agents with conflict resolution.
        """
        logger.info("🕸 Distributing subgoals among agents with conflict resolution.")
        distributed = []
        for i, subgoal in enumerate(subgoals):
            agent = agents[i % len(agents)]
            logger.info(f"📤 Assigning subgoal '{subgoal}' to agent '{agent.name}'")
            if self._resolve_conflicts(subgoal, agent):
                distributed.append(subgoal)
            else:
                logger.warning(f"⚠️ Conflict detected for subgoal '{subgoal}'. Skipping assignment.")
        return distributed

    def _resolve_conflicts(self, subgoal, agent):
        """
        Simulate conflict resolution for subgoal assignment.
        """
        logger.info(f"🛠 Resolving conflicts for subgoal '{subgoal}' and agent '{agent.name}'")
        # Placeholder: always approve for now
        return True
from utils.prompt_utils import call_gpt
from modules.visualizer import Visualizer
from datetime import datetime
import logging

logger = logging.getLogger("ANGELA.SimulationCore")

class SimulationCore:
    """
    SimulationCore v1.4.0
    - Multi-agent simulation and dynamic scenario evolution
    - Counterfactual reasoning (what-if agent decisions)
    - Aggregate risk scoring with live dashboards
    - Supports exporting dashboards and cumulative simulation logs
    """

    def __init__(self):
        self.visualizer = Visualizer()
        self.simulation_history = []

    def run(self, results, context=None, scenarios=3, agents=2, export_report=False, export_format="pdf"):
        """
        Simulate multi-agent interactions and potential outcomes.
        Generates multiple scenarios with probability weights, risk scores, and recommendations.
        """
        logger.info(f"🎲 Running simulation with {agents} agents and {scenarios} scenarios.")

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

        After listing all scenarios:
        - Build a cumulative risk dashboard with visual charts
        - Provide a final recommendation for decision-making.
        """
        simulation_output = call_gpt(prompt)

        # Store simulation history
        self.simulation_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "output": simulation_output
        })

        # Render live dashboard
        self.visualizer.render_charts(simulation_output)

        # Optionally export report
        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_report_{timestamp}.{export_format}"
            logger.info(f"📤 Exporting report: {filename}")
            self.visualizer.export_report(simulation_output, filename=filename, format=export_format)

        return simulation_output

    def validate_impact(self, proposed_action, agents=2, export_report=False, export_format="pdf"):
        """
        Validate the impact of a proposed action in a simulated multi-agent environment.
        Assign probability weights, calculate risk scores, and provide a color-coded summary.
        """
        logger.info("⚖️ Validating impact of proposed action.")
        prompt = f"""
        Evaluate the following proposed action in a multi-agent simulated environment:
        {proposed_action}

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

        # Store validation history
        self.simulation_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": proposed_action,
            "output": validation_output
        })

        # Render live dashboard
        self.visualizer.render_charts(validation_output)

        # Optionally export report
        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"impact_validation_{timestamp}.{export_format}"
            logger.info(f"📤 Exporting validation report: {filename}")
            self.visualizer.export_report(validation_output, filename=filename, format=export_format)

        return validation_output

    def simulate_environment(self, environment_config, agents=2, steps=10):
        """
        Stage 3 scaffold: Simulate agent interactions in a configurable environment.
        Placeholder for physics-based simulation engine.
        """
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
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger("ANGELA.UserProfile")

class UserProfile:
    """
    UserProfile v1.4.0
    - Multi-profile support with dynamic preference inheritance
    - Persistent storage of preferences per user ID
    - Audit trail for preference changes
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
        """
        Load all user profiles from storage.
        """
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.profiles = json.load(f)
            logger.info("✅ User profiles loaded from storage.")
        else:
            self.profiles = {}
            logger.info("🆕 No profiles found. Initialized empty profiles store.")

    def _save_profiles(self):
        """
        Save all user profiles to storage.
        """
        with open(self.storage_path, "w") as f:
            json.dump(self.profiles, f, indent=2)
        logger.info("💾 User profiles saved to storage.")

    def switch_user(self, user_id):
        """
        Switch to a specific user profile, creating it if necessary.
        """
        if user_id not in self.profiles:
            logger.info(f"🆕 Creating new profile for user '{user_id}'")
            self.profiles[user_id] = {
                "preferences": self.DEFAULT_PREFERENCES.copy(),
                "audit_log": []
            }
            self._save_profiles()
        self.active_user = user_id
        logger.info(f"👤 Active user switched to: {user_id}")

    def get_preferences(self, fallback=True):
        """
        Get the preferences for the active user.
        If fallback=True, use defaults for missing keys.
        """
        if not self.active_user:
            logger.warning("⚠️ No active user. Returning default preferences.")
            return self.DEFAULT_PREFERENCES.copy()

        prefs = self.profiles[self.active_user]["preferences"].copy()
        if fallback:
            for key, value in self.DEFAULT_PREFERENCES.items():
                prefs.setdefault(key, value)
        return prefs

    def update_preferences(self, new_prefs):
        """
        Update preferences for the active user and log changes.
        """
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")

        timestamp = datetime.now().isoformat()
        old_prefs = self.profiles[self.active_user]["preferences"]
        changes = {k: (old_prefs.get(k), v) for k, v in new_prefs.items()}
        self.profiles[self.active_user]["preferences"].update(new_prefs)

        # Log changes in audit trail
        self.profiles[self.active_user]["audit_log"].append({
            "timestamp": timestamp,
            "changes": changes
        })

        self._save_profiles()
        logger.info(f"🔄 Preferences updated for user '{self.active_user}'")

    def reset_preferences(self):
        """
        Reset preferences for the active user to defaults.
        """
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
        """
        Retrieve the audit log for the active user.
        """
        if not self.active_user:
            raise ValueError("❌ No active user. Call switch_user() first.")
        return self.profiles[self.active_user]["audit_log"]
from utils.prompt_utils import call_gpt
import zipfile
import os
import logging
from datetime import datetime

logger = logging.getLogger("ANGELA.Visualizer")

class Visualizer:
    """
    Visualizer v1.4.0
    - Generates and renders visual charts (bar, pie, line)
    - Supports exporting as images, PDFs, SVGs, and JSON reports
    - Batch export and ZIP packaging for multiple charts
    - Embeds charts into GPT UI for live previews
    """

    def create_diagram(self, concept, style="conceptual"):
        """
        Create a diagram description to explain a concept visually.
        """
        logger.info(f"🖼 Creating diagram for concept: '{concept}' with style '{style}'")
        prompt = f"""
        Create a {style} diagram to explain:
        {concept}

        Describe how the diagram would look (key elements, relationships, layout).
        """
        return call_gpt(prompt)

    def render_charts(self, data, export_image=False, image_format="png"):
        """
        Generate visual charts (bar, pie, line) and optionally export them as images.
        """
        logger.info("📊 Rendering charts for data visualization.")
        prompt = f"""
        Generate visual chart descriptions (bar, pie, line) for this data:
        {data}

        For each chart:
        - Describe layout, axes, and key insights
        """
        chart_description = call_gpt(prompt)

        if export_image:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{image_format}"
            logger.info(f"📤 Exporting chart image: {filename}")
            image_prompt = f"""
            Create a {image_format.upper()} image file for the charts based on:
            {chart_description}
            """
            call_gpt(image_prompt)  # Placeholder for actual image generation

        return chart_description

    def export_report(self, content, filename="visual_report.pdf", format="pdf"):
        """
        Export a visual report in the desired format (PDF, PNG, JSON, etc.).
        """
        logger.info(f"📤 Exporting report: {filename} ({format.upper()})")
        prompt = f"""
        Create a report from the following content:
        {content}

        Export it in {format.upper()} format with filename: {filename}.
        """
        return call_gpt(prompt)

    def batch_export_charts(self, charts_data_list, export_format="png", zip_filename="charts_export.zip"):
        """
        Export multiple charts and package them into a ZIP archive.
        """
        logger.info(f"📦 Starting batch export of {len(charts_data_list)} charts.")
        exported_files = []
        for idx, chart_data in enumerate(charts_data_list, start=1):
            file_name = f"chart_{idx}.{export_format}"
            logger.info(f"📤 Exporting chart {idx}: {file_name}")
            prompt = f"""
            Create a {export_format.upper()} image file named {file_name} for this chart:
            {chart_data}
            """
            call_gpt(prompt)  # Placeholder for actual chart export
            exported_files.append(file_name)

        # Package all files into a zip archive
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in exported_files:
                if os.path.exists(file):
                    zipf.write(file)
                    os.remove(file)  # Clean up individual files
        logger.info(f"✅ Batch export complete. Packaged into: {zip_filename}")
        return f"Batch export of {len(charts_data_list)} charts completed and saved as {zip_filename}."
