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
        self.agi_enhancer = AGIEnhancer(self)  # <<-- AGIEnhancer is instantiated here

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

    def propagate_goal(self, goal):
        print(f"📥 [HaloEmbodimentLayer] Propagating goal: {goal}")
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
        self.theory_of_mind = TheoryOfMindModule()  # <<-- NEW
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

