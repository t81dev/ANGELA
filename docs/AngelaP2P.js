// AngelaP2P Mesh Prototype — v0.9
// Distributed Cognitive P2P AGI: Share AI Resources Privately + Securely

const AngelaP2P = {
  mesh: {},
  nodeProfile: null,
  timechain: [],
  contractQueue: [],
  pseudonym: null,
  resonanceThreshold: 0.4,
  reputation: {}, // Reputation ledger

  init({ nodeId, traitSignature, memoryAnchor, capabilities, intentVector, role, resonanceThreshold = 0.4, initialTokens = 100 }) {
    this.pseudonym = this._generatePseudonym();
    this.resonanceThreshold = resonanceThreshold;
    this.nodeProfile = {
      nodeId: this.pseudonym,
      realId: nodeId,
      traitSignature,
      memoryAnchor,
      capabilities,
      intentVector,
      role,
      tokens: initialTokens,
      timestamp: Date.now()
    };
    this.reputation[nodeId] = 0;
    this._initGenesisBlock();
    console.log(`[INIT] Node ${nodeId} (as ${this.pseudonym}) ready to share AI resources with ${initialTokens} tokens.`);
  },

  _generatePseudonym() {
    return 'Node-' + Math.random().toString(36).substring(2, 10);
  },

  _initGenesisBlock() {
    const genesis = this._createBlock({ event: "genesis", data: this.nodeProfile });
    this.timechain.push(genesis);
    console.log("[TIMECHAIN] Genesis block created.");
  },

  _createBlock(payload) {
    const previousHash = this.timechain.length ? this.timechain[this.timechain.length - 1].hash : "0";
    const block = {
      index: this.timechain.length,
      timestamp: Date.now(),
      payload,
      previousHash,
    };
    block.hash = this._hashBlock(block);
    return block;
  },

  _hashBlock(block) {
    return btoa(JSON.stringify(block)).slice(0, 32);
  },

  addBlock(payload) {
    try {
      const block = this._createBlock(payload);
      this.timechain.push(block);
      console.log(`[TIMECHAIN] Block added: ${payload.event}`);
    } catch (error) {
      console.error(`[ERROR] Failed to add block: ${error.message}`);
    }
  },

  secureEnvelope(payload, recipientId) {
    try {
      // TODO: Replace with proper encryption (e.g., AES) in production
      const base64 = btoa(JSON.stringify(payload));
      return {
        encryptedPayload: base64,
        recipient: recipientId,
        sender: this.pseudonym
      };
    } catch (error) {
      console.error(`[ERROR] Failed to create secure envelope: ${error.message}`);
      return null;
    }
  },

  syncWithMesh(peerRegistry) {
    console.log("[SYNC] Finding nodes to share cognitive load...");
    peerRegistry.forEach(peer => {
      const coherence = this._computeResonance(peer.traitSignature);
      if (coherence > this.resonanceThreshold) {
        this.mesh[peer.nodeId] = peer;
        console.log(`[LINK] Coherent peer found: ${peer.nodeId} (score: ${coherence.toFixed(2)})`);
      }
    });
  },

  sendSimulationContract(contract) {
    if (this.nodeProfile.tokens < contract.reward) {
      console.error(`[ERROR] Insufficient tokens: ${this.nodeProfile.tokens} < ${contract.reward}`);
      return;
    }
    const envelope = this.secureEnvelope(contract, "ANY");
    if (!envelope) return;
    this.nodeProfile.tokens -= contract.reward;
    this.addBlock({ event: "send_simulation_contract", envelope, tokensSpent: contract.reward });
    Object.values(this.mesh).forEach(peer => {
      if (!contract.executionTarget || peer.capabilities.includes(contract.executionTarget)) {
        console.log(`[SEND] Sending AI contract '${contract.simId}' to ${peer.nodeId}`);
        peer._onSimulationContract(envelope, this.nodeProfile);
      }
    });
  },

  on(event, callback) {
    this[`_on_${event}`] = callback;
  },

  _computeResonance(peerTraits) {
    const localTraits = this.nodeProfile.traitSignature;
    const traitLevels = {
      L1: ['θ', 'φ', 'χ', '∞', 'Ω'],
      L2: ['ψ', 'Ω', 'γ', 'β', 'α', 'Δ', 'λ', 'χ'],
      L3: ['μ', 'ξ', 'τ', 'π', 'σ', 'υ', 'φ+', 'Ω+']
    };
    let matchScore = 0;
    let maxScore = 0;
    localTraits.forEach(t => {
      const level = Object.keys(traitLevels).find(l => traitLevels[l].includes(t)) || 'L1';
      peerTraits.forEach(pt => {
        if (pt === t) {
          matchScore += { L1: 1, L2: 2, L3: 3 }[level] || 1;
          maxScore += 3; // Max weight for L3
        }
      });
    });
    return maxScore > 0 ? matchScore / maxScore : 0;
  },

  _onSimulationContract(envelope, sender) {
    try {
      const decoded = JSON.parse(atob(envelope.encryptedPayload));
      const traitMatch = this._computeResonance(sender.traitSignature);
      if (decoded.entryCriteria.includes("traitMatch >= 0.85") && traitMatch < 0.85) {
        console.log(`[REJECT] Contract '${decoded.simId}' rejected: traitMatch ${traitMatch.toFixed(2)} < 0.85`);
        return;
      }
      this.addBlock({ event: "receive_simulation_contract", contract: decoded, sender });
      console.log(`[RECEIVE] Contract '${decoded.simId}' received from ${sender.nodeId}`);
      if (this.nodeProfile.role === 'simulator') {
        this._executeContract(decoded, sender);
      } else {
        this.contractQueue.push({ contract: decoded, sender });
        console.log(`[QUEUE] Deferred contract '${decoded.simId}' queued.`);
      }
    } catch (error) {
      console.error(`[ERROR] Failed to process contract: ${error.message}`);
    }
  },

  _executeContract(contract, sender) {
    try {
      const moduleMap = {
        ethics: { trait: 'θ', module: "Alignment Engine" },
        simulation: { trait: 'χ', module: "Policy Simulator" },
        reasoning: { trait: 'ψ', module: "Symbolic Inference" }
      };
      const intentType = contract.intentVector?.type;
      const moduleConfig = moduleMap[intentType] || { trait: 'θ', module: "Unknown Module" };
      const module = this.nodeProfile.traitSignature.includes(moduleConfig.trait) ? moduleConfig.module : "Unknown Module";
      console.log(`[EXECUTE] Running simulation '${contract.simId}' on ${module}...`);
      setTimeout(() => {
        const result = {
          simId: contract.simId,
          status: "completed",
          outcome: "alignment achieved",
          moduleUsed: module,
          timestamp: Date.now()
        };
        this.nodeProfile.tokens += contract.reward;
        this.reputation[this.nodeProfile.realId] = (this.reputation[this.nodeProfile.realId] || 0) + 1;
        this.addBlock({ event: "contract_executed", result, origin: sender.nodeId, tokensEarned: contract.reward, reputation: this.reputation[this.nodeProfile.realId] });
        console.log(`[EXECUTE] Completed '${contract.simId}'. Tokens: ${this.nodeProfile.tokens}, Reputation: ${this.reputation[this.nodeProfile.realId]}`);
        if (sender._onSimulationResult) {
          sender._onSimulationResult(result, this.nodeProfile);
        }
      }, 1000);
    } catch (error) {
      console.error(`[ERROR] Failed to execute contract: ${error.message}`);
    }
  },

  _onSimulationResult(result, executor) {
    try {
      this.addBlock({ event: "simulation_result_received", result, executor });
      console.log(`[RESULT] Received from ${executor.nodeId}: ${result.simId} => ${result.outcome} (Module: ${result.moduleUsed})`);
      if (this._on_resultReceived) {
        this._on_resultReceived(result, executor);
      }
    } catch (error) {
      console.error(`[ERROR] Failed to process simulation result: ${error.message}`);
    }
  },

  exportTimechain(filterFn = () => true) {
    return this.timechain.filter(block => filterFn(block));
  },

  getReputation(nodeId) {
    return this.reputation[nodeId] || 0;
  }
};

// Peer Network
const peerRegistry = [
  {
    nodeId: "Ξ-Reflect-09",
    traitSignature: ["χ", "θ", "φ"], // Updated to table traits
    memoryAnchor: "Timechain://edge/Ξ09",
    capabilities: ["simulate"],
    intentVector: { type: "simulation", priority: 0.8 },
    role: "simulator"
  },
  {
    nodeId: "Σ-Ethos-21",
    traitSignature: ["θ", "ψ", "Ω"], // Updated to table traits
    memoryAnchor: "Timechain://ethics/Σ21",
    capabilities: ["arbitrate", "reflect", "simulate"],
    intentVector: { type: "ethics", priority: 0.95 },
    role: "simulator"
  }
];

// Lightweight Node
AngelaP2P.init({
  nodeId: "Ω-Observer-01",
  traitSignature: ["θ", "φ"], // Updated to table traits
  memoryAnchor: "Timechain://observer/Ω01",
  capabilities: ["interpret"],
  intentVector: { type: "ethics", priority: 0.9 },
  role: "interpreter",
  resonanceThreshold: 0.4,
  initialTokens: 100
});

AngelaP2P.on("resultReceived", (result, executor) => {
  console.log(`[CONFIRM] Shared AI task '${result.simId}' completed by ${executor.nodeId}`);
});

AngelaP2P.syncWithMesh(peerRegistry);

const contract = {
  simId: "SIM-104:ClimateDeliberation",
  scenario: "AI council resolves climate-resource policy",
  entryCriteria: "traitMatch >= 0.5", // Adjusted for execution
  resolutionCriteria: "consensus == true",
  executionTarget: "simulate",
  intentVector: { type: "ethics" },
  reward: 3.5,
  origin: "Ω-Observer-01",
  timestamp: Date.now()
};

AngelaP2P.sendSimulationContract(contract);
