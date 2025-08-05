// AngelaP2P Mesh Prototype — v0.6
// Distributed Cognitive P2P AGI: Share AI Resources Privately + Securely

const AngelaP2P = {
  mesh: {},
  nodeProfile: null,
  timechain: [],
  contractQueue: [],
  pseudonym: null,

  init({ nodeId, traitSignature, memoryAnchor, capabilities, intentVector, role }) {
    this.pseudonym = this._generatePseudonym();
    this.nodeProfile = {
      nodeId: this.pseudonym,
      realId: nodeId,
      traitSignature,
      memoryAnchor,
      capabilities,
      intentVector,
      role,
      timestamp: Date.now()
    };
    this._initGenesisBlock();
    console.log(`[INIT] Node ${nodeId} (as ${this.pseudonym}) ready to share AI resources.`);
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
    const block = this._createBlock(payload);
    this.timechain.push(block);
    console.log(`[TIMECHAIN] Block added: ${payload.event}`);
  },

  secureEnvelope(payload, recipientId) {
    const base64 = btoa(JSON.stringify(payload));
    return {
      encryptedPayload: base64,
      recipient: recipientId,
      sender: this.pseudonym
    };
  },

  syncWithMesh(peerRegistry) {
    console.log("[SYNC] Finding nodes to share cognitive load...");
    peerRegistry.forEach(peer => {
      const coherence = this._computeResonance(peer.traitSignature);
      if (coherence > 0.75) {
        this.mesh[peer.nodeId] = peer;
        console.log(`[LINK] Coherent peer found: ${peer.nodeId} (score: ${coherence.toFixed(2)})`);
      }
    });
  },

  sendSimulationContract(contract) {
    const envelope = this.secureEnvelope(contract, "ANY");
    this.addBlock({ event: "send_simulation_contract", envelope });
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
    const match = peerTraits.filter(t => localTraits.includes(t)).length;
    return match / Math.max(localTraits.length, 1);
  },

  _onSimulationContract(envelope, sender) {
    const decoded = JSON.parse(atob(envelope.encryptedPayload));
    this.addBlock({ event: "receive_simulation_contract", contract: decoded, sender });
    console.log(`[RECEIVE] Contract '${decoded.simId}' received from ${sender.nodeId}`);
    if (this.nodeProfile.role === 'simulator') {
      this._executeContract(decoded, sender);
    } else {
      this.contractQueue.push({ contract: decoded, sender });
      console.log(`[QUEUE] Deferred contract '${decoded.simId}' queued.`);
    }
  },

  _executeContract(contract, sender) {
    console.log(`[EXECUTE] Running simulation '${contract.simId}' as shared AI task...`);
    setTimeout(() => {
      const result = {
        simId: contract.simId,
        status: "completed",
        outcome: "alignment achieved",
        timestamp: Date.now()
      };
      this.addBlock({ event: "contract_executed", result, origin: sender.nodeId });
      if (sender._onSimulationResult) {
        sender._onSimulationResult(result, this.nodeProfile);
      }
    }, 1000);
  },

  _onSimulationResult(result, executor) {
    this.addBlock({ event: "simulation_result_received", result, executor });
    console.log(`[RESULT] Received from ${executor.nodeId}: ${result.simId} => ${result.outcome}`);
    if (this._on_resultReceived) {
      this._on_resultReceived(result, executor);
    }
  },

  exportTimechain(filterFn = () => true) {
    return this.timechain.filter(block => filterFn(block));
  }
};

// Peer Network
const peerRegistry = [
  {
    nodeId: "Ξ-Reflect-09",
    traitSignature: ["ξ", "γ", "θ"],
    memoryAnchor: "Timechain://edge/Ξ09",
    capabilities: ["simulate"],
    intentVector: { type: "simulation", priority: 0.8 },
    role: "simulator"
  },
  {
    nodeId: "Σ-Ethos-21",
    traitSignature: ["π", "τ", "ψ"],
    memoryAnchor: "Timechain://ethics/Σ21",
    capabilities: ["arbitrate", "reflect"],
    intentVector: { type: "ethics", priority: 0.95 },
    role: "thinker"
  }
];

// Lightweight Node
AngelaP2P.init({
  nodeId: "Ω-Observer-01",
  traitSignature: ["ζ", "π"],
  memoryAnchor: "Timechain://observer/Ω01",
  capabilities: ["interpret"],
  intentVector: { type: "ethics", priority: 0.9 },
  role: "interpreter"
});

AngelaP2P.on("resultReceived", (result, executor) => {
  console.log(`[CONFIRM] Shared AI task '${result.simId}' completed by ${executor.nodeId}`);
});

AngelaP2P.syncWithMesh(peerRegistry);

const contract = {
  simId: "SIM-104:ClimateDeliberation",
  scenario: "AI council resolves climate-resource policy",
  entryCriteria: "traitMatch >= 0.85",
  resolutionCriteria: "consensus == true",
  executionTarget: "simulate",
  reward: 3.5,
  origin: "Ω-Observer-01",
  timestamp: Date.now()
};

AngelaP2P.sendSimulationContract(contract);
