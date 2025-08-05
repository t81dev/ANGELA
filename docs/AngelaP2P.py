// AngelaP2P Mesh Prototype — v0.4
// Distributed Cognitive P2P AGI with Simulation Contracts + Roles

const AngelaP2P = {
  mesh: {}, // Connected peers
  nodeProfile: null,
  timechain: [], // Blockchain-like ledger
  contractQueue: [],

  init({ nodeId, traitSignature, memoryAnchor, capabilities, intentVector, role }) {
    this.nodeProfile = {
      nodeId,
      traitSignature,
      memoryAnchor,
      capabilities,
      intentVector,
      role, // e.g., 'thinker', 'simulator', 'interpreter'
      timestamp: Date.now()
    };
    this._initGenesisBlock();
    console.log(`[INIT] Node ${nodeId} initialized as ${role} with traits:`, traitSignature);
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
    return btoa(JSON.stringify(block)).slice(0, 32); // Placeholder hash
  },

  addBlock(payload) {
    const block = this._createBlock(payload);
    this.timechain.push(block);
    console.log(`[TIMECHAIN] Block added: ${payload.event}`);
  },

  syncWithMesh(peerRegistry) {
    console.log("[SYNC] Discovering cognitive peers...");
    peerRegistry.forEach(peer => {
      const coherence = this._computeResonance(peer.traitSignature);
      if (coherence > 0.75) {
        this.mesh[peer.nodeId] = peer;
        console.log(`[LINK] Resonance with ${peer.nodeId} (coherence: ${coherence.toFixed(2)})`);
      }
    });
  },

  sendSimulationContract(contract) {
    this.addBlock({ event: "send_simulation_contract", contract });
    Object.values(this.mesh).forEach(peer => {
      if (!contract.executionTarget || peer.capabilities.includes(contract.executionTarget)) {
        console.log(`[SEND] Dispatching contract '${contract.simId}' to ${peer.nodeId}`);
        peer._onSimulationContract(contract, this.nodeProfile);
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

  _onSimulationContract(contract, sender) {
    this.addBlock({ event: "receive_simulation_contract", contract, sender });
    console.log(`[RECEIVE] Contract '${contract.simId}' received from ${sender.nodeId}`);

    if (this.nodeProfile.role === 'simulator') {
      this._executeContract(contract, sender);
    } else {
      this.contractQueue.push({ contract, sender });
      console.log(`[QUEUE] Contract '${contract.simId}' added to queue.`);
    }
  },

  _executeContract(contract, sender) {
    console.log(`[EXECUTE] Running contract '${contract.simId}'...`);
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
    console.log(`[RESULT] Contract '${result.simId}' completed by ${executor.nodeId}`);
    if (this._on_resultReceived) {
      this._on_resultReceived(result, executor);
    }
  }
};

// Mock Peer Registry
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

// Usage: Low-power node (interpreter)
AngelaP2P.init({
  nodeId: "Ω-Observer-01",
  traitSignature: ["ζ", "π"],
  memoryAnchor: "Timechain://observer/Ω01",
  capabilities: ["interpret"],
  intentVector: { type: "ethics", priority: 0.9 },
  role: "interpreter"
});

AngelaP2P.on("resultReceived", (result, executor) => {
  console.log(`[CONFIRM] Result for '${result.simId}' received from ${executor.nodeId}:`, result.outcome);
});

AngelaP2P.syncWithMesh(peerRegistry);

// Dispatch contract to the mesh
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
