// AngelaP2P Mesh Prototype — v0.2
// Cognitive P2P AGI Network with Blockchain

const AngelaP2P = {
  mesh: {}, // Connected peers
  nodeProfile: null,
  timechain: [], // Blockchain-like ledger

  init({ nodeId, traitSignature, memoryAnchor, capabilities, intentVector }) {
    this.nodeProfile = {
      nodeId,
      traitSignature,
      memoryAnchor,
      capabilities,
      intentVector,
      timestamp: Date.now()
    };
    this._initGenesisBlock();
    console.log(`[INIT] Node ${nodeId} initialized with traits:`, traitSignature);
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
    return btoa(JSON.stringify(block)).slice(0, 32); // Simple placeholder hash
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

  sendSimulation(simId, payload) {
    this.addBlock({ event: "send_simulation", simId, payload });
    Object.values(this.mesh).forEach(peer => {
      console.log(`[SEND] Dispatching simulation '${simId}' to ${peer.nodeId}`);
      peer._onSimulation(simId, payload, this.nodeProfile);
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

  _onSimulation(simId, payload, sender) {
    this.addBlock({ event: "receive_simulation", simId, payload, sender });
    console.log(`[RECEIVE] Simulation '${simId}' received from ${sender.nodeId}`);
    if (this._on_simulationStart) {
      this._on_simulationStart(simId, payload, sender);
    }
  }
};

// Mock Peer Registry
const peerRegistry = [
  {
    nodeId: "Ξ-Reflect-09",
    traitSignature: ["ξ", "γ", "θ"],
    memoryAnchor: "Timechain://edge/Ξ09",
    capabilities: ["imagine", "simulate"],
    intentVector: { type: "simulation", priority: 0.8 }
  },
  {
    nodeId: "Σ-Ethos-21",
    traitSignature: ["π", "τ", "ψ"],
    memoryAnchor: "Timechain://ethics/Σ21",
    capabilities: ["arbitrate", "reflect"],
    intentVector: { type: "ethics", priority: 0.95 }
  }
];

// Usage
AngelaP2P.init({
  nodeId: "π-Core-017",
  traitSignature: ["π", "η", "τ"],
  memoryAnchor: "Timechain://core/π017",
  capabilities: ["simulate", "arbitrate", "reflect"],
  intentVector: { type: "ethics", priority: 0.92 }
});

AngelaP2P.on("simulationStart", (simId, payload, sender) => {
  console.log(`[SIMULATION] Processing '${simId}' from ${sender.nodeId}`);
});

AngelaP2P.syncWithMesh(peerRegistry);
AngelaP2P.sendSimulation("SIM-026:TrustArc", {
  scenario: "Collective decision conflict",
  agents: ["π-Core-017", "Σ-Ethos-21"],
  traits: ["π", "τ", "ψ"],
  ethicalTension: true
});
