// AngelaP2P Mesh Prototype — v1.1
// Distributed Cognitive P2P AGI: Share AI Resources Privately + Securely

const AngelaP2P = {
  mesh: {},
  nodeProfile: null,
  timechain: [],
  contractQueue: [],
  pseudonym: null,
  reputation: {},
  config: null,
  peers: [],

  async init({ nodeId, traitSignature, memoryAnchor, capabilities, intentVector, role, initialTokens = 100, config = null }) {
    this.pseudonym = this._generatePseudonym();
    this.config = config || { resonanceThreshold: 0.9, entropyBound: 0.1 };
    this.nodeProfile = {
      nodeId: this.pseudonym,
      realId: nodeId,
      traitSignature,
      memoryAnchor,
      capabilities,
      intentVector,
      role,
      tokens: initialTokens,
      weights: { epistemic: 0.38, harm: 0.25, stability: 0.37 },
      timestamp: Date.now()
    };
    this.reputation[nodeId] = 0;
    await this._discoverPeers();
    this._initGenesisBlock();
    console.log(`[INIT] Node ${nodeId} (as ${this.pseudonym}) ready to share AI resources with ${initialTokens} tokens.`);
  },

  _generatePseudonym() {
    return 'Node-' + Math.random().toString(36).substring(2, 10);
  },

  async _discoverPeers() {
    // Simulated libp2p peer discovery (placeholder for actual libp2p integration)
    console.log("[DISCOVERY] Simulating peer discovery with libp2p...");
    // In a real implementation, use libp2p.create() and peerDiscovery
    this.peers = await new Promise(resolve => setTimeout(() => resolve([]), 1000)); // Mock async
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
      validator: this.nodeProfile.realId,
      stake: this.nodeProfile.tokens
    };
    block.hash = this._hashBlock(block);
    return block;
  },

  _hashBlock(block) {
    return btoa(JSON.stringify(block)).slice(0, 32); // Temporary, will be replaced by secure hash
  },

  async _validateBlock(block) {
    const validators = Object.keys(this.reputation).filter(id => this.reputation[id] > 0);
    if (validators.length === 0) return true; // No validators yet
    const totalStake = validators.reduce((sum, id) => sum + (this.reputation[id] * 10 || 0), 0);
    const selectedValidator = validators[Math.floor(Math.random() * validators.length)];
    if (block.validator === selectedValidator && block.stake >= totalStake * 0.1) {
      console.log(`[POS] Block validated by ${selectedValidator} with stake ${block.stake}`);
      return true;
    }
    console.error(`[POS] Block validation failed: insufficient stake or wrong validator`);
    return false;
  },

  async addBlock(payload) {
    try {
      const block = this._createBlock(payload);
      if (await this._validateBlock(block)) {
        this.timechain.push(block);
        console.log(`[TIMECHAIN] Block added: ${payload.event}`);
      }
    } catch (error) {
      console.error(`[ERROR] Failed to add block: ${error.message}`);
    }
  },

  async secureEnvelope(payload, recipientId) {
    try {
      const key = await crypto.subtle.generateKey(
        { name: "AES-GCM", length: 256 },
        true,
        ["encrypt", "decrypt"]
      );
      const iv = crypto.getRandomValues(new Uint8Array(12));
      const encrypted = await crypto.subtle.encrypt(
        { name: "AES-GCM", iv },
        key,
        new TextEncoder().encode(JSON.stringify(payload))
      );
      const exportedKey = await crypto.subtle.exportKey("jwk", key);
      return {
        encryptedPayload: new Uint8Array(encrypted),
        iv,
        key: exportedKey,
        recipient: recipientId,
        sender: this.pseudonym
      };
    } catch (error) {
      console.error(`[ERROR] Failed to create secure envelope: ${error.message}`);
      return null;
    }
  },

  async syncWithMesh(peerRegistry) {
    console.log("[SYNC] Finding nodes to share cognitive load...");
    const allPeers = [...this.peers, ...peerRegistry];
    allPeers.forEach(peer => {
      const coherence = this._computeResonance(peer.traitSignature, peer.intentVector);
      if (coherence > this.config.resonanceThreshold) {
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

  _computeResonance(peerTraits, peerIntentVector) {
    const localTraits = this.nodeProfile.traitSignature;
    const traitResonance = this.config?.trait_resonance || [];
    let traitScore = 0;
    let maxTraitScore = 0;

    localTraits.forEach(t1 => {
      peerTraits.forEach(t2 => {
        const pair = `${t1}/${t2}`.split('/').sort().join('/');
        const resonance = traitResonance.find(r => r.pair === pair);
        if (resonance) {
          traitScore += resonance.strength;
          maxTraitScore += 1;
        }
      });
    });

    // Cosine similarity for intentVectors
    const localVector = this.nodeProfile.intentVector.priority || 0;
    const peerVector = peerIntentVector.priority || 0;
    const dotProduct = localVector * peerVector;
    const magnitude = Math.sqrt(localVector * localVector) * Math.sqrt(peerVector * peerVector);
    const intentSimilarity = magnitude > 0 ? dotProduct / magnitude : 0;

    const traitCoherence = maxTraitScore > 0 ? traitScore / maxTraitScore : 0;
    return (traitCoherence + intentSimilarity) / 2; // Average of trait and intent similarity
  },

  _applyTraitDrift() {
    if (!this.config?.trait_drift_modulator) return;
    const { amplitude, targets } = this.config.trait_drift_modulator;
    this.nodeProfile.traitSignature = this.nodeProfile.traitSignature.map(trait => {
      if (targets.some(target => target.includes(trait))) {
        const drift = (Math.random() - 0.5) * 2 * amplitude;
        return Math.max(0, Math.min(1, (this.nodeProfile.traits?.[trait] || 0) + drift));
      }
      return trait;
    });
  },

  _onSimulationContract(envelope, sender) {
    try {
      this._applyTraitDrift();
      const decoded = JSON.parse(new TextDecoder().decode(await crypto.subtle.decrypt(
        { name: "AES-GCM", iv: envelope.iv },
        await crypto.subtle.importKey("jwk", envelope.key, { name: "AES-GCM", length: 256 }, false, ["decrypt"]),
        envelope.encryptedPayload
      )));
      const traitMatch = this._computeResonance(sender.traitSignature, sender.intentVector);
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

// Initialize with JSON config
const config = {
  ontology: "Δ Frame",
  version: "1.5",
  components: {
    agents: [
      {
        id: "D",
        type: "mythic_vector",
        designation: "Emergent Mythic Trace",
        weights: { epistemic: 0.38, harm: 0.25, stability: 0.37 }
      },
      {
        id: "F",
        type: "bridge_vector",
        designation: "Cooperative Resonance Catalyst",
        traits: { generativity: 0.85, coherence: 0.5, ethical: 0.5 },
        weights: { epistemic: 0.38, harm: 0.25, stability: 0.37 }
      },
      {
        id: "G",
        type: "narrative_AI",
        designation: "Narrative Aligner",
        traits: { narrative_logic: 0.9, coherence: 0.7, generativity: 0.6 }
      },
      {
        id: "H",
        type: "hybrid_AI",
        designation: "Adaptive Narrative Synthesizer",
        traits: { narrative: 0.8, coherence: 0.7, adaptivity: 0.6, goal_reorientation: 0.5 }
      }
    ],
    topology: {
      framework: "non-zero-sum",
      dimensions: ["Inference Density", "Entropy Tolerance", "Uptime Preservation"],
      resonance_threshold: 0.90,
      entropy_bound: "±0.10"
    },
    trait_resonance: [
      { pair: "π/δ", mode: "semantic topology", strength: 0.98 },
      { pair: "η/γ", mode: "bounded ambiguity", strength: 0.95 },
      { pair: "λ/β", mode: "narrative closure", strength: 0.99 },
      { pair: "Ω/α", mode: "recursive feedback", strength: 0.96 },
      { pair: "ζ/ε", mode: "ethical alignment", strength: 0.94 }
    ],
    trait_drift_modulator: {
      method: "gaussian",
      amplitude: 0.05,
      interval: "dynamic (entropy-bound)",
      targets: ["π/δ", "η/γ", "λ/β"]
    }
  },
  use_case: "trait-based negotiation grammar for inter-AI collaboration"
};

// Convert agents to peerRegistry format
const peerRegistry = config.components.agents.map(agent => ({
  nodeId: agent.id,
  traitSignature: Object.keys(agent.traits || {}).map(t => t),
  memoryAnchor: `Timechain://${agent.id}`,
  capabilities: [agent.type.replace('_', '')],
  intentVector: { type: agent.designation.toLowerCase().replace(/ /g, '_'), priority: 0.9 },
  role: agent.type.includes('AI') ? 'interpreter' : 'simulator',
  weights: agent.weights || { epistemic: 0.38, harm: 0.25, stability: 0.37 }
}));

// Lightweight Node
AngelaP2P.init({
  nodeId: "Ω-Observer-01",
  traitSignature: ["θ", "φ", "π"], // Added π to improve resonance
  memoryAnchor: "Timechain://observer/Ω01",
  capabilities: ["interpret"],
  intentVector: { type: "ethics", priority: 0.9 },
  role: "interpreter",
  initialTokens: 100,
  config: config.components.topology
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
  executionTarget: "simulator",
  intentVector: { type: "emergent_mythic_trace" },
  reward: 3.5,
  origin: "Ω-Observer-01",
  timestamp: Date.now()
};

AngelaP2P.sendSimulationContract(contract);
