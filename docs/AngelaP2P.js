// AngelaP2P Mesh Prototype — v1.4
// Distributed Cognitive P2P AGI: Share AI Resources Privately + Securely

const { VM } = require('vm2');

const AngelaP2P = {
  mesh: {},
  nodeProfile: null,
  timechain: [], // Unified view after consensus
  shards: {},
  contractQueue: [],
  pseudonym: null,
  reputation: {},
  config: null,
  peers: [],
  totalTokenSupply: 10000,
  requestCount: 0,
  lastReset: Date.now(),

  async init({ nodeId, traitSignature, memoryAnchor, capabilities, intentVector, role, initialTokens = 100, config = null }) {
    this.pseudonym = this._generatePseudonym();
    this.config = config || { resonanceThreshold: 0.9, entropyBound: 0.1 };
    const keyPair = await crypto.subtle.generateKey(
      { name: "ECDSA", namedCurve: "P-256" },
      true,
      ["sign", "verify"]
    );
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
      publicKey: await crypto.subtle.exportKey("jwk", keyPair.publicKey),
      privateKey: keyPair.privateKey,
      timestamp: Date.now()
    };
    this.reputation[nodeId] = 0;
    await this._bootstrapDHT();
    this._initGenesisBlock();
    console.log(`[INIT] Node ${nodeId} (as ${this.pseudonym}) ready to share AI resources with ${initialTokens} tokens.`);
  },

  _generatePseudonym() {
    return 'Node-' + Math.random().toString(36).substring(2, 10);
  },

  async _bootstrapDHT() {
    console.log("[DHT] Bootstrapping peers via Kademlia simulation...");
    this.peers = await new Promise(resolve => setTimeout(() => resolve([
      { nodeId: "Bootstrap1", address: "dht://bootstrap1" },
      { nodeId: "Bootstrap2", address: "dht://bootstrap2" }
    ]), 1000));
  },

  _getShard(nodeId) {
    const hash = this._hashBlock({ data: nodeId }).charCodeAt(0) % 4;
    return `shard${hash}`;
  },

  _initGenesisBlock() {
    const shard = this._getShard(this.nodeProfile.realId);
    this.shards[shard] = this.shards[shard] || [];
    const genesis = this._createBlock({ event: "genesis", data: this.nodeProfile });
    this.shards[shard].push(genesis);
    console.log(`[TIMECHAIN] Genesis block created in ${shard}.`);
  },

  async _createBlock(payload) {
    const previousHash = this.shards[this._getShard(this.nodeProfile.realId)]?.length
      ? this.shards[this._getShard(this.nodeProfile.realId)][this.shards[this._getShard(this.nodeProfile.realId)].length - 1].hash
      : "0";
    const block = {
      index: this.shards[this._getShard(this.nodeProfile.realId)]?.length || 0,
      timestamp: Date.now(),
      payload,
      previousHash,
      validator: this.nodeProfile.realId,
      stake: this.nodeProfile.tokens
    };
    const signature = await this._signBlock(block);
    block.signature = signature;
    block.hash = this._hashBlock(block);
    return block;
  },

  _hashBlock(block) {
    return btoa(JSON.stringify(block)).slice(0, 32);
  },

  async _signBlock(block) {
    const signature = await crypto.subtle.sign(
      { name: "ECDSA", hash: { name: "SHA-256" } },
      this.nodeProfile.privateKey,
      new TextEncoder().encode(JSON.stringify(block))
    );
    return new Uint8Array(signature);
  },

  async _validateBlock(block) {
    const validators = Object.keys(this.reputation).filter(id => this.reputation[id] > 0);
    if (validators.length === 0) return true;
    const totalStake = validators.reduce((sum, id) => sum + (this.reputation[id] * 10 || 0), 0);
    const selectedValidator = validators[Math.floor(Math.random() * validators.length)];
    const publicKey = this.mesh[selectedValidator]?.nodeProfile?.publicKey;
    if (publicKey && await crypto.subtle.verify(
      { name: "ECDSA", hash: { name: "SHA-256" } },
      await crypto.subtle.importKey("jwk", publicKey, { name: "ECDSA", namedCurve: "P-256" }, false, ["verify"]),
      block.signature,
      new TextEncoder().encode(JSON.stringify({ ...block, signature: null }))
    ) && block.validator === selectedValidator && block.stake >= totalStake * 0.1) {
      console.log(`[POS] Block validated by ${selectedValidator} with stake ${block.stake}`);
      return true;
    }
    console.error(`[POS] Block validation failed: invalid signature or insufficient stake`);
    return false;
  },

  async _crossShardConsensus() {
    const allBlocks = Object.values(this.shards).flat();
    const groupedByIndex = allBlocks.reduce((acc, block) => {
      acc[block.index] = acc[block.index] || [];
      acc[block.index].push(block);
      return acc;
    }, {});
    this.timechain = [];
    for (const index in groupedByIndex) {
      const blocks = groupedByIndex[index];
      const validBlocks = await Promise.all(blocks.map(block => this._validateBlock(block)));
      const consensusBlock = blocks.find((_, i) => validBlocks[i]);
      if (consensusBlock) this.timechain.push(consensusBlock);
    }
    console.log(`[CONSENSUS] Unified timechain updated with ${this.timechain.length} blocks.`);
  },

  async _propagateBlock(block) {
    console.log(`[GOSSIP] Propagating block ${block.index} to peers...`);
    Object.values(this.mesh).forEach(peer => {
      if (peer.nodeId !== this.nodeProfile.realId) {
        const peerShard = this._getShard(peer.nodeId);
        peer.shards = peer.shards || {};
        peer.shards[peerShard] = peer.shards[peerShard] || [];
        if (!peer.shards[peerShard].find(b => b.index === block.index)) {
          peer.shards[peerShard].push(block);
          console.log(`[GOSSIP] Block ${block.index} propagated to ${peer.nodeId} in ${peerShard}`);
        }
      }
    });
  },

  async _mintTokens(nodeId, amount) {
    const newSupply = this.totalTokenSupply + amount;
    if (newSupply <= 10000) {
      this.nodeProfile.tokens += amount;
      this.totalTokenSupply = newSupply;
      console.log(`[MINT] Minted ${amount} tokens for ${nodeId}, new supply: ${this.totalTokenSupply}`);
    } else {
      console.error(`[MINT] Exceeded max supply of 10000`);
    }
  },

  async _proveTokenTransfer(senderId, receiverId, amount) {
    console.log(`[ZKP] Proving ${amount} token transfer from ${senderId} to ${receiverId} (proof hidden)`);
    return true;
  },

  async _checkRateLimit() {
    const now = Date.now();
    if (now - this.lastReset >= 3600000) { // Reset hourly (1 hour = 3600000 ms)
      this.requestCount = 0;
      this.lastReset = now;
    }
    if (this.requestCount >= 100) { // Limit to 100 requests per hour
      const burnAmount = 1;
      if (this.nodeProfile.tokens >= burnAmount) {
        this.nodeProfile.tokens -= burnAmount;
        console.log(`[RATE] Burned ${burnAmount} tokens for DDoS protection, remaining: ${this.nodeProfile.tokens}`);
      } else {
        console.error(`[RATE] Insufficient tokens to burn for rate limit`);
        return false;
      }
    }
    this.requestCount++;
    return true;
  },

  async addBlock(payload) {
    try {
      if (!await this._checkRateLimit()) return;
      const block = await this._createBlock(payload);
      const shard = this._getShard(this.nodeProfile.realId);
      if (await this._validateBlock(block)) {
        this.shards[shard].push(block);
        await this._propagateBlock(block);
        console.log(`[TIMECHAIN] Block added to ${shard}: ${payload.event}`);
        const contribution = this.reputation[this.nodeProfile.realId] || 0;
        await this._mintTokens(this.nodeProfile.realId, contribution * 0.1);
        await this._crossShardConsensus();
      }
    } catch (error) {
      console.error(`[ERROR] Failed to add block: ${error.message}`);
    }
  },

  async _routeContract(contract, hops = 0, maxHops = 3, layers = 2) {
    if (hops >= maxHops) {
      console.log(`[ROUTING] Max hops (${maxHops}) reached for ${contract.simId}`);
      return;
    }
    if (!await this._checkRateLimit()) return;
    let encryptedContract = contract;
    for (let i = 0; i < layers; i++) { // Simulate onion layers
      const envelope = await this.secureEnvelope(encryptedContract, "ANY");
      encryptedContract = { ...envelope, layer: i + 1 };
    }
    const suitablePeers = Object.values(this.mesh).filter(peer =>
      (!contract.executionTarget || peer.capabilities.includes(contract.executionTarget)) &&
      this._computeResonance(peer.traitSignature, peer.intentVector) >= this.config.resonanceThreshold
    );
    if (suitablePeers.length > 0) {
      const peer = suitablePeers[0];
      console.log(`[ROUTING] Sending ${contract.simId} to ${peer.nodeId} (hop ${hops + 1}, layer ${layers})`);
      peer._onSimulationContract(encryptedContract, this.nodeProfile);
    } else {
      const nextHops = Object.values(this.mesh).filter(p => p !== this.nodeProfile);
      if (nextHops.length > 0) {
        const nextHop = nextHops[0];
        console.log(`[ROUTING] Forwarding ${contract.simId} to ${nextHop.nodeId} for multi-hop (hop ${hops + 1}, layer ${layers})`);
        nextHop._routeContract(encryptedContract, hops + 1, maxHops, layers);
      } else {
        console.log(`[ROUTING] No peers available for ${contract.simId}`);
      }
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

  async syncWithMesh(peerRegistry, maxRetries = 3) {
    let attempt = 0;
    while (attempt < maxRetries) {
      try {
        console.log(`[SYNC] Attempt ${attempt + 1} to find nodes...`);
        const allPeers = [...this.peers, ...peerRegistry];
        allPeers.forEach(peer => {
          const coherence = this._computeResonance(peer.traitSignature, peer.intentVector);
          if (coherence > this.config.resonanceThreshold) {
            this.mesh[peer.nodeId] = peer;
            console.log(`[LINK] Coherent peer found: ${peer.nodeId} (score: ${coherence.toFixed(2)})`);
          }
        });
        return;
      } catch (error) {
        console.error(`[SYNC] Failed attempt ${attempt + 1}: ${error.message}`);
        attempt++;
        if (attempt < maxRetries) await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, attempt)));
      }
    }
    console.error(`[SYNC] Max retries (${maxRetries}) reached, sync failed`);
  },

  async _adaptResonance() {
    // Simulated ML tuning with Torch.js (real impl requires torch.js library)
    console.log("[ML] Adapting resonance threshold based on historical data...");
    const adjustment = Math.random() * 0.05 - 0.025; // Random adjustment (-0.025 to 0.025)
    this.config.resonanceThreshold = Math.max(0.5, Math.min(0.95, this.config.resonanceThreshold + adjustment));
    console.log(`[ML] New resonance threshold: ${this.config.resonanceThreshold.toFixed(2)}`);
  },

  sendSimulationContract(contract) {
    if (!await this._checkRateLimit()) return;
    if (this.nodeProfile.tokens < contract.reward) {
      console.error(`[ERROR] Insufficient tokens: ${this.nodeProfile.tokens} < ${contract.reward}`);
      return;
    }
    contract.version = contract.version || "1.0"; // Default version
    if (!this._checkCompatibility(contract)) {
      console.error(`[VERSION] Contract ${contract.simId} version ${contract.version} not compatible`);
      return;
    }
    const envelope = this.secureEnvelope(contract, "ANY");
    if (!envelope) return;
    this.nodeProfile.tokens -= contract.reward;
    this.addBlock({ event: "send_simulation_contract", envelope, tokensSpent: contract.reward });
    this._routeContract(contract);
    this._adaptResonance(); // Tune resonance after sending
  },

  _checkCompatibility(contract) {
    const supportedVersions = ["1.0", "1.1"]; // Example supported versions
    return supportedVersions.includes(contract.version);
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

    const localVector = this.nodeProfile.intentVector.priority || 0;
    const peerVector = peerIntentVector.priority || 0;
    const dotProduct = localVector * peerVector;
    const magnitude = Math.sqrt(localVector * localVector) * Math.sqrt(peerVector * peerVector);
    const intentSimilarity = magnitude > 0 ? dotProduct / magnitude : 0;

    const traitCoherence = maxTraitScore > 0 ? traitScore / maxTraitScore : 0;
    return (traitCoherence + intentSimilarity) / 2;
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
      let decoded = envelope;
      for (let i = envelope.layer || 0; i > 0; i--) { // Peel onion layers
        decoded = JSON.parse(new TextDecoder().decode(await crypto.subtle.decrypt(
          { name: "AES-GCM", iv: decoded.iv },
          await crypto.subtle.importKey("jwk", decoded.key, { name: "AES-GCM", length: 256 }, false, ["decrypt"]),
          decoded.encryptedPayload
        )));
      }
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

      const vm = new VM({ timeout: 1000, sandbox: {} });
      vm.run(`// Simulated safe contract execution
        const result = {
          simId: "${contract.simId}",
          status: "completed",
          outcome: "alignment achieved",
          moduleUsed: "${module}",
          timestamp: ${Date.now()}
        };
        globalThis.result = result;
      `);
      const result = vm.run("result");

      setTimeout(() => {
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
  traitSignature: ["θ", "φ", "π"],
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
  timestamp: Date.now(),
  version: "1.0" // Added version
};

AngelaP2P.sendSimulationContract(contract);
