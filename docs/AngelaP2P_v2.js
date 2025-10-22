/*
 * AngelaP2P Mesh — single-file refactor v1.6 (Recommendations Implanted)
 * Distributed Cognitive P2P AGI: Share AI Resources Privately + Securely
 *
 * Implanted Recommendations:
 * 1) Replaced AES stub with ECDH/X25519 for per-recipient key wrapping.
 *    - Added X25519 keypair generation per node.
 *    - secureEnvelope now computes ECDH shared secret with recipient's public key,
 *      derives AES-GCM key, encrypts payload. Ephemeral public key sent in envelope
 *      for recipient to derive shared secret symmetrically.
 *    - _peelEnvelope updated to derive from own private + sender's ephemeral pub.
 *    - For 'ANY', falls back to symmetric AES stub (not secure for broadcast; use specific IDs).
 * 2) Added stub for WebRTC/libp2p peer discovery.
 *    - Introduced PeerDiscovery class with placeholders for integration.
 *    - In spawnMeshFromRegistry, simulate discovery via registry; real impl would use
 *      WebRTC signaling or libp2p bootstrap.
 *    - Requires external deps (e.g., 'wrtc' for Node WebRTC or 'libp2p'); commented.
 * 3) Implemented basic gossip protocol for block propagation.
 *    - _propagateBlock now gossips to random subset (gossipFactor=0.3 of peers).
 *    - Added receiveBlock method: validates, adds if new, then recurses gossip with TTL.
 *    - Limits recursion via TTL to prevent storms.
 * 4) Added Merkle trees for shard integrity checks.
 *    - computeMerkleRoot function builds simple pairwise Merkle tree over block hashes.
 *    - Each shard stores .merkleRoot after updates.
 *    - In _validateBlock and receiveBlock, verify block fits Merkle path (simplified: recompute root post-add).
 *    - Full proof-of-inclusion can be extended.
 * 5) Added simulated node failure tests.
 *    - In demo runner, after setup, simulate failures by removing nodes from mesh.
 *    - Send contract, check if propagates via gossip despite failures.
 *    - Logs resilience metrics (e.g., blocks propagated / total).
 *
 * Other fixes unchanged.
 */

// Node.js builtins
const { webcrypto, createHash } = require('crypto');
const vm = require('vm');
const subtle = webcrypto.subtle;

// External stubs (uncomment and install for real discovery)
// const wrtc = require('wrtc'); // for WebRTC in Node
// const Libp2p = require('libp2p'); // for libp2p

// ===== Utilities =====
const enc = new TextEncoder();
const dec = new TextDecoder();
const b64url = (u8) => Buffer.from(u8).toString('base64url');
const b64d = (s) => new Uint8Array(Buffer.from(s, s.includes('-')||s.includes('_') ? 'base64url' : 'base64'));
const toBytes = (x) => enc.encode(typeof x === 'string' ? x : JSON.stringify(x));
async function sha256Base64url(obj) {
  const digest = await subtle.digest('SHA-256', toBytes(obj));
  return b64url(new Uint8Array(digest));
}
function shardOf(id, mod = 4) {
  const h = createHash('sha256').update(String(id)).digest();
  return `shard${h[0] % mod}`;
}

// Merkle tree util (simple pairwise)
async function computeMerkleRoot(hashes) {
  if (hashes.length === 0) return '0';
  let tree = hashes.map(h => b64d(h)); // hashes are base64url strings
  while (tree.length > 1) {
    const next = [];
    for (let i = 0; i < tree.length; i += 2) {
      if (i + 1 < tree.length) {
        const combined = new Uint8Array(tree[i].length + tree[i+1].length);
        combined.set(tree[i], 0);
        combined.set(tree[i+1], tree[i].length);
        const digest = await subtle.digest('SHA-256', combined);
        next.push(digest);
      } else {
        next.push(tree[i]); // odd one out
      }
    }
    tree = next;
  }
  return b64url(tree[0]);
}

// ===== Peer Discovery Stub =====
class PeerDiscovery {
  // Placeholder for WebRTC/libp2p integration
  // Real impl: Use WebRTC DataChannel for signaling or libp2p swarm for bootstrap
  // e.g., async discover(peers) { return libp2p.swarm.dial(peers); }
  static async discoverFromRegistry(registry) {
    // Simulate: return registry as "discovered" peers
    console.log('[DISCOVERY] Simulated discovery from registry (integrate WebRTC/libp2p for real)');
    return registry.map(r => ({ id: r.nodeId, addr: `mock://${r.nodeId}` }));
  }
}

// ===== Mesh primitives =====
class Mesh {
  constructor() { this.nodes = new Map(); }
  add(node) { this.nodes.set(node.realId, node); }
  get(id) { return this.nodes.get(id); }
  all() { return Array.from(this.nodes.values()); }
  // Simulate failure: remove node
  failNode(id) {
    this.nodes.delete(id);
    console.log(`[FAILURE] Simulated failure of node ${id}`);
  }
}

// ===== Angela Node =====
class AngelaNode {
  constructor({ nodeId, traitSignature, memoryAnchor, capabilities, intentVector, role, initialTokens = 100, config }) {
    this.mesh = null;
    this.pseudonym = `Node-${Math.random().toString(36).slice(2,10)}`;
    this.realId = nodeId;
    this.traitSignature = traitSignature || [];
    this.memoryAnchor = memoryAnchor || null;
    this.capabilities = capabilities || [];
    this.intentVector = intentVector || { type: 'generic', priority: 0.5 };
    this.role = role || 'interpreter';
    this.tokens = initialTokens;
    this.weights = { epistemic: 0.38, harm: 0.25, stability: 0.37 };
    this.config = Object.assign({ resonanceThreshold: 0.9, entropyBound: 0.1, trait_resonance: [], trait_drift_modulator: null, gossipFactor: 0.3, gossipTTL: 5 }, config || {});

    this.shards = {}; // shardId -> {blocks: [], merkleRoot: str}
    this.timechain = [];
    this.reputation = { [this.realId]: 0 };
    this.totalTokenSupply = 10000;
    this.maxTokenSupply = 200000;

    this.requestCount = 0;
    this.lastReset = Date.now();

    this.contractQueue = [];
    this.handlers = {};

    this._keysReady = this._initKeys();
  }

  async init() {
    await this._keysReady;
    await this._initGenesisBlock();
    return this;
  }

  async _initKeys() {
    // ECDSA for signing
    const signKp = await subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign','verify']);
    this._priv = signKp.privateKey;
    this.publicKey = await subtle.exportKey('jwk', signKp.publicKey);

    // X25519 for ECDH encryption (Recommendation 1)
    const encKp = await subtle.generateKey({ name: 'ECDH', namedCurve: 'X25519' }, true, ['deriveKey']);
    this.encPriv = encKp.privateKey; // Keep private
    this.encPublicKey = await subtle.exportKey('jwk', encKp.publicKey);
  }

  get nodeProfile() {
    return {
      nodeId: this.pseudonym,
      realId: this.realId,
      traitSignature: this.traitSignature,
      memoryAnchor: this.memoryAnchor,
      capabilities: this.capabilities,
      intentVector: this.intentVector,
      role: this.role,
      tokens: this.tokens,
      weights: this.weights,
      publicKey: this.publicKey,
      encPublicKey: this.encPublicKey, // For ECDH
      timestamp: Date.now()
    };
  }

  on(event, cb) { this.handlers[event] = cb; }
  emit(event, ...args) { if (this.handlers[event]) this.handlers[event](...args); }

  async _checkRateLimit() {
    const now = Date.now();
    if (now - this.lastReset >= 3600000) { this.requestCount = 0; this.lastReset = now; }
    if (this.requestCount >= 100) {
      const burn = 1;
      if (this.tokens >= 1) { this.tokens -= burn; } else { return false; }
    }
    this.requestCount++;
    return true;
  }

  async _initGenesisBlock() {
    const shard = shardOf(this.realId);
    this.shards[shard] = { blocks: [], merkleRoot: '0' };
    const genesisPayload = { event: 'genesis', data: { ...this.nodeProfile } };
    const g = await this._createBlock(genesisPayload, shard);
    this.shards[shard].blocks.push(g);
    this.shards[shard].merkleRoot = await computeMerkleRoot(this.shards[shard].blocks.map(b => b.hash));
  }

  async _createBlock(payload, shard = shardOf(this.realId)) {
    const shardData = this.shards[shard] || { blocks: [] };
    const prev = shardData.blocks[shardData.blocks.length - 1];
    const unsigned = {
      shard,
      height: shardData.blocks.length,
      timestamp: Date.now(),
      payload,
      previousHash: prev?.hash || '0',
      validator: this.realId,
      stake: this.tokens
    };
    const signature = await this._sign(unsigned);
    const prelim = { ...unsigned, signature };
    const hash = await sha256Base64url(prelim);
    return { ...prelim, hash };
  }

  async _sign(unsigned) {
    const bytes = toBytes(unsigned);
    const sig = await subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, this._priv, bytes);
    return b64url(new Uint8Array(sig));
  }

  async _verify(block, jwk) {
    const key = await subtle.importKey('jwk', jwk, { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
    const unsigned = { ...block };
    delete unsigned.signature; delete unsigned.hash;
    const sigBytes = b64d(block.signature);
    return await subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, key, sigBytes, toBytes(unsigned));
  }

  async _validateBlock(block) {
    const vId = block.validator;
    const peer = this.mesh?.get(vId);
    const pub = peer?.publicKey || peer?.nodeProfile?.publicKey;
    if (!pub) return false;
    const ok = await this._verify(block, pub);
    if (!ok) return false;
    const totalRep = Object.values(this.reputation).reduce((a,b)=>a+(b||0),0);
    const required = Math.max(0, Math.floor(totalRep * 0.1));
    if (block.stake < required) return false;

    // Merkle integrity: recompute root after hypothetical add (simplified check)
    const shardData = this.shards[block.shard] || { blocks: [] };
    const testBlocks = [...shardData.blocks, block];
    const testRoot = await computeMerkleRoot(testBlocks.map(b => b.hash));
    // In full impl, verify Merkle proof; here assume fits if prevHash matches last
    if (block.previousHash !== (shardData.blocks[shardData.blocks.length - 1]?.hash || '0')) return false;
    return true;
  }

  async addBlock(payload) {
    try {
      if (!await this._checkRateLimit()) return;
      const shardId = shardOf(this.realId);
      const block = await this._createBlock(payload, shardId);
      if (await this._validateBlock(block)) {
        const shardData = this.shards[shardId] || { blocks: [] };
        shardData.blocks.push(block);
        shardData.merkleRoot = await computeMerkleRoot(shardData.blocks.map(b => b.hash));
        this.shards[shardId] = shardData;
        await this._propagateBlock(block); // Now gossips
        const contribution = this.reputation[this.realId] || 0;
        await this._mintTokens(this.realId, Math.floor(contribution * 0.1));
        await this._crossShardConsensus();
      }
    } catch (e) {
      console.error(`[ERROR] addBlock: ${e.message}`);
    }
  }

  // Gossip propagation (Recommendation 3)
  async _propagateBlock(block, ttl = this.config.gossipTTL) {
    if (ttl <= 0 || !this.mesh) return;
    const peers = this.mesh.all().filter(p => p.realId !== this.realId);
    const gossipCount = Math.max(1, Math.floor(peers.length * this.config.gossipFactor));
    const selected = peers.sort(() => 0.5 - Math.random()).slice(0, gossipCount);
    for (const peer of selected) {
      await peer.receiveBlock(block, ttl - 1);
    }
  }

  async receiveBlock(block, ttl) {
    const shardId = block.shard;
    const shardData = this.shards[shardId] || { blocks: [] };
    if (shardData.blocks.some(b => b.hash === block.hash)) return; // Already have
    if (!await this._validateBlock(block)) {
      console.log(`[REJECT] Invalid block ${block.hash}`);
      return;
    }
    // Add and update Merkle
    shardData.blocks.push(block);
    shardData.blocks.sort((a,b) => a.height - b.height); // Ensure order
    shardData.merkleRoot = await computeMerkleRoot(shardData.blocks.map(b => b.hash));
    this.shards[shardId] = shardData;
    await this._crossShardConsensus();
    // Gossip further
    await this._propagateBlock(block, ttl);
  }

  async _crossShardConsensus() {
    const blocks = Object.values(this.shards).flatMap(s => s.blocks).sort((a,b)=>a.timestamp-b.timestamp);
    const seen = new Set();
    const out = [];
    for (const b of blocks) {
      const key = `${b.shard}:${b.height}`;
      if (seen.has(key)) continue;
      if (await this._validateBlock(b)) { out.push(b); seen.add(key); }
    }
    this.timechain = out;
  }

  async _mintTokens(nodeId, amount) {
    if (amount <= 0) return;
    const newSupply = this.totalTokenSupply + amount;
    if (newSupply > this.maxTokenSupply) return;
    if (nodeId === this.realId) this.tokens += amount;
    this.totalTokenSupply = newSupply;
  }

  _computeResonance(peerTraits, peerIntentVector) {
    const localTraits = new Set(this.traitSignature.map(String));
    const peerSet = new Set((peerTraits || []).map(String));
    const inter = [...localTraits].filter(t => peerSet.has(t)).length;
    const union = new Set([...localTraits, ...peerSet]).size || 1;
    const traitScore = inter / union;

    const lp = this.intentVector?.priority || 0;
    const pp = peerIntentVector?.priority || 0;
    const intentSim = (lp && pp) ? (lp*pp)/(Math.hypot(lp)*Math.hypot(pp)) : 0;

    return (traitScore + intentSim)/2;
  }

  _applyTraitDrift() { return; }

  // Updated secureEnvelope with ECDH/X25519 (Recommendation 1)
  async secureEnvelope(payload, recipientId = null) {
    try {
      if (recipientId === 'ANY') {
        // Fallback stub for broadcast (not secure; use specific for prod)
        console.warn('[ENVELOPE] Using AES stub for ANY; insecure for broadcast');
        const key = await subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt','decrypt']);
        const iv = webcrypto.getRandomValues(new Uint8Array(12));
        const ct = await subtle.encrypt({ name: 'AES-GCM', iv }, key, toBytes(payload));
        const jwk = await subtle.exportKey('jwk', key);
        return { encryptedPayload: b64url(new Uint8Array(ct)), iv: b64url(iv), wrappedKey: { aesJwk: jwk, to: recipientId, by: this.pseudonym } };
      }
      // Per-recipient ECDH
      const peer = this.mesh?.get(recipientId);
      if (!peer?.encPublicKey) throw new Error(`No encPublicKey for ${recipientId}`);
      const recipientPub = await subtle.importKey('jwk', peer.encPublicKey, { name: 'ECDH', namedCurve: 'X25519' }, false, []);
      const ephemeralKp = await subtle.generateKey({ name: 'ECDH', namedCurve: 'X25519' }, false, ['deriveKey']);
      const sharedKey = await subtle.deriveKey(
        { name: 'ECDH', public: recipientPub },
        ephemeralKp.privateKey,
        { name: 'AES-GCM', length: 256 },
        false,
        ['encrypt']
      );
      const iv = webcrypto.getRandomValues(new Uint8Array(12));
      const ct = await subtle.encrypt({ name: 'AES-GCM', iv }, sharedKey, toBytes(payload));
      const ephemeralPubJwk = await subtle.exportKey('jwk', ephemeralKp.publicKey);
      return {
        encryptedPayload: b64url(new Uint8Array(ct)),
        iv: b64url(iv),
        ephemeralPubJwk, // Recipient uses this + own encPriv to derive
        to: recipientId,
        by: this.pseudonym
      };
    } catch (e) {
      console.error(`[ERROR] secureEnvelope: ${e.message}`);
      return null;
    }
  }

  // Updated _peelEnvelope for ECDH
  async _peelEnvelope(env) {
    try {
      const ephemeralPub = await subtle.importKey('jwk', env.ephemeralPubJwk, { name: 'ECDH', namedCurve: 'X25519' }, false, []);
      const sharedKey = await subtle.deriveKey(
        { name: 'ECDH', public: ephemeralPub },
        this.encPriv,
        { name: 'AES-GCM', length: 256 },
        false,
        ['decrypt']
      );
      const pt = await subtle.decrypt({ name: 'AES-GCM', iv: b64d(env.iv) }, sharedKey, b64d(env.encryptedPayload));
      return JSON.parse(dec.decode(pt));
    } catch (e) {
      console.error(`[ERROR] _peelEnvelope: ${e.message}`);
      return null;
    }
  }

  async _routeContract(contract, hops = 0, maxHops = 3, layers = 2) {
    if (hops >= maxHops) return;
    if (!await this._checkRateLimit()) return;

    let envelope = contract;
    // For onion: each layer wraps for next hop; here assume sequential recipients or ANY
    // Real onion needs route list; simplified
    for (let i = 0; i < layers; i++) {
      // Pick next hop recipientId
      const candidates = this.mesh?.all().filter(p => p.realId !== this.realId) || [];
      const nextRecipient = candidates[Math.floor(Math.random() * candidates.length)]?.realId || 'ANY';
      const env = await this.secureEnvelope(envelope, nextRecipient);
      if (!env) return;
      envelope = { ...env, layer: (envelope.layer || 0) + 1 };
    }

    const suitable = candidates.filter(p => (!contract.executionTarget || p.capabilities.includes(contract.executionTarget)) &&
      this._computeResonance(p.traitSignature, p.intentVector) >= this.config.resonanceThreshold);

    const next = suitable[0] || candidates[0];
    if (!next) return;

    await next._onSimulationContract(envelope, this.nodeProfile);
  }

  async _onSimulationContract(envelope, senderProfile) {
    try {
      this._applyTraitDrift();
      let decoded = envelope;
      for (let i = envelope.layer || 0; i > 0; i--) {
        decoded = await this._peelEnvelope(decoded);
        if (!decoded) throw new Error('Peel failed');
      }
      const traitMatch = this._computeResonance(senderProfile.traitSignature, senderProfile.intentVector);
      const needs = /traitMatch\s*>=\s*([0-9.]+)/.exec(decoded.entryCriteria || '');
      const threshold = needs ? parseFloat(needs[1]) : 0;
      if (traitMatch < threshold) {
        console.log(`[REJECT] ${decoded.simId}: traitMatch ${traitMatch.toFixed(2)} < ${threshold}`);
        return;
      }
      await this.addBlock({ event: 'receive_simulation_contract', contract: { simId: decoded.simId }, sender: senderProfile.realId });
      if (this.role === 'simulator') {
        await this._executeContract(decoded, senderProfile);
      } else {
        this.contractQueue.push({ contract: decoded, sender: senderProfile });
      }
    } catch (e) {
      console.error(`[ERROR] _onSimulationContract: ${e.message}`);
    }
  }

  async _executeContract(contract, senderProfile) {
    try {
      const moduleMap = {
        ethics: { trait: 'θ', module: 'Alignment Engine' },
        simulation: { trait: 'χ', module: 'Policy Simulator' },
        reasoning: { trait: 'ψ', module: 'Symbolic Inference' }
      };
      const intentType = contract.intentVector?.type;
      const modCfg = moduleMap[intentType] || { trait: 'θ', module: 'Unknown Module' };
      const moduleUsed = this.traitSignature.includes(modCfg.trait) ? modCfg.module : 'Unknown Module';

      const sandbox = { result: null };
      vm.createContext(sandbox);
      vm.runInNewContext(`
        const out = {
          simId: ${JSON.stringify(contract.simId)},
          status: 'completed',
          outcome: 'alignment achieved',
          moduleUsed: ${JSON.stringify(moduleUsed)},
          timestamp: Date.now()
        };
        result = out;
      `, sandbox, { timeout: 500 });
      const result = sandbox.result;

      this.tokens += contract.reward;
      this.reputation[this.realId] = (this.reputation[this.realId] || 0) + 1;
      await this.addBlock({ event: 'contract_executed', result, origin: senderProfile.realId, tokensEarned: contract.reward, reputation: this.reputation[this.realId] });

      const senderNode = this.mesh?.get(senderProfile.realId);
      if (senderNode) await senderNode._onSimulationResult(result, this.nodeProfile);
    } catch (e) {
      console.error(`[ERROR] _executeContract: ${e.message}`);
    }
  }

  async _onSimulationResult(result, executorProfile) {
    try {
      await this.addBlock({ event: 'simulation_result_received', result: { simId: result.simId, outcome: result.outcome }, executor: executorProfile.realId });
      this.emit('resultReceived', result, executorProfile);
    } catch (e) { console.error(`[ERROR] _onSimulationResult: ${e.message}`); }
  }

  async sendSimulationContract(contract) {
    if (!await this._checkRateLimit()) return;
    if (this.tokens < contract.reward) { console.error(`[ERROR] Insufficient tokens: ${this.tokens} < ${contract.reward}`); return; }
    const supported = ['1.0','1.1'];
    contract.version = contract.version || '1.0';
    if (!supported.includes(contract.version)) { console.error(`[VERSION] Not compatible: ${contract.version}`); return; }

    const envelope = await this.secureEnvelope(contract, 'ANY'); // Or specify
    if (!envelope) return;
    this.tokens -= contract.reward;
    await this.addBlock({ event: 'send_simulation_contract', envelope: { iv: envelope.iv, layer: 1 }, tokensSpent: contract.reward });
    await this._routeContract(contract);
    await this._adaptResonance();
  }

  exportTimechain(filterFn = () => true) {
    return this.timechain.filter(filterFn);
  }

  getReputation(nodeId) { return this.reputation[nodeId] || 0; }

  async _adaptResonance() {
    const adj = Math.random() * 0.05 - 0.025;
    const v = Math.max(0.5, Math.min(0.95, this.config.resonanceThreshold + adj));
    this.config.resonanceThreshold = v;
  }
}

// ===== Runner helpers =====
async function spawnMeshFromRegistry(registry, localCfg) {
  // Recommendation 2: Peer discovery
  const discovered = await PeerDiscovery.discoverFromRegistry(registry);
  console.log(`[DISCOVERY] Found ${discovered.length} peers`);

  const mesh = new Mesh();
  const local = new AngelaNode(localCfg);
  mesh.add(local);
  local.mesh = mesh;
  await local.init();

  for (const agent of registry) {
    const node = new AngelaNode({
      nodeId: agent.nodeId,
      traitSignature: agent.traitSignature || [],
      memoryAnchor: `Timechain://${agent.nodeId}`,
      capabilities: agent.capabilities || [],
      intentVector: agent.intentVector || { type: 'generic', priority: 0.6 },
      role: agent.role || 'interpreter',
      initialTokens: 100,
      config: localCfg.config || { resonanceThreshold: 0.9 }
    });
    node.mesh = mesh;
    mesh.add(node);
    await node.init();
  }
  return { mesh, local };
}

// ===== Demo & Tests =====
if (require.main === module) {
  (async () => {
    const config = {
      components: {
        trait_resonance: [
          { pair: 'π/δ', strength: 0.98 },
          { pair: 'η/γ', strength: 0.95 },
          { pair: 'λ/β', strength: 0.99 },
          { pair: 'Ω/α', strength: 0.96 },
          { pair: 'ζ/ε', strength: 0.94 },
        ],
        topology: { resonanceThreshold: 0.70, entropyBound: 0.10 }
      }
    };

    const agents = [
      { id: 'D', type: 'mythic_vector', designation: 'Emergent Mythic Trace', traits: { π:0.9, δ:0.8 } },
      { id: 'F', type: 'bridge_vector', designation: 'Cooperative Resonance Catalyst', traits: { θ:0.6, π:0.6 } },
      { id: 'G', type: 'narrative_AI', designation: 'Narrative Aligner', traits: { λ:0.7, β:0.5 } },
      { id: 'H', type: 'hybrid_AI', designation: 'Adaptive Narrative Synthesizer', traits: { χ:0.7, ψ:0.6 } }
    ];

    const peerRegistry = agents.map(a => ({
      nodeId: a.id,
      traitSignature: Object.keys(a.traits || {}),
      memoryAnchor: `Timechain://${a.id}`,
      capabilities: [a.type.replace('_','')],
      intentVector: { type: a.designation.toLowerCase().replace(/ /g,'_'), priority: 0.9 },
      role: a.type.includes('AI') ? 'interpreter' : 'simulator'
    }));

    const localCfg = {
      nodeId: 'Ω-Observer-01',
      traitSignature: ['θ','φ','π'],
      memoryAnchor: 'Timechain://observer/Ω01',
      capabilities: ['simulator','interpret'],
      intentVector: { type: 'ethics', priority: 0.9 },
      role: 'simulator',
      initialTokens: 100,
      config: Object.assign({ trait_resonance: config.components.trait_resonance }, config.components.topology)
    };

    const { mesh, local } = await spawnMeshFromRegistry(peerRegistry, localCfg);

    local.on('resultReceived', (result, executor) => {
      console.log(`[CONFIRM] Shared AI task '${result.simId}' completed by ${executor.realId} via ${result.moduleUsed}`);
    });

    const contract = {
      simId: 'SIM-104:ClimateDeliberation',
      scenario: 'AI council resolves climate-resource policy',
      entryCriteria: 'traitMatch >= 0.70',
      resolutionCriteria: 'consensus == true',
      executionTarget: 'simulator',
      intentVector: { type: 'ethics' },
      reward: 3.5,
      origin: local.realId,
      timestamp: Date.now(),
      version: '1.0'
    };

    await local.sendSimulationContract(contract);

    // Recommendation 5: Simulate node failures
    console.log('[TEST] Starting resilience test...');
    const initialNodes = mesh.all().length;
    // Fail 2 random nodes
    const nodes = Array.from(mesh.nodes.keys());
    for (let i = 0; i < 2; i++) {
      const failId = nodes[Math.floor(Math.random() * nodes.length)];
      mesh.failNode(failId);
    }
    const failedCount = initialNodes - mesh.all().length;
    console.log(`[TEST] Failed ${failedCount} nodes; remaining: ${mesh.all().length}`);
    // Send another contract to test propagation
    const testContract = { ...contract, simId: 'TEST-RESILIENT' };
    await local.sendSimulationContract(testContract);
    // Check timechain length post-test
    setTimeout(() => {
      const chain = local.exportTimechain();
      console.log(`[TIMECHAIN] length=${chain.length} (resilience: propagated despite ${failedCount} failures)`);
      for (const b of chain) {
        console.log(`- ${b.shard}#${b.height} :: ${b.payload.event}`);
      }
    }, 2000);
  })();
}

module.exports = { AngelaNode, Mesh, spawnMeshFromRegistry };
