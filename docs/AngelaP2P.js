/*
 * AngelaP2P Mesh — single-file refactor v1.5
 * Distributed Cognitive P2P AGI: Share AI Resources Privately + Securely
 *
 * Key fixes vs original:
 * - Node crypto (webcrypto) + stable hashing (SHA-256 base64url)
 * - Deterministic sign/verify (no signature in signed payload)
 * - Async hygiene (await only in async fns)
 * - Safer sandbox via Node 'vm' (not vm2)
 * - No private keys serialized into blocks
 * - Sharding uses sha256(id) % N (stable)
 * - Per-shard linearity; cross-shard merge by timestamp
 * - Basic envelope encryption stub (replace with ECDH/X25519 for real)
 * - Peer registry spawns real node instances (methods exist)
 * - Trait vocabulary alignment hooks
 */

const { webcrypto, createHash } = require('crypto');
const vm = require('vm');
const subtle = webcrypto.subtle;

// ===== Utilities =====
const enc = new TextEncoder();
const dec = new TextDecoder();
const b64 = (u8) => Buffer.from(u8).toString('base64');
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

// ===== Mesh primitives =====
class Mesh {
  constructor() { this.nodes = new Map(); }
  add(node) { this.nodes.set(node.realId, node); }
  get(id) { return this.nodes.get(id); }
  all() { return Array.from(this.nodes.values()); }
}

// ===== Angela Node =====
class AngelaNode {
  constructor({ nodeId, traitSignature, memoryAnchor, capabilities, intentVector, role, initialTokens = 100, config }) {
    this.mesh = null; // set externally
    this.pseudonym = `Node-${Math.random().toString(36).slice(2,10)}`;
    this.realId = nodeId;
    this.traitSignature = traitSignature || [];
    this.memoryAnchor = memoryAnchor || null;
    this.capabilities = capabilities || [];
    this.intentVector = intentVector || { type: 'generic', priority: 0.5 };
    this.role = role || 'interpreter';
    this.tokens = initialTokens;
    this.weights = { epistemic: 0.38, harm: 0.25, stability: 0.37 };
    this.config = Object.assign({ resonanceThreshold: 0.9, entropyBound: 0.1, trait_resonance: [], trait_drift_modulator: null }, config || {});

    this.shards = {}; // shardId -> blocks[]
    this.timechain = [];
    this.reputation = { [this.realId]: 0 };
    this.totalTokenSupply = 10000;
    this.maxTokenSupply = 200000;

    this.requestCount = 0;
    this.lastReset = Date.now();

    this.contractQueue = [];

    this.handlers = {}; // event -> fn

    this._keysReady = this._initKeys();
  }

  // ---- lifecycle ----
  async init() {
    await this._keysReady;
    await this._initGenesisBlock();
    return this;
  }

  async _initKeys() {
    const kp = await subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign','verify']);
    this._priv = kp.privateKey; // keep private key off-chain
    this.publicKey = await subtle.exportKey('jwk', kp.publicKey);
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
      timestamp: Date.now()
    };
  }

  // ---- events ----
  on(event, cb) { this.handlers[event] = cb; }
  emit(event, ...args) { if (this.handlers[event]) this.handlers[event](...args); }

  // ---- rate limiting ----
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

  // ---- blocks ----
  async _initGenesisBlock() {
    const shard = shardOf(this.realId);
    this.shards[shard] = this.shards[shard] || [];
    const genesisPayload = { event: 'genesis', data: { ...this.nodeProfile } }; // JSON-safe
    const g = await this._createBlock(genesisPayload, shard);
    this.shards[shard].push(g);
  }

  async _createBlock(payload, shard = shardOf(this.realId)) {
    const prev = this.shards[shard]?.[this.shards[shard].length - 1];
    const unsigned = {
      shard,
      height: (this.shards[shard]?.length || 0),
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
    return subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, key, sigBytes, toBytes(unsigned));
  }

  async _validateBlock(block) {
    // require the validator to be known in mesh and signature to match
    const vId = block.validator;
    const peer = this.mesh?.get(vId);
    const pub = peer?.publicKey || (peer?.nodeProfile?.publicKey);
    if (!pub) return false;
    const ok = await this._verify(block, pub);
    if (!ok) return false;
    // simple stake rule proportional to total reputation
    const totalRep = Object.values(this.reputation).reduce((a,b)=>a+(b||0),0);
    const required = Math.max(0, Math.floor(totalRep * 0.1));
    return block.stake >= required;
  }

  async addBlock(payload) {
    try {
      if (!await this._checkRateLimit()) return;
      const shard = shardOf(this.realId);
      const block = await this._createBlock(payload, shard);
      if (await this._validateBlock(block)) {
        this.shards[shard].push(block);
        await this._propagateBlock(block);
        // mint proportional to self reputation (very small inflation)
        const contribution = this.reputation[this.realId] || 0;
        await this._mintTokens(this.realId, Math.floor(contribution * 0.1));
        await this._crossShardConsensus();
      }
    } catch (e) {
      console.error(`[ERROR] addBlock: ${e.message}`);
    }
  }

  async _propagateBlock(block) {
    if (!this.mesh) return;
    for (const peer of this.mesh.all()) {
      if (peer.realId === this.realId) continue;
      const s = block.shard;
      peer.shards[s] = peer.shards[s] || [];
      if (!peer.shards[s].some(b => b.hash === block.hash)) {
        peer.shards[s].push(block);
      }
    }
  }

  async _crossShardConsensus() {
    // merge all shards by timestamp; keep first valid per (shard,height)
    const blocks = Object.values(this.shards).flat().sort((a,b)=>a.timestamp-b.timestamp);
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

  // ---- routing & envelopes ----
  _computeResonance(peerTraits, peerIntentVector) {
    // unify vocabularies: treat trait labels as strings and compute simple overlap + intent cosine
    const localTraits = new Set(this.traitSignature.map(String));
    const peerSet = new Set((peerTraits || []).map(String));
    const inter = [...localTraits].filter(t => peerSet.has(t)).length;
    const union = new Set([...localTraits, ...peerSet]).size || 1;
    const traitScore = inter / union; // Jaccard

    const lp = this.intentVector?.priority || 0;
    const pp = peerIntentVector?.priority || 0;
    const intentSim = (lp && pp) ? (lp*pp)/(Math.hypot(lp)*Math.hypot(pp)) : 0;

    return (traitScore + intentSim)/2;
  }

  _applyTraitDrift() {
    // keep symbols stable; if you want numeric drift, store separately
    return;
  }

  async secureEnvelope(payload, recipientId = 'ANY') {
    // NOTE: This is a confidentiality stub. For production, use ECDH(X25519) to wrap AES key per recipient.
    try {
      const key = await subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt','decrypt']);
      const iv = webcrypto.getRandomValues(new Uint8Array(12));
      const ct = await subtle.encrypt({ name: 'AES-GCM', iv }, key, toBytes(payload));
      const jwk = await subtle.exportKey('jwk', key);
      return { encryptedPayload: b64url(new Uint8Array(ct)), iv: b64url(iv), wrappedKey: { aesJwk: jwk, to: recipientId, by: this.pseudonym } };
    } catch (e) {
      console.error(`[ERROR] secureEnvelope: ${e.message}`);
      return null;
    }
  }

  async _peelEnvelope(env) {
    const key = await subtle.importKey('jwk', env.wrappedKey.aesJwk, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
    const pt = await subtle.decrypt({ name: 'AES-GCM', iv: b64d(env.iv) }, key, b64d(env.encryptedPayload));
    return JSON.parse(dec.decode(pt));
  }

  async _routeContract(contract, hops = 0, maxHops = 3, layers = 2) {
    if (hops >= maxHops) return;
    if (!await this._checkRateLimit()) return;

    // onionize: repeatedly secureEnvelope around the payload
    let envelope = contract;
    for (let i=0;i<layers;i++) {
      const env = await this.secureEnvelope(envelope, 'ANY');
      if (!env) return;
      envelope = { ...env, layer: (envelope.layer||0) + 1 };
    }

    const candidates = this.mesh?.all().filter(p => p.realId !== this.realId) || [];
    // find peers satisfying executionTarget and resonance
    const suitable = candidates.filter(p => (!contract.executionTarget || p.capabilities.includes(contract.executionTarget)) &&
      this._computeResonance(p.traitSignature, p.intentVector) >= this.config.resonanceThreshold);

    const next = (suitable[0] || candidates[0]);
    if (!next) return;

    await next._onSimulationContract(envelope, this.nodeProfile);
  }

  async _onSimulationContract(envelope, senderProfile) {
    try {
      this._applyTraitDrift();
      let decoded = envelope;
      for (let i = envelope.layer || 0; i > 0; i--) {
        decoded = await this._peelEnvelope(decoded);
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

      // reward + rep + block
      this.tokens += contract.reward;
      this.reputation[this.realId] = (this.reputation[this.realId] || 0) + 1;
      await this.addBlock({ event: 'contract_executed', result, origin: senderProfile.realId, tokensEarned: contract.reward, reputation: this.reputation[this.realId] });

      // deliver result back to sender if that node is present
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

  // ---- external API ----
  async sendSimulationContract(contract) {
    if (!await this._checkRateLimit()) return;
    if (this.tokens < contract.reward) { console.error(`[ERROR] Insufficient tokens: ${this.tokens} < ${contract.reward}`); return; }
    const supported = ['1.0','1.1'];
    contract.version = contract.version || '1.0';
    if (!supported.includes(contract.version)) { console.error(`[VERSION] Not compatible: ${contract.version}`); return; }

    const envelope = await this.secureEnvelope(contract, 'ANY');
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
  const mesh = new Mesh();
  // create local observer node first
  const local = new AngelaNode(localCfg);
  mesh.add(local);
  local.mesh = mesh;
  await local.init();

  // spawn peers
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

// ===== Demo (run: node angela-p2p-mesh.single.js) =====
if (require.main === module) {
  (async () => {
    // Config (trimmed for demo)
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

    // Build peer registry from your JSON agents (symbol-only traits for demo)
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
      capabilities: ['simulator','interpret'], // allow executing
      intentVector: { type: 'ethics', priority: 0.9 },
      role: 'simulator',
      initialTokens: 100,
      config: Object.assign({ trait_resonance: config.components.trait_resonance }, config.components.topology)
    };

    const { mesh, local } = await spawnMeshFromRegistry(peerRegistry, localCfg);

    // wire a confirmation handler
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

    // small delay then print timechain summary
    setTimeout(() => {
      const chain = local.exportTimechain();
      console.log(`\n[TIMECHAIN] length=${chain.length}`);
      for (const b of chain) {
        console.log(`- ${b.shard}#${b.height} :: ${b.payload.event}`);
      }
    }, 1200);
  })();
}

module.exports = { AngelaNode, Mesh, spawnMeshFromRegistry };
