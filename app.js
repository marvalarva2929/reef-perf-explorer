/* Reef Perf — kernel performance explorer.
 * A port of inference-engine/src/csrc/src/simulations/analysis/roofline.py:
 * measured RTL tile cycles × tile counts, plus host↔DTCM transfer at the
 * measured rate (no DMA, so transfers are serialized with compute). */
'use strict';

/* ------------------------------------------------------------------ data */

// Kernel results sets. Cycles are mcycle around one kernel call on Coral's
// VmeCoreMiniAxi RTL (cocotb + Verilator, coralnpu 382b5c12). Attention values
// are the full stream (4 KV blocks of 64 keys + finish) for one head.
const RESULTS = {
  branch: {
    label: 'inference-engine-simulation · 376cd28+',
    short: 'sim 376cd28+',
    date: '2026-10-20',
    note: 'PR #16 trim. Scalar pointwise kernels, VME attention with LUT exp.',
    vmeOverhead: 250, rvvTile: 9315,
    attn: { 64: { prefill: 358439, decode: 39934 }, 32: { prefill: 259219, decode: 25746 } },
    layernorm: { 768: 243105 },
    gelu: 53286, residual: 241941
  },
  coral: {
    label: 'coral-npu-kernels · 5fb006c+',
    short: 'coral-npu 5fb006c+',
    date: '2026-10-21',
    note: 'Every integer loop vectorized on RVV; fixed-point requantization.',
    vmeOverhead: 251, rvvTile: 9316,
    attn: { 64: { prefill: 230527, decode: 31665 }, 32: { prefill: 137562, decode: 17912 } },
    split: { 64: { prefill: [40416.5, 68861], decode: [6827.5, 4355] }, 32: { prefill: [25735.25, 34621], decode: [3922.25, 2223] } },
    layernorm: { 768: 13955, 512: 9534 },
    gelu: 7040, residual: 7594
  },
  stash: {
    label: 'stash on simulation · 4d1a0d9+ (unpushed)',
    short: 'stash 4d1a0d9+',
    date: '2026-10-22',
    note: 'Vectorized pointwise and attention softmax rows, rescale and finish. Lives only in a git stash.',
    vmeOverhead: 250, rvvTile: 9314,
    attn: { 64: { prefill: 105359, decode: 23886 }, 32: { prefill: 74087, decode: 13981 } },
    split: { 64: { prefill: [25255.25, 4338], decode: [5884.5, 348] }, 32: { prefill: [17933.25, 2354], decode: [3439.25, 224] } },
    layernorm: { 768: 21343, 512: 14367 },
    gelu: 7029, residual: 17987
  }
};

// Per-tile shapes and traffic (same across result sets; from model.json tiles).
const TILE = {
  layernorm: { rows: 4 },          // ops 8/elem, host write = elems*1.5, out = elems
  gelu: { elems: 4096, ops: 4096, hw: 4096, out: 4352 },
  residual: { elems: 4096, ops: 12288, hw: 8192, out: 4096 }
};

// TinyStories (GPT-Neo) presets: d and
// layers from the roneneldan/TinyStories-* config.json files, MLP 4·d, vocab
// 50257, local window 256, 16 heads in every config (so d_head = d / 16).
const tinyStories = (label, d, layers) => ({
  label, file: label + ' (HF config)', name: label.toLowerCase(), n_head: 16, d_head: d / 16, d_model: d, d_ff: 4 * d,
  layers, window: 256, vocab: 50257, n_ctx: 2048, alternating: true,
  assumed: { n_head: '16 in every TinyStories config' }
});
const MODELS = {
  ts1m: tinyStories('TinyStories-1M', 64, 8),
  ts3m: tinyStories('TinyStories-3M', 128, 8),
  ts8m: tinyStories('TinyStories-8M', 256, 8),
  neo: { label: 'TinyStories-28M', file: 'gpt_neo_28m.json', name: 'gpt-neo-28m', n_head: 16, d_head: 32, d_model: 512, d_ff: 2048, layers: 8, window: 256, vocab: 50257, n_ctx: 2048, alternating: true, assumed: { n_head: '16 (TinyStories HF configs)' } },
  ts33m: tinyStories('TinyStories-33M', 768, 4),
  gpt2: { label: 'GPT-2 small', file: 'model.json', name: 'gpt2-small', n_head: 12, d_head: 64, d_model: 768, d_ff: 3072, layers: 12, window: 1024, vocab: 50257, n_ctx: 1024, alternating: false, assumed: {} }
};

// hyperram: the SoC architecture (lhs-soc.vercel.app/architecture, soc.yaml).
// Weights, KV cache and spilled activations live in external HyperRAM; one DMA
// channel moves tiles between it and the Coral core's TCMs (one outstanding
// transaction). The site gives no bandwidth: the controller's [10:0] external
// interface matches an x8 HyperBus (DQ[7:0], RWDS, CK, CS#), which moves 2 B
// per bus clock (DDR); the bus clock defaults to the core clock. Whether DMA
// overlaps compute is not stated, so it defaults to serialized.
const HARDWARE = {
  hyperram: {
    label: 'atlas · HyperRAM + DMA', file: 'soc.yaml (atlas)', name: 'HyperRAM via DMA', clock_mhz: 50, dbus: 16, vme_step: 84,
    clockConfirmed: false, ext: true, hb_mhz: 50, hb_bytes: 2, dma_overlap: 0
  },
  a0: { label: 'hardware.json · host AXI', file: 'hardware.json', name: 'host AXI, no DMA', clock_mhz: 50, host_bw: 15.745, dbus: 16, vme_step: 84, clockConfirmed: false, ext: false }
};

const BOUNDS = {
  compute: { label: 'Compute', color: '#6b98ff' },
  dtcm: { label: 'Memory · DTCM', color: '#f5c26b' },
  host: { label: 'Host transfer', color: '#e07b3c' },
  latency: { label: 'Latency / overhead', color: '#737b88' }
};

const DESCS = {
  ln1: 'Pre-attention LayerNorm over d_model.',
  qkv: 'Fused Q, K, V projection: d_model → 3 · n_head · d_head.',
  attention: 'Causal block-flash attention: q·kᵀ and P·v per 64-key KV block, LUT-exp softmax, then finish.',
  attn_out: 'Attention output projection: n_head · d_head → d_model.',
  residual1: 'Residual add after attention.',
  ln2: 'Pre-MLP LayerNorm over d_model.',
  fc1: 'MLP up-projection: d_model → d_ff.',
  gelu: 'GeLU activation over d_ff.',
  fc2: 'MLP down-projection: d_ff → d_model.',
  residual2: 'Residual add after the MLP.',
  ln_f: 'Final LayerNorm, newest position only.',
  lm_head: 'LM head: one row, d_model → vocab. Streams the whole weight matrix through DTCM every token.'
};

/* ---------------------------------------------------------------- engine */

const cdiv = (a, b) => Math.ceil(a / b);
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');

function matmul(id, n, k, m, HW, L) {
  const vme = n % 16 === 0;
  const tn = vme ? 16 : 1, tk = 512, tm = vme ? 16 : 32;
  const chunks = [];
  for (let k0 = 0; k0 < k; k0 += tk) chunks.push(Math.min(tk, k - k0));
  let kc = 0, dt = 0, host = 0;
  const at = cdiv(n, tn), mt = cdiv(m, tm);
  for (const c of chunks) {
    kc += vme ? L.vmeOverhead + HW.vme_step * c / 4 : L.rvvTile * c / tk;
    dt += vme ? (c / 4) * 128 + tn * tm * 4 : tn * cdiv(tm, 16) * c * 17 + tn * tm * 4;
    host += tn * c + mt * (c * tm + tn * tm * 4);
  }
  const src = vme
    ? `vme_matmul_u8 16×512×16 = ${fmtInt(L.vmeOverhead + 84 * 128)} cyc · fit ${L.vmeOverhead} + ${HW.vme_step}·k/4`
    : `matmul_rvv_s8u8 1×512×32 = ${fmtInt(L.rvvTile)} cyc`;
  return {
    id, engine: vme ? 'VME' : 'RVV', kind: 'matmul · ' + (vme ? 'VME' : 'RVV'),
    shape: `${n}×${k}×${m}`, ops: 2 * n * k * m, kc: at * mt * kc, dtcm: at * mt * dt,
    host: at * host, tiles: at * mt * chunks.length, src, flag: false
  };
}

function attention(P, nq, ctx, window, L) {
  const DH = P.d_head, H = P.n_head, block = 64, qb = nq > 1 ? 16 : 1;
  let pairs = 0;
  for (let q0 = 0; q0 < nq; q0 += qb) {
    const f = ctx - nq + q0, l = ctx - nq + Math.min(q0 + qb, nq) - 1, lo = Math.max(0, f + 1 - window);
    pairs += Math.floor(l / block) - Math.floor(lo / block) + 1;
  }
  let visible = 0;
  for (let i = 0; i < nq; i++) visible += Math.min(ctx - nq + i + 1, window);
  const qBlocks = cdiv(nq, qb);
  const ph = nq > 1 ? 'prefill' : 'decode';
  // [cycles per KV block, finish cycles per query block]; split measured
  // per block when the result set has it, else the stream total over 4 blocks.
  const cost = (d) => (L.split && L.split[d] ? L.split[d][ph] : [L.attn[d][ph] / 4, 0]);
  let per, fin, flag = false;
  if (L.attn[DH]) [per, fin] = cost(DH);
  else {
    flag = true;
    const a = cost(32), b = cost(64), t = (DH - 32) / 32;
    per = Math.max(1, a[0] + (b[0] - a[0]) * t);
    fin = Math.max(0, a[1] + (b[1] - a[1]) * t);
  }
  const mm = nq > 1
    ? ((qb / 16) * (block / 16) * (DH / 4) + (qb / 16) * (DH / 16) * (block / 4)) * 128
    : DH * cdiv(block, 16) * 17 + block * cdiv(DH, 16) * 17;
  const state = qb * block * 10 + qb * DH * 16;
  const host = H * (pairs * 2 * block * DH + qBlocks * (qb * DH + qb * (DH + 2) * 4 + qb * DH));
  const stream = (DH === 64 ? 'vme' : 'vme_d32') + '_' + ph;
  return {
    id: 'attention', engine: nq > 1 ? 'VME' : 'RVV', kind: 'flash attention · ' + (nq > 1 ? 'VME' : 'RVV'),
    shape: `q${nq} kv${Math.min(ctx, window)} d${DH} ×${H}h`, ops: H * 4 * visible * DH,
    kc: H * (pairs * per + qBlocks * fin), dtcm: H * pairs * (mm + state), host, tiles: H * pairs, flag,
    src: flag ? `no stream at d_head ${DH} · interpolated between d32 and d64`
              : L.split && L.split[DH]
                ? `${stream} = ${fmtInt(L.split[DH][ph][0])} cyc / KV block + ${fmtInt(L.split[DH][ph][1])} finish`
                : `${stream} stream = ${fmtInt(L.attn[DH][ph])} cyc, spread over 4 KV blocks`,
    flagText: flag ? `d_head ${DH} has no RTL run. Cost per KV block is interpolated between the measured d32 and d64 streams.` : ''
  };
}

function pointwise(id, kind, elems, d, L) {
  let cyc, te, ops, hw, out, flag = false, srcName, flagText = '';
  if (kind === 'layernorm') {
    const rows = TILE.layernorm.rows;
    const ds = Object.keys(L.layernorm).map(Number);
    const exact = L.layernorm[d];
    const base = exact ? d : ds.reduce((a, b) => (Math.abs(b - d) < Math.abs(a - d) ? b : a));
    cyc = L.layernorm[base] * (d / base);
    te = rows * d; ops = 8 * te; hw = 1.5 * te; out = te;
    flag = !exact;
    srcName = `layernorm 4×${base} = ${fmtInt(L.layernorm[base])} cyc` + (flag ? ` · scaled to d=${d}` : '');
    if (flag) flagText = `No layernorm 4×${d} run in this result set. Cost is the 4×${base} tile scaled linearly by row width.`;
  } else {
    const t = TILE[kind];
    cyc = L[kind]; te = t.elems; ops = t.ops; hw = t.hw; out = t.out;
    srcName = `${kind === 'residual' ? 'residual_add' : kind} 4096 = ${fmtInt(cyc)} cyc`;
  }
  const tiles = cdiv(elems, te), ht = hw + out, dt = ht + (kind === 'layernorm' ? te : 0);
  return {
    id, engine: 'RVV', kind: kind + ' · RVV', shape: kind === 'layernorm' ? `${elems / d}×${d}` : `${elems} elems`,
    ops: ops * tiles, kc: cyc * tiles, dtcm: dt * tiles, host: ht * tiles, tiles, src: srcName, flag, flagText
  };
}

function layerOps(P, HW, L, nq, ctx, w) {
  const D = P.d_model, A = P.n_head * P.d_head;
  return [
    pointwise('ln1', 'layernorm', nq * D, D, L),
    matmul('qkv', nq, D, 3 * A, HW, L),
    attention(P, nq, ctx, w, L),
    matmul('attn_out', nq, A, D, HW, L),
    pointwise('residual1', 'residual', nq * D, D, L),
    pointwise('ln2', 'layernorm', nq * D, D, L),
    matmul('fc1', nq, D, P.d_ff, HW, L),
    pointwise('gelu', 'gelu', nq * P.d_ff, D, L),
    matmul('fc2', nq, P.d_ff, D, HW, L),
    pointwise('residual2', 'residual', nq * D, D, L)
  ];
}

function classify(o, HW, L) {
  o.peak = o.engine === 'VME' ? 2048 / HW.vme_step : 32768 / L.rvvTile;
  o.cu = (o.ops / o.peak) / o.kc;
  o.du = (o.dtcm / HW.dbus) / o.kc;
  o.hu = o.hc / o.tc;
  if (o.hc > o.kc) o.bound = 'host';
  else if (o.cu >= 0.6) o.bound = 'compute';
  else if (o.du >= 0.6) o.bound = 'dtcm';
  else o.bound = 'latency';
}

function run(P, HW, L, nq, ctx) {
  const counts = P.alternating
    ? { global: Math.ceil(P.layers / 2), local: Math.floor(P.layers / 2) }
    : { global: P.layers, local: 0 };
  const res = {
    global: layerOps(P, HW, L, nq, ctx, P.n_ctx),
    local: layerOps(P, HW, L, nq, ctx, P.window)
  };
  const head = [pointwise('ln_f', 'layernorm', P.d_model, P.d_model, L), matmul('lm_head', 1, P.d_model, P.vocab, HW, L)];
  // Every tile's inputs (weights, activations, K/V blocks) come in and its
  // outputs go out over the transfer path: host AXI writes, or the HyperRAM
  // DMA. Serialized with the kernel unless the DMA is set to overlap.
  const overlap = HW.ext && HW.dma_overlap;
  const cost = (o) => {
    o.hc = o.host / HW.host_bw;
    o.tc = overlap ? Math.max(o.kc, o.hc) : o.kc + o.hc;
    classify(o, HW, L);
  };
  let total = 0, kern = 0, moved = 0;
  for (const t of ['global', 'local']) for (const o of res[t]) { cost(o); total += counts[t] * o.tc; kern += counts[t] * o.kc; moved += counts[t] * o.host; }
  for (const o of head) { cost(o); total += o.tc; kern += o.kc; moved += o.host; }
  return { res, head, counts, total, kern, moved };
}

/* -------------------------------------------------------------- format */

function fmtT(cyc, HW) {
  const ms = cyc / (HW.clock_mhz * 1e3);
  if (ms >= 60000) return (ms / 60000).toFixed(2) + ' min';
  if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
  if (ms >= 100) return ms.toFixed(0) + ' ms';
  if (ms >= 10) return ms.toFixed(1) + ' ms';
  return ms.toFixed(2) + ' ms';
}
function fmtC(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'G';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(Math.round(n));
}
function fmtB(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' MB';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + ' KB';
  return Math.round(n) + ' B';
}
const pct = (x) => (x * 100).toFixed(x < 0.1 ? 1 : 0) + '%';
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* --------------------------------------------------------------- state */

const S = {
  model: 'neo', hw: 'hyperram', res: 'coral', phase: 'decode', seq: 1024, layerType: 'global', sel: null,
  p: {}, hwp: {}, whatif: false, zoom: null, fit: 0.9
};

(function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (RESULTS[h.get('res')]) S.res = h.get('res');
  if (MODELS[h.get('model')]) S.model = h.get('model');
  if (HARDWARE[h.get('hw')]) S.hw = h.get('hw');
  if (['prefill', 'decode'].includes(h.get('phase'))) S.phase = h.get('phase');
  if ([128, 256, 512, 1024, 2048].includes(+h.get('seq'))) S.seq = +h.get('seq');
  if (['global', 'local'].includes(h.get('layer'))) S.layerType = h.get('layer');
  if (h.get('sel') && /^[a-z_0-9]+$/.test(h.get('sel'))) S.sel = h.get('sel');
})();

function writeHash() {
  const h = new URLSearchParams({ res: S.res, model: S.model, hw: S.hw, phase: S.phase, seq: S.seq, layer: S.layerType });
  if (S.sel) h.set('sel', S.sel);
  history.replaceState(null, '', '#' + h.toString());
}

const INTS = new Set(['n_head', 'd_head', 'd_model', 'd_ff', 'layers', 'window', 'vocab']);
function resolved() {
  const BP = MODELS[S.model], BH = HARDWARE[S.hw];
  const apply = (base, over) => {
    const out = Object.assign({}, base);
    for (const k of Object.keys(over)) {
      const v = parseFloat(over[k]);
      if (isFinite(v) && v > 0) out[k] = INTS.has(k) ? Math.max(1, Math.round(v)) : v;
    }
    return out;
  };
  const P = apply(BP, S.p), HW = apply(BH, S.hwp);
  if (P.d_head % 4) P.d_head = Math.max(4, Math.round(P.d_head / 4) * 4);
  if (HW.ext) {
    if (S.hwp.dma_overlap !== undefined) HW.dma_overlap = S.hwp.dma_overlap ? 1 : 0;
    HW.host_bw = HW.hb_mhz * HW.hb_bytes / HW.clock_mhz; // B per core cycle
  }
  BOUNDS.host.label = HW.ext ? 'Memory · HyperRAM' : 'Host transfer';
  return { P, HW, BP, BH, L: RESULTS[S.res] };
}

/* -------------------------------------------------------------- render */

const $ = (id) => document.getElementById(id);
const GEO = { W: 196, H: 148, X0: 264, DX: 240, R1: 136, R2: 348, RH: 600 };

function edgePath() {
  const { W, H, X0, DX, R1, R2, RH } = GEO;
  const ar = (x, y) => ` M${x - 7} ${y - 5} L${x} ${y} L${x - 7} ${y + 5}`;
  const ad = (x, y) => ` M${x - 5} ${y - 7} L${x} ${y} L${x + 5} ${y - 7}`;
  let d = 'M115 160 V198' + ad(115, 198);
  d += ` M190 248 H226 V${R1 + H / 2} H${X0 - 2}` + ar(X0 - 2, R1 + H / 2);
  for (const y of [R1, R2]) for (let i = 0; i < 4; i++) {
    const a = X0 + i * DX + W, b = X0 + (i + 1) * DX - 2;
    d += ` M${a} ${y + H / 2} H${b}` + ar(b, y + H / 2);
  }
  const lastX = X0 + 4 * DX + W / 2, firstX = X0 + W / 2;
  d += ` M${lastX} ${R1 + H} V${R1 + H + 30} H${firstX} V${R2 - 2}` + ad(firstX, R2 - 2);
  d += ` M${lastX} ${R2 + H} V${R2 + H + 54} H${firstX} V${RH - 2}` + ad(firstX, RH - 2);
  d += ` M${X0 + W} ${RH + 74} H${X0 + DX - 2}` + ar(X0 + DX - 2, RH + 74);
  d += ` M${X0 + DX + W} ${RH + 74} H742` + ar(742, RH + 74);
  return d;
}

function segHTML(opts, current, attr) {
  return opts.map(([v, label]) =>
    `<button type="button" data-${attr}="${esc(v)}" aria-pressed="${String(v) === String(current)}">${esc(label)}</button>`).join('');
}

function optionsHTML(map, labelFn, current, extra) {
  return Object.keys(map).map((k) => `<option value="${esc(k)}"${k === current ? ' selected' : ''}>${esc(labelFn(map[k], k))}</option>`).join('') + (extra || '');
}

function render() {
  const { P, HW, BP, BH, L } = resolved();
  const seq = Math.min(S.seq, P.n_ctx);
  const lt = P.alternating ? S.layerType : 'global';
  const pre = run(P, HW, L, seq, seq);
  const dec = run(P, HW, L, 1, seq);
  const cur = S.phase === 'prefill' ? pre : dec;
  const phase = S.phase;
  const T = (c) => fmtT(c, HW);

  // ---- ranking
  const order = ['ln1', 'qkv', 'attention', 'attn_out', 'residual1', 'ln2', 'fc1', 'gelu', 'fc2', 'residual2'];
  const rows = [];
  order.forEach((id, i) => {
    if (id === 'attention') {
      for (const t of ['global', 'local']) {
        const c = cur.counts[t];
        if (!c) continue;
        const o = cur.res[t][i];
        rows.push({ o, key: id, t, label: (P.alternating ? 'attention · ' + t : 'attention') + ' ×' + c, tot: o.tc * c });
      }
    } else {
      const o = cur.res.global[i];
      rows.push({ o, key: id, t: '', label: id + ' ×' + P.layers, tot: o.tc * P.layers });
    }
  });
  for (const o of cur.head) rows.push({ o, key: o.id, t: '', label: o.id, tot: o.tc });
  rows.sort((a, b) => b.tot - a.tot);
  const maxShare = rows[0].tot / cur.total;

  const layer = cur.res[lt];
  const ncount = (o) => (o.id === 'attention' ? (cur.counts[lt] || 1) : P.layers);
  const all = layer.concat(cur.head);
  if (S.sel && !all.some((o) => o.id === S.sel)) S.sel = null;
  const sel = S.sel;

  // ---- nav
  $('model-select').innerHTML = optionsHTML(MODELS, (m) => m.label || m.file, S.model, '<option value="__import">Import JSON…</option>');
  $('model-select').title = BP.file;
  $('hw-select').innerHTML = optionsHTML(HARDWARE, (h) => h.label || h.file, S.hw, '<option value="__import">Import JSON…</option>');
  $('hw-select').title = BH.file;
  const XF = HW.ext ? 'HyperRAM↔TCM' : 'Host↔DTCM';
  $('res-select').innerHTML = optionsHTML(RESULTS, (r) => r.short, S.res);
  $('res-select').title = L.note;
  $('phase-seg').innerHTML = segHTML([['prefill', 'Prefill'], ['decode', 'Decode']], S.phase, 'phase');
  $('seq-seg').innerHTML = segHTML([128, 256, 512, 1024, 2048].filter((v) => v <= P.n_ctx).map((v) => [v, v]), seq, 'seq');
  $('seq-label').textContent = S.phase === 'prefill' ? 'Prompt' : 'Context';

  // ---- what-if
  const fields = [
    ['p', 'n_head', 'Attention heads', 1, BP.assumed && BP.assumed.n_head || 'from ' + BP.file],
    ['p', 'd_head', 'Head dim', 4, 'measured streams: 32, 64'],
    ['p', 'd_model', 'Model width', 64, 'd_model'],
    ['p', 'd_ff', 'MLP hidden', 64, 'd_ff'],
    ['p', 'layers', 'Layers', 1, P.alternating ? 'alternating global / local' : 'all global'],
    ['p', 'window', 'Local window', 64, P.alternating ? 'local-attention layers' : 'unused (all global)'],
    ['p', 'vocab', 'Vocab size', 1024, 'LM head width'],
    ['hwp', 'clock_mhz', 'Clock (MHz)', 5, BH.clockConfirmed ? 'confirmed' : 'unconfirmed · not in soc.yaml'],
    ...(HW.ext ? [
      ['hwp', 'hb_mhz', 'HyperBus clock (MHz)', 5, 'not stated · defaults to core clock'],
      ['hwp', 'hb_bytes', 'Bytes per bus clock', 1, 'x8 DDR from [10:0] interface · derived'],
      ['hwp', 'dma_overlap', 'DMA overlaps compute', 'bool', 'not stated · 1 outstanding txn']
    ] : [['hwp', 'host_bw', 'Host→DTCM (B/cycle)', 0.5, 'measured · RTL bound 16']]),
    ['hwp', 'dbus', 'DTCM bus (B/cycle)', 4, 'confirmed · 128-bit LSU'],
    ['hwp', 'vme_step', 'VME cycles / step', 1, 'measured · k-sweep slope']
  ];
  let changed = 0;
  const fieldHTML = (f) => {
    const [grp, key, label, step, note] = f;
    const baseV = grp === 'p' ? BP[key] : BH[key];
    const raw = S[grp][key];
    if (step === 'bool') {
      const on = raw !== undefined ? !!raw : !!baseV;
      const isB = raw !== undefined && !!raw !== !!baseV;
      if (isB) changed++;
      return `<div class="wf${isB ? ' changed' : ''}">
        <label for="f-${key}"><span>${esc(label)}</span><span class="note small muted">${esc(isB ? 'edited · file says ' + (baseV ? 'on' : 'off') : note)}</span></label>
        <input id="f-${key}" type="checkbox" class="chk" data-grp="${grp}" data-key="${key}"${on ? ' checked' : ''}>
      </div>`;
    }
    const isCh = raw !== undefined && parseFloat(raw) !== baseV;
    if (isCh) changed++;
    return `<div class="wf${isCh ? ' changed' : ''}">
      <label for="f-${key}"><span>${esc(label)}</span><span class="note small muted">${esc(isCh ? 'edited · file says ' + baseV : note)}</span></label>
      <input id="f-${key}" type="number" inputmode="decimal" step="${step}" data-grp="${grp}" data-key="${key}" value="${esc(raw !== undefined ? raw : baseV)}">
    </div>`;
  };
  const modelFields = fields.filter((f) => f[0] === 'p').map(fieldHTML).join('');
  const hwFields = fields.filter((f) => f[0] === 'hwp').map(fieldHTML).join('');
  const active = document.activeElement && document.activeElement.id;
  $('whatif').hidden = !S.whatif;
  $('whatif').innerHTML = `
    <div style="display:flex;align-items:center;gap:8px"><h2>What-if</h2>
      <button type="button" class="btn sm" data-act="reset">Reset</button>
      <button type="button" class="btn sm" data-act="whatif" aria-label="Close what-if">✕</button></div>
    <fieldset><legend class="eyebrow">Model · ${esc(BP.file)}</legend>${modelFields}</fieldset>
    <fieldset><legend class="eyebrow">Hardware · ${esc(BH.file)}</legend>${hwFields}</fieldset>
    <p class="small muted" style="margin:0">Edits override the loaded files for this session. Kernel cycles stay as measured on the VME RTL; the VME step re-costs VME tiles.${HW.ext ? ' HyperRAM bandwidth: ' + (HW.hb_mhz * HW.hb_bytes).toFixed(0) + ' MB/s = ' + HW.host_bw.toFixed(2) + ' B/cycle.' : ''}</p>`;
  if (active && active.startsWith('f-') && $(active)) { const el = $(active); el.focus(); }
  $('whatif-btn').setAttribute('aria-pressed', String(S.whatif));
  $('whatif-count').hidden = !changed;
  $('whatif-count').textContent = changed;

  // ---- canvas world
  const nodeHTML = (o, x, y, n) => {
    const share = (o.tc * n) / cur.total;
    const B = BOUNDS[o.bound];
    return `<button type="button" class="node" style="left:${x}px;top:${y}px" data-sel="${o.id}" aria-pressed="${sel === o.id}"
      aria-label="${esc(`${o.id}, ${B.label}, ${T(o.tc)}, ${pct(share)} of ${phase}`)}">
      <span class="strip" style="background:${B.color}"></span>
      <span class="body">
        <span style="display:flex;align-items:center;gap:6px"><span class="id">${o.id}</span>${o.flag ? '<span class="tag">scaled</span>' : ''}</span>
        <span class="kind">${esc(o.kind)}</span>
        <span class="shape">${esc(o.shape)}</span>
        <span style="flex-grow:1"></span>
        <span style="display:flex;align-items:baseline;justify-content:space-between;gap:4px"><span class="time">${T(o.tc)}</span><span class="share">${pct(share)}</span></span>
        <span class="bar"><span style="width:${Math.min(100, share / maxShare * 100).toFixed(1)}%;background:${B.color}"></span></span>
      </span></button>`;
  };
  let nodes = '';
  layer.forEach((o, i) => { nodes += nodeHTML(o, GEO.X0 + (i % 5) * GEO.DX, i < 5 ? GEO.R1 : GEO.R2, ncount(o)); });
  cur.head.forEach((o, i) => { nodes += nodeHTML(o, GEO.X0 + i * GEO.DX, GEO.RH, 1); });
  const layerSeg = P.alternating
    ? segHTML([['global', 'Global ×' + cur.counts.global], ['local', 'Local ×' + cur.counts.local]], lt, 'layer')
    : `<button type="button" aria-pressed="true" disabled>Global ×${P.layers}</button>`;
  $('world').innerHTML = `
    <div class="abs frame"></div>
    <div class="abs frame-head"><span class="title">Transformer layer</span><span class="mono muted">× ${P.layers}</span>
      <div class="seg" role="group" aria-label="Layer type">${layerSeg}</div>
      <span class="small muted">${!P.alternating ? 'All layers attend to the full context' : lt === 'local' ? `Local layers attend to the last ${P.window} keys` : 'Global layers attend to the full context'}</span></div>
    <div class="abs row-label" style="left:264px;top:112px">Attention sub-block</div>
    <div class="abs row-label" style="left:264px;top:324px">MLP sub-block</div>
    <div class="abs row-label" style="left:264px;top:576px">Output head · once per step</div>
    <svg class="abs" width="1500" height="820" viewBox="0 0 1500 820" aria-hidden="true" style="left:0;top:0"><path d="${edgePath()}" fill="none" stroke="#5a616c" stroke-width="1.6" stroke-linejoin="round"/></svg>
    <div class="abs endpoint" style="left:40px;top:120px;width:150px">${S.phase === 'prefill' ? 'tokens in · ' + seq : 'token in · ctx ' + seq}</div>
    <div class="abs ghost-node"><span class="mono" style="font-weight:600;font-size:14px">embedding</span><span class="small muted">host gather</span><span class="small muted">not modeled</span></div>
    <div class="abs endpoint" style="left:744px;top:654px;width:180px">logits · ${P.vocab}</div>
    ${nodes}`;

  const zoom = S.zoom || S.fit;
  $('world').style.transform = `scale(${zoom})`;
  $('sizer').style.width = Math.ceil(1500 * zoom + 420) + 'px';
  $('sizer').style.height = Math.ceil(820 * zoom + 60) + 'px';
  $('zoom-pct').textContent = Math.round(zoom * 100) + '%';
  $('context-label').textContent = `${P.name} · ${phase}${S.phase === 'prefill' ? ' ' + seq + ' tokens' : ' @ ctx ' + seq} · ${BH.name} · ${L.short}`;
  $('legend').innerHTML = Object.values(BOUNDS).map((b) => `<span><span class="sw" style="background:${b.color}"></span>${b.label}</span>`).join('');

  // ---- right popups
  const scrollKeep = [...$('right').querySelectorAll('.grow')].map((e) => e.scrollTop);
  if (!sel) {
    const mixAcc = Object.fromEntries(Object.keys(BOUNDS).map((k) => [k, 0]));
    const moveLine = HW.ext
      ? `Kernel ${pct(cur.kern / cur.total)} · HyperRAM DMA ${pct(1 - cur.kern / cur.total)} of time. ${fmtB(cur.moved)} moved at ${(HW.hb_mhz * HW.hb_bytes).toFixed(0)} MB/s (${HW.host_bw.toFixed(2)} B/cycle, derived). DMA ${HW.dma_overlap ? 'overlaps compute: each block takes the longer of the two' : 'serialized with compute (overlap not stated)'}.`
      : `Kernel time ${pct(cur.kern / cur.total)} · host↔DTCM transfer ${pct(1 - cur.kern / cur.total)} (no DMA, serialized)`;
    for (const r of rows) mixAcc[r.o.bound] += r.tot;
    const mix = Object.keys(BOUNDS).map((k) => ({ ...BOUNDS[k], w: mixAcc[k] / cur.total }));
    const hz = HW.clock_mhz * 1e6;
    // compare against the other result sets
    const others = Object.keys(RESULTS).map((k) => {
      const R = RESULTS[k], p2 = run(P, HW, R, seq, seq), d2 = run(P, HW, R, 1, seq);
      return { k, R, pre: p2.total, dec: d2.total };
    });
    const base = others.find((x) => x.k === S.res);
    $('right').innerHTML = `
      <section class="popup fixed" aria-label="Overall estimate">
        <div><div class="eyebrow">Overall estimate</div>
          <div class="small" style="color:var(--text-2)">${esc(P.name)} on ${esc(BH.name)} · ${HW.clock_mhz} MHz · ${esc(L.short)}</div></div>
        <div class="kpis">
          <div class="kpi${S.phase === 'prefill' ? ' on' : ''}"><div class="small muted">Prefill · ${seq} tok</div><div class="v">${T(pre.total)}</div><div class="small muted">${(seq / (pre.total / hz)).toFixed(1)} tok/s</div></div>
          <div class="kpi${S.phase === 'decode' ? ' on' : ''}"><div class="small muted">Decode · ctx ${seq}</div><div class="v">${T(dec.total)}</div><div class="small muted">${(hz / dec.total).toFixed(2)} tok/s</div></div>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px">
          <div class="small" style="color:var(--text-2)">What limits ${phase} time</div>
          <div class="stack" role="img" aria-label="${esc(mix.map((m) => m.label + ' ' + pct(m.w)).join(', '))}">${mix.map((m) => `<div style="width:${(m.w * 100).toFixed(2)}%;background:${m.color}"></div>`).join('')}</div>
          <div class="grid2">${mix.map((m) => `<div class="sw-row"><span class="sw" style="background:${m.color}"></span><span style="flex-grow:1">${m.label}</span><span class="mono">${pct(m.w)}</span></div>`).join('')}</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:4px">
          <div class="small" style="color:var(--text-2)">Same model on each kernel results set</div>
          ${others.map((x) => `<button type="button" class="rank-row" data-res="${x.k}" style="min-height:32px;flex-direction:row;align-items:baseline;gap:8px;${x.k === S.res ? 'background:#1f2a40' : ''}">
            <span class="mono small" style="flex-grow:1">${esc(x.R.short)}</span>
            <span class="mono small">${T(x.dec)}/tok</span>
            <span class="mono small muted" style="width:76px;text-align:right">${T(x.pre)}</span></button>`).join('')}
          <div class="small muted">decode @ ctx ${seq} · prefill ${seq} tokens${base ? '' : ''}</div>
        </div>
        <div class="small muted">${esc(moveLine)}</div>
      </section>
      <section class="popup grow" aria-label="Where the time goes">
        <div style="display:flex;align-items:baseline;gap:8px"><div class="eyebrow" style="flex-grow:1">Where the time goes · ${phase}</div><span class="mono small" style="color:var(--text-2)">${T(cur.total)}</span></div>
        ${rows.map((r) => {
          const sh = r.tot / cur.total, B = BOUNDS[r.o.bound];
          return `<button type="button" class="rank-row" data-sel="${r.key}" data-layer-set="${r.t}">
            <span class="top"><span class="lbl">${esc(r.label)}</span><span class="mono small" style="color:var(--text-2)">${T(r.tot)}</span><span class="mono small muted" style="width:44px;text-align:right">${pct(sh)}</span></span>
            <span class="track"><span style="width:${(sh / maxShare * 100).toFixed(1)}%;background:${B.color}"></span></span></button>`;
        }).join('')}
      </section>`;
  } else {
    const inHead = cur.head.find((o) => o.id === sel);
    const o = inHead || layer.find((x) => x.id === sel);
    const n = inHead ? 1 : ncount(o);
    const B = BOUNDS[o.bound];
    const reasons = {
      compute: `Runs at ${pct(o.cu)} of the measured ${o.engine} peak (${o.peak.toFixed(1)} ops/cycle). Only more ${o.engine} throughput or fewer ops moves it.`,
      dtcm: `The DTCM bus is ${pct(o.du)} busy at ${HW.dbus} B/cycle. A wider bus or better reuse moves it.`,
      host: HW.ext
        ? `Moving ${fmtB(o.host)} over the HyperRAM DMA (${fmtC(o.hc)} cycles at ${HW.host_bw.toFixed(2)} B/cycle) takes longer than the kernel (${fmtC(o.kc)}). ${HW.dma_overlap ? 'With overlap the transfer sets the time.' : 'It is serialized with the kernel.'} More HyperRAM bandwidth or tile reuse moves it.`
        : `Host↔DTCM transfer (${fmtC(o.hc)} cycles) is longer than the kernel (${fmtC(o.kc)}). No DMA, so it is not overlapped.`,
      latency: `Neither roof is close: ${pct(o.cu)} of ${o.engine} peak, ${pct(o.du)} of DTCM bandwidth. Time goes to loop overhead, scalar work and load-use stalls — a kernel-side fix.`
    };
    const stats = [
      [inHead ? 'Time' : 'Time / layer', T(o.tc)],
      [`In model (×${n})`, `${T(o.tc * n)} · ${pct(o.tc * n / cur.total)}`],
      ['Shape', o.shape], ['Tiles / launches', fmtInt(o.tiles)],
      ['Kernel cycles', fmtC(o.kc)], [XF + ' cycles', fmtC(o.hc)],
      ['DTCM↔core bytes', fmtB(o.dtcm)], [XF + ' bytes', fmtB(o.host)],
      ['Ops (2·MAC)', fmtC(o.ops)], ['Attained', (o.ops / o.kc).toFixed(2) + ' ops/cyc'],
      ['Intensity', (o.ops / o.dtcm).toFixed(2) + ' ops/B'], ['Engine', `${o.engine} · peak ${o.peak.toFixed(1)}`]
    ];
    const meters = [
      [o.engine + ' compute peak', Math.min(1, o.cu), BOUNDS.compute.color],
      ['DTCM bus bandwidth', Math.min(1, o.du), BOUNDS.dtcm.color],
      [(HW.ext ? 'HyperRAM DMA' : 'Host transfer') + ' share of block time', o.hu, BOUNDS.host.color]
    ];
    // same block across result sets
    const across = Object.keys(RESULTS).map((k) => {
      const r2 = run(P, HW, RESULTS[k], S.phase === 'prefill' ? seq : 1, seq);
      const o2 = inHead ? r2.head.find((x) => x.id === sel) : r2.res[lt].find((x) => x.id === sel);
      return { k, short: RESULTS[k].short, tc: o2.tc, bound: o2.bound };
    });
    const maxA = Math.max(...across.map((a) => a.tc));
    // roofline
    const lg = (v) => Math.log10(Math.max(1e-6, v));
    const rx = (x) => Math.min(348, Math.max(44, 44 + (lg(x) + 2) / 5 * 304));
    const ry = (y) => Math.min(200, Math.max(12, 200 - (lg(y) + 2) / 4 * 188));
    const pV = 2048 / HW.vme_step, pR = 32768 / L.rvvTile;
    const pts = all.filter((p) => p !== o).map((p) =>
      `<span class="pt" title="${p.id}" style="left:${rx(p.ops / p.dtcm) - 4}px;top:${ry(p.ops / p.kc) - 4}px;width:8px;height:8px;background:#4a515c;border:1px solid #171a1f"></span>`).join('') +
      `<span class="pt" title="${o.id}" style="left:${rx(o.ops / o.dtcm) - 7}px;top:${ry(o.ops / o.kc) - 7}px;width:14px;height:14px;background:${B.color};border:2px solid #e8eaed"></span>`;
    const gx = [44, 104.8, 165.6, 226.4, 287.2, 348], gy = [12, 59, 106, 153, 200];
    $('right').innerHTML = `
      <section class="popup fixed sel-card" aria-label="Block stats">
        <div style="height:5px;background:${B.color}"></div>
        <div class="inner">
          <div class="sel-head"><div style="flex-grow:1"><div class="eyebrow">${inHead ? 'Output head · ' + phase : (o.id === 'attention' && P.alternating ? lt + ' layers · ' : 'every layer · ') + phase}</div><h2>${o.id}</h2></div>
            <button type="button" class="btn sm" data-act="clear">✕ Overview</button></div>
          <div class="small" style="color:var(--text-2);font-size:13px">${DESCS[o.id]}</div>
          <div class="bound-box"><div style="display:flex;align-items:center;gap:8px"><span class="sw" style="width:12px;height:12px;border-radius:3px;background:${B.color}"></span><b>${B.label}</b></div>
            <div class="small" style="color:var(--text-2)">${esc(reasons[o.bound])}</div></div>
          <div class="stats">${stats.map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${esc(v)}</span></div>`).join('')}</div>
        </div>
      </section>
      <section class="popup grow" aria-label="Block limits">
        <div class="eyebrow">How close to each limit</div>
        ${meters.map(([l, v, c]) => `<div class="meter" style="display:flex;flex-direction:column;gap:3px"><div style="display:flex;justify-content:space-between" class="small"><span style="color:var(--text-2)">${l}</span><span class="mono">${pct(v)}</span></div><span class="track"><span style="width:${(v * 100).toFixed(1)}%;background:${c}"></span></span></div>`).join('')}
        <div class="roof">
          <svg width="348" height="232" viewBox="0 0 352 232" role="img" aria-label="Roofline: this block against the VME and RVV ceilings">
            <g stroke="#262a31">${gx.map((x) => `<line x1="${x}" y1="12" x2="${x}" y2="200"/>`).join('')}${gy.map((y) => `<line x1="44" y1="${y}" x2="348" y2="${y}"/>`).join('')}</g>
            <g font-family="IBM Plex Mono, monospace" font-size="10" fill="#a3a9b3">
              ${['0.01', '0.1', '1', '10', '100', '1k'].map((t, i) => `<text x="${gx[i]}" y="214" text-anchor="${i === 5 ? 'end' : 'middle'}">${t}</text>`).join('')}
              ${['100', '10', '1', '0.1', '0.01'].map((t, i) => `<text x="38" y="${gy[i] + 3}" text-anchor="end">${t}</text>`).join('')}
              <text x="196" y="230" text-anchor="middle">ops per DTCM↔core byte</text></g>
            <path d="M${rx(0.01)} ${ry(HW.dbus * 0.01)} L${rx(pV / HW.dbus)} ${ry(pV)} L348 ${ry(pV)}" fill="none" stroke="#e8eaed" stroke-width="1.6"/>
            <path d="M${rx(pR / HW.dbus)} ${ry(pR)} L348 ${ry(pR)}" fill="none" stroke="#e8eaed" stroke-width="1.2" stroke-dasharray="4 3"/>
            <text x="346" y="${ry(pV) - 4}" text-anchor="end" font-size="10" fill="#e8eaed" font-family="IBM Plex Sans, sans-serif">VME peak ${pV.toFixed(1)}</text>
            <text x="346" y="${ry(pR) - 4}" text-anchor="end" font-size="10" fill="#e8eaed" font-family="IBM Plex Sans, sans-serif">RVV peak ${pR.toFixed(2)}</text>
          </svg>${pts}
        </div>
        <div class="small muted">y: attained ops/cycle, kernel only. Grey dots: the other blocks in this view.</div>
        <div style="display:flex;flex-direction:column;gap:6px">
          <div class="eyebrow">This block on each result set</div>
          ${across.map((a) => `<div style="display:flex;flex-direction:column;gap:3px"><div class="small" style="display:flex;justify-content:space-between;${a.k === S.res ? 'color:var(--text)' : 'color:var(--text-2)'}"><span class="mono">${esc(a.short)}${a.k === S.res ? ' ·  current' : ''}</span><span class="mono">${T(a.tc)}</span></div><span class="track"><span style="width:${(a.tc / maxA * 100).toFixed(1)}%;background:${BOUNDS[a.bound].color}"></span></span></div>`).join('')}
        </div>
        <div class="src"><div class="small" style="font-weight:600">Measured source · ${esc(L.short)}</div>
          ${o.engine === 'VME' ? '<div class="small muted">VME cycles stand in for the 16×16 systolic array (assumed the same unit).</div>' : ''}
          <div class="mono small" style="color:var(--text-2)">${esc(o.src)}</div>
          ${o.flag ? `<div class="flag">${esc(o.flagText)}</div>` : ''}</div>
      </section>`;
  }
  [...$('right').querySelectorAll('.grow')].forEach((e, i) => { if (scrollKeep[i] != null) e.scrollTop = scrollKeep[i]; });
  writeHash();
}

/* -------------------------------------------------------------- events */

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.h);
  toast.h = setTimeout(() => { t.hidden = true; }, 6000);
}

function fitZoom() {
  const st = document.querySelector('.stage').getBoundingClientRect();
  const narrow = st.width < 760;
  const f = Math.min(1.1, (st.width - (narrow ? 32 : 420)) / 1500, (st.height - (narrow ? st.height * 0.46 + 40 : 40)) / 820);
  S.fit = Math.max(0.35, Math.round(f * 100) / 100);
}

document.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) {
    // click on empty canvas clears the selection
    if (S.sel && e.target.closest('#scroller')) { S.sel = null; render(); }
    return;
  }
  const d = b.dataset;
  if (d.phase) S.phase = d.phase;
  else if (d.seq) S.seq = +d.seq;
  else if (d.layer) S.layerType = d.layer;
  else if (d.res) S.res = d.res;
  else if (d.sel) {
    if (d.layerSet) S.layerType = d.layerSet;
    S.sel = S.sel === d.sel && !d.layerSet ? null : d.sel;
  } else if (d.zoom) {
    const z = S.zoom || S.fit;
    S.zoom = d.zoom === 'fit' ? null : Math.max(0.35, Math.min(1.6, Math.round((z + 0.1 * +d.zoom) * 10) / 10));
  } else if (d.act === 'clear') S.sel = null;
  else if (d.act === 'whatif' || b.id === 'whatif-btn') S.whatif = !S.whatif;
  else if (d.act === 'reset') { S.p = {}; S.hwp = {}; }
  else return;
  render();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && S.sel) { S.sel = null; render(); }
});

document.addEventListener('input', (e) => {
  const t = e.target;
  if (t.dataset && t.dataset.grp) { S[t.dataset.grp][t.dataset.key] = t.type === 'checkbox' ? (t.checked ? 1 : 0) : t.value; render(); }
});

$('res-select').addEventListener('change', (e) => { S.res = e.target.value; render(); });
$('model-select').addEventListener('change', (e) => {
  if (e.target.value === '__import') { e.target.value = S.model; $('model-file').click(); return; }
  S.model = e.target.value; S.p = {}; S.sel = null; S.layerType = 'global'; render();
});
$('hw-select').addEventListener('change', (e) => {
  if (e.target.value === '__import') { e.target.value = S.hw; $('hw-file').click(); return; }
  S.hw = e.target.value; S.hwp = {}; render();
});

/* ------------------------------------------------------------- imports */

const num = (v) => (v && typeof v === 'object' && 'value' in v ? v.value : v);

// Accepts the repo's model.json / gpt_neo_28m.json ({"model": {...}}) or
// reef.model/v1 ({"params": {...}}).
function parseModel(j, file) {
  const m = j.model || j.params;
  if (!m || typeof m !== 'object') throw new Error('expected a "model" or "params" object');
  const need = ['d_model', 'n_head', 'd_ff', 'layers', 'vocab'];
  const miss = need.filter((k) => !(+num(m[k]) > 0));
  if (miss.length) throw new Error('missing ' + miss.join(', '));
  const pattern = j.layer_pattern || m.attention_layers || ['global'];
  const out = {
    file, name: String(m.name || j.name || file.replace(/\.json$/, '')),
    d_model: +num(m.d_model), n_head: +num(m.n_head), d_ff: +num(m.d_ff), layers: +num(m.layers), vocab: +num(m.vocab),
    n_ctx: +num(m.n_ctx) || 1024, alternating: pattern.includes('local'), assumed: {}
  };
  out.d_head = +num(m.d_head) || out.d_model / out.n_head;
  out.window = +num(m.window) || out.n_ctx;
  if (j.assumed && j.assumed.n_head) out.assumed.n_head = 'assumed: ' + String(j.assumed.n_head).slice(0, 40);
  return out;
}

// Accepts the repo's hardware.json (leaves of {value, units, source, confirmed})
// or reef.hw/v1 (nested memory / interconnect / engines).
function parseHW(j, file) {
  const g = (...paths) => {
    for (const p of paths) {
      let v = j;
      for (const k of p.split('.')) v = v && v[k];
      v = num(v);
      if (+v > 0) return +v;
    }
    return null;
  };
  const ext = g('hyperram.bus_mhz', 'memory.hyperram.bus_mhz', 'hyperram.mb_per_s', 'memory.hyperram.mb_per_s');
  if (ext) {
    const mbs = g('hyperram.mb_per_s', 'memory.hyperram.mb_per_s');
    const busMhz = g('hyperram.bus_mhz', 'memory.hyperram.bus_mhz') || (g('clock_mhz') || 50);
    const out = {
      file, label: String(j.name || file), name: 'HyperRAM via DMA', ext: true,
      clock_mhz: g('clock_mhz') || 50, dbus: g('dbus_bytes_per_cycle', 'memory.dtcm.bus_bytes_per_cycle') || 16,
      vme_step: g('vme_cycles_per_matmul_step', 'engines.vme.cycles_per_step') || 84,
      hb_mhz: busMhz, hb_bytes: mbs ? mbs / busMhz : (g('hyperram.bytes_per_clock', 'memory.hyperram.bytes_per_clock') || 2),
      dma_overlap: (j.dma && num(j.dma.overlap)) ? 1 : 0, clockConfirmed: !!(j.clock_mhz && j.clock_mhz.confirmed)
    };
    return out;
  }
  const out = {
    file, name: String(j.name || file.replace(/\.json$/, '')),
    clock_mhz: g('clock_mhz'),
    host_bw: g('host_to_dtcm_bytes_per_cycle', 'interconnect.host_to_dtcm_bytes_per_cycle'),
    dbus: g('dbus_bytes_per_cycle', 'memory.dtcm.bus_bytes_per_cycle'),
    vme_step: g('vme_cycles_per_matmul_step', 'engines.vme.cycles_per_step') || 84,
    clockConfirmed: !!(j.clock_mhz && j.clock_mhz.confirmed)
  };
  const miss = ['clock_mhz', 'host_bw', 'dbus'].filter((k) => !out[k]);
  if (miss.length) throw new Error('missing ' + miss.join(', '));
  return out;
}

function importInto(input, map, parse, key) {
  input.addEventListener('change', async () => {
    const f = input.files[0];
    input.value = '';
    if (!f) return;
    try {
      const j = JSON.parse(await f.text());
      const id = 'u_' + f.name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
      map[id] = parse(j, f.name);
      S[key] = id;
      if (key === 'model') { S.p = {}; S.sel = null; } else S.hwp = {};
      render();
      toast(`Loaded ${f.name}`);
    } catch (err) {
      toast(`Couldn't import ${f.name}: ${err.message}`);
    }
  });
}
importInto($('model-file'), MODELS, parseModel, 'model');
importInto($('hw-file'), HARDWARE, parseHW, 'hw');

window.addEventListener('resize', () => { fitZoom(); render(); });
fitZoom();
render();
