/* Reef Perf — kernel performance explorer.
 * A port of inference-engine/src/csrc/src/simulations/analysis/roofline.py:
 * measured RTL tile cycles × tile counts, plus host↔DTCM transfer at the
 * measured rate (no DMA, so transfers are serialized with compute). */
'use strict';

/* ------------------------------------------------------------------ data */

// Kernel results: one set, measured end to end on reef-perf
// (yuvesh-kernel-integration a660183: M3 core + Zvt matrix engine + DMA + host
// boot) running the e2e-npu-firmware branch (inference-engine/coral/e2e), with
// mcycle read after every kernel. MEASURED_CYCLES (below) gives each module's
// cycles per model, phase, XSPI bandwidth and decode context; run() charges
// those. The remaining fields only shape the resource model (ops, DTCM and DMA
// bytes per module, VME/RVV peaks) and come from the earlier RTL kernel runs.
const RESULTS = {
  reef: {
    label: 'reef-perf e2e · a660183 · measured',
    short: 'reef-perf e2e (measured)',
    date: '2026-11-09',
    note: 'Every module measured end to end on reef-perf: prefill = 16-token prompt; decode at context 17, 128, 256, 512, 1024, 2048.',
    vmeOverhead: 136, gemvOverhead: 35, vmeStep: 25, rvvTile: 9316,
    attn: { 64: { prefill: 230527, decode: 31665 }, 32: { prefill: 137562, decode: 17912 } },
    split: { 64: { prefill: [40416.5, 68861], decode: [6827.5, 4355] }, 32: { prefill: [25735.25, 34621], decode: [3922.25, 2223] } },
    layernorm: { 768: 13955, 512: 9534 },
    gelu: 7040, residual: 7594
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
  ts28m: tinyStories('TinyStories-28M', 512, 8),
  ts33m: tinyStories('TinyStories-33M', 768, 4)
};

// xspi: the RTL team's data path (2026-11). Weights and, for now, the KV cache
// in XSPI flash (16 bit per XSPI clock, ~6 ns); one DMA channel into the 32 KB
// DTCM, double buffered so transfers overlap compute; the host boots the 8 KB
// ITCM over SPI (1 bit/cycle). XSPI clock assumed equal to the 50 MHz core.
const HARDWARE = {
  xspi: {
    label: 'reef-perf · XSPI + DMA', file: 'reef-perf a660183', name: 'XSPI flash via DMA', clock_mhz: 50, dbus: 16, vme_step: 84,
    clockConfirmed: false, dtcm_c: 3.1, ext: true, hb_mhz: 50, hb_bytes: 2, dma_overlap: 1
  }
};

// Module cycles measured on reef-perf (see RESULTS). MEASURED_CYCLES[model] =
// { order, prefill: { bw: [cycles in order] }, decode: { bw: { ctx: [...] } } }.
// Prefill is a 16-token prompt; bw is XSPI bytes per core cycle. Residual
// adds and GeLU run inside attn_out, fc1 and fc2, so they are 0.
const MEASURED_CYCLES = {
  ts28m: {
    order: ['ln1',  'qkv',  'attention_global',  'attention_local',  'attn_out',  'ln2',  'fc1',  'fc2',  'ln_f',  'lm_head',  'embedding'],
    prefill: { 0.25: [86972,  6419012,  317740,  325940,  2142098,  86952,  8539804,  8520158,  43553,  104877942,  81162], 0.5: [86972,  3266114,  248108,  256308,  1103228,  86952,  4357448,  4340666,  43553,  52965771,  48394], 1: [86972,  1764020,  235578,  243778,  607612,  86952,  2374984,  2272186,  43553,  27010955,  32010], 2: [86972,  1127006,  234554,  242754,  395274,  86952,  1527322,  1394476,  43553,  15124450,  23818], 4: [86972,  1127006,  234042,  242242,  395274,  86952,  1527322,  1394476,  43553,  15075497,  19722], 8: [86972,  1127006,  233786,  241986,  395274,  86952,  1527322,  1394476,  43553,  15075497,  17674] },
    decode: {
      0.25: { 17: [43486,  3207762,  227062,  235258,  1070953,  43476,  4269806,  4259983,  43549,  104877942,  5136], 128: [43483,  3204104,  639667,  672439,  1069755,  43476,  4264131,  4254306,  43480,  104680519,  5054], 256: [43483,  3204104,  1189539,  1255111,  1069755,  43476,  4264131,  4254306,  43480,  104680519,  5054], 512: [43483,  3204104,  2289155,  1255113,  1069755,  43476,  4264131,  4254306,  43480,  104680519,  5054], 1024: [43483,  3204104,  4488291,  1255113,  1069755,  43476,  4264131,  4254306,  43480,  104680519,  5054], 2048: [43483,  3204104,  8885859,  1255113,  1069755,  43476,  4264131,  4254306,  43480,  104680519,  5054] },
      0.5: { 17: [43486,  1629969,  190198,  198394,  550846,  43476,  2175652,  2169565,  43549,  52965771,  3088], 128: [43483,  1625833,  602803,  635575,  549408,  43476,  2169054,  2163660,  43480,  52768350,  3006], 256: [43483,  1625833,  1152675,  1218247,  549408,  43476,  2169054,  2163660,  43480,  52768350,  3006], 512: [43483,  1625833,  2252291,  1218249,  549408,  43476,  2169054,  2163660,  43480,  52768350,  3006], 1024: [43483,  1625833,  4451427,  1218249,  549408,  43476,  2169054,  2163660,  43480,  52768350,  3006], 2048: [43483,  1625833,  8848995,  1218249,  549408,  43476,  2169054,  2163660,  43480,  52768350,  3006] },
      1: { 17: [43486,  872209,  182909,  191105,  303038,  43476,  1184420,  1135325,  43549,  27010955,  2064], 128: [43483,  868073,  595514,  628286,  301600,  43476,  1177822,  1129420,  43480,  26813534,  1982], 256: [43483,  868073,  1145386,  1210958,  301600,  43476,  1177822,  1129420,  43480,  26813534,  1982], 512: [43483,  868073,  2245002,  1210960,  301600,  43476,  1177822,  1129420,  43480,  26813534,  1982], 1024: [43483,  868073,  4444138,  1210960,  301600,  43476,  1177822,  1129420,  43480,  26813534,  1982], 2048: [43483,  868073,  8841706,  1210960,  301600,  43476,  1177822,  1129420,  43480,  26813534,  1982] },
      2: { 17: [43486,  516617,  181885,  190081,  185891,  43476,  716641,  648610,  43549,  15124450,  1552], 128: [43483,  513381,  594490,  627262,  184976,  43476,  712950,  645760,  43480,  15052396,  1470], 256: [43483,  513381,  1144362,  1209934,  184976,  43476,  712950,  645760,  43480,  15052396,  1470], 512: [43483,  513381,  2243978,  1209936,  184976,  43476,  712950,  645760,  43480,  15052396,  1470], 1024: [43483,  513381,  4443114,  1209936,  184976,  43476,  712950,  645760,  43480,  15052396,  1470], 2048: [43483,  513381,  8840682,  1209936,  184976,  43476,  712950,  645760,  43480,  15052396,  1470] },
      4: { 17: [43486,  514991,  181373,  189569,  185349,  43476,  714509,  648086,  43549,  15075497,  1296], 128: [43483,  511839,  593978,  626750,  184462,  43476,  711006,  645278,  43480,  15011847,  1214], 256: [43483,  511839,  1143850,  1209422,  184462,  43476,  711006,  645278,  43480,  15011847,  1214], 512: [43483,  511839,  2243466,  1209424,  184462,  43476,  711006,  645278,  43480,  15011847,  1214], 1024: [43483,  511839,  4442602,  1209424,  184462,  43476,  711006,  645278,  43480,  15011847,  1214], 2048: [43483,  511839,  8840170,  1209424,  184462,  43476,  711006,  645278,  43480,  15011847,  1214] },
      8: { 17: [43486,  514991,  181117,  189313,  185349,  43476,  714509,  648086,  43549,  15075497,  1168], 128: [43483,  511839,  593722,  626494,  184462,  43476,  711006,  645278,  43480,  15011847,  1086], 256: [43483,  511839,  1143594,  1209166,  184462,  43476,  711006,  645278,  43480,  15011847,  1086], 512: [43483,  511839,  2243210,  1209168,  184462,  43476,  711006,  645278,  43480,  15011847,  1086], 1024: [43483,  511839,  4442346,  1209168,  184462,  43476,  711006,  645278,  43480,  15011847,  1086], 2048: [43483,  511839,  8839914,  1209168,  184462,  43476,  711006,  645278,  43480,  15011847,  1086] }
    }
  },
  ts33m: {
    order: ['ln1',  'qkv',  'attention_global',  'attention_local',  'attn_out',  'ln2',  'fc1',  'fc2',  'ln_f',  'lm_head',  'embedding'],
    prefill: { 0.25: [257428,  28755240,  877096,  893496,  9590944,  257364,  38317272,  38270440,  64418,  156935240,  118778], 0.5: [257428,  14591288,  668200,  684600,  4905448,  257364,  19466880,  19429256,  64418,  79269960,  69626], 1: [257428,  7683728,  626308,  642708,  2632168,  257364,  10373760,  10078088,  64418,  40437320,  45050], 2: [257428,  4537148,  623236,  639636,  1583308,  257364,  6186348,  5783468,  64418,  22500656,  32762], 4: [257428,  4530728,  621700,  638100,  1581168,  257364,  6178012,  5781432,  64418,  22450582,  26618], 8: [257428,  4530728,  620936,  637336,  1581168,  257364,  6178012,  5781432,  64418,  22450582,  23546] },
    decode: {
      0.25: { 17: [64357,  7188090,  309051,  315852,  2397704,  64341,  9579286,  9567578,  64414,  156935040,  7490], 128: [64346,  7179074,  911630,  917862,  2394718,  64341,  9565172,  9553457,  64346,  156628259,  7432], 256: [64346,  7179074,  1714959,  1721222,  2394718,  64341,  9565172,  9553457,  64346,  156628259,  7432], 512: [64346,  7179074,  3321614,  1721223,  2394718,  64341,  9565172,  9553457,  64346,  156628259,  7432], 1024: [64346,  7179074,  6534926,  1721223,  2394718,  64341,  9565172,  9553457,  64346,  156628259,  7432], 2048: [64346,  7179074,  12961550,  1721223,  2394718,  64341,  9565172,  9553457,  64346,  156628259,  7432] },
      0.5: { 17: [64357,  3646398,  255432,  263628,  1225978,  64341,  4865184,  4856930,  64414,  79269760,  4418], 128: [64346,  3636752,  794337,  827110,  1222677,  64341,  4850096,  4842516,  64346,  78962979,  4360], 256: [64346,  3636752,  1513489,  1579062,  1222677,  64341,  4850096,  4842516,  64346,  78962979,  4360], 512: [64346,  3636752,  2951665,  1579063,  1222677,  64341,  4850096,  4842516,  64346,  78962979,  4360], 1024: [64346,  3636752,  5827921,  1579063,  1222677,  64341,  4850096,  4842516,  64346,  78962979,  4360], 2048: [64346,  3636752,  11579729,  1579063,  1222677,  64341,  4850096,  4842516,  64346,  78962979,  4360] },
      1: { 17: [64357,  1918910,  244959,  253155,  657658,  64341,  2591904,  2519138,  64414,  40437120,  2882], 128: [64346,  1909264,  783864,  816637,  654357,  64341,  2576816,  2504724,  64346,  40130339,  2824], 256: [64346,  1909264,  1503016,  1568589,  654357,  64341,  2576816,  2504724,  64346,  40130339,  2824], 512: [64346,  1909264,  2941192,  1568590,  654357,  64341,  2576816,  2504724,  64346,  40130339,  2824], 1024: [64346,  1909264,  5817448,  1568590,  654357,  64341,  2576816,  2504724,  64346,  40130339,  2824], 2048: [64346,  1909264,  11569256,  1568590,  654357,  64341,  2576816,  2504724,  64346,  40130339,  2824] },
      2: { 17: [64357,  1100303,  244191,  252387,  386995,  64341,  1511259,  1409387,  64414,  22355840,  2114], 128: [64346,  1097764,  783096,  815869,  386399,  64341,  1509086,  1408770,  64346,  22338351,  2056], 256: [64346,  1097764,  1502248,  1567821,  386399,  64341,  1509086,  1408770,  64346,  22338351,  2056], 512: [64346,  1097764,  2940424,  1567822,  386399,  64341,  1509086,  1408770,  64346,  22338351,  2056], 1024: [64346,  1097764,  5816680,  1567822,  386399,  64341,  1509086,  1408770,  64346,  22338351,  2056], 2048: [64346,  1097764,  11568488,  1567822,  386399,  64341,  1509086,  1408770,  64346,  22338351,  2056] },
      4: { 17: [64357,  1097546,  243807,  252003,  386076,  64341,  1507639,  1408494,  64414,  22299478,  1730], 128: [64346,  1095355,  782712,  815485,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1672], 256: [64346,  1095355,  1501864,  1567437,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1672], 512: [64346,  1095355,  2940040,  1567438,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1672], 1024: [64346,  1095355,  5816296,  1567438,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1672], 2048: [64346,  1095355,  11568104,  1567438,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1672] },
      8: { 17: [64357,  1097546,  243616,  251812,  386076,  64341,  1507639,  1408494,  64414,  22299478,  1538], 128: [64346,  1095355,  782520,  815293,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1480], 256: [64346,  1095355,  1501672,  1567245,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1480], 512: [64346,  1095355,  2939848,  1567246,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1480], 1024: [64346,  1095355,  5816104,  1567246,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1480], 2048: [64346,  1095355,  11567912,  1567246,  385596,  64341,  1506045,  1408015,  64346,  22294762,  1480] }
    }
  }
};
const nearest = (keys, v) => keys.map(Number).reduce((a, b) => (Math.abs(Math.log(b / v)) < Math.abs(Math.log(a / v)) ? b : a));
// The measured run nearest this XSPI bandwidth and context, or null.
function measuredRun(model, phase, bw, ctx) {
  const m = MEASURED_CYCLES[model];
  if (!m) return null;
  const b = nearest(Object.keys(m[phase]), bw);
  let row = m[phase][b], c = 16;
  if (phase === 'decode') { c = nearest(Object.keys(row), ctx); row = row[c]; }
  const out = { bw: b, ctx: c };
  m.order.forEach((k, i) => { out[k] = row[i]; });
  return out;
}

// Side-panel resource list. Every panel view reads this; order is display order.
const RESOURCES = [
  { id: 'vme', name: () => 'VME', short: () => 'VME', unit: 'ops', rate: 'ops/cyc', color: '#6b98ff' },
  { id: 'rvv', name: () => 'RVV', short: () => 'RVV', unit: 'ops', rate: 'ops/cyc', color: '#4fd1b5' },
  { id: 'dtcm', name: () => 'DTCM ↔ core', short: () => 'DTCM', unit: 'bytes', rate: 'B/cyc', color: '#f5c26b' },
  { id: 'xfer', name: (HW) => (HW.ext ? 'XSPI DMA' : 'Host AXI'), short: (HW) => (HW.ext ? 'DMA' : 'Host'), unit: 'bytes', rate: 'B/cyc', color: '#e07b3c' }
];
const OVERHEAD = { id: 'overhead', color: '#aab3c0' };
// vtmms-only loop, operands loaded once (DECODE_GEMV_PROPOSAL): cycles per 16x16x4 step.
const VME_ARRAY_STEP = 6.9;
const DTCM_WIDEST = 16;

const BOUNDS = {
  compute: { label: 'Compute', color: '#6b98ff' },
  dtcm: { label: 'Memory · DTCM', color: '#f5c26b' },
  host: { label: 'Host transfer', color: '#e07b3c' },
  latency: { label: 'Latency / overhead', color: '#aab3c0' }
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
  const gemm = n % 16 === 0, gemv = !gemm && L.gemvOverhead != null, vme = gemm || gemv;
  const over = gemv ? L.gemvOverhead : L.vmeOverhead;
  const tn = gemm ? 16 : 1, tk = 512, tm = vme ? 16 : 32;
  const chunks = [];
  for (let k0 = 0; k0 < k; k0 += tk) chunks.push(Math.min(tk, k - k0));
  let kc = 0, dt = 0, host = 0, acc = 0;
  const at = cdiv(n, tn), mt = cdiv(m, tm);
  for (const c of chunks) {
    kc += vme ? over + HW.vme_step * c / 4 : L.rvvTile * c / tk;
    dt += vme ? (c / 4) * 128 + tn * tm * 4 : tn * cdiv(tm, 16) * c * 17 + tn * tm * 4;
    host += tn * c + mt * (c * tm + tn * tm * 4);
    // VME step: 8 contiguous 16 B loads transposed, 20 accesses (6.4 B each) otherwise.
    // RVV: one A byte and one 16 B B load per k. Results drain at 16 B.
    acc += (vme ? (c / 4) * (L.vmeStep ? 8 : 20) : tn * cdiv(tm, 16) * c * 2) + tn * tm * 4 / DTCM_WIDEST;
  }
  const src = gemv
    ? `mtx_gemv_int8 1×512×16 = ${fmtInt(over + HW.vme_step * 128)} cyc · A transposed`
    : vme
    ? `${L.vmeStep ? 'mtx_gemm_int8' : 'vme_matmul_u8'} 16×512×16 = ${fmtInt(over + HW.vme_step * 128)} cyc${L.vmeStep ? ' · A transposed' : ''}`
    : `matmul_rvv_s8u8 1×512×32 = ${fmtInt(L.rvvTile)} cyc`;
  return {
    id, engine: vme ? 'VME' : 'RVV', kind: 'matmul · ' + (vme ? 'VME' : 'RVV'),
    shape: `${n}×${k}×${m}`, ops: 2 * n * k * m, kc: at * mt * kc, dtcm: at * mt * dt,
    host: at * host, tiles: at * mt * chunks.length, src, flag: false,
    // The array computes all 16 rows on a GEMV tile; only one is useful.
    usedOps: 2 * at * mt * chunks.reduce((a, c) => a + (gemv ? 16 : tn) * c * tm, 0),
    reqOps: 2 * n * k * m, reqBytes: n * k + k * m + 4 * n * m, dtcmAcc: at * mt * acc
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
    usedOps: H * pairs * qb * block * DH * 4, reqOps: H * 4 * visible * DH, otherUnknown: true,
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
    ops: ops * tiles, kc: cyc * tiles, dtcm: dt * tiles, host: ht * tiles, tiles, src: srcName, flag, flagText,
    usedOps: ops * tiles
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
  // Demand / capacity per resource, in cycles; the bottleneck is the largest.
  const compute = o.ops / o.peak, dtcm = o.dtcm / HW.dbus;
  o.busy = { compute, dtcm, host: o.hc, latency: Math.max(0, o.kc - Math.max(compute, dtcm)) };
  o.bound = Object.keys(o.busy).reduce((a, b) => (o.busy[b] > o.busy[a] ? b : a));
}

// Per-module record for the side panel. segments sum to total_cycles.
function moduleData(o, HW, L, overlap) {
  const eng = o.engine === 'VME' ? 'vme' : 'rvv', oth = eng === 'vme' ? 'rvv' : 'vme';
  const other = o.otherUnknown ? null : 0, bytes = o.reqBytes != null ? o.reqBytes : null;
  const used = { [eng]: o.usedOps, [oth]: other, dtcm: o.dtcm, xfer: o.host };
  const required = { [eng]: o.reqOps != null ? o.reqOps : null, [oth]: other, dtcm: bytes, xfer: bytes };
  const peak_rate = { vme: 2048 / VME_ARRAY_STEP, rvv: 32768 / L.rvvTile, dtcm: HW.dbus, xfer: HW.host_bw };
  // RTL mcycle around the kernel, charged to the engine it runs on.
  const active_cycles = { [eng]: o.kc };
  const used_accesses = o.dtcmAcc != null ? o.dtcmAcc : o.dtcm / DTCM_WIDEST;
  const bytes_per_access = used_accesses ? o.dtcm / used_accesses : 0;
  const required_accesses = bytes != null ? bytes / DTCM_WIDEST : null;
  const c = HW.dtcm_c, Bus = HW.dbus;
  const dtcm = {
    used_bytes: o.dtcm, used_accesses, bytes_per_access, required_bytes: bytes, required_accesses,
    c, cycles: used_accesses * Math.max(c, bytes_per_access / Bus),
    min_cycles: required_accesses != null ? required_accesses * Math.max(c, DTCM_WIDEST / Bus) : null
  };
  const cycles = {}, min_cycles = {}, standalone = {};
  for (const { id } of RESOURCES) {
    const modeled = id === 'dtcm' ? dtcm.cycles : (used[id] == null ? null : used[id] / peak_rate[id]);
    standalone[id] = modeled;
    cycles[id] = active_cycles[id] != null ? active_cycles[id] : modeled;
    min_cycles[id] = id === 'dtcm' ? dtcm.min_cycles : (required[id] == null ? null : required[id] / peak_rate[id]);
  }
  // A serial DMA gets its full time. On-chip resources overlap, so only the
  // largest is charged, capped at kernel time; the rest of the kernel is overhead.
  const segments = { vme: 0, rvv: 0, dtcm: 0, xfer: 0, overhead: 0 };
  const pool = overlap ? ['vme', 'rvv', 'dtcm', 'xfer'] : ['vme', 'rvv', 'dtcm'];
  const top = pool.reduce((x, y) => ((standalone[y] || 0) > (standalone[x] || 0) ? y : x));
  if (!overlap) segments.xfer = o.hc;
  segments[top] = Math.min(overlap ? o.tc : o.kc, standalone[top] || 0);
  segments.overhead = o.tc - RESOURCES.reduce((a, r) => a + segments[r.id], 0);
  return { total_cycles: o.tc, shape: o.shape, segments, peak_rate, required, used, active_cycles, dtcm, cycles, min_cycles };
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
  const meas = measuredRun(S.model, nq > 1 ? 'prefill' : 'decode', HW.host_bw, ctx);
  const key = (o, t) => (o.id === 'attention' ? 'attention_' + t : o.id);
  const cost = (o, t) => {
    o.hc = o.host / HW.host_bw;
    // Measured: the module's whole time, DMA included (it overlaps compute).
    if (meas) o.kc = meas[key(o, t)] || 0;
    o.tc = meas ? o.kc : overlap ? Math.max(o.kc, o.hc) : o.kc + o.hc;
    classify(o, HW, L);
    o.m = moduleData(o, HW, L, overlap);
  };
  let total = 0, kern = 0, moved = 0;
  for (const t of ['global', 'local']) for (const o of res[t]) { cost(o, t); total += counts[t] * o.tc; kern += counts[t] * o.kc; moved += counts[t] * o.host; }
  for (const o of head) { cost(o); total += o.tc; kern += o.kc; moved += o.host; }
  if (meas) total += meas.embedding;
  return { res, head, counts, total, kern, moved, meas };
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
// Panel number format: 3 significant figures, K/M/G suffixes.
function f3(n) {
  if (n == null) return '—';
  if (n === 0) return '0';
  const sfx = ['', 'K', 'M', 'G', 'T'];
  let a = n, i = 0;
  while (a >= 1000 && i < 4) { a /= 1000; i++; }
  let t = +a.toPrecision(3);
  if (t >= 1000 && i < 4) { t /= 1000; i++; }
  return (t >= 100 ? t.toFixed(0) : t >= 10 ? t.toFixed(1) : t.toFixed(2)) + sfx[i];
}
function fB2(n) {
  if (n == null) return '—';
  if (n === 0) return '0';
  const u = ['B', 'KB', 'MB', 'GB'];
  let a = n, i = 0;
  while (a >= 1024 && i < 3) { a /= 1024; i++; }
  const t = +a.toPrecision(3);
  return (t >= 100 ? t.toFixed(0) : t >= 10 ? t.toFixed(1) : i ? t.toFixed(2) : t.toFixed(0)) + ' ' + u[i];
}
function p2(x) {
  if (x == null) return '—';
  if (x === 0) return '0%';
  return (x >= 0.1 ? (x * 100).toFixed(0) : (x * 100).toFixed(1)) + '%';
}
const hzFmt = (v) => f3(v);
function t3(cyc, HW) {
  const ms = cyc / (HW.clock_mhz * 1e3);
  if (ms >= 60000) return f3(ms / 60000) + ' min';
  if (ms >= 1000) return f3(ms / 1000) + ' s';
  return f3(ms) + ' ms';
}
const withUnit = (s, u) => (s === '—' ? s : `${s}<span class="u">${u}</span>`);
const amount = (r, v) => {
  if (r.unit === 'ops') return withUnit(f3(v), 'ops');
  if (v === 0) return withUnit('0', 'B');
  const [n, u] = fB2(v).split(' ');
  return withUnit(n, u);
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* --------------------------------------------------------------- state */

const S = {
  model: 'ts28m', hw: 'xspi', res: 'reef', phase: 'decode', seq: 1024, layerType: 'global', sel: null,
  p: {}, hwp: {}, whatif: false, zoom: null, fit: 0.9, bars: 'abs', grid: 'achieved', roof: 'xfer'
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
  const L = RESULTS[S.res], BP = MODELS[S.model];
  // The transposed layout changes the step cost, so that set's step is the file value.
  const BH = L.vmeStep ? { ...HARDWARE[S.hw], vme_step: L.vmeStep } : HARDWARE[S.hw];
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
  BOUNDS.host.label = HW.ext ? 'Memory · XSPI' : 'Host transfer';
  return { P, HW, BP, BH, L };
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
      ['hwp', 'hb_mhz', 'XSPI clock (MHz)', 5, 'not stated · defaults to core clock'],
      ['hwp', 'hb_bytes', 'Bytes per XSPI clock', 1, '16 bit/cycle · RTL team'],
      ['hwp', 'dma_overlap', 'DMA overlaps compute', 'bool', 'double-buffered firmware']
    ] : [['hwp', 'host_bw', 'Host→DTCM (B/cycle)', 0.5, 'measured · RTL bound 16']]),
    ['hwp', 'dbus', 'DTCM bus (B/cycle)', 4, 'confirmed · 128-bit LSU'],
    ['hwp', 'dtcm_c', 'DTCM cycles / access', 0.1, 'loads-only loop · 25 cyc / 8 loads'],
    ['hwp', 'vme_step', 'VME cycles / step', 1, L.vmeStep ? 'measured · transposed A' : 'measured · k-sweep slope']
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
    <p class="small muted" style="margin:0">Edits override the loaded files for this session. Kernel cycles stay as measured on the VME RTL; the VME step re-costs VME tiles.${HW.ext ? ' XSPI bandwidth: ' + (HW.hb_mhz * HW.hb_bytes).toFixed(0) + ' MB/s = ' + HW.host_bw.toFixed(2) + ' B/cycle.' : ''}</p>`;
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
    <svg class="abs" width="1500" height="820" viewBox="0 0 1500 820" aria-hidden="true" style="left:0;top:0"><path d="${edgePath()}" fill="none" stroke="#98a2b1" stroke-width="1.6" stroke-linejoin="round"/></svg>
    <div class="abs endpoint" style="left:40px;top:120px;width:150px">${S.phase === 'prefill' ? 'tokens in · ' + seq : 'token in · ctx ' + seq}</div>
    <div class="abs ghost-node"><span class="mono" style="font-weight:600;font-size:14px">embedding</span><span class="small muted">DMA gather</span><span class="small muted">${cur.meas ? f3(cur.meas.embedding) + ' cycles' : 'not modeled'}</span></div>
    <div class="abs endpoint" style="left:744px;top:654px;width:180px">logits · ${P.vocab}</div>
    ${nodes}`;

  const zoom = S.zoom || S.fit;
  $('world').style.transform = `scale(${zoom})`;
  $('sizer').style.width = Math.ceil(1500 * zoom + 510) + 'px';
  $('sizer').style.height = Math.ceil(820 * zoom + 60) + 'px';
  $('zoom-pct').textContent = Math.round(zoom * 100) + '%';
  $('context-label').textContent = `${P.name} · ${phase}${S.phase === 'prefill' ? ' ' + seq + ' tokens' : ' @ ctx ' + seq} · ${BH.name} · ${L.short}`;
  $('legend').innerHTML = Object.values(BOUNDS).map((b) => `<span><span class="sw" style="background:${b.color}"></span>${b.label}</span>`).join('');

  // ---- right panel
  const scrollKeep = [...$('right').querySelectorAll('.grow')].map((e) => e.scrollTop);
  const segOrder = [...RESOURCES.map((r) => r.id), OVERHEAD.id];
  const segColor = (id) => (id === OVERHEAD.id ? OVERHEAD.color : RESOURCES.find((r) => r.id === id).color);
  const segName = (id) => (id === OVERHEAD.id ? 'overhead' : RESOURCES.find((r) => r.id === id).short(HW));
  // Baseline: the loaded configs, when What-if edits are active.
  let base = null;
  if (changed) {
    const BHW = { ...BH };
    if (BHW.ext) BHW.host_bw = BHW.hb_mhz * BHW.hb_bytes / BHW.clock_mhz;
    const bq = Math.min(S.seq, BP.n_ctx);
    base = run(BP, BHW, L, S.phase === 'prefill' ? bq : 1, bq);
    base.hw = BHW;
  }
  const segSum = (segs) => segOrder.reduce((a, id) => a + segs[id], 0);
  const barSegs = (segs) => segOrder.filter((id) => segs[id] > 0)
    .map((id) => `<span style="width:${(segs[id] / segSum(segs) * 100).toFixed(3)}%;background:${segColor(id)}" title="${segName(id)} ${fmtInt(segs[id])} cycles"></span>`).join('');

  function donut(segs, total, W, H, r, compact) {
    const cx = W / 2, cy = H / 2, sw = r * 0.42, C = 2 * Math.PI * r, sum = segSum(segs);
    let acc = 0, arcs = '';
    const labels = [];
    for (const id of segOrder) {
      if (!(segs[id] > 0)) continue;
      const w = segs[id] / sum;
      arcs += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${segColor(id)}" stroke-width="${sw}" stroke-dasharray="${(w * C).toFixed(3)} ${C.toFixed(3)}" stroke-dashoffset="${(-acc * C).toFixed(3)}" transform="rotate(-90 ${cx} ${cy})"><title>${segName(id)} ${fmtInt(segs[id])} cycles</title></circle>`;
      const ang = (acc + w / 2) * 2 * Math.PI;
      labels.push({ id, w, right: ang <= Math.PI, ax: cx + Math.sin(ang) * (r + sw / 2 + 3), y: cy - Math.cos(ang) * (r + sw / 2 + 10) });
      acc += w;
    }
    const gap = compact ? 37 : 26;
    for (const side of [true, false]) {
      const g = labels.filter((l) => l.right === side).sort((a, b) => a.y - b.y);
      g.forEach((l, i) => { l.y = Math.max(l.y, i ? g[i - 1].y + gap : 12); });
      for (let i = g.length - 1; i >= 0; i--) g[i].y = Math.min(g[i].y, (i < g.length - 1 ? g[i + 1].y - gap : H - 16));
    }
    const text = labels.map((l) => {
      const x = cx + (l.right ? 1 : -1) * (r + sw / 2 + 8), a = l.right ? 'start' : 'end';
      return `<text x="${x}" y="${l.y}" text-anchor="${a}" class="d-n" fill="${segColor(l.id)}">${segName(l.id)}</text>` + (compact
        ? `<text x="${x}" y="${l.y + 12}" text-anchor="${a}" class="d-v">${f3(segs[l.id])}</text><text x="${x}" y="${l.y + 24}" text-anchor="${a}" class="d-v">${p2(l.w)}</text>`
        : `<text x="${x}" y="${l.y + 12}" text-anchor="${a}" class="d-v">${f3(segs[l.id])} ${p2(l.w)}</text>`);
    }).join('');
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Time by segment">${arcs}${text}
      <text x="${cx}" y="${cy + 2}" text-anchor="middle" class="d-t" style="font-size:${(r * 0.36).toFixed(1)}px">${f3(total)}</text>${compact ? '' : `<text x="${cx}" y="${cy + r * 0.36}" text-anchor="middle" class="d-v">cycles</text>`}</svg>`;
  }

  // Roofline: y matches the table's Achieved (used ops over total cycles); x and
  // the memory roof come from the selected memory level.
  function roofline() {
    const W = 386, H = 214, L0 = 36, R0 = W - 6, T0 = 8, B0 = H - 30;
    const res = (id) => RESOURCES.find((r) => r.id === id);
    const mem = res(S.roof), peak = cur.head[0].m.peak_rate;
    const B = peak[mem.id], pV = peak.vme, pR = peak.rvv;
    const lg = (v) => Math.log10(Math.max(1e-6, v));
    const X = (v) => Math.min(R0, Math.max(L0, L0 + (lg(v) + 2) / 5 * (R0 - L0)));
    const Y = (v) => Math.min(B0, Math.max(T0, B0 - (lg(v) + 2) / 5 * (B0 - T0)));
    const xt = [0.01, 0.1, 1, 10, 100, 1000], yt = [1000, 100, 10, 1, 0.1, 0.01];
    const xl = (v) => (v >= 1000 ? '1K' : String(v));
    const x0 = Math.max(0.01, 0.01 / B);
    // Modules within a few px share one dot; rows are sorted, so the first is the largest.
    const groups = [];
    for (const r of rows) {
      const m = r.o.m, eng = r.o.engine === 'VME' ? 'vme' : 'rvv';
      const ops = m.used[eng], x = ops / m.used[mem.id], y = ops / m.total_cycles;
      const px = X(x), py = Y(y), on = !!sel && r.key === sel && (!r.t || r.t === lt);
      const name = r.key === 'attention' ? 'attention ' + r.t : r.key;
      const g = groups.find((d) => d.eng === eng && Math.hypot(d.px - px, d.py - py) < 4);
      if (g) { g.names.push(name); if (on) { g.on = true; g.r = r; } }
      else groups.push({ eng, px, py, x, y, on, r, names: [name] });
    }
    const dots = groups.sort((a, b) => a.on - b.on).map((g) =>
      `<button type="button" class="rp${g.on ? ' on' : ''}${sel && !g.on ? ' dim' : ''}" data-sel="${g.r.key}" data-layer-set="${g.r.t || lt}" aria-label="${esc(g.names.join(', '))}" title="${esc(g.names.join(', '))} · ${g.x.toPrecision(4)} ops/B · ${g.y.toPrecision(4)} ops/cyc" style="left:${(g.px / W * 100).toFixed(2)}%;top:${(g.py / H * 100).toFixed(2)}%;background:${res(g.eng).color}"></button>`).join('');
    return `<div class="sec-row"><div class="sec">Roofline</div><div class="seg mini">${segHTML(['dtcm', 'xfer'].map((id) => [id, res(id).short(HW)]), S.roof, 'roof')}</div></div>
      <div class="roofc"><svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Roofline">
        <g stroke="var(--line)">${xt.map((v) => `<line x1="${X(v)}" y1="${T0}" x2="${X(v)}" y2="${B0}"/>`).join('')}${yt.map((v) => `<line x1="${L0}" y1="${Y(v)}" x2="${R0}" y2="${Y(v)}"/>`).join('')}</g>
        ${xt.map((v, i) => `<text x="${X(v)}" y="${B0 + 13}" text-anchor="${i === 5 ? 'end' : 'middle'}" class="d-v">${xl(v)}</text>`).join('')}
        ${yt.map((v) => `<text x="${L0 - 5}" y="${Y(v) + 4}" text-anchor="end" class="d-v">${xl(v)}</text>`).join('')}
        <text x="${(L0 + R0) / 2}" y="${H - 3}" text-anchor="middle" class="d-v">ops / ${mem.short(HW)} byte</text>
        <text x="${L0 + 4}" y="${T0 + 11}" class="d-v">ops/cyc</text>
        <path d="M${X(x0)} ${Y(B * x0)} L${X(pV / B)} ${Y(pV)}" fill="none" stroke="${mem.color}" stroke-width="1.6"/>
        <path d="M${X(pV / B)} ${Y(pV)} L${R0} ${Y(pV)}" fill="none" stroke="${res('vme').color}" stroke-width="1.6"/>
        <path d="M${X(pR / B)} ${Y(pR)} L${R0} ${Y(pR)}" fill="none" stroke="${res('rvv').color}" stroke-width="1.4" stroke-dasharray="5 3"/>
        <text x="${R0 - 2}" y="${Y(pV) + 12}" text-anchor="end" class="d-n" fill="${res('vme').color}">VME ${f3(pV)}</text>
        <text x="${R0 - 2}" y="${Y(pR) - 4}" text-anchor="end" class="d-n" fill="${res('rvv').color}">RVV ${f3(pR)}</text>
        <text x="${X(0.1 / B) + 10}" y="${Y(0.1) + 15}" class="d-n" fill="${mem.color}">${mem.short(HW)} ${f3(B)} B/cyc</text>
      </svg>${dots}</div>`;
  }

  if (!sel) {
    const sumSegs = (list) => {
      const out = Object.fromEntries(segOrder.map((id) => [id, 0]));
      for (const [o, n] of list) for (const id of segOrder) out[id] += o.m.segments[id] * n;
      return out;
    };
    const instances = (r) => (r.o.id === 'attention' ? cur.counts[r.t] : r.o.id === 'ln_f' || r.o.id === 'lm_head' ? 1 : P.layers);
    const segs = sumSegs(rows.map((r) => [r.o, instances(r)]));
    const baseOf = (r) => {
      if (!base) return null;
      if (r.key === 'attention') return base.res[r.t].find((x) => x.id === 'attention');
      return base.head.find((x) => x.id === r.key) || base.res.global.find((x) => x.id === r.key);
    };
    const baseInst = (r) => (r.key === 'attention' ? base.counts[r.t] : r.t === '' && (r.key === 'ln_f' || r.key === 'lm_head') ? 1 : BP.layers);
    const baseTot = (r) => { const o = baseOf(r); return o ? o.tc * baseInst(r) : 0; };
    let baseSegs = null;
    if (base) {
      baseSegs = Object.fromEntries(segOrder.map((id) => [id, 0]));
      for (const r of rows) { const o = baseOf(r), n = baseInst(r); if (o) for (const id of segOrder) baseSegs[id] += o.m.segments[id] * n; }
    }
    const scale = Math.max(rows[0].tot, ...(base ? rows.map(baseTot) : [0]));
    const hz = HW.clock_mhz * 1e6;
    const rowName = (r) => (r.key === 'attention' ? (P.alternating ? 'attn ' + r.t : 'attention') : r.key) + (instances(r) > 1 ? ' ×' + instances(r) : '');
    const peakCell = (r, res) => {
      const m = r.o.m, u = m.used[res.id];
      if (u == null) return null;
      return (u / m.total_cycles) / m.peak_rate[res.id];
    };
    const ratioCell = (r, res) => {
      const m = r.o.m, q = m.required[res.id], u = m.used[res.id];
      return q > 0 && u != null ? u / q : null;
    };
    const maxRatio = Math.max(1.0001, ...rows.flatMap((r) => RESOURCES.map((res) => ratioCell(r, res) || 0)));
    const violet = (a) => `background:rgba(167,139,250,${(a * 0.8).toFixed(3)})`;
    const gridCell = (r, res) => {
      if (S.grid === 'achieved') {
        const v = peakCell(r, res);
        if (v == null) return '<span class="gc">—</span>';
        if (v === 0) return '<span class="gc">0</span>';
        const num = v * 100 >= 10 ? (v * 100).toFixed(0) : (v * 100).toFixed(1);
        return `<span class="gc" style="${violet(Math.min(1, v))}">${num}</span>`;
      }
      const v = ratioCell(r, res);
      if (v == null) return '<span class="gc">—</span>';
      return `<span class="gc" style="${violet(Math.min(1, (v - 1) / (maxRatio - 1)))}">×${f3(v)}</span>`;
    };
    $('right').innerHTML = `
      <section class="popup grow" aria-label="Model">
        <div class="ph">
          <div class="ph-top"><span class="ph-big">${base ? f3(base.total) + ' → ' : ''}${f3(cur.total)} cycles${base ? ` <small>×${(cur.total / base.total).toFixed(2)}</small>` : ''}</span><span class="ph-time">${t3(cur.total, HW)}</span></div>
          <div class="ph-sub">${S.phase === 'decode' ? (hzFmt(hz / cur.total)) + ' tok/s' : 'latency ' + t3(cur.total, HW)}${cur.meas ? ` · measured: ${S.phase === 'decode' ? 'context ' + cur.meas.ctx : cur.meas.ctx + '-token prompt'}, ${cur.meas.bw} B/cycle` : ''}</div>
        </div>
        <div class="sec">Time by segment</div>
        ${base
          ? `<div class="two">${donut(baseSegs, base.total, 190, 160, 30, true)}${donut(segs, cur.total, 190, 160, 30, true)}</div>`
          : donut(segs, cur.total, 380, 190, 56)}
        ${roofline()}
        <div class="sec-row"><div class="sec">Modules by time</div><div class="seg mini">${segHTML([['abs', 'Cycles'], ['pct', '100%']], S.bars, 'bars')}</div></div>
        <div class="mods">${rows.map((r) => {
          const w = S.bars === 'abs' ? r.tot / scale * 100 : 100;
          const bw = base && S.bars === 'abs' ? baseTot(r) / scale * 100 : null;
          return `<button type="button" class="mod" data-sel="${r.key}" data-layer-set="${r.t}">
            <span class="mn">${esc(rowName(r))}</span>
            <span class="mb"><span class="mf" style="width:${w.toFixed(2)}%">${barSegs(r.o.m.segments)}</span>${bw != null ? `<span class="mo" style="width:${bw.toFixed(2)}%"></span>` : ''}</span>
            <span class="mv">${f3(r.tot)}</span><span class="mv mp">${p2(r.tot / cur.total)}</span></button>`;
        }).join('')}</div>
        <div class="sec-row"><div class="sec">${S.grid === 'achieved' ? 'Achieved %, over module' : 'Used ÷ required'}</div><div class="seg mini">${segHTML([['achieved', 'Achieved'], ['ratio', 'Used ÷ req']], S.grid, 'grid')}</div></div>
        <div class="ugrid">
          <div class="ug head"><span></span>${RESOURCES.map((res) => `<span class="gh">${res.short(HW)}</span>`).join('')}</div>
          ${rows.map((r) => `<button type="button" class="ug" data-sel="${r.key}" data-layer-set="${r.t}"><span class="mn">${esc(rowName(r))}</span>${RESOURCES.map((res) => gridCell(r, res)).join('')}</button>`).join('')}
        </div>
      </section>`;
  } else {
    const inHead = cur.head.find((o) => o.id === sel);
    const o = inHead || layer.find((x) => x.id === sel);
    const n = inHead ? 1 : ncount(o);
    const m = o.m;
    const share = m.total_cycles * n / cur.total;
    const bo = base && (base.head.find((x) => x.id === sel) || base.res[lt].find((x) => x.id === sel));
    const BAR_W = 462;
    const timeBar = (mm, tag, scale) => {
      const segs = mm.segments, sum = segSum(segs), wf = sum / scale;
      const shown = segOrder.filter((id) => segs[id] > 0);
      const label = (id) => `${segName(id)} ${f3(segs[id])}`;
      const fits = (id) => label(id).length * 6.3 + 6 <= BAR_W * wf * segs[id] / sum;
      const spill = shown.filter((id) => !fits(id));
      return `<div>${tag ? `<div class="tb-tag">${tag}</div>` : ''}
        <div style="width:${(wf * 100).toFixed(2)}%">
          <div class="tbar">${barSegs(segs)}</div>
          <div class="tlabels">${shown.map((id) => `<span style="width:${(segs[id] / sum * 100).toFixed(3)}%">${fits(id) ? label(id) : ''}</span>`).join('')}</div>
        </div>
        ${spill.length ? `<div class="tspill">${spill.map((id) => `<span><span class="dot" style="background:${segColor(id)}"></span>${label(id)}</span>`).join('')}</div>` : ''}
      </div>`;
    };
    const bm = bo ? bo.m : null;
    const barScale = bm ? Math.max(segSum(m.segments), segSum(bm.segments)) : segSum(m.segments);
    const hasActive = RESOURCES.some((r) => m.active_cycles[r.id] != null);
    const maxCyc = Math.max(1, ...RESOURCES.map((r) => m.cycles[r.id] || 0));
    const change = (fmt, cur2, was) => {
      const a = fmt(cur2);
      if (!bm) return a;
      const b = fmt(was);
      return a === b ? a : `<small class="was">${b} →</small>${a}`;
    };
    const acc = (v) => withUnit(f3(v), 'acc');
    const cycles = (v) => withUnit(f3(v), 'cyc');
    const blank = '<span></span>';
    const rowsHTML = RESOURCES.map((r) => {
      const id = r.id, u = m.used[id], q = m.required[id], cyc = m.cycles[id], mn = m.min_cycles[id];
      const isD = id === 'dtcm';
      const x = u === 0 ? '0' : cyc != null && mn > 0 ? (cyc / mn).toFixed(1) : '—';
      const usedCell = change((v) => amount(r, v), u, bm && bm.used[id]);
      const isModeled = m.active_cycles[id] == null;
      const cycFmt = (v) => (isModeled && v > 0 ? '<span class="u">~</span>' : '') + cycles(v);
      const cycCell = change(cycFmt, cyc, bm && bm.cycles[id]);
      return `<div class="rt${!u ? ' zero' : ''}">
        <span class="rn"><span class="dot" style="background:${r.color}"></span><span>${esc(r.name(HW))}</span></span>
        <span class="num">${withUnit(f3(m.peak_rate[id]), r.rate)}</span>
        <span class="num">${usedCell}</span>
        <span class="num">${amount(r, q)}</span>
        <span class="num">${cycles(mn)}</span>
        <span class="num">${cycCell}</span>
        <span class="num">${x}</span>
        ${isD ? `<span class="rs">accesses</span>
        <span class="num">${withUnit(String(m.dtcm.c), 'cyc/acc')}</span>
        <span class="num">${change(acc, m.dtcm.used_accesses, bm && bm.dtcm.used_accesses)}</span>
        <span class="num">${acc(m.dtcm.required_accesses)}</span>
        ${blank.repeat(3)}
        <span class="rs">bytes per access</span>${blank}
        <span class="num dim">${withUnit(f3(m.dtcm.bytes_per_access), 'B/acc')}</span>
        ${blank.repeat(4)}` : ''}
        <span class="cbar"><span style="width:${((cyc || 0) / maxCyc * 100).toFixed(2)}%;background:${r.color}"></span>${mn != null ? `<i style="left:${Math.min(100, mn / maxCyc * 100).toFixed(2)}%"></i>` : ''}</span>
      </div>`;
    }).join('');
    $('right').innerHTML = `
      <section class="popup grow" aria-label="${esc(o.id)}">
        <div class="chrome"><b class="mono">${esc(o.id)}</b><button type="button" class="icon-btn" data-act="clear" aria-label="Close">✕</button></div>
        <div class="ph">
          <div class="ph-top"><span class="ph-big">${bm ? f3(bm.total_cycles) + ' → ' : ''}${f3(m.total_cycles)} cycles</span><span class="ph-time">${bm ? t3(bm.total_cycles, base.hw) + ' → ' : ''}${t3(m.total_cycles, HW)}</span></div>
          <div class="ph-sub">${esc(m.shape)} · ×${n} · ${p2(share)} of model${o.flag ? ' · scaled' : ''}</div>
        </div>
        ${roofline()}
        <div class="tbs">${bm ? timeBar(bm, 'before', barScale) + timeBar(m, 'after', barScale) : timeBar(m, '', barScale)}</div>
        <div class="rtw"><div class="rt rh"><span>Resource</span><span class="num">Peak</span><span class="num">Used</span><span class="num">Required</span><span class="num">Min cycles</span><span class="num" title="${hasActive ? 'Measured; ~ modeled' : '~ modeled'}">Cycles</span><span class="num">×</span></div>${rowsHTML}</div>
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
  const f = Math.min(1.1, (st.width - (narrow ? 32 : 510)) / 1500, (st.height - (narrow ? st.height * 0.46 + 40 : 40)) / 820);
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
  if (d.roof) S.roof = d.roof;
  else if (d.bars) S.bars = d.bars;
  else if (d.grid) S.grid = d.grid;
  else if (d.phase) S.phase = d.phase;
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
      dtcm_c: g('dtcm_cycles_per_access', 'memory.dtcm.cycles_per_access') || 3.1,
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
    dtcm_c: g('dtcm_cycles_per_access', 'memory.dtcm.cycles_per_access') || 3.1,
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
