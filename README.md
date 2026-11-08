# Reef Perf — kernel performance explorer

A static page that estimates where a transformer model spends its time on the
Reef NPU (Coral VME + RVV), block by block, from RTL-simulated kernel cycles.

Pick a **model config**, a **hardware config** and a **kernel results** set in the
nav. The canvas shows the model as a pipeline. Each block is one kernel family,
coloured by what limits it (compute, DTCM memory, host transfer, or
latency/overhead). With no block selected, the right-hand panels give the
overall prefill and decode estimates and a ranked list of where the time goes;
select a block to see its cycles, bytes moved, limits and roofline.

## How the estimate works

It is a port of `inference-engine/src/csrc/src/simulations/analysis/roofline.py`
from [LonghornSilicon/reef](https://github.com/LonghornSilicon/reef):

- matmuls are tiled 16×512×16 on the VME (prefill) or 1×512×32 on RVV (decode)
  and charged the measured tile cycles;
- attention streams 64-key KV blocks per 16-query block and is charged the
  measured `vme_attention_block` + `attention_flash_finish` cycles;
- pointwise kernels scale their measured tile;
- every tile's inputs (weights, activations, K/V blocks) come in and its outputs
  go out over the transfer path set by the hardware config (below).

Bound labels: each block puts demand on a set of resources (its compute engine,
the DTCM bus, the transfer path). Demand ÷ capacity gives each one's busy cycles,
and kernel time not explained by compute or DTCM counts as overhead. The block is
bound by whichever is largest.

The side panels use one resource list (VME, RVV, DTCM ↔ core, HyperRAM DMA). Each
module's time is split into segments: the serial transfer gets its full time, the
on-chip resource with the largest used ÷ peak gets that many cycles, and the rest
of the kernel is overhead. Required amounts come from the shape for matmuls and
attention ops; anything not derivable shows "—".

## Hardware configs

| Config | Transfer path | Source |
|---|---|---|
| `atlas · HyperRAM + DMA` (default) | Weights, KV cache and spilled activations in external HyperRAM; one DMA channel to the Coral core's TCMs, one outstanding transaction | [SoC architecture](https://lhs-soc.vercel.app/architecture) (`soc.yaml`) |
| `hardware.json · host AXI` | Host writes DTCM over the 128-bit AXI slave at the measured 15.745 B/cycle, no DMA | `simulations/analysis/hardware.json` |

The architecture site gives no HyperRAM bandwidth, clock or DMA-overlap rule, so
these are What-if parameters marked unconfirmed:

- **HyperRAM bandwidth:** the controller's `[10:0]` external interface matches an
  x8 HyperBus (DQ[7:0], RWDS, CK, CS#), which moves 2 B per bus clock (DDR). The
  bus clock defaults to the core clock: 100 MB/s, 2 B/cycle at 50 MHz. Per-transaction
  latency is not modeled.
- **DMA overlap:** off by default (serialized), since the site lists one outstanding
  transaction and does not say transfers overlap compute.
- **Clock:** 50 MHz, an open decision on the site.
- **16×16 systolic array:** the site's integrated accelerator is assumed to be the
  same unit as the Coral VME whose RTL cycles the estimate uses.

## Models

TinyStories-1M, 3M, 8M, 28M and 33M (GPT-Neo: d and layers from the
roneneldan/TinyStories-* configs, MLP 4·d, vocab 50257, local window 256,
alternating global/local layers, 16 heads) and GPT-2 small. Only d_head 32 and 64
have measured attention streams; other head sizes (TinyStories-1M/3M/8M/33M:
d_head 4, 8, 16, 48) interpolate between them and are flagged "scaled". Treat
attention for those models as a rough estimate.

## Kernel results sets

| Set | Source | GPT-Neo 28M decode @1024 / prefill 1024 (host AXI config) |
|---|---|---|
| `sim 376cd28+` (2026-10-20) | `feature/inference-engine-simulation`, RESULTS.md | 0.96 s / 2.37 min |
| `coral-npu 5fb006c+` (2026-10-21) | `feature/coral-npu-kernels`, RESULTS.md: every integer loop vectorized | 0.76 s / 1.35 min |
| `stash 4d1a0d9+` (2026-10-22) | unpushed stash on the simulation branch: vectorized pointwise and softmax | 0.75 s / 1.14 min |
| `coral + transposed` (default) | `coral-npu 5fb006c+` with matmuls at the `reef_perf/rtl_matrix.csv` microbenchmarks, A stored transposed (coralnpu 8a5e1b31): GEMM 16×512×16 = 3,336 cycles, GEMV 1×512×16 = 3,235, 25 cycles per step. Decode matmuls use the VME GEMV tile instead of RVV. Attention keeps its measured streams | |

Cycles are from Coral's `VmeCoreMiniAxi` RTL (cocotb + Verilator, coralnpu
382b5c12). The `sim 376cd28+` set only logs attention stream totals, so its
per-block cost is the total spread over 4 blocks (about 1% high). It also has no
d=512 LayerNorm run, so LayerNorm is scaled from 4×768 and flagged "scaled".

Not measured: the clock (50 MHz, unconfirmed), host-side K-split sums and
orchestration, contention between host writes and a running kernel. DTCM byte
counts are derived from loop structure.

## Importing configs

**Model**: the repo's `model.json` / `gpt_neo_28m.json` (a `"model"` object with
`d_model`, `n_head`, `d_head`, `d_ff`, `layers`, `vocab`, `n_ctx`, `window`,
`attention_layers`) or a `reef.model/v1` file with a `"params"` object.

**Hardware**: the repo's `hardware.json` (leaves of `{value, units, source,
confirmed}`: `clock_mhz`, `host_to_dtcm_bytes_per_cycle`, `dbus_bytes_per_cycle`,
optional `vme_cycles_per_matmul_step`) or `reef.hw/v1` (nested `memory`,
`interconnect`, `engines`).

Imported files stay in the browser; nothing is uploaded.

## Run locally

    python3 -m http.server 8000

No build step; `index.html`, `style.css` and `app.js` are the whole site.
