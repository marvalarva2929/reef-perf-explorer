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

The side panels use one resource list (VME, RVV, DTCM ↔ core, HyperRAM DMA). A
module's engine row shows its measured RTL kernel cycles. Other resources show the
time they would need on its own, marked ~: used ÷ peak,
except DTCM, which is accesses × max(c, bytes per access ÷ B), with c = 3.1 cycles
per access and B = 16 B/cycle. Min cycles use the required amounts. The VME peak is
the array alone (6.9 cycles per 16×16×4 step), so operand loads count under DTCM.

The time bar sums to the module's total. Serial DMA gets its full time. On-chip
resources overlap, so only the largest is charged, capped at kernel cycles, and the
rest of the kernel is overhead. With DMA overlap on, DMA joins the on-chip group.
DTCM access counts are modeled from loop structure; pointwise and attention assume
16 B accesses.

The roofline's y axis is used ops over total cycles, the same value as Achieved in
the table. Its x axis and memory roof follow the DTCM / DMA toggle.

## Hardware config

One config, `reef-perf · XSPI + DMA`: the RTL team's data path. Weights and
(for now) the KV cache sit in XSPI flash, 16 bit per XSPI clock (~6 ns); one DMA
channel moves them into the 32 KB DTCM, double buffered so transfers overlap
compute; the host boots the 8 KB ITCM over SPI. The XSPI clock defaults to the
50 MHz core clock (2 B/cycle, 100 MB/s); change it under What-if.

The headline (cycles, tok/s) is **measured end to end on reef-perf**
(`yuvesh-kernel-integration` a660183, M3 core + Zvt matrix engine + DMA + host
boot) running the `e2e-npu-firmware` branch's firmware
(`inference-engine/coral/e2e`): 16-token prompt, then decode. It shows the run
nearest the What-if XSPI bandwidth. The module breakdown below it is still the
kernel-results estimate.

## Models

TinyStories-1M, 3M, 8M, 28M and 33M (GPT-Neo: d and layers from the
roneneldan/TinyStories-* configs, MLP 4·d, vocab 50257, local window 256,
alternating global/local layers, 16 heads). Only d_head 32 and 64
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
