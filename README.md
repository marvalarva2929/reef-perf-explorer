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

The cycles and tok/s are **measured end to end on reef-perf**
(`yuvesh-kernel-integration` a660183, M3 core + Zvt matrix engine + DMA + host
boot) running the `e2e-npu-firmware` branch's firmware
(`inference-engine/coral/e2e`): 16-token prompt, then decode. The module
breakdown is measured too (see Kernel results).

## Models

TinyStories-28M and 33M (GPT-Neo: d and layers from the roneneldan/TinyStories-*
configs, MLP 4·d, vocab 50257, local window 256, alternating global/local
layers, 16 heads).

## Kernel results

One set, `reef-perf e2e (measured)`: every module's cycles measured end to end
on reef-perf, with an mcycle read after each kernel. Runs cover XSPI bandwidths
0.25, 0.5, 1, 2, 4 and 8 B/cycle; prefill is a 16-token prompt (28M in blocks of
8 tokens, 33M in blocks of 4, to fit the 32 KB DTCM); decode is measured at
contexts 17, 128, 256, 512, 1024 and 2048. The app shows the run nearest the
What-if bandwidth and the selected context. Residual adds and GeLU are fused into
attn_out, fc1 and fc2, so they show 0. The resource columns (VME, RVV, DTCM, DMA
use) are still modeled from loop structure, not measured.

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
