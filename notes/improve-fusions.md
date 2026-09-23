# ProjectsCodex's fusions, ported: 18 % at 720p, bit-identical

ProjectsCodex fused three things in its own tree between 2026-09-13 and 09-16 and
measured each. None of it reached this repository: that tree is a separate git history
with no common ancestor, and `improve` had since rewritten the same files — FFN batching,
compact I/O, joint QKV, the unmapped memory path. So this is a port onto `improve`'s code,
not a merge, and it was measured again here rather than trusted.

## What moved

| | Codex's phase | what it removes |
|---|---|---|
| dense residual in the GEMM epilogue | 38 | the float32 branch write, its read, one pass per projection |
| window residual, same, with the window reverse | 39 | the above, plus the residual's own gather back into the image |
| window attention QK^T, softmax and PV in one pass | 42 | the scores and probabilities' two round trips through memory |

These are not dispatch-count reductions for their own sake. `phase45` found every pass that
only moves data already at the memory ceiling; each fusion here deletes a buffer's round
trip, which is why they pay where cutting dispatches alone does not.

## Measured on this tree

Paired, alternating, all three on against all three off, every head bit-identical:

| output | off | on | gain |
|---|---|---|---|
| 384x384 | 82.37 ms | 69.24 ms | **15.9 %** |
| 1280x720 | 470.97 ms | 386.66 ms | **17.9 %** |
| 1920x1080 | 1020.37 ms | 828.90 ms | **18.8 %** |
| 384x384, `XMX_STAGING=1` | | | 14.1 % |

Ranges do not overlap at any size. Dispatches 1128 -> 864, on top of `improve`'s FFN
batching; the two compose. Codex measured the same three at about 18 % in its own tree
(497 -> 407 ms at 720p), and an independent re-run there reproduced it: 496.96 -> 406.77.

## Two things a straight copy would have broken, silently

**The graph-cache key.** Codex keys its fusions at bits 5-7. Here those are `input_fp16`,
`compact_head` and `joint_qkv`. Copied as they were, a graph captured under one setting
would replay for another — a wrong picture and no error anywhere. They take bits 8-11, and
`test_ffn_batch.py` now demands 4096 distinct keys across every combination.

**The publish order.** `gemm_resident.comp` here publishes — rounds to E4M3, applies the
gate — on the accumulator before storing. That is right for everything else and wrong for
a residual, which the two-pass path adds *before* rounding. So the residual has its own
path: raw accumulator through shared memory, then residual, publish, store.
`gemm_staged.comp` already staged the raw value and needed only the addition.

## The scratch arena, checked by hand

The fused kernels write their output while still reading their input, which the two-pass
path never did. The arena aliases roles with disjoint lifetimes, so a fusion could write
into memory its own input occupies. Every site was checked: input and output hold
different roles in all six residuals, and the fused attention writes `context`, not
`merged16`, because `merged16` shares `q16`'s role and another workgroup may still be
reading Q. Codex had made that last choice for the same reason; the role table is
identical in both trees.

## The head merge, added after (2026-09-23)

The merged-output mode of `window_attention.comp` was left behind at first: off in Codex's
tree and unmeasured there. A per-pass profile put it back on the table — `merge heads`
was 16.3 ms of a 382 ms frame at 1280x768, a pass that reads the FP32 context only to
publish it as E4M3 in (window, token, C) order, which the attention's own store can do.
Paired, alternating, `NR_FUSE_ATTENTION_MERGE` off against on, every head bit-identical:

| output | off | on | gain |
|---|---|---|---|
| 384x384 | 69.25 ms | 67.40 ms | 2.7 % |
| 1280x720 | 386.25 ms | 370.08 ms | **4.2 %** |
| 1920x1080 | 838.38 ms | 810.21 ms | 3.4 % |
| 384x384, `XMX_STAGING=1` | 66.90 ms | 65.41 ms | 2.2 % |

A second 720p run gave 384.53 -> 370.87. 62 dispatches fewer, 864 -> 802. The merged store
goes into `context16`, a new name in the scores' arena role, which the fused path leaves
unused — not into `merged16`, for the reason above — so it costs no memory.

The benchmark also showed the head read 1.5-2x slower with the merge on (720p 4.2 -> 6-7 ms,
1080p 8.7 -> 16.7). It did not survive a direct probe: 3.99 against 3.98 ms over twelve
alternating frames each, with and without the benchmark's per-frame comparison. The graph
saving did: 13.6 ms at 720p there too.

## Left behind, deliberately

- `window_attention_qkv.comp`: off by default in Codex's tree (`NR_FUSE_QKV_ATTENTION`) and
  not measured there as enabled. It reads the FP32 QKV projection inside the attention;
  normalising in the QKV GEMM's own epilogue removes more traffic, and is next.
- `DIRECT_EPILOGUE`: an experiment there, and `improve` already stores float32 epilogues
  straight from the accumulator.
- phase37's paired cosine conversions: measured 0.6 % *slower* by Codex and never enabled.

## Switches

`NR_FUSE_RESIDUAL=0`, `NR_FUSE_WINDOW_RESIDUAL=0`, `NR_FUSE_WINDOW_ATTENTION=0` and
`NR_FUSE_ATTENTION_MERGE=0` each restore their separate passes, which are also the
reference the fused one is bit-identical to. The kernel tests are Codex's: 120 dense
residual cases, 192 window cases with padding and shifted windows, and 48 attention cases,
each in both output layouts, covering zero, negative zero, subnormals, the clamp
boundaries and every finite half as a bias. All of them pass under `XMX_STAGING=1` too —
they did not until 2026-09-23, because they wrote buffers through `.view()`, which an
unmapped buffer refuses.
