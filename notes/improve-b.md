# improve-b — the bottleneck's small-M GEMMs

A research branch, opened 2026-09-25 at the owner's request, for the one lever the small
network frames `min_extent` allows left standing: the bottleneck's GEMMs. At 192x128 the
eight bottleneck blocks are about 30 % of the graph; their GEMMs have 16-64 rows and K up
to 4096, and run at 30-45 GB/s of weight streaming against a machine ceiling of 70-91.

## What bounds them (measured on the staged kernel, 64x1024x4096, 200 us)

- Taking the global loads out (constants in their place): A 165 us, B 140, both 119.5. So
  even with no memory traffic a 32-deep K step costs ~0.93 us of shared-memory stores,
  barriers and fragment loads, and with 32 workgroups on 8 cores nothing hides it.
- The three kernels on the bottleneck's shapes, bit-identical to each other: staged is the
  fastest everywhere but 16x3072x1024, where tiled and base tie it (16x1024x4096: staged
  188 us, tiled 399, base 401; 64x1024x4096: 200, 703, 775).

## A deeper K step (tried)

`STAGED_BK` makes the step a build parameter; the loaders cover an A row 32 columns at a
time and the fragment loop runs k in the same order, so every variant gives the same bits
(checked on six shapes). Standalone:

| shape | BK 32 | BK 64 | BK 128 |
| --- | ---: | ---: | ---: |
| 64x1024x4096 | 196-203 | 168-169 | 273 |
| 64x4096x1024 | 93-95 | 118-122 | 272 |
| 64x3072x1024 | 97-98 | 112-124 | 204-214 |
| 64x1024x1024 | 48-49 | 44-47 | 63-64 |
| 144x512x512 | 36-37 | 33-37 | 69-70 |

64 wins only where there are few workgroups (32 of them at N=1024), 15 %; with 128 it
halves occupancy (16 KB of shared memory) and loses. 128 loses everywhere — past 64 the
step's registers spill. The time of a step grows with its work, not with its barriers:
the loop is not waiting on barriers. As a routed variant for K >= 2048 and <= 64
workgroups it would be worth about 0.26 ms a frame (the FFN's down projection, eight
blocks). Kept as a parameter, not routed.

## Not repeated

Register prefetch in the staged kernel was measured on exactly this shape before
(`improve-fusions.md`): twice as slow. Split-K changes the order of summation.

## First, how many tokens the bottleneck has

**64 at 320x320, not 25.** Every level is padded to a multiple of 8 before it is halved
(`nr_frame_resident.py`, the level table), so 320 goes 20 -> 24 -> 12 -> 16 -> 8: an 8x8
bottleneck. At the vendor's minimum extent the pad to 64 rows costs nothing, because
there is none. Fewer than 64 tokens needs `min_extent` below 320: 16 at 256x128 and at
192x128, 32 at 320x192 and 256x192.

## A kernel without the shared-memory round trip (tried, not routed)

`src/gpu/gemm_smallm.comp`: one subgroup per 16x16 block of the output, fragments straight
from memory, two K steps of loads in flight, B packed so a 16x16 tile is 512 contiguous
bytes. Bit-identical to the staged kernel (same fragments, same order). Three things
decided it:

- **It wins only on weights that are still in cache.** Timed the way the other kernels
  were — the same B every call — 16x1024x4096 is 126-146 us against staged's 188 at 16
  rows. Rotating eight copies of B, so it comes from DRAM as in a frame, the four
  bottleneck GEMMs at 16 rows take 400 us a block against staged's 423 at 64 rows. At 32
  rows 451, at 48 rows 586. Nothing to route.
- **Registers cap the loads in flight at two steps.** Four steps spilled (11:26, then
  27:48 with the addresses spelled out) and ran twice as slow; three do not divide K.
  Thirty-two rows a subgroup spilled 49:75. An array of cooperative matrices indexed in a
  loop compiled a third slower than the same tiles written out by macro.
- **SIMD16 does not buy registers.** ANV compiles a cooperative-matrix shader SIMD32 unless
  the pipeline asks for a size (`anv_fixup_subgroup_size`); with
  `requiredSubgroupSize = 16` it is SIMD16 and bit-identical, but a matrix takes the same
  bytes at either width, four stages spill just the same, and two stages take twice the
  sends and run 25 % slower.

Little's law says the rest: at two steps a subgroup holds ~2 KB of loads in flight, 64
subgroups hold 128 KB, and at ~1 us of loaded latency that is the 127 GB/s they reach.

## The weights are not what the bottleneck waits for

Reading half of B's bytes (a timing proxy: each load's address halved) on cold weights:
64x4096x1024 105 -> 87 us, 64x3072x1024 84 -> 78, the other two unchanged — 426 -> 395 us a
block, 0.25 ms a frame at best before paying for any decode. Storing the weights as their
E4M3 bytes would buy that; `improve-fusions.md`'s proxy found the same. Neither a 320 MB
footprint of rotating weights (TLB) nor the real weights against random ones moves any
of it. In a frame the four take ~540 us a block against 426 standalone; the epilogues are
not the difference (the gate and E4M3 publish cost nothing measurable) and the rest of it
is not explained.

## Kept: a 32-row block for a bottleneck of 32 tokens or fewer

The staged kernel with `STAGED_BM=32` (and `STAGED_BK=64` where N <= 1024 and K allows it),
routed by libxmx for M <= 32; the bottleneck of 32 tokens or fewer is padded to 32 rows,
not 64. Each row's sums are its own, so the output is the 64-row block's byte for byte
(`src/gpu/test_staged32.py`, 50 cases, and a 32-token case in `test_gemm_qkv.py`).
Standalone, cold weights, the four GEMMs at 32 rows: 32-row block 132 + 109 + 69 + 37 us
against 186 + 107 + 85 + 52 on the 64-row one. Replayed graph, three alternating rounds,
heads unchanged:

| network | before | after |
| --- | ---: | ---: |
| 192x128 | 13.4-14.2 ms | 12.1-13.2 |
| 256x128 | 13.8-14.7 | 13.1-13.5 |
| 256x192 | 16.9-17.9 | 16.6-16.9 |
| 320x192 | 20.1-20.9 | 19.7-19.8 |

Only `min_extent` below 320 reaches it; at the default the bottleneck is 64 tokens and
nothing changes. `XMX_STAGED32=0` is the comparison.
