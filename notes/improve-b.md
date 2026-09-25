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

## Next

A kernel for small M without the shared-memory round trip: fragments straight from
memory, several K steps of loads in flight, B stored in the order its fragments load
(pairs of K rows interleaved), so a 16x16 tile is 512 contiguous bytes instead of sixteen
gathered half-rows. Where it would pay: the direct kernel (tiled) is latency-bound today —
256 dependent 16-deep steps at ~1.5 us for K=4096.
