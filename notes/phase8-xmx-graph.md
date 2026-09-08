# The graph runs on XMX, and the graph is chaotic
2026-09-09

Two results. The first is Phase 4's: every weight GEMM in the recovered 71-block graph
now runs on the Xe2 cooperative-matrix units. The second was found while trying to
validate the first, and is the more important one.

## The graph amplifies any perturbation to a fixed floor

`nr_model` computes in float32 with the vendor's rounding points. Running the GEMMs in
FP16 instead perturbs each one by ~3e-04 relative. The head then moves by mean 0.0174
on the RGB channels — 12 % of its own sd. That looked like a bug.

It is not. Perturbing the **input frame** by a relative 1e-06 — four orders of magnitude
below E4M3's quantum and two below FP16's — and running the *unchanged CPU reference*:

| input perturbation | head RGB mean\|d\| | max\|d\| |
|---|---|---|
| 1e-06 | 0.011783 | 0.17181 |
| 1e-04 | 0.015628 | 0.17787 |

Head RGB sd is 0.1395. A hundredfold larger perturbation produces only a 1.3x larger
divergence: **the response saturates**. The E4M3 publishes are hard quantizers with a
6.25 % step, so a relative perturbation p flips a rounding with probability ~p/0.0625
per element per publish; across ~100 publishes any p above ~1e-06 has flipped most
elements at least once, and the divergence sits at a floor of 9-11 % of the signal.

Consequences, and they are not small:

- **Bitwise CPU/GPU agreement is unattainable by construction.** So is bitwise agreement
  with NVIDIA. Any acceptance test phrased per-element is measuring this floor.
- `notes/phase4-end-to-end.md`'s "CPU and XMX agree to 9.7e-07" was measured on a
  pass-through pipeline with no E4M3 publishes in it. It does not transfer.
- The composed image damps the floor by 4x (the residual enters at 0.25 and is clipped):
  CPU vs XMX is mean 0.0044 on the image against the model's own change of 0.0261.
  MLX-DLSS report 0.0041-0.0048 MAE against NVIDIA on native game-face crops — the same
  number. The XMX path sits inside the noise floor of the whole reconstruction.
- Chunk size is part of the numerics. A different `CHUNK_TOKENS` gives a different row
  count to the BLAS, and one reassociated FMA is enough to flip a publish.

## Phase 4: the GEMMs

`src/gpu/nr_xmx.py` points `nr_model.MATMUL` at `xmx.gemm_mapped`. Every
`[..., K] @ [K, N]` above 2^20 multiply-accumulates goes to the GPU; the per-head score
and context matmuls stay on the CPU because they are genuinely batched (a different B
per head and window) and a dispatch each would cost more than numpy takes.

**Every GEMM's right-hand operand converts to FP16 losslessly.** 579 of the 649 logical
tensors are stored F16; the other 70 are the `attn_scale` vectors, which are not GEMM
operands. So the FP16 error is entirely on the activation side, and most activations
reach a GEMM already E4M3- or half-published, which is also lossless.

### The memory type was the whole story

`libxmx.c` chose the first host-visible memory type, which on this device is
`memoryTypes[1]` = DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT — uncached. `memoryTypes[2]`
adds HOST_CACHED. Reading results back through the uncached mapping ran at ~80 MB/s and
buried a 1.35 TFLOP/s kernel:

| GEMM | before | after |
|---|---|---|
| 2048x512x512 | 31.2 ms | **2.26 ms** |
| 147456x32x128 | 1073.6 ms | **10.0 ms** |

A one-line preference for HOST_CACHED. Then `xmx_reserve` hands the mapped buffers
back to Python so A is built and C is read in place — no staging copies at all on a
shared-memory APU — and prepared weights are uploaded once rather than per call.

Kernel throughput measured with `iters=50` inside one submit: **1348 GFLOP/s**.
Fixed dispatch cost, 8x16x16: 0.31 ms.

## Elementwise work, in numpy, is now the bulk of a frame

Three rewrites, each verified **bit-identical** to the form it replaces on 3.4 M random
values across four magnitude regimes:

- `e4m3`: clamping the *exponent field* at 121 produces the normal and subnormal step
  from one expression (121 - 3 = 118 is exactly the exponent of 2^-9), so there is no
  branch and no second `where`; and the reciprocal of a power of two is exact, so the
  division becomes a multiply. **111 ms -> 33 ms** per 4 M elements.
- `quadratic_gate_activation`: numpy has no SIMD path for float16 *arithmetic* and falls
  back to scalar, while the float16<->float32 conversions are hardware. Every vendor
  half operation is now float32 arithmetic with an explicit rounding pass — identical,
  because float32 holds the exact product and sum of two halves. **200 ms -> 107 ms**.
- `vendor_cosine_normalize`, `vendor_cosine_publish`, the softmax division: same change.

`CHUNK_TOKENS` default 2^18 -> **2^13**, so a chunk's working set stays in L2 across the
chain of elementwise passes. Swept on a 384x384 frame: 29.1 / 27.2 / 26.5 / 28.5 s at
2^18 / 2^15 / 2^13 / 2^11.

## Where a frame goes now

384x384 Cyberpunk face crop, network extent 384x384:

```
                 before   after
CPU  numpy        45.1 s   37.9 s
XMX               29.5 s   22.2 s     2.03x against the original CPU baseline
```

At 22.2 s: **4.86 s** GPU GEMM, **4.95 s** CPU batched attention matmuls, **12.4 s**
elementwise numpy. The GEMMs are no longer the problem. Going much below this means
moving the E4M3 publishes, the softmax and the window shuffles onto the GPU as well —
a real compute-shader port of the graph, not a hook.
