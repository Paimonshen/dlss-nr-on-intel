# The resident runtime: 16-30x, and the phase13 thesis confirmed

2026-09-09

`notes/phase13-torch-and-blas.md` concluded that the XMX path loses to a good CPU BLAS
not because the kernel is slow — it measures 1348 GFLOP/s and beats OpenBLAS on every
shape — but because every GEMM round-trips its activation through host memory. That
made residency the precondition rather than an optimisation. This is the first slice of
it, and the thesis holds: **the same arithmetic, kept on the device, runs 16-30x faster
than the fair CPU baseline.**

## The design: pointers, not descriptors

The device reports `bufferDeviceAddress = true` and 256 bytes of push constants. So
operands travel as 64-bit device addresses inside the push constants and there are no
descriptor sets at all. Recording a pass is push-and-dispatch, which means a whole
chain goes into **one command buffer with one fence at the end** instead of one submit
per GEMM.

`coopMatLoad` reads through a `buffer_reference` — worth checking before designing
around it, and it compiles and runs.

```
src/gpu/gemm_resident.comp   the cooperative-matrix GEMM, operands by pointer
src/gpu/resident.comp        e4m3, quadratic gate, half rounding, f32<->f16, scale, residual
libxmx.c                     buffer pool, recording, one submit
src/gpu/xmxres.py            Runtime / Buffer
```

Because the APU's memory is shared and the pool is HOST_CACHED, `Buffer.view()` hands
back a numpy array over the same bytes the GPU reads. Feeding an input or reading an
output is an address, not a transfer.

## The subnormal rescale turned out to be unnecessary — and phase4 was wrong

`xmx.py` rescales both operands by a power of two before every GEMM, because
`notes/phase4-subnormal-flush.md` found **27.22 %** of parameters were FP16 subnormal
and XMX flushes those to zero. Measured on the *correctly decoded* logical weights and
on a real frame's activations:

| | float16-subnormal |
|---|---|
| GEMM activations | 0.0015 % |
| GEMM weights | 0.00006 % |

**The 27 % was an artifact of the wrong decode** — reading packed E4M3 and permuted
bytes as dense FP16 produces garbage that is largely subnormal. Flushing every
subnormal operand to zero for a whole frame moves the head by 8.08 % of its sd, *below*
the 11-12 % floor the FP16 path already sits on. So the resident path needs no
max-reduction, no scale buffers and no rescale at all, and it says so rather than
carrying machinery for a problem that does not exist.

## A trap: `float(float16_t(x))` is not a rounding

The shader's half rounding was written as the obvious round trip. The compiler folds it
away, the value stays float32, and **every one of the vendor's rounding points silently
disappeared** — the gate came out as the pure float32 expression. It is not a small
error: the graph's whole character lives in those roundings.

Replaced with the explicit form on the float32 exponent and mantissa — round-half-even
at bit 13, the subnormal range at its fixed 2^-24 step, overflow to infinity — which is
the algorithm already verified bit-exact in numpy and cannot be elided. `precise` is
needed on the gate for the same class of reason: an FMA contraction would skip the
rounding between the multiply and the add.

## What it does

`src/gpu/test_resident.py`. The operators are **bit-identical** to `nr_model` over four
magnitude regimes including the inf/NaN one: e4m3, the quadratic gate, half rounding.

A whole feed-forward — `to_half, gemm, gate, e4m3, to_half, gemm, residual` — is
**7 passes in one submit**. Against the reference given the same half inputs it agrees
to 2.6e-04, which is the E4M3 publish amplifying a float32 accumulation-order
difference; against the float32 reference it differs by 7.2e-03, the known FP16-operand
effect.

Against the fair CPU baseline — OpenBLAS numpy with the torch rounding of
`notes/phase14-npu-and-rounding.md`:

| tokens | host | resident | |
|---|---|---|---|
| 4096 | 22.8 ms | 1.41 ms | **16.1x** |
| 36864 | 133.2 ms | 4.43 ms | **30.1x** |
| 147456 (block 0's extent at 384x384) | 432.4 ms | 16.3 ms | **26.5x** |

## What is still on the host

This is one block family's feed-forward, not the graph. Still to port: the bit-affine
softmax, the fragment-tree cosine normalise and publish, the window partition and
reverse, average pool, the nearest and learned upsamples, the decoder merges, and the
concatenations the branched and split feed-forwards need. Then the block dispatch has
to record a whole frame rather than a chain.

The arithmetic ceiling from `notes/phase11-what-is-left.md` is unchanged: 340 ms for a
720p frame at the kernel's measured throughput, so full-frame real-time remains out of
reach by about 20x. What residency buys is the distance between today's 64.4 s and that
floor.
