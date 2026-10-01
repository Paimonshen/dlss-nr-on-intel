# Phase 71 — Intel's own driver on Windows: the first session

2026-09-29/30. The `windows` branch on this machine under Windows 11, built with MSYS2's UCRT64
gcc 16.2 and the Vulkan SDK 1.4.357, against Intel's driver **101.8991**, replaced by **101.9033**
(WHQL, released 2026-09-29) on the second day. `docs/WINDOWS.md`'s checklist was written on
Linux, and this is what the machine said. Unless marked, a finding was made on 101.8991 and
checked again on 101.9033. The last section is from 2026-10-01, on 101.9033, with Linux's
captures beside it.

## The configuration is there

`coopmat_probe`, **verified** on both drivers:

| idx | M | N | K | A | B | C | Result |
|---|---|---|---|---|---|---|---|
| 0 | 8 | 16 | 16 | fp16 | fp16 | fp32 | fp32 |
| 1 | 8 | 16 | 32 | uint8 | uint8 | uint32 | uint32 |
| 2 | 8 | 16 | 32 | sint8 | sint8 | sint32 | sint32 |
| 3 | 8 | 16 | 16 | fp16 | fp16 | fp16 | fp16 |

That is four configs to Mesa's six. There is no bf16 (`VK_KHR_shader_bfloat16` is absent) and
no `VK_NV_cooperative_matrix2`. `cooperativeMatrixRobustBufferAccess` is **true** here; it is
false on Mesa. Subgroups are 16-32, and libxmx pins 32 without its warning. A probe written into
`window_attention.comp` read back 32 lanes and eight subgroups. Shared memory is capped at 48 KB
a workgroup.

## The compiler folds packHalf2x16's round trip

`src/bench/half_probe.py` over 360 704 values (ordinary, half-subnormal, far below, overflow):

| spelling | Mesa | Intel's Windows compiler |
|---|---|---|
| bit-twiddled | exact | exact |
| `unpackHalf2x16(packHalf2x16(x))` | exact | **360 428 left in float32** |
| `float(float16_t(x))` | folded (the old `publish.glsl` comment) | exact |

The other 276 values were halves already. This is a fold, not a rounding mode: asking for
`RoundingModeRTE` and `DenormPreserve` through `SPV_KHR_float_controls` changes nothing. So
`half_round` rounded nothing, and every vendor rounding point in the graph vanished. **17 of 34
CTest checks failed**, among them `gpu_resident`'s gate and half operators in all four magnitude
regimes. The two drivers keep opposite spellings, so neither is portable. The bit-twiddled one
is, at ten instructions and two branches against two.

**Fix:** `half_round` takes its spelling from specialization constant 1, `HALF_BY_CAST`. libxmx
adds it to every pipeline from `VkPhysicalDeviceDriverProperties::driverID`: the cast on
`VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS`, packHalf2x16 everywhere else. `XMX_HALF_ROUND=pack` or
`cast` overrides it. With the cast, **30 of 33** pass (see below for `gpu_window_attention`).
Forced back to pack on 101.9033, the gate and half operators fail again. This has not yet run on
Mesa, where the specialised code should be what it was.

## shaderInt64 was never enabled

The validation layer said so on its first run here. Nine shader modules declare SPIR-V's Int64
capability, because their buffer-reference arithmetic is `uint64_t`, and the device was created
without `shaderInt64` (`VUID-VkShaderModuleCreateInfo-pCode-08740`). Mesa never minded. libxmx
now asks for the feature where the device has it, and nothing else changed.

## What still differs

- **The sign of a zero.** In `test_gemm_qkv.py`, K differs at 2 of 245 824 values: `-0.0`
  against `0.0` (1 window, 240 tokens, C 1024, mask 7). In `test_gemm_residual.py`, one byte of
  131 136 differs, the sign byte of a half. PR #3's author reports the same failure on a B580, so
  it comes from the driver, not his card. `SignedZeroInfNanPreserve` does not change it.
- **The ViT attention against its unfused passes.** In `test_global_attention.py`, 378 of 65 536
  values differ at specialization mask 7, and mask 0 agrees. The fused kernel is not specialised,
  so it is the reference's specialised passes that move. The frame does not: with
  `XMX_SPECIALIZE=0` the Tekken frames below come out byte-identical to the default.

## The unmerged window attention hangs the engine

`window_attention.comp` with `MERGED_OUTPUT` false is used by the graph only with
`NR_FUSE_ATTENTION_MERGE=0`, but `test_window_attention.py` uses it in every case. On 101.8991
**it hangs the GPU from 32 windows up**: `VK_ERROR_DEVICE_LOST`, and Windows records a
LiveKernelEvent 141, an engine timeout and reset. At 1 and 6 windows it runs, and matches the
three-pass result bit for bit. At 32 it hung every time, with one head or sixteen. It is **not**
the subgroup width (see the probe above), not the bias pointer (it hangs with a live one), and not
shared memory (16 KB).

The one thing the unmerged variant does on its own is store its accumulators straight to the
buffer. Staged through shared memory instead, with the same bytes, it hangs only sometimes. One
run of the whole test passed its 48 cases, and the next hung at case 44. Both 300-submit loops
hung, and the second, with `shaderInt64` on, sat for 60 s (`VK_TIMEOUT`) before the reset. The
staging moves the hang and does not remove it, so it was not kept. The merged variant passed
300 submits at 32 windows and 16 heads, and every whole-graph test.

There were about ten engine resets over the two days, and none took the adapter down. The hang
has not been re-tested on 101.9033. On this driver, leave the test out:
`ctest ... -E gpu_window_attention`.

## Against Linux, on the same frames

Python on Windows has no `AF_UNIX`. So the comparison ran the daemon's own `serve()` in-process on
a loopback TCP socket, with `restill.py`'s protocol and settings. Its input was the Tekken 7
capture that Linux re-rendered on 2026-09-28 through the same graph: 1920x1080, network
1920x1152, history and all. On 101.9033:

| frame | pixels that differ | max | mean, levels of 255 | PSNR |
|---|---|---|---|---|
| 001 (no history) | 72 % | 24 | 0.62 | 47.5 dB |
| 002 | 68 % | 19 | 0.54 | 48.6 dB |
| 003 | 65 % | 18 | 0.49 | 49.1 dB |
| 004 | 63 % | 18 | 0.47 | 49.4 dB |

The frames are close but not identical, and the difference covers most of the frame. That is
what a different head looks like in a chaotic graph (`phase8`); it is not a few pixels at a
rounding boundary. Frame 001 has no history, so no temporal table touches it, and `nr_image.c`
makes no libm call that could differ between the two C libraries. Finding where the head first
differs needs numbers from Linux: `frame_replay.py`'s head hashes at the same commit, then per
block. *They came on 2026-10-01: the first GEMM, and why, is the last section.*

## Speed

On 101.9033, on a machine checked quiet before and after (mains power, best-performance mode,
CPU 1-2 %):

| | Windows | Linux, graph (HANDOFF, 2026-09-28) |
|---|---|---|
| 320x320, warm replay | 28.1 ms | 23.3 ms |
| 1280x720 on 1344x768, warm replay | 199.6 ms | 143 ms |
| 1920x1080, one daemon frame | 0.47 s, 406 ms on the GPU | |

The heads were `e62005b80145b97a…` at 320x320 and `c217fd2fdbbe6b79…` at 720p, the same on
every run. The kernels' shapes, shared-memory sizes and occupancy were tuned against Mesa's
compiler, and this is Intel's; it has not been profiled yet. The daemon's `--dump` costs another
1.6 s a 1080p frame here, for the PNGs.

The in-process replay, the quiet check and the probes live outside the tree, in the checkout's
`work/tools-win/`.

## Where the two drivers part: Mesa flushes float16 subnormals (2026-10-01)

Linux ran `src/bench/capture_compare.py` on the input Windows had recorded, bit for bit the same
(`b1b4ff26…`), and the two graphs parted at the stem, the first GEMM. 205 254 of its 3 276 800
float32 outputs differed, by up to 4.7e-5 (HANDOFF, 2026-10-01). The stem is
`features (pixels x 16) @ input_adapter_weight (16 x 32)`. Both operands are float16, and the
products are exact in float32, so its exact sum can be taken in float64 from the two captures
alone. Against that sum:

| | equal to the exact sum, float32-rounded | with float16 subnormals flushed in both operands |
|---|---|---|
| Windows | **98.96 %** | 92.72 % |
| Linux | 92.72 % | **98.98 %** |

**The adapter holds exactly two float16 subnormals**: [9, 21] = 2.07e-5 and [14, 7] = 2.96e-5.
Columns 21 and 7 are 204 447 of the differing values: every pixel in column 7, and every pixel
but 353 in column 21. The other 807 are the 27 pixel rows whose features hold a subnormal, in
channels 0-2. Flushing explains 205 245 of the 205 254. The values that miss the exact sum on
either side sit below it, never above. Mostly the miss is one or two ulps, more only where the
sum cancels to near zero, and never more than 1.2e-7. That is the accumulator's own
truncation, the same on both drivers: 33 240 of those values are the same bits on Linux and
Windows.

**The mechanism is Mesa's default, not the hardware.** In Mesa 26.2.3, brw writes `cr0` only
for an explicit float-controls execution mode (`emit_shader_float_controls_execution_mode` in
`brw_from_nir.cpp`). ANV never sets the interface descriptor's Denorm Mode, which stays at
`Ftz`. Our shaders declare no mode, so `cr0`'s FP16_DENORM_PRESERVE (bit 10) stays clear, and
the cooperative-matrix multiply flushes float16 subnormal operands. ANV reports
`shaderDenormPreserveFloat16 = true` and `shaderDenormFlushToZeroFloat16 = false`. Intel on
101.9033 reports both true, with `denormBehaviorIndependence = ALL`. Its compiler keeps
subnormals when nothing is declared.

Shown on Windows by declaring the mode in SPIR-V, with the shaders' code untouched
(`src/bench/denorm_mode.py`):

- every graph shader with `DenormFlushToZero 16`: the stem is **bit-identical to Linux's**;
- every graph shader with `DenormPreserve 16`: the whole graph is bit-identical to the default,
  head `e62005b8…`;
- `DenormFlushToZero` for 32-bit floats as well changes nothing beyond the 16-bit flush.

**This corrects `phase4-subnormal-flush.md`.** What it measured on 2026-09-08 was real on Mesa,
and its flush-to-zero model reproduced the GPU. But the flush is Mesa's default mode, not the
XMX units' behaviour: the same units keep float16 subnormals under Intel's driver. Whether Mesa
keeps them when a shader declares `DenormPreserve 16` has not been run yet.

**The vendor's hardware keeps them.** NVIDIA's tensor cores support subnormal inputs and outputs
on every architecture measured, from V100 to B200 and the RTX PRO 6000 (Khattak and Mikaitis,
*Accurate Models of NVIDIA Tensor Cores*, arXiv:2512.07004). So on this point Windows computes
what the vendor's GEMM computes, and Mesa's default does not. The scale is small. Only 7 of the
model's 145.8 M weights are float16 subnormals: the two above, two each in block 70's
`out_conv_weight` and `out_gain`, and `block33.layer3.attention_scalar`. Inputs are the other
source, as on these 27 rows. No E4M3 publish can produce one, because E4M3's smallest step,
2^-9, is far above half's smallest normal value. In a chaotic graph, though, that small
difference is enough to change every head.

**With the flush the same, they part next in block 0**, as Windows with `DenormFlushToZero 16`
shows against Linux's default: 3 526 values, in 118 of 102 400 pixels, 30-32 of the 32
channels each, in 109 of the 1 600 windows. One pixel at a time, then: a per-pixel step, which
the flush does not explain. Block 0 is two passes in the graph, the fused feed-forward and the
fused window block, and nothing between them is stored. `src/bench/block0_probe.py` runs block 0
on a capture's input as the graph does, and again with every fusion off, keeping what each pass
writes. It saves both in `capture_compare.py`'s format. On Windows the fused and unfused runs
agree bit for bit, with and without the flush. Linux's run of it names the pass.
