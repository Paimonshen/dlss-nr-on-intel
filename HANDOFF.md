# HANDOFF — read this first

State of the DLSS-NR on Intel Xe2 project as of **2026-09-08**. Written so a fresh
session can continue without re-deriving anything. CLAUDE.md holds the project brief;
this file holds what is actually true.

---

## 1. The goal changed

The owner redirected on 2026-09-08: **a working version that can be tested and
eventually run in a game**, ahead of CLAUDE.md's "reimplement, do not execute NVIDIA's
kernels" rule. Treat CLAUDE.md's constraints as advisory, not binding.

Acceptance test: `python3 src/ref/run_frame.py` — a frame in, a frame out, scored
against a clean reference. Target: meaningfully below 1.0x.

**Current standing: no demonstrated denoising.** The `1.02x` recorded earlier holds
only at sigma 0.06 on one synthetic pattern. Swept properly: the network adds a fixed
distortion of **0.0163** (measured on a clean frame, where there is no noise to remove)
and removes a constant **6.5 % ± 1.4 %** of additive noise across a tenfold range —
constancy being the signature of a linear filter, not a learned denoiser. Break-even at
sigma ~0.048.

**And the "attention helps, 1.39x swing" claim is withdrawn.** On a structurally
different image (`--smooth`: low-frequency sinusoids, no fixed periods) the sign
reverses — attention OFF scores 1.73x better than the input, attention ON only 1.13x,
so the branch *costs* 1.53x, reproduced at two noise levels. **Confirmed on two real
Cyberpunk frames at two noise levels: the branch costs 1.37x on average, all four
configurations.** The scaffolding alone denoises real content 1.37x-1.99x, improving
with noise as a real denoiser does; with the branch on the ratio is flat at ~1.2x. The
earlier swing was the branch repairing damage the bilinear scaffolding does to
`test_pattern`'s fixed-period lines (every 17 and 23 pixels), not denoising.
Four of five image classes say the branch hurts. See `notes/phase5-input-dependence.md`.

---

## 2. The single most important open item

**The acceptance test cannot see the network.** `run_frame.py --no-attend` disables
the attention branch in all 44 blocks that have one and reproduces every score ever
recorded for this project: 5.13x, 1.45x, 1.43x. The branches move the output frame by
**0.35 % relative** and the score by 3e-5. The "1.43x, best non-degenerate result" is
the cost of bilinear resampling through five levels plus an RMS skip blend — the
scaffolding, not the model.

It is not a missing gain. A global multiplier on the branch makes it monotonically
worse (1.43x -> 1.44x -> 1.61x -> 29.6x -> NaN at gains 1/4/16/64/256), so the branch
direction is wrong rather than its magnitude, and the "~450x branch/skip factor" would
not rescue it.

`run_frame.py` now carries a second trunk through the identical scaffolding with no
branches and prints `NETWORK-INERT` when the difference is below 0.1 %. **Run
`--no-attend` first whenever a score moves.**

The next real work is the parts that are still stand-ins — stem, output head, learned
resampling, the unapplied leading `2.5C^2 + 64C` — not further tuning of what is
already recovered.

*(Superseded: "the attention is not softmax" was the previous entry here. It was
wrong; see section 3.)*

## 3. What is solid — do not re-derive

| Fact | Where |
|---|---|
| `ref/nvngx_dlssnr.dll`, 310.8.0.0, Authenticode-verified against NVIDIA's cert | `notes/phase0-acquire.md` |
| Weights are **FP16**, not FP8 E4M3. 73,841,889 parameters, 153 tensors | `notes/phase3-weight-format.md` |
| Our decode is **byte-identical to a real RTX capture** (SHA-256 `A5513B18…BD4EE3E5`) | `notes/phase3-execution-order.md` |
| All 15 `.data` containers are **Zstandard**, decompressing to PTX sm_120, 231 entries | `notes/phase3-ptx-unlock.md` |
| Complete network spec, every parameter placed, regenerable and self-checking | `notes/MODEL-SPEC.txt`, `src/tools/model_spec.py` |
| Execution order: 156-slot graph in 14 stages, matching our blocks slot-for-slot | `notes/phase3-execution-order.md` |
| Head dim 32, Swin window 8x8, GQA 4:1 (Q=C², K=V=C²/4) | `notes/phase3-architecture.md` |
| Block layouts for all five block types | `notes/phase3-block-internals.md`, `notes/phase3-block-layout.md` |
| **XMX flushes subnormal FP16 to zero; 27.22% of the model is subnormal.** Fixed by a per-tensor 2^k rescale | `notes/phase4-subnormal-flush.md` |
| **NVIDIA accumulates in FP16**, not FP32 — zero of 218 kernels use an FP32 accumulator | `notes/phase4-accumulation-choice.md` |
| GPU path validated layer-by-layer on all three block families, worst 2.8e-04 | `notes/phase4-subnormal-flush.md` |
| End-to-end CPU vs XMX agree to 9.7e-07 over 4.2M elements | `notes/phase4-end-to-end.md` |
| Attention **is** softmax: logits hard-clamped (Swin +-6, ViT +-3), `exp` hand-rolled in f16x2 bit arithmetic, **no max subtraction** | `notes/phase5-softmax-found.md` |
| `ptx_trace.py` was dropping every `{ ... }`-scoped statement — 26k `fma.f16x2`, 48k `mul.f16x2` invisible. Fixed | `notes/phase5-softmax-found.md` |
| QK L2-norm re-derived mechanically: mma -> square -> butterfly sum -> eps -> rsqrt -> mma operand B | `notes/phase5-softmax-found.md` |
| The scaffolding, not the network, produces every score so far | `notes/phase5-visual-loop.md` |
| **`attn_scale` is H FP32 scalars, not 2H FP16**; `unknown_even` is the low half, not a parameter. Only this reading ever reaches the +-6 clamp | `notes/phase5-attn-scale-fp32.md` |
| Kernel parameter block: `+0` input, `+8` output, `+16` weight/scalar arena (FP32 scalars live *in* the weight arena), `+24/+32` dims | `notes/phase5-attn-scale-fp32.md` |
| No input normalisation inside the block kernels — all 32 `rsqrt` trace to `mma` outputs (Q/K), none to a memory load | `notes/phase5-softmax-found.md` |
| **The gate multiplies the SKIP**, fused into the mma accumulator: `D = A.B + gate(*)x`; the branch is added unscaled | `notes/phase5-gate-on-skip.md` |
| The branch is short by **~8x**, not ~450x — that figure followed from the wrong gate form | `notes/phase5-gate-on-skip.md` |
| The fused Swin block is six GEMM classes, all operands sourced | `notes/phase5-block-skeleton.md` |
| **Every weight matrix is stored 2x in the arena**; gate, bias table and scale are 1x. Exact at C=64/128/256; C=32 short by 0.5C^2, reproducing the known C=32 anomaly from an independent measurement | `notes/phase5-block-skeleton.md` |
| **head_dim = 32 at every width**, measured from the QK-norm reduction (4 lanes x 4 squares x 2). Head count = C/32, confirmed twice | `notes/phase5-narrow-blocks.md` |
| Bias and scale share one indexing scheme: `head = 4*ctaid.z + tid.y`, one 64x64 table per head | `notes/phase5-narrow-blocks.md` |
| `src/tools/ptx_addrform.py` — PTX address to linear form; validated against a hand-derived address | `notes/phase5-narrow-blocks.md` |
| **All ten U-Net resampling matrices located** by block surplus: encoder `C^2` at the back, decoder `C^2`+vector at the front, five levels. Applying them as plain projections regresses to a pass-through, so `--resample` is opt-in | `notes/phase5-resample.md` |
| **`block39` (decoder input upsample) is recovered**: `512x512` + a `512` bias (container holds `512^2+512` exactly); one GEMM, the only layer with a bias. Wiring it makes the attention branch matter for the first time — a **1.39x swing** | `notes/phase5-dec-upsample.md` |
| **The output head is recovered**: block70's last 512 are a padded `(64,8)` holding a real `32 x 4`, 384 structural zeros. `blend_scale` = 0.73975 | `notes/phase5-output-head.md` |
| block0 carries 512 extra at the FRONT, dense (zero zeros) — a different kind of object from the head | `notes/phase5-stem.md` |
| The I/O kernels take **descriptor tables**: `cg2r_post_process` has 9 buffers at stride 32 (ptr + 3 `v2.b32`), the pre-block opens with 5 pointers. Inputs are five separate tensors, not one packed image | `notes/phase5-io-structs.md` |
| **16-channel packing order measured**: ch4-6 colour RGB, ch7-9 reprojected history RGB (same affine `(x-a)*b`), ch12-14 sign-encoded validity, rest uniforms | `notes/phase5-channel-order.md` |
| The pre-block is a **whole block** (528 mma, ~10,240 weight elements) that takes 16 channels in — not a small adapter. `input_adapter` framing withdrawn | `notes/phase5-channel-order.md` |
| **The input is five optional 2D textures** read `tex.2d.v4.f32`: colour(3, the only required one), a 5-tap 3-channel plane, 2ch, a 1-channel thresholded mask (5-tap), 2ch — eleven components | `notes/phase5-input-contract.md` |
| Shared-memory APU does not affect correctness or capacity (141 MiB of weights against an 11.46 GiB heap); it caps **bandwidth**, which is already the measured ceiling | verified 2026-09-08 |
| The attention accumulator is initialised from the arena: `S = Q.K^T + table` — an additive logit bias is real, but our `attn_bias` bytes cannot be it | `notes/phase5-block-skeleton.md` |
| One weight blob, not three models: `RCDATA/WEIGHTS_HT` is the only data resource and all 73,841,889 params close in one U-Net | `src/tools/pe_resources.py` |
| All 111 `_fp8` kernels have an FP16 twin, none unpaired — the FP16 path is native, so no FP8 problem exists here | verified 2026-09-08 |
| The colour/tone path is `cg2r_post_process_kernel`: pure f32, `ex2`+`lg2` (pow), `tex.2d`, saturate — a ~370-byte scalar launch struct. The intensity controls live here, not in the blocks | verified 2026-09-08 |

---

## 4. What is still unknown

1. **Why the branch is ~8x too small** — see section 2 and `notes/phase5-gate-on-skip.md`.
   This is the blocker, and it now has a number attached to it.
2. **The `128C` region's role.** Now further constrained: because the softmax has no
   max subtraction, a near-constant -56 additive term saturates the clamp and makes
   attention *uniform*. Measured — adding it raises row entropy toward the 4.159-nat
   maximum. It is not a pre-softmax additive bias. `notes/phase3-bias-region.md`
   (whose shift-invariance argument is void), `notes/phase5-softmax-found.md`.
3. *(Resolved — `attn_scale` is FP32, see section 3.)*
4. **The leading `2.5C^2 + 64C` of each fused block.** Boundaries known, roles not.
5. **The Q/K/V split at C=32 and C=64.** 15 blocks, undetermined structure. The 29
   fused-Swin blocks also have no `attn_scale` extracted, so their attention is a bare
   cosine and uniform by construction.
6. **Stem and output head** — `input_adapter_weight`, `out_gain`, `out_conv_weight`,
   `blend_scale` named but not located.
7. **Learned resampling** — bilinear stand-ins, and per section 2 they are currently
   the dominant term in the score.

## 5. Withdrawn conclusions — do not resurrect

- **"Missing ~2^8.5 scale factor."** Inferred from a `cos*skip + sin*branch` gate form
  that the PTX rules out (no sqrt, no fma against 1.0f anywhere). Withdrawn.
- **"The 128C region is a shifted-window mask."** The test was degenerate — every
  entry counted as masked, so the "agreement" was the mask's own density.
- **"The leading region contains the output projection."** The apparent improvement
  was the branch being attenuated 11x-18x.
- **"1.01x parity."** That configuration was a pass-through, correlation +1.0000.
- **"The attention is not softmax."** The `ex2` census was right, the inference wrong:
  the exponential is hand-rolled in f16x2 and `ptx_trace.py` was not parsing it.
  `notes/phase5-no-softmax.md` is marked withdrawn.
- **"Softmax is shift-invariant, so the `128C` region's -56.2 constant has no
  effect."** There is no max subtraction, so it has a large effect — it saturates
  the clamp.
- **"5.13x -> 1.43x progress."** Real, but it was the scaffolding being tuned; the
  same numbers appear with the network disabled.
- **"`attn_scale` is stored as a log."** Right symptom, wrong mechanism — the value is
  FP32, not a log. `--exp-scale` is kept only for comparison.

---

## 6. Traps that cost time here

- **The tools skip things.** `ptx_trace.py` dropped every `{ ... }` scope — where NVCC
  puts hand-written f16x2 arithmetic — which hid the whole attention non-linearity and
  produced a confident, wrong, documented conclusion. Before trusting a census, check
  that it *covers* the input: parsed-statement count against `;` count, and whether the
  dataflow graph is connected (no `mma` consumed another `mma` — impossible in a fused
  attention kernel, and the tell that something was missing).
- **Four metrics in a row had holes**, each found by an adversarial configuration and
  none by reasoning: correlation with the input rewards inaction; dividing by a fitted
  gain becomes 0/0 on collapse; a pass-through scores exactly 1.00x; and the fourth and
  worst, the score was measuring the stand-in scaffolding rather than the model for a
  whole phase. `denoise_score` now reports COLLAPSED and PASS-THROUGH, and `run_frame`
  reports NETWORK-INERT. Distrust the next metric too.
- **Reading PTX by eye produced three wrong conclusions.** Use `src/tools/ptx_trace.py`.
  It needed two fixes to be trustworthy: vector destinations in braces, and joining
  PTX instructions split across lines (until then every `mma` parsed with no operands).
- **Statistical segmentation cannot find Q/K/V boundaries.** Two methods (column norms,
  sd profile) both failed calibration on C=512 where the answer is known. Do not trust
  a third without calibrating it first.
- `bsdtar` reads the driver SFX but silently drops every file and exits 0; carve the
  payload from the `37 7A BC AF 27 1C` signature first.

---

## 7. Layout and commands

```
ref/     nvngx_dlssnr.dll (0444) + sha256          NEVER modify, NEVER commit
work/    weights_ht.bin, modules/*.ptx, libxmx.so, build artifacts, frames
notes/   18 findings documents, MODEL-SPEC.txt, ptx-kernel-configs.md
src/tools/  pe_inspect, pe_resources, hnet_weights, model_spec, extract_modules,
            authenticode_verify, ptx_trace
src/ref/    hnet_model (loader), hnet_ops, hnet_ref, forward, run_frame, image_io
src/gpu/    gemm_coopmat.comp, gemm_coopmat_f16acc.comp, libxmx.c, xmx.py, tests
src/probe/  coopmat_probe.c
```

Regression — all should exit 0:

```
python3 src/tools/model_spec.py work/weights_ht.bin   # 73,841,889 ALL ACCOUNTED
python3 src/ref/hnet_model.py                         # every parameter placed
python3 src/ref/hnet_ops.py
python3 src/gpu/test_gemm.py                          # XMX vs numpy
python3 src/gpu/test_attention_gpu.py                 # three block families
python3 src/ref/run_frame.py --skip-rms               # the acceptance test
python3 src/ref/run_frame.py --skip-rms --no-attend  # ALWAYS compare against this
```

Rebuild if `work/` is lost:

```
python3 src/tools/pe_resources.py work/dl/nvngx_dlssnr.dll --dump RCDATA/WEIGHTS_HT work/weights_ht.bin
python3 src/tools/extract_modules.py work/dl/nvngx_dlssnr.dll work/modules
glslc -fshader-stage=compute --target-env=vulkan1.3 -O src/gpu/gemm_coopmat.comp -o work/gemm_coopmat.spv
cc -O2 -shared -fPIC -Iwork/vulkan-headers/include -o work/libxmx.so src/gpu/libxmx.c -lvulkan
```

---

## 8. Hardware facts (probed, trust them)

Intel Arc 140V-class Xe2, Mesa ANV, Vulkan 1.4.354, subgroup 32.
Six cooperative matrix configs, all scope=subgroup, all M=8 N=16:
fp16/bf16 each with same-type or fp32 accumulate, sint8/uint8 -> int32 (K=32).
`cooperativeMatrixRobustBufferAccess = false` — out-of-bounds loads are UB, so edge
tiles must be padded explicitly. No TF32 or INT4 through Vulkan. `VK_NV_cooperative_matrix2`
advertises zero flexible-dimension configs. XMX throughput measured at 0.7–1.9 TFLOP/s
with a naive kernel (no shared-memory staging, no K-blocking) — a floor, not a ceiling.

There is also an NPU (`/dev/accel/accel0`, `intel_vpu` loaded) — **not useful here**:
no Vulkan access, graph-compiler only, which conflicts with layer-by-layer validation.

---

## 9. Related work

`skchen17/dlssnr-amd-lab` — serious RE, reached bitwise parity for slots 1-98 by
capturing exact RTX state and replaying translated PTX under ZLUDA. Both halves are
closed to us (no RTX, no PTX consumer on Intel). Their own evidence log (E-24) states
the **semantics were never recovered by anyone**. Useful for cross-checking facts, not
for answers.

`danielblnc/DLSS-NR-on-AMD` — 816 stars, no source code at all.

If the reimplementation route stalls, the alternative is what worked for AMD:
translate PTX and execute NVIDIA's own kernels. For Intel the chain PTX -> SPIR-V ->
Level Zero is shorter than AMD's.

**The census that made this look cheap was wrong** (2026-09-08). It was taken with the
`ptx_trace` that dropped every `{ ... }` scope, so it missed ~120k instructions. Redone
with the fixed parser over the same 120 non-FP8 kernels:

```
distinct full opcodes  170   (recorded as 46)
distinct opcode bases   48
```

And the hard set is not "only `mma` and `movmatrix`". It also contains the Blackwell
async data-movement pipeline: **`cp.async.bulk` (480) with `mbarrier` (798)** -- that is
TMA, which Intel has no equivalent for and which would have to be replaced by ordinary
loads in every kernel -- plus `elect` (456), `bar` (518), `shfl` (7,880), `tex` (130)
and `red` (192). `shfl`, `elect` and `tex` map onto subgroup operations and images;
TMA does not map at all.

So route B is a rewrite of every kernel's data movement, not a mechanical opcode
translation. It is still the proven route, but it is not the cheap one the old note
implied.
