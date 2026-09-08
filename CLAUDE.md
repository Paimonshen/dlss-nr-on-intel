# DLSS 5 Neural Rendering on Intel Xe2 (Lunar Lake)

## What this project is

Reimplement the inference pass of NVIDIA's DLSS 5 Neural Rendering (DLSS-NR) so it
runs on an Intel Xe2 integrated GPU.

As of 2026-09-07 no public port exists for **any** Intel GPU. The only non-NVIDIA
implementation is `danielblnc/DLSS-NR-on-AMD` for RDNA4, which works because RDNA4
has native FP8 E4M3 matrix hardware. That approach does not transfer here — see
"No FP8" below.

**Success = one frame goes in, a neurally-rendered frame comes out, and it matches
a CPU reference.** Playable framerates are explicitly *not* a goal. Do not propose
optimisations that trade correctness for speed until Phase 3 is done.

> **REACHED 2026-09-09.** `python3 src/ref/nr_frame.py IN.png OUT.png` renders a frame
> with the real effect — lashes and hair resolved, skin pores synthesised — in 45 s on
> CPU. Two adversarial controls pass. **Read `HANDOFF.md` first**; it overrides this
> file, and large parts of what follows are superseded. `notes/phase7-first-render.md`.

---

## Hardware (already probed — do not re-probe, trust these values)

- Acer Aspire A14-52M, Intel Lunar Lake, Arc 140V-class iGPU (Xe2 / Battlemage-family)
- Arch Linux, Mesa ANV Vulkan driver, `intel-compute-runtime` for OpenCL
- **15 GiB total RAM, shared with the iGPU.** 7.6 GiB swap. There is no dedicated VRAM.
- OpenCL stack alive: `Platform #0: Intel(R) OpenCL Graphics / Device #0: Intel(R) Arc(TM) Graphics`

Vulkan capabilities, confirmed via `vulkaninfo`:

```
VK_KHR_cooperative_matrix                : extension revision 2
cooperativeMatrix                        = true
cooperativeMatrixRobustBufferAccess      = false
cooperativeMatrixSupportedStages         = SHADER_STAGE_COMPUTE_BIT   (compute only)
shaderBFloat16CooperativeMatrix          = true                        <-- primary path
VK_NV_cooperative_matrix2                : extension revision 1        (Mesa impl)
```

XMX matrix engines support: TF32, FP16, BF16, INT8, INT4, INT2.
**They do not support FP8.** Xe3 / Panther Lake does not add it either.

**Verified 2026-09-07** (`src/probe/coopmat_probe.c`, full table in
`notes/hw-coopmat.md`): Mesa exposes exactly **6** cooperative matrix configs, all
scope = subgroup, all **M=8 N=16** (K=16 float, K=32 int8) — fp16 and bf16, each
with either same-type or **fp32** accumulate, plus sint8/uint8 → int32.
`subgroupSize` = 32. TF32 and INT4/INT2 are *not* reachable from Vulkan even though
the DPAS hardware has TF32 (OpenCL advertises
`cl_intel_subgroup_matrix_multiply_accumulate_tf32`). `VK_NV_cooperative_matrix2` is
advertised but reports **zero** flexible-dimension configs.
The **primary path is config 1, `fp16 × fp16 → fp32`** — the shipped weights are FP16
(see below), so nothing is converted. The bf16 config is a fallback, not the plan.

---

## SUPERSEDED 2026-09-09 — see HANDOFF.md section 0

**The section below is wrong.** The container holds *packed backend payloads* —
permuted into `mma` fragment order and partly E4M3 — not plain dense FP16. Decoding
them as FP16 gives values that correlate **−0.02** with the true logical tensors.
`data_len == 2*n_elem` fixes the byte count, not the encoding. Use
`work/mlxw/dlssnr-logical.safetensors` (649 named tensors). Kept below for the record.

## The weights are FP16 — there is no FP8 anywhere

**Corrected 2026-09-07. This section previously said the model ships FP8 E4M3
weights, and everything about dequantisation followed from that. It was wrong.**
Full evidence in `notes/phase3-weight-format.md`; reader in
`src/tools/hnet_weights.py`.

The weight container states its own element count. `data_len == 2 * n_elem` holds in
**153 of 153** tensors, so one byte per element is arithmetically impossible.
Decoding confirms it: FP16 gives sane weights (sd ≈ 2e-03), BF16 gives 1e-18
nonsense under either byte order. Little-endian, settled by an exact NaN/Inf census
over all 73.8 M elements (11 non-finite vs 61 092 byte-swapped).

**73 841 889 parameters, stored little-endian FP16.** The reported "~148 M
parameters" was the byte count divided by one, on the assumption of FP8.

What follows:

- **No dequantisation step exists.** No E4M3 decoder, no calibration, no error budget.
- **Do not convert to BF16.** That was only a workaround for FP8 hardware we lack.
  FP16 → BF16 would throw away 3 of 10 mantissa bits for nothing.
- **The primary path is FP16, not BF16.** Config 1 of the cooperative matrix table is
  `fp16 × fp16 → fp32`, the same shape and the same FP32 accumulation as the bf16
  config. Stored weights reach the XMX units with **zero conversion**.
- The debugging invariant is stronger than before: the weights are used exactly as
  stored, so **any divergence from the CPU reference is an implementation bug.**

Accumulate in FP32 — config 1 provides it. The Turing FP8-unpacking penalty was never
relevant here and is now doubly moot.

---

## The target binary

`nvngx_dlssnr.dll`, ~158 MB.

- ~~Officially shipped since NVIDIA driver 616.56 / 616.64 WHQL. Prefer the official
  driver copy over the leaked NBA 2K27 early-access build 310.8.0.0.~~
  **Both halves falsified 2026-09-07** — see `notes/phase0-acquire.md`. Driver 616.64
  was unpacked in full and does not contain it; the public `NVIDIA/DLSS` SDK is still
  at v310.7.0 (2026-06-23, pre-DLSS-5) and ships only `nvngx_dlss.dll`,
  `nvngx_dlssd.dll`, `nvngx_dlssg.dll`. There is **no official copy to prefer**: the
  NBA 2K27 early-access 310.8.0.0 build is currently the only one in existence.
  Acquisition stays with the owner, per the redistribution rule below.
- **Verified:** 89.1 % of the file is weights (`.rsrc`, 147 697 152 B), and they are
  **73 841 889 parameters in FP16**, not ~148 M in FP8 — that figure was the byte
  count read as one byte per parameter. No re-encoding is needed, so the on-disk
  147 MB is also the in-memory footprint. Trivial for this machine.
- Architecture, per NVIDIA's own research page (research.nvidia.com/labs/adlr/DLSS5/):
  a **one-step pixel-space diffusion model**, conditioned on the current rendered
  frame, engine motion vectors, carried temporal state, and artistic-direction values.
  Deterministic and temporally stable by design.
- Single-frame capture analysis on an RTX 2070 reported 174 CUDA kernels and 176
  compiled CUBIN modules, containing **Swin and ViT blocks, QKV projections**, and
  FP8 E4M3-specific kernels. Original binaries target `sm_120` (Blackwell) only.

Treat all of the above as *reported*, not verified. Verifying it is Phase 1's job.

### Useful reference implementations (read, do not depend on)

- `Dagherbou/OptiScaler_DLSSNR` — NVIDIA-only, but the cleanest example of how the
  NR pass is driven directly with no vendor integration. v0.2.0 as of 2026-09-06.
- `lisitskyaa/ComfyUI-DLSS5-NR` — runs the pass headless on a single image through a
  small native D3D12 bridge. Closest thing to the I/O contract we need to reproduce.
- `NIGos/dlss5-dx11-bridge` — documents the NGX call contract
  (`NVSDK_NGX_D3D12_CreateFeature` / `EvaluateFeature`).

---

## Hard constraints

1. **Memory.** 15 GiB shared, total, for the whole system. The NVIDIA-side OptiScaler
   mod reserves a fixed 13 GiB. We cannot. Design for a low internal model resolution
   and **tile the frame** from the start; do not write a whole-frame path and plan to
   retrofit tiling.
2. **No NVIDIA GPU exists on this machine or anywhere accessible.** There is no way to
   produce reference activations from the original binary. Ground truth must come from
   our own CPU reference implementation (see Phase 3). *Partially mitigated
   2026-09-07:* our weight decode is byte-identical to a model arena captured from a
   running RTX 50 by `skchen17/dlssnr-amd-lab` (SHA-256 `A5513B18…BD4EE3E5`), so the
   **weights** are anchored to real hardware even though activations are not.
   See `notes/phase3-execution-order.md`.
3. Cooperative matrix is **compute-stage only**. Everything goes through compute
   shaders; no graphics-pipeline path.

---

## Roadmap

- **Phase 0 — Acquire. DONE 2026-09-07.** `ref/nvngx_dlssnr.dll`, 165 840 496 B,
  sha256 `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`,
  version 310.8.0.0, mode 0444. **Not** from the driver — 616.64 was unpacked in full
  and does not contain it (it ships only the NGX runtime, which knows the feature and
  fetches the DLL out of band). Provenance is cryptographic: full Authenticode
  verification against `CN=NVIDIA Corporation` chaining to DigiCert Trusted Root G4.
  See `notes/phase0-acquire.md`; tooling in `src/tools/`.
- **Phase 1 — Confirm the weight blob. DONE 2026-09-07** — `notes/phase1-binary-map.md`.
  radare2 was never needed; `src/tools/pe_inspect.py` does the section map. Weights are
  in **`.rsrc`, 147 697 152 B = 89.1 % of the file, entropy 5.89**, confirming the
  reported ~89 %. Imports are only ADVAPI32/KERNEL32/USER32/VERSION — no cudart, cudnn,
  cublas, nvinfer or onnxruntime, so the graph is entirely inside this DLL.
  **There is no `.nv_fatbin` section** — that string does not occur in the file. The
  CUDA code is 15 fatbin containers (magic `BA55ED50`, first at 0xdf0e0) and 15 ELF
  cubins (first at 0x1f9220), all embedded in `.data`. `-arch sm_120` confirmed.
- **Phase 2 — Recover the graph. DONE 2026-09-07.** The written spec exists:
  `notes/MODEL-SPEC.txt`, regenerated by `src/tools/model_spec.py`, accounts for
  **all 73 841 889 parameters with zero remainder**. Grouped-query attention
  (QKV = 1.5C^2 = Q(C^2) + K(C^2/4) + V(C^2/4), 4:1), attention bias = heads x 64 x 64
  = 128C, transition blocks = standard + one extra C^2. Original notes: `notes/phase2-graph-rtti.md`.
  The CUDA toolkit turned out to be unnecessary: the DLL is MSVC-built with **RTTI
  intact**, and 35 mangled class names in the `HNetCpp` namespace give the layer
  taxonomy outright — Swin blocks at 1/2/4/8/16 heads (`CCSplitSwin16H*`,
  `CCTinlayoutFusedSwin{1,2,4,8}HLayer`), ViT blocks in 2D and 1D flavours
  (`CCVit*Layer`, `CCVit1D*Layer`) with QKV / attention / projection / FFN
  expand+contract, and `CCDecInputUpsampleLayer`. Every one is a GEMM-family op, so all
  of it maps onto our single 8×16×16 bf16→fp32 shape. **Still missing:** layer ordering,
  widths, depths, Swin window size/shift, and where the diffusion conditioning enters.
  Recover from `CCNetwork` construction code in `.text` and from tensor shapes in the
  `.rsrc` blob; carve the 15 fatbins out of `.data` by magic if kernel-level detail is
  needed.
- **Phase 3 — CPU reference. DONE 2026-09-09** — `src/ref/nr_model.py`,
  `src/ref/nr_frame.py`, `notes/phase7-first-render.md`. Not by finishing the recovery
  below, but by porting the graph `iamwavecut/MLX-DLSS` recovered from vendor captures
  (PyTorch -> numpy; Apache-2.0) onto the correctly-decoded logical weights. Our 649
  tensors match their `weight_spec.json` exactly — 0 missing, 0 extra, 0 shape
  mismatches — so the two independent extractions of this DLL agree. Regression:
  `python3 src/ref/test_nr_model.py`. *(Historical, superseded: The `.data` containers are **Zstandard**,
  not a proprietary codec — they decompress to PTX source for all 231 kernels, which
  carries the full layer configuration in mangled template parameters. See
  `notes/phase3-ptx-unlock.md`; this supersedes guesswork about shapes.)* The weight container is fully
  decoded — `notes/phase3-weight-format.md`, `src/tools/hnet_weights.py`: 153 tensors,
  73 841 889 FP16 parameters, walked to EOF exactly. **The dequantisation step does
  not exist** — weights are used as stored. Block sizes describe a symmetric U-Net of
  71 blocks with an 8-block bottleneck. Remaining: recover 2-D tensor shapes and the
  block→layer-class mapping from `CCNetwork` construction code in `.text`, then
  rebuild the graph and run one frame. Minutes per frame is acceptable. **This is the
  ground truth for everything after.** First "it's alive" milestone.
  **First numerics are running** — `src/ref/hnet_ref.py`, `notes/phase3-first-numerics.md`:
  41 of 153 tensors bound to kernel shapes, the 512x512 reshape verified against a
  shuffle control (row-norm cv 1.12 vs 0.17), and the accumulation choice measured on
  real weights — fp32 accumulate is **284x** more accurate than fp16 on one layer.
  Architecture is now recovered too — `notes/phase3-architecture.md`: 5 encoder /
  5 decoder Swin stages at **32/64/128/256/512 channels, 32 per head**, a ViT-1D
  bottleneck (blocks 31-38), `dec_input_upsample 1024->512`, shifted-window MSA
  confirmed by `_shifted` kernels, and the full parameter-name inventory. Each
  `block{N}.layer{M}.layer` is a **packed blob of several parameters concatenated**,
  which is why the sizes never factored into `in x out`. The model was exported from
  PyTorch (ATen op names survive in `.rdata`). Codenames: feature CG2R, engine HNet,
  configs `crazy-cuckoo` and `hnet-vigilant-squid`.
- **Phase 4 — GPU path.** *(Started, and already productive.)* Port to XMX via Vulkan
  cooperative matrix in **FP16** (not BF16 — the weights ship as FP16), compute
  shaders, tiled. Validate layer-by-layer against Phase 3.
  `src/gpu/gemm_coopmat.comp` + `src/gpu/gemm_runner.c` run FP16 x FP16 -> FP32 GEMM
  on the XMX units and match numpy to 2.4e-06 on synthetic data.
  **Critical finding — `notes/phase4-subnormal-flush.md`: XMX flushes subnormal FP16
  operands to zero, and 27.22 % of this model's parameters (20.1 M of 73.8 M) are FP16
  subnormals.** Median matrix is 9.64 % subnormal, worst is 82.92 %, and 20 matrices
  are over half. Run naively a quarter of the network evaluates to zero, silently.
  Fixed exactly by a per-tensor `2^k` rescale (`fp16_shift()` in `hnet_model.py`);
  residual error returns to the 5e-06 of ordinary FP32 accumulation.
  **All three block families now validated on hardware** — 20 layer evaluations,
  worst deviation 2.8e-04 (`src/gpu/test_attention_gpu.py`). The fused Swin layout is
  now anchored on the cosine gate, which is exactly `C` long after 8 zero pad bytes at
  every width; that fixed a real extraction bug where the gate sat inside `wq`.
  Open: the Q/K/V split needs a fractional KV head at C=32 and C=64 under GQA 4:1, so
  those 15 narrow blocks — built from different kernel classes — use some other
  structure, and the loader deliberately does not guess it.
  **NVIDIA accumulates in FP16, not FP32** — zero of 218 PTX kernels use an FP32
  accumulator (`notes/phase4-accumulation-choice.md`). Config 1 is 400–800x *more*
  accurate than the original rather than a reproduction of it; an FP16-accumulate
  shader is built and tested alongside, since it is the only way to compare against
  hardware-captured activations. Which is wanted is the owner's call.
  **End-to-end pass working** — `notes/phase4-end-to-end.md`, `src/ref/forward.py`:
  a tensor goes through all 71 blocks in execution order at 512x256 down to a 16x8
  bottleneck and back, on CPU and on XMX, agreeing to **9.7e-07 relative** over 4.2 M
  output elements. Plumbing only: 27 blocks pass through, the fused blocks' leading
  region is unapplied, and the runtime scale is still missing. **Resident context built**
  (`notes/phase4-resident-context.md`): `src/gpu/libxmx.c` keeps the Vulkan device,
  pipeline and buffers alive across calls, cutting the end-to-end pass from 13.1 s to
  **6.3 s** (CPU numpy: 11.5 s) with correctness unchanged. First honest throughput:
  **0.7–1.9 TFLOP/s** FP16 with FP32 accumulate. That is a floor — the kernel has no
  shared-memory staging, no K-blocking and no register reuse, so it is memory-bound
  rather than matrix-unit-bound. Optimisation stays deferred until Phase 3 correctness
  is settled, per the roadmap. The 6.3 s is not a frame time either: most of it is the
  CPU-side softmax and window shuffling in numpy.
- **Phase 5 (later) — Integration.** Wire into a real game. Target: Control (DX12, has
  DLSS, light enough for this iGPU under Proton, and the most published RTX 50
  before/after comparisons to sanity-check against). Cyberpunk 2077 photo mode is the
  better A/B still-frame source — the model is trained on skin, hair and subsurface
  scattering, so faces show the effect most clearly.

---

## Resolved 2026-09-07 — do not re-litigate

1. **Cooperative matrix table.** Answered by our own probe, not by `vulkaninfo`.
   6 configs, single shape 8×16×16, subgroup scope. See `notes/hw-coopmat.md`.
2. **OpenCL DPAS shapes.** `cl_intel_subgroup_matrix_multiply_accumulate` *and*
   `..._tf32` are both present, confirming DPAS hardware beyond what Vulkan exposes.
   Using it would mean a second, OpenCL backend — not a flag.
3. **FP32 accumulation for BF16 — yes.** `bf16 × bf16 → fp32` (C and Result both
   fp32) is config 3. This is the path: BF16 in, FP32 accumulate.

## SUPERSEDED 2026-09-09 — the section below measured the scaffolding

`run_frame.py` and `denoise_score` were built on the wrong weight decode, and
DLSS-NR is a detail re-render, not a denoiser, so a denoise score was never the right
objective either. The live acceptance test is `src/ref/nr_frame.py` plus the two
controls in `notes/phase7-first-render.md`. Kept below for the record.

## Goal, as redirected by the owner 2026-09-08

A **working** version that can be tested and eventually run in a game, ahead of the
"reimplement, do not execute NVIDIA's kernels" constraint written below. The AMD route
(capture exact RTX state, replay translated PTX) is closed to us on both halves, and
their own evidence log says the *semantics* were never recovered by anyone. So the
acceptance test is `src/ref/run_frame.py`: a frame in, a frame out, scored by
`denoise_score` against a clean reference — an objective that attenuation cannot game.

Current standing: **no result yet.** The score reads 1.43x worse than doing nothing,
but `--no-attend` — the network switched off entirely — scores the same to four
decimals. Every figure this project has recorded (5.13x, 1.45x, 1.43x) is the cost of
the stand-in bilinear resampling and skip blend; the recovered network moves the frame
by 0.35 % and the score by 3e-5. Amplifying the branch only makes it worse
(1.43 -> 1.61 -> 29.6x at gains 1/16/64), so its *direction* is wrong, not its scale.
`run_frame.py` now prints `NETWORK-INERT` and a scaffolding-only baseline; run
`--no-attend` first whenever a score moves. Target: meaningfully below 1.0x, which
needs the stand-ins replaced (stem, output head, learned resampling, skip blend
weights, the unapplied leading region). See `notes/phase5-visual-loop.md`.

## Open questions to resolve next

0. **The `128C` region is not the attention bias.** It is neither a shifted-window
   mask nor a 2-D relative-position bias, and only ~8 bits of each 16-bit element
   carry information (sign and exponent are constant, 321 distinct values in 65,536).
   Its role is open; it stays wired into the softmax as a dimensionally-correct
   placeholder. See `notes/phase3-bias-region.md`. **Narrowed 2026-09-08:** that note
   argues the near-constant -56.2 "has no effect, softmax is shift-invariant". The
   softmax has **no max subtraction** (`notes/phase5-softmax-found.md`), so a constant
   of -56 pushes every logit past the -6 clamp and makes the attention exactly uniform
   — measured. It cannot be a pre-softmax additive term.

0b. *(Withdrawn 2026-09-08.)* The "missing ~2^8.5 scale factor" is no longer an open
   problem. It was inferred from a norm-preserving `cos*skip + sin*branch` gate form
   that the PTX rules out — no kernel computes a sqrt or touches the immediate 1.0f.
   With the gate as a plain multiply the trunk does not decay and a small branch is
   ordinary residual behaviour. See `notes/phase3-first-operator.md`.

0c. **The attention non-linearity is solved.** It *is* a softmax; `ex2` never appears
   because `exp` is hand-rolled in `f16x2` from fma/max/min plus a shift-and-add on
   the f16 bit pattern. Logits are hard-clamped — **Swin +-6, ViT +-3** — instead of
   max-subtracted. Constants recovered and verified bit-exact;
   `ptx_exp()` / `ptx_softmax()` in `src/ref/hnet_ops.py`. This also withdraws
   `notes/phase5-no-softmax.md`. See `notes/phase5-softmax-found.md`.

0d. **`attn_scale` is FP32 — SOLVED.** The `2H` run is H **FP32** scalars, not 2H FP16
   ones, and `unknown_even` is the low half rather than an unidentified parameter: in
   the ViT blocks those slots take exactly 8 distinct values, all multiples of 0x2000
   (an FP16 widened to FP32); in the Swin blocks they hold NaNs and 1e4 magnitudes. The
   kernel confirms it — `ld.global.b32` + `cvt.rn.f16.f32`, stride 4, indexed per head.
   Scales become 0.015-28 instead of 1.1-2.8, and this is the **only** reading under
   which a logit ever reaches the hardware's +-6 clamp. The parameter block is also
   decoded: `+0` input, `+8` output, `+16` weight/scalar arena, `+24`/`+32` dims — so
   the "launch parameter block" is recoverable after all. This supersedes the
   stored-as-a-log hypothesis. See `notes/phase5-attn-scale-fp32.md`.

1. **Write the operators.** *Started* — `src/ref/hnet_ops.py`,
   `notes/phase3-first-operator.md`: the split-Swin-16H **attention path runs on real
   weights**. GQA splits the qkv slice exactly (C² + C²/4 + C²/4), the bias reshapes
   to (16, 64, 64), attention rows sum to 1 to 4e-16 and per-head entropy is 3.4–3.9
   of a possible 4.159 nats — a trained distribution, not a degenerate one.
   Cosine gates (values in [-1,1], max exactly 1.0) are **universal** — every block
   type has them, and those matrices carry no bias. QK-normalisation is confirmed
   (32 `rsqrt` for 16 heads; the branch is input-scale-invariant to 1.2%), and
   `attn_scale` is located: the odd slots of the `2H` run, one per head.
   **All block types now have an ordered internal layout** —
   `notes/phase3-block-internals.md`. Remaining: the ~450x branch/skip factor, which
   lives in the launch parameter block and is *not* recoverable from PTX alone.
2. **Which named parameter is which inside the first `4C^2 + 64C`.** The block's
   *ordered* layout is now solved — `notes/phase3-block-layout.md`, exact for 45 of
   46 fused blocks — and the gates and attention bias are identified by their value
   signatures. What remains is naming the individual matrices inside the leading
   weight region (`qkv_weight` is 1.5C^2 and `projection_weight` is C^2 by size).
3. **The 57 kernels with no `tin3_1` template** — post-block, blend, control-mask,
   RGB output head, `cc_cb_clear`. These carry the output path; shapes still to be
   read from the PTX body.
4. `VkPhysicalDeviceCooperativeMatrix2FeaturesNV` — flexible dimensions are empty,
   but the extension also carries reduction/conversion ops. Check before locking
   down the Phase 4 design.
5. **`cooperativeMatrixRobustBufferAccess = false`** means out-of-bounds coopmat
   loads are UB, not zero-fill. Since tiling is mandatory, decide the edge-tile
   padding scheme *before* the first shader, not after.

*(Answered and closed: cooperative matrix table and FP32 accumulation
(`notes/hw-coopmat.md`); FP8 vs FP16 (`notes/phase3-weight-format.md`); `.rsrc`
entropy; block-to-stage map, Swin window 8x8, head dim 32, and the full parameter
accounting (`notes/MODEL-SPEC.txt`).)*

## Repo layout

```
ref/     immutable originals — the DLL, recorded hashes. NEVER modified, NEVER committed.
work/    working copies, carved sections, dumps, scratch
notes/   findings, section maps, kernel name lists, the model graph spec
src/     our code
```

---

## Rules

- **Never commit or redistribute `nvngx_dlssnr.dll` or any weights derived from it.**
  They are NVIDIA's. Every existing project in this space requires the user to supply
  the DLL themselves, and this one does the same. The repo is code only.
- Never modify anything in `ref/`. Copy into `work/` first.
- Distinguish *reported* from *verified* in `notes/`. Most of what is written above came
  from press coverage and community frame captures, not from our own analysis.
- When something contradicts this file, the machine is right and this file is wrong —
  update it.

---

*Last updated 2026-09-09 (the network renders a frame; Phase 3 closed). Owner runs Arch Linux, is comfortable at kernel/driver level,
prefers C for low-level work, and does not need concepts explained from scratch.*
