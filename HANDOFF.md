# HANDOFF — read this first

State of the DLSS-NR on Intel Xe2 project as of **2026-09-10**. CLAUDE.md holds the
original brief; **this file overrides it wherever they disagree**, and after
2026-09-09 they disagree about something foundational.

---

## Latest: softmax packing and GEMM sweep (2026-09-10)

**A smaller, exact optimization is now enabled:** softmax uses native FP16 pack/unpack
instead of manual bit conversion. The isolated kernel is 14% faster, but paired full
frames improve only about 2% at 384x384 and 1% at 720p; **1080p is unchanged within
noise**. Do not quote the kernel gain as a frame gain. All heads remain bit-identical.
`notes/phase29-softmax-pack.md` contains measurements and reproduction commands.

Six GEMM tile sizes and transposed static weights were evaluated; neither produced
a worthwhile replacement for the current path. The ablation benchmark now forces
fresh recording so cached replay cannot ignore its disabled operations. Attention
layout/conversion fusion is a better next experiment than another blind tile sweep.
`make test` passes, including exhaustive affine-pair and finite-FP16 softmax checks.

## Previous: captured frame replay (2026-09-10)

**A frame now uses one GPU submission instead of 73, and subsequent frames reuse
recorded commands.** Encoder skip copies and bottleneck conversions run on the GPU.
The output is bit-identical to the previous resident path, including changed inputs.
`notes/phase28-frame-replay.md` records the implementation, limits and raw evidence.

Incremental warm measurements over phase27, paired on the same buffers:

| Output | Previous block mode | Captured replay | Time reduction |
|---|---:|---:|---:|
| 384x384 | 123.5 ms | 91.0 ms | 26% |
| 1280x720 | 541.3 ms | 510.6 ms | 6% |
| 1920x1080 | 1139.0 ms | 1102.7 ms | 3% |

These are graph timings, excluding feature assembly/composition and image I/O;
first-use allocation/compilation/capture is separate. This remains photo mode.
`NR_FRAME_MODE=replay` is the default; `block` restores the old submission strategy,
`single` records once per frame. Diagnostic stage capture/timing uses block mode.

The backend now retains one extent by default and frees old GPU resources on a
resolution change. Temporary image files are cleaned up. `.gitignore` no longer
hides `src/ref/`; include that source in the next commit. Clean-build dependencies
and pinned upstream revisions are in `notes/reproduce.md`.

GPU/CPU regressions, cold capture, replay with changed input, resolution eviction,
recording-error recovery, clean native/shader build, and a three-frame temporal pan
all passed. `src/bench/frame_replay.py` reproduces the exact-output comparison.

## Previous: pipeline specialization (2026-09-10)

**The resident Vulkan path is faster, with bit-identical output.** Operation flags
are now specialized before pipeline compilation; 33–34 cached variants cover a
frame. All specialized variants in the compiler check are spill-free, against the
generic 16x32 GEMM's 15:15 spills:fills. The claim that OpenCL was the only remaining
route to improvement is withdrawn. `notes/phase27-pipeline-specialization.md`.

Warm, paired measurements on a real Cyberpunk frame, with the same buffers:

| Output | Generic median | Specialized median | Time reduction |
|---|---:|---:|---:|
| 384x384 | 126 ms | 114 ms | 9% |
| 1280x720, two separate processes | 614–670 ms | 536–550 ms | 13–18% |
| 1920x1080 | 1457 ms | 1179 ms | 19% |

These measure the graph, not image I/O, feature assembly or composition. First use
also compiles the variants; startup is not represented by these warm numbers.
The old 2.9 s 1080p record was not reproduced with this protocol and must not be
used to inflate the improvement. This remains photo mode, not real-time rendering.

Enabled automatically after `make`; `--resident` and the game daemon both benefit.
`XMX_SPECIALIZE=0` restores the generic path for comparison; default is `7`.
`make test` passes, including the new strided/batched GEMM specialization tests.
`src/bench/specialization.py --size 720 1280 --masks 0,7 --pairs 6` reproduces the
paired comparison and checks exact outputs before and after changing frame input.

---

## IT RENDERS (2026-09-09, later)

A frame goes in and a neurally-rendered frame comes out, with the real effect:
eyelashes and eyebrow hairs resolved out of a smeared input, skin pores synthesised,
iris and eyeliner sharpened. `notes/phase7-first-render.md`.

```
make                                                         # Vulkan runtime, layer, shaders
python3 src/ref/nr_frame.py IN.png OUT.png --resident        # the fast path
python3 src/ref/nr_temporal.py IN.png OUT --resident --pan 6,0 --frames 5
work/venv/bin/python src/ref/nr_frame.py IN.png OUT.png --accel   # 10 s, best on CPU
python3 src/ref/nr_frame.py IN.png OUT.png                   # 38 s, netlib reference
```

Before phase27, a full **1280x720** frame rendered in **0.59-0.72 s** (network extent
1280x768), with a **2.9 s** 1080p record. Current paired results are above. Quote the
band, not a single number. Frames inside one process were steady to 2 %, but the same binary spread ten
percent either side of a ~625 ms median *between* processes — buffer placement, not
clocks; the GPU holds 1950 MHz at 43-45 C throughout. A first frame after an idle costs
two to four times the rest while the clock ramps.

- `src/ref/nr_model.py` — the recovered 71-block graph in numpy (a port of MLX-DLSS's
  PyTorch `model.py`, Apache-2.0; no torch on this machine). One GEMM entry point,
  `nr_model.MATMUL`, for the XMX swap.
- `src/ref/nr_frame.py` — frame in / frame out, using MLX-DLSS's pure-numpy
  `features.py` and `composition.py` loaded by path from `work/mlx-dlss`.
- Our 649 logical tensors match their `weight_spec.json` **exactly** — 0 missing,
  0 extra, 0 shape mismatches. Two independent extractions of the same DLL agree.
- Two controls passed. Shuffled weights give a flat red tint with no pores and no
  lashes (structure-blind, ratio 0.98 vs the trained 1.18). The recovered control
  profiles work: `neutral` switches the network off by **37x** (change 0.00071 vs
  0.02610), and `natural` / `cinematic` are distinct styles.
- **`src/ref/{hnet_model,hnet_ops,hnet_ref,forward,run_frame}.py` are superseded.**
  They decode the packed container as dense FP16 and guess the block layout. Keep them
  for the PTX-derived findings they encode; do not build on them.
- **Phase 4 is done**: `src/gpu/nr_xmx.py` puts every GEMM on the XMX units, the
  batched attention included. `notes/phase8-xmx-graph.md`.
- **`src/gpu/nr_xmx.py` (the GEMM-hook path) is superseded by residency** and kept
  only for comparison. It lost to a good CPU BLAS, because a dispatch round-trips its
  activation through host memory; that is what made residency the precondition rather
  than an optimisation. The CPU baselines are worth remembering: the system numpy is
  netlib reference at ~3 GFLOP/s, a pip numpy is OpenBLAS at 214, and on the 384 face
  netlib CPU is 38.0 s against OpenBLAS CPU 17.5 s. `notes/phase13-torch-and-blas.md`.
- **The numpy port is bit-identical to MLX-DLSS's PyTorch original** — every primitive,
  every layout operator, all four block families on real weights, and the whole
  71-block forward, once the same GEMM is given to both.
  `python3 src/ref/test_against_torch.py` under `work/venv`.
- **Per-element agreement is not a property a port can have.** Each precision regime
  annihilates perturbations below its own working precision *exactly* and jumps
  straight to 9-12 % of the head's sd above it — float32 flips between 1e-09 and
  1e-07, half between 1e-05 and 1e-03. numpy's own float32 GEMM carries 5.2e-07 of
  error, above the float32 threshold, so **any** correct float32 implementation would
  diverge from this reference by the whole floor. Judge on the composed image and on
  the controls, never per-element. `notes/phase9-numerics.md`.

---

- **The NPU is not worth using.** Present and driver-ready (`/dev/accel/accel0`,
  `intel_vpu`), but the GPU is the faster engine in this SoC (67 TOPS against 48), our
  own shader runs at 4 % of it, all three engines share one memory pool so the actual
  bottleneck does not move, and the graph's bit-level ops do not fit an NPU compiler's
  operator set. `notes/phase14-npu-and-rounding.md`.

---

## 0. The one thing to know

**The weight container does not hold plain dense FP16, and almost every dead end in
this project traces to assuming it does.**

`iamwavecut/MLX-DLSS` ships working extraction code for the same DLL. Run on ours it
turns the 153 packed tensors into **649 named, shaped, logical tensors** — none
unsupported, none opaque. The payload is packed into `mma` fragment order and partly
E4M3; our reader was interpreting those bytes as dense FP16.

Proof it matters: `block0.layer0.input_adapter_weight` is `(16, 32)`, exactly the
512-element region we had located at the front of block0. Its logical values have
**correlation −0.02** with our raw FP16 read of the same bytes, and sd **0.2489**
against our 0.0306 — the `1/sqrt(16) = 0.25` that `notes/phase5-stem.md` predicted a
16-input projection should have. Right location, right shape, wrong decode.

So: **start from `work/mlxw/dlssnr-logical.safetensors`, not from
`src/tools/hnet_weights.py`.** Everything we derived from the *kernels* survives;
everything we derived from container *bytes* has to be re-checked against the logical
tensors.

```
git clone --depth 1 https://github.com/iamwavecut/MLX-DLSS work/mlx-dlss
T=work/mlx-dlss/python/mlxdlss/tools
PYTHONPATH=work/shim python3 $T/extract_dlssnr_weights.py work/dl/nvngx_dlssnr.dll work/mlxw/dlssnr-packed.safetensors
PYTHONPATH=work/shim python3 $T/unpack_dlssnr_weights.py work/mlxw/dlssnr-packed.safetensors work/mlxw/dlssnr-logical.safetensors
```

`work/shim/safetensors/` is a 60-line format-compatible shim — this machine has no pip
and the real package needs sudo. Round-trip verified. Both files already exist.

---

## 1. Do this first

**Done 2026-09-09.** The graph runs, on CPU and on XMX. What is left, in order of
value:

1. ~~**The GEMM kernel's operand staging.**~~ **DONE and it does not pay** —
   `notes/phase22-staging-and-storage.md`. `src/gpu/gemm_staged.comp` exists, is
   correct, and wins 19-24 % on deep-K shapes; it loses 44 % on the shallow-K, huge-M
   shapes this graph actually spends its GEMM time on. Gated at K >= 128 the frame is
   unchanged. Storing every published buffer as float16 is done too, and also changed
   nothing measurable. **The frame is no longer bandwidth-bound** — 30.9 GB in ~600 ms
   is 50 GB/s against a ~90 GB/s ceiling — and both halves are latency-bound.
   **And the GEMM's ceiling is now understood** — `notes/phase26-the-register-ceiling.md`.
   All 64 XMX engines are busy (throughput scales linearly to 256 workgroups, four per
   engine, and flattens exactly there), but each is ~90 % idle because a SIMD32 subgroup
   runs out of registers: one accumulator is 4 of the 128 GRF, so a 16x32 block already
   spills and a 32x64 block spills 474 times **in the old universal shaders**.
   **Phase27 removes the 16x32 spills by specializing operation flags** and saves
   13–18% at 720p without changing any result. The old conclusion that only OpenCL
   could help was too strong. `cl_intel_subgroup_2d_block_io` and wider DPAS shapes
   remain a possible OpenCL research direction, requiring a second backend.

   **Phase28 also removes per-block submissions and repeated Python recording.**
   The old phase25 fit (`20 ms + 632 ms per megapixel`) predates both changes and
   cannot describe today's fixed overhead. Current paired timings are at the top.
   They still do not demonstrate real-time rendering. Profile the specialized,
   replayed graph before choosing another kernel change. The historical reasoning
   below is retained as evidence of earlier experiments, not a current cost split.

   *A 720p frame is ~600 ms, and*
   `src/bench/split_cost.py` splits it almost exactly in half: **327 ms of GEMM,
   366 ms of everything else.** The second half is close to its floor — the passes run
   at 50-90 GB/s against a ~90 GB/s ceiling — and the first is not: 459.6 GFLOP in
   327 ms is **1.4 TFLOP/s, 4.4 % of the ~32 TFLOP/s peak**. The kernel now keeps a
   16x32 block of the output in registers, but still has **no shared-memory staging of
   the operands**, which is the one lever left. Two things already tried and rejected
   with numbers, so do not repeat them: a 32x32 register block (faster in isolation,
   *slower* in a frame — occupancy) and software pipelining the K loop (619 -> 760 ms —
   register pressure). `notes/phase21-fusion-and-tiling.md`.
   *(No longer blocked: the coopmat epilogue works if the accumulator is stored
   untouched to shared memory first and transformed there — llama.cpp's `mul_mm.comp`
   does the same. The underlying **Mesa/ANV bug is still live in 26.2.1 and unreported**;
   there is a two-line reproducer and a five-variant table in `notes/phase18-fusion.md`,
   and it is the one piece of this work that clearly belongs upstream. A second one, if
   wanted: llama.cpp issue #13530 has coopmat disabled for all Intel on the strength of
   an Alchemist regression, and the only Xe2 rebuttal in it is a discrete B580 with
   GDDR6. Arc 140V on a UMA LPDDR5X pool is unmeasured in public and we have the
   numbers.)*
2. ~~**A real game, not `vkcube`.**~~ **The pipeline is proven; the content is not** —
   `notes/phase24-a-real-game.md`. The layer loads inside a Wine prefix, intercepts the
   swapchain DXVK creates for a translated D3D11 game, and the frame comes back into the
   game's own image: `[nr_layer] swapchain 1280x720 format 44` then `processed 1280x720`.
   `src/layer/nr-photo --proton <appid> <exe>` is the reproducible way in.

   **What is left is a frame worth looking at.** The one captured is a warning screen —
   2D interface over pixel art. Reaching a face needs someone to drive the game's menus,
   which is the one part of this that cannot be done unattended. **Dead or Alive 5**
   (appid 311730, D3D11, installed) is the right target; its character models are exactly
   what the model is trained on. Counter-Strike 2 is native Vulkan and would skip Proton
   entirely, but it is VAC-protected — not somewhere to experiment with someone's
   account.
3. ~~**The temporal path**~~ **DONE 2026-09-09** — `src/ref/nr_temporal.py`,
   `notes/phase12-temporal.md`. The history gate is the head's fourth channel, and it
   works: alpha goes 0.008 with no history -> **0.705** with correctly reprojected
   history (ceiling 0.7397) -> **0.032** when the motion is wrong. Flicker on a static
   scene falls 3.6x by frame 3 and 6.3x at the peak, with no high-frequency loss.
   Optical-flow motion and processing_scale != 1 still need OpenCV/Pillow, which this
   machine lacks; engine motion works.
4. ~~**The DX12/Proton integration**~~ — the door turned out to be a Vulkan layer
   rather than an NGX hook; see item 2.

**Already done:** published inter-block buffers use float16 (phase22); per-block
host copies/fences and repeated command recording are removed (phase28). Tuning
`MIN_MACS` concerns the superseded `nr_xmx.py` hook path, not the resident default.

---

## 2. What is solid — measured from the kernels, unaffected by the packing

| Fact | Where |
|---|---|
| Attention **is** softmax: logits hard-clamped (Swin ±6, ViT ±3), `exp` hand-rolled in f16x2 bit arithmetic, **no max subtraction**. Constants exact, verified bit-exact | `notes/phase5-softmax-found.md` |
| The gate multiplies the **skip**, fused into the `mma` C operand: `D = A·B + gate⊙x`; the branch is added unscaled | `notes/phase5-gate-on-skip.md` |
| `attn_scale` is **FP32 per head** (`ld.global.b32` + `cvt.rn.f16.f32`, stride 4, indexed `head = 4·ctaid.z + tid.y`) | `notes/phase5-attn-scale-fp32.md` |
| **head_dim = 32 at every width**, from the QK-norm reduction (4 lanes × 4 squares × 2). Head count = C/32 | `notes/phase5-narrow-blocks.md` |
| Kernel parameter block: `+0` input, `+8` output, `+16` weight arena, `+24/+32` dims. I/O kernels take descriptor tables | `notes/phase5-io-structs.md` |
| **Input is five optional 2D textures**, `tex.2d.v4.f32`; only colour is required | `notes/phase5-input-contract.md` |
| **16-channel packing order**: ch4-6 colour, ch7-9 reprojected history (same affine `(x−a)·b`), ch12-14 sign-encoded validity. MLX-DLSS agrees independently | `notes/phase5-channel-order.md` |
| Output head is **32 → 4**; three channels become display RGB | `notes/phase5-output-head.md` |
| XMX flushes subnormal FP16 to zero; fixed by a per-tensor 2^k rescale | `notes/phase4-subnormal-flush.md` |
| **Each precision regime has a sharp threshold**: below it a perturbation is annihilated exactly, above it the head jumps to 9-12 % of its sd. float32 ~1e-07, half ~1e-04. The half path is the **more stable** of the two | `notes/phase9-numerics.md` |
| The whole CPU/XMX gap is the FP16 rounding of GEMM *activations*, and **97.3 % of them are already half-valued** — only 186 of 6987 calls are touched. Weights change nothing: 579 of 649 tensors are stored F16, the other 70 are `attn_scale`, not a GEMM operand | `notes/phase9-numerics.md` |
| Batched attention and the folded branched FFN are **bit-identical** to the plain GEMM hook; the two 720p renders are pixel-identical | `notes/phase9-numerics.md` |
| ~~CPU and XMX agree to 9.7e-07 over 4.2M elements~~ — measured on a pass-through with no E4M3 publishes in it; does not transfer | `notes/phase4-end-to-end.md` |

---

## 3. Withdrawn — do not resurrect

Today alone: **eight**. This is the normal rate here; assume the next confident claim is
also wrong until it survives an adversarial test.

- **"The weights are plain FP16, used exactly as stored."** Section 0.
- **"GQA 4:1, Q=C², K=V=C²/4."** `qkv_weight` is `(C, 3C)` — **full MHA, no GQA**. The
  narrow-block Q/K/V split that resisted three methods does not exist.
- **"DLSS-NR does not use softmax."** The `ex2` census was right, the inference wrong;
  `ptx_trace.py` was dropping every `{ … }` scope, which is where the f16x2 exp lives.
- **"3C² means K and V are pre-replicated."** The factor 2 is uniform across all
  matrices — it is the packing, not GQA.
- **"The 128C region is an attention mask."** `max = 0.00` was a rounded print; the
  fraction of exact zeros is 0.0000.
- **"`attn_scale` is stored as a log."** Right symptom, wrong mechanism — it is FP32.
- **"The attention branch helps, 1.39x swing."** Reverses on a structurally different
  image and on two real Cyberpunk frames, where the branch **costs 1.37x**. The swing
  was the branch repairing damage the bilinear scaffolding does to `test_pattern`'s
  fixed-period lines. `notes/phase5-input-dependence.md`
- **"1.02x better than the input."** True only at sigma 0.06 on one synthetic pattern.
  Swept: the network adds a fixed distortion of 0.0163 and removes a constant 6.5 % of
  noise — constancy being the signature of a linear filter, not a denoiser.

Older, still withdrawn: the "missing 2^8.5 factor", the "~450x branch/skip factor" (it
followed from the wrong gate form), the leading-region projection, and "1.01x parity"
(a pass-through).

---

## 4. Traps this project keeps falling into

- **Tools that skip input silently.** `ptx_trace.py` dropped brace scopes and hid 120k
  instructions. Always check coverage: parsed statements vs `;` count, and whether the
  dataflow graph is connected.
- **PTX is not SSA, and it has loops.** "Last definition before the use" is only valid
  in straight-line code. The pre-block's store tail sits *textually before* the texture
  reads it consumes, because it is a loop body. A def/use map over all definitions
  invents edges; program order alone is not enough either.
- **The metric has had five holes.** Correlation rewards inaction; dividing by a fitted
  gain is 0/0 on collapse; a pass-through scores 1.00x; the score measured the
  scaffolding not the model for a whole phase; and results depended on the test image
  and the noise level. `run_frame.py` now reports COLLAPSED, PASS-THROUGH and
  NETWORK-INERT — **and `--no-attend` is the control to run whenever a score moves.**
- **Statistical segmentation cannot find tensor boundaries.** Three methods failed
  calibration at C=512 where the answer was known. The notes said so; I re-ran one of
  them anyway. Read section 3 before trying a fourth.
- **External write-ups are summaries, not sources.** A WebFetch of `weight_spec.json`
  returned plausible-looking shapes with a confabulated label (`block31` as "final
  output stage"). Clone the repo and read the file.

---

## 5. Layout, commands, repo

```
ref/        the DLL (0444) + sha256      NEVER modify, NEVER commit
work/       weights, PTX modules, mlx-dlss clone, mlxw/*.safetensors, shim, builds
notes/      24 findings documents
src/tools/  PE/resource readers, the weight reader (now superseded), model_spec,
            and the PTX analysis tools: ptx_trace (dataflow), ptx_addrform
            (address → linear form), ptx_chains (accumulator chains)
src/ref/    nr_model, nr_frame, nr_temporal, nr_display, nr_accel, image_io
src/gpu/    xmxres, nr_resident, nr_frame_resident   <- the device path
src/layer/  nr_layer.c, nr_daemon.py, nr-photo       <- into a game
            hnet_*, forward, run_frame            <- superseded, kept for their findings
src/gpu/    gemm_coopmat*.comp, libxmx.c, xmx.py, tests
```

Git repo initialised 2026-09-08, three commits, **code only** — `.gitignore` keeps the
DLL, the weights, the driver payload and the game screenshots out. `/ultrareview` is
ready to run (user-triggered; the model cannot launch it).

Regression, all should exit 0:

```
python3 src/ref/test_nr_model.py                  # primitives, codec, temporal, graph
work/venv/bin/python src/ref/test_against_torch.py # bit-identical to the original
python3 src/gpu/test_resident.py                  # the device path, ~2 min
python3 src/gpu/test_gemm.py                      # worst rel 2.4e-06
python3 src/ref/nr_frame.py IN.png OUT.png --gpu  # the visual check
python3 src/ref/nr_frame.py IN.png OUT.png --profile neutral   # the control: ~37x smaller
```

`nr_frame.py` flags: `--gpu --size HxW --profile {standard,neutral,natural,cinematic}
--intensity F --detail-strength F --colour-strength F --frame-index N -v`.

**Controls** (`notes/phase10-controls.md`). Free, post-network — one pass covers the
whole range: `--intensity` (exactly linear, 0 an exact no-op, clamped to [0,1]),
`--detail-strength` / `--colour-strength` (the change is 0.0049 high frequency against
0.0249 low, so `--colour-strength 0` is detail with no tonal shift; past
`--detail-strength 2` it over-sharpens), `--intensity-ladder` renders once and writes
one file per value. Costing a pass each: `--style-index` (a different character, corr
0.50 with style 0 at index 1 — but only 0-8 are sane, 64 gives a magenta cast),
`--local-tone` / `--local-structure` (smooth monotone gains, colour-neutral and safe to
over-drive: tone reaches 1.24x at 2.0, structure peaks near 1.5), `--skin-structure`,
`--auto-mask`, `--control-mask`. Model A/B/C is **not** reproducible — the shipped
weights prove only slot 0.

`nr_xmx.install(fuse_branched=True, exact=False)`. `exact=True` carries activations
half cannot hold as a sum of two halves — more accurate than the reference's own
float32 GEMM — for 16.8 s -> 21.2 s. It moves the head gap from 0.0174 to 0.0124 and
no further; see `notes/phase9-numerics.md` for why zero is not reachable.

The old harness (`src/tools/model_spec.py`, `src/ref/hnet_*.py`, `run_frame.py`) still
runs but is built on the wrong weight decode; its scores measure scaffolding.

---

## 6. Honest standing

**The prototype works.** A frame goes in, a neurally-rendered frame comes out, on
NVIDIA's own weights, with the effect the feature is sold on. Two adversarial controls
pass (`notes/phase7-first-render.md`). The previous entry here — "no demonstrated
denoising" — is retired; its cause was the weight decode, exactly as section 0 predicted.

What is *not* claimed:

- **No NVIDIA parity gate.** There is still no NVIDIA GPU here, so there are still no
  reference activations. The graph is MLX-DLSS's recovery from vendor captures, and it
  is validated against their spec and against behaviour, not against the DLL.
- **The whole graph is resident on the GPU**: phase27 warm medians are **0.114 s** at
  384x384, **0.536–0.550 s** at 720p and **1.179 s** at 1080p. The optimization is
  bit-identical to the generic GPU path. Earlier comparisons reported head correlation
  0.9918 with the CPU reference and visually indistinguishable pictures.
  5.6 GB of device buffers at 720p.
  `src/gpu/nr_frame_resident.py`, `notes/phase15-residency.md`,
  `notes/phase18-fusion.md`, `notes/phase21-fusion-and-tiling.md`.
- **It runs in a game.** A Vulkan layer captures the presented frame, a daemon runs
  the model, and the result goes back into the swapchain — a photo mode, triggered by
  a file. Verified end to end on `vkcube`.
  `src/layer/`, `notes/phase17-integration-landscape.md`, `notes/phase19-photo-mode.md`.
- **HDR is handled**: `src/ref/nr_display.py`, the recovered display codec — encode a
  linear-HDR frame to an sRGB proxy with a soft knee, run the model, fold it back by
  luminance ratio onto the untouched original. Clamping instead destroys 97 % of the
  highlight structure. `notes/phase16-hdr.md`.
- **The machine's real limits, measured** (`notes/phase20-machine-limits.md`):
  **70-91 GB/s** of memory bandwidth against 136.5 theoretical, the GPU holding its
  **1950 MHz ceiling** throughout a run at 46-48 C, and our GEMM at **8-12 %** of the
  ~32 TFLOP/s FP16 peak — **4.4 %** averaged over the frame's real shapes **before
  specialization**. The GPU has **64** XMX engines. Phase27 demonstrates that some
  register pressure was avoidable shader code; these older throughput measurements
  do not establish the optimized kernel's ceiling. Earlier notes quoted 23 GB/s,
  which was single-threaded numpy and wrong by 3x. Both GEMM and elementwise graph
  operations now run on the GPU.
- **Temporal processing exists** in `nr_temporal.py`; the game's photo daemon uses
  the single-frame path and does not supply engine motion/history.
- The graph recovery is **not ours**. Ours is the Xe2 execution path, the numpy
  reference, the independent second extraction that confirms their weight spec, and
  the PTX findings in section 2 that their write-up and ours agree on.

## 7. Hardware (probed, trust it)

Intel Arc 140V-class Xe2, Mesa ANV, Vulkan 1.4.354, subgroup 32. Six cooperative-matrix
configs, all scope=subgroup, all M=8 N=16; the path is **fp16 × fp16 → fp32** (config 1).
`cooperativeMatrixRobustBufferAccess = false`, so edge tiles need explicit padding.
Shared-memory APU: correctness and capacity are unaffected (141 MiB of weights against
an 11.46 GiB heap); it caps **bandwidth**, already the measured ceiling at 0.7–1.9
TFLOP/s with a naive kernel. `libxmx.c` still memcpys into a mapped HOST_VISIBLE buffer
on every dispatch — on an integrated GPU those copies are free to remove.
