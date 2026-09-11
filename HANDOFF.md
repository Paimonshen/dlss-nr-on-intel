# HANDOFF — read this first

State of the DLSS-NR on Intel Xe2 project as of **2026-09-11**. CLAUDE.md holds the
original brief; **this file overrides it wherever they disagree**, and after
2026-09-09 they disagree about something foundational.

`notes/INDEX.md` says what each of the fifty-four phase notes settles — go there when
you need the evidence behind a line in this file, rather than reading them in order.

---

## Latest: it stops flickering (2026-09-11)

The picture in live mode shimmered, and the cause was not noise in the input. Measured on
22 consecutive presents of a real game: **2.3 % of a frame is byte-identical between two
presents, and the network moved those pixels 3.25 levels of 255 anyway**, while pixels
that actually moved came back amplified 1.01x. The graph is global — five downsamples into
a ViT-1D bottleneck whose attention sees the whole frame — so a fighter moving in the
middle moves the decoder's answer over a crowd that did not move at all. Vendor stability
comes from the temporal path, not from the network (`notes/phase53`).

**The temporal path now runs in live mode, and the flicker is 3.7x smaller for 3.7 % of
the frame time** (215 -> 224 ms; static-pixel invention 3.25 -> 0.87 levels; moving pixels
untouched at 0.98x). `notes/phase54`. Three things a next reader needs:

**1. Identity reprojection is free and exact.** A `vkQueuePresentKHR` layer has no motion
vectors, so the previous output goes into feature channels 7-9 where it sits.
`sample_history` at pixel centres is **bit-identical** to the history — the five-tap
Catmull-Rom collapses to its middle tap — so there is no gather and no blur, and
`nr_frame.apply_history` matches MLX-DLSS's `make_temporal_features` bit for bit.

**2. The learned gate is not local, and this is the surprise.** `phase12` measured it at
**0.705** with correct history. In a fight it reads **0.12** over pixels that did not move
and 0.014 over pixels that did. It discriminates 10x, but the whole scale is down 5x,
because most of the frame moved and a global network distrusts the history everywhere.
**Optical flow does not fix it** — measured, not assumed: DIS costs 7 ms, raises the
whole-frame gate 0.076 -> 0.111, and nearly all of that lands on *moving* pixels, which is
the ghosting case rather than the flicker one.

**3. So there is a floor under the gate, and it is ours, not the vendor's.**
`history_confidence` can only scale down. What a present-time layer has instead is the
game's own frame: where the game handed back the same pixel, the previous output is right
for that pixel by construction. `nr_daemon.hold_floor` turns that into a per-pixel lower
bound, full at zero change and gone by four levels of 255, never above `blend_scale`. It
cannot ghost — the frame that changes a pixel is the frame that releases it — and it costs
**2.2 ms**. Knobs: `temporal`, `hold`, `cut_limit`, all live through `nr-ctl`.

**The daemon is now stateful across frames.** Any test that sends a sequence and expects
each frame to stand alone has to pass `--temporal 0`; `test_ui_mask.py` does.

## It runs in a game, live, at 10 fps (2026-09-10 evening)

Ten phases in one session, `notes/phase39` through `phase48`. The three things a next
reader most needs:

**1. The frame is fully accounted for, and performance really is finished — inside the
graph.** `xmx_profile()` puts a GPU timestamp after every recorded pass, so there is now
a per-pass breakdown instead of two ablations (`src/bench/frame_profile.py`,
`notes/phase45`). GEMM is 216 ms of 488 at 720p; of the other 272, **every pass that only
moves data is at the memory ceiling** (residual 104 GB/s, merge heads 84, partition 69,
split heads 65, to-half 61, against the machine's 70-91). Only softmax and cosine publish
sit below it, at ~45 %, and they are arithmetic — that is what an arithmetic pass looks
like measured with a bandwidth ruler. GEMM is register-bound (`phase26`). There is nothing
left to win inside the graph.

**2. The bottleneck moved out of the graph.** Shrink the extent and the network stops
being the frame. What is left is a stack of independent full-frame passes over the
*output* resolution — feature assembly, composition, the head upscale, the detail blur —
each reading and writing the whole picture, and none of which shrinks with the render
scale. Two of them were pure waste and are fixed (`phase47`, `phase48`): a resample
written in the obvious 2-D form cost **193 ms of a 350 ms frame** until the axes were
separated, and the diffusion noise was rebuilt from four transcendentals a pixel every
frame despite depending only on the extent and a frame index nobody sets.
**This is where the next work is.**

**3. There is a live mode and a control tool.** `NR_LAYER_LIVE=N` in the layer,
`--render-scale` in the daemon, `src/layer/nr-ctl` to drive both without reloading the
model. Measured end to end: **512x288 at scale 0.35 is 94 ms, 10.6 fps**; 640x360 is 9.5,
854x480 is 5.4. Set the *game* to that size and the compositor does the stretch for free.
The network runs on the reduced frame but the **head** is scaled back and composed against
the full-resolution original, so the game's own pixels are never resampled.

**4. Read the parallel tree before assuming this one is ahead.** `~/ProjectsCodex` is
driven by a different model on the same problem. Twice now it has held something this
tree lacked: `.gitignore` hiding the entire CPU reference from thirteen commits, and —
this session — seven Vulkan-layer guards plus a real bug, the interface mask silently
failing whenever a strength knob moved (`phase49`). Its git history being behind ours
says nothing about its working tree.

Also this session: the interface mask proven in a live fight (**ten times less HUD
damage**) and then caught making a *worse* artefact on a near-static frame, diagnosed and
fixed (`phase42`, `phase43`); the profiles measured as a real trade-off — everything added
to skin texture comes out of speculars and colour, and `--colour-strength` runs *opposite*
to its name (`phase44`); the layer proven under **VKD3D-Proton** on a 64-bit D3D12 game
(`phase41`); and four busy E-cores measured to cost the GPU **7 %** for a theoretical
gain of under 2 % (`phase46`).

## Earlier entries, now folded into the notes

softmax native packing (`phase29`), captured frame replay (`phase28`), pipeline
specialization (`phase27`). All three are in effect and none is contested; the numbers
live in their notes.

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

The graph runs, on CPU and on XMX, and the kernel work is finished. **720p is 495 ms
and dead stable — 494/494/494 across three processes.** What is left, in order of
value:

1. ~~**A frame worth looking at, from a real game.**~~ **Done** —
   `notes/phase34-doa5.md`. Dead or Alive 5 (appid 311730, 32-bit D3D9 through DXVK,
   not D3D11 as written here before) was driven live and the pass ran on real faces:
   lashes and skin resolved in the game's own swapchain image.
   `src/layer/nr-photo --proton <appid> <exe>` is the way in.
   What is *not* done is the interface mask on a live HUD. It was written and it was
   broken in the one place tests did not reach — `exchange()` used one size for the
   request and the answer, so every masked frame came back unchanged
   (`notes/phase39-layer-review.md`). Fixed and covered by a test that fails on the old
   code, but **never yet run in a game**: that needs a restart with `NR_UI_MASK=1`.

2. ~~**The Mesa/ANV cooperative-matrix store bug.**~~ **It does not exist** —
   `notes/phase38-there-was-no-bug.md`. The reproducer written to file it found that the
   last surviving variant was our own invalid shader: a float16 matrix stored into a
   `float[]`, where the component type does not match the destination. Given a matching
   destination every variant is exact. Three phases of design rested on it. What is left
   upstream is llama.cpp issue #13530
   has coopmat disabled for all Intel on the strength of an Alchemist regression, and
   its only Xe2 rebuttal is a discrete B580 with GDDR6; Arc 140V on a UMA LPDDR5X pool
   is unmeasured in public and this project has the numbers.

3. **A live mode exists and reaches 10 fps.** `notes/phase47-live-mode.md`:
   `NR_LAYER_LIVE=N` in the layer, `--render-scale` in the daemon, `src/layer/nr-ctl` to
   drive both without restarting the model. Measured end to end, **512x288 at scale 0.35
   is 98.6 ms, 10.15 fps**; 640x360 is 8.6 and 854x480 is 5.7. Set the *game* to that
   size and the compositor does the stretch for nothing. Note what this changed: below
   about 640x360 the network is no longer the frame — the numpy at output resolution
   (composition, feature assembly) is, and it does not shrink with the render scale.

4. ~~**If more speed is wanted, measure before choosing.**~~ **Measured, and there is
   nothing left inside the frame.** `notes/phase45-frame-profile.md`: every pass now has
   a GPU timestamp (`xmx_profile`, `src/bench/frame_profile.py`), not an ablation. The
   split is **216 ms of GEMM against 272 ms of everything else** — the old ablation said
   221/290, so it was right. What is new is the inside of that 272:

   | pass | ms | GB/s | of the machine's ~80 |
   | --- | --- | --- | --- |
   | softmax | 74.4 | 36 | 44 % — arithmetic |
   | cosine publish | 67.9 | 38 | 48 % — arithmetic |
   | residual | 51.3 | 104 | **130 %** |
   | split heads / partition / merge heads | 52.9 | 65-84 | **81-105 %** |
   | to half | 11.3 | 61 | **77 %** |

   **Every pass that only moves data is already at the memory ceiling.** The two below it
   do per-element arithmetic, so 45 % of bandwidth is what that looks like, not a
   deficiency — and softmax has already had one round of exactly this work for 1.1 % of
   the frame (phase 29). GEMM is register-bound (phase 26). The frame is fully accounted
   for. Also killed, with a probe kept at `src/bench/bank_probe.*`: the 32-way shared
   memory bank conflict in `attention.comp` is real and costs **1.11x**, not 32x.

   The extent curve is **17 ms + 488 ms per megapixel** — down from 20 + 632 before
   specialization and replay. Things already tried, with numbers, that should not be
   repeated: shared-memory operand staging (phase 22), integer weights (phase 23),
   register blocks past 16x32 and software pipelining the K loop (phases 21 and 26),
   and storing published buffers as float16 (phase 22 — correct, and no faster).
   ~~The open ones: attention layout/conversion fusion, and OpenCL.~~ Both are now
   closed. Attention Q/K fusion is **done** (phase 32: 1804 -> 1664 dispatches, 4 %).
   **OpenCL is measured and is not the lever** — phase 33: its DPAS path reaches
   3533 GFLOP/s against Vulkan's 3828 on the same shape, peaks at the same 16x32 block
   and collapses beyond it the same way. Two APIs, two subgroup widths, one curve.
   `cl_intel_subgroup_2d_block_io` is present and untried, and would have to buy more
   than 8 % just to reach parity.

**Memory is no longer the constraint it was**: the shared scratch arena took 720p from
5041 to 2303 MiB and 1080p now fits without swapping (phase 32).

**Real time is still not on the table.** At `17 ms + 488 ms/Mpixel`, 30 fps needs about
a 243x137 extent and 15 fps about 425x239. On this hardware with this graph, DLSS-NR is
a photo mode — which is what the Vulkan layer delivers.

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
- **A lookup table written from memory mislabels everything downstream.**
  `frame_profile.py` shipped a hand-written map of unary kind numbers to names. It was
  wrong, and under the wrong names one pass appeared to run at a fifth of memory speed,
  which produced a confident and entirely false diagnosis — with a proposed fix — before
  anyone noticed the table. The 50 ms belonged to the one pass that needed nothing.
  **Generate name tables from the source they name.** The file now regexes them out of
  `resident.comp` and `attention.comp` at import. `notes/phase45`.
- **A textbook mechanism is a hypothesis, not a finding.** `attention.comp` gives each of
  a subgroup's 32 lanes one row of a stride-64 shared tile: bank `i mod 32` for all 32
  lanes, a perfect 32-way conflict, arithmetic beyond dispute. Measured on Xe2 with a
  30-line probe it costs **1.11x**, not 32x, and the `gather`/`scatter` surgery it would
  have justified was not worth doing. `src/bench/bank_probe.*`, `notes/phase45`.
- **On this model, looking at the picture disagrees with measuring it.** The pass moves
  the *level* — it pulls back blown-out skin by 30-40 % — and that shift dominates every
  raw statistic and every impression. Twice in one session a confident visual reading was
  wrong: "more pores" where fine texture had *fallen* 16 % (it rose 31 % once normalised
  for level), and "the irises turned brown" where the hue moved 18° to 21° and had been
  brown all along. **Normalise for level, and sample the exact pixels you are claiming
  about.** `notes/phase42`, `phase44`.
- **One frame is not a recommendation.** `cinematic` was declared the defensible default
  for faces on the strength of a single cutscene where it landed slightly positive. Three
  frames later it was removing detail — on a bright scene it *smooths*. Any statement of
  the form "profile X is right for Y" needs several scenes. `notes/phase44`.
- **`pgrep -f` and `pkill -f` match the shell that runs them.** Four times now. The
  `[n]ame` bracket trick fixes the pattern but not the case where the string also appears
  elsewhere in your own command line — a heredoc, a comment, an echo. `pkill -f
  nr_daemon.py` killed the very shell launching the daemon. **List with
  `ps -eo pid,args | awk '/pattern/ && !/awk/'` and kill by explicit PID.**
- **The test suite and a loaded daemon do not fit together.** 15 GiB shared with the iGPU;
  a resident daemon holds the model, `make test` allocates its own device buffers, and the
  run gets OOM-killed. Stop the daemon before the suite, not after.
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
  **2.3 GiB** of device buffers at 720p since the scratch arena (`phase32`); the 5.6 GB
  this used to say predates it, and 1080p did not fit at all before.
  `src/gpu/nr_frame_resident.py`, `notes/phase15-residency.md`,
  `notes/phase18-fusion.md`, `notes/phase21-fusion-and-tiling.md`.
- **It runs in a real game, in two modes.** A Vulkan layer captures the presented frame,
  a daemon runs the model, and the result goes back into the swapchain. Proven in **Dead
  or Alive 5** (32-bit D3D9 through DXVK) with faces enhanced and measured, and the layer
  proven to attach under **VKD3D-Proton** on a 64-bit D3D12 title. Photo mode is triggered
  by a file; live mode (`NR_LAYER_LIVE=N`) runs continuously and reaches 10.6 fps at
  512x288. `src/layer/`, `notes/phase34-doa5.md`, `phase41`, `phase47`.
- **HDR is handled**: `src/ref/nr_display.py`, the recovered display codec — encode a
  linear-HDR frame to an sRGB proxy with a soft knee, run the model, fold it back by
  luminance ratio onto the untouched original. Clamping instead destroys 97 % of the
  highlight structure. `notes/phase16-hdr.md`.
- **The interface mask works and has a failure mode.** In a live fight it cut HUD damage
  **tenfold** (mean |d| 9.69 -> 0.97 on the health bar). On a near-static frame it covered
  the *subject* in a speckle instead, and since the pass moves skin by 20-30 levels the
  interleaving read as a mottled crust — worse than the softened HUD it prevents. Two
  guards now: a majority filter narrowing the mask to solid blocks, and a coverage limit
  that drops it entirely above 55 %. `notes/phase42`, `phase43`.
- **The machine's real limits, measured** (`notes/phase20-machine-limits.md`):
  **70-91 GB/s** of memory bandwidth against 136.5 theoretical, the GPU holding its
  **1950 MHz ceiling** throughout a run at 46-48 C, and our GEMM at **8-12 %** of the
  ~32 TFLOP/s FP16 peak — **4.4 %** averaged over the frame's real shapes **before
  specialization**. The GPU has **64** XMX engines. Phase27 demonstrates that some
  register pressure was avoidable shader code; these older throughput measurements
  do not establish the optimized kernel's ceiling. Earlier notes quoted 23 GB/s,
  which was single-threaded numpy and wrong by 3x. Both GEMM and elementwise graph
  operations now run on the GPU.
- **Temporal processing exists** in `nr_temporal.py`; both game modes use the
  single-frame path and supply no engine motion or history. Feature channels 7:10 *are*
  the history slot and the still path fills them with the current colour. Live mode has a
  previous output but no motion vectors, and `phase12` measured the gate at **0.032 with
  wrong motion** — un-reprojected history would be rejected, so it is not worth wiring
  blind. `notes/phase48`.
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
