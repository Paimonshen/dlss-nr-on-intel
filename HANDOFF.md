# HANDOFF — read this first

State of the DLSS-NR on Intel Xe2 project as of **2026-09-09**. CLAUDE.md holds the
original brief; **this file overrides it wherever they disagree**, and after
2026-09-09 they disagree about something foundational.

---

## IT RENDERS (2026-09-09, later)

A frame goes in and a neurally-rendered frame comes out, with the real effect:
eyelashes and eyebrow hairs resolved out of a smeared input, skin pores synthesised,
iris and eyeliner sharpened. `notes/phase7-first-render.md`.

```
python3 src/ref/nr_frame.py IN.png OUT.png --gpu      # 17 s for 384x384 on XMX
python3 src/ref/nr_frame.py IN.png OUT.png            # 38 s, the CPU reference
```

A full **1280x720** frame renders too, network extent 1280x768, 9 GiB peak, no tiling
artefacts.

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
- **Phase 4 is done too**: `src/gpu/nr_xmx.py` puts every GEMM on the XMX units,
  the batched attention included. 384x384 goes 45.1 s -> 16.8 s. All the arithmetic
  is on the GPU; 71 % of a frame is now elementwise numpy.
  `notes/phase8-xmx-graph.md`.
- **The graph is chaotic.** Perturbing the input by a relative 1e-06 moves the head
  by 0.0118 mean on RGB, and by 1e-04 only 0.0156 — it saturates. The E4M3 publishes
  are 6.25 %-step quantizers and about a hundred of them stand between input and
  output. Bitwise CPU/GPU agreement is unattainable **by construction**, and so is
  bitwise agreement with NVIDIA. Judge any change on the composed image and on the
  controls, never per-element.

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

1. **The elementwise work is 71 % of a frame** and it is all in numpy: the E4M3
   publishes, the bit-affine softmax, the fragment-tree cosine normalise, the window
   partition and reverse. Compute shaders for those would be the next real step, and
   it is a port of the graph rather than a hook on its GEMMs. Note this needs the
   activations to *stay* on the GPU between blocks — the win is not in any single
   kernel but in not round-tripping through host memory 71 times.
2. **The temporal path** (`work/mlx-dlss/python/mlxdlss/temporal.py`, pure numpy):
   motion vectors, the reprojected history in channels 7-9, the sign-encoded validity
   in 12-14, the five-tap history filter. This is what "run it in a game" actually
   needs; the single-frame path is only the first frame of a sequence.
3. **The DX12/Proton integration** (Phase 5). Nothing has been attempted here.

Smaller, if wanted: `MIN_MACS` in `nr_xmx.py` was tuned against the reference BLAS
and should be re-swept; `split_group_feed_forward` still issues `2*groups` GEMMs and
could fold the same way the branched one now does.

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
| **The graph amplifies any perturbation above ~1e-06 to a fixed floor** of 9-11 % of the head's sd. CPU vs XMX sits on that floor (0.0044 on the image, against MLX-DLSS's 0.0041-0.0048 gap to NVIDIA itself) | `notes/phase8-xmx-graph.md` |
| Every GEMM weight converts to FP16 losslessly: 579 of the 649 logical tensors are stored F16 and the other 70 are `attn_scale`, which is not a GEMM operand | `notes/phase8-xmx-graph.md` |
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
src/ref/    nr_model, nr_frame, image_io          <- the live path
            hnet_*, forward, run_frame            <- superseded, kept for their findings
src/gpu/    gemm_coopmat*.comp, libxmx.c, xmx.py, tests
```

Git repo initialised 2026-09-08, three commits, **code only** — `.gitignore` keeps the
DLL, the weights, the driver payload and the game screenshots out. `/ultrareview` is
ready to run (user-triggered; the model cannot launch it).

Regression, all should exit 0:

```
python3 src/ref/test_nr_model.py                  # primitives + graph, ~40 s
python3 src/gpu/test_gemm.py                      # worst rel 2.4e-06
python3 src/ref/nr_frame.py IN.png OUT.png --gpu  # the visual check
python3 src/ref/nr_frame.py IN.png OUT.png --profile neutral   # the control: ~37x smaller
```

`nr_frame.py` flags: `--size HxW --profile {standard,neutral,natural,cinematic}
--intensity F --detail-strength F --colour-strength F --frame-index N -v`.

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
- **Not fast.** 16.8 s for 384x384 on XMX, ~90 s for 720p. Every GEMM is on the GPU;
  the remaining 71 % is elementwise numpy.
- **Single frame.** No motion vectors, no history, no temporal path.
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
