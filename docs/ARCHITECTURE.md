# DLSS-NR, as recovered

What the network is, what it expects, and what it returns — written for someone who wants
to run it somewhere it was not meant to run.

The `notes/` directory is the record of how each of these was established, including the
attempts that were wrong; this file is the answer without the archaeology. Where the two
disagree, **this file is current** and the note is dated.

Everything here was recovered from a shipped `nvngx_dlssnr.dll`, build 310.8.0.0, by
reading its container and its embedded code — and cross-checked against
[MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS), an independent extraction of the same
binary from vendor captures. The two agree on all 649 tensors with 0 missing, 0 extra and
0 shape mismatches, which is the strongest evidence available without NVIDIA hardware.

**No weights are here.** This is the shape of the thing, not the thing.

---

## 1. What it is

A **one-step pixel-space diffusion model** that re-renders a frame's detail, conditioned
on the rendered frame, carried temporal state, and three artistic-direction scalars. It is
not a denoiser and not an upscaler: input and output are the same extent, and what it
returns is a *residual* to add to the frame it was given.

Internally: a symmetric U-Net of **71 blocks** — five encoder and five decoder stages of
shifted-window (Swin) attention, an eight-block ViT-1D bottleneck, and an upsample between
them. **73 841 889 parameters, FP16.** Vendor codenames in the binary: feature `CG2R`,
engine `HNet`, configs `crazy-cuckoo` and `hnet-vigilant-squid`.

## 2. The contract — this is the reusable part

### Extent

The network runs on an extent that is **at least 320 and a multiple of 64** on each axis.
A frame smaller than that, or not on the grid, is mirrored outward to fit and cropped back
afterwards; the mirror is a reflection of row and column indices, not padding.

```
1024x576 -> 1024x576      already aligned
 563x317 ->  576x320      rounded up
1920x1080 -> 1920x1088
   64x48 ->  320x320      the floor
```

### Input: 16 channels, float32

| channel | contents |
| --- | --- |
| 0–2 | deterministic noise for this extent and frame index |
| 3 | constant 1 |
| 4–6 | the current frame, scaled |
| 7–9 | the **previous output**, reprojected along motion, scaled the same way |
| 10 | normalised style index |
| 11 | local tone strength |
| 12 | local structure strength |
| 13 | skin structure strength, or −1 when the automatic mask is off |
| 14 | automatic-mask structure strength, or −1 |
| 15 | unused, zero |

The colour scaling is three FP16 roundings and is **not** a single multiply:

```
scaled(x) = half(half(half(x) - 0.5) * 0.125)
```

On a first frame, channels 7–9 repeat channels 4–6 — the model is told the history is the
current frame. The noise is a function of the extent and the frame index and nothing else,
so it is worth memoising; it is a Gaussian pair built from a hash of the pixel coordinates
and the frame index, rounded to FP16.

With a per-pixel **control mask**, channels 11 and 12 become that mask's green and blue
times the corresponding strength, and 13/14 go to zero.

### Output: a 4-channel head

| channel | contents |
| --- | --- |
| 0–2 | RGB residual |
| 3 | the temporal gate, as a logit |

The composition, with every rounding point that matters:

```
predicted = clamp(colour + half(head.rgb) * 0.25, 0, 1)
alpha     = clamp(sigmoid(half(head.a)) * half(0.73974609375), 0, 1)
output    = predicted + alpha * (history - predicted)
```

`history` here is channels 7–9 recovered as `channels * 8 + 0.5`. An `intensity` control
then blends the result against the untouched frame — above 1 it extrapolates past the
model's own picture, which the vendor's panel allows up to 2.

### The controls

Four profiles, which are three scalars and nothing more:

| profile | style | tone | structure |
| --- | --- | --- | --- |
| `standard` | 0 | 1 | 1 |
| `natural` | 1/128 | 1 | 1 |
| `cinematic` | 2/128 | 1 | 1 |
| `neutral` | 0 | 0 | 0 |

They are a trade, not a quality ladder: everything the pass adds to skin texture it takes
out of speculars and colour. `notes/phase44-profile-tradeoff.md` measures the curve.

## 3. The graph

Five encoder stages at **32 / 64 / 128 / 256 / 512 channels**, head dimension **32**, so
1/2/4/8/16 heads; an eight-block **ViT-1D** bottleneck at C = 1024; `dec_input_upsample
1024 -> 512`; five decoder stages back. Attention is shifted-window over **8x8 windows, 64
tokens**.

Every block is a packed blob of several parameters concatenated, which is why the on-disk
sizes never factor into `in x out`. The budgets solve exactly:

- **Grouped-query attention, 4:1.** `QKV = 1.5C² = Q(C²) + K(C²/4) + V(C²/4)`.
- **Attention bias** is `heads * 64 * 64 = 128C`, stored in a 12-bit fragment permutation.
- **Transition blocks** are a standard block plus one extra `C²`.
- **The cosine gate** is exactly `C` long, after eight zero pad bytes, at every width. It
  is the anchor that makes the fused Swin layout readable — getting it wrong puts the gate
  inside `wq` and everything downstream is quietly wrong.

The non-linearity in attention is a **softmax**, hand-rolled in `f16x2` with hard logit
clamps and **no max subtraction**. `attn_scale` is FP32, per head.

`notes/MODEL-SPEC.txt` is the per-block table: width, sub-layer count, element count and
layout for all 71 blocks, accounting for every one of the 73 841 889 parameters with zero
remainder. `src/tools/model_spec.py` regenerates it.

## 4. The weights

**The container does not hold plain dense FP16.** It holds packed backend payloads —
permuted into `mma` fragment order, partly E4M3 — and the fact that `data_len == 2 *
n_elem` fixes the byte count, not the encoding. Read as dense FP16 the values correlate
**−0.02** with the truth. Use a logical, named, shaped extraction; MLX-DLSS's tools
produce one from a DLL you supply.

There is **no dequantisation step**: the weights are used as they arrive.

### The trap that costs a quarter of the network

**27.22 % of the parameters — 20.1 M of 73.8 M — are FP16 subnormals.** The median tensor
is 9.64 % subnormal, the worst is 82.92 %, and 20 tensors are over half. Intel's XMX units
flush subnormal FP16 operands to zero, so run naively a quarter of the network evaluates
to zero, silently, with no error anywhere. A per-tensor `2^k` rescale fixes it exactly and
returns the residual error to the 5e-06 of ordinary FP32 accumulation.

Any matrix unit with flush-to-zero behaviour will have this problem. It is the single most
expensive thing to discover late.

## 5. Numerics, and what agreement is possible

**NVIDIA accumulates in FP16.** Zero of 218 PTX kernels use an FP32 accumulator. An
FP32-accumulate implementation is therefore 400–800x *more* accurate than the original on
an isolated GEMM — it is not a reproduction of it. Which you want depends on whether you
are matching their output or making a good picture.

**Per-element agreement is not a property a port can have.** The graph is chaotic: a
relative 1e-06 perturbation of the input moves the head as much as an FP16 GEMM does,
because roughly 100 E4M3 publishes, each with a 6.25 % quantum, stand between input and
output. NumPy's own float32 GEMM carries more error than the threshold below which
perturbations vanish. Judge on the composed image and on whether the controls behave;
`notes/phase9-numerics.md` has the measurements.

## 6. The temporal path

The previous output, reprojected along motion vectors into channels 7–9, blended back
through the head's fourth channel. The gate is learned and it discriminates:

| history given | gate |
| --- | --- |
| none | 0.008 |
| correct, reprojected | **0.705** |
| wrong — zero motion on a panning scene | **0.032** |

That last row is ghosting rejection, and it is why the vendor's output is temporally
stable when a single frame through the same network is not. Static-scene flicker falls
3.6x by the fourth frame.

**The gate is not local.** On a frame where most of the picture moves it reads about 0.12
even over pixels that did not move at all — the network is global, so a history that
disagrees over most of the frame is distrusted everywhere. `notes/phase54-flicker-fix.md`
measures this and what to do about it when you have no motion vectors.

## 7. What was not recovered

- **Which named parameter occupies which slice of each packed blob.** The totals are
  pinned and the names are known; the internal assignment is not. It does not matter if
  you use a logical extraction, which is why this stopped being urgent.
- **Which of `crazy-cuckoo` / `hnet-vigilant-squid` this blob is**, and whether both
  configurations ship.
- **The window shift offset.** Shifted windows are confirmed by `_shifted` kernel names
  and the window is 8x8; the shift itself is inferred, not read.

## 8. Where the evidence is

| question | note |
| --- | --- |
| the per-block table | `notes/MODEL-SPEC.txt` |
| widths, stages, kernel inventory | `notes/phase3-architecture.md` |
| kernel → layer class → template config | `notes/ptx-kernel-configs.md` |
| the container format | `notes/phase3-weight-format.md` |
| the subnormal flush | `notes/phase4-subnormal-flush.md` |
| the accumulator choice | `notes/phase4-accumulation-choice.md` |
| softmax and `attn_scale` | `notes/phase5-softmax-found.md`, `phase5-attn-scale-fp32.md` |
| the attention bias region | `notes/phase3-bias-region.md` |
| the 16 input channels | `notes/phase48-feature-inputs.md` |
| the controls | `notes/phase30-control-atlas.md` |
| the temporal path | `notes/phase12-temporal.md`, `phase54-flicker-fix.md` |
| why bit-exactness is impossible | `notes/phase9-numerics.md` |

`notes/INDEX.md` maps all of them.
