# Phase 72 — preset B, NVIDIA's faster model: read out of `.text`, unpacked, running (2026-10-17)

`nvngx_dlssnr.dll` 310.9.16.0 (sha256 `2bc19bc0…28789f`, HANDOFF 10-17) carries two networks.
**Preset B is preset A's graph with every encoder and decoder stage cut to depth 2**: the same
block families, the same eight-block bottleneck, 31 blocks against 71. One function in `.text`
builds both from five stage depths. Unpacked through MLX-DLSS's own decoders and run through
this port, it renders a frame much as A does. On this Arc 140V's resident path it takes
0.54-0.58 of A's time: the full-resolution blocks and the bottleneck are A's and cost the same.

Everything below was read with REA 6.5.0 and Ghidra 12.1.2 on the DLL (image base
`0x180000000`), or measured here. What was not established is listed at the end.

## The builder: one function, two depth arrays

`FUN_1800602e0` makes the descriptor list. It calls the builder `FUN_18005c5a0(descriptor, 0,
depths)` twice:

| preset | depths, `.rdata` | name the descriptor gets | blocks |
| --- | --- | --- | --- |
| A (id 1) | `{4, 4, 6, 8, 8}` at `0x1800de990` | `CC_Control_History_Blend_Quantize_With_Teacher_honest_tench_2026_07_04_22_30_weights` | 71 |
| B (id 2) | `{2, 2, 2, 2, 2}` at `0x1800de9a8` | `hnet-crazy-cuckoo-small-fp8` | 31 |

The builder is 9 KB. It names the network `CCNetwork` and its family `crazy-cuckoo`, and
gives it the name `hnet-vigilant-squid`, which the caller overwrites with the preset's.
It is generic over the depths:

- the stem, a `single_layer_block` with `layer0.dw_weight`;
- four encoder stages at 32, 64, 128 and 256 channels, `depths[0..3]` blocks each. The last
  block of each also registers `layer0.dw_weight` and `layer0.conv_weight`: the transition
  that downsamples;
- the 512 stage, `depths[4]` split blocks, the last flagged to carry the projection to 1024;
- **eight ViT blocks, a loop fixed at 8** (flags for the first and the last), for both presets;
- the merge (`0x400 -> 0x200`: `conv_weight`, `dw_weight`, `sin`), then the decoder mirrored:
  `depths[4]` split blocks, and at each lower level an upsampling block and `depth - 1`
  window blocks; then the head (`FUN_18009e080(.., 1, 1)`).

**The shifted windows** come from two helpers. `FUN_180059c80(i)` for the encoder and
`FUN_1800568f0(depth, i)` for the decoder give a phase, `i & 3` and `(depth + i) & 3`. The
phase picks none, both, x or y (`(-1,-1)`, `(1,1)`, `(1,0)`, `(0,1)`). The stem is phase 0
and the head phase 1. That rule reproduces the table this port took from MLX-DLSS for every
one of A's 71 blocks, including the decoder's offset of two in the depth-6 stage
(`src/ref/test_topology.py`). For B it gives the encoder's two blocks of a stage phases 0
and 1, and the decoder's phases 2 and 3.

**The skips** are wired by role, as for A. Each decoder level's first block takes the block
before it and that level's encoder transition. The merge takes the last ViT block and the
last 512 block, and the head takes the last decoder block and block 0.

**The registry** that maps presets to weights (`FUN_180023b80` walks it) holds one 656-byte
entry per preset, at `0x1800dbe30` (A) and `0x1800dc0c0` (B). They differ in the name, the
id, the resource (`WEIGHTS_HT` / `WEIGHTS_CC_SMALL_FP8`) and the short name
(`CC_SILVER_AARDWOLD` / `CC_SMALL_FP8`), and in nothing else. The style grades of `phase70`
(-0.1, -0.25, -0.1, -0.15) sit in both, unchanged. The debug overlay (`FUN_18001c120`)
calls B `hnet-almond-eagle-11m`.

`src/ref/nr_topology.py` is that builder: block indices, roles and window phases from five
depths, and the depths read back from a set of weight names.

## The weights

`WEIGHTS_CC_SMALL_FP8` is 113 893 343 bytes in the same container as `WEIGHTS_HT`: 77
records against 153. **Every record has the byte size of one of A's record families**
(B's block 1 is A's 1-3, 2 is 4, 9 is 23-29, 10 is 30, 11-18 are 31-38, 19 is 39, and so
on), and **none is byte-identical to any of A's**: B is trained separately, not cut down
from A.

MLX-DLSS's unpacker decodes record families by A's block numbers, so
`src/tools/unpack_preset.py` decodes each of B's records as the A block of the same role
and renames the tensors back. Its `--preset A` reproduces all 649 of
`dlssnr-logical.safetensors`'s tensors exactly, which checks the renaming end to end. B
comes out as 281 tensors over 31 blocks, each through that family's size and trailer
checks.

- **113 342 375 parameters**, counted as `phase61` counted A's 145 755 123: 22% fewer.
  NVIDIA reported "24% smaller" and "roughly 118 million". The bottleneck is 100 679 944
  in both, which is 89% of B.
- Each matrix has its family's `1/sqrt(K)` spread, as in A (sd 0.1767 at K=32, 0.0625 at
  256, 0.0442 at 512, 0.0312 at 1024). The gates lie in [0, 1] and the scales are positive.
- `blend_scale`, the history gate's ceiling, is **0.76708984375** (A: 0.73974609375).
  The daemon reads it from the weights now (`nr_frame.blend_scale_of`).
- The bottleneck's `attention_scalar` is about 1e-6 in B (0.109 and -558.5 in A).
  Neither MLX-DLSS's graph nor ours uses it.

## The kernels of 310.9.16

The PTX is FP8 only now: 120 entries, 112 names, every one of them present in 310.8.
With labels and symbol names made canonical, **66 are instruction for instruction 310.8's**.
The other 54 differ by register numbering, scheduling, two memory fences, a store loop's
bounds halved from 256 to 128, or integer addressing. Float arithmetic changed in
one place, the pre-block (`cc_tinlayout_fused_pre_block_*`): its parameter struct grew by a
pair of fields at +160, with new multiplies and selects around constants 0.5 and 1.0, and
the only new hint string in the DLL is `DLSSNR.MaskBlendMode`. That is the hint's
arithmetic, and the default path is unchanged (below). The bottleneck's
`cc_vit_1d_attention` gained a loop bound of `ceil(tokens / 128)`. Both kernels serve both
presets, so these are differences between 310.8 and 310.9.16, not between A and B.

## Running it

| | A | B |
| --- | --- | --- |
| CPU reference, 320x320 | 10.4 s | 5.7 s |
| device against the CPU reference, 320x320 (`test_resident.py`) | corr 0.9809, spread 1.0067, 415 passes | corr 0.9870, spread 1.0068, 179 passes |
| `nr_frame.py --gpu`, DOA5 1280x720, whole run with the load | 41.5 s | 23.8 s |

The resident path, presets alternating, median of 7 after 5 warm-up frames:

| network field | A | B | B/A |
| --- | --- | --- | --- |
| 576x320 | 34.0 ms | 19.2 ms | 0.56 |
| 640x384 | 41.4 ms | 22.5 ms | 0.54 |
| 896x512 | 64.8 ms | 37.2 ms | 0.57 |
| 1280x736 | 124.3 ms | 70.3 ms | 0.57 |
| 1920x1088 | 267.6 ms | 154.1 ms | 0.58 |

Why not the 0.44 the block count suggests: at 1280x736, one submit per block, A's 137 ms is
86.8 window stages, 18.1 split, 10.9 bottleneck, 10.8 last block and head and 10.6 stem and
block 0. B's 72 ms is 35.7, 4.3, 10.9, 10.9 and 10.5. **The full-resolution blocks and the
bottleneck do not shrink, and they are 45% of B's frame.** They are the next place to look
for B's speed.

On Dead or Alive 5's 1280x720 frame (`work/doa5/in.png`, through `--gpu`), B changes the
frame by 0.0351 on average against A's 0.0364. On the two faces, luma with fine detail as
the residual of a sigma-2 blur, level-normalised as `phase42` asks:

| face | level: game, A, B | fine / level: game, A, B |
| --- | --- | --- |
| Bayman | 0.419, 0.381, 0.368 | 0.0990, 0.1080, 0.1114 |
| the left one | 0.589, 0.511, 0.492 | 0.0597, 0.0695, 0.0682 |

Both presets pull the level down by 9-16% and add 9-16% of fine detail for it. B sits
within A's range on both faces, a little further on the level. The images stay local.

## `DLSSNR.MaskBlendMode`, and whether faces are milder

News of 10-13 say NVIDIA's revised comparison images show milder changes to faces, alongside
the smaller model. The owner asked whether `DLSSNR.MaskBlendMode`, the only new hint, is
behind it.

**What the hint does.** 310.9.16's parameter parser (`FUN_18001a020`) reads it as 0-3, and
anything else becomes 0, the default. 310.8's (`FUN_180019f30`) had no such hint: a control
mask always switched the automatic mask off. 310.9.16 does so only in mode 0. The pre-block
kernel then fills channels 11-14 per pixel; its feature filler is
`HNetCpp::cc_fused_pre_swin_impl::CCPreBlockShmemFiller`:

- tone is the mask's green times the tone strength, and structure the mask's blue times the
  structure strength, with 1 where there is no mask;
- with the automatic mask off, the skin and automatic-mask lanes are -1;
- with the automatic mask on, structure is 1 and the lanes carry the two strengths;
- with a control mask and mode 1, 2 or 3, the automatic mask stays on, and each strength is
  averaged with, lowered to or raised to the masked structure.

That is all of the pre-block's new float arithmetic: the two 0.5s, the selects and the
compares. Both DLLs default to the automatic mask off and a skin strength of -1, so **310.9.16's
default output is 310.8's**. The mode matters only to a game that sends a control mask and
turns the automatic mask on.

**A bug in our port that it exposed.** With a control mask, the vendor puts -1 in the two
automatic lanes, in both versions. MLX-DLSS's feature builder, which `nr_frame.py
--control-mask` used, put 0. An all-ones mask should change nothing, but with the 0 it
moved Dead or Alive 5's frame by 0.025 on average under A (0.022 under B), max 0.40, which
is two thirds of the pass. With -1 it reproduces the unmasked frame exactly.
`nr_frame.make_features` now fills the lanes as the vendor does and takes the blend mode
(`--mask-blend-mode`), and `nr_temporal.py` follows it. The live daemon never puts a mask
into the features, so it is unaffected.

The kernel writes -1 there only while a flag in its parameters (+200) is set. That the flag
is set is an inference: OpenDLSS-NR, which claims the vendor's inputs bit for bit, writes -1
or the strengths, never 0.

**Is B milder on faces?** Measured here on five frames of Dead or Alive 5 and Mortal Kombat 1,
with eight faces boxed by hand. The figures are the mean |change| in luma on the faces and on
the rest of the frame, through the resident path:

| setting | faces, A → B | rest, A → B | faces / rest, A and B |
| --- | --- | --- | --- |
| standard, no automatic mask (the live default) | 0.0751 → 0.0717 (−5 %) | 0.0372 → 0.0333 (−11 %) | 1.71-3.00, 1.70-3.21 |
| the vendor panel's: structure 1.5, automatic mask, skin 2.0 | 0.0743 → **0.0553 (−26 %)** | 0.0377 → 0.0259 (−31 %) | 1.64-2.95, 1.73-2.99 |
| automatic mask, skin structure 0 | 0.0504 → 0.0522 (+4 %) | 0.0338 → 0.0307 (−9 %) | 1.09-2.49, 1.13-2.73 |

B is not milder on faces in particular. Under the vendor panel's settings it changes the
whole frame about 30 % less than A, faces and environment alike, which a comparison of faces
would show as milder faces. The skin control works in both, a little more selectively in A.
Five frames from two games is a small sample, and the vendor's own output was not compared.

(Both DLLs also map the app's quality mode to the network's scale, MaxQuality and DLAA 1.0
and UltraQuality 0.85, in `ComputeScalingRatioCommon`. 310.8's per-frame parser read
`DLSSNR.ScalingRatio` and overwrote it with 1.0, and 310.9.16's does not read it. Not
followed further.)

## Not established

- **The vendor's own output for B.** Nothing here was compared with NVIDIA's B. That needs
  the RTX 5060 Ti with `DLSSNR.Hint.Render.Preset` set to B.
- Whether the attention's new loop bound changes what the default path computes. The port
  follows 310.8 there, for A and for B. (The pre-block's new arithmetic does not: it is the
  mask blend modes, above.)
- B in a game. The daemon runs it with `NR_PRESET=B`, which has not been tried in a game.
- Windows. The setup window extracts preset A only.

## Reproduce

```
python3 src/tools/unpack_preset.py work/dl-310.9.16/nvngx_dlssnr.dll \
    work/mlxw/dlssnr-b-logical.safetensors                       # preset B
python3 src/tools/unpack_preset.py work/dl-310.9.16/nvngx_dlssnr.dll /tmp/a.safetensors \
    --preset A --check work/mlxw/dlssnr-logical.safetensors      # the renaming, end to end
python3 src/ref/test_topology.py
python3 src/gpu/test_resident.py                                 # both presets' whole graph
python3 src/ref/nr_frame.py IN.png OUT.png --gpu --weights work/mlxw/dlssnr-b-logical.safetensors
python3 src/bench/frame_profile.py --weights work/mlxw/dlssnr-b-logical.safetensors
NR_PRESET=B ...                                                  # the daemon, on B
```
