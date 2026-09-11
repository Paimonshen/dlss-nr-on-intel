# Index of the notes

Eighty-seven files, fifty-six phases. This is what each one settles, so a reader arriving
cold can go straight to the answer instead of the archaeology.

**Read `HANDOFF.md` first** — it carries the current state, the standing conclusions and
the traps. This index is for going deeper on one question.

Notes are the record of what was *measured*, including measurements that turned out
wrong. Where a note is superseded it says so at the top and is kept anyway: several of
this project's worst hours went into re-deriving something a withdrawn note already
disproved.

## Start here

| note | what it settles |
| --- | --- |
| `phase45-frame-profile.md` | where every millisecond of a frame goes, per pass, from GPU timestamps |
| `phase47-live-mode.md` | the live mode, 10.6 fps at 512x288, and the bottleneck leaving the graph |
| `phase38-there-was-no-bug.md` | the "driver bug" that shaped three phases does not exist |
| `phase9-numerics.md` | why bit-identical agreement with a CPU reference is impossible here |
| `reviewing.md` | how to run `/ultrareview` on this repo without wasting a run |
| `reproduce.md` | how to run the resident path from a clean checkout |

## The binary, the weights, the graph

| note | what it settles |
| --- | --- |
| `phase0-acquire.md` | provenance of the DLL, cryptographically |
| `phase1-binary-map.md` | 89 % of the file is weights; no CUDA runtime imports |
| `phase2-graph-rtti.md` | the layer taxonomy, out of intact MSVC RTTI |
| `phase3-weight-format.md` | FP16, not FP8 — the figure that was wrong for days |
| `phase3-ptx-unlock.md` | the `.data` containers are Zstandard and decompress to PTX |
| `phase3-architecture.md` | five encoder/decoder stages, ViT-1D bottleneck, 32 per head |
| `phase3-execution-order.md` | our decode is byte-identical to a capture from real RTX hardware |
| `phase6-mlx-dlss-unpack.md` | **the container is packed backend payloads, not dense FP16** |
| `phase3-block-layout.md`, `-block-internals.md` | the ordered layout inside each block family |
| `phase3-bias-region.md` | the 128C region is the attention bias, in a 12-bit permutation |
| `phase5-softmax-found.md` | the attention is a softmax, hand-rolled in `f16x2` |
| `phase5-attn-scale-fp32.md` | `attn_scale` is FP32 per head |
| `phase5-*` (the rest) | stem, output head, channel order, resampling, I/O structs, narrow blocks |
| `phase5-no-softmax.md` | **withdrawn** — kept so nobody re-derives it |

## Running it

| note | what it settles |
| --- | --- |
| `hw-coopmat.md` | six cooperative-matrix configs, all M=8 N=16, subgroup scope |
| `phase4-subnormal-flush.md` | XMX flushes subnormal FP16 operands (premise later corrected) |
| `phase4-accumulation-choice.md` | NVIDIA accumulates in FP16; we use FP32 and are more accurate |
| `phase7-first-render.md` | the first real frame, with its two adversarial controls |
| `phase8-xmx-graph.md` | the whole graph on XMX, and the graph is chaotic |
| `phase15-residency.md` | activations never return to the host |
| `phase16-hdr.md` | the display codec, so an HDR frame survives the round trip |
| `phase12-temporal.md` | the temporal gate discriminates: 0.705 right, 0.032 wrong motion |
| `phase28-frame-replay.md` | one submission a frame, commands reused |
| `phase27-pipeline-specialization.md` | specialize before believing a register ceiling |
| `phase32-scratch-and-qk.md` | the scratch arena: 720p from 5041 to 2303 MiB |

## Performance, and the levers that are closed

| note | what it settles |
| --- | --- |
| `phase20-machine-limits.md` | 70-91 GB/s of 136.5, the clock ceiling held, 64 XMX engines |
| `phase26-the-register-ceiling.md` | every engine busy, each ~90 % idle; the register file is the wall |
| `phase45-frame-profile.md` | per-pass timings; everything that moves data is at the memory ceiling |
| `phase21`, `phase22`, `phase23`, `phase31`, `phase33` | tiling, staging, integer weights, the accumulator, OpenCL — all measured, all closed |
| `phase25-the-frame-rate-wall.md` | `17 ms + 488 ms per megapixel`, and what that forbids |
| `phase51-output-resolution-costs.md` | what costs the output extent rather than the network's, and a profile that was measuring swap |
| `phase50-what-the-model-computes-in.md` | 36 % of the shipped model's mma is already FP8; what FP4 would and would not change |
| `phase37-neural-upstream.md` | half the extent is 3x faster and keeps 62 % of the high band |
| `phase46-cpu-share.md` | four busy E-cores cost the GPU 7 % for a theoretical +2 % |
| `phase14-npu-and-rounding.md` | the NPU is not worth using; the rounding point was |
| `phase13-torch-and-blas.md` | the system numpy is netlib; every early CPU baseline was 67x too slow |

## In a game

| note | what it settles |
| --- | --- |
| `phase17-integration-landscape.md` | why a Vulkan layer and not NGX |
| `phase19-photo-mode.md` | the loop closed on `vkcube` |
| `phase24-a-real-game.md` | the layer loads inside a Wine prefix |
| `phase34-doa5.md` | a face, from a real game |
| `phase42-doa5-fight-scene.md` | the pass measured on a live fight; the tone shift that fools the eye |
| `phase35-what-the-operator-does.md` | the interface-mask detector, on three real frames |
| `phase43-mask-mottling.md` | the mask making a worse artefact than it prevents, and both fixes |
| `phase44-profile-tradeoff.md` | profiles trade speculars for skin texture; one knob runs backwards |
| `phase41-doa6-and-vkd3d.md` | the layer under VKD3D-Proton; why DOA6LR will not start |
| `phase47-live-mode.md` | live mode, the control tool, 10.6 fps |
| `phase48-feature-inputs.md` | all sixteen feature channels, and two costs that were pure waste |
| `phase52-live-in-a-game.md` | live neural rendering inside a running game, measured |
| `phase53-why-it-flickers.md` | why a static pixel moves 3.3 levels: the network is global |
| `phase54-flicker-fix.md` | the temporal path in live mode; 3.7x less flicker for 3.7 % of the frame |
| `phase55-a-switch-on-a-key.md` | one key over a fullscreen game, and the compositor crash from binding it the wrong way |
| `phase56-the-panel-and-the-manual.md` | every knob on one screen, one table behind the tools and the README, three bugs in a terminal's input path |
| `phase30-control-atlas.md` | what each vendor slider does, and one that does nothing |
| `morning-doa5.md` | a session log, in Russian, kept for its screenshots |

## Reviews

| note | what it settles |
| --- | --- |
| `phase39-layer-review.md` | eight findings in `src/layer`; one was a feature that never worked |
| `phase40-gpu-review.md` | four nits in `src/gpu`, and why that area was already right |
| `phase49-from-the-other-tree.md` | seven layer guards and one real bug taken from `ProjectsCodex` |

## Withdrawn or superseded, kept deliberately

`phase5-no-softmax.md`, `phase18-fusion.md` and `phase36-the-bug-is-narrower.md` (both
superseded by `phase38`), the FP8 reading in the early part of `phase3-weight-format.md`,
the dense-FP16 decode behind `phase4-subnormal-flush.md`, and the shortcut-installing
recipe at the top of `phase55-a-switch-on-a-key.md`. Each says so at the top.
**A withdrawn finding lives on wherever it was written down** — `phase40` found five
places in the *code* still asserting the phase-18 bug as fact, months after the note
retired it.
