# Small E4M3 experiment and work after the limit reset

2026-10-13. The owner requested only the cheapest task now and the remaining
work after the limits reset. Branch: improve-e4m3, based on master d29b665.

Completed: src/bench/e4m3_probe.comp and .py test the rounding-constant approach
against the actual current publish.glsl on Intel Xe2. The candidate rounds the
magnitude, preserving our existing min/NaN handling and x<0 zero-sign rule.
The source of the idea is mauri870/DLSSNR-RDNA3, commit 5f099e2, MIT,
copyright mochizuki0323; its license is retained under licenses/.
This is a separate public-source engine, not Daniel's closed AMD installer.

## Follow-up: complete graph checked, 2026-10-13

The owner requested the next inexpensive task and testing if usage allowed.
publish.glsl now has compile-time NR_E4M3_MAGIC, **default 0**. Setting it to 1
uses the magnitude-based candidate. No runtime default is changed.
frame_replay.py --shader-dir selects a complete variant set before constructing
the backend; separate processes and directories avoid overwriting baseline shaders.

On Linux, device **Intel(R) Graphics (LNL)**, six paired rounds per invocation:

| Output | Network field | Order | Baseline replay ms | Candidate replay ms | Lower time |
| --- | --- | --- | ---: | ---: | ---: |
| 320x320 | 320x320 | baseline then candidate | 23.840 | 22.949 | 3.74% |
| 320x320 | 320x320 | candidate then baseline | 24.090 | 22.923 | 4.85% |
| 1280x720 | 1344x768 | baseline then candidate | 147.434 | 142.742 | 3.18% |

Both the original and changed-input head SHA256 match across variants in every
comparison. Within each invocation block, single and replay paths also agree.
Additional candidate runs pass test_gemm_contract.py and test_frame_execution.py,
covering epilogues, resolution changes, specialization, scratch planning, cached
replay, changed inputs and recording-failure recovery.

These are short synthetic backend-frame timings, including input/output movement
around the resident graph, not game FPS or complete daemon preprocessing/composition.
320 reverse order corroborates a small gain on this machine; 720 has one ordering.
Windows/B580 has not been tested. Keep the option off pending that check.

The two directories were compiled from the same Makefile shader inventory with
glslangValidator --target-env vulkan1.3 -Isrc/gpu -DNR_E4M3_MAGIC=0 or =1;
source/defines for each shader are read by src/tools/build_check.py:make_shaders.
Runtime libraries and local weights were reused. Reproduction after compiling:

```sh
python3 src/bench/frame_replay.py --size 320 320 --pairs 6 --shader-dir work/e4m3-baseline --json work/e4m3-baseline-320.json
python3 src/bench/frame_replay.py --size 320 320 --pairs 6 --shader-dir work/e4m3-magic --json work/e4m3-magic-320.json
```

Raw JSON/logs are kept under work/e4m3-*. No model outputs or weights are committed.

## Result on the local Linux Intel GPU

- 1,114,875 inputs: every FP16 bit pattern, random FP32 bits, E4M3 midpoint
  neighbours, positive/negative zero, infinities and NaNs.
- GPU current versus candidate: **zero bit mismatches**, including 6,064 NaN
  inputs and 6,068 nonfinite inputs in total.
- Current GPU versus the independent CPU oracle: zero mismatches for normal
  finite values and zeros. 4,060 FP32 subnormal inputs are excluded from that
  CPU comparison; an initial CPU run differed only in this domain. GPU-vs-GPU
  still includes all of them. This does not establish behavior on other drivers.
- One short alternating timing run, 25 warm-up pairs and 10 measured pairs:
  current median 0.05311 ms/pass; candidate 0.05170 ms/pass (~2.7% lower).
  Ranges overlap: current 0.05201–0.05593, candidate 0.05059–0.05260 ms.
  Each measurement includes recording/submission/fence for 32 dependent passes
  over one reused buffer. It is not a complete model or game-frame benchmark.
  An earlier exploratory run was closer to 1%; a stable speed gain is not claimed.

At the initial probe commit 37ae423 the production e4m3 function was unchanged.
The follow-up above adds a gated candidate while retaining the default algorithm.
Private output: work/e4m3-result.json and work/e4m3-probe.log.

To reproduce, build the regular runtime/shaders, then:

```sh
glslangValidator --target-env vulkan1.3 -Isrc/gpu -o work/e4m3_probe.spv src/bench/e4m3_probe.comp
python3 src/bench/e4m3_probe.py
```

## Next after limits reset, in order

1. Repeat the gated candidate on B580/Windows, and repeat larger-field timings
   in reversed order before enabling it. The Linux whole-graph checks above pass;
   independent-driver validation remains necessary.
2. Join the existing fused FFN and C32 attention block. Measure register pressure,
   spills, image correctness and graph time; do not duplicate the FFN fusion
   already present in the current source.
3. GPU feature generation and output composition, connected to shared game GPU
   resources. Design adapter identity, ownership, synchronization and lifetime
   together to remove full-frame host/IPC trips.
4. Consider dependency-driven multi-block execution only after the simpler work.
   Removing barriers alone does not implement this safely.

Research baseline: the current graph plans 415 passes and one cached submission;
the old 1664-submit explanation is obsolete. Daniel v0.6.0's RDNA3 code really
uses FP16 WMMA; its Swin chain/run kernels are trap stubs, unlike RDNA4.
The complete local investigation is under work/amd-review-20261013/REPORT.md
in the workspace root, with retained REA/LLVM evidence. No extracted AMD binaries
or raw decompiled code belong in this source branch.
