# Small E4M3 experiment and work after the limit reset

2026-10-13. The owner requested only the cheapest task now and the remaining
work after the limits reset. Branch: improve-e4m3, based on master d29b665.

Completed: src/bench/e4m3_probe.comp and .py test the rounding-constant approach
against the actual current publish.glsl on Intel Xe2. The candidate rounds the
magnitude, preserving our existing min/NaN handling and x<0 zero-sign rule.
The source of the idea is mauri870/DLSSNR-RDNA3, commit 5f099e2, MIT,
copyright mochizuki0323; its license is retained under licenses/.
This is a separate public-source engine, not Daniel's closed AMD installer.

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

The production e4m3 function and runtime defaults have not changed. Only the
isolated probe and notes are added; whole-frame validation is deferred.
Private output: work/e4m3-result.json and work/e4m3-probe.log.

To reproduce, build the regular runtime/shaders, then:

```sh
glslangValidator --target-env vulkan1.3 -Isrc/gpu -o work/e4m3_probe.spv src/bench/e4m3_probe.comp
python3 src/bench/e4m3_probe.py
```

## Next after limits reset, in order

1. Try the candidate behind a controlled shader switch. Preserve numerical
   boundaries, validate whole-frame outputs and pair old/new graph timings.
   Repeat on B580/Windows before enabling it by default. A tiny isolated-kernel
   difference is insufficient evidence for a production change.
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
