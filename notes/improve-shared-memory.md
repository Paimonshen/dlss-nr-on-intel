# Shared memory on Xe2: how the driver sizes it, and what that cost the staged GEMM

2026-09-24. The owner asked why window attention has exactly 2 KB of shared memory and what
1 KB or 512 B would do. Answering it turned up a rule in Mesa and a 10 % frame.

## The rule — verified, in the driver and on the hardware

Mesa 26.2.2, `src/intel/vulkan/genX_shader.c:1183`, fills two fields for every compute
pipeline from the same number, `total_shared` — the bytes the shader declares:

- `SharedLocalMemorySize`, what each workgroup is given: the declaration rounded up to the
  Xe2 allocation table — **1 KB at least**, then 2, 4, 8, 16, 24, 32, 48, 64 ... KB
  (`intel_compute_slm_calculate_size`);
- `PreferredSLMAllocationSize`, the shared-memory partition of each core: *workgroups a
  core's threads can hold* x **the declared bytes**, rounded up to 16, 32, 64, 96, 128 ... KB
  and capped — at 128 KB on this machine, measured below
  (`intel_compute_preferred_slm_calc_encode_size`, `src/intel/common/intel_compute_slm.c`).

So the partition is sized from the declaration and filled with the rounded allocation.
Whenever the two differ, fewer workgroups fit than the threads allow. For a 32-lane
workgroup — 64 a core by threads — resident = min(64, partition / allocation):

| declared | allocation | partition | resident | predicted ms | measured ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | — | — | 64 | 11 | 11.0 |
| 64 B | 1 KB | 16 KB | 16 | 44 | 41.5 |
| 128 B | 1 KB | 16 KB | 16 | 44 | 41.3 |
| 256 B | 1 KB | 16 KB | 16 | 44 | 41.3 |
| 384 B | 1 KB | 32 KB | 32 | 22 | 21.1 |
| 512 B | 1 KB | 32 KB | 32 | 22 | 21.1 |
| 768 B | 1 KB | 64 KB | 64 | 11 | 11.0 |
| 1 KB | 1 KB | 64 KB | 64 | 11 | 11.0 |
| **1.25 KB** | 2 KB | 96 KB | **48** | 14.7 | **14.2** |
| **1.5 KB** | 2 KB | 96 KB | **48** | 14.7 | **14.2** |
| 1.75 KB | 2 KB | 128 KB | 64 | 11 | 11.0 |
| 2 KB | 2 KB | 128 KB | 64 | 11 | 11.0 |
| 3 KB | 4 KB | 128 KB (cap) | 32 | 22 | 21.1 |
| 4 KB | 4 KB | 128 KB (cap) | 32 | 22 | 21.1 |

The probe is a chain of 4096 dependent FMAs per lane, no memory traffic, 65 536
workgroups, with `shared uint pad[N]` declared and touched — so its time is only how many
workgroups are resident to interleave. Fourteen sizes, every one where the rule puts it,
including the two that look wrong: **1.25-1.5 KB is 29 % slower than 2 KB**, and 768 B is as
fast as 1 KB. Larger, measured earlier: 8 KB 42.4, 16 KB 86.7, 32 KB 181.6 ms — the cap
halving the resident count per doubling.

For a pipeline here the rule reads: **declare exactly an allocation size, and keep
(workgroups a core holds by threads) x size <= 128 KB.** For 32 lanes that is 1 or 2 KB;
for 128 lanes, up to 8 KB. Anything between two allocation sizes, or under 1 KB, pays.

It is arguably a driver bug — the partition's own comment says it estimates how many
workgroups run at once and multiplies by their size, which only works with the size they
are given — and the fix would be one line: pass `intel_compute_slm_calculate_size(GFX_VER,
total_shared)` instead of `total_shared` (and the same for the task and mesh stages).
**Not reported upstream**; that is the owner's call.

## What it cost: the staged GEMM at half its threads

`gemm_staged.comp` is 128 lanes and declared 15.5 KB — the A and B tiles, 7.5 KB with their
padding, and an 8 KB float stage for the epilogue. Allocation 16 KB; partition 16 x 15.5 KB,
capped at 128 KB: **eight workgroups a core, 32 threads of 64**.

The tiles and the stage are never live at once — the stage is written after the last K
step has read its fragments — so they now share their bytes, as two `shared` blocks
(`GL_EXT_shared_memory_block`, `VK_KHR_workgroup_memory_explicit_layout`, which libxmx now
enables). 8 KB, sixteen workgroups, every thread. One `barrier()` before the stage is the
whole price, and it is load-bearing: without it all three head hashes change and
`test_gemm_qkv.py` fails.

1280x720, paired, two runs each: **staged GEMM 90.5 -> 70.9 ms, device total 219 -> 198.**
Bit-identical — the head hashes of the three reference frames unchanged, `make test` green
in both memory modes. (The tiled GEMM at 2 KB, window attention at 2 KB and the fused FFN at
2 KB were already at 64.)

## And the L1: real, and worth nothing here

Shared memory and the L1 data cache are one array. A chain of 48 dependent loads through a
4 KB table: 2.4 ms at 256 B declared, 3.6 at 1 KB, **10.7 at 2 KB** and above — at 2 KB x 64
the partition is the whole array and the table no longer stays in L1.

Getting it back was tried on the kernels that sit at 2 KB, and did not pay:

- the fused FFN with its hidden chunk and its stage aliased into 1 KB (they are never live
  at once either): 22.1 -> 21.9 ms, inside the noise;
- the tiled GEMM's stage cut to 1 KB, on the float32 path that never touches it: 2-6 % on a
  few memory-bound shapes, the same on the rest.

## Also measured and dropped

- **The softmax pipeline** (`attention.comp`) declares 8 KB for 32 lanes, a quarter of the
  threads, and the frame only runs its bottleneck softmax, which never touches the stage.
  At 1 KB: 4.44 -> 4.04 ms. Not worth a second pipeline.
- **The base GEMM** (`gemm_resident.comp`, RM = RN = 1) declares 512 B, half the threads.
  Padded to 1 KB: no change; it is one pass a frame.

## Reproduce

The probe is small enough to rewrite: a shader with the window-attention push block, `shared
uint pad[N]` touched by every lane and either the FMA chain or a pointer chase, timed through
`Runtime.window_attention` with `XMX_WINDOW_SPV` pointing at each build. Variants of the
real kernels go in through `XMX_STAGED_SPV`, `XMX_ROW_SPV`, `XMX_GEMM_SPV` and
`XMX_FFN_SPV`; `src/bench/frame_profile.py` times them in a frame.
