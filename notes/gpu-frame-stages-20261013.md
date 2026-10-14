# Reproducing GPU image-path timing

Public samples: [measurements/windows-arc140v-20261014.json](measurements/windows-arc140v-20261014.json).
This allowlisted export includes the six image-stage cases and four paired E4M3
cases, without local paths, process logs, weights or image arrays.

`src/bench/gpu_frame_stages.py` compares the current full-model CPU image path
with GPU preparation/composition. It modifies no installed game or registry.
The first grouped diagnostic completed. A controlled rotating comparison on the
new driver completed on 2026-10-14; see the controlled results below.

## Controlled 101.9150 results, 2026-10-14

Each fixture frame ran in all three modes with rotating order, 25 warm-up frames
and 30 measured frames. Full-model output bytes match exactly at every recorded
position. These are image/daemon processing durations; game capture, kernel IPC,
WSI and return into the game remain outside the benchmark.

| Frame / scale | CPU daemon median / p95 | GPU host-image median / p95 | Device processing median / p95 |
| --- | ---: | ---: | ---: |
| 1280x720 / 0.4 | 56.30 / 61.87 ms | 44.24 / 49.23 ms | 41.72 / 46.11 ms |
| 1280x720 / 1.0 | 229.44 / 249.62 ms | 188.93 / 206.88 ms | 185.21 / 193.99 ms |
| 1920x1080 / 0.4 | 105.49 / 119.33 ms | 75.49 / 84.55 ms | 69.16 / 74.66 ms |
| 1920x1080 / 1.0 | 527.04 / 560.96 ms | 515.90 / 525.09 ms | 512.59 / 519.83 ms |
| 2560x1440 / 0.4 | 179.59 / 201.09 ms | 122.14 / 127.16 ms | 115.19 / 118.95 ms |
| 2560x1440 / 1.0 | 786.39 / 811.04 ms | 743.07 / 755.64 ms | 738.63 / 748.94 ms |

Preparation and composition improve substantially, especially large output with
a lower-resolution network. At full network resolution, a network slowdown in
DEVICE I/O mode offsets much of that saving. Separate timestamped probes confirm
it is also visible on the GPU, not just Python timing: at 1440p the instrumented
network medians were about 628 ms for CPU image I/O and 710 ms for GPU image I/O.
The controlled allocation-policy experiment below did not improve that cost.
The full-resolution DEVICE-I/O network gap remains unresolved; the allocation
and E4M3 probes must not be presented as solving it.

Raw results: work/gpu-frame-stages-interleaved-9150.json and
work/gpu-frame-stages-highres-9150.json. First three cases precede the settings
cache; high-resolution cases include it. The cache removes unchanged 64-byte
control uploads, with no shader or image/history arithmetic change. Exact old
composer source is retained under work/gpu_image_compose.pre-cache-20261014.py.

## Paired DEVICE allocation-policy experiment

Two persistent Python/runtime workers ran the same FP16 features, weights and
graph key. Their order alternated every round, with 25 warm-up and 30 measured
pairs per resolution. The timer covered `ResidentFrame.run_device` only: input
uploads and diagnostic full-head readbacks occurred outside it. This isolates
network I/O allocation from image preparation, composition and temporal history;
its absolute durations are not interchangeable with the three-mode table above.

| Source size, scale 1.0 | Default DEVICE median / p95 | Cached DEVICE median / p95 | Median paired cached/default |
| --- | ---: | ---: | ---: |
| 1920×1080 | 350.65 / 360.17 ms | 377.27 / 387.03 ms | 1.07458 |
| 2560×1440 | 659.99 / 671.63 ms | 692.85 / 701.52 ms | 1.05038 |

The padded network fields were 1920×1152 and 2624×1536, respectively; both
workers used identical field geometry and bytes.

Default DEVICE selected memory type 0, `DEVICE_LOCAL`. The opt-in selected type
3, `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`. Real DEVICE input
and head buffers remained **unmapped in every round** in both workers. Input
hashes, complete head bits, changed-input head bits and graph keys matched
exactly. Cached allocation was slower in all 30 pairs at each resolution, and
the slower medians appeared with either worker running first.

Thus explicitly selecting HOST_CACHED memory without mapping it was approximately
7.5% slower at 1080p and 5.0% slower at 1440p in this network-only experiment.
It was not retained: `NR_DEVICE_MEMORY_CACHED` was removed from production, with
the default policy unchanged. The exact experimental C source and DLL are saved
only in ignored `work/libxmx.memory-cached-probe.c` and
`work/libxmx.memory-cached-probe.dll`. Reproducing that historical experiment
requires those artifacts in an isolated runtime; the removed opt-in is not a
current production switch.

Raw data and ignored harness: `work/device-memory-paired-9150.json` and
`work/device_memory_paired_workers.py`. Metadata records the actual Windows
`32.0.101.9150` driver and source/DLL hashes. The experiment rejects this proposed
allocation change as an optimisation; it does not identify the remaining
whole-pipeline network slowdown's cause.

## Paired E4M3 candidate at higher resolutions

The isolated `codex-improve-e4m3-20261013` checkout at `40cb8e5` compared baseline
and magic E4M3 shader variants with separate persistent model/runtime workers,
alternating order, 25 warm-up and 30 measured pairs. This timer covered complete
model replay with its existing input/head transport. It is a different scope
from the DEVICE allocation probe and from full daemon/image processing.

| Source size | Baseline median / p95 | Magic median / p95 | Median paired magic/baseline |
| --- | ---: | ---: | ---: |
| 1920×1080 | 367.60 / 377.51 ms | 404.69 / 419.45 ms | 1.10150 |
| 2560×1440 | 653.20 / 662.50 ms | 800.95 / 820.30 ms | 1.22433 |

All complete heads and changed-input heads matched exactly. The candidate was
slower in every measured pair at both sizes: approximately **10.1% at 1080p** and
**22.4% at 1440p**. Earlier paired checks had found about 9.6% slower at 320×320
and effectively neutral at 720p. These Windows `101.9150` results provide no
performance reason to adopt this E4M3 candidate for the current path.

The E4 checkout's production files remained unchanged. Its ignored
`work/paired-9150-e4m3-highres.json` stores raw pairs, driver/build metadata and
complete/changed-head hashes; `work/e4_paired_workers.py` is the harness.

## Three timing scopes

| Mode | What is timed |
| --- | --- |
| `cpu-daemon` | Real `nr_daemon.process_connection`: CPU decode, crop validation, preparation, full GPU model, native CPU composition, reply, statistics and history |
| `gpu-image-host-daemon` | Real `process_connection` with `NR_GPU_IMAGE=1`: raw upload, GPU preparation/model/composition, reduced metadata and history, final byte download and reply |
| `gpu-device` | `GPUImagePipeline.process_device`: GPU preparation/model/composition plus reduced metadata and history; fixture upload and diagnostic result download occur outside its timer |

The daemon modes use the same in-memory socket fixture. Receiving a frame copies
bytes into the daemon's reusable inbox, while sending retains a view of the reply
without an extra output copy. Kernel named-pipe traffic, game capture, WSI waits,
copying the result into the game, and the game's own rendering are outside all
three timings. The device timing is therefore the processing component of a
shared-GPU transport, not a measured game FPS or a complete shared-transport frame.

`answer_ready_ms` and `daemon_total_ms` are both recorded. The CPU path maintains
history and logs after sending the reply; the GPU image path currently maintains
history before returning its result. Compare total duration as well as reply
latency, because after-reply work affects the next request.

## Fixed configuration and correctness

Default cases are 1280×720 at network scales 0.4 and 1.0, and 1920×1080 at 0.4.
The current default is 25 warm-up frames and 30 measured frames per mode/case. All modes
receive the same repeating four-frame synthetic sequence, full model, profile,
SDR format and temporal settings:

- Standard profile, intensity/detail/colour strength 1.
- Temporal confidence 1, hold 1, release 24, shot-cut limit 0.15.
- Minimum network extent 320, BGRA8 UNORM.
- Windows' existing NumPy block allocator enabled where available, matching `serve`.

The sequence contains gradients, edges, chroma and small changes below the cut
threshold; it is reproducible without distributing a game capture. Frame hashes
must match exactly across all three modes. Hashing/readback for that check occurs
after the relevant timer. Raw timings and median/p95/min/max are saved in JSON.

CPU `preprocess_ms` runs from completion of inbox reception to `run_features`;
`network_ms` wraps that full method, and its existing input/graph/read split is
saved separately. CPU `compose_ms` ends when the reply is ready. GPU stage values
come from the image pipeline; composition includes its finite-head check and
metadata/history includes its output-change reduction. GPU host transfer and
dispatch time is the remaining interval around those device stages.

`--interleaved` retains a CPU graph frame and a GPU image graph frame using the
same uploaded weights, while each of the three modes keeps its own history. Every
fixture frame is processed in all three modes; their order rotates before the
next frame. All modes warm before the first recorded sample. This reduces clock
and thermal drift between long mode groups.

`--profile-network` adds a separate rotating probe after the primary measurement.
It uses single-recording execution, 25 warm-up frames and 10 measured frames per
mode by default, collecting Vulkan timestamps immediately after the network.
Timestamp instrumentation and command recording change overhead, so those values
are reported separately and never mixed into the replay-mode primary timing.
Allocation notes record the runtime's chosen memory type/properties for GRAPH,
HOST_READ, HOST_WRITE and DEVICE; these notes alone do not establish a speed cause.

Run on AC power with other GPU tests/benchmarks idle. Report the device, runtime
configuration and raw data alongside the timings; the same sequence does not
guarantee identical clocks, temperature or driver scheduling on separate runs.

## Commands

From the continuation checkout on Windows:

```powershell
$env:NR_ROOT = 'D:\NRonWindows\codex-continuation-20261013'
$env:NR_DLL_PATH = 'C:\msys64\ucrt64\bin;C:\VulkanSDK\1.4.357.0\Bin'
.\work\python\Scripts\python.exe .\src\bench\gpu_frame_stages.py --describe
.\work\python\Scripts\python.exe .\src\bench\gpu_frame_stages.py --smoke --output work/gpu-frame-smoke.json
.\work\python\Scripts\python.exe .\src\bench\gpu_frame_stages.py --interleaved --profile-network --driver 101.9150 --output work/gpu-frame-stages-interleaved-9150.json
```

`--describe` creates no GPU runtime. `--smoke` runs one warm-up and one measured
96×160 frame per mode; it validates daemon integration and output parity and is
not a performance result. The default benchmark needs the existing shaders,
native image library and logical model weights. All GPU runs must be serial.

On Linux, use the checkout's configured Python with the same script/options.
`--case 1280x720@0.4` can select a single case; repeat `--case` for more.
Set `--driver` to the actual Vulkan probe version; the script also records the
Windows video driver's CIM version and hashes the runtime DLL and image shaders.

## First grouped diagnostic: correctness passes, performance unresolved

`work/gpu-frame-smoke.json` confirmed the actual daemon CPU and GPU host-image
branches return identical bytes at 160×96. `work/gpu-frame-stages.json` contains
the first grouped run on Arc 140V, Python 3.12.12 and NumPy 2.5.3, with 10 warm-up
and 30 measured frames per mode/case. All 90 sample positions have identical
output hashes across the three modes. The driver version was not captured
in that initial JSON, so these results must not be labelled as driver 101.9150.

| Frame / scale | CPU daemon median / p95 | GPU host-image daemon median / p95 | GPU device processing median / p95 |
| --- | ---: | ---: | ---: |
| 1280×720 / 0.4 | 37.97 / 42.06 ms | 62.58 / 66.72 ms | 66.96 / 70.66 ms |
| 1280×720 / 1.0 | 206.78 / 213.45 ms | 189.27 / 191.94 ms | 206.47 / 220.97 ms |
| 1920×1080 / 0.4 | 95.47 / 101.34 ms | 117.77 / 122.41 ms | 97.97 / 102.36 ms |

The recorded network wall times vary considerably between mode groups, despite
matching pixels. For 720p/0.4 they were 31.16, 49.66 and 56.32 ms. At full 720p
they were 186.35, 178.18 and 197.63 ms. Therefore the first run demonstrates
correctness and produces a reproducible harness, but does not establish a general
performance win or attribute the slower cases to memory type. The rotating run
and separate network timestamps are intended to distinguish image-pass overhead
from changes in network execution and scheduling.

## Read-only overhead audit before further changes

For a stable crop and an established temporal history, `_process_device` currently
performs nine synchronous GPU submissions: crop metadata; current RGB decode;
cut reduction; history resize/features; network; finite-head reduction;
composition; output-change reduction; and three history copies. Both resident
and transfer submission helpers wait on a fence before returning. Before the
identical-settings cache, the composer also uploaded 64 bytes of scalar settings
to a DEVICE buffer every frame, adding a staged transfer/wait. The cache removes
that wait for unchanged settings without changing shaders or image/history
arithmetic. Host transport adds raw upload and final output download. With the
cache, stable unchanged settings yield approximately nine waits for device
processing and eleven for GPU images with host transport,
compared with usually one replay wait in the CPU image path on the shared-memory
140V. A failed cached crop check adds a full crop scan and another wait.

The main full-frame work duplicated outside the network is:

- Decoding raw RGB into `active`, then decoding/resampling it into `inner`.
- Decoding/resampling those raw pixels again inside feature preparation.
- Reading raw RGB again in the composition pass.
- Copying current inner/original/neural images into their previous-frame buffers.
- Writing and scanning a full RGB composition solely for the change statistic.

These passes are real GPU work but do not establish the cause of the earlier
network-time variation. CPU HOST_WRITE/HOST_READ prefer HOST_CACHED memory on
UMA; DEVICE uses the first compatible device-local type and intentionally stays
unmapped. The completed equal-kernel allocation comparison above recorded those
selections and found the proposed cached DEVICE policy slower.

Identical-settings caching is now implemented and current-driver byte/history
parity passed. The next low-risk measurement is per-submit wall cost, followed
by wrapper-only candidates such as recording decode
and cut reduction together, and recording change reduction and history copies
together before reading their reduced metadata. Both preserve the current GPU
shader operation order while reducing host waits. Ping-pong history buffers can
remove the three copies, provided allocated-buffer ownership is tracked
independently of the current/previous roles and resize/fallback teardown is
tested. Those batching and ping-pong changes remain proposals; no kernels were
changed during the read-only audit or to obtain the allocation comparison.
