# Experimental GPU frame path, Windows continuation

Local work toward 0.0.1.3, based on 7abed9d. No publication or release.
The window's GPU image option remains off by default. A real game benchmark
and matched desktop measurements remain necessary before release claims.

## What is implemented

The same-Arc SDR path keeps raw input, feature preparation, network head,
composition and temporal image history in unmapped GPU buffers. The layer
exports dedicated BDA buffers; the daemon imports their actual allocation size
and memory type after authenticating the pipe client and matching deviceUUID
and driverUUID. The pipe carries 144 bytes of request control and a 16-byte
reply instead of whole images. Small statistics and control values still use
the CPU; this is not a claim that the CPU does no work.

Protocol 1 uses external binary semaphores. Protocol 2 handles unavailable or
invalid exports with completed private fences and ordered control replies.
Both use external ownership release/acquire barriers; neither maps pixels.
Successful FRAME replies follow completion of the daemon's output release.
Resource cleanup requires confirmed release or retained process identity;
unfinished submissions are pinned after timeout. Device/swapchain teardown
drains private fences before destroying referenced objects.

The first implementation requires synchronous live processing on every frame,
SDR RGBA8/BGRA8, no interface mask/capture/dump/meter, and FP16 feature input.
With `NR_GPU_ONLY=1`, unsupported modes retain the game's original GPU image.
The experimental window option now selects this strict policy. The existing host
transport remains available when strict mode is unset; GPU preparation/composition
can also run behind it. See gpu-only-20261014.md for the latest policy and tests.
Tiny images requiring the CPU algorithm's minimum-size enlargement are declined
before GPU capture; there is no unconditional 320-pixel surface cutoff.

Per-device primary surface selection uses the largest recently presented live
surface, stable ties and an optional NR_LAYER_SWAPCHAIN handle. Owner locking
precedes model loading, and session reports use the actual runtime PID.
Transfer command buffers/fences are reused after confirmed completion.
NR_LAYER_TIMING=1 enables sampled bounded median/p95 host timings; fence waits
include scheduling and application waits, not just GPU copy execution.

## Validation on Intel 101.9150

Hardware: Arc 140V, Vulkan 1.4.359; Windows driver 32.0.101.9150.
MSVC builds the runtime and x64/x86 game layers. CPU checks cover ownership,
protocol/session authentication, partial allocation failure, cancellation,
private loader dispatch initialization and timeout-safe teardown.

The controlled Win32 fixture in normal-user execution passed with Vulkan
instance API 1.1 and 1.3. Each run presented 24 frames through the actual layer,
with 22 shared-GPU full-model frames after two trigger-off frames; both extents
(128x64 and 160x96), off/on, scale 0.4 -> 1.0 -> 0.4, resize and cleanup passed.
Khronos synchronization validation reported no errors. Both selected protocol 1.
First-frame graph construction took several seconds and is excluded from any
warm-speed claim. This is a swapchain integration fixture, not a game FPS test.

Earlier restricted-process runs aborted at Win32 swapchain creation even with
NR absent. The identical normal-user baseline passed. Missing private command
buffer loader initialization caused a separate first-present abort with NR;
capturing PFN_vkSetDeviceLoaderData and initializing allocated commands fixed it.
CreateDevice now resolves through its created instance rather than NULL.

A restricted two-process interop fixture passed six exact copies across two
geometries using protocol 2. It observed successful semaphore exports with NULL
handles; that observation must not be attributed solely to the Intel driver:
the normal-user full-path runs exported and used semaphores successfully.
The full-image protocol-2 fixture also passed in normal-user execution on
2026-10-14: 24 presents, 22 gpu-shared-fence frames, both extents, controls,
resize/cleanup and zero synchronization-validation errors.

GPU preprocessing, composition and full-model parity passed before the update:
78 composition cases, exact FP16 preparation/RGB resize checks, eight consecutive
full-model frames with exact output and neural-history equality. Current-driver
image-parity rerun passed all four GPU image CTests after the settings cache.
Production process_device checks forbid full-image
host views/downloads; only reduced statistics are allowed. Cut/delta reduction
differs from NumPy ordering by up to about 1e-6, so a cut exactly at its threshold
can choose a different history reset. This numerical limit remains explicit.

## Measurements and next checks

The initial grouped image benchmark preserved exact output but did not establish
a general speed gain. At that point GPU preparation/composition had nine synchronous
submissions plus an unchanged-settings upload; identical settings are now cached.
Host transport adds upload/download. See gpu-frame-stages-20261013.md for measured
scopes and raw results.
The first interleaved command was blocked by an exhausted escalation-review
limit. After access recovered, the 25-warm/30-sample benchmark completed on
101.9150, including full 1080p/1440p at the owner's request. Exact outputs are
preserved; GPU host-image duration at 1440p/0.4 fell from 179.59 to 122.14 ms.
At full resolution the gain is smaller because DEVICE I/O network execution is
slower. The allocation-policy cause is being isolated; see the measured table
and scopes in gpu-frame-stages-20261013.md.

The equal-network allocation-policy comparison preserved exact heads and
unmapped buffers, but cached DEVICE memory was 7.5%/5.0% slower at 1080p/1440p.
Its opt-in was removed; exact test source/DLL are retained privately under work/.
The final daemon stop/restart smoke passed: 16 presents, two initial shared
frames, an exact retained original pixel during loss, then 13 host-image fallback
frames after restart, zero synchronization-validation errors and owned cleanup.
Shared transport stays disabled after a failed exchange for that device lifetime;
restarting the game/device re-enables negotiation. This recovery policy is explicit.

Next: a real game A/B and B580 measurement. Identical composer settings are now
cached and its host/GPU tests pass.
After measuring, consider batching
reductions/history copies to remove avoidable fence waits without changing pixels.
B580 is not present locally; no desktop FPS or threefold gain is claimed.

An independent improve-e4m3 checkout at 40cb8e5 was checked on 101.9150:
1,114,875 probe inputs and full/changed head hashes match, and candidate GEMM/frame
regressions pass. Alternating two warm contexts gave about 9.6% slower paired
time at 320x320 and essentially equal time at 720p. Added full 1080p and 1440p
paired runs remained exact but were about 10.1% and 22.4% slower respectively.
NR_E4M3_MAGIC stays off;
that branch has not been merged into this work.

All private results, model weights and binaries remain under ignored work/.
The owned fixture harnesses use unique endpoints and stop only their own children.
Original Claude and the independent older GEMM checkout remain untouched.

Final review also covered same-family queue-handle changes and temporary
unsupported settings: queues rebuild the bridge before capture, and a safely
released unsupported FRAME uses deferred retry. Fatal errors retain the block.
Full regression run covered 61 CTests (window_attention excluded); two failures
revealed a stale cache stub and parent-repository release provenance. Both were
fixed, and all 14 affected CPU/GPU regression entries passed on the final build.
Wizard checks: 79 tests with one platform skip, plus six bridge tests.
