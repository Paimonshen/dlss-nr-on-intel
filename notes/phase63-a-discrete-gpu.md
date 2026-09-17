# Phase 63 — the first report from someone else's machine: a discrete GPU

2026-09-18. Someone on Reddit ran this on an **Intel Arc B580** (BMG G21, discrete
Battlemage) and reported that the effect made their frame rate worse and that the daemon
ended every frame with `xmx_graph_run: resident submit (-4)`. They also ran
`src/probe/coopmat_probe.c` and `src/gpu/bench.py` and posted both. That output is the
first measurement this project has from hardware other than the Arc 140V it was written on,
and it says something the owner's machine could never have shown.

Everything below about their machine is **reported** — their terminal, not ours. What is
verified here is the code it points at and the numbers on this machine.

## Their hardware is right, and the probe proves it

Their probe output matches ours line for line: `subgroup size 32`, `cooperativeMatrix = 1`,
`supportedStages = compute`, `cooperativeMatrixRobustBufferAccess = 0`, and the same **six**
configs, all scope=subgroup, all M=8 N=16 — including **config 1, `fp16 × fp16 → fp32`**,
which is the path this port takes. `NV_cooperative_matrix2` present with zero flexible
configs, exactly as in `notes/hw-coopmat.md`. Two generations of Xe2, one table.

So their question — "dont really know where to put matrix or if its done automatically" —
has a short answer: nothing to place. The probe only reports. The shaders ask for that
config themselves.

## Their benchmark says the card is 50x slower than an integrated one

Same `src/gpu/bench.py`, same shapes:

| shape | Arc B580 (reported) | Arc 140V (here, measured) |
| --- | --- | --- |
| 512x512 K=512 | 141.1 GFLOP/s | 1269.0 |
| 1024x1024 K=1024 | 52.8 | 3517.2 |
| 2048x2048 K=512 | 49.8 | 3536.3 |
| 4096x1024 K=1024 | 65.0 | 3295.2 |

A discrete Battlemage has several times the matrix hardware of this iGPU and its own
memory. It should win every row. Losing them by 50x is not a slow GPU; it is a GPU reading
its operands from somewhere else.

## Where: `memtype()` asked for host-cached memory on a card that has none

`libxmx.c` chose the memory type for every buffer — weights, activations, scratch — by
preferring `HOST_CACHED` above everything else. That preference is right here and was
measured: on this shared-memory APU the uncached host-visible type ran readback at 80 MB/s
and buried a 1.35 TFLOP/s kernel (`phase8`). Both types are the same physical RAM, so
nothing is lost by asking for the cached one.

On a **discrete** GPU they are not the same memory. `HOST_CACHED` there is system RAM by
construction: the card cannot cache host memory in its own. So every operand sat behind
PCIe and 12 GB of VRAM went unused. The one machine this was written on could not exhibit
it, and no test could have caught it, because the property being asked for is legal and
available on both.

**Changed**: the preference now depends on `VkPhysicalDeviceProperties::deviceType` —
device-local first on a discrete GPU, host-cached first otherwise — with a **1 GiB floor on
the heap**, because with the BAR unresized the host-visible VRAM window is 256 MB, far less
than a frame needs, and preferring it would turn a slow run into a failed allocation. Such a
card falls back to system memory as before.

**Unverified.** There is no discrete GPU here. What is verified is that this machine is
untouched: it is integrated, so it takes the same branch it always took, `bench.py` reads
1269–3709 GFLOP/s before and after, and `make test` is green.

## The benchmark hid the real failure behind an impossible number

Their last row read `8192x512 K=512 … 0.0003 s … 495058.6 GFLOP/s … 0.009 ms`. Half a
petaflop from a card that had just managed 65 GFLOP/s. `bench.py` never checked what
`xmx_gemm` returned, so a call that failed instantly was divided into the FLOP count and
printed as throughput. Whatever went wrong on that shape — and it is the interesting event
in the whole log, since it is where their run stopped working — was reported as a record.

Now checked: a failed call prints `xmx_gemm:` and the library's own message. **A tool that
reports a rate has to check the call before it divides by the time.** This is the second
instance of `phase45`'s lesson: the arithmetic is fine, the input to it was never verified.

## Two more things that came out of the same report

**The daemon now says which GPU it is on** — `model ready in 0.4s on Intel(R) Graphics
(LNL)`. `xmx_init` takes `pds[0]`, the first device Vulkan enumerates, with no vendor check
and until now no record of the choice. On a machine with more than one GPU that is a guess
nobody could audit from the log.

**Their `make test` fails in `test_ui_mask`** with `cannot reshape array of size 0 into
shape (256,384,4)`: the daemon answered nothing, which is what it does when a frame fails.
The harness keeps the daemon's stdout and prints it only if the process exits, so the
message that would name the failure never reached them. The reproducer that does print it
is `python3 src/ref/nr_frame.py in.png out.png --resident` — no sockets, no game, the
exception on the terminal.

## Still open

- **The `-4` itself.** A 50x slowdown makes a submission long enough to hit the driver's
  job timeout far more likely, and a reset is exactly what `VK_ERROR_DEVICE_LOST` reports.
  Plausible, not established — their first failure line would settle more than any amount
  of reasoning here.
- **Readback from write-combined VRAM.** Host reads of device-local host-visible memory are
  slow on a discrete card. If the new preference trades compute speed for readback time,
  the answer is a staging copy, which this code has never needed on an APU.
- **They have offered remote access to a B580 or B570.** That is the only way anything in
  this note gets verified. Owner's call.
