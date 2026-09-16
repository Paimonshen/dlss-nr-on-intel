# Phase 59 — a third kind of client, and the best live rate so far

2026-09-16. **Tekken 7**, Steam appid 389730, Unreal Engine 4,
`TekkenGame-Win64-Shipping.exe`. **10.5 fps sustained at 640x360**, measured over 20
seconds of play: 210 frames, 91 ms each.

## What it settles

**The layer attaches to 64-bit D3D11 through DXVK**, with no change to anything. That is
the third kind of Vulkan client it has run under:

| game | API | bits | translation | note |
| --- | --- | --- | --- | --- |
| Dead or Alive 5 | D3D9 | 32 | DXVK | `phase34`, needed the 32-bit layer library |
| a D3D12 title | D3D12 | 64 | VKD3D-Proton | `phase41`, attach proven, no picture |
| **Tekken 7** | **D3D11** | **64** | **DXVK** | here |

The API was not taken on trust. UE4 carries the names of every RHI module — `D3D11RHI`,
`D3D12RHI`, `VulkanRHI`, `OpenGLDrv` — in the string table of any build, so the binary
proves nothing; the game's configuration is inside its pak files. What settles it is
`/proc/<pid>/maps` on the running process: `d3d11.dll` and `dxgi.dll` mapped, and
`libnr_layer` present in five mappings.

## Measured

| | |
| --- | --- |
| swapchain | 640x360, no letterbox |
| render scale | 0.55 |
| frame | **91 ms**, 210 frames in 20 s = **10.5 fps** |
| history gate | **0.573** mean, 0.556-0.661 over the window |
| frame held by the floor | **86 %** |
| frame-to-frame change | 0.006-0.022, far under the 0.15 cut limit |

Power-saver, 1.38 GHz, on battery — as `phase57`, so the two are comparable with each
other and not with anything earlier.

**No letterbox.** UE4 hands over a swapchain the size of the window. Dead or Alive 5
letterboxes 16:9 inside a 4:3 window and a quarter of every frame was bars (`phase51`),
which is why `Letterbox` exists; here it finds nothing and costs its eight lines.

## The gate is the highest this project has seen

0.573 mean, against 0.38-0.54 in a Dead or Alive 5 fight and 0.12 on the slow replay that
`phase54` was measured on. The reason is what that phase predicted: the gate is **global**,
so it reads how much of the *whole frame* agrees with its history. Tekken's camera barely
moves, one fighter occupies the middle, and the background is still — so the history is
correct over most of the frame and the model trusts it. A DoA5 fight moves the camera.

With 86 % of the frame also getting the floor, this is the most stable live picture the
project has produced, and it is stable for a reason that is a property of the game rather
than of anything done here.

## On "the best rate so far"

`phase47` recorded 10.6 fps, at **512x288**. This is 10.5 fps at 640x360 — **1.56x the
pixels for the same rate**. The honest caveat: the power profile of that earlier
measurement is not recorded, and this one is on power-saver, so the comparison is
suggestive rather than clean. What is not in doubt is that the frame at this extent is
91 ms today and the `phase47` table has 640x360 at scale **0.35** costing 116.8 ms — a
lower render scale for a longer frame. The native host passes (`phase57`) are the
difference.

## A mistake worth writing down

The daemon was started by the toggle, and this session deleted `/tmp/nr_daemon.log` before
starting a second daemon — which exited, correctly, because one was already listening. The
first daemon's output then existed only through `/proc/<pid>/fd/1`, an unlinked inode. It
was readable there, and the measurements above came from it.

`nr-panel` reads the log **by path** and would have shown nothing. Deleting a file another
process is writing to does not give you a fresh one; it gives that process a private one.
