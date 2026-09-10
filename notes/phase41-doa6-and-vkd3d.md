# Phase 41 — the layer works under VKD3D-Proton, and why DOA6LR does not start

2026-09-10.

## The result that matters for this project

**The Vulkan layer attaches to a D3D12 game under VKD3D-Proton.** Until now it had only
ever been driven through DXVK on a 32-bit D3D9 title (`notes/phase24`, `phase34`), and
VKD3D is a different Vulkan client entirely. Attached to DOA6LR it reported:

```
[nr_layer] active; socket=/tmp/nr_layer.sock trigger=/tmp/nr_trigger
[nr_layer] swapchain 800x450 format 44, 3 images
[nr_layer] the daemon did not answer; frame unchanged      (x4)
```

Format 44 is `VK_FORMAT_B8G8R8A8_UNORM`, one of the five the daemon decodes, so the
new four-bytes-a-pixel guard (`notes/phase39`) passes it. Four presents were
intercepted. Nothing about the 64-bit path or the D3D12 swapchain needed changing.

So the photo mode is ready for D3D12 titles. What is not ready is this particular game.

## Where DOA6LR dies

Verified from a Proton log, a Wine `+file` trace, and our own layer:

| stage | result |
| --- | --- |
| module load, `SteamAPI_Init` | OK |
| reading `fdata_package/*.fdata` | OK — the data layer works |
| `D3D12CreateDevice` via vkd3d-proton | OK — **DX Ultimate, SM 6.8, DXR 1.1** |
| swapchain 800x450, 3 images | OK — 4 frames presented |
| `ChangeProperties: Reallocating swapchain (1920 x 1200)` | logged |
| anything after that | **nothing, for 4.4 s** |
| exit | orderly: Streamline plugins unloaded, no exception |

The game asks to go fullscreen and then stops presenting. vkd3d recreates a swapchain
inside the present task, so no present means no recreation — our layer never saw a
1920x1200 swapchain either. Four seconds later the process shuts down cleanly. There is
no `c0000005`, no Vulkan error, no failed file open in the game's own tree.

## What it is not

Each of these was tested and changed nothing — same stage, same 4.4 s, same exit:

- **Proton version.** Hotfix in a fresh prefix fails identically to Experimental.
- **Saved graphics settings.** Moved `GRAPHICSSETTING` aside; identical, and the game
  did not even rewrite it.
- **esync / fsync.** `PROTON_NO_ESYNC=1 PROTON_NO_FSYNC=1`; identical.
- **XWayland vs native Wayland.** `PROTON_ENABLE_WAYLAND=1`; identical.
- **The Streamline plugins.** `sl.dlss`, `sl.dlss_g`, `sl.reflex` disabled; identical.
- **Aspect ratio.** A 1280x720 (16:9) Wine virtual desktop fails at the same point as
  the 1920x1200 (16:10) panel, so the panel's shape is not it.
- **The GPU stack.** The device exposes every modern feature the title could want.

## The one real configuration bug, and why fixing it is not enough

`version.dll` and `winmm.dll` sitting next to the exe are **Ultimate ASI Loader** — the
PDB path inside them says so. Wine's default override order is builtin-first, so it
loaded its *own* `version.dll` and the injection chain never ran:

```
...\Dead or Alive 6 Last Round\VERSION.dll  ::  builtin      <- Wine's, not the game's
```

`WINEDLLOVERRIDES="version=n,b;winmm=n,b"` fixes that, and `nt_file_dupe.asi` then
loads. It also immediately fails:

```
3542: Loaded   ...\nt_file_dupe.asi : native
3543: Unloaded ...\nt_file_dupe.asi : native
```

Same millisecond, adjacent lines — `DllMain` returned FALSE. It hooks NT file APIs,
which is not something that survives Wine's ntdll.

Everything the game needs *from the system* works. What does not work is the
third-party shim bolted onto this build, and that is outside anything this project
controls. The deterministic, orderly, always-at-the-same-point exit fits that and fits
nothing else that was tested.

## Standing conclusion

For this project's purposes **Dead or Alive 5 remains the live target** — it runs, the
pass runs inside it, and the untested piece (the interface mask on a moving HUD) needs
that game, not this one. DOA6LR is worth returning to only if a build appears whose
startup path works under Proton; the Streamline prize described in `phase37` — the
`kBufferTypeHUDLessColor` / depth / motion-vector tags, verified present in this game's
`sl.common.dll` — is unreachable while the game will not start.
