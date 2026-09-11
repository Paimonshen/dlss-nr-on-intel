# Phase 52 — live neural rendering, running in a game

2026-09-11, 09:07. Dead or Alive 5 launched from Steam with `NR_LAYER_LIVE=1`, the daemon
holding the model, `nr-ctl` driving it. Not a photo mode and not a held frame: **every
present goes through the network, continuously.**

```
198 frames in 30 s — 6.60 fps
frame time 0.13-0.16 s, tight
change 0.02350 to 0.03144 across those frames — the scene is moving, every frame is its own
```

1280x720 output, render scale 0.35. This is the thing `phase47` built and neither tree had
ever run in a game.

## What it does to a frame, live

Frame 012 of the captured run, the character-select screen:

| region | luma | mean abs diff | relative fine texture |
| --- | --- | --- | --- |
| Kasumi's face | 98.5 -> 93.5 | 8.56 | **+6.3 %** |
| Mai's face | 98.2 -> 93.6 | 6.19 | +5.3 % |
| armour | 134.1 -> 123.4 | 12.74 | **+8.8 %** |
| background, logo | 70.2 -> 72.1 | 3.89 | -2.9 % |

Whole frame: `|d|` 7.23, brightness 93.0 -> 91.3, speckle x0.99. At 4x the skin gains pores
and the hair separates into finer strands; the background, which is flat art, is left
alone. The tone pull-back of `phase42` is there and smaller than at full scale.

## The finding: the render scale barely matters here

Swept live, without restarting the game or reloading the model — which is what `nr-ctl`
was built for:

| render scale | fps | frame |
| --- | --- | --- |
| 0.35 | 6.60 | 130 ms |
| 0.30 | 7.27 | 128 ms |
| 0.25 | 6.20 | 148 ms |
| 0.20 | 7.53 | 124 ms |

**The network shrinks threefold and the frame moves by a tenth.** That is `phase51`
confirmed from the other side: at a 1280x720 output the network is the minority of the
cost and the numpy either side of it — feature assembly, composition, the head upscale,
the codec — is measured at the *output* resolution and does not shrink with the scale.
The non-monotonic row at 0.25 is the network's alignment padding, familiar from the extent
curve.

**So the lever is the game's own resolution, not the render scale.** To go faster the game
must present a smaller frame. `phase47` measured 512x288 at 10.6 fps end to end; the route
to keeping that watchable on a 1080p panel is gamescope, which is now installed and which
the layer has been verified to load inside — it forces the swapchain to its nested size and
upscales with FSR 1.0. XeSS does not exist for Linux; checked twice.

## Two operational notes

`nr-photo` picks the last `Proton*` directory alphabetically, so when Steam downloaded
**Proton Hotfix** it silently displaced Experimental as the runtime. Worth pinning.

Launching the game from two places at once — `nr-photo --proton` and Steam — leaves both
fighting over one Wine prefix and neither starts; the second attempt then returns 53
because the first left `wineserver` alive. Check for a running game before launching one.
