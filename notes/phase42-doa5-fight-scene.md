# Phase 42 — the pass on a live fight, measured

2026-09-10, 17:16. Dead or Alive 5 Last Round running from Steam with the layer in its
launch options, daemon started separately with `--dump`. Ayane vs Raidou on the rooftop
stage, 1280x720, `VK_FORMAT_B8G8R8A8_UNORM`. One triggered frame, 2.37 s wall in the
daemon — that figure includes the socket round trip and writing two PNGs, not just the
~490 ms of graph.

This is the first in-game frame this project has measured rather than looked at.

## What the model did

| region | mean abs diff /255 | high-frequency energy |
| --- | --- | --- |
| whole frame | 5.37 (max 49) | +1.9 % |
| Ayane, face and hair | 9.37 | +6.4 % |
| Raidou, head and shoulder | 4.35 | **+36.1 %** |

**75.8 % of pixels moved by more than 2/255.** At full-frame scale the picture looks
almost unchanged, which is the honest impression and also why the crops matter: this is
a fighting-game wide shot, the characters occupy a small part of the frame, and the
model works on exactly the parts it was trained for. Zoomed 4x, Ayane's hair separates
into strands, the flower ornament recovers petal edges, and the eye and lips sharpen;
Raidou's mask picks up leather grain and his shoulder plate recovers form.

The +36 % on Raidou against +6.4 % on Ayane is the interesting split. His head is dark,
low-contrast and mostly texture; hers is already high-contrast hair against a bright
ornament. The model has the most to add where the input carries the least.

## The interface, and the case for the mask

`NR_LAYER_UI_MASK` was **not** in the game's environment for this run — the mask was off.
The result is the first direct measurement of what that costs:

```
health bar and name    mean |d| 5.98   max 38
DXVK overlay text      mean |d| 3.77   max 15
```

The health bar loses its dark outline, its edges bleed into the background, and the
"AYANE" lettering softens. The model treated the HUD as scene content, which is exactly
what it should do with no mask and exactly what the mask exists to prevent.

So the mask is still unproven on a live HUD — but the thing it is meant to prevent is now
measured, on a real frame, instead of argued. `notes/phase35-what-the-operator-does.md`
for the detector, `notes/phase39-layer-review.md` for the bug that stopped it working at
all until today.

Frames and crops: `work/doa5live/`.
