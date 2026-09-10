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


## A close-up, and a correction to what "more detail" means

17:31, same session, a story-mode still of Hitomi at **1920x1080**, 5.05 s in the daemon.
The largest face this project has put through the pass. First impression from the two
full frames was "pores and freckles appeared". The measurement says that is half right,
and the half it gets wrong matters.

```
face, mean colour   in  R 184.9  G 125.3  B 98.0   luma 136.1   saturation 87.3
                    out R 120.8  G  80.4  B 61.0   luma  87.4   saturation 60.0
```

Mean absolute difference over the face is **48.8/255**, and compensating nothing but the
mean brightness shift drops it to **17.6**. So most of that number is tone, not texture.
Raw fine-texture energy (deviation from a local 3x3 mean) *falls* on the face, by 16 %.

Normalised for level, it rises everywhere:

| region | luma | relative fine texture |
| --- | --- | --- |
| face | 136.1 -> 87.4 | **+30.6 %** |
| hair | 55.0 -> 37.0 | +8.5 % |
| jacket | 60.1 -> 52.8 | +8.6 % |
| background | 41.6 -> 46.4 | +8.9 % |

**The whole frame's brightness is unchanged (-1 %).** The pass is not darkening the
picture; it is specifically pulling back a blown-out skin highlight — the game's shader
puts the face at R=185, near clipping and waxy — and putting structure into the range it
frees. That is a defensible thing for a detail re-render trained on real skin to do, and
whether it is *wanted* is an artistic call rather than a correctness one. The knobs exist
already: `--profile`, `--intensity`, `--detail-strength`, `--colour-strength`, all at
their defaults for this frame.

The lesson for measuring this model: **raw high-frequency energy is the wrong statistic
when the pass also moves the level.** Normalise, or the tone change masquerades as lost
detail. The earlier fight-scene figures in this note are unaffected — the change there
was 5.4/255 with no comparable level shift — but they are the exception, not the rule.

## The mask refused, correctly

Relaunched with `NR_LAYER_UI_MASK=1`, verified present in the game's own `/proc` environ.
The layer said:

```
[nr_layer] ui mask on: the first present after the trigger is kept to find what held still
[nr_layer] 90% of the frame held still; ui mask refused
[nr_layer] processed 1920x1080
```

The frame was a story-mode still. 90 % of it had not moved between the two presents, and
the detector cannot separate an interface over a still scene from a still scene, so it
declined rather than guessed — exactly the designed behaviour
(`notes/phase35-what-the-operator-does.md`). **The mask is still unproven on a live HUD**,
and proving it needs a frame captured mid-round with the characters actually moving.
