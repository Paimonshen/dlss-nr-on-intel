# Phase 44 — the profiles are a measured trade-off, and one knob runs backwards

2026-09-10, 18:0x. A cutscene frame from DoA5 — Kasumi masked and in shadow beside a
second character in armour — put through the graph four times **on the same input**, so
this is a controlled comparison rather than four different scenes.

## Why the frame was worth it

The live frame showed the pass behaving in opposite directions in different places:

| region | luma | relative fine texture |
| --- | --- | --- |
| shoulder, blown-out skin | 162.4 -> 123.7 | **+52.0 %** |
| armour | 160.8 -> 151.4 | +20.0 % |
| second face | 119.0 -> 105.3 | +1.0 % |
| **Kasumi's face, in shadow** | 51.8 -> 43.5 | **-13.9 %** |

The loss is real and it is specific: her eyes. Peak in the left eye fell **204 -> 117**,
iris saturation **35.4 -> 16.8**, local contrast **107 -> 59**. Meanwhile the fabric of
her mask *gained* — contrast 22.3 -> 34.3, lifted out of near-black. So the pass lifts
shadow detail and crushes speculars, and on a dark face held together by two bright
irises that is a net loss.

## The four runs

Same 1920x1080 input, `src/ref/nr_frame.py --resident`:

| variant | eye peak | eye saturation | eye contrast | skin luma | skin rel. texture |
| --- | --- | --- | --- | --- | --- |
| input | 204 | 35.4 | 107.3 | 162.4 | — |
| `--profile cinematic` | **169** | **32.7** | **97.0** | 157.0 | +3.7 % |
| `--profile natural` | 144 | 26.6 | 86.6 | 151.7 | +8.9 % |
| `--profile standard` (default) | 117 | 16.8 | 59.0 | 123.7 | **+52.0 %** |
| `--colour-strength 1.5` | 87 | 8.8 | 40.7 | 104.3 | **+80.0 %** |

It is a clean monotone curve. **Everything the pass adds to skin texture it takes out of
speculars and colour**, and the profile chooses where on that curve to sit. There is no
setting that does both, at least not among these.

Visually the ranking is unmistakable at 5x on the eyes: cinematic keeps the amber irises
almost as the game drew them, standard mutes them, and `colour-strength 1.5` leaves them
nearly grey.

## The knob runs backwards from its name

`--colour-strength 1.5` does **not** restore colour. It is the most aggressive setting of
the five — iris saturation 16.8 -> 8.8, half again below the default. It scales the
strength of the pass's colour term, not the colour that survives. Worth stating plainly
because the name invites the opposite reading, and this project spent a measurement
finding out.

## What to use

Nothing here says one profile is correct; it says the choice is a real one with a
measured cost. For a photo mode on faces where the eyes carry the shot, **cinematic** is
the defensible default — it keeps 83 % of the highlight and 92 % of the iris saturation
while still adding texture. For blown-out skin and armour, **standard** is worth four
times the texture. The daemon takes `--profile`, so this is a launch-time choice today
and could be a per-frame one if it ever matters.

Frames: `work/doa5ab/`.
