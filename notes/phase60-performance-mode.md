# Phase 60 — the performance mode buys this workload nothing

> **Correction, the same afternoon.** Every number below was measured with Steam
> recompiling **Counter-Strike 2**'s shader cache in the background — two `fossilize_replay`
> workers at ~98 % of a core each, 6.5 GB of pipeline cache, running the whole time and not
> noticed until afterwards. What survives is the **frequency cap**: the GPU's own `act_freq`
> sitting at `rp0` is a direct reading of the GPU and does not depend on CPU load. What does
> not survive is every **number** — the fitted curve, the per-extent comparison and the two
> whole-frame times were taken on a machine with two cores gone. Background load would
> slow them, so a clean run may be faster still, but that is a guess and is not claimed.
> A Steam cache replay of this size is also what a Vulkan driver update triggers, which is a
> second candidate for the 9 % below. Re-measure on an idle machine before quoting any of it.

2026-09-16. The machine was switched to its performance mode — in Windows, through a
firmware setting Linux cannot reach, and in Linux as well. The question was whether the
graph got faster. **It did not, and the reason is measured, not assumed.**

## The state it was measured in

| | |
| --- | --- |
| mains | online, battery held at 80 % |
| Linux profile / EPP | `performance` / `performance` |
| RAPL package limits | PL1 **35 W**, PL2 **37 W** |
| CPU at idle | 3.86 GHz, against 1.2-1.4 on power-saver |
| GPU power profile | `base` |
| GPU hardware ceiling (`rp0`) | **1950 MHz** |

35 W is the high end for a 17 W Lunar Lake part and is presumably what the Windows-side
setting chose. That is an inference from the value: the limits before the change were never
recorded, so "raised" is not a measured delta.

## Measured

`src/bench/extent_curve.py` — the graph alone, best of five warm frames per extent — with
the GPU's `act_freq` sampled every 250 ms throughout:

| network extent | ms | fps | against 17 + 488 |
| --- | ---: | ---: | ---: |
| 448x256 | 64 | 15.7 | -12.3 % |
| 512x320 | 82 | 12.1 | -15.4 % |
| 640x384 | 126 | 7.9 | -8.0 % |
| 768x448 | 168 | 5.9 | -9.1 % |
| 1024x576 | 280 | 3.6 | -8.1 % |
| 1280x768 | 459 | 2.2 | -7.6 % |

Fitted: **11 ms + 457 ms per megapixel**, residual within 6 ms.

**The GPU spent 89 % of the run at 1950 MHz**, 9 % at 1900 and 2 % at 1450 — pinned at its
hardware ceiling. And the whole-frame stage profile of `phase57` came out at 215 and 245 ms
in two processes, against 214 on power-saver.

## Why that is "nothing" and not "9 %"

The curve is about 9 % under the old formula. It is not attributable to the power mode:

- **The graph runs on the GPU, and the GPU is frequency-capped, not power-capped.** It sat at
  `rp0` — the most the hardware will do — and earlier runs recorded the same 1950 MHz
  ceiling. A package power limit and a CPU energy preference cannot lift a frequency
  ceiling.
- **9 % is inside the band.** The same binary spreads about ten percent either side of its
  median *between processes*, from buffer placement (`HANDOFF`, "quote the band"). The two
  stage-profile processes today differed by 14 % from each other.
- **The system has been updated since `17 + 488` was measured**, and that formula's power
  profile was not recorded.

The host passes did not visibly move either. That fits too: after `phase57` they are small,
and the ones that remain move whole frames through memory, which is bounded by bandwidth —
70-91 GB/s (`phase20`) — not by the CPU's clock.

**A clean attribution would need a paired run in one session**, flipping only the profile.
The Linux half can be flipped; the firmware half cannot be reached from Linux at all, which
is the reason it was set in Windows. Not done, because the practical answer does not depend
on it: nothing about this workload is limited by the thing the mode changes.

## What that means for anyone tuning this

The lever is the **extent**, as it has been since `phase25`. The formula to plan with is now
`11 ms + 457 ms per megapixel` on this machine — call it the same curve as before within its
band — and the laptop's power mode is not a variable worth recording against it, except to
say the GPU was at its ceiling.
