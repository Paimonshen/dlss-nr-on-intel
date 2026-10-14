## `--dump` was inflating every measurement it appeared in, and now does not (2026-10-10)

**The trap.** `nr_daemon.py --dump` writes two images a frame from inside its own timed region
(`clock` at :626, the dump at :809-823, the `in X s` report at :851), and `image_io.save` spawns
ImageMagick. Measured on the B580 at 1280x720: **143 ms an image, 287 ms a frame, against a 70 ms
frame.** Every `in X s` and every fps taken with `--dump` on was wrong by 4x. It also made a dump
useless as a record of a session: the old path managed **101 frames in 30 s**, so a "run" produced
two files.

**The fix.** `image_io.save_raw` writes the same bytes with no subprocess and no header, and
`src/tools/dump_to_png.py` converts a dump directory afterwards, outside the timing.
`test_dump_raw.py` checks the pixels against a real PNG from the old path, byte for byte.

| 1280x720 | per image | per frame |
|---|---:|---:|
| `image_io.save` (ImageMagick) | 143 ms | 287 ms |
| `image_io.save_raw` (fixed) | 2.31 ms | **4.61 ms** |

**The one thing to know if you edit it: the buffers must be reused.** The allocating spelling was
5.4 ms an image -- not arithmetic and not bandwidth (13.8 MB at this machine's 380 GB/s is 0.04 ms)
but first-touch page faults on a fresh 11 MB float array. Reused: 2.31 ms.

**Three-group baseline, same scene, 30 s x 3 each, median frame time from the daemon's own log:**

| group | median frame | p95 | p99 | disk written |
|---|---:|---:|---:|---:|
| no `--dump` | **70 ms** | 90 | 150 | 0 |
| old `--dump` | **280 ms** | 440 | 550 | ~3 MB |
| new `--dump` | **70 ms** | 140 | 300 | **~1084 MB** |

**Reported median frame: 280 -> 70 ms, 4.00x.** The daemon log rounds to 10 ms, so the equal 70 ms medians do not prove zero overhead compared with no dump. The apparent 15 % fps
gap between groups 1 and 3 is a frame-count artefact -- the old arm only got through 101 frames,
the new one 371, so fps never compared like with like.

**What remains is disk, not encoding.** Raw does not compress: 2.7 MB x 2 = 5.4 MB a frame, 1.08 GB
in 30 s, which is what lifts p95/p99. The lever, if it is needed, is `--dump-every N`, not a faster
encoder. `NR_DUMP_SLOW=1` restores the old call so the two can be compared in one session; it is a
debugging switch, not a mode.

**So every performance number in this file needs two things stated with it**: whether `--dump` was
on, and whether the machine was otherwise idle (an earlier round of agent measurements ate 40 % of
GPU contention from a game running beside them).

---

## Rules for measuring performance (2026-10-10)

**This section is the contract for any performance number quoted from this repository.** It
exists because two rounds of measurements on this machine disagreed by 12-26 % and the cause
turned out to be the measurement, not the code. Read it before quoting a frame time, and quote
the five declarations with it.

### Every performance number must declare these five

| # | What | Why it changes the answer |
|---|---|---|
| 1 | **`--dump` on or off** | On, the dump was 287 ms a frame against a 70 ms frame -- a 4x error. Now 4.6-6.7 ms serial; no difference resolved by the 10 ms median log. |
| 2 | **Machine otherwise idle** | One round of agent measurements ate 40 % of GPU contention from a game running beside them. |
| 3 | **Resolution AND render scale, and the network extent the daemon reported** | Scale is the only lever that changes the frame rate; the same 1280x720 at 0.5 and 1.0 differ 3x. The daemon prints `network WxH scale S`. |
| 4 | **`NR_LAYER_ASYNC` state** | On, the cycle becomes `max(G, D)` instead of `G + D`. A different quantity, not a faster one. |
| 5 | **The scene: game, where in it, and how long** | Animated menus and static splashes give different distributions. Thirty seconds minimum, and see the warm-up rule below. |

### Baselines must carry the distribution, not just the rate

A baseline is **median frame time, p95, p99 and the sample count** -- taken from the daemon's own
`in X s`, not from `frames / wall_seconds`. A rate alone cannot be compared with another rate,
because it silently includes whatever startup the window happened to contain.

| quantity | 2026-10-10, Arc B580 |
|---|---|
| steady-state median frame, 1280x720, scale 1.0, sync, **in a game** | **70 ms** |
| the same frame with **only the daemon running** (no game contending) | **56 ms** |
| p95 / p99 | 90 ms / 110-150 ms |
| sample count behind it | 9 runs over 3 sessions, 175-479 frames each |

**The two figures differ by ~14 ms and both are real** -- state which one a number is. With a game
running, its previously submitted work is still executing on the GPU while the neural pass runs, so
the pass is slower than it is alone. The 70 ms is what a player sees; the 56 ms is what the daemon
costs in isolation, and it decomposes as **44 ms GPU (40 of it the graph) + 11.5 ms host + 0.7 ms
unattributed**, each segment measured (in the contributor's own per-pass profile, which is not
in this repository).

### Rules, each of which was broken at least once here

1. **Discard the first 25 frames, or warm up first.** Two separate startup costs, both large.
   The first frame is 0.75-1.09 s -- 11-16x the median (model load, first pipeline builds, the
   half-rounding probe), and the first five average 0.22-0.48 s. And the GPU itself ramps:
   measured 2026-10-10 over 50 consecutive frames, the graph's own cost (`split[1]`, what the
   daemon logs as `gpu N+M+Kms`) **decays monotonically from 72 ms to 40 ms** --

   ```
   gpu_run ms, frame order:  72.4 75.0 74.9 74.8 74.6 75.0 67.7 62.2 62.2 60.5
                             56.9 53.7 50.7 50.0 48.0 45.6 45.8 43.6 43.8 43.4
                             43.7 43.3 42.1 41.8 40.2 40.1 40.4 39.7 39.9 39.8  ... 39.5-40.2
   ```

   It settles after about 20 frames. **A 10-frame warm-up therefore reports `gpu_run` at 52 ms
   where the steady state is 40 -- a 12 ms error, 21 % of the whole frame.** Ten is not enough;
   twenty-five is the number this repository uses.
   The window-length point stands too: a 14 s window that keeps the startup reports 11.57 fps
   where the steady state is 12.50, and a 30 s window with an 8 s warm-up reports 14.57 for the
   same code -- **the difference was the window, not the program.**
   **Two tools in this repository used to default below it:** `src/bench/daemon_stages.py`
   (was 4) and `src/bench/live_rates.py` (was 3). Both now default to 25, so their numbers are
   steady-state without a flag. Anything older than 2026-10-10 that quotes those tools is a
   startup measurement.
2. **Compare processed frames per second over matched steady-state intervals.** Different
   frame counts over the same 30 s are a valid throughput difference; state warm-up, scene,
   settings and whether dumping is enabled. Also report frame-time percentiles, which answer
   a different question from throughput.
3. **Do not put measurement overhead and real performance in one table.** "The frame takes 70 ms"
   and "the dump adds 210 ms" are different claims about different things. State which each row is.
4. **Never quote a number without the five declarations above.** If a past measurement lacks them,
   it is not a baseline -- re-measure it.
5. **Say when a number is a projection.** A quantity derived from other measurements (the disk a
   `--dump-every 10` session will write, say) is not measured, and must be marked as derived.

### Two live traps on this platform

- **`image_io.save` spawns ImageMagick (143 ms a 1280x720 image).** Never call it from anything
  being timed. `image_io.save_raw` is the same pixels at 2.3-3.3 ms; convert afterwards with
  `src/tools/dump_to_png.py`.
- **`image_io.save_raw` reuses a module-level scratch buffer** and is therefore not reentrant.
  Fine for the daemon, which serves one connection at a time; not fine from a thread pool.

---
