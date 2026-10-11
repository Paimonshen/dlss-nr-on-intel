#!/usr/bin/env python3
"""`save_raw` must produce the same bytes `save` does, and be cheap enough to run per frame.

`save` is ImageMagick and `save_raw` is not, so the two are different code paths over the same
quantisation. If they ever disagree, a dump stops being a record of what the game was shown --
which is the only reason the dump exists. This pins that down, and the cost, in one command:

    python3 src/ref/test_dump_raw.py
"""
import pathlib
import statistics
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import image_io  # noqa: E402

BUDGET_MS = 5.0          # per frame, for both images, as the fix was specified


def main():
    failures = []
    temporary = pathlib.Path(tempfile.mkdtemp(prefix="test_dump_raw_"))
    rng = np.random.default_rng(7)

    cases = {
        "float ordinary":   rng.random((64, 96, 3)).astype(np.float32),
        "float extremes":   np.array([[[0.0, 0.5, 1.0], [1e-9, 0.999999, 0.0]]], np.float32),
        "float out of 0-1": (rng.random((16, 16, 3)).astype(np.float32) * 4.0) - 1.5,
        "float all zeros":  np.zeros((16, 16, 3), np.float32),
        "float all ones":   np.ones((16, 16, 3), np.float32),
    }

    # uint8 is not in the cases above on purpose: `save` is documented for float in [0,1], and
    # feeding it uint8 would be comparing against a path nobody uses. `save_raw`'s uint8
    # passthrough is checked separately below instead.

    for name, arr in cases.items():
        src = np.asarray(arr)
        quantised = (np.clip(src.astype(np.float32), 0, 1) * 255.0 + 0.5).astype(np.uint8)

        png = temporary / "a.png"
        image_io.save(src, png)
        # what the PNG holds, read back through ImageMagick
        got = np.rint(image_io.load(png) * 255.0).astype(np.uint8)

        raw = temporary / "a.raw"
        image_io.save_raw(src, raw)
        h, w = quantised.shape[:2]
        back = np.frombuffer(raw.read_bytes(), np.uint8).reshape(h, w, 3)

        same_png = bool((got == quantised).all())
        same_self = bool((back == quantised).all())
        mark = "ok  " if (same_png and same_self) else "FAIL"
        if not (same_png and same_self):
            failures.append(name)
        print(f"  [{mark}] {name:<20} png==quantised {same_png}  raw==quantised {same_self}")

    # uint8 passthrough: no quantisation should happen to it at all
    u8 = (rng.random((32, 48, 3)) * 255).astype(np.uint8)
    raw = temporary / "u8.raw"
    image_io.save_raw(u8, raw)
    back = np.frombuffer(raw.read_bytes(), np.uint8).reshape(32, 48, 3)
    ok = bool((back == u8).all())
    print(f"  [{'ok  ' if ok else 'FAIL'}] uint8 passthrough    unchanged {ok}")
    if not ok:
        failures.append("uint8 passthrough")

    # round trip
    big = rng.random((720, 1280, 3)).astype(np.float32)
    raw = temporary / "big.raw"
    image_io.save_raw(big, raw)
    back = image_io.load_raw(raw, 1280, 720, 3)
    rt = bool((np.rint(back * 255).astype(np.uint8)
               == (big * 255.0 + 0.5).astype(np.uint8)).all())
    print(f"  [{'ok  ' if rt else 'FAIL'}] load_raw round trip  {rt}")
    if not rt:
        failures.append("round trip")

    # cost, at the extent the daemon actually saw
    def timeit(fn, n=10):
        fn()
        s = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            s.append((time.perf_counter() - t0) * 1000)
        return statistics.median(s)

    old = timeit(lambda: image_io.save(big, temporary / "c.png"))
    new = timeit(lambda: image_io.save_raw(big, temporary / "c.raw"))
    per_frame = 2 * new
    print()
    print(f"  1280x720, one image : save {old:.2f} ms   save_raw {new:.2f} ms")
    print(f"  the daemon's two    : save {2 * old:.1f} ms/frame   "
          f"save_raw {per_frame:.2f} ms/frame   budget {BUDGET_MS} ms")
    if per_frame >= BUDGET_MS:
        failures.append(f"save_raw {per_frame:.2f} ms/frame is over the {BUDGET_MS} ms budget")

    print()
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("save_raw matches save byte for byte and is inside the per-frame budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
