#!/usr/bin/env python3
"""A recording replay_nr_xess can read, made up rather than captured: a textured background
panning right, a square of another texture moving left over it, each frame jittered the way
XeSS asks and its vectors at the output's extent. For checking the replay itself, and for
seeing what the proxy does where one surface uncovers another, without a game.

    python src/xess/make_synthetic_recording.py <folder> [--frames 40]

The shape is Hogwarts Legacy's at XeSS's Performance setting: 835x522 to 1920x1200.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

WIDTH, HEIGHT = 835, 522
OUT_W, OUT_H = 1920, 1200
PAN = 2.0          # background, colour pixels a frame, rightward
SPEED = 5.0        # the square, leftward
SIDE = 120
FORMAT_RGBA16F = 10


def halton(index, base):
    result, fraction = 0.0, 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


def background(x, y):
    """A texture with detail at every scale: stripes, a checker and fine noise-like ripples."""
    checker = ((np.floor(x / 24) + np.floor(y / 24)) % 2) * 0.15
    stripes = 0.1 * np.sin(x * 0.35) * np.cos(y * 0.21)
    fine = 0.05 * np.sin(x * 1.7 + y * 2.3)
    base = 0.25 + checker + stripes + fine
    return np.stack([base * 1.1, base * 0.9, base * 0.7], -1)


def square(x, y):
    rings = 0.5 + 0.3 * np.sin(np.hypot(x - SIDE / 2, y - SIDE / 2) * 0.6)
    return np.stack([rings * 0.4, rings * 0.8, rings], -1)


def frame(t, jitter):
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    ux, uy = xx - jitter[0], yy - jitter[1]           # the pixel's sample, unjittered
    colour = background(ux - PAN * t, uy)
    left = 560 - SPEED * t
    inside = (ux >= left) & (ux < left + SIDE) & (uy >= 200) & (uy < 200 + SIDE)
    colour[inside] = square(ux[inside] - left, uy[inside] - 200)
    alpha = np.full((HEIGHT, WIDTH, 1), 1.0)
    return np.concatenate([colour, alpha], -1).astype(np.float16)


def vectors(t):
    """Current to previous, in output pixels, at the output's extent, nearest surface."""
    oy, ox = np.indices((OUT_H, OUT_W), dtype=np.float64)
    ux = (ox + 0.5) * WIDTH / OUT_W - 0.5
    uy = (oy + 0.5) * HEIGHT / OUT_H - 0.5
    left = 560 - SPEED * t
    inside = (ux >= left) & (ux < left + SIDE) & (uy >= 200) & (uy < 200 + SIDE)
    motion = np.zeros((OUT_H, OUT_W, 2))
    motion[..., 0] = np.where(inside, SPEED, -PAN) * OUT_W / WIDTH
    return motion.astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    folder = pathlib.Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.txt").write_text(
        f"colour {WIDTH} {HEIGHT} {FORMAT_RGBA16F} 8\nmotion {OUT_W} {OUT_H}\noutput {OUT_W} {OUT_H}\n"
        "quality 101\nflags 257\nvelocity_scale 1 1\njitter_scale 1 1\nwhite 1\n")
    lines = []
    for t in range(args.frames):
        jitter = (halton(t + 1, 2) - 0.5, halton(t + 1, 3) - 0.5)
        with open(folder / f"{t:04d}.bin", "wb") as out:
            out.write(frame(t, jitter).tobytes())
            out.write(vectors(t).tobytes())
        lines.append(f"{t} {t + 1} {jitter[0]:.6f} {jitter[1]:.6f} 1.000000 {1 if t == 0 else 0}\n")
    (folder / "frames.txt").write_text("".join(lines))
    print(f"{args.frames} frames in {folder}")


if __name__ == "__main__":
    main()
