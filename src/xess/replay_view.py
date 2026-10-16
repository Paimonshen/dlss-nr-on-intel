#!/usr/bin/env python3
"""Pictures from replay_nr_xess's frames: crops of two replays side by side, one PNG a frame,
and the difference between them, so what the proxy did to a frame can be looked at.

    python src/xess/replay_view.py <replay A> <replay B> <out> [--frames 10,20,30]
        [--crop x,y,w,h] [--gain 4]

A and B are folders of NNNN.ppm (say, NR and --off); each picture is A | B | |A - B| * gain.
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import zlib

import numpy as np


def read_ppm(path):
    data = path.read_bytes()
    parts = data.split(b"\n", 3)
    width, height = map(int, parts[1].split())
    return np.frombuffer(parts[3], np.uint8).reshape(height, width, 3)


def write_png(path, image):
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("out")
    parser.add_argument("--frames", default="10,20,30")
    parser.add_argument("--crop", default="0,0,0,0", help="x,y,w,h; 0 for the whole frame")
    parser.add_argument("--gain", type=float, default=4.0)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x, y, w, h = map(int, args.crop.split(","))
    for index in map(int, args.frames.split(",")):
        a = read_ppm(pathlib.Path(args.a) / f"{index:04d}.ppm")
        b = read_ppm(pathlib.Path(args.b) / f"{index:04d}.ppm")
        if w and h:
            a, b = a[y:y + h, x:x + w], b[y:y + h, x:x + w]
        gap = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * args.gain, 0, 255).astype(np.uint8)
        write_png(out / f"{index:04d}.png", np.concatenate([a, b, gap], 1))
        print(f"{index:04d}: mean |A - B| {np.abs(a.astype(np.int16) - b).mean():.2f}")


if __name__ == "__main__":
    main()
