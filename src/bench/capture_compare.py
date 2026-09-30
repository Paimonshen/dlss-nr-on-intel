#!/usr/bin/env python3
"""Where two machines' graphs part: every level's output captured on one frame, compared.

A head hash says two machines disagree, not where. This captures what the resident frame
keeps under `capture` — the stem, block 0, each encoder level, the bottleneck, each decoder
level — on frame_replay.py's synthetic frame, and saves it; the same command on the other
machine saves its own, and `--compare` walks both in graph order. For each point it reports
how many values are the same bits, how many differ only in the sign of a zero, and how far
the rest are apart, so the first point that parts — and by how much — is plain.

    python3 src/bench/capture_compare.py --size 320 320 --save linux-320.npz
    python3 src/bench/capture_compare.py --compare linux-320.npz windows-320.npz
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "ref"))


def synthetic(h, w):
    """frame_replay.py's frame, so the two tools' heads can be checked against each other."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return np.stack((0.5 + 0.3 * np.sin(xx / 9), 0.5 + 0.3 * np.cos(yy / 11),
                     0.5 + 0.3 * np.sin((xx + yy) / 13)), -1)


def save(args):
    import nr_frame
    h, w = args.size
    geometry = nr_frame.NetworkGeometry.vendor_aligned(w, h)
    features = nr_frame.make_features(synthetic(h, w), geometry=geometry,
                                      **nr_frame.PROFILES["standard"])
    backend = nr_frame.ResidentBackend()
    frame = backend.frame(*features.shape[:2])
    plain = frame.run(features, execution="block").copy()
    captured = {}
    head = frame.run(features, capture=captured)
    frame.close()
    if not np.array_equal(head, plain):
        raise SystemExit("the captured run's head is not the plain run's: a capture must not "
                         "change what the graph computes")
    order = list(captured) + ["head"]
    captured["head"] = head
    meta = {"size": [h, w], "network": list(features.shape[:2]), "order": order,
            "head_sha256": hashlib.sha256(head.tobytes()).hexdigest()}
    np.savez(args.save, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
             **{name: np.ascontiguousarray(captured[name], np.float32) for name in order})
    print(f"  {args.save}: network {meta['network'][0]}x{meta['network'][1]}, "
          f"head {meta['head_sha256'][:16]}")
    for name in order:
        array = captured[name]
        print(f"  {name:10} {str(array.shape):24} "
              f"{hashlib.sha256(np.ascontiguousarray(array, np.float32).tobytes()).hexdigest()[:16]}")


def compare(args):
    a, b = (np.load(path) for path in args.compare)
    meta_a, meta_b = (json.loads(bytes(x["meta"]).decode()) for x in (a, b))
    if meta_a["network"] != meta_b["network"]:
        raise SystemExit(f"different networks: {meta_a['network']} against {meta_b['network']}")
    print(f"  {args.compare[0]} against {args.compare[1]}, network "
          f"{meta_a['network'][0]}x{meta_a['network'][1]}")
    print(f"  {'point':10} {'same bits':>10} {'only ±0':>9} {'differ':>8} {'max |d|':>10} "
          f"{'mean |d|':>10} {'of mean |x|':>12}")
    first = None
    for name in meta_a["order"]:
        x, y = a[name], b[name]
        if x.shape != y.shape:
            print(f"  {name:10} shapes differ: {x.shape} against {y.shape}")
            continue
        same = x.view(np.uint32) == y.view(np.uint32)
        signed_zero = ~same & (x == 0) & (y == 0)
        differ = ~same & ~signed_zero & ~(np.isnan(x) & np.isnan(y))
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))[differ]
        scale = np.abs(x.astype(np.float64)).mean() or 1.0
        n = x.size
        print(f"  {name:10} {100 * same.mean():9.3f}% {int(signed_zero.sum()):9d} "
              f"{int(differ.sum()):8d} {d.max() if d.size else 0:10.3g} "
              f"{d.mean() if d.size else 0:10.3g} {(d.mean() / scale) if d.size else 0:12.3g}")
        if first is None and (differ.any() or signed_zero.any()):
            first = name
    print(f"\n  first point that parts: {first or 'none — the two graphs agree bit for bit'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, nargs=2, default=(320, 320), metavar=("H", "W"))
    parser.add_argument("--save", type=pathlib.Path, help="capture this machine's graph to .npz")
    parser.add_argument("--compare", type=pathlib.Path, nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if bool(args.save) == bool(args.compare):
        parser.error("give --save or --compare")
    save(args) if args.save else compare(args)


if __name__ == "__main__":
    main()
