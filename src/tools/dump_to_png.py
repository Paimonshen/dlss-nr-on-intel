#!/usr/bin/env python3
"""Convert a `--dump` directory of raw frames into PNGs.

`nr_daemon.py --dump` writes `NNN_in.raw` / `NNN_out.raw` plus one `dump.json`, because the dump
sits inside the daemon's per-frame timing and encoding a PNG there cost more than the network did
(143 ms an image through ImageMagick, against a ~70 ms frame). This is that conversion, paid
here instead — outside the timed region, where it affects nothing.

    python3 src/tools/dump_to_png.py <dump-dir> [--out <dir>] [--keep]

Reads `dump.json` for the geometry. Without `--out` the PNGs land beside the raws; `--keep`
leaves the raws in place, otherwise they are removed once their PNG exists.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ref"))

import image_io  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", help="the directory --dump wrote into")
    parser.add_argument("--out", default=None, help="where the PNGs go (default: in place)")
    parser.add_argument("--keep", action="store_true", help="do not delete the .raw files")
    args = parser.parse_args()

    source = pathlib.Path(args.dump)
    if not source.is_dir():
        raise SystemExit(f"not a directory: {source}")

    meta_path = source / "dump.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        width, height = int(meta["width"]), int(meta["height"])
        channels = int(meta.get("channels", 3))
    else:
        # No sidecar: the geometry has to come from somewhere, and guessing would silently
        # produce a scrambled picture, so say so rather than guess.
        raise SystemExit(f"no {meta_path}; the dump has no geometry to convert with")

    destination = pathlib.Path(args.out) if args.out else source
    destination.mkdir(parents=True, exist_ok=True)

    raws = sorted(source.glob("*.raw"))
    if not raws:
        print(f"no .raw files in {source}")
        return 0

    print(f"  {len(raws)} raw frame(s) at {width}x{height}x{channels} -> {destination}")
    done = 0
    for raw in raws:
        try:
            arr = image_io.load_raw(raw, width, height, channels)
        except ValueError as error:
            print(f"  {raw.name}: {error}", flush=True)
            continue
        png = destination / (raw.stem + ".png")
        image_io.save(arr, png)
        done += 1
        print(f"  {raw.name} -> {png.name}", flush=True)
        if not args.keep and destination == source:
            raw.unlink()

    print(f"  converted {done} of {len(raws)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
