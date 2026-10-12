#!/usr/bin/env python3
"""Convert a `--dump` directory of raw frames into PNGs.

`nr_daemon.py --dump` writes `NNN_in.raw` / `NNN_out.raw` with per-file geometry, because the dump
sits inside the daemon's per-frame timing and encoding a PNG there cost more than the network did
(143 ms an image through ImageMagick, against a ~70 ms frame). This is that conversion, paid
here instead — outside the timed region, where it affects nothing.

    python3 src/tools/dump_to_png.py <dump-dir> [--out <dir>] [--keep]

Reads each raw's `.raw.json` geometry, or an older dump's `dump.json`. Without `--out` the
PNGs land beside the raws; `--keep` leaves the raws in place. New raws are removed only after
checking the PNG's dimensions and pixels. Legacy raws are retained because one directory's
geometry cannot establish the original dimensions of every frame after a window resize.
"""
import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

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

    destination = pathlib.Path(args.out) if args.out else source
    destination.mkdir(parents=True, exist_ok=True)

    raws = sorted(source.glob("*.raw"))
    if not raws:
        print(f"no .raw files in {source}")
        return 0

    print(f"  {len(raws)} raw image(s) -> {destination}")
    done = 0
    failed = 0
    for raw in raws:
        temporary = None
        try:
            arr = image_io.load_raw(raw)
            png = destination / (raw.stem + ".png")
            # Keep an existing PNG until the replacement has been encoded and checked.
            with tempfile.NamedTemporaryFile(dir=destination, suffix=".png", delete=False) as handle:
                temporary = pathlib.Path(handle.name)
            image_io.save(arr, temporary)
            back = image_io.load(temporary)
            if back.shape != arr.shape or not np.array_equal(
                    np.rint(back * 255).astype(np.uint8), np.rint(arr * 255).astype(np.uint8)):
                raise ValueError("PNG verification failed; raw retained")
            os.replace(temporary, png)
            done += 1
            print(f"  {raw.name} -> {png.name}", flush=True)
            sidecar = image_io.raw_metadata_path(raw)
            if not args.keep and destination.resolve() == source.resolve() and sidecar.is_file():
                raw.unlink()
                sidecar.unlink()
            elif not sidecar.is_file():
                print("    legacy geometry: raw retained", flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            failed += 1
            print(f"  {raw.name}: {error}", flush=True)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    print(f"  converted {done} of {len(raws)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
