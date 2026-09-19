#!/usr/bin/env python3
"""Paired FFN schedule benchmark: replayed full graph, no game or image codecs.

Report warm wall times, not per-dispatch estimates. The input, buffers and shaders
are shared; both graphs are captured before timing. Every result must match exactly.
"""
import argparse
import pathlib
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src/gpu'), str(ROOT / 'src/ref')]
import nr_frame
import xmx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'),
                        default=(576, 1024), help='input extent, before model padding')
    parser.add_argument('--pairs', type=int, default=8)
    args = parser.parse_args()
    if min(*args.size, args.pairs) <= 0:
        parser.error('size and pairs must be positive')
    color = np.random.default_rng(67).random((*args.size, 3), dtype=np.float32)
    features = nr_frame.make_features(color, **nr_frame.PROFILES['standard'])
    backend = nr_frame.ResidentBackend()
    try:
        rt = backend.runtime
        frame = backend.frame(*features.shape[:2])
        samples = {False: [], True: []}
        counts = {}
        expected = None
        for mode in (False, True):
            rt.batch_ffn = mode
            passes = []
            head = frame.run(features, execution='replay', submits=passes)
            counts[mode] = passes[0]
            if expected is None:
                expected = head
            else:
                np.testing.assert_array_equal(head, expected)
            # Warm both captured graphs; never time weight uploads or shader compilation.
            np.testing.assert_array_equal(frame.run(features, execution='replay'), expected)
        print(f'device: {xmx.device_name()}', flush=True)
        print(f'buffers: {xmx.memory_note()}', flush=True)
        print(f'network extent: {features.shape[1]}x{features.shape[0]}; '
              f'{args.pairs} alternating pairs', flush=True)
        for pair in range(args.pairs):
            for mode in ((False, True) if pair % 2 == 0 else (True, False)):
                rt.batch_ffn = mode
                started = time.perf_counter()
                head = frame.run(features, execution='replay')
                samples[mode].append(1000 * (time.perf_counter() - started))
                np.testing.assert_array_equal(head, expected)
        for mode in (False, True):
            values = samples[mode]
            print(f'NR_BATCH_FFN={int(mode)}: {counts[mode]} recorded passes, '
                  f'median {statistics.median(values):.3f} ms, '
                  f'range {min(values):.3f}..{max(values):.3f} ms')
        before, after = (statistics.median(samples[m]) for m in (False, True))
        print(f'pass reduction: {counts[False] - counts[True]}; '
              f'warm frame speedup: {before / after:.3f}x; '
              f'time saved: {before - after:.3f} ms')
        print('All heads bit-identical. Includes input write and head read; excludes '
              'feature assembly, composition, IPC and game time. This is not game FPS.')
    finally:
        backend.close()


if __name__ == '__main__':
    main()
