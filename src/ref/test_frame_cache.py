#!/usr/bin/env python3
"""Check extent eviction without loading weights or allocating GPU memory."""
from types import SimpleNamespace
import nr_frame


class Frame:
    def __init__(self, runtime, weights, height, width):
        self.closed = False
        self.height, self.width = height, width

    def close(self):
        self.closed = True


def main():
    backend = object.__new__(nr_frame.ResidentBackend)
    backend._module = SimpleNamespace(ResidentFrame=Frame)
    backend.runtime = backend.weights = None
    backend._frames = {}
    backend.max_cached_frames = 2
    first = backend.frame(384, 384)
    second = backend.frame(768, 1280)
    assert backend.frame(384, 384) is first
    third = backend.frame(1088, 1920)
    assert second.closed and not first.closed and not third.closed
    assert len(backend._frames) == 2
    backend.close()
    assert first.closed and third.closed and not backend._frames
    backend.max_cached_frames = 1
    first = backend.frame(384, 384)
    assert backend.frame(384, 384) is first
    second = backend.frame(768, 1280)
    assert first.closed and len(backend._frames) == 1
    backend.close()
    assert second.closed
    print('frame cache: reuse, LRU eviction and release OK')


if __name__ == '__main__':
    main()
