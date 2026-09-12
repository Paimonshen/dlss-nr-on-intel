#!/usr/bin/env python3
"""Exercise request bounds, codecs and the native layer's disconnect behavior."""
import contextlib
import io
import pathlib
import socket
import struct
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import numpy as np
import nr_daemon as daemon

ROOT = pathlib.Path(__file__).resolve().parents[2]


class Request:
    def __init__(self, header, payload=b''):
        self.header, self.payload, self.reads, self.reply = header, payload, 0, None

    def recv(self, count):
        self.reads += 1
        if self.header:
            value, self.header = self.header, b''
            return value
        value, self.payload = self.payload[:count], self.payload[count:]
        return value

    def sendall(self, data):
        self.reply = data


def request_tests():
    args = SimpleNamespace(max_pixels=1024, profile='standard', intensity=1,
                           detail_strength=1, colour_strength=1, dump=None)
    for magic, width, height in ((0, 1, 1), (daemon.MAGIC, 0, 8),
                                 (daemon.MAGIC, 8, 0), (daemon.MAGIC, 1024, 1024)):
        request = Request(struct.pack('<4I', magic, width, height, 44))
        try:
            daemon.process_connection(request, None, args)
        except ValueError:
            assert request.reads == 1 and request.reply is None
        else:
            raise AssertionError('invalid header accepted')
    raw = bytes(range(64))
    request = Request(struct.pack('<4I', daemon.MAGIC, 4, 4, 999), raw)
    with contextlib.redirect_stdout(io.StringIO()):
        daemon.process_connection(request, None, args)
    assert request.reply == raw
    for vk_format in daemon.FORMATS:
        rgb = daemon.decode(raw, 4, 4, vk_format)
        assert daemon.encode(rgb, raw, vk_format) == raw
    try:
        daemon.receive(Request(b'', b'123'), 4)
    except EOFError:
        pass
    else:
        raise AssertionError('truncated body accepted')


def resample_tests():
    """`resample` runs per frame and was 55% of it in its first form (notes/phase47)."""
    rng = np.random.default_rng(0)
    image = rng.random((17, 23, 4)).astype(np.float32)

    def reference(source, size):
        """The obvious two-dimensional form the separable one replaced."""
        height, width = source.shape[:2]
        new_height, new_width = size
        ys = (np.arange(new_height, dtype=np.float32) + 0.5) * (height / new_height) - 0.5
        xs = (np.arange(new_width, dtype=np.float32) + 0.5) * (width / new_width) - 0.5
        y0 = np.clip(np.floor(ys), 0, height - 1).astype(np.int32)
        x0 = np.clip(np.floor(xs), 0, width - 1).astype(np.int32)
        y1, x1 = np.clip(y0 + 1, 0, height - 1), np.clip(x0 + 1, 0, width - 1)
        wy = np.clip(ys - y0, 0, 1)[:, None, None].astype(np.float32)
        wx = np.clip(xs - x0, 0, 1)[None, :, None].astype(np.float32)
        top = source[y0][:, x0] * (1 - wx) + source[y0][:, x1] * wx
        bottom = source[y1][:, x0] * (1 - wx) + source[y1][:, x1] * wx
        return (top * (1 - wy) + bottom * wy).astype(np.float32)

    for size in ((41, 37), (9, 11), (17, 40)):
        assert np.allclose(daemon.resample(image, size), reference(image, size), atol=1e-6), size
    assert daemon.resample(image, image.shape[:2]) is image, "the identity must not copy"
    # a whole factor takes the area mean, which point sampling would not
    block = np.arange(4 * 6 * 1, dtype=np.float32).reshape(4, 6, 1)
    assert np.allclose(daemon.resample(block, (2, 3)),
                       block.reshape(2, 2, 3, 2, 1).mean((1, 3))), "area mean"
    flat = np.full((8, 8, 3), 0.25, np.float32)
    assert np.allclose(daemon.resample(flat, (19, 5)), 0.25), "a constant must stay constant"
    print("resample: separable form matches the two-dimensional one, area-averages on a\n"
          "  whole factor, and leaves a constant alone")


def letterbox_tests():
    """`active_region` finds the bars a 4:3 window puts around a 16:9 render.

    DoA5's smallest window is 1024x768 and it letterboxes 16:9 inside it, so a quarter of
    every frame is black. Every stage downstream is measured at the output resolution
    (notes/phase51), so skipping the bars is worth a quarter of them.
    """
    rng = np.random.default_rng(0)
    boxed = np.zeros((768, 1024, 3), np.float32)
    boxed[96:672] = rng.random((576, 1024, 3)).astype(np.float32)
    assert daemon.active_region(boxed) == (96, 672, 0, 1024), "letterbox"
    pillared = np.zeros((768, 1024, 3), np.float32)
    pillared[:, 128:896] = 0.5
    assert daemon.active_region(pillared) == (0, 768, 128, 896), "pillarbox"
    plain = rng.random((720, 1280, 3)).astype(np.float32)
    assert daemon.active_region(plain) == (0, 720, 0, 1280), "a full frame keeps its edges"
    # a dark sky is black at the top and not at the bottom; a letterbox is symmetric
    sky = np.zeros((720, 1280, 3), np.float32); sky[200:] = 0.4
    assert daemon.active_region(sky) == (0, 720, 0, 1280), "asymmetric dark is not a bar"
    # a fade to black is bars all the way in from both sides and must not leave a sliver
    assert daemon.active_region(np.zeros((720, 1280, 3), np.float32)) == (0, 720, 0, 1280), \
        "a black frame is a frame, not a letterbox"
    # the bars are handed back untouched, which needs the codec to round-trip exactly
    raw = np.zeros((1, 256, 4), np.uint8)
    for channel in range(3):
        raw[0, :, channel] = np.arange(256)
    payload = raw.tobytes()
    assert daemon.encode(daemon.decode(payload, 256, 1, 44), payload, 44) == payload, \
        "encode(decode(v)) must be the identity for every byte"
    # Found once and afterwards only checked, because reducing the whole frame twice was
    # the most expensive host pass left once everything around it went native. A wrong
    # cache is a wrong crop, so it has to let go of every change that matters.
    cache = daemon.Letterbox()
    key = (1024, 768, 44)
    assert cache.region(boxed, key) == (96, 672, 0, 1024), "the cache finds them"
    moved = boxed.copy()
    moved[96:672] = 0.5                       # same bars, completely different content
    assert cache.region(moved, key) == (96, 672, 0, 1024), "and keeps them"
    lit = rng.random((768, 1024, 3)).astype(np.float32) * 0.8 + 0.1
    assert cache.region(lit, key) == (0, 768, 0, 1024), "bars gone, cache let go"
    wider = lit.copy()
    wider[:160] = 0.0
    wider[-160:] = 0.0
    assert cache.region(wider, key) == (160, 608, 0, 1024), "bars grew, cache let go"
    assert cache.region(boxed, (800, 600, 44)) == (96, 672, 0, 1024), "a new extent rescans"
    cache.key, cache.bounds = (1280, 720, 44), (96, 672, 0, 1024)
    assert cache.region(plain, (1280, 720, 44)) == (0, 720, 0, 1280), \
        "bounds that do not fit the frame are refused rather than used"
    print("letterbox: bars found, a dark frame and an asymmetric sky refused, the cache\n"
          "  lets go when they move or vanish, and the\n"
          "  codec round-trips so the bars can be left alone")


def native_exchange_tests():
    with tempfile.TemporaryDirectory(prefix='nr-exchange-') as temporary:
        for mode in ('echo', 'reject', 'partial', 'masked'):
            path = str(pathlib.Path(temporary) / mode)
            errors = []
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(path)
                server.listen(1)
                server.settimeout(5)

                def serve():
                    try:
                        connection, _ = server.accept()
                        with connection:
                            connection.settimeout(5)
                            header = daemon.receive(connection, 16)
                            magic, width, height, _ = struct.unpack('<4I', header)
                            if mode == 'reject':
                                return  # client may still be writing a large payload
                            body = daemon.receive(connection, width * height * 4)
                            if mode == 'masked':
                                # the request carries a mask plane the answer does not,
                                # which is exactly what the daemon does
                                assert magic == daemon.MAGIC_MASKED, hex(magic)
                                daemon.receive(connection, width * height)
                                connection.sendall(body)
                                return
                            connection.sendall(body if mode == 'echo' else body[:17])
                    except Exception as error:
                        errors.append(error)

                thread = threading.Thread(target=serve)
                thread.start()
                result = subprocess.run([str(ROOT / 'work' / 'test_exchange'), path,
                                         mode if mode in ('echo', 'masked') else 'reject'],
                                        capture_output=True, timeout=10)
                thread.join(timeout=6)
                assert not thread.is_alive() and not errors, errors
                assert result.returncode == 0, (mode, result.returncode, result.stderr)
    print('daemon/layer: early bounds, HDR/alpha round trips, truncation, a masked\n  request whose answer is smaller, and a SIGPIPE-safe exchange OK')


if __name__ == '__main__':
    request_tests()
    resample_tests()
    letterbox_tests()
    native_exchange_tests()
