#!/usr/bin/env python3
"""A 32-channel window block's attention half in one pass, against the three it replaces.

`window_block` must write what the window-gathered QKV projection with its epilogue,
merged window attention and the window residual write, byte for byte: whole and cropped
windows, both window origins, a float32 and a half image, and the three publishes the
graph uses — a float32 output, a half one, and E4M3 published as half. Nothing outside the
image may be written.
"""
import numpy as np
import xmxres as X

GUARD = 64
FILL16 = np.array([0x7E5A], np.uint16).view(np.float16)[0]
FILL32 = np.array([0x7FC0DEAD], np.uint32).view(np.float32)[0]


def fill(dtype):
    return FILL16 if np.dtype(dtype) == np.float16 else FILL32


def filled(values, dtype):
    bits = np.uint16 if np.dtype(dtype) == np.float16 else np.uint32
    return np.asarray(values).view(bits) == np.asarray(fill(dtype)).view(bits)


def main():
    rt = X.Runtime()
    rng = np.random.default_rng(5150)
    cases = 0
    for height, width in ((8, 8), (16, 24), (20, 36), (13, 21), (40, 56)):
        for origin in ((0, 0), (-4, -4)):
            ph, pw, _ = rt.window_extent(height, width, origin)
            windows, rows, pixels = (ph // 8) * (pw // 8), ph * pw, height * width
            buffers = []

            def alloc(n, dtype):
                b = rt.buffer(n, dtype)
                buffers.append(b)
                return b
            try:
                image32, image16 = alloc(pixels * 32, np.float32), alloc(pixels * 32, np.float16)
                qkv, projection = alloc(32 * 96, np.float16), alloc(32 * 32, np.float16)
                bias, cosine, scale = alloc(64 * 64, np.float32), alloc(32, np.float32), alloc(1, np.float32)
                q, k, v = (alloc(rows * 32, np.float16) for _ in range(3))
                attended = alloc(rows * 32, np.float16)
                for spread in (0.3, 2.0):
                    values = rng.normal(0, spread, pixels * 32)
                    X.host_write(image32, values.astype(np.float32))
                    X.host_write(image16, values.astype(np.float16))
                    X.host_write(qkv, rng.normal(0, 0.25, 32 * 96).astype(np.float16))
                    X.host_write(projection, rng.normal(0, 0.25, 32 * 32).astype(np.float16))
                    X.host_write(bias, rng.normal(0, 1.0, 64 * 64).astype(np.float32))
                    X.host_write(cosine, rng.uniform(-1.5, 1.5, 32).astype(np.float32))
                    X.host_write(scale, rng.uniform(2.0, 20.0, 1).astype(np.float32))
                    for image, image_half in ((image32, False), (image16, True)):
                        for epilogue, narrow in ((0, False), (0, True), (X.EPI_E4M3, True)):
                            dtype = np.float16 if narrow else np.float32
                            for mask in (0, 7):
                                rt.specialize(mask)
                                want = alloc(pixels * 32 + GUARD, dtype)
                                got = alloc(pixels * 32 + GUARD, dtype)
                                for b in (want, got):
                                    X.host_write(b, np.full(pixels * 32 + GUARD, fill(dtype), dtype))
                                rt.begin()
                                rt.gemm_qkv(image, qkv, q, k, v, scale, rows, 32, 1, 64,
                                            window=(height, width, origin), image_half=image_half)
                                rt.window_attention(q, k, v, attended, windows, 1, bias=bias,
                                                    merged=True)
                                rt.gemm_residual(attended, projection, image, cosine, want, rows,
                                                 32, 32, reverse=(height, width, 8, origin),
                                                 epilogue=epilogue, narrow=narrow,
                                                 skip_half=image_half)
                                rt.window_block(image, qkv, projection, got, bias, cosine, scale,
                                                height, width, origin, epilogue=epilogue,
                                                narrow=narrow, image_half=image_half)
                                rt.submit()
                                name = (f"{height}x{width} origin {origin} spread {spread} "
                                        f"{'half' if image_half else 'float'} image, epilogue "
                                        f"{epilogue}{' narrow' if narrow else ''} mask {mask}")
                                g, w_ = X.host_view(got, dtype), X.host_view(want, dtype)
                                n = pixels * 32
                                if not np.array_equal(g[:n].view(np.uint8), w_[:n].view(np.uint8)):
                                    bad = np.flatnonzero(g[:n].view(np.uint16 if narrow else np.uint32)
                                                         != w_[:n].view(np.uint16 if narrow else np.uint32))
                                    raise AssertionError(f"{name}: {bad.size} of {n} differ, first "
                                                         f"at {bad[0]}: {g[bad[0]]} vs {w_[bad[0]]}")
                                assert filled(g[n:], dtype).all(), f"{name}: written past the image"
                                assert not filled(g[:n], dtype).any(), f"{name}: left unwritten"
                                cases += 1
            finally:
                for b in buffers:
                    b.free()
    before = rt.graph_key()
    rt.fuse_window_block = not rt.fuse_window_block
    assert rt.graph_key() != before, "the window block's switch must change the graph key"
    print(f"window block: {cases} cases bit-identical to the three passes it replaces, "
          f"graph key OK; staging={rt.staging}")


if __name__ == "__main__":
    main()
