#!/usr/bin/env python3
"""The fused feed-forward against the two GEMMs it replaces, on identical inputs.

Reference: the expand into a half hidden buffer with the gate and E4M3 publish, then the
projection with the residual in its epilogue — the graph's own path. Fused: `ffn_fused`.
The outputs must match byte for byte, every element must have been written, and the guards
past the end must survive. Row counts that are and are not whole 64-row blocks take the
projection down both of its GEMM paths.
"""
import numpy as np
import xmxres as X

GUARD = 64
FILL16 = np.array([0x7E5A], np.uint16).view(np.float16)[0]
FILL32 = np.array([0x7FC0DEAD], np.uint32).view(np.float32)[0]


def filled(values, dtype):
    bits = np.uint16 if np.dtype(dtype) == np.float16 else np.uint32
    fill = FILL16 if np.dtype(dtype) == np.float16 else FILL32
    return np.asarray(values).view(bits) == np.asarray(fill).view(bits)


def main():
    rt = X.Runtime()
    rng = np.random.default_rng(90210)
    cases, channels = 0, 32
    for rows, hidden in ((16, 128), (48, 128), (64, 128), (1024, 128), (1024, 64), (80, 32)):
        buffers = []

        def alloc(n, dtype):
            b = rt.buffer(n, dtype)
            buffers.append(b)
            return b
        try:
            a = alloc(rows * channels, np.float16)
            expand = alloc(channels * hidden, np.float16)
            projection = alloc(hidden * channels, np.float16)
            cosine = alloc(channels, np.float32)
            hidden16 = alloc(rows * hidden, np.float16)
            skip32 = alloc(rows * channels, np.float32)
            skip16 = alloc(rows * channels, np.float16)
            X.host_write(expand, rng.normal(0, 0.25, channels * hidden).astype(np.float16))
            X.host_write(projection, rng.normal(0, 0.1, hidden * channels).astype(np.float16))
            X.host_write(cosine, rng.uniform(-1.5, 1.5, channels).astype(np.float32))
            for spread in (0.05, 1.0, 12.0):
                values = rng.normal(0, spread, rows * channels).astype(np.float16)
                values[:3] = [0, -0.0, 2 ** -24]
                X.host_write(a, values)
                skip = rng.normal(0, spread, rows * channels)
                X.host_write(skip32, skip.astype(np.float32))
                X.host_write(skip16, skip.astype(np.float16))
                for skip_half in (False, True):
                    for epilogue, narrow in ((0, False), (X.EPI_E4M3, True)):
                        dtype = np.float16 if narrow else np.float32
                        want = alloc(rows * channels + GUARD, dtype)
                        got = alloc(rows * channels + GUARD, dtype)
                        source = skip16 if skip_half else skip32
                        for mask in (0, 7):
                            rt.specialize(mask)
                            fill = FILL16 if narrow else FILL32
                            for buf in (want, got):
                                X.host_write(buf, np.full(rows * channels + GUARD, fill, dtype))
                            rt.begin()
                            rt.gemm(a, expand, hidden16, rows, hidden, channels,
                                    epilogue=X.EPI_GATE_E4M3, narrow=True)
                            rt.gemm_residual(hidden16, projection, source, cosine, want, rows,
                                             channels, hidden, epilogue=epilogue, narrow=narrow,
                                             skip_half=skip_half)
                            rt.ffn_fused(a, expand, projection, got, source, cosine, rows,
                                         channels, hidden, epilogue=epilogue, narrow=narrow,
                                         skip_half=skip_half)
                            rt.submit()
                            w, g = X.host_view(want, dtype), X.host_view(got, dtype)
                            if not np.array_equal(w.view(np.uint8), g.view(np.uint8)):
                                bad = np.flatnonzero(w.view(np.uint8) != g.view(np.uint8))
                                raise AssertionError(
                                    f"{rows}x{hidden} spread {spread} skip_half {skip_half} "
                                    f"epilogue {epilogue} mask {mask}: {bad.size} bytes differ")
                            assert filled(g[rows * channels:], dtype).all(), "guard overwritten"
                            assert not filled(g[:rows * channels], dtype).any(), "left unwritten"
                            cases += 1
            for args in ((a, expand, projection, a, skip32, cosine, rows, channels, hidden),
                         (a, expand, projection, skip16, skip32, cosine, rows + 8, channels, hidden),
                         (a, expand, projection, skip16, skip32, cosine, rows, 64, hidden)):
                try:
                    rt.ffn_fused(*args)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid fused feed-forward accepted")
        finally:
            for b in buffers:
                b.free()
    before = rt.graph_key()
    rt.fuse_ffn = not rt.fuse_ffn
    assert rt.graph_key() != before
    print(f"fused feed-forward: {cases} bit-exact cases, both projection paths, guards and "
          f"graph key OK; staging={rt.staging}")


if __name__ == "__main__":
    main()
