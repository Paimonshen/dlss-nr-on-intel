#!/usr/bin/env python3
"""The QKV projection's epilogue against the projection plus the three passes it replaces.

Reference: the GEMM into float32, `cosine_publish` for Q (with its head's scale) and K,
`split_heads` with the E4M3 publish for V — the graph's own path. Fused: `gemm_qkv`. The
three targets must match byte for byte, and the guard values past their ends must
survive. Both GEMM paths the epilogue lives in are exercised: the register-tiled one
(C below 128, or rows that are not whole 64-row blocks) and the staged one.
"""
import numpy as np
import xmxres as X

GUARD = 64
# not an E4M3 value, so no correct output can ever equal it
FILL = np.float16(-11.1)


def reference(rt, a, weight, scale, proj, q, k, v, windows, tokens, channels, heads):
    rows = windows * tokens
    rt.gemm(a, weight, proj, rows, 3 * channels, channels)
    rt.cosine_publish(proj, q, windows * heads * tokens, tokens=tokens, heads=heads,
                      scale=scale, narrow=True, qkv_part=0)
    rt.cosine_publish(proj, k, windows * heads * tokens, tokens=tokens, heads=heads,
                      narrow=True, qkv_part=1)
    rt.split_heads(proj, v, windows, tokens, channels, heads, 2, epilogue=X.EPI_E4M3,
                   narrow=True)


def main():
    rt = X.Runtime()
    rng = np.random.default_rng(1729)
    cases, paths = 0, set()
    # (windows, tokens, channels): window blocks at every width, then the bottleneck's
    # single window of all tokens, tile-aligned but not a whole 64-row block
    shapes = ((6, 64, 32), (3, 64, 64), (2, 64, 128), (2, 64, 256), (1, 64, 512),
              (1, 240, 1024), (3, 16, 256), (1, 256, 128))
    for windows, tokens, channels in shapes:
        heads, rows = channels // 32, windows * tokens
        paths.add("staged" if channels >= 128 and rows % 64 == 0 else "tiled")
        buffers = []

        def alloc(count, dtype):
            buf = rt.buffer(count, dtype)
            buffers.append(buf)
            return buf
        try:
            a = alloc(rows * channels, np.float16)
            weight = alloc(channels * 3 * channels, np.float16)
            scale = alloc(heads, np.float32)
            proj = alloc(rows * 3 * channels, np.float32)
            targets = [alloc(rows * channels + GUARD, np.float16) for _ in range(6)]
            X.host_write(weight, rng.normal(0, .2, channels * 3 * channels).astype(np.float16))
            X.host_write(scale, rng.uniform(.5, 12, heads).astype(np.float32))
            for spread in (.01, .3, 30.0):
                values = rng.normal(0, spread, (rows, channels)).astype(np.float16)
                # a zero row takes the norm floor; at the widest spreads the squares
                # overflow half, the reciprocal is zero and so is the row
                values[1] = 0
                values[-1] = values[-1] * np.float16(4)
                X.host_write(a, values.reshape(-1))
                for mask in (0, 7):
                    rt.specialize(mask)
                    for buf in targets:
                        X.host_write(buf, np.full(rows * channels + GUARD, FILL, np.float16))
                    rt.begin()
                    reference(rt, a, weight, scale, proj, *targets[:3], windows, tokens,
                              channels, heads)
                    rt.gemm_qkv(a, weight, *targets[3:], scale, rows, channels, heads, tokens)
                    rt.submit()
                    for name, want, got in zip("QKV", targets[:3], targets[3:]):
                        want = X.host_view(want, np.float16)
                        got = X.host_view(got, np.float16)
                        if not np.array_equal(got.view(np.uint16), want.view(np.uint16)):
                            bad = np.flatnonzero(got.view(np.uint16) != want.view(np.uint16))
                            raise AssertionError(
                                f"{name} differs at {bad.size} of {got.size} "
                                f"(first {bad[0]}: {got[bad[0]]} against {want[bad[0]]}) "
                                f"windows={windows} tokens={tokens} C={channels} "
                                f"spread={spread} mask={mask}")
                        assert (got[rows * channels:] == FILL).all(), f"{name} guard overwritten"
                        # and every element was written, so matching is not two untouched fills
                        assert not (got[:rows * channels] == FILL).any(), f"{name} left unwritten"
                    cases += 1
            # guards on the host side of the call
            q, k, v = targets[3:]
            for args, why in (((a, weight, q, q, v, scale, rows, channels, heads, tokens), "alias"),
                              ((a, weight, a, k, v, scale, rows, channels, heads, tokens), "input"),
                              ((a, weight, q, k, v, scale, rows, channels, heads + 1, tokens), "heads"),
                              ((a, weight, q, k, v, scale, rows + 8, channels, heads, tokens), "windows")):
                try:
                    rt.gemm_qkv(*args)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"invalid QKV projection accepted: {why}")
        finally:
            for buf in buffers:
                buf.free()
    before = rt.graph_key()
    rt.qkv_epilogue = not rt.qkv_epilogue
    assert rt.graph_key() != before
    assert paths == {"tiled", "staged"}, paths
    print(f"QKV epilogue: {cases} bit-exact cases on both GEMM paths, guards and graph key "
          f"OK; staging={rt.staging}")


if __name__ == "__main__":
    main()
