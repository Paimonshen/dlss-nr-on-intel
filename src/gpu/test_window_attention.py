#!/usr/bin/env python3
"""Fused window attention against the three-pass path it replaces, on identical inputs.

ProjectsCodex's test, ported without its merged-output mode: that mode and
`fuse_attention_merge` are off by default in Codex's tree and were not carried over.
"""
import numpy as np
import xmxres as X


def main():
    rt = X.Runtime()
    rng = np.random.default_rng(42)
    cases = 0
    for batches, heads in ((1, 1), (6, 3), (32, 16)):
        buffers = []
        def alloc(count, dtype=np.float32):
            buf = rt.buffer(count, dtype)
            buffers.append(buf)
            return buf
        try:
            q, k, v = [alloc(batches*2048, np.float16) for _ in range(3)]
            bias = alloc(heads*4096)
            scores = alloc(batches*4096)
            probs = alloc(batches*4096, np.float16)
            expected, actual = [alloc(batches*2048+32) for _ in range(2)]
            for scale in (.01, .3, 5.0, None):
                for buf in (q, k, v):
                    buf.view(np.float16)[:] = rng.normal(0, scale or .3, batches*2048)
                    buf.view(np.float16)[:8] = [0, -0., 2**-20, -2**-20, .125, -.25, 1, -2]
                bias.view()[:] = rng.uniform(-7, 7, heads*4096)
                if scale is None:
                    # Zero scores expose every finite-half bias to the bit-affine
                    # transform, including clamp boundaries and half overflow in sums.
                    q.view(np.float16)[:] = k.view(np.float16)[:] = 0
                    finite = np.arange(65536, dtype=np.uint16).view(np.float16)
                    finite = finite[np.isfinite(finite)].astype(np.float32)
                    bias.view()[:] = np.resize(finite, heads*4096)
                for use_bias in (False, True):
                    selected = bias if use_bias else None
                    for mask in (0, 7):
                        rt.specialize(mask)
                        expected.view()[:] = actual.view()[:] = -11
                        rt.begin()
                        rt.gemm(q, k, scores, 64, 64, 32, batch=batches, transpose_b=True)
                        rt.softmax(scores, probs, batches*64, 64, narrow=True,
                                   bias=selected, heads=heads)
                        rt.gemm(probs, v, expected, 64, 32, 64, batch=batches)
                        rt.window_attention(q, k, v, actual, batches, heads, bias=selected)
                        rt.submit()
                        np.testing.assert_array_equal(actual.view().view(np.uint8),
                                                      expected.view().view(np.uint8))
                        cases += 1
            for target, b, h in ((q, batches, heads), (actual, 0, heads),
                                 (actual, batches, 0), (actual, 2**32, 1)):
                try:
                    rt.window_attention(q, k, v, target, b, h, bias=bias)
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid attention accepted')
            tiny = alloc(1)
            try:
                rt.window_attention(q, k, v, tiny, batches, heads, bias=bias)
            except ValueError:
                pass
            else:
                raise AssertionError('undersized output accepted')
        finally:
            for buf in buffers: buf.free()
    old = rt.graph_key()
    rt.fuse_window_attention = not rt.fuse_window_attention
    assert old != rt.graph_key()
    print(f'window attention: {cases} bit-exact cases, guards and graph key OK')


if __name__ == '__main__':
    main()
