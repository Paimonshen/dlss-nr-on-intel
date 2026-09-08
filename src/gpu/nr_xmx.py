#!/usr/bin/env python3
"""
nr_xmx — run the recovered graph's GEMMs on the Xe2 XMX units.

`nr_model` funnels every matrix multiply through one hook, `nr_model.MATMUL`.
This module points that hook at `xmx.gemm`, which is FP16 x FP16 -> FP32 on the
cooperative-matrix path (config 1 of notes/hw-coopmat.md).

Two matmuls in the graph are *not* routed: the per-head score `Q @ K^T` and
context `P @ V`. Both are genuinely batched — a different B per head and per
window — and small enough (64x32x64) that a dispatch each costs more than numpy
takes. They are a few percent of the arithmetic.

Right-hand operands are the model's weights, so their FP16 conversion, power-of-two
rescale and tile padding are cached per tensor and done once.

    import nr_xmx; nr_xmx.install()
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ref"))

import xmx  # noqa: E402

# Below this many multiply-accumulates the ~0.4 ms fixed dispatch cost exceeds what
# numpy takes, even at the ~3 GFLOP/s of the reference BLAS this machine ships.
MIN_MACS = 1 << 20

_prepared: dict[tuple, tuple] = {}
STATS = {"gpu": 0, "gpu_macs": 0, "gpu_seconds": 0.0,
         "cpu": 0, "cpu_macs": 0, "cpu_seconds": 0.0}


def _operand(weight):
    """Cache the converted, rescaled, padded right-hand operand per tensor.

    Keyed by data pointer plus shape and strides, not by object identity: the
    branched feed-forward indexes `expansion_weight[head, branch, input]`, which
    builds a fresh view object on every access but always over the same bytes.
    The key doubles as the upload key, so a weight reused across chunks of one
    block is written into the device buffer once.
    """
    key = (weight.ctypes.data, weight.shape, weight.strides)
    hit = _prepared.get(key)
    if hit is None:
        hit = (weight, xmx.prepare_b(weight))
        _prepared[key] = hit
    return key, hit[1]


def matmul(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if b.ndim == 2 and a.ndim >= 2 and a.shape[-1] == b.shape[0]:
        rows = int(np.prod(a.shape[:-1], dtype=np.int64))
        inner, cols = b.shape
        if rows * inner * cols >= MIN_MACS:
            started = time.perf_counter()
            flat = np.ascontiguousarray(a.reshape(rows, inner))
            key, operand = _operand(b)
            out = xmx.gemm_mapped(flat, operand, b_key=key)
            STATS["gpu"] += 1
            STATS["gpu_macs"] += rows * inner * cols
            STATS["gpu_seconds"] += time.perf_counter() - started
            return out.reshape(*a.shape[:-1], cols)
    started = time.perf_counter()
    out = a @ b
    STATS["cpu"] += 1
    STATS["cpu_macs"] += int(np.prod(a.shape[:-1], dtype=np.int64)) * int(
        np.prod(b.shape[-2:], dtype=np.int64))
    STATS["cpu_seconds"] += time.perf_counter() - started
    return out


def install():
    import nr_model
    nr_model.MATMUL = matmul
    return xmx.device_name()


def uninstall():
    import nr_model
    nr_model.MATMUL = None


def reset_stats():
    for key in STATS:
        STATS[key] = 0 if isinstance(STATS[key], int) else 0.0


def report():
    lines = []
    for where in ("gpu", "cpu"):
        calls, macs, seconds = (STATS[where], STATS[f"{where}_macs"],
                                STATS[f"{where}_seconds"])
        rate = 2 * macs / seconds / 1e9 if seconds > 0 else 0.0
        lines.append(f"  {where.upper()}  {calls:6d} calls  {2 * macs / 1e9:8.1f} GFLOP  "
                     f"{seconds:7.2f} s  {rate:7.1f} GFLOP/s")
    return "\n".join(lines)
