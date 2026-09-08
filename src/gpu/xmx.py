#!/usr/bin/env python3
"""
xmx — host-side interface to the Xe2 cooperative-matrix GEMM.

Backed by `work/libxmx.so`, a resident Vulkan context: the instance, device,
pipeline and buffers are created once and reused, so a call costs a memcpy, a submit
and a fence wait rather than ~80 ms of setup.

Two things this layer must do that the kernel does not:

  1. **Rescale both operands by a power of two.** XMX flushes subnormal FP16 to zero
     and 27% of this model is FP16-subnormal (notes/phase4-subnormal-flush.md).
     A power-of-two scale is exact, so this is lossless.
  2. **Pad to the tile shape.** The only float configuration is M=8 N=16 K=16.
"""
import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FP16_MAX = 65504.0
TM, TN, TK = 8, 16, 16

_lib = None


def _load(spv="gemm_coopmat.spv"):
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(str(ROOT / "work" / "libxmx.so"))
    lib.xmx_init.argtypes = [ctypes.c_char_p]
    lib.xmx_init.restype = ctypes.c_int
    lib.xmx_gemm.argtypes = [ctypes.c_uint] * 3 + [ctypes.c_void_p] * 3 + [ctypes.c_uint]
    lib.xmx_gemm.restype = ctypes.c_int
    lib.xmx_error.restype = ctypes.c_char_p
    lib.xmx_device.restype = ctypes.c_char_p
    if lib.xmx_init(str(ROOT / "work" / spv).encode()) != 0:
        raise RuntimeError("xmx_init: " + lib.xmx_error().decode())
    _lib = lib
    return lib


def device_name():
    return _load().xmx_device().decode()


def _shift(x):
    mx = float(np.abs(np.asarray(x, dtype=np.float32)).max())
    return int(np.floor(np.log2(FP16_MAX / mx))) if mx > 0 else 0


def _pad(a, m, n):
    out = np.zeros((m, n), dtype=np.float16)
    out[:a.shape[0], :a.shape[1]] = a
    return out


def gemm(A, B, rescale=True, iters=1):
    """C = A @ B, FP16 operands on XMX, FP32 accumulation, returned as float32."""
    lib = _load()
    A = np.asarray(A, dtype=np.float16)
    B = np.asarray(B, dtype=np.float16)
    M0, K0 = A.shape
    K1, N0 = B.shape
    assert K0 == K1, "inner dimensions disagree: %d vs %d" % (K0, K1)

    ka = _shift(A) if rescale else 0
    kb = _shift(B) if rescale else 0
    As = (A.astype(np.float32) * 2.0 ** ka).astype(np.float16)
    Bs = (B.astype(np.float32) * 2.0 ** kb).astype(np.float16)

    M = -(-M0 // TM) * TM
    N = -(-N0 // TN) * TN
    K = -(-K0 // TK) * TK
    Ap = np.ascontiguousarray(_pad(As, M, K))
    Bp = np.ascontiguousarray(_pad(Bs, K, N))
    C = np.empty((M, N), dtype=np.float32)
    if lib.xmx_gemm(M, N, K, Ap.ctypes.data, Bp.ctypes.data, C.ctypes.data, iters) != 0:
        raise RuntimeError("xmx_gemm: " + lib.xmx_error().decode())
    return C[:M0, :N0] / (2.0 ** (ka + kb))
