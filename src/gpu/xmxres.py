#!/usr/bin/env python3
"""
xmxres — the device-resident runtime: buffers that persist, dispatches that batch.

`xmx.py` runs one GEMM per submit and hands the result back to the host, which is
why the XMX path does not beat a good CPU BLAS: a 720p frame moves 16.46 GB across
the boundary and 62x the kernel's own time goes into the trip
(`notes/phase11-what-is-left.md`, `notes/phase13-torch-and-blas.md`).

Here the operands are device buffers addressed by pointer, the elementwise passes
run on the GPU too, and a whole chain records into one command buffer with a single
fence at the end. Because the APU's memory is shared and HOST_CACHED, a buffer's
contents are also a numpy array — `Buffer.view()` — so feeding an input or reading
an output is an address, not a transfer.

    rt = Runtime()
    a = rt.buffer_from(activations)          # float32, on device
    rt.begin()
    rt.to_half(a, a16, n)
    rt.gemm(a16, w16, out, M, N, K)
    rt.submit()
"""
from __future__ import annotations

import ctypes
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
TM, TN, TK = 8, 16, 16

(E4M3, GATE, HALF, TO_HALF, SCALE, RESIDUAL, FROM_HALF, PARTITION, REVERSE, ADD_BIAS,
 SPLIT_HEADS, MERGE_HEADS, POOL2, UPSAMPLE2, SCALE_CHANNEL, ADD, PAD_END,
 GATE_E4M3_HALF, E4M3_HALF, GATE_HALF) = range(20)
COSINE_PUBLISH, SOFTMAX = 0, 1

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    import xmx
    lib = xmx._load()                      # shares the instance, device and queue
    for name, args in (
            ("xmx_res_init", [ctypes.c_char_p] * 4),
            ("xmx_buf_create", [ctypes.c_ulonglong]),
            ("xmx_buf_destroy", [ctypes.c_int]),
            ("xmx_begin", []),
            ("xmx_submit", []),
            ("xmx_rec_gemm", [ctypes.c_int] * 3 + [ctypes.c_uint] * 14),
            ("xmx_rec_unary", [ctypes.c_uint] + [ctypes.c_int] * 4
             + [ctypes.c_uint, ctypes.c_uint, ctypes.c_float] + [ctypes.c_uint] * 5),
            ("xmx_rec_row", [ctypes.c_uint] + [ctypes.c_int] * 3 + [ctypes.c_uint] * 5
             + [ctypes.c_float]),
            ("xmx_rec_history", [ctypes.c_int] * 3 + [ctypes.c_uint] * 5)):
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = ctypes.c_int
    lib.xmx_buf_ptr.argtypes = [ctypes.c_int]
    lib.xmx_buf_ptr.restype = ctypes.c_void_p
    lib.xmx_buf_bytes.argtypes = [ctypes.c_int]
    lib.xmx_buf_bytes.restype = ctypes.c_ulonglong
    if lib.xmx_res_init(str(ROOT / "work" / "gemm_resident.spv").encode(),
                        str(ROOT / "work" / "resident.spv").encode(),
                        str(ROOT / "work" / "attention.spv").encode(),
                        str(ROOT / "work" / "history.spv").encode()) != 0:
        raise RuntimeError("xmx_res_init: " + lib.xmx_error().decode())
    _lib = lib
    return lib


def align(value, multiple):
    return -(-value // multiple) * multiple


class Buffer:
    """A device buffer that is also addressable as a numpy array."""

    __slots__ = ("id", "nbytes", "_lib")

    def __init__(self, nbytes):
        self._lib = _load()
        self.id = self._lib.xmx_buf_create(int(nbytes))
        if self.id < 0:
            raise RuntimeError("xmx_buf_create: " + self._lib.xmx_error().decode())
        self.nbytes = int(nbytes)

    def view(self, dtype=np.float32, shape=None):
        pointer = self._lib.xmx_buf_ptr(self.id)
        count = self.nbytes // np.dtype(dtype).itemsize
        array = np.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8)), shape=(self.nbytes,))
        array = array.view(dtype)[:count]
        return array if shape is None else array[:int(np.prod(shape))].reshape(shape)

    def zero(self):
        self.view(np.uint8)[:] = 0
        return self

    def free(self):
        if self.id >= 0:
            self._lib.xmx_buf_destroy(self.id)
            self.id = -1

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class Runtime:
    """Records a chain of GPU passes and submits it once."""

    def __init__(self):
        self.lib = _load()
        self.recorded = 0

    # -- allocation ----------------------------------------------------

    def buffer(self, count, dtype=np.float32):
        return Buffer(int(count) * np.dtype(dtype).itemsize)

    def buffer_from(self, array, dtype=np.float32, pad=0):
        """A device buffer holding `array`, optionally padded with zeros at the end."""
        array = np.ascontiguousarray(array, dtype=dtype)
        buffer = Buffer((array.size + int(pad)) * array.dtype.itemsize)
        flat = buffer.view(dtype)
        flat[:array.size] = array.reshape(-1)
        if pad:
            flat[array.size:] = 0
        return buffer

    # -- recording -----------------------------------------------------

    def begin(self):
        if self.lib.xmx_begin() != 0:
            raise RuntimeError("xmx_begin: " + self.lib.xmx_error().decode())
        self.recorded = 0
        return self

    def gemm(self, a, b, c, rows, cols, inner, *, batch=1, strides=None, transpose_b=False,
             leading=None, offsets=(0, 0, 0)):
        """C = A @ B for tile-aligned extents; A and B are float16, C float32.

        `strides` are element counts per batch item, defaulting to the dense packing;
        `leading` overrides the row strides of A, B and C, and `offsets` shifts each
        operand's base in elements, so a GEMM can read or write a slice of a wider
        buffer — which is how the branched and split feed-forwards place their heads.
        Extents must already be multiples of 8 / 16 / 16: cooperative-matrix loads are
        not bounds-checked on this device (`cooperativeMatrixRobustBufferAccess` is
        false), so the padding has to be in the buffer, not in a guard.
        """
        for extent, multiple, name in ((rows, TM, "rows"), (cols, TN, "cols"), (inner, TK, "inner")):
            if extent % multiple:
                raise ValueError(f"{name}={extent} must be a multiple of {multiple}")
        if strides is None:
            strides = (rows * inner, cols * inner if transpose_b else inner * cols, rows * cols)
        lda, ldb, ldc = leading or (0, 0, 0)
        flags = 1 if transpose_b else 0
        if self.lib.xmx_rec_gemm(a.id, b.id, c.id, rows, cols, inner, batch,
                                 strides[0], strides[1], strides[2], flags,
                                 lda, ldb, ldc, *offsets) != 0:
            raise RuntimeError("xmx_rec_gemm: " + self.lib.xmx_error().decode())
        self.recorded += 1
        return self

    def unary(self, kind, source, target, count, *, scale=1.0, second=None,
              third=None, channels=0, _dims=None, _pad=0):
        second = second if second is not None else source
        third = third if third is not None else source
        batch, height, width, across = _dims or (0, 0, 0, 0)
        if self.lib.xmx_rec_unary(kind, source.id, second.id, target.id, third.id,
                                  int(count), int(channels), float(scale),
                                  int(batch), int(height), int(width), int(across),
                                  int(_pad)) != 0:
            raise RuntimeError("xmx_rec_unary: " + self.lib.xmx_error().decode())
        self.recorded += 1
        return self

    def e4m3(self, source, target, count, scale=1.0):
        return self.unary(E4M3, source, target, count, scale=scale)

    def gate(self, source, target, count, scale=1.0):
        return self.unary(GATE, source, target, count, scale=scale)

    def half(self, source, target, count, scale=1.0):
        return self.unary(HALF, source, target, count, scale=scale)

    def gate_e4m3_half(self, source, target, count):
        """gate, publish and narrow in one read and one write.

        The graph writes a float32 hidden buffer, gates it, publishes it and narrows
        it to half — four trips over 503 MB at 720p in block 0 alone. This is one.
        """
        return self.unary(GATE_E4M3_HALF, source, target, count)

    def e4m3_half(self, source, target, count):
        return self.unary(E4M3_HALF, source, target, count)

    def gate_half(self, source, target, count):
        return self.unary(GATE_HALF, source, target, count)

    def to_half(self, source, target, count, scale=1.0):
        """float32 -> float16, for a GEMM operand."""
        return self.unary(TO_HALF, source, target, count, scale=scale)

    def from_half(self, source, target, count, scale=1.0):
        return self.unary(FROM_HALF, source, target, count, scale=scale)

    def scale(self, source, target, count, factor):
        return self.unary(SCALE, source, target, count, scale=factor)

    def residual(self, branch, skip, cosine, target, count, channels):
        """target = branch + skip * cosine, one cosine per channel."""
        return self.unary(RESIDUAL, branch, target, count, second=skip, third=cosine,
                          channels=channels)

    @staticmethod
    def window_extent(height, width, origin=(0, 0), size=8):
        """-> (padded height, padded width, pads) for a shifted-window partition."""
        pad_top, pad_left = -origin[0], -origin[1]
        padded_height = pad_top + height + (-(height + pad_top)) % size
        padded_width = pad_left + width + (-(width + pad_left)) % size
        return padded_height, padded_width, (pad_top, pad_left)

    def partition(self, source, target, height, width, channels, size=8, origin=(0, 0)):
        """NHWC -> (windows, tokens, channels), the shifted-window origin folded in.

        The vendor pads by up to one window before partitioning; rather than write that
        padded copy, the gather reads zero outside the image.
        """
        ph, pw, (top, left) = self.window_extent(height, width, origin, size)
        return self.unary(PARTITION, source, target, ph * pw * channels,
                          channels=channels, scale=1.0,
                          _dims=(size, height, width, pw // size),
                          _pad=(top << 16) | left)

    def reverse(self, source, target, height, width, channels, size=8, origin=(0, 0)):
        """(windows, tokens, channels) -> NHWC, cropping the shifted-window pad away."""
        ph, pw, (top, left) = self.window_extent(height, width, origin, size)
        return self.unary(REVERSE, source, target, height * width * channels,
                          channels=channels, scale=1.0,
                          _dims=(size, height, width, pw // size),
                          _pad=(top << 16) | left)

    def split_heads(self, source, target, windows, tokens, channels, heads, part):
        """(windows, tokens, 3C) -> Q, K or V as (windows, heads, tokens, 32)."""
        return self.unary(SPLIT_HEADS, source, target, windows * tokens * channels,
                          channels=channels, _dims=(heads, tokens, part, 0))

    def merge_heads(self, source, target, windows, tokens, channels, heads):
        """(windows, heads, tokens, 32) -> (windows, tokens, C)."""
        return self.unary(MERGE_HEADS, source, target, windows * tokens * channels,
                          channels=channels, _dims=(heads, tokens, 0, 0))

    def pool2(self, source, target, height, width, channels):
        """2x2 average pool, NHWC."""
        return self.unary(POOL2, source, target, (height // 2) * (width // 2) * channels,
                          channels=channels, _dims=(0, height, width, 0))

    def upsample2(self, source, target, source_width, height, width, channels):
        """Nearest 2x upsample, cropped to (height, width)."""
        return self.unary(UPSAMPLE2, source, target, height * width * channels,
                          channels=channels, _dims=(0, width, source_width, 0))

    def pad_end(self, source, target, height, width, padded_height, padded_width, channels):
        """Extend to a larger extent with zeros, as `pad_spatial_end` does."""
        return self.unary(PAD_END, source, target,
                          padded_height * padded_width * channels, channels=channels,
                          _dims=(0, height, width, padded_width))

    def scale_channel(self, source, factors, target, count, channels):
        """target = source * factors, one factor per channel."""
        return self.unary(SCALE_CHANNEL, source, target, count, channels=channels,
                          third=factors)

    def add(self, left, right, target, count):
        return self.unary(ADD, left, target, count, second=right)

    def add_bias(self, source, bias, target, count, tokens, heads):
        """scores + the per-head attention bias."""
        return self.unary(ADD_BIAS, source, target, count, channels=tokens,
                          third=bias, _dims=(heads, 0, 0, 0))

    def cosine_publish(self, source, target, rows, *, tokens=0, heads=0, scale=None):
        """Normalise rows of 32 through the kernel's fragment tree, then publish as E4M3.

        With `scale` the query path also multiplies by its head's `attn_scale`; rows are
        ordered (batch, head, token), so the head follows from the row index.
        """
        third = scale if scale is not None else source
        if self.lib.xmx_rec_row(COSINE_PUBLISH, source.id, target.id, third.id,
                                int(rows), int(tokens), int(heads),
                                1 if scale is not None else 0, 0, 0.0) != 0:
            raise RuntimeError("xmx_rec_row: " + self.lib.xmx_error().decode())
        self.recorded += 1
        return self

    def softmax(self, source, target, rows, width, *, stride=0, cap=0.0):
        """The bit-affine softmax, one row per invocation.

        `stride` lets a row be wider than its token count, which the global blocks
        need: their token count is the bottleneck's pixel count and need not be a
        multiple of the tile. `cap` is the symmetric logit clamp the vit_1d kernels
        apply.
        """
        if self.lib.xmx_rec_row(SOFTMAX, source.id, target.id, source.id,
                                int(rows), int(width), 0, 0, int(stride), float(cap)) != 0:
            raise RuntimeError("xmx_rec_row: " + self.lib.xmx_error().decode())
        self.recorded += 1
        return self

    def sample_history(self, history, motion, target, height, width, channels=3,
                       absolute=False):
        """Reproject `history` with the recovered five-tap Catmull-Rom filter.

        `motion` holds either the offsets, or the sample coordinates themselves when
        `absolute`, which is the form `sample_history` in the reference takes.
        """
        if self.lib.xmx_rec_history(history.id, motion.id, target.id,
                                    height * width, channels, height, width,
                                    1 if absolute else 0) != 0:
            raise RuntimeError("xmx_rec_history: " + self.lib.xmx_error().decode())
        self.recorded += 1
        return self

    def submit(self):
        count = self.lib.xmx_submit()
        if count < 0:
            raise RuntimeError("xmx_submit: " + self.lib.xmx_error().decode())
        return count
