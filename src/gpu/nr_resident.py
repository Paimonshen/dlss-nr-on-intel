#!/usr/bin/env python3
"""
nr_resident — whole blocks of the graph recorded as one device submit.

`xmxres` gives the primitives; this assembles them into the shapes the recovered
graph actually has. One block is a single command buffer: its feed-forward, its
window attention and both residuals, with nothing crossing back to the host in
between.

The E4M3 publishes are what make the layout work. Because the publish is
elementwise, a branched feed-forward can write each of its heads straight into a
slice of one wide buffer and publish the whole thing in a single dense pass — no
concatenation, no strided elementwise kernel.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ref"))

import nr_model  # noqa: E402
import xmxres  # noqa: E402


class BlockWeights:
    """One block's weights, uploaded once and kept on the device."""

    def __init__(self, runtime, weights, index, *, heads):
        prefix = f"block{index}.layer0"
        self.index, self.heads = index, heads
        self.origin = nr_model.recovered_window_origin(index)
        self.projection = weights[f"{prefix}.projection_weight"]
        self.channels = self.projection.shape[0]

        bias = weights[f"{prefix}.attn_bias"]
        if nr_model.uses_fragment_swizzle(index, heads):
            bias = nr_model.recover_attention_bias_layout(bias)

        take = lambda name, dtype=np.float16: runtime.buffer_from(
            weights[f"{prefix}.{name}"], dtype)
        self.qkv = take("qkv_weight")
        self.out = take("projection_weight")
        self.bias = runtime.buffer_from(bias, np.float32)
        self.scale = runtime.buffer_from(weights[f"{prefix}.attn_scale"], np.float32)
        self.attn_cos = runtime.buffer_from(weights[f"{prefix}.attn_cos_skip"])
        self.ffn_cos = runtime.buffer_from(weights[f"{prefix}.ffn_cos_skip"])

        self.branched = f"{prefix}.ffn_expand_weight" in weights
        if self.branched:
            expansion, branch = nr_model._fused_branched_weights(
                weights[f"{prefix}.ffn_expand_weight"],
                weights[f"{prefix}.ffn_branch_projection_weight"])
            self.groups = expansion.shape[0]
            self.expand = runtime.buffer_from(expansion, np.float16)
            self.branch = runtime.buffer_from(branch, np.float16)
            self.ffn_out = take("ffn_output_projection_weight")
        else:
            self.groups = 0
            self.expand = take("weight1")
            self.branch = take("weight2")
            self.ffn_out = None


class BlockScratch:
    """Working buffers for one block at one extent, allocated once and reused."""

    def __init__(self, runtime, weights, height, width):
        channels, heads = weights.channels, weights.heads
        padded_height, padded_width, _ = runtime.window_extent(
            height, width, weights.origin)
        self.height, self.width = height, width
        self.tokens = 64
        self.windows = (padded_height // 8) * (padded_width // 8)
        self.batch = self.windows * heads
        pixels = height * width
        windowed = self.windows * self.tokens * channels
        hidden = weights.groups * 128 if weights.branched else weights.expand.nbytes // 2 // channels

        make = runtime.buffer
        self.value = make(pixels * channels)
        self.value16 = make(pixels * channels, np.float16)
        self.hidden = make(pixels * hidden)
        self.hidden16 = make(pixels * hidden, np.float16)
        self.heads_out = make(pixels * channels)
        self.heads16 = make(pixels * channels, np.float16)
        self.branch = make(pixels * channels)
        self.ffn = make(pixels * channels)
        self.win = make(windowed)
        self.win16 = make(windowed, np.float16)
        self.proj = make(windowed * 3)
        self.q, self.k, self.v = (make(windowed) for _ in range(3))
        self.q16, self.k16, self.v16 = (make(windowed, np.float16) for _ in range(3))
        self.scores = make(self.batch * self.tokens * self.tokens)
        self.probs16 = make(self.batch * self.tokens * self.tokens, np.float16)
        self.context = make(self.batch * self.tokens * 32)
        self.merged = make(windowed)
        self.merged16 = make(windowed, np.float16)
        self.attended = make(windowed)
        self.attention = make(pixels * channels)
        self.out = make(pixels * channels)
        self.hidden_width = hidden

    def free(self):
        for name in dir(self):
            value = getattr(self, name)
            if isinstance(value, xmxres.Buffer):
                value.free()


def record_feed_forward(runtime, w, s):
    """The block's feed-forward, into `s.ffn`. Branched or plain, as the block is."""
    pixels, channels = s.height * s.width, w.channels
    runtime.to_half(s.value, s.value16, pixels * channels)
    if w.branched:
        for head in range(w.groups):
            runtime.gemm(s.value16, w.expand, s.hidden, pixels, 128, channels,
                         leading=(0, 0, s.hidden_width),
                         offsets=(0, head * channels * 128, head * 128))
        runtime.gate(s.hidden, s.hidden, pixels * s.hidden_width)
        runtime.e4m3(s.hidden, s.hidden, pixels * s.hidden_width)
        runtime.to_half(s.hidden, s.hidden16, pixels * s.hidden_width)
        for head in range(w.groups):
            runtime.gemm(s.hidden16, w.branch, s.heads_out, pixels, 32, 128,
                         leading=(s.hidden_width, 0, channels),
                         offsets=(head * 128, head * 128 * 32, head * 32))
        runtime.e4m3(s.heads_out, s.heads_out, pixels * channels)
        runtime.to_half(s.heads_out, s.heads16, pixels * channels)
        runtime.gemm(s.heads16, w.ffn_out, s.branch, pixels, channels, channels)
        runtime.residual(s.branch, s.value, w.ffn_cos, s.ffn, pixels * channels, channels)
        # the fused multi-head kernels publish the residual before attention reads it
        runtime.e4m3(s.ffn, s.ffn, pixels * channels)
    else:
        runtime.gemm(s.value16, w.expand, s.hidden, pixels, s.hidden_width, channels)
        runtime.gate(s.hidden, s.hidden, pixels * s.hidden_width)
        runtime.e4m3(s.hidden, s.hidden, pixels * s.hidden_width)
        runtime.to_half(s.hidden, s.hidden16, pixels * s.hidden_width)
        runtime.gemm(s.hidden16, w.branch, s.branch, pixels, channels, s.hidden_width)
        runtime.residual(s.branch, s.value, w.ffn_cos, s.ffn, pixels * channels, channels)


def record_window_attention(runtime, w, s, source):
    """Window attention over `source`, into `s.attention`."""
    channels, heads, tokens = w.channels, w.heads, s.tokens
    windowed = s.windows * tokens * channels
    runtime.partition(source, s.win, s.height, s.width, channels, origin=w.origin)
    runtime.to_half(s.win, s.win16, windowed)
    runtime.gemm(s.win16, w.qkv, s.proj, s.windows * tokens, 3 * channels, channels)
    for index, part in enumerate((s.q, s.k, s.v)):
        runtime.split_heads(s.proj, part, s.windows, tokens, channels, heads, index)
    runtime.cosine_publish(s.q, s.q, s.batch * tokens, tokens=tokens, heads=heads,
                           scale=w.scale)
    runtime.cosine_publish(s.k, s.k, s.batch * tokens, tokens=tokens, heads=heads)
    runtime.e4m3(s.v, s.v, windowed)
    for part, half in ((s.q, s.q16), (s.k, s.k16), (s.v, s.v16)):
        runtime.to_half(part, half, windowed)
    runtime.gemm(s.q16, s.k16, s.scores, tokens, tokens, 32, batch=s.batch,
                 strides=(tokens * 32, tokens * 32, tokens * tokens), transpose_b=True)
    runtime.add_bias(s.scores, w.bias, s.scores, s.batch * tokens * tokens, tokens, heads)
    runtime.softmax(s.scores, s.scores, s.batch * tokens, tokens)
    runtime.to_half(s.scores, s.probs16, s.batch * tokens * tokens)
    runtime.gemm(s.probs16, s.v16, s.context, tokens, 32, tokens, batch=s.batch,
                 strides=(tokens * tokens, tokens * 32, tokens * 32))
    runtime.merge_heads(s.context, s.merged, s.windows, tokens, channels, heads)
    runtime.e4m3(s.merged, s.merged, windowed)
    runtime.to_half(s.merged, s.merged16, windowed)
    runtime.gemm(s.merged16, w.out, s.attended, s.windows * tokens, channels, channels)
    runtime.reverse(s.attended, s.attention, s.height, s.width, channels,
                    origin=w.origin)


def record_block(runtime, w, s):
    """A whole window block: feed-forward, attention, both residuals."""
    pixels = s.height * s.width
    record_feed_forward(runtime, w, s)
    record_window_attention(runtime, w, s, s.ffn)
    runtime.residual(s.attention, s.ffn, w.attn_cos, s.out, pixels * w.channels,
                     w.channels)


def run_block(runtime, w, s, value):
    """Host convenience: one block, one submit, numpy in and numpy out."""
    s.value.view(shape=value.shape)[...] = value
    runtime.begin()
    record_block(runtime, w, s)
    passes = runtime.submit()
    return s.out.view(shape=value.shape).copy(), passes
