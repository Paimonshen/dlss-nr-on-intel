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


class SplitBlockWeights:
    """A split-family block (23-30, 40-47): four layers, sixteen heads, C=512."""

    def __init__(self, runtime, weights, index):
        self.index, self.heads = index, 16
        self.origin = nr_model.recovered_window_origin(index)
        self.projection = weights[f"block{index}.layer3.projection_weight"]
        self.channels = self.projection.shape[0]
        self.groups = self.channels // 64
        bias = weights[f"block{index}.layer2.attn_bias"]
        if nr_model.uses_fragment_swizzle(index, 16):
            bias = nr_model.recover_attention_bias_layout(bias)
        self.first = runtime.buffer_from(weights[f"block{index}.layer0.first_projection_weight"],
                                         np.float16)
        self.expand = runtime.buffer_from(weights[f"block{index}.layer0.group_expand_weight"],
                                          np.float16)
        self.project = runtime.buffer_from(weights[f"block{index}.layer0.group_project_weight"],
                                           np.float16)
        self.weight3 = runtime.buffer_from(weights[f"block{index}.layer1.weight3"], np.float16)
        self.ffn_cos = runtime.buffer_from(weights[f"block{index}.layer1.ffn_cos_skip"])
        self.qkv = runtime.buffer_from(weights[f"block{index}.layer2.qkv_weight"], np.float16)
        self.scale = runtime.buffer_from(weights[f"block{index}.layer2.attn_scale"])
        self.bias = runtime.buffer_from(bias)
        self.out = runtime.buffer_from(self.projection, np.float16)
        self.attn_cos = runtime.buffer_from(weights[f"block{index}.layer3.attn_cos_skip"])
        self.branched = False
        self.split = True


class BlockWeights:
    """One block's weights, uploaded once and kept on the device."""

    def __init__(self, runtime, weights, index, *, heads):
        prefix = f"block{index}.layer0"
        self.split = False
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


class GlobalBlockWeights:
    """A bottleneck block (31-38): every token attends to every other, 32 heads, C=1024."""

    def __init__(self, runtime, weights, index):
        import math
        self.index, self.heads, self.split, self.branched = index, 32, False, False
        self.projection = weights[f"block{index}.layer4.projection_weight"]
        self.channels = self.projection.shape[0]
        self.expand = runtime.buffer_from(weights[f"block{index}.layer0.weight"], np.float16)
        self.ffn_proj = runtime.buffer_from(weights[f"block{index}.layer1.weight"], np.float16)
        self.hidden_width = weights[f"block{index}.layer0.weight"].shape[1]
        self.ffn_cos = runtime.buffer_from(weights[f"block{index}.layer1.ffn_cos_skip"])
        self.qkv = runtime.buffer_from(weights[f"block{index}.layer2.qkv_weight"], np.float16)
        # the global kernels fold sqrt(head_dim) into the per-head scale
        scale = (weights[f"block{index}.layer2.attn_scale"]
                 * np.float32(math.sqrt(self.channels // self.heads)))
        self.scale = runtime.buffer_from(scale)
        self.out = runtime.buffer_from(self.projection, np.float16)
        self.attn_cos = runtime.buffer_from(weights[f"block{index}.layer4.attn_cos_skip"])
        self.logit_cap = nr_model.GLOBAL_ATTENTION_LOGIT_CAP


class GlobalScratch:
    """Working buffers for one bottleneck block over `tokens` tokens."""

    def __init__(self, runtime, weights, tokens):
        channels, heads = weights.channels, weights.heads
        # the token count is the bottleneck's pixel count and need not be tile-aligned
        self.tokens, self.padded = tokens, xmxres.align(tokens, 16)
        padded, hidden = self.padded, weights.hidden_width
        make = runtime.buffer
        self.value = make(padded * channels).zero()
        self.value16 = make(padded * channels, np.float16)
        self.hidden = make(padded * hidden)
        self.hidden16 = make(padded * hidden, np.float16)
        self.branch = make(padded * channels)
        self.ffn = make(padded * channels)
        self.ffn16 = make(padded * channels, np.float16)
        self.proj = make(padded * channels * 3)
        self.q, self.k, self.v = (make(padded * channels) for _ in range(3))
        self.q16, self.k16, self.v16 = (make(padded * channels, np.float16) for _ in range(3))
        self.scores = make(heads * padded * padded)
        self.probs16 = make(heads * padded * padded, np.float16)
        self.context = make(heads * padded * 32)
        self.merged = make(padded * channels)
        self.merged16 = make(padded * channels, np.float16)
        self.attention = make(padded * channels)
        self.out = make(padded * channels)

    def free(self):
        for name in dir(self):
            value = getattr(self, name)
            if isinstance(value, xmxres.Buffer):
                value.free()


def record_global_block(runtime, w, s, source=None, target=None):
    """A bottleneck block: the wide feed-forward, then attention over every token."""
    source = source or s.value
    target = target or s.out
    channels, heads, padded = w.channels, w.heads, s.padded
    runtime.to_half(source, s.value16, padded * channels)
    runtime.gemm(s.value16, w.expand, s.hidden16, padded, w.hidden_width, channels,
                 epilogue=xmxres.EPI_GATE_E4M3, narrow=True)
    runtime.gemm(s.hidden16, w.ffn_proj, s.branch, padded, channels, w.hidden_width)
    runtime.residual(s.branch, source, w.ffn_cos, s.ffn, padded * channels, channels)

    runtime.to_half(s.ffn, s.ffn16, padded * channels)
    runtime.gemm(s.ffn16, w.qkv, s.proj, padded, 3 * channels, channels)
    for index, part in enumerate((s.q, s.k, s.v)):
        runtime.split_heads(s.proj, part, 1, padded, channels, heads, index)
    runtime.cosine_publish(s.q, s.q, heads * padded, tokens=padded, heads=heads,
                           scale=w.scale)
    runtime.cosine_publish(s.k, s.k, heads * padded, tokens=padded, heads=heads)
    runtime.e4m3_half(s.v, s.v16, padded * channels)
    for part, half in ((s.q, s.q16), (s.k, s.k16)):
        runtime.to_half(part, half, padded * channels)
    runtime.gemm(s.q16, s.k16, s.scores, padded, padded, 32, batch=heads,
                 strides=(padded * 32, padded * 32, padded * padded), transpose_b=True)
    # no attention bias here, and the logits are clamped symmetrically
    runtime.softmax(s.scores, s.scores, heads * padded, s.tokens,
                    stride=padded, cap=w.logit_cap)
    runtime.to_half(s.scores, s.probs16, heads * padded * padded)
    runtime.gemm(s.probs16, s.v16, s.context, padded, 32, padded, batch=heads,
                 strides=(padded * padded, padded * 32, padded * 32))
    runtime.merge_heads(s.context, s.merged, 1, padded, channels, heads)
    runtime.e4m3_half(s.merged, s.merged16, padded * channels)
    runtime.gemm(s.merged16, w.out, s.attention, padded, channels, channels)
    runtime.residual(s.attention, s.ffn, w.attn_cos, target, padded * channels, channels)


def run_global_block(runtime, w, s, value):
    """Host convenience: `value` is (tokens, channels)."""
    tokens, channels = value.shape
    s.value.view(shape=(s.padded, channels))[:tokens] = value
    runtime.begin()
    record_global_block(runtime, w, s)
    passes = runtime.submit()
    return s.out.view(shape=(s.padded, channels))[:tokens].copy(), passes


class BlockScratch:
    """Working buffers for one block at one extent, allocated once and reused."""

    def __init__(self, runtime, weights, height, width):
        channels, heads = weights.channels, weights.heads
        # Sized for the largest window count any origin can produce, so blocks at the
        # same level share one scratch even though their shifts differ. Getting this
        # wrong is silent: a shifted block has more windows than an unshifted one and
        # would write past the end of buffers cut to the unshifted size.
        padded_height, padded_width, _ = runtime.window_extent(height, width, (-4, -4))
        self.height, self.width = height, width
        self.tokens = 64
        self.windows = (padded_height // 8) * (padded_width // 8)
        self.batch = self.windows * heads
        pixels = height * width
        windowed = self.windows * self.tokens * channels
        hidden = weights.groups * 128 if weights.branched else weights.expand.nbytes // 2 // channels

        if getattr(weights, "split", False):
            hidden = weights.groups * 256
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
        self.merged_core = make(pixels * channels)
        self.core16 = make(pixels * channels, np.float16)
        self.hidden_width = hidden

    def free(self):
        for name in dir(self):
            value = getattr(self, name)
            if isinstance(value, xmxres.Buffer):
                value.free()


def record_feed_forward(runtime, w, s, source):
    """The block's feed-forward, into `s.ffn`. Branched or plain, as the block is."""
    pixels, channels = s.height * s.width, w.channels
    runtime.to_half(source, s.value16, pixels * channels)
    if w.branched:
        for head in range(w.groups):
            runtime.gemm(s.value16, w.expand, s.hidden16, pixels, 128, channels,
                         leading=(0, 0, s.hidden_width),
                         offsets=(0, head * channels * 128, head * 128),
                         epilogue=xmxres.EPI_GATE_E4M3, narrow=True)
        for head in range(w.groups):
            runtime.gemm(s.hidden16, w.branch, s.heads16, pixels, 32, 128,
                         leading=(s.hidden_width, 0, channels),
                         offsets=(head * 128, head * 128 * 32, head * 32),
                         epilogue=xmxres.EPI_E4M3, narrow=True)
        runtime.gemm(s.heads16, w.ffn_out, s.branch, pixels, channels, channels)
        runtime.residual(s.branch, source, w.ffn_cos, s.ffn, pixels * channels, channels)
        # the fused multi-head kernels publish the residual before attention reads it
        runtime.e4m3(s.ffn, s.ffn, pixels * channels)
    else:
        runtime.gemm(s.value16, w.expand, s.hidden16, pixels, s.hidden_width, channels,
                     epilogue=xmxres.EPI_GATE_E4M3, narrow=True)
        runtime.gemm(s.hidden16, w.branch, s.branch, pixels, channels, s.hidden_width)
        runtime.residual(s.branch, source, w.ffn_cos, s.ffn, pixels * channels, channels)


def record_split_feed_forward(runtime, w, s, source):
    """The split family's core: e4m3(x @ first), then a per-64-group 64 -> 256 -> 64 MLP.

    The gate sits between the two group GEMMs with no publish, so the wide buffer is
    gated in one dense pass; the group outputs are published once, together.
    """
    pixels, channels, groups = s.height * s.width, w.channels, w.groups
    wide = groups * 256
    runtime.to_half(source, s.value16, pixels * channels)
    runtime.gemm(s.value16, w.first, s.heads16, pixels, channels, channels,
                 epilogue=xmxres.EPI_E4M3, narrow=True)
    for group in range(groups):
        runtime.gemm(s.heads16, w.expand, s.hidden16, pixels, 256, 64,
                     leading=(channels, 0, wide),
                     offsets=(group * 64, group * 64 * 256, group * 256),
                     epilogue=xmxres.EPI_GATE, narrow=True)
    for group in range(groups):
        runtime.gemm(s.hidden16, w.project, s.core16, pixels, 64, 256,
                     leading=(wide, 0, channels),
                     offsets=(group * 256, group * 256 * 64, group * 64),
                     epilogue=xmxres.EPI_E4M3, narrow=True)
    runtime.gemm(s.core16, w.weight3, s.branch, pixels, channels, channels)
    runtime.residual(s.branch, source, w.ffn_cos, s.ffn, pixels * channels, channels)


def record_window_attention(runtime, w, s, source):
    """Window attention over `source`, into `s.attention`.

    The window count follows this block's own origin, not the scratch's worst case.
    """
    channels, heads, tokens = w.channels, w.heads, s.tokens
    padded_height, padded_width, _ = runtime.window_extent(s.height, s.width, w.origin)
    windows = (padded_height // 8) * (padded_width // 8)
    batch = windows * heads
    windowed = windows * tokens * channels
    runtime.partition(source, s.win, s.height, s.width, channels, origin=w.origin)
    runtime.to_half(s.win, s.win16, windowed)
    runtime.gemm(s.win16, w.qkv, s.proj, windows * tokens, 3 * channels, channels)
    for index, part in enumerate((s.q, s.k, s.v)):
        runtime.split_heads(s.proj, part, windows, tokens, channels, heads, index)
    runtime.cosine_publish(s.q, s.q, batch * tokens, tokens=tokens, heads=heads,
                           scale=w.scale)
    runtime.cosine_publish(s.k, s.k, batch * tokens, tokens=tokens, heads=heads)
    runtime.e4m3_half(s.v, s.v16, windowed)
    for part, half in ((s.q, s.q16), (s.k, s.k16)):
        runtime.to_half(part, half, windowed)
    runtime.gemm(s.q16, s.k16, s.scores, tokens, tokens, 32, batch=batch,
                 strides=(tokens * 32, tokens * 32, tokens * tokens), transpose_b=True)
    runtime.add_bias(s.scores, w.bias, s.scores, batch * tokens * tokens, tokens, heads)
    runtime.softmax(s.scores, s.scores, batch * tokens, tokens)
    runtime.to_half(s.scores, s.probs16, batch * tokens * tokens)
    runtime.gemm(s.probs16, s.v16, s.context, tokens, 32, tokens, batch=batch,
                 strides=(tokens * tokens, tokens * 32, tokens * 32))
    runtime.merge_heads(s.context, s.merged, windows, tokens, channels, heads)
    runtime.e4m3_half(s.merged, s.merged16, windowed)
    runtime.gemm(s.merged16, w.out, s.attended, windows * tokens, channels, channels)
    runtime.reverse(s.attended, s.attention, s.height, s.width, channels,
                    origin=w.origin)


def record_block(runtime, w, s, source=None, target=None):
    """A whole window block: feed-forward, attention, both residuals.

    `source` and `target` default to the scratch's own buffers; passing them lets one
    level's blocks chain into the next without a copy.
    """
    source = source or s.value
    target = target or s.out
    pixels = s.height * s.width
    if getattr(w, "split", False):
        record_split_feed_forward(runtime, w, s, source)
    else:
        record_feed_forward(runtime, w, s, source)
    record_window_attention(runtime, w, s, s.ffn)
    runtime.residual(s.attention, s.ffn, w.attn_cos, target, pixels * w.channels,
                     w.channels)


def run_block(runtime, w, s, value):
    """Host convenience: one block, one submit, numpy in and numpy out."""
    s.value.view(shape=value.shape)[...] = value
    runtime.begin()
    record_block(runtime, w, s)
    passes = runtime.submit()
    return s.out.view(shape=value.shape).copy(), passes


# --------------------------------------------------------------------------
# transitions between levels
# --------------------------------------------------------------------------


class Transition:
    """The weights a level change needs, uploaded once."""

    def __init__(self, runtime, weights, index, *, kind):
        self.kind = kind
        prefix = f"block{index}.layer0"
        if kind in ("down", "up"):
            self.weight0 = runtime.buffer_from(weights[f"{prefix}.weight0"], np.float16)
            self.out_channels = weights[f"{prefix}.weight0"].shape[1]
        if kind == "up":
            self.sine = runtime.buffer_from(weights[f"{prefix}.sin"])


def record_downsample(runtime, transition, scratch, source, target, height, width,
                      channels, *, pad_to=0):
    """Pool the block's unpublished output, publish it, then project.

    The fused `ds` kernels pool the half-precision output before its E4M3 publish and
    publish the pooled tensor again before the QMMA projection, so both are here.
    """
    if pad_to:
        padded_height = -(-height // pad_to) * pad_to
        padded_width = -(-width // pad_to) * pad_to
        runtime.pad_end(source, scratch.padded, height, width, padded_height,
                        padded_width, channels)
        source, height, width = scratch.padded, padded_height, padded_width
    half_height, half_width = height // 2, width // 2
    pixels = half_height * half_width
    runtime.pool2(source, scratch.pooled, height, width, channels)
    runtime.e4m3_half(scratch.pooled, scratch.pooled16, pixels * channels)
    runtime.gemm(scratch.pooled16, transition.weight0, target, pixels,
                 transition.out_channels, channels, epilogue=xmxres.EPI_E4M3)
    return half_height, half_width


def record_upsample_merge(runtime, transition, scratch, source, skip, target,
                          source_height, source_width, height, width, channels,
                          out_channels):
    """Project, nearest-upsample onto the skip, add the scaled skip, publish.

    The fused `upsample` kernels read the merged tensor as E4M3, so the publish is
    part of the transition rather than of the block that follows.
    """
    source_pixels = source_height * source_width
    runtime.to_half(source, scratch.projected16, source_pixels * channels)
    runtime.gemm(scratch.projected16, transition.weight0, scratch.projected,
                 source_pixels, out_channels, channels)
    runtime.upsample2(scratch.projected, scratch.upsampled, source_width, height, width,
                      out_channels)
    runtime.scale_channel(skip, transition.sine, scratch.scaled, height * width * out_channels,
                          out_channels)
    runtime.add(scratch.upsampled, scratch.scaled, target, height * width * out_channels)
    runtime.e4m3(target, target, height * width * out_channels)


class TransitionScratch:
    """Buffers a transition needs, sized for the largest level that uses it."""

    def __init__(self, runtime, elements, half_elements):
        make = runtime.buffer
        self.padded = make(elements)
        self.pooled = make(elements)
        self.pooled16 = make(elements, np.float16)
        self.projected = make(elements)
        self.projected16 = make(elements, np.float16)
        self.upsampled = make(elements)
        self.scaled = make(elements)

    def free(self):
        for name in dir(self):
            value = getattr(self, name)
            if isinstance(value, xmxres.Buffer):
                value.free()


def record_plain_downsample(runtime, edge, scratch, source, target, height, width,
                            channels, *, pad_to=0):
    """`downsample()`: pool then project, with no publish between.

    Block 30's bridge into the bottleneck, which unlike the encoder's `ds` kernels
    does not republish the pooled tensor before the projection.
    """
    if pad_to:
        padded_height = -(-height // pad_to) * pad_to
        padded_width = -(-width // pad_to) * pad_to
        runtime.pad_end(source, scratch.padded, height, width, padded_height,
                        padded_width, channels)
        source, height, width = scratch.padded, padded_height, padded_width
    pixels = (height // 2) * (width // 2)
    runtime.pool2(source, scratch.pooled, height, width, channels)
    runtime.to_half(scratch.pooled, scratch.pooled16, pixels * channels)
    runtime.gemm(scratch.pooled16, edge.weight0, target, pixels, edge.out_channels,
                 channels, epilogue=xmxres.EPI_E4M3)
