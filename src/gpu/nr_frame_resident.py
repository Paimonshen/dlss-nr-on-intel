#!/usr/bin/env python3
"""
nr_frame_resident — the whole 71-block graph on the device.

`nr_resident` records a block; this records a frame. Activations never come back to
the host: the stem writes a device buffer, every block, transition and skip reads and
writes device buffers, and only the head is read out.

Levels, for a network extent (H, W). The encoder halves five times and the decoder
mirrors it; the two deepest transitions pad to a multiple of the window first.

    L0  H     x W      C=32     block 0, and block 70 at the end
    L1  H/2   x W/2    C=32     blocks 1-3,   67-69
    L2  H/4   x W/4    C=64     blocks 5-7,   63-65
    L3  H/8   x W/8    C=128    blocks 9-13,  57-61
    L4  H/16  x W/16   C=256    blocks 15-21, 49-55
    L5  pad8(L4)/2     C=512    blocks 23-30, 40-47
    L6  pad8(L5)/2     C=1024   blocks 31-38, every token attending to every other

A block is safe writing over its own input — its target is touched only by the final
residual, after every read of the source — so a level needs one value buffer, not two.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ref"))

import nr_model  # noqa: E402
import nr_resident as R  # noqa: E402
import xmxres  # noqa: E402

ENCODER = ((range(1, 4), 4, 1), (range(5, 8), 8, 2),
           (range(9, 14), 14, 4), (range(15, 22), 22, 8))
DECODER = ((48, range(49, 56), 4, 8), (56, range(57, 62), 3, 4),
           (62, range(63, 66), 2, 2), (66, range(67, 70), 1, 1))


def pad8(extent):
    return -(-extent // 8) * 8


class Edge:
    """A transition's weights: a projection and, for the decoder, a skip scale."""

    def __init__(self, runtime, weight, sine=None):
        self.weight0 = runtime.buffer_from(weight, np.float16)
        self.out_channels = weight.shape[1]
        self.sine = None if sine is None else runtime.buffer_from(sine)


class ResidentFrame:
    """The graph, its weights and its buffers, for one network extent."""

    def __init__(self, runtime, weights, height, width):
        self.rt, self.weights = runtime, weights
        self.height, self.width = height, width
        self.levels = self._plan(height, width)
        self._scratch, self._blocks, self._buffers, self._edges = {}, {}, {}, {}
        self._upload_edges()

    @staticmethod
    def _plan(height, width):
        levels = [(height, width, 32), (height // 2, width // 2, 32),
                  (height // 4, width // 4, 64), (height // 8, width // 8, 128),
                  (height // 16, width // 16, 256)]
        h, w = pad8(levels[4][0]) // 2, pad8(levels[4][1]) // 2
        levels.append((h, w, 512))
        levels.append((pad8(h) // 2, pad8(w) // 2, 1024))
        return levels

    def _upload_edges(self):
        take = lambda name: self.weights[name]
        self.adapter = self.rt.buffer_from(take("block0.layer0.input_adapter_weight"),
                                           np.float16)
        self.bottleneck = Edge(self.rt, take("block30.layer4.weight"))
        self.decoder_input = Edge(self.rt, take("block39.layer0.conv_weight"),
                                  take("block39.layer0.inp_upsample_sin"))
        self.merge_sin = self.rt.buffer_from(take("block70.layer0.inp_merge_sin"))
        self.merge_cos = self.rt.buffer_from(take("block70.layer0.inp_merge_cos"))
        # the head is 32 -> 4, and the cooperative matrix wants a multiple of 16
        # columns; both halves go into one padded matrix and the first four columns
        # of the product are the head
        head = np.zeros((32, 16), dtype=np.float32)
        head[:16, :4] = take("block70.layer0.out_gain")
        head[16:, :4] = take("block70.layer0.out_conv_weight")
        self.head = self.rt.buffer_from(head, np.float16)

    # -- lazily built and reused -----------------------------------------

    def block(self, index, heads, family="window"):
        if (index, family) not in self._blocks:
            builder = {"split": lambda: R.SplitBlockWeights(self.rt, self.weights, index),
                       "global": lambda: R.GlobalBlockWeights(self.rt, self.weights, index),
                       "window": lambda: R.BlockWeights(self.rt, self.weights, index,
                                                        heads=heads)}[family]
            self._blocks[(index, family)] = builder()
        return self._blocks[(index, family)]

    def scratch(self, block, height, width, tokens=None):
        key = (height, width, tokens, block.channels, block.heads,
               getattr(block, "split", False), getattr(block, "branched", False))
        if key not in self._scratch:
            self._scratch[key] = (R.GlobalScratch(self.rt, block, tokens) if tokens
                                  else R.BlockScratch(self.rt, block, height, width))
        return self._scratch[key]

    def transition_scratch(self, elements):
        rounded = 1 << max(1, int(elements - 1)).bit_length()
        if rounded not in self._edges:
            self._edges[rounded] = R.TransitionScratch(self.rt, rounded, 0)
        return self._edges[rounded]

    def buffer(self, name, elements, dtype=np.float32):
        existing = self._buffers.get(name)
        if existing is None or existing.nbytes < elements * np.dtype(dtype).itemsize:
            if existing is not None:
                existing.free()
            existing = self.rt.buffer(elements, dtype)
            self._buffers[name] = existing
        return existing

    def edge(self, index, kind):
        if (index, kind) not in self._edges:
            prefix = f"block{index}.layer0"
            self._edges[(index, kind)] = Edge(
                self.rt, self.weights[f"{prefix}.weight0"],
                self.weights.get(f"{prefix}.sin") if kind == "up" else None)
        return self._edges[(index, kind)]

    # -- the frame --------------------------------------------------------

    def run(self, features, submits=None, capture=None):
        rt = self.rt

        def keep(name, buffer, count, shape=None):
            if capture is not None:
                data = buffer.view()[:count].copy()
                capture[name] = data if shape is None else data.reshape(shape)

        height, width = self.height, self.width
        pixels = height * width
        counter = [0]

        def submit():
            counter[0] += rt.submit()

        stem = self.buffer("stem", pixels * 32)
        source = self.buffer("features", pixels * 16)
        source.view(shape=(pixels, 16))[...] = features.reshape(-1, 16)
        rt.begin()
        rt.to_half(source, self.buffer("features16", pixels * 16, np.float16), pixels * 16)
        rt.gemm(self.buffer("features16", pixels * 16, np.float16), self.adapter, stem,
                pixels, 32, 16)
        submit()

        # block 0 runs at full resolution; its output is both the skip the post block
        # merges and, pooled, the encoder's input
        block0 = self.block(0, 1)
        raw = self.buffer("block0", pixels * 32)
        full_skip = self.buffer("full_skip", pixels * 32)
        h, w, channels = self.levels[1]
        value = self.buffer("l1", h * w * 32)
        rt.begin()
        R.record_block(rt, block0, self.scratch(block0, height, width),
                       source=stem, target=raw)
        # the post block's skip is block 0 published; the encoder pools the
        # *unpublished* output, so both come from `raw` and neither from the other
        rt.e4m3(raw, full_skip, pixels * 32)
        rt.pool2(raw, value, height, width, 32)
        rt.e4m3(value, value, h * w * 32)
        submit()
        keep("stem", stem, pixels * 32, (1, height, width, 32))
        keep("block0", raw, pixels * 32, (1, height, width, 32))
        keep("full_skip", full_skip, pixels * 32, (1, height, width, 32))
        keep("l1_in", value, h * w * 32, (1, h, w, 32))

        skips, level = {}, 1
        for regular, transition, heads in ENCODER:
            h, w, channels = self.levels[level]
            for index in regular:
                block = self.block(index, heads)
                rt.begin()
                R.record_block(rt, block, self.scratch(block, h, w), source=value,
                               target=value)
                rt.e4m3(value, value, h * w * channels)
                submit()
            keep(f"l{level}", value, h * w * channels, (1, h, w, channels))
            skips[level] = self.buffer(f"skip{level}", h * w * channels)
            skips[level].view()[:h * w * channels] = value.view()[:h * w * channels]

            block = self.block(transition, heads)
            edge = self.edge(transition, "down")
            nh, nw, nchannels = self.levels[level + 1]
            unpublished = self.buffer("unpublished", h * w * channels)
            nxt = self.buffer(f"l{level + 1}", nh * nw * nchannels)
            padded = pad8(h) * pad8(w) * channels
            rt.begin()
            R.record_block(rt, block, self.scratch(block, h, w), source=value,
                           target=unpublished)
            R.record_downsample(rt, edge, self.transition_scratch(padded), unpublished,
                                nxt, h, w, channels, pad_to=8 if transition == 22 else 0)
            submit()
            value, level = nxt, level + 1
            keep(f"l{level}_in", value, nh * nw * nchannels, (1, nh, nw, nchannels))

        # the split family, then the bottleneck
        h, w, channels = self.levels[5]
        for index in range(23, 31):
            block = self.block(index, 16, "split")
            rt.begin()
            R.record_block(rt, block, self.scratch(block, h, w), source=value, target=value)
            rt.e4m3(value, value, h * w * channels)
            submit()
        keep("l5", value, h * w * channels, (1, h, w, channels))
        split_skip = self.buffer("split_skip", h * w * channels)
        split_skip.view()[:h * w * channels] = value.view()[:h * w * channels]

        gh, gw, gchannels = self.levels[6]
        deep = self.buffer("l6", gh * gw * gchannels)
        rt.begin()
        R.record_plain_downsample(rt, self.bottleneck, self.transition_scratch(
            pad8(h) * pad8(w) * channels), value, deep, h, w, channels, pad_to=8)
        submit()

        tokens = gh * gw
        for index in range(31, 39):
            block = self.block(index, 32, "global")
            scratch = self.scratch(block, gh, gw, tokens=tokens)
            rt.begin()
            scratch.value.view()[:tokens * gchannels] = deep.view()[:tokens * gchannels]
            R.record_global_block(rt, block, scratch)
            rt.e4m3(scratch.out, scratch.out, scratch.padded * gchannels)
            submit()
            deep.view()[:tokens * gchannels] = scratch.out.view()[:tokens * gchannels]

        # the decoder input merge, then the split family again
        rt.begin()
        R.record_upsample_merge(rt, self.decoder_input, self.transition_scratch(
            h * w * channels), deep, split_skip, value, gh, gw, h, w, gchannels, channels)
        submit()
        for index in range(40, 48):
            block = self.block(index, 16, "split")
            rt.begin()
            R.record_block(rt, block, self.scratch(block, h, w), source=value, target=value)
            rt.e4m3(value, value, h * w * channels)
            submit()

        for transition, regular, skip_level, heads in DECODER:
            sh, sw, schannels = self.levels[skip_level]
            edge = self.edge(transition, "up")
            target = self.buffer(f"d{skip_level}", sh * sw * schannels)
            rt.begin()
            R.record_upsample_merge(rt, edge, self.transition_scratch(
                sh * sw * max(channels, schannels)), value, skips[skip_level], target,
                h, w, sh, sw, channels, schannels)
            block = self.block(transition, heads)
            R.record_block(rt, block, self.scratch(block, sh, sw), source=target,
                           target=target)
            rt.e4m3(target, target, sh * sw * schannels)
            submit()
            value, h, w, channels = target, sh, sw, schannels
            for index in regular:
                block = self.block(index, heads)
                rt.begin()
                R.record_block(rt, block, self.scratch(block, h, w), source=value,
                               target=value)
                rt.e4m3(value, value, h * w * channels)
                submit()

        # back to full resolution, merged with block 0's output, then the head
        merged = self.buffer("merged", pixels * 32)
        upsampled = self.buffer("upsampled", pixels * 32)
        block70 = self.block(70, 1)
        out = self.buffer("out", pixels * 32)
        rt.begin()
        rt.upsample2(value, upsampled, w, height, width, 32)
        rt.scale_channel(upsampled, self.merge_sin, merged, pixels * 32, 32)
        rt.residual(merged, full_skip, self.merge_cos, merged, pixels * 32, 32)
        R.record_block(rt, block70, self.scratch(block70, height, width),
                       source=merged, target=out)
        rt.to_half(out, self.buffer("out16", pixels * 32, np.float16), pixels * 32)
        rt.gemm(self.buffer("out16", pixels * 32, np.float16), self.head,
                self.buffer("head", pixels * 16), pixels, 16, 32)
        submit()

        if submits is not None:
            submits.append(counter[0])
        return self.buffer("head", pixels * 16).view(shape=(pixels, 16))[:, :4].reshape(
            height, width, 4).copy()
