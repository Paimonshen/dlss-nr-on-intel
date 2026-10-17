#!/usr/bin/env python3
"""
nr_topology — where each block of a DLSS-NR network sits, from five stage depths.

nvngx_dlssnr.dll 310.9.16 builds both of its networks with one function (`FUN_18005c5a0`
in `.text`, called twice from `FUN_1800602e0`) out of five stage depths read from `.rdata`:
preset A, the model of 310.8, is {4, 4, 6, 8, 8} at 0x1800de990, and preset B,
`hnet-crazy-cuckoo-small-fp8`, is {2, 2, 2, 2, 2} at 0x1800de9a8. The rest is fixed: the
stem, eight ViT blocks in the bottleneck, the merge into the decoder and the head.
notes/phase72-preset-b.md.

An encoder stage of depth d at 32, 64, 128 or 256 channels is d - 1 window blocks and a
transition block that downsamples; the 512-channel stage is d split blocks, the last of
which also projects to 1024. The decoder mirrors it: d split blocks after the merge, then at
each lower level an upsampling block that takes the transition's output as its skip,
followed by d - 1 window blocks.

A block's shifted window cycles through four phases, none, both, x, y: position & 3 in an
encoder stage and (depth + position) & 3 in a decoder one (`FUN_180059c80`, `FUN_1800568f0`),
phase 0 for the stem and 1 for the head.
"""
from __future__ import annotations

PRESETS = {"A": (4, 4, 6, 8, 8), "B": (2, 2, 2, 2, 2)}
BOTTLENECK_DEPTH = 8

# The window origin (y, x) of each phase, eight-pixel windows shifted by half.
_PHASE_ORIGIN = ((0, 0), (-4, -4), (0, -4), (-4, 0))


class Topology:
    """Block indices and window phases of one network."""

    def __init__(self, depths):
        depths = tuple(int(depth) for depth in depths)
        if len(depths) != 5 or min(depths) < 1:
            raise ValueError(f"five stage depths of at least 1 expected, got {depths}")
        self.depths = depths
        self.stem = 0
        phases = {0: 0}
        index = 1
        # (window blocks, transition block, head count) for levels 0-3
        self.encoder = []
        for level, depth in enumerate(depths[:4]):
            blocks = tuple(range(index, index + depth - 1))
            self.encoder.append((blocks, index + depth - 1, 1 << level))
            for position in range(depth):
                phases[index + position] = position & 3
            index += depth
        self.split_encoder = tuple(range(index, index + depths[4]))
        for position, block in enumerate(self.split_encoder):
            phases[block] = position & 3
        index += depths[4]
        self.bottleneck = tuple(range(index, index + BOTTLENECK_DEPTH))
        index += BOTTLENECK_DEPTH
        self.merge = index
        index += 1
        self.split_decoder = tuple(range(index, index + depths[4]))
        for position, block in enumerate(self.split_decoder):
            phases[block] = (depths[4] + position) & 3
        index += depths[4]
        # (upsampling block, window blocks, skip level, head count) for levels 3-0
        self.decoder = []
        for level in (3, 2, 1, 0):
            depth = depths[level]
            self.decoder.append((index, tuple(range(index + 1, index + depth)), level,
                                 1 << level))
            for position in range(depth):
                phases[index + position] = (depth + position) & 3
            index += depth
        self.head = index
        phases[self.head] = 1
        self.block_count = index + 1
        self._phases = phases

    def __repr__(self):
        return f"Topology({self.depths})"

    @property
    def preset(self):
        """The preset these depths are, or None."""
        for name, depths in PRESETS.items():
            if depths == self.depths:
                return name
        return None

    def window_origin(self, index):
        """The vendor window origin as (y, x); (0, 0) for a block without windows."""
        phase = self._phases.get(index)
        return (0, 0) if phase is None else _PHASE_ORIGIN[phase]

    def role(self, index):
        """What block `index` is, independent of the depths around it."""
        if index == self.stem:
            return ("stem",)
        if index == self.head:
            return ("head",)
        if index == self.merge:
            return ("merge",)
        if index in self.bottleneck:
            return ("vit", index - self.bottleneck[0])
        if index in self.split_encoder:
            return ("split", "encoder", index == self.split_encoder[-1])
        if index in self.split_decoder:
            return ("split", "decoder")
        for level, (blocks, transition, _) in enumerate(self.encoder):
            if index == transition:
                return ("transition", level)
            if index in blocks:
                return ("window", level, "encoder")
        for upsample, blocks, level, _ in self.decoder:
            if index == upsample:
                return ("upsample", level)
            if index in blocks:
                return ("window", level, "decoder")
        raise ValueError(f"no block {index} in {self!r}")

    def same_role(self, index, other):
        """A block of `other` doing what `index` does here: the window blocks of one level
        and side, and the split blocks other than the encoder's last, are interchangeable."""
        role = self.role(index)
        kind = role[0]
        if kind in ("stem", "head", "merge"):
            return getattr(other, kind)
        if kind == "vit":
            return other.bottleneck[role[1]]
        if kind == "split":
            if role[1] == "decoder":
                return other.split_decoder[0]
            return other.split_encoder[-1] if role[2] else other.split_encoder[0]
        if kind == "transition":
            return other.encoder[role[1]][1]
        upsample, decoded = next((block, blocks) for block, blocks, level, _ in other.decoder
                                 if level == role[1])
        if kind == "upsample":
            return upsample
        blocks = other.encoder[role[1]][0] if role[2] == "encoder" else decoded
        if not blocks:
            raise ValueError(f"{other!r} has no {role[2]} window block at level {role[1]}")
        return blocks[0]

    @classmethod
    def from_weights(cls, names):
        """The depths a set of logical weight names was built with, checked block by block.

        Each encoder transition is the first block with a `layer0.weight0`, and the
        512-channel stage ends at the block with `layer4.weight`."""
        names = set(names)

        def has(block, suffix):
            return f"block{block}.{suffix}" in names

        depths, index = [], 1
        for level in range(4):
            start = index
            while not has(index, "layer0.weight0"):
                if not has(index, "layer0.qkv_weight"):
                    raise ValueError(f"block{index}: no window block at level {level}")
                index += 1
            index += 1
            depths.append(index - start)
        start = index
        while not has(index, "layer4.weight"):
            if not has(index, "layer2.attn_bias"):
                raise ValueError(f"block{index}: no split block in the 512 stage")
            index += 1
        depths.append(index + 1 - start)
        topology = cls(depths)
        expected = {
            topology.merge: "layer0.conv_weight",
            topology.head: "layer0.blend_scale",
            **{block: "layer2.qkv_weight" for block in topology.bottleneck},
            **{block: "layer2.attn_bias" for block in topology.split_decoder},
            **{upsample: "layer0.sin" for upsample, _, _, _ in topology.decoder},
        }
        for block, suffix in expected.items():
            if not has(block, suffix):
                raise ValueError(f"depths {topology.depths}: block{block} lacks {suffix}")
        if any(has(topology.block_count, suffix)
               for suffix in ("layer0.qkv_weight", "layer0.weight")):
            raise ValueError(f"depths {topology.depths}: blocks past {topology.head}")
        return topology


PRESET_A = Topology(PRESETS["A"])
PRESET_B = Topology(PRESETS["B"])
