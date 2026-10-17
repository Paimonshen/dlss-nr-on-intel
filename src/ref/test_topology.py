#!/usr/bin/env python3
"""
test_topology — the block layout of both presets, without weights.

Preset A against the explicit window-origin table this port used until the depths were read
out of `.text` (MLX-DLSS's recovered graph, which `test_against_torch.py` holds to its
original), preset B against what the builder's rule gives it, and the inference of the
depths from weight names both ways. Exits 0 on success.

    python3 src/ref/test_topology.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nr_topology as T  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(name)


def preset_a_origin(index):
    """The table `nr_model.recovered_window_origin` was before 2026-10-17, as written."""
    if index == 0:
        phase = 0
    elif 1 <= index <= 4:
        phase = index - 1
    elif 5 <= index <= 8:
        phase = index - 5
    elif 9 <= index <= 14:
        phase = index - 9
    elif 15 <= index <= 22:
        phase = index - 15
    elif 23 <= index <= 30:
        phase = index - 23
    elif 40 <= index <= 55:
        phase = index - (40 if index < 48 else 48)
    elif 56 <= index <= 61:
        phase = index - 54
    elif 62 <= index <= 69:
        phase = index - (62 if index < 66 else 66)
    elif index == 70:
        phase = 1
    else:
        return (0, 0)
    return ((0, -4, 0, -4)[phase % 4], (0, -4, -4, 0)[phase % 4])


def names_of(topology):
    """The logical weight names that identify each block's family, for `from_weights`."""
    names = set()
    for block in range(topology.block_count):
        kind = topology.role(block)[0]
        suffixes = {
            "stem": ["layer0.qkv_weight", "layer0.input_adapter_weight"],
            "window": ["layer0.qkv_weight"],
            "transition": ["layer0.qkv_weight", "layer0.weight0"],
            "upsample": ["layer0.qkv_weight", "layer0.weight0", "layer0.sin"],
            "split": ["layer2.attn_bias", "layer2.qkv_weight"],
            "vit": ["layer2.qkv_weight", "layer0.weight"],
            "merge": ["layer0.conv_weight"],
            "head": ["layer0.qkv_weight", "layer0.blend_scale"],
        }[kind]
        if topology.role(block) == ("split", "encoder", True):
            suffixes.append("layer4.weight")
        names.update(f"block{block}.{suffix}" for suffix in suffixes)
    return names


def main():
    A, B = T.PRESET_A, T.PRESET_B
    print("preset A, {4, 4, 6, 8, 8}")
    check("71 blocks", A.block_count == 71, str(A.block_count))
    wrong = [i for i in range(90) if A.window_origin(i) != preset_a_origin(i)]
    check("window origins equal the explicit table, blocks 0-89", not wrong, str(wrong[:4]))
    check("transitions are 4, 8, 14, 22", [t for _, t, _ in A.encoder] == [4, 8, 14, 22])
    check("split 23-30, bottleneck 31-38, merge 39, split 40-47, head 70",
          (A.split_encoder, A.bottleneck, A.merge, A.split_decoder, A.head)
          == (tuple(range(23, 31)), tuple(range(31, 39)), 39, tuple(range(40, 48)), 70))
    check("upsampling blocks 48, 56, 62, 66 merge levels 3, 2, 1, 0",
          [(u, level) for u, _, level, _ in A.decoder] == [(48, 3), (56, 2), (62, 1), (66, 0)])

    print("preset B, {2, 2, 2, 2, 2}")
    check("31 blocks", B.block_count == 31, str(B.block_count))
    check("transitions are 2, 4, 6, 8", [t for _, t, _ in B.encoder] == [2, 4, 6, 8])
    check("split 9-10, bottleneck 11-18, merge 19, split 20-21, head 30",
          (B.split_encoder, B.bottleneck, B.merge, B.split_decoder, B.head)
          == ((9, 10), tuple(range(11, 19)), 19, (20, 21), 30))
    # encoder: phase = position & 3; decoder: (depth + position) & 3 = position + 2
    expected = {0: (0, 0), 30: (-4, -4)}
    expected.update({block: (0, 0) for block in (1, 3, 5, 7, 9)})
    expected.update({block: (-4, -4) for block in (2, 4, 6, 8, 10)})
    expected.update({block: (0, -4) for block in (20, 22, 24, 26, 28)})
    expected.update({block: (-4, 0) for block in (21, 23, 25, 27, 29)})
    expected.update({block: (0, 0) for block in range(11, 20)})
    wrong = [i for i in range(31) if B.window_origin(i) != expected[i]]
    check("window phases as the builder sets them", not wrong, str(wrong[:4]))
    roles = [B.same_role(i, A) for i in range(31)]
    check("each block has preset A's block of the same role", roles == [
        0, 1, 4, 5, 8, 9, 14, 15, 22, 23, 30, *range(31, 39), 39, 40, 40,
        48, 49, 56, 57, 62, 63, 66, 67, 70], str(roles))

    print("depths from weight names")
    for topology in (A, B, T.Topology((1, 3, 2, 5, 1))):
        found = T.Topology.from_weights(names_of(topology))
        check(f"{topology.depths} read back", found.depths == topology.depths, repr(found))
    names = names_of(B) - {"block19.layer0.conv_weight"}
    try:
        T.Topology.from_weights(names)
        check("a missing merge is refused", False)
    except ValueError as error:
        check("a missing merge is refused", "block19" in str(error), str(error))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("both presets lay out as the builder makes them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
