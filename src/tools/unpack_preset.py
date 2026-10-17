#!/usr/bin/env python3
"""
unpack_preset — a preset's weights as logical tensors, through MLX-DLSS's unpacker.

nvngx_dlssnr.dll 310.9.16 carries two networks in one container format: preset A in
`WEIGHTS_HT`, byte for byte the model of 310.8, and preset B in `WEIGHTS_CC_SMALL_FP8`.
Each of B's 77 records is exactly the size of one of A's record families
(notes/phase72-preset-b.md), and MLX-DLSS's unpacker decodes those families by A's block
indices. So each record is decoded under the index of the preset A block with the same role
(`nr_topology.Topology.same_role`), and its tensors are renamed back to its own block.
Run with `--preset A` it reproduces `work/mlxw/dlssnr-logical.safetensors`, which checks the
renaming end to end.

Usage:
    unpack_preset.py <nvngx_dlssnr.dll> <out.safetensors> [--preset A|B] [--check REF]
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "ref"))
sys.path.append(str(ROOT / "work" / "shim"))
import nr_topology  # noqa: E402

RESOURCES = {"A": "WEIGHTS_HT", "B": "WEIGHTS_CC_SMALL_FP8"}


def load_tool(directory, name):
    """One of MLX-DLSS's tool modules, by its path rather than through sys.path."""
    spec = importlib.util.spec_from_file_location(f"mlxdlss_{name}", directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # its dataclasses look themselves up there
    spec.loader.exec_module(module)
    return module


def unpack(records, topology, unpacker):
    """{name: payload} of one preset -> its logical tensors under its own block indices."""
    reference = nr_topology.PRESET_A
    blocks = {int(name.split(".")[0][5:]) for name in records}
    if blocks != set(range(topology.block_count)):
        raise ValueError(f"records name blocks {sorted(blocks)[:3]}..., "
                         f"{topology!r} has {topology.block_count}")
    decoded = {}
    for name, payload in sorted(records.items()):
        block, rest = name.split(".", 1)
        own = int(block[5:])
        alias = topology.same_role(own, reference)
        tensors = unpacker.unpack_known_tensor(f"block{alias}.{rest}", payload)
        if not tensors:
            raise ValueError(f"{name}: no decoder for its family (as block{alias})")
        for logical, value in tensors.items():
            renamed = f"block{own}." + logical.split(".", 1)[1]
            if renamed in decoded:
                raise ValueError(f"{name}: duplicate tensor {renamed}")
            decoded[renamed] = value
    found = nr_topology.Topology.from_weights(decoded)
    if found.depths != topology.depths:
        raise ValueError(f"decoded weights read as {found!r}, expected {topology!r}")
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dll", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--preset", choices=sorted(RESOURCES), default="B")
    parser.add_argument("--mlx-dlss", type=pathlib.Path, default=ROOT / "work" / "mlx-dlss")
    parser.add_argument("--check", type=pathlib.Path,
                        help="a logical safetensors the result must equal, tensor for tensor")
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists")

    tools = args.mlx_dlss / "python" / "mlxdlss" / "tools"
    extractor = load_tool(tools, "extract_dlssnr_weights")
    unpacker = load_tool(tools, "unpack_dlssnr_weights")
    from safetensors.numpy import save_file

    dll = args.dll.read_bytes()
    resource = extractor.extract_pe_resource(dll, RESOURCES[args.preset])
    records = {name: record.payload
               for name, record in extractor.parse_weight_map(resource).items()}
    topology = nr_topology.Topology(nr_topology.PRESETS[args.preset])
    decoded = unpack(records, topology, unpacker)

    metadata = {
        "format": unpacker.LOGICAL_FORMAT,
        "fully_logical": "true",
        "preset": args.preset,
        "depths": json.dumps(list(topology.depths)),
        "resource": RESOURCES[args.preset],
        "resource_sha256": hashlib.sha256(resource).hexdigest(),
        "source_sha256": hashlib.sha256(dll).hexdigest(),
        "source_tensor_count": str(len(records)),
        "decoded_tensor_count": str(len(decoded)),
        "matrix_contract": "K-by-N; output = input @ weight",
        "renaming": "each record decoded as the preset A block of the same role",
    }
    if args.check:
        sys.path.insert(0, str(ROOT / "src" / "ref"))
        import nr_model
        reference, _ = nr_model.load_logical(args.check)
        if set(reference) != set(decoded):
            sys.exit(f"--check: {len(set(reference) ^ set(decoded))} names differ")
        unequal = [name for name in sorted(decoded)
                   if not np.array_equal(np.asarray(decoded[name], np.float32),
                                         reference[name])]
        if unequal:
            sys.exit(f"--check: {len(unequal)} tensors differ, first {unequal[0]}")
        print(f"--check: all {len(decoded)} tensors equal {args.check}")
    save_file(dict(sorted(decoded.items())), str(args.out), metadata=metadata)
    print(f"preset {args.preset} {topology.depths}: {len(records)} records -> "
          f"{len(decoded)} tensors, {topology.block_count} blocks -> {args.out}")


if __name__ == "__main__":
    main()
