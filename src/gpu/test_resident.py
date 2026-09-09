#!/usr/bin/env python3
"""
test_resident — the device-resident runtime against the CPU reference.

Two things are checked separately, because they fail for different reasons:

  * the **operators**, which must be bit-identical to `nr_model`'s — the E4M3
    publish, the quadratic gate and the half rounding, over four magnitude regimes
    including the inf/NaN one;
  * the **chain**, where the GPU's float16 GEMM operands are a real difference from
    the float32 reference, so it is compared both ways: against the reference as
    written, and against the reference with its GEMM inputs rounded to half, which
    isolates the plumbing from the precision.

    python3 src/gpu/test_resident.py
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "ref"))

import nr_model as M  # noqa: E402
import xmxres  # noqa: E402

WEIGHTS = ROOT / "work" / "mlxw" / "dlssnr-logical.safetensors"
FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(name)


def test_operators(runtime):
    print("elementwise operators, against nr_model")
    rng = np.random.default_rng(0)
    regimes = {
        "linear sweep": np.arange(8192, dtype=np.float32) * 0.001 - 4.0,
        "wide random": (rng.standard_normal(8192) * 20).astype(np.float32),
        "tiny": (rng.standard_normal(8192) * 1e-5).astype(np.float32),
        "huge (inf/NaN)": (rng.standard_normal(8192) * 1e5).astype(np.float32),
    }
    for label, values in regimes.items():
        count = values.size
        source = runtime.buffer_from(values)
        outputs = [runtime.buffer(count) for _ in range(3)]
        runtime.begin()
        runtime.e4m3(source, outputs[0], count)
        runtime.gate(source, outputs[1], count)
        runtime.half(source, outputs[2], count)
        runtime.submit()
        with np.errstate(invalid="ignore", over="ignore"):
            expected = (M.e4m3(values), M.quadratic_gate_activation(values),
                        M._half_rounded(values))
        for name, got, want in zip(("e4m3", "gate", "half"), outputs, expected):
            check(f"{name} ({label})",
                  np.array_equal(got.view()[:count], want, equal_nan=True))
        for buffer in [source, *outputs]:
            buffer.free()


def test_chain(runtime, weights, tokens=4096):
    print("a whole feed-forward, one submit")
    rng = np.random.default_rng(1)
    expand = weights["block1.layer0.weight1"]
    project = weights["block1.layer0.weight2"]
    cosine = weights["block1.layer0.ffn_cos_skip"]
    channels, hidden = expand.shape
    value = (rng.standard_normal((tokens, channels)) * 0.3).astype(np.float32)

    reference = M.cosine_residual(
        value, M.e4m3(M.quadratic_gate_activation(value @ expand)) @ project, cosine)

    def half(array):
        return array.astype(np.float16).astype(np.float32)

    half_input = M.cosine_residual(
        value, M.e4m3(M.quadratic_gate_activation(half(value) @ half(expand)))
        @ half(project), cosine)

    device = {
        "x": runtime.buffer_from(value),
        "x16": runtime.buffer(tokens * channels, np.float16),
        "w1": runtime.buffer_from(expand, np.float16),
        "w2": runtime.buffer_from(project, np.float16),
        "cos": runtime.buffer_from(cosine),
        "h": runtime.buffer(tokens * hidden),
        "h16": runtime.buffer(tokens * hidden, np.float16),
        "branch": runtime.buffer(tokens * channels),
        "out": runtime.buffer(tokens * channels),
    }

    def record():
        runtime.begin()
        runtime.to_half(device["x"], device["x16"], tokens * channels)
        runtime.gemm(device["x16"], device["w1"], device["h"], tokens, hidden, channels)
        runtime.gate(device["h"], device["h"], tokens * hidden)
        runtime.e4m3(device["h"], device["h"], tokens * hidden)
        runtime.to_half(device["h"], device["h16"], tokens * hidden)
        runtime.gemm(device["h16"], device["w2"], device["branch"], tokens, channels, hidden)
        runtime.residual(device["branch"], device["x"], device["cos"], device["out"],
                         tokens * channels, channels)
        return runtime.submit()

    passes = record()
    result = device["out"].view(shape=(tokens, channels)).copy()
    check("the chain records as one submit", passes == 7, f"{passes} passes")
    scale = float(np.abs(reference).max())
    plumbing = float(np.abs(result - half_input).max()) / scale
    precision = float(np.abs(result - reference).max()) / scale
    check("matches the reference given the same half inputs", plumbing < 1e-3,
          f"max rel {plumbing:.2e}")
    check("differs from the float32 reference only by the FP16 operands",
          precision < 5e-2, f"max rel {precision:.2e}")

    record()
    started = time.perf_counter()
    for _ in range(5):
        record()
    device_seconds = (time.perf_counter() - started) / 5
    started = time.perf_counter()
    for _ in range(5):
        M.cosine_residual(value, M.e4m3(M.quadratic_gate_activation(value @ expand))
                          @ project, cosine)
    host_seconds = (time.perf_counter() - started) / 5
    print(f"  {tokens} tokens: resident {device_seconds * 1e3:.2f} ms, "
          f"host {host_seconds * 1e3:.2f} ms, {host_seconds / device_seconds:.1f}x")
    for buffer in device.values():
        buffer.free()


def main():
    runtime = xmxres.Runtime()
    test_operators(runtime)
    if WEIGHTS.exists():
        weights, _ = M.load_logical(WEIGHTS)
        test_chain(runtime, weights)
    else:
        print(f"weights not found at {WEIGHTS}; skipping the chain")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("the resident runtime matches the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
