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


def test_attention(runtime, weights, block=9, heads=4, height=24, width=32):
    """The whole attention path, both sides starting from the same projection.

    Feeding the reference's float32 projection to the device isolates the port from
    the one real difference — the qkv GEMM's float16 activation — and every stage
    then has to agree exactly.
    """
    print("window attention, stage by stage from a shared projection")
    rng = np.random.default_rng(2)
    prefix = f"block{block}.layer0"
    qkv = weights[f"{prefix}.qkv_weight"]
    projection = weights[f"{prefix}.projection_weight"]
    bias = weights[f"{prefix}.attn_bias"]
    scale = weights[f"{prefix}.attn_scale"]
    channels, size, tokens = projection.shape[0], 8, 64
    windows = (height // size) * (width // size)
    batch = windows * heads
    value = (rng.standard_normal((1, height, width, channels)) * 0.3).astype(np.float32)

    partitioned = M.partition_windows(value, size)
    projected = partitioned @ qkv
    head_shape = (windows, tokens, heads, 32)
    parts = [np.ascontiguousarray(part.reshape(head_shape).transpose(0, 2, 1, 3))
             for part in np.split(projected, 3, axis=-1)]
    query = M.vendor_cosine_publish(parts[0], scale)
    key = M.vendor_cosine_publish(parts[1])
    values = M.e4m3(parts[2])
    scores = (query.reshape(batch, tokens, 32)
              @ key.reshape(batch, tokens, 32).transpose(0, 2, 1))
    scores = scores + np.tile(bias, (windows, 1, 1))
    probabilities = M.vendor_approximate_softmax(scores)
    context = probabilities @ values.reshape(batch, tokens, 32)
    merged = M.e4m3(context.reshape(windows, heads, tokens, 32)
                    .transpose(0, 2, 1, 3).reshape(windows, tokens, channels))
    expected = M.reverse_windows(merged @ projection, batch_count=1, height=height,
                                 width=width, window_size=size)

    n = windows * tokens * channels
    device = {name: runtime.buffer(n) for name in ("q", "k", "v", "merged", "attended")}
    device.update({name: runtime.buffer(n, np.float16) for name in ("q16", "k16", "v16", "m16")})
    device["proj"] = runtime.buffer_from(projected)
    device["scores"] = runtime.buffer(batch * tokens * tokens)
    device["p16"] = runtime.buffer(batch * tokens * tokens, np.float16)
    device["ctx"] = runtime.buffer(batch * tokens * 32)
    device["out"] = runtime.buffer(height * width * channels)
    device["bias"] = runtime.buffer_from(bias)
    device["scale"] = runtime.buffer_from(scale)
    device["wp"] = runtime.buffer_from(projection, np.float16)

    runtime.begin()
    for index, name in enumerate(("q", "k", "v")):
        runtime.split_heads(device["proj"], device[name], windows, tokens, channels,
                            heads, index)
    runtime.cosine_publish(device["q"], device["q"], batch * tokens, tokens=tokens,
                           heads=heads, scale=device["scale"])
    runtime.cosine_publish(device["k"], device["k"], batch * tokens, tokens=tokens,
                           heads=heads)
    runtime.e4m3(device["v"], device["v"], n)
    for name in ("q", "k", "v"):
        runtime.to_half(device[name], device[name + "16"], n)
    runtime.gemm(device["q16"], device["k16"], device["scores"], tokens, tokens, 32,
                 batch=batch, strides=(tokens * 32, tokens * 32, tokens * tokens),
                 transpose_b=True)
    runtime.add_bias(device["scores"], device["bias"], device["scores"],
                     batch * tokens * tokens, tokens, heads)
    runtime.softmax(device["scores"], device["scores"], batch * tokens, tokens)
    runtime.to_half(device["scores"], device["p16"], batch * tokens * tokens)
    runtime.gemm(device["p16"], device["v16"], device["ctx"], tokens, 32, tokens,
                 batch=batch, strides=(tokens * tokens, tokens * 32, tokens * 32))
    runtime.merge_heads(device["ctx"], device["merged"], windows, tokens, channels, heads)
    runtime.e4m3(device["merged"], device["merged"], n)
    runtime.to_half(device["merged"], device["m16"], n)
    runtime.gemm(device["m16"], device["wp"], device["attended"], windows * tokens,
                 channels, channels)
    runtime.reverse(device["attended"], device["out"], height, width, channels, size)
    passes = runtime.submit()
    check("the attention path records as one submit", passes == 19, f"{passes} passes")

    for name, buffer, want in (
            ("cosine publish Q", device["q"], query),
            ("cosine publish K", device["k"], key),
            ("e4m3 V", device["v"], values),
            ("softmax(scores)", device["scores"], probabilities),
            ("context", device["ctx"], context),
            ("merged publish", device["merged"], merged),
            ("attention output", device["out"], expected)):
        check(name, np.array_equal(buffer.view(shape=want.shape), want))
    for buffer in device.values():
        buffer.free()


def test_permutations(runtime):
    print("window and head permutations")
    rng = np.random.default_rng(3)
    height, width, channels, heads = 24, 32, 128, 4
    value = rng.standard_normal((1, height, width, channels)).astype(np.float32)
    source = runtime.buffer_from(value)
    windowed = runtime.buffer(value.size)
    back = runtime.buffer(value.size)
    runtime.begin()
    runtime.partition(source, windowed, height, width, channels)
    runtime.reverse(windowed, back, height, width, channels)
    runtime.submit()
    expected = M.partition_windows(value, 8)
    check("partition_windows", np.array_equal(windowed.view(shape=expected.shape), expected))
    check("reverse_windows round trip",
          np.array_equal(back.view(shape=value.shape), value))

    windows, tokens = expected.shape[0], 64
    projected = rng.standard_normal((windows, tokens, 3 * channels)).astype(np.float32)
    device = runtime.buffer_from(projected)
    parts = [runtime.buffer(windows * tokens * channels) for _ in range(3)]
    merged = runtime.buffer(windows * tokens * channels)
    context = rng.standard_normal((windows, heads, tokens, 32)).astype(np.float32)
    device_context = runtime.buffer_from(context)
    runtime.begin()
    for index, part in enumerate(parts):
        runtime.split_heads(device, part, windows, tokens, channels, heads, index)
    runtime.merge_heads(device_context, merged, windows, tokens, channels, heads)
    runtime.submit()
    for index, (name, part) in enumerate(zip("qkv", np.split(projected, 3, axis=-1))):
        want = part.reshape(windows, tokens, heads, 32).transpose(0, 2, 1, 3)
        check(f"split_heads {name}", np.array_equal(parts[index].view(shape=want.shape), want))
    want = context.transpose(0, 2, 1, 3).reshape(windows, tokens, channels)
    check("merge_heads", np.array_equal(merged.view(shape=want.shape), want))
    for buffer in [source, windowed, back, device, merged, device_context, *parts]:
        buffer.free()


def main():
    runtime = xmxres.Runtime()
    test_operators(runtime)
    test_permutations(runtime)
    if WEIGHTS.exists():
        weights, _ = M.load_logical(WEIGHTS)
        test_chain(runtime, weights)
        test_attention(runtime, weights)
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
