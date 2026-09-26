/* The vendor's rounding points, shared by every shader that has to reproduce them.
 *
 * `precise` is not decoration here: the gate rounds to half between the multiply and
 * the add, and an FMA contraction would skip that rounding and give a different answer
 * from the reference.
 */
#ifndef PUBLISH_GLSL
#define PUBLISH_GLSL

/* The hardware float16 conversion. `float(float16_t(x))` is the obvious spelling and
 * the compiler folds it away, leaving the value in float32 — the bug that once made
 * every vendor rounding point in this graph silently vanish. `packHalf2x16` changes
 * the bit representation so it cannot be elided, and it is bit-exact against numpy's
 * float16 over ordinary values, half subnormals and overflow to infinity
 * (`src/bench/half_probe.py`): two instructions and no branches, where doing the
 * exponent and mantissa by hand took ten and two branches.
 *
 * That is the default and it stays the default: on Mesa the other spelling is folded
 * away, so it is not a substitute there — it is only a fallback for a driver that
 * cannot run this one at all. The B580's Windows driver (101.8993) is such a driver:
 * `packHalf2x16` makes it lose the device (`VK_ERROR_DEVICE_LOST`). A build for it
 * defines HALF_ROUND_FLOAT16, where `float(float16_t(x))` is used instead. On the
 * B580 the two agree bit for bit — 0 differences over 12020 values covering ordinary
 * values, subnormals, NaN and Inf (`src/bench/half_probe.py`) — so this changes which
 * instruction runs, not the number it produces. */
#ifdef HALF_ROUND_FLOAT16
float half_round(float x) { return float(float16_t(x)); }
#else
float half_round(float x) { return unpackHalf2x16(packHalf2x16(vec2(x, 0.0))).x; }
#endif

/* Packing two floats into a half pair, and reading them back.
 *
 * `packHalf2x16` is not only used for rounding: the softmax's exponential is a bit
 * trick on the packed word - `(packHalf2x16(affine) << 5) + 0x7ff88000u`, then read
 * back - in attention.comp, global_attention.comp, window_attention.comp and
 * window_block.comp. That trick needs the *sixteen bits*, not just a rounded value, so
 * a driver that produces the wrong bits breaks every attention path at once rather
 * than one rounding point.
 *
 * The B580's Windows driver (101.8993) is such a driver. Measured here by the daemon's
 * own start-up probe: packHalf2x16 disagrees with float16 on 90109 of 90368 values,
 * while the bit-twiddled conversion and `float16_t` agree on all of them. With that
 * driver the graph's output is NaN, which is what the probe exists to catch.
 *
 * So a build for such a driver defines PACK_HALF_BITS, and both spells below are done
 * by hand instead. The rounding is the same arithmetic the reference uses - normal
 * halves keep ten mantissa bits with ties to even, subnormals are multiples of 2^-24,
 * overflow to infinity, NaN keeps its sign and top payload bits - and the packing is
 * exact, so the bit trick still lands where it did on a driver that works. */
#ifdef PACK_HALF_BITS
uint half16_bits(float x) {
    uint f = floatBitsToUint(x);
    uint sign = (f >> 16u) & 0x8000u;
    uint a = f & 0x7fffffffu;
    /* Inf, NaN: the payload's top bits survive, quieted. */
    if (a > 0x7f800000u) return 0x7e00u | ((a >> 13u) & 0x1ffu);
    if (a >= 0x477ff000u) return 0x7c00u;              /* >= 65520: infinity */
    uint normal = (a + 0x0fffu + ((a >> 13u) & 1u)) >> 13u;
    if (a >= 0x38800000u) return sign | (normal - 0x1c000u);
    /* Subnormal: the float adder's own rounding finds the multiple of 2^-24 at 0.5. */
    float magnitude = uintBitsToFloat(a);
    float tiny = (magnitude + 0.5f) - 0.5f;
    return sign | (floatBitsToUint(tiny) >> 13u);
}

float half16_value(uint h) {
    uint sign = (h & 0x8000u) << 16u;
    uint e = (h >> 10u) & 0x1fu, m = h & 0x3ffu;
    if (e == 0u) {
        if (m == 0u) return uintBitsToFloat(sign);      /* signed zero */
        /* Subnormal: shift up until the implicit bit appears; half's exponent is then
         * 1 - shift, and float32's bias is 112 more, so 113 - shift. */
        uint shift = 0u;
        while ((m & 0x400u) == 0u) { m <<= 1u; shift++; }
        m &= 0x3ffu;
        return uintBitsToFloat(sign | ((113u - shift) << 23u) | (m << 13u));
    }
    if (e == 0x1fu) return uintBitsToFloat(sign | 0x7f800000u | (m << 13u));
    return uintBitsToFloat(sign | ((e + 112u) << 23u) | (m << 13u));
}

uint pack_half2(vec2 v) { return half16_bits(v.x) | (half16_bits(v.y) << 16u); }
vec2 unpack_half2(uint w) { return vec2(half16_value(w & 0xffffu),
                                        half16_value(w >> 16u)); }
#else
uint pack_half2(vec2 v) { return packHalf2x16(v); }
vec2 unpack_half2(uint w) { return unpackHalf2x16(w); }
#endif

float e4m3(float x) {
    float magnitude = min(abs(x), 448.0);
    int exponent = (floatBitsToInt(magnitude) >> 23) & 0xFF;
    exponent = max(exponent, 121) - 3;                 // 121-3 = the 2^-9 subnormal step
    float step = intBitsToFloat(exponent << 23);
    float reciprocal = intBitsToFloat((254 - exponent) << 23);
    float rounded = roundEven(magnitude * reciprocal) * step;
    return x < 0.0 ? -rounded : rounded;
}

float gate_activation(float x) {
    precise float wide = half_round(x);
    precise float clamped = clamp(wide, -4.0, 4.0);
    precise float linear = abs(clamped) * -0.055908203125;
    linear += 0.447265625;
    linear = half_round(linear);
    linear *= clamped;
    linear += 0.89453125;
    linear = half_round(linear);
    return half_round(wide * linear);
}

/* The publish an epilogue applies: bits 8-11 of a pass's `flags` pick the transform. */
float publish(uint epilogue, float value) {
    if (epilogue == 2u || epilogue == 3u) value = gate_activation(value);
    if (epilogue == 1u || epilogue == 3u) value = e4m3(value);
    if (epilogue == 4u) value = half_round(value);
    return value;
}

#endif
