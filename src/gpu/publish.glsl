/* The vendor's rounding points, shared by every shader that has to reproduce them.
 *
 * `precise` is not decoration here: the gate rounds to half between the multiply and
 * the add, and an FMA contraction would skip that rounding and give a different answer
 * from the reference.
 */
#ifndef PUBLISH_GLSL
#define PUBLISH_GLSL

/* The hardware float16 conversion, in whichever spelling the compiler keeps. Mesa folds
 * `float(float16_t(x))` away, leaving the value in float32 — the bug that once made every
 * vendor rounding point in this graph silently vanish — and keeps packHalf2x16's round
 * trip, which changes the bit representation. Intel's Windows compiler does the opposite:
 * it folds the round trip to x and keeps the cast. Where it survives, each is bit-exact
 * against numpy's float16 over ordinary values, half subnormals and overflow to infinity
 * (`src/bench/half_probe.py`), in two instructions and no branches, where doing the
 * exponent and mantissa by hand took ten and two branches. libxmx sets constant 1 from
 * the driver (`xmx_init`, `XMX_HALF_ROUND`). */
layout(constant_id = 1) const bool HALF_BY_CAST = false;
float half_round(float x) {
    return HALF_BY_CAST ? float(float16_t(x)) : unpackHalf2x16(packHalf2x16(vec2(x, 0.0))).x;
}

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
