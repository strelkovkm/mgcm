struct ComputeParams {
    uint  iters;
    float alpha;
    float beta;
    float impossible_sink;
};
[[vk::push_constant]] ComputeParams pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float> out_buf;

float eval_erf_approx(float x) {
    float s = (x < 0.0f) ? -1.0f : 1.0f;
    float a = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - poly * exp(-a * a));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mb5_fma16(uint3 tid : SV_DispatchThreadID) {
    float acc[16];
    [ForceUnroll]
    for (int k = 0; k < 16; ++k) {
        acc[k] = float(tid.x & 255u) * 1e-5f + float(k) * 0.05f + 1.012345f;
    }

    [loop]
    for (uint i = 0; i < pc.iters; ++i) {
        [ForceUnroll]
        for (int k = 0; k < 16; ++k) {
            acc[k] = fma(acc[k], pc.alpha, pc.beta);
        }
    }

    precise float sum = 0.0f;
    [ForceUnroll]
    for (int k = 0; k < 16; ++k) sum += acc[k];
    if (sum == pc.impossible_sink) out_buf[tid.x] = sum;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mb5_gelu_tanh(uint3 tid : SV_DispatchThreadID) {
    const float kSqrt2OverPi = 0.79788456f;
    const float kCubic       = 0.044715f;
    float acc[8];
    [ForceUnroll]
    for (int k = 0; k < 8; ++k) {
        acc[k] = 0.2f + float((tid.x + uint(k)) & 15u) * 0.03f;
    }

    [loop]
    for (uint i = 0; i < pc.iters; ++i) {
        [ForceUnroll]
        for (int k = 0; k < 8; ++k) {
            float x   = acc[k];
            float arg = kSqrt2OverPi * fma(kCubic, x * x * x, x);
            float g   = 0.5f * x * (1.0f + tanh(arg));
            // Contractive affine feedback keeps x in (0.25, 0.75) without denormals or saturation
            acc[k]    = fma(g, pc.alpha, pc.beta);
        }
    }

    precise float sum = 0.0f;
    [ForceUnroll]
    for (int k = 0; k < 8; ++k) sum += acc[k];
    if (sum == pc.impossible_sink) out_buf[tid.x] = sum;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mb5_gelu_erf(uint3 tid : SV_DispatchThreadID) {
    const float kInvSqrt2 = 0.70710678f;
    float acc[8];
    [ForceUnroll]
    for (int k = 0; k < 8; ++k) {
        acc[k] = 0.2f + float((tid.x + uint(k)) & 15u) * 0.03f;
    }

    [loop]
    for (uint i = 0; i < pc.iters; ++i) {
        [ForceUnroll]
        for (int k = 0; k < 8; ++k) {
            float x = acc[k];
            float g = 0.5f * x * (1.0f + eval_erf_approx(x * kInvSqrt2));
            acc[k]  = fma(g, pc.alpha, pc.beta);
        }
    }

    precise float sum = 0.0f;
    [ForceUnroll]
    for (int k = 0; k < 8; ++k) sum += acc[k];
    if (sum == pc.impossible_sink) out_buf[tid.x] = sum;
}
