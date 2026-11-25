struct MatmulParams {
    uint  M;
    uint  N;
    uint  K;
    float impossible_sink;
};
[[vk::push_constant]] MatmulParams pc;
[[vk::binding(0, 0)]] StructuredBuffer<float>   mat_a;
[[vk::binding(1, 0)]] StructuredBuffer<float>   mat_b;
[[vk::binding(2, 0)]] StructuredBuffer<float>   bias_vec;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> mat_c;

groupshared float smem_a[64 * 16];
groupshared float smem_b[16 * 64];

float eval_erf_approx(float x) {
    // Abramowitz & Stegun 7.1.26 rational polynomial (matches MLIR math.erf lowering)
    float s = (x < 0.0f) ? -1.0f : 1.0f;
    float a = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - poly * exp(-a * a));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void mb4_matmul_smem(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    uint tid = ltid.y * 16u + ltid.x; // 0..255
    uint base_m = gid.y * 64u;
    uint base_n = gid.x * 64u;

    float acc[4][4];
    [ForceUnroll]
    for (int m = 0; m < 4; ++m) {
        [ForceUnroll]
        for (int n = 0; n < 4; ++n) {
            acc[m][n] = 0.0f;
        }
    }

    uint load_a_r0 = tid / 16u;        // 0..15
    uint load_a_c  = tid % 16u;        // 0..15
    uint load_b_r  = tid / 64u;        // 0..3
    uint load_b_c  = tid % 64u;        // 0..63

    [loop]
    for (uint k0 = 0; k0 < pc.K; k0 += 16u) {
        [ForceUnroll]
        for (uint s = 0; s < 4u; ++s) {
            uint row_a = load_a_r0 + s * 16u;
            smem_a[row_a * 16u + load_a_c] = mat_a[(base_m + row_a) * pc.K + (k0 + load_a_c)];
            uint row_b = load_b_r + s * 4u;
            smem_b[row_b * 64u + load_b_c] = mat_b[(k0 + row_b) * pc.N + (base_n + load_b_c)];
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < 16u; ++kk) {
            float reg_a[4];
            float reg_b[4];
            [ForceUnroll]
            for (uint m = 0; m < 4u; ++m) reg_a[m] = smem_a[(ltid.y * 4u + m) * 16u + kk];
            [ForceUnroll]
            for (uint n = 0; n < 4u; ++n) reg_b[n] = smem_b[kk * 64u + (ltid.x * 4u + n)];
            [ForceUnroll]
            for (uint m = 0; m < 4u; ++m) {
                [ForceUnroll]
                for (uint n = 0; n < 4u; ++n) {
                    acc[m][n] = fma(reg_a[m], reg_b[n], acc[m][n]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float salt_guard = 1.135791f;
    [ForceUnroll]
    for (uint m = 0; m < 4u; ++m) {
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) {
            uint gm = base_m + ltid.y * 4u + m;
            uint gn = base_n + ltid.x * 4u + n;
            float v = acc[m][n] + ((pc.impossible_sink == -12345.0f) ? (bias_vec[gn] * salt_guard) : 0.0f);
            mat_c[gm * pc.N + gn] = v;
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void mb4_matmul_cache(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    uint base_m = gid.y * 64u;
    uint base_n = gid.x * 64u;

    float acc[4][4];
    [ForceUnroll]
    for (int m = 0; m < 4; ++m) {
        [ForceUnroll]
        for (int n = 0; n < 4; ++n) {
            acc[m][n] = 0.0f;
        }
    }

    [loop]
    for (uint kk = 0; kk < pc.K; ++kk) {
        float reg_a[4];
        float reg_b[4];
        [ForceUnroll]
        for (uint m = 0; m < 4u; ++m) reg_a[m] = mat_a[(base_m + ltid.y * 4u + m) * pc.K + kk];
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) reg_b[n] = mat_b[kk * pc.N + (base_n + ltid.x * 4u + n)];
        [ForceUnroll]
        for (uint m = 0; m < 4u; ++m) {
            [ForceUnroll]
            for (uint n = 0; n < 4u; ++n) {
                acc[m][n] = fma(reg_a[m], reg_b[n], acc[m][n]);
            }
        }
    }

    float salt_guard = 1.246813f;
    [ForceUnroll]
    for (uint m = 0; m < 4u; ++m) {
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) {
            uint gm = base_m + ltid.y * 4u + m;
            uint gn = base_n + ltid.x * 4u + n;
            float v = acc[m][n] + ((pc.impossible_sink == -12345.0f) ? (bias_vec[gn] * salt_guard) : 0.0f);
            mat_c[gm * pc.N + gn] = v;
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void mb6_matmul_fused_gelu_erf(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    uint tid = ltid.y * 16u + ltid.x;
    uint base_m = gid.y * 64u;
    uint base_n = gid.x * 64u;

    float acc[4][4];
    [ForceUnroll]
    for (int m = 0; m < 4; ++m) {
        [ForceUnroll]
        for (int n = 0; n < 4; ++n) {
            acc[m][n] = 0.0f;
        }
    }

    uint load_a_r0 = tid / 16u;
    uint load_a_c  = tid % 16u;
    uint load_b_r  = tid / 64u;
    uint load_b_c  = tid % 64u;

    [loop]
    for (uint k0 = 0; k0 < pc.K; k0 += 16u) {
        [ForceUnroll]
        for (uint s = 0; s < 4u; ++s) {
            uint row_a = load_a_r0 + s * 16u;
            smem_a[row_a * 16u + load_a_c] = mat_a[(base_m + row_a) * pc.K + (k0 + load_a_c)];
            uint row_b = load_b_r + s * 4u;
            smem_b[row_b * 64u + load_b_c] = mat_b[(k0 + row_b) * pc.N + (base_n + load_b_c)];
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < 16u; ++kk) {
            float reg_a[4];
            float reg_b[4];
            [ForceUnroll]
            for (uint m = 0; m < 4u; ++m) reg_a[m] = smem_a[(ltid.y * 4u + m) * 16u + kk];
            [ForceUnroll]
            for (uint n = 0; n < 4u; ++n) reg_b[n] = smem_b[kk * 64u + (ltid.x * 4u + n)];
            [ForceUnroll]
            for (uint m = 0; m < 4u; ++m) {
                [ForceUnroll]
                for (uint n = 0; n < 4u; ++n) {
                    acc[m][n] = fma(reg_a[m], reg_b[n], acc[m][n]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // 16x unrolled Add(bias) + GELU(ERF) epilogue in registers (realistic register/SSA pressure!)
    const float inv_sqrt2 = 0.70710678f;
    [ForceUnroll]
    for (uint m = 0; m < 4u; ++m) {
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) {
            uint gm = base_m + ltid.y * 4u + m;
            uint gn = base_n + ltid.x * 4u + n;
            float x = acc[m][n] + bias_vec[gn];
            float y = 0.5f * x * (1.0f + eval_erf_approx(x * inv_sqrt2));
            mat_c[gm * pc.N + gn] = y;
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void mb4_matmul_smem_tail(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    uint tid    = ltid.y * 16u + ltid.x;
    uint base_m = gid.y * 64u;
    uint base_n = gid.x * 64u;

    float acc[4][4];
    [ForceUnroll]
    for (int m = 0; m < 4; ++m) {
        [ForceUnroll]
        for (int n = 0; n < 4; ++n) {
            acc[m][n] = 0.0f;
        }
    }

    uint load_a_r0 = tid / 16u;
    uint load_a_c  = tid % 16u;
    uint load_b_r  = tid / 64u;
    uint load_b_c  = tid % 64u;

    [loop]
    for (uint k0 = 0; k0 < pc.K; k0 += 16u) {
        [ForceUnroll]
        for (uint s = 0; s < 4u; ++s) {
            uint row_a = load_a_r0 + s * 16u;
            uint gm_a  = base_m + row_a;
            uint gk_a  = k0 + load_a_c;
            smem_a[row_a * 16u + load_a_c] = (gm_a < pc.M && gk_a < pc.K) ? mat_a[gm_a * pc.K + gk_a] : 0.0f;

            uint row_b = load_b_r + s * 4u;
            uint gk_b  = k0 + row_b;
            uint gn_b  = base_n + load_b_c;
            smem_b[row_b * 64u + load_b_c] = (gk_b < pc.K && gn_b < pc.N) ? mat_b[gk_b * pc.N + gn_b] : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < 16u; ++kk) {
            float reg_a[4];
            float reg_b[4];
            [ForceUnroll]
            for (uint m = 0; m < 4u; ++m) reg_a[m] = smem_a[(ltid.y * 4u + m) * 16u + kk];
            [ForceUnroll]
            for (uint n = 0; n < 4u; ++n) reg_b[n] = smem_b[kk * 64u + (ltid.x * 4u + n)];
            [ForceUnroll]
            for (uint m = 0; m < 4u; ++m) {
                [ForceUnroll]
                for (uint n = 0; n < 4u; ++n) {
                    acc[m][n] = fma(reg_a[m], reg_b[n], acc[m][n]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float salt_guard = 1.135791f;
    [ForceUnroll]
    for (uint m = 0; m < 4u; ++m) {
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) {
            uint gm = base_m + ltid.y * 4u + m;
            uint gn = base_n + ltid.x * 4u + n;
            if (gm < pc.M && gn < pc.N) {
                float v = acc[m][n] + ((pc.impossible_sink == -12345.0f) ? (bias_vec[gn] * salt_guard) : 0.0f);
                mat_c[gm * pc.N + gn] = v;
            }
        }
    }
}
