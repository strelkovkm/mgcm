struct MlpParams {
    uint  M;
    uint  N;
    uint  K;
    float salt_guard;
};
[[vk::push_constant]] MlpParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float>   buf0;
[[vk::binding(1, 0)]] StructuredBuffer<float>   buf1;
[[vk::binding(2, 0)]] StructuredBuffer<float>   buf2;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> out_buf;

groupshared float smem_a[64 * 16];
groupshared float smem_b[16 * 64];

float eval_erf_approx(float x) {
    float s = (x < 0.0f) ? -1.0f : 1.0f;
    float a = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - poly * exp(-a * a));
}

void compute_tile_64x64(uint3 gid, uint3 ltid, out float acc[4][4]) {
    uint tid    = ltid.y * 16u + ltid.x;
    uint base_m = gid.y * 64u;
    uint base_n = gid.x * 64u;

    [ForceUnroll]
    for (int m = 0; m < 4; ++m) {
        [ForceUnroll]
        for (int n = 0; n < 4; ++n) acc[m][n] = 0.0f;
    }

    uint load_a_r0 = tid / 16u, load_a_c = tid % 16u;
    uint load_b_r  = tid / 64u, load_b_c = tid % 64u;

    [loop]
    for (uint k0 = 0; k0 < pc.K; k0 += 16u) {
        [ForceUnroll]
        for (uint s = 0; s < 4u; ++s) {
            uint ra = load_a_r0 + s * 16u;
            uint gm = base_m + ra, gk = k0 + load_a_c;
            smem_a[ra * 16u + load_a_c] = (gm < pc.M && gk < pc.K) ? buf0[gm * pc.K + gk] : 0.0f;

            uint rb = load_b_r + s * 4u;
            uint gkb = k0 + rb, gn = base_n + load_b_c;
            smem_b[rb * 64u + load_b_c] = (gkb < pc.K && gn < pc.N) ? buf1[gkb * pc.N + gn] : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < 16u; ++kk) {
            float ra[4], rb[4];
            [ForceUnroll] for (uint m = 0; m < 4u; ++m) ra[m] = smem_a[(ltid.y * 4u + m) * 16u + kk];
            [ForceUnroll] for (uint n = 0; n < 4u; ++n) rb[n] = smem_b[kk * 64u + (ltid.x * 4u + n)];
            [ForceUnroll]
            for (uint m = 0; m < 4u; ++m) {
                [ForceUnroll]
                for (uint n = 0; n < 4u; ++n) acc[m][n] = fma(ra[m], rb[n], acc[m][n]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// 1. Standalone MatMul: A(0), B(1), dummy_sink(2) -> C(3)
[shader("compute")]
[numthreads(16, 16, 1)]
void ora_matmul_only(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    float acc[4][4];
    compute_tile_64x64(gid, ltid, acc);
    float guard = (pc.salt_guard == -12345.0f) ? (buf2[0] * 1.135791f) : 0.0f;
    [ForceUnroll]
    for (uint m = 0; m < 4u; ++m) {
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) {
            uint gm = gid.y * 64u + ltid.y * 4u + m;
            uint gn = gid.x * 64u + ltid.x * 4u + n;
            if (gm < pc.M && gn < pc.N) out_buf[gm * pc.N + gn] = acc[m][n] + guard;
        }
    }
}

// 2. Fused MatMul + Bias (P1): A(0), B(1), bias(2) -> C(3)
[shader("compute")]
[numthreads(16, 16, 1)]
void ora_matmul_bias(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    float acc[4][4];
    compute_tile_64x64(gid, ltid, acc);
    float salt = (pc.salt_guard == -12345.0f) ? 1.135791f : 1.0f;
    [ForceUnroll]
    for (uint m = 0; m < 4u; ++m) {
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) {
            uint gm = gid.y * 64u + ltid.y * 4u + m;
            uint gn = gid.x * 64u + ltid.x * 4u + n;
            if (gm < pc.M && gn < pc.N) out_buf[gm * pc.N + gn] = acc[m][n] + buf2[gn] * salt;
        }
    }
}

// 3. Full Fusion MatMul + Bias + GELU_ERF (P3): A(0), B(1), bias(2) -> Y(3)
[shader("compute")]
[numthreads(16, 16, 1)]
void ora_matmul_bias_gelu(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    float acc[4][4];
    compute_tile_64x64(gid, ltid, acc);
    const float inv_sqrt2 = 0.70710678f;
    [ForceUnroll]
    for (uint m = 0; m < 4u; ++m) {
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) {
            uint gm = gid.y * 64u + ltid.y * 4u + m;
            uint gn = gid.x * 64u + ltid.x * 4u + n;
            if (gm < pc.M && gn < pc.N) {
                float x = acc[m][n] + buf2[gn];
                out_buf[gm * pc.N + gn] = 0.5f * x * (1.0f + eval_erf_approx(x * inv_sqrt2));
            }
        }
    }
}

// 4. Full Fusion MatMul + Bias + ReLU (P3): A(0), B(1), bias(2) -> Y(3)
[shader("compute")]
[numthreads(16, 16, 1)]
void ora_matmul_bias_relu(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    float acc[4][4];
    compute_tile_64x64(gid, ltid, acc);
    float z = (pc.salt_guard == -12345.0f) ? 1.135791f : 0.0f;
    [ForceUnroll]
    for (uint m = 0; m < 4u; ++m) {
        [ForceUnroll]
        for (uint n = 0; n < 4u; ++n) {
            uint gm = gid.y * 64u + ltid.y * 4u + m;
            uint gn = gid.x * 64u + ltid.x * 4u + n;
            if (gm < pc.M && gn < pc.N) out_buf[gm * pc.N + gn] = max(acc[m][n] + buf2[gn], z);
        }
    }
}

// 5. Standalone 1D Add(bias): C(0), bias(1), dummy(2) -> C2(3)
[shader("compute")]
[numthreads(256, 1, 1)]
void ora_elem_add(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x, total = pc.M * pc.N;
    if (idx < total) {
        float g = (pc.salt_guard == -12345.0f) ? (buf2[0] * 1.135791f) : 0.0f;
        out_buf[idx] = buf0[idx] + buf1[idx % pc.N] + g;
    }
}

// 6. Standalone 1D GELU_ERF: C(0), dummy(1), dummy(2) -> Y(3)
[shader("compute")]
[numthreads(256, 1, 1)]
void ora_elem_gelu(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x, total = pc.M * pc.N;
    if (idx < total) {
        float x = buf0[idx] + ((pc.salt_guard == -12345.0f) ? (buf1[0] + buf2[0]) : 0.0f);
        out_buf[idx] = 0.5f * x * (1.0f + eval_erf_approx(x * 0.70710678f));
    }
}

// 7. Standalone 1D ReLU: C(0), dummy(1), dummy(2) -> Y(3)
[shader("compute")]
[numthreads(256, 1, 1)]
void ora_elem_relu(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x, total = pc.M * pc.N;
    if (idx < total) {
        float z = (pc.salt_guard == -12345.0f) ? (buf1[0] + buf2[0] + 1.135791f) : 0.0f;
        out_buf[idx] = max(buf0[idx], z);
    }
}

// 8. Fused 1D Add + GELU_ERF (P2): C(0), bias(1), dummy(2) -> Y(3)
[shader("compute")]
[numthreads(256, 1, 1)]
void ora_elem_add_gelu(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x, total = pc.M * pc.N;
    if (idx < total) {
        float x = buf0[idx] + buf1[idx % pc.N] + ((pc.salt_guard == -12345.0f) ? buf2[0] : 0.0f);
        out_buf[idx] = 0.5f * x * (1.0f + eval_erf_approx(x * 0.70710678f));
    }
}

// 9. Fused 1D Add + ReLU (P2): C(0), bias(1), dummy(2) -> Y(3)
[shader("compute")]
[numthreads(256, 1, 1)]
void ora_elem_add_relu(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x, total = pc.M * pc.N;
    if (idx < total) {
        float z = (pc.salt_guard == -12345.0f) ? (buf2[0] + 1.135791f) : 0.0f;
        out_buf[idx] = max(buf0[idx] + buf1[idx % pc.N], z);
    }
}