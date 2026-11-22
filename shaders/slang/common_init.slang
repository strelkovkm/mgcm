struct InitParams {
    uint num_vec4;
    uint total_threads;
    uint seed;
};
[[vk::push_constant]] InitParams pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> out_buf;

uint xorshift32(uint x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
}
float to_normal_float(uint u) {
    return asfloat(0x3F800000u | (u & 0x007FFFFFu)) - 0.5f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void init_prng_vec4(uint3 tid : SV_DispatchThreadID) {
    [loop]
    for (uint idx = tid.x; idx < pc.num_vec4; idx += pc.total_threads) {
        uint s = xorshift32(idx * 747796405u + pc.seed + 2891336453u);
        float x = to_normal_float(s); s = xorshift32(s);
        float y = to_normal_float(s); s = xorshift32(s);
        float z = to_normal_float(s); s = xorshift32(s);
        float w = to_normal_float(s);
        out_buf[idx] = float4(x, y, z, w);
    }
}
