struct WarmupParams {
    uint  iters;
    float alpha;
    float beta;
    float impossible_sink;
};
[[vk::push_constant]] WarmupParams wpc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float> warm_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void mb0_warmup(uint3 tid : SV_DispatchThreadID) {
    float acc[8];
    [ForceUnroll]
    for (int k = 0; k < 8; ++k) acc[k] = float(tid.x) * 1e-6f + float(k) * 0.1f;

    [loop]
    for (uint i = 0; i < wpc.iters; ++i) {
        [ForceUnroll]
        for (int k = 0; k < 8; ++k) acc[k] = fma(acc[k], wpc.alpha, wpc.beta);
    }
    precise float sum = (acc[0] + acc[1]) + (acc[2] + acc[3])
                      + (acc[4] + acc[5]) + (acc[6] + acc[7]);
    if (sum == wpc.impossible_sink) warm_out[tid.x] = sum;
}
