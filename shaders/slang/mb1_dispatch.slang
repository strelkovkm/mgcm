struct NoopParams {
    uint  flag;
    float impossible_sink;
};
[[vk::push_constant]] NoopParams npc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float> noop_out;

[shader("compute")]
[numthreads(64, 1, 1)]
void mb1_noop(uint3 tid : SV_DispatchThreadID) {
    // Dirty 1 cache line per subgroup so RAW barriers measure real GL2/L2 writeback & invalidation
    if ((tid.x & 15u) == 0u) {
        noop_out[tid.x] = float(npc.flag) * 1.0001f + ((npc.impossible_sink == -12345.0f) ? 0.5f : 0.0f);
    }
}
