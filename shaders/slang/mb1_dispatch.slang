struct NoopParams {
    uint  flag;
    float impossible_sink;
};
[[vk::push_constant]] NoopParams npc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float> noop_out;

[shader("compute")]
[numthreads(64, 1, 1)]
void mb1_noop(uint3 tid : SV_DispatchThreadID) {
    if (float(tid.x + npc.flag) == npc.impossible_sink) noop_out[tid.x] = 1.0f;
}
