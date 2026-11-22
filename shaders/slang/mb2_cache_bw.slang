struct StreamParams {
    uint  num_vec4;
    uint  steps;
    uint  num_groups;
    float impossible_sink;
};
[[vk::push_constant]] StreamParams pc;
[[vk::binding(0, 0)]] StructuredBuffer<float4>   src_buf;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> dst_sink;

[shader("compute")]
[numthreads(256, 1, 1)]
void mb2_stream_vec4(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    // Stagger workgroup start offsets across [0, num_vec4) aligned to 256 vec4 (4 KB)
    uint wg_offset = uint(((uint64_t(gid.x) * uint64_t(pc.num_vec4)) / uint64_t(pc.num_groups)) & ~255ull);
    uint idx = (wg_offset + ltid.x) % pc.num_vec4;
    float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);

    [loop]
    for (uint s = 0; s < pc.steps; ++s) {
        acc += src_buf[idx];
        idx += 256u;
        if (idx >= pc.num_vec4) {
            idx -= pc.num_vec4;
        }
    }

    precise float sum = (acc.x + acc.y) + (acc.z + acc.w);
    if (sum == pc.impossible_sink) {
        dst_sink[dtid.x] = acc;
    }
}
