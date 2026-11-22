struct StreamParams {
    uint  num_blocks;
    uint  passes_per_sweep;
    uint  sweeps;
    float impossible_sink;
};
[[vk::push_constant]] StreamParams pc;
[[vk::binding(0, 0)]] StructuredBuffer<float4>   src_buf;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> dst_sink;

[shader("compute")]
[numthreads(256, 1, 1)]
void mb2_stream_vec4(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);

    [loop]
    for (uint r = 0; r < pc.sweeps; ++r) {
        uint base_blk = (gid.x + r * 17u) % pc.num_blocks;
        [loop]
        for (uint p = 0; p < pc.passes_per_sweep; ++p) {
            uint blk = base_blk + p * 128u;
            if (blk >= pc.num_blocks) {
                blk -= pc.num_blocks;
            }
            acc += src_buf[blk * 256u + ltid.x];
        }
    }

    precise float sum = (acc.x + acc.y) + (acc.z + acc.w);
    if (sum == pc.impossible_sink) {
        dst_sink[dtid.x] = acc;
    }
}
