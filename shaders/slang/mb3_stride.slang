struct StrideParams {
    uint  num_tiles;
    uint  passes_per_sweep;
    uint  sweeps;
    uint  stride;
    float salt_scale;
    float impossible_sink;
};
[[vk::push_constant]] StrideParams pc;
[[vk::binding(0, 0)]] StructuredBuffer<float>   src_buf;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst_sink;

[shader("compute")]
[numthreads(256, 1, 1)]
void mb3_stride_f32(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    float acc = 0.0f * (1.1234567f + pc.salt_scale);
    uint tile_span = 256u * pc.stride;

    [loop]
    for (uint r = 0; r < pc.sweeps; ++r) {
        uint base_tile = (gid.x + r * 17u) % pc.num_tiles;
        uint col = r % pc.stride;
        [loop]
        for (uint p = 0; p < pc.passes_per_sweep; ++p) {
            uint tile = base_tile + p * 128u;
            if (tile >= pc.num_tiles) {
                tile -= pc.num_tiles;
            }
            acc += src_buf[tile * tile_span + ltid.x * pc.stride + col];
        }
    }

    if (acc == pc.impossible_sink) {
        dst_sink[dtid.x] = acc;
    }
}
