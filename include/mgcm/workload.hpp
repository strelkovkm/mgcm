#pragma once
#include "mgcm/backend.hpp"
#include <cstdint>
#include <vector>

namespace mgcm {

enum class PartitionPlan : uint8_t {
    P0_SplitAll      = 0, // {MatMul} | {Add} | {Act}
    P1_FusedBias     = 1, // {MatMul + Add} | {Act}
    P2_FusedEpilogue = 2, // {MatMul} | {Add + Act}
    P3_FullFusion    = 3  // {MatMul + Add + Act}
};

inline constexpr uint8_t plan_bit(PartitionPlan p) {
    return static_cast<uint8_t>(1u << static_cast<uint8_t>(p));
}
inline constexpr uint8_t kAllPlansMask =
    plan_bit(PartitionPlan::P0_SplitAll) | plan_bit(PartitionPlan::P1_FusedBias) |
    plan_bit(PartitionPlan::P2_FusedEpilogue) | plan_bit(PartitionPlan::P3_FullFusion);

inline const char* to_string(PartitionPlan p) {
    switch (p) {
        case PartitionPlan::P0_SplitAll:      return "P0_SplitAll";
        case PartitionPlan::P1_FusedBias:     return "P1_FusedBias";
        case PartitionPlan::P2_FusedEpilogue: return "P2_FusedEpilogue";
        case PartitionPlan::P3_FullFusion:    return "P3_FullFusion";
    }
    return "Unknown";
}

enum class ActivationPipe : uint8_t { None = 0, GeluErf = 1, GeluTanh = 2 };

struct EpilogueProfile {
    uint32_t       fma_ops_per_elem       = 0;
    ActivationPipe act_pipe               = ActivationPipe::None;
    bool           has_bias_buffer        = false;
    uint32_t       spirv_words_per_scalar = 0;

    static EpilogueProfile none()      { return {0, ActivationPipe::None, false, 0}; }
    static EpilogueProfile bias_only() { return {1, ActivationPipe::None, true, 14}; }
    static EpilogueProfile relu()      { return {1, ActivationPipe::None, false, 6}; }
    static EpilogueProfile silu()      { return {3, ActivationPipe::GeluTanh, false, 45}; }
    static EpilogueProfile gelu_tanh() { return {0, ActivationPipe::GeluTanh, false, 55}; }
    static EpilogueProfile gelu_erf()  { return {0, ActivationPipe::GeluErf, false, 111}; }

    EpilogueProfile operator+(const EpilogueProfile& rhs) const {
        EpilogueProfile out{};
        out.fma_ops_per_elem       = fma_ops_per_elem + rhs.fma_ops_per_elem;
        out.act_pipe               = (rhs.act_pipe != ActivationPipe::None) ? rhs.act_pipe : act_pipe;
        out.has_bias_buffer        = has_bias_buffer || rhs.has_bias_buffer;
        out.spirv_words_per_scalar = spirv_words_per_scalar + rhs.spirv_words_per_scalar;
        return out;
    }
};

struct MatmulSchedule {
    uint32_t tile_m         = 64;
    uint32_t tile_n         = 64;
    uint32_t tile_k         = 16;
    uint32_t reg_m          = 4;
    uint32_t reg_n          = 4;
    uint32_t unroll_k       = 16;   // 16 if inner K-loop is statically unrolled, 1 if rolled
    uint32_t split_k        = 1;
    bool     use_shared_mem = true;

    uint32_t wg_threads() const { return (tile_m / reg_m) * (tile_n / reg_n); }

    static MatmulSchedule tiled_64x64_smem()  { return {64, 64, 16, 4, 4, 16, 1, true}; }
    static MatmulSchedule tiled_64x64_cache() { return {64, 64, 1, 4, 4, 1, 1, false}; }
    static MatmulSchedule tiled_32x32_smem()  { return {32, 32, 16, 2, 2, 16, 1, true}; }
    static MatmulSchedule naive_untiled()     { return {16, 16, 1, 1, 1, 1, 1, false}; }
};

struct KernelWorkload {
    CompilerKind   frontend                 = CompilerKind::GraphCompiler;
    bool           fe_cache_miss            = true;
    bool           drv_cache_miss           = true;
    uint32_t       est_spirv_words          = 0;

    uint32_t       workgroups               = 1;
    double         flops_contraction_padded = 0.0; // Uses MB4 GEMM pipe
    double         flops_vector_fma_padded  = 0.0; // Uses MB5a FMA pipe
    double         elems_act_padded         = 0.0; // Uses MB5b GElem/s pipe (1 call/element)
    ActivationPipe act_pipe                 = ActivationPipe::None;

    bool           uses_shared_mem          = false;
    bool           has_unaligned_tail       = false;
    bool           has_fused_heavy_act      = false;

    double         bytes_cold               = 0.0;
    double         cold_working_set_bytes   = 0.0;
    double         bytes_reuse              = 0.0;
    double         reuse_working_set_bytes  = 0.0;
    uint32_t       read_stride_elems        = 1;
};

struct PartitionWorkload {
    PartitionPlan               plan            = PartitionPlan::P0_SplitAll;
    std::vector<KernelWorkload> kernels;
    uint32_t                    host_sync_count = 0;
    double                      ext_cut_bytes   = 0.0;
};

class IBlockModel {
public:
    virtual ~IBlockModel() = default;
    virtual uint32_t                       instance_count() const = 0;
    virtual std::vector<PartitionWorkload> build_candidates(uint32_t device_wave_quantum) const = 0;
};

} // namespace mgcm
