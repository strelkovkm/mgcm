#pragma once
#include "mgcm/workload.hpp"
#include <cstdint>

namespace mgcm {

struct MlpConfig {
    uint32_t        batch              = 1;
    uint32_t        M                  = 128;
    uint32_t        N                  = 512;
    uint32_t        K                  = 256;
    bool            broadcast_b        = true; // true: [B,M,K]x[K,N]; false: independent BatchMatMul
    bool            transpose_b        = false;
    bool            is_dynamic_m       = false;
    EpilogueProfile activation         = EpilogueProfile::gelu_erf();
    MatmulSchedule  schedule           = MatmulSchedule::tiled_64x64_smem();
    CompilerKind    frontend           = CompilerKind::GraphCompiler;
    bool            fe_cached          = false;
    bool            base_mm_drv_cached = false;
    uint32_t        instance_count     = 1; // K_same: identical blocks in the model graph
    uint8_t         allowed_plans_mask = kAllPlansMask;
};

class MlpBlock final : public IBlockModel {
public:
    explicit MlpBlock(MlpConfig cfg);

    const MlpConfig& config() const { return cfg_; }
    uint32_t         instance_count() const override { return cfg_.instance_count; }

    std::vector<PartitionWorkload> build_candidates(uint32_t device_wave_quantum) const override;

    static uint32_t estimate_spirv_words(CompilerKind compiler, bool has_matmul, uint32_t num_buffers,
                                         const MatmulSchedule& sched, const EpilogueProfile& ep,
                                         uint32_t out_unroll);

private:
    MlpConfig cfg_;
};

} // namespace mgcm
