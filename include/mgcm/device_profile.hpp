#pragma once
#include "mgcm/backend.hpp"
#include <cstdint>
#include <vector>

namespace mgcm {

inline constexpr uint32_t kDeviceProfileSchemaVersion = 2;

struct BandwidthPoint {
    size_t working_set_bytes = 0;
    double bw_median_gbps    = 0.0;
    double bw_iqr_gbps       = 0.0;
};

struct WaveSaturationPoint {
    uint32_t workgroups  = 0;
    double   gflops      = 0.0;
    double   gpu_time_us = 0.0;
};

struct DispatchSamplePoint {
    uint32_t chain_len    = 0;
    double   nobar_gpu_us = 0.0;
    double   bar_gpu_us   = 0.0;
};

struct CompileSamplePoint {
    uint32_t spirv_words = 0;
    double   time_ms     = 0.0;
};

struct StageTimingsMs {
    double init_prng_ms    = 0.0;
    double mb0_warmup_ms   = 0.0;
    double mb1_dispatch_ms = 0.0;
    double mb2_cache_ms    = 0.0;
    double mb3_stride_ms   = 0.0;
    double mb4_tiled_ms    = 0.0;
    double mb5_compute_ms  = 0.0;
    double mb6_drv_ms      = 0.0;
    double mb6_fe_ms       = 0.0;
};

struct DeviceProfile {
    uint32_t          schema_version           = kDeviceProfileSchemaVersion;
    double            calibration_wall_time_ms = 0.0;
    StageTimingsMs    stage_timings{};
    StaticDeviceProps props;

    // Stage 1 (MB6): Frontend & Driver Compilation OLS Parameters (Fields 1-9)
    double                          alpha_0_slang_ms          = 0.0;
    double                          alpha_1_slang_ms_per_word = 0.0;
    double                          alpha_0_gc_ms             = 0.0;
    double                          alpha_1_gc_ms_per_word    = 0.0;
    double                          beta_0_drv_ms             = 0.0;
    double                          beta_1_drv_ms_per_word    = 0.0;
    double                          beta_0_drv_smem_ms        = 0.0;
    double                          beta_1_drv_smem_ms_per_word = 0.0;
    double                          t_cache_hit_ms            = 0.0;
    double                          r2_fe_slang               = 0.0;
    double                          r2_fe_gc                  = 0.0;
    double                          r2_drv                    = 0.0;
    std::vector<CompileSamplePoint> mb6_drv_points;
    std::vector<CompileSamplePoint> mb6_fe_points;

    // Stage 2 (MB1): Dispatch & Synchronization Overheads (Fields 10-12)
    double t_submit_cpu_us  = 0.0;
    double t_launch_gpu_us  = 0.0;
    double t_barrier_gpu_us = 0.0;
    double                           r2_dispatch = 0.0;
    std::vector<DispatchSamplePoint> mb1_points;

    // Stage 2 (MB2): Memory Hierarchy Curve & Summary (Fields 13-16)
    std::vector<BandwidthPoint> bw_curve;
    size_t                      l2_effective_bytes = 0;
    double                      bw_l2_gbps         = 0.0;
    double                      bw_dram_gbps       = 0.0;

    // Stage 2 (MB3): Strided Coalescing Factors (Fields 17-18)
    double bw_l2_strided32_gbps   = 0.0;
    double coal_factor_l2_s32     = 1.0;
    double bw_dram_strided32_gbps = 0.0;
    double coal_factor_dram_s32   = 1.0;

    // Stage 2 (MB4): 2D-Tiled GEMM, Tail Predication & Fusion Register Pressure (Fields 19-23)
    double gflops_matmul_smem    = 0.0;
    double gflops_matmul_cache   = 0.0;
    double smem_vs_cache_speedup = 1.0;
    bool   prefer_shared_mem     = true;
    double tail_efficiency       = 1.0;
    double fusion_reg_factor     = 1.0;

    // Stage 2 (MB5): Compute Roofs & Hardware Wave Quantization (Field 24)
    double                           pi_fma_gflops        = 0.0;
    double                           pi_gelu_erf_gelem_s  = 0.0;
    double                           pi_gelu_tanh_gelem_s = 0.0;
    uint32_t                         g_wave_quantum       = 1;
    double                           pi_wave1_gflops      = 0.0;
    uint32_t                         g_half_workgroups    = 1;
    uint32_t                         g_sat_workgroups     = 1;
    std::vector<WaveSaturationPoint> wave_curve;
};

} // namespace mgcm
