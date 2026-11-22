#pragma once
#include "mgcm/backend.hpp"
#include <vector>

namespace mgcm {

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

struct DeviceProfile {
    StaticDeviceProps props;

    // MB1: Dispatch & Synchronization overheads
    double t_submit_cpu_us  = 0.0;
    double t_launch_gpu_us  = 0.0;
    double t_barrier_gpu_us = 0.0;

    // MB2: Cache Hierarchy & Memory Bandwidth
    size_t l2_effective_bytes = 0;
    double bw_l2_gbps         = 0.0;
    double bw_dram_gbps       = 0.0;
    std::vector<BandwidthPoint> bw_curve;

    // MB3: Uncoalesced / Strided Access Penalty (stride = 32 floats = 128 B)
    double bw_dram_strided32_gbps = 0.0;
    double coal_factor_dram_s32   = 1.0;
    double bw_l2_strided32_gbps   = 0.0;
    double coal_factor_l2_s32     = 1.0;

    // MB4: 2D-Tiled MatMul (64x64) Shared Memory vs Direct L1/L2 Cache
    double gflops_matmul_smem   = 0.0;
    double gflops_matmul_cache  = 0.0;
    double smem_vs_cache_speedup = 1.0;
    bool   prefer_shared_mem    = true;

    // MB5: Compute Roofs & Wave Saturation
    double   pi_fma_gflops         = 0.0;
    double   pi_gelu_tanh_gelem_s  = 0.0;
    double   pi_gelu_erf_gelem_s   = 0.0;
    uint32_t g_wave_quantum    = 1;
    uint32_t g_half_workgroups = 1;
    uint32_t g_sat_workgroups  = 1;
    std::vector<WaveSaturationPoint> wave_curve;

    // MB6: Uncached Driver Compilation + Lazy First-Infer JIT
    double t_compile_small_cpu_ms      = 0.0;
    double t_compile_small_lazy_ms     = 0.0;
    double t_compile_tiled_cpu_ms      = 0.0;
    double t_compile_tiled_lazy_ms     = 0.0;
    double t_compile_fused_erf_cpu_ms  = 0.0;
    double t_compile_fused_erf_lazy_ms = 0.0;
};

} // namespace mgcm
