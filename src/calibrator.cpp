#include "mgcm/calibrator.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace mgcm {
namespace {

struct RobustStats {
    double median = 0.0;
    double iqr    = 0.0;
};

RobustStats calc_median_iqr(std::vector<double> v) {
    if (v.empty()) {
        return {};
    }
    std::sort(v.begin(), v.end());
    auto pct = [&](double q) {
        double pos = q * static_cast<double>(v.size() - 1);
        auto lo    = static_cast<size_t>(std::floor(pos));
        auto hi    = static_cast<size_t>(std::ceil(pos));
        double f   = pos - static_cast<double>(lo);
        return v[lo] * (1.0 - f) + v[hi] * f;
    };
    return {pct(0.50), pct(0.75) - pct(0.25)};
}

} // namespace

Calibrator::Calibrator(ICalibBackend& backend, CalibratorConfig cfg) : backend_(backend), cfg_(cfg) {}

void Calibrator::run_warmup() {
    BufferHandle buf = backend_.allocate_buffer(64 * 1024);
    KernelHandle krn = backend_.compile_kernel(BenchKernelId::MB0_Warmup, 0);

    std::array<BufferHandle, 1> bufs{buf};
    std::array<Scalar32, 4> scalars{
        Scalar32::U32(8192), Scalar32::F32(1.00001f), Scalar32::F32(0.00001f), Scalar32::F32(-999999.0f)};
    DispatchDesc d{krn, bufs, scalars, {128, 1, 1}, {256, 1, 1}, 1, false};

    double elapsed_ms = 0.0;
    while (elapsed_ms < cfg_.warmup_ms) {
        auto tr = backend_.execute(d);
        elapsed_ms += tr.cpu_wall_ns * 1e-6;
    }
    backend_.destroy_kernel(krn);
    backend_.free_buffer(buf);
}

void Calibrator::run_mb1_dispatch(DeviceProfile& out) {
    BufferHandle buf = backend_.allocate_buffer(4096);
    KernelHandle krn = backend_.compile_kernel(BenchKernelId::MB1_DispatchNoop, 0);

    std::array<BufferHandle, 1> bufs{buf};
    std::array<Scalar32, 2> scalars{Scalar32::U32(1), Scalar32::F32(-999999.0f)};

    auto measure_chain = [&](uint32_t repeats, bool barriers) {
        DispatchDesc d{krn, bufs, scalars, {1, 1, 1}, {64, 1, 1}, repeats, barriers};
        for (int w = 0; w < 4; ++w) {
            backend_.execute(d);
        }
        std::vector<double> cpu_us, gpu_us;
        for (uint32_t it = 0; it < cfg_.timed_iters; ++it) {
            auto tr = backend_.execute(d);
            cpu_us.push_back(tr.cpu_wall_ns * 1e-3);
            gpu_us.push_back(tr.gpu_execution_ns * 1e-3);
        }
        return std::pair{calc_median_iqr(cpu_us).median, calc_median_iqr(gpu_us).median};
    };

    auto [cpu_1, gpu_nobar_1]     = measure_chain(1, false);
    auto [cpu_128, gpu_nobar_128] = measure_chain(128, false);
    auto [cpu_b1, gpu_bar_1]      = measure_chain(1, true);
    auto [cpu_b128, gpu_bar_128]  = measure_chain(128, true);

    double slope_nobar = (gpu_nobar_128 - gpu_nobar_1) / 127.0;
    double slope_bar   = (gpu_bar_128 - gpu_bar_1) / 127.0;

    out.t_launch_gpu_us  = std::max(0.0, slope_nobar);
    out.t_barrier_gpu_us = std::max(0.0, slope_bar - slope_nobar);
    out.t_submit_cpu_us  = std::max(0.0, cpu_1 - gpu_nobar_1);

    if (cfg_.verbose) {
        std::cout << std::fixed << std::setprecision(3) << "[MB1 Dispatch] t_submit_cpu = " << out.t_submit_cpu_us
                  << " us | t_launch_gpu = " << out.t_launch_gpu_us << " us | t_barrier_gpu = " << out.t_barrier_gpu_us
                  << " us\n";
    }

    backend_.destroy_kernel(krn);
    backend_.free_buffer(buf);
}

void Calibrator::run_mb2_cache_hierarchy(DeviceProfile& out, BufferHandle shared_src_buf, size_t max_bytes) {
    constexpr uint32_t kNumGroups    = 128;
    constexpr uint32_t kLocalSize    = 256;
    constexpr uint32_t kTotalThreads = kNumGroups * kLocalSize;

    BufferHandle sink_buf   = backend_.allocate_buffer(kTotalThreads * 16);
    KernelHandle stream_krn = backend_.compile_kernel(BenchKernelId::MB2_CacheStreamVec4, 0);

    const std::vector<size_t> sizes_kb = {32,   64,    128,   256,   512,   1024,  1536,  2048,  3072,  4096,  6144,
                                          8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144};

    out.bw_curve.clear();
    for (size_t kb : sizes_kb) {
        size_t w_bytes = kb * 1024ull;
        if (w_bytes > max_bytes) {
            break;
        }

        auto num_blocks       = static_cast<uint32_t>(w_bytes / 4096ull);
        uint32_t passes       = std::max<uint32_t>(1u, num_blocks / kNumGroups);
        uint32_t sweeps       = std::max<uint32_t>(1u, (64u + passes - 1u) / passes);
        std::array<BufferHandle, 2> bufs{shared_src_buf, sink_buf};

        {
            std::array<Scalar32, 4> psc{
                Scalar32::U32(num_blocks), Scalar32::U32(passes), Scalar32::U32(sweeps), Scalar32::F32(-999999.0f)};
            backend_.execute({stream_krn, bufs, psc, {kNumGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false});
            auto pr = backend_.execute({stream_krn, bufs, psc, {kNumGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false});
            if (pr.gpu_execution_ns < 450000.0 && pr.gpu_execution_ns > 10.0) {
                auto mul = static_cast<uint32_t>(std::ceil(450000.0 / pr.gpu_execution_ns));
                sweeps *= std::clamp<uint32_t>(mul, 1u, 64u);
            }
        }

        std::array<Scalar32, 4> sc{
            Scalar32::U32(num_blocks), Scalar32::U32(passes), Scalar32::U32(sweeps), Scalar32::F32(-999999.0f)};
        DispatchDesc d{stream_krn, bufs, sc, {kNumGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false};

        for (int w = 0; w < 3; ++w) {
            backend_.execute(d);
        }

        double bytes_per_dispatch =
            static_cast<double>(kTotalThreads) * static_cast<double>(passes) * static_cast<double>(sweeps) * 16.0;
        std::vector<double> gbps_samples;
        uint32_t iters =
            (w_bytes >= 64 * 1024 * 1024) ? std::max<uint32_t>(11u, cfg_.timed_iters / 2) : cfg_.timed_iters;
        for (uint32_t it = 0; it < iters; ++it) {
            auto tr = backend_.execute(d);
            gbps_samples.push_back(bytes_per_dispatch / tr.gpu_execution_ns);
        }

        auto st = calc_median_iqr(gbps_samples);
        out.bw_curve.push_back({w_bytes, st.median, st.iqr});

        if (cfg_.verbose) {
            double mb = static_cast<double>(w_bytes) / (1024.0 * 1024.0);
            std::cout << "  [MB2] W = " << std::setw(7) << std::setprecision(2) << mb << " MB -> " << std::setw(8)
                      << std::setprecision(2) << st.median << " GB/s (IQR " << std::setw(5) << st.iqr << ")\n";
        }
    }

    const auto& c = out.bw_curve;
    if (c.size() >= 6) {
        out.bw_dram_gbps = 0.5 * (c[c.size() - 1].bw_median_gbps + c[c.size() - 2].bw_median_gbps);
        int l2_top_idx   = static_cast<int>(c.size()) - 3;
        while (l2_top_idx > 1 && c[l2_top_idx].bw_median_gbps < out.bw_dram_gbps * 1.35) {
            --l2_top_idx;
        }
        int cliff_idx   = l2_top_idx;
        double max_drop = 0.0;
        int search_lo   = std::max(1, l2_top_idx - 3);
        int search_hi   = std::min<int>(static_cast<int>(c.size()) - 2, l2_top_idx + 2);
        for (int idx = search_lo; idx <= search_hi; ++idx) {
            double drop = c[idx].bw_median_gbps - c[idx + 1].bw_median_gbps;
            if (drop > max_drop) {
                max_drop  = drop;
                cliff_idx = idx;
            }
        }
        out.l2_effective_bytes = c[cliff_idx].working_set_bytes;
        out.bw_l2_gbps         = c[std::max(0, cliff_idx - 2)].bw_median_gbps;
    }

    if (cfg_.verbose) {
        std::cout << "[MB2 Summary] Detected L2 ~= "
                  << (static_cast<double>(out.l2_effective_bytes) / (1024.0 * 1024.0))
                  << " MB | BW_L2 = " << out.bw_l2_gbps << " GB/s | BW_DRAM = " << out.bw_dram_gbps << " GB/s\n";
    }

    backend_.destroy_kernel(stream_krn);
    backend_.free_buffer(sink_buf);
}

void Calibrator::run_mb3_stride(DeviceProfile& out, BufferHandle shared_src_buf, size_t max_bytes) {
    constexpr uint32_t kNumGroups    = 128;
    constexpr uint32_t kLocalSize    = 256;
    constexpr uint32_t kTotalThreads = kNumGroups * kLocalSize;

    BufferHandle sink_buf = backend_.allocate_buffer(kTotalThreads * 4);
    KernelHandle krn      = backend_.compile_kernel(BenchKernelId::MB3_StrideF32, 0);
    std::array<BufferHandle, 2> bufs{shared_src_buf, sink_buf};

    auto measure_stride_bw = [&](size_t w_bytes, uint32_t stride) {
        size_t tile_bytes = 256ull * stride * sizeof(float);
        auto num_tiles    = static_cast<uint32_t>(w_bytes / tile_bytes);
        uint32_t passes   = std::max<uint32_t>(1u, num_tiles / kNumGroups);
        uint32_t sweeps   = (w_bytes <= 16 * 1024 * 1024) ? 16u : 2u;

        std::array<Scalar32, 6> sc{Scalar32::U32(num_tiles),
                                   Scalar32::U32(passes),
                                   Scalar32::U32(sweeps),
                                   Scalar32::U32(stride),
                                   Scalar32::F32(0.0f),
                                   Scalar32::F32(-999999.0f)};
        DispatchDesc d{krn, bufs, sc, {kNumGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false};
        for (int w = 0; w < 3; ++w) {
            backend_.execute(d);
        }

        double useful_bytes =
            static_cast<double>(kTotalThreads) * static_cast<double>(passes) * static_cast<double>(sweeps) * 4.0;
        std::vector<double> samples;
        for (uint32_t it = 0; it < 15; ++it) {
            auto tr = backend_.execute(d);
            samples.push_back(useful_bytes / tr.gpu_execution_ns);
        }
        return calc_median_iqr(samples).median;
    };

    size_t w_l2 = std::clamp<size_t>(out.l2_effective_bytes / 2, 1024ull * 1024ull, 16ull * 1024ull * 1024ull);
    if (out.l2_effective_bytes > 0 && out.l2_effective_bytes <= 1024ull * 1024ull) {
        w_l2 = 512ull * 1024ull;
    }
    size_t w_dram = std::min<size_t>(max_bytes, std::max<size_t>(128ull * 1024ull * 1024ull, out.l2_effective_bytes * 4));

    double l2_s1    = measure_stride_bw(w_l2, 1);
    double l2_s32   = measure_stride_bw(w_l2, 32);
    double dram_s1  = measure_stride_bw(w_dram, 1);
    double dram_s32 = measure_stride_bw(w_dram, 32);

    out.bw_l2_strided32_gbps   = l2_s32;
    out.coal_factor_l2_s32     = std::clamp(l2_s32 / std::max(1e-6, l2_s1), 0.005, 1.0);
    out.bw_dram_strided32_gbps = dram_s32;
    out.coal_factor_dram_s32   = std::clamp(dram_s32 / std::max(1e-6, dram_s1), 0.005, 1.0);

    if (cfg_.verbose) {
        std::cout << "[MB3 Stride=32] L2 useful BW = " << out.bw_l2_strided32_gbps << " GB/s (factor "
                  << std::setprecision(3) << out.coal_factor_l2_s32
                  << ") | DRAM useful BW = " << std::setprecision(2) << out.bw_dram_strided32_gbps
                  << " GB/s (factor " << std::setprecision(3) << out.coal_factor_dram_s32 << ")\n";
    }

    backend_.destroy_kernel(krn);
    backend_.free_buffer(sink_buf);
}

void Calibrator::run_mb4_smem_vs_cache(DeviceProfile& out, BufferHandle shared_src_buf) {
    constexpr uint32_t M = 1024, N = 1024, K = 512;
    BufferHandle buf_b    = backend_.allocate_buffer(K * N * sizeof(float));
    BufferHandle buf_bias = backend_.allocate_buffer(N * sizeof(float));
    BufferHandle buf_c    = backend_.allocate_buffer(M * N * sizeof(float));

    // Initialize B and bias with PRNG so no NaN/denormal stalls occur
    {
        KernelHandle init_krn = backend_.compile_kernel(BenchKernelId::InitPrngVec4, 0);
        std::array<BufferHandle, 1> b1{buf_b};
        std::array<Scalar32, 3> s1{Scalar32::U32((K * N) / 4), Scalar32::U32(16384), Scalar32::U32(0x12345678u)};
        backend_.execute({init_krn, b1, s1, {64, 1, 1}, {256, 1, 1}, 1, false});

        std::array<BufferHandle, 1> b2{buf_bias};
        std::array<Scalar32, 3> s2{Scalar32::U32(N / 4), Scalar32::U32(256), Scalar32::U32(0x87654321u)};
        backend_.execute({init_krn, b2, s2, {1, 1, 1}, {256, 1, 1}, 1, false});
        backend_.destroy_kernel(init_krn);
    }

    auto bench_variant = [&](BenchKernelId kid) {
        KernelHandle krn = backend_.compile_kernel(kid, 0);
        std::array<BufferHandle, 4> bufs{shared_src_buf, buf_b, buf_bias, buf_c};
        std::array<Scalar32, 4> sc{Scalar32::U32(M), Scalar32::U32(N), Scalar32::U32(K), Scalar32::F32(-999999.0f)};
        DispatchDesc d{krn, bufs, sc, {N / 64, M / 64, 1}, {16, 16, 1}, 1, false};

        for (int w = 0; w < 4; ++w) {
            backend_.execute(d);
        }
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        std::vector<double> gflops;
        for (uint32_t it = 0; it < cfg_.timed_iters; ++it) {
            auto tr = backend_.execute(d);
            gflops.push_back(flops / tr.gpu_execution_ns);
        }
        backend_.destroy_kernel(krn);
        return calc_median_iqr(gflops).median;
    };

    out.gflops_matmul_smem    = bench_variant(BenchKernelId::MB4_MatmulSmem);
    out.gflops_matmul_cache   = bench_variant(BenchKernelId::MB4_MatmulDirectCache);
    out.smem_vs_cache_speedup = out.gflops_matmul_smem / std::max(1e-6, out.gflops_matmul_cache);
    out.prefer_shared_mem     = (out.smem_vs_cache_speedup > 1.05);

    if (cfg_.verbose) {
        std::cout << std::setprecision(2) << "[MB4 Tiled 64x64] SMEM = " << out.gflops_matmul_smem
                  << " GFLOPS | DirectCache = " << out.gflops_matmul_cache
                  << " GFLOPS | Speedup = " << std::setprecision(3) << out.smem_vs_cache_speedup << "x (prefer_smem="
                  << (out.prefer_shared_mem ? "true" : "false") << ")\n";
    }

    backend_.free_buffer(buf_c);
    backend_.free_buffer(buf_bias);
    backend_.free_buffer(buf_b);
}

void Calibrator::run_mb5_compute_and_waves(DeviceProfile& out) {
    constexpr uint32_t kMaxGroups = 512;
    constexpr uint32_t kLocalSize = 256;
    BufferHandle sink_buf         = backend_.allocate_buffer(kMaxGroups * kLocalSize * sizeof(float));
    std::array<BufferHandle, 1> bufs{sink_buf};

    KernelHandle krn_fma  = backend_.compile_kernel(BenchKernelId::MB5_Fma16, 0);
    KernelHandle krn_tanh = backend_.compile_kernel(BenchKernelId::MB5_GeluTanh, 0);
    KernelHandle krn_erf  = backend_.compile_kernel(BenchKernelId::MB5_GeluErf, 0);

    // 5a. Peak FP32 FMA throughput
    {
        constexpr uint32_t kGroups = 256, kIters = 4096;
        std::array<Scalar32, 4> sc{
            Scalar32::U32(kIters), Scalar32::F32(1.000001f), Scalar32::F32(0.000001f), Scalar32::F32(-999999.0f)};
        DispatchDesc d{krn_fma, bufs, sc, {kGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false};
        for (int w = 0; w < 4; ++w) {
            backend_.execute(d);
        }
        double flops = static_cast<double>(kGroups * kLocalSize) * static_cast<double>(kIters) * 32.0;
        std::vector<double> samples;
        for (uint32_t it = 0; it < cfg_.timed_iters; ++it) {
            samples.push_back(flops / backend_.execute(d).gpu_execution_ns);
        }
        out.pi_fma_gflops = calc_median_iqr(samples).median;
    }

    // 5b. GELU(TANH) and GELU(ERF) SFU throughput (Giga-elements/s)
    auto measure_sfu = [&](KernelHandle krn) {
        constexpr uint32_t kGroups = 256, kIters = 512;
        std::array<Scalar32, 4> sc{
            Scalar32::U32(kIters), Scalar32::F32(0.5f), Scalar32::F32(0.3f), Scalar32::F32(-999999.0f)};
        DispatchDesc d{krn, bufs, sc, {kGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false};
        for (int w = 0; w < 3; ++w) {
            backend_.execute(d);
        }
        double elems = static_cast<double>(kGroups * kLocalSize) * static_cast<double>(kIters) * 8.0;
        std::vector<double> samples;
        for (uint32_t it = 0; it < 15; ++it) {
            samples.push_back(elems / backend_.execute(d).gpu_execution_ns);
        }
        return calc_median_iqr(samples).median;
    };
    out.pi_gelu_tanh_gelem_s = measure_sfu(krn_tanh);
    out.pi_gelu_erf_gelem_s  = measure_sfu(krn_erf);

    // 5c. Wave saturation curve across workgroup counts G
    const std::vector<uint32_t> grid_sizes = {1,  2,  4,  6,   8,   12,  16,  24,  32,  36,
                                              48, 64, 72, 96, 108, 144, 192, 256, 384, 512};
    out.wave_curve.clear();
    double peak_wave_gflops = 0.0;
    for (uint32_t g : grid_sizes) {
        constexpr uint32_t kIters = 2048;
        std::array<Scalar32, 4> sc{
            Scalar32::U32(kIters), Scalar32::F32(1.000001f), Scalar32::F32(0.000001f), Scalar32::F32(-999999.0f)};
        DispatchDesc d{krn_fma, bufs, sc, {g, 1, 1}, {kLocalSize, 1, 1}, 1, false};
        backend_.execute(d);
        double flops = static_cast<double>(g * kLocalSize) * static_cast<double>(kIters) * 32.0;
        std::vector<double> ns_vec;
        for (int it = 0; it < 9; ++it) {
            ns_vec.push_back(backend_.execute(d).gpu_execution_ns);
        }
        double med_ns = calc_median_iqr(ns_vec).median;
        double gf     = flops / med_ns;
        peak_wave_gflops = std::max(peak_wave_gflops, gf);
        out.wave_curve.push_back({g, gf, med_ns * 1e-3});
    }

    // Detect hardware wave quantum from the first discrete step jump (>1.35x) in GPU execution time
    out.g_wave_quantum = grid_sizes.front();
    if (out.wave_curve.size() >= 2) {
        double base_us = out.wave_curve.front().gpu_time_us;
        for (size_t i = 0; i + 1 < out.wave_curve.size(); ++i) {
            double curr_us = out.wave_curve[i].gpu_time_us;
            double next_us = out.wave_curve[i + 1].gpu_time_us;
            if (next_us > 1.35 * curr_us && next_us > 1.35 * base_us) {
                out.g_wave_quantum = out.wave_curve[i].workgroups;
                break;
            }
        }
    }

    out.g_half_workgroups = grid_sizes.back();
    out.g_sat_workgroups  = grid_sizes.back();
    for (const auto& pt : out.wave_curve) {
        if (pt.gflops >= 0.50 * peak_wave_gflops) {
            out.g_half_workgroups = pt.workgroups;
            break;
        }
    }
    for (const auto& pt : out.wave_curve) {
        if (pt.gflops >= 0.85 * peak_wave_gflops) {
            out.g_sat_workgroups = pt.workgroups;
            break;
        }
    }

    if (cfg_.verbose) {
        std::cout << std::setprecision(2) << "[MB5 Compute] Peak FMA = " << out.pi_fma_gflops
                  << " GFLOPS | GELU_TANH = " << out.pi_gelu_tanh_gelem_s
                  << " GElem/s | GELU_ERF = " << out.pi_gelu_erf_gelem_s << " GElem/s | G_wave = " << out.g_wave_quantum << " WGs, G_half = " << out.g_half_workgroups << " WGs, G_sat(85%) = " << out.g_sat_workgroups << " WGs\n";
    }

    backend_.destroy_kernel(krn_erf);
    backend_.destroy_kernel(krn_tanh);
    backend_.destroy_kernel(krn_fma);
    backend_.free_buffer(sink_buf);
}

void Calibrator::run_mb6_compile_cost(DeviceProfile& out, BufferHandle shared_src_buf) {
    BufferHandle buf_b    = backend_.allocate_buffer(64 * 64 * sizeof(float));
    BufferHandle buf_bias = backend_.allocate_buffer(64 * sizeof(float));
    BufferHandle buf_c    = backend_.allocate_buffer(64 * 64 * sizeof(float));

    std::random_device rd;
    std::mt19937 rng(rd());

    auto measure_compile = [&](BenchKernelId kid, bool is_matmul) {
        std::vector<double> cpu_ms, lazy_ms;
        for (uint32_t s = 0; s < cfg_.compile_samples; ++s) {
            uint32_t salt = ((rng() & 0x007FFFFEu) | 1u);
            double comp_ns = 0.0;
            KernelHandle krn = backend_.compile_kernel(kid, salt, &comp_ns);

            TimingResult tr1{}, tr2{};
            if (!is_matmul) {
                std::array<BufferHandle, 1> b1{buf_c};
                std::array<Scalar32, 2> sc{Scalar32::U32(1), Scalar32::F32(-999999.0f)};
                DispatchDesc d{krn, b1, sc, {1, 1, 1}, {64, 1, 1}, 1, false};
                tr1 = backend_.execute(d);
                tr2 = backend_.execute(d);
            } else {
                std::array<BufferHandle, 4> b4{shared_src_buf, buf_b, buf_bias, buf_c};
                std::array<Scalar32, 4> sc{
                    Scalar32::U32(64), Scalar32::U32(64), Scalar32::U32(16), Scalar32::F32(-999999.0f)};
                DispatchDesc d{krn, b4, sc, {1, 1, 1}, {16, 16, 1}, 1, false};
                tr1 = backend_.execute(d);
                tr2 = backend_.execute(d);
            }

            cpu_ms.push_back(comp_ns * 1e-6);
            lazy_ms.push_back(std::max(0.0, (tr1.cpu_wall_ns - tr2.cpu_wall_ns) * 1e-6));
            backend_.destroy_kernel(krn);
        }
        return std::pair{calc_median_iqr(cpu_ms).median, calc_median_iqr(lazy_ms).median};
    };

    std::tie(out.t_compile_small_cpu_ms, out.t_compile_small_lazy_ms) =
        measure_compile(BenchKernelId::MB1_DispatchNoop, false);
    std::tie(out.t_compile_tiled_cpu_ms, out.t_compile_tiled_lazy_ms) =
        measure_compile(BenchKernelId::MB4_MatmulSmem, true);
    std::tie(out.t_compile_fused_erf_cpu_ms, out.t_compile_fused_erf_lazy_ms) =
        measure_compile(BenchKernelId::MB6_MatmulFusedGeluErf, true);

    if (cfg_.verbose) {
        std::cout << std::setprecision(3)
                  << "[MB6 Compile (CPU + Lazy ms)] Small = " << out.t_compile_small_cpu_ms << " + "
                  << out.t_compile_small_lazy_ms << " ms | TiledMatMul = " << out.t_compile_tiled_cpu_ms << " + "
                  << out.t_compile_tiled_lazy_ms << " ms | FusedMatMulGeluErf = " << out.t_compile_fused_erf_cpu_ms
                  << " + " << out.t_compile_fused_erf_lazy_ms << " ms\n";
    }

    backend_.free_buffer(buf_c);
    backend_.free_buffer(buf_bias);
    backend_.free_buffer(buf_b);
}

DeviceProfile Calibrator::run_full() {
    DeviceProfile prof{};
    prof.props = backend_.query_static_props();
    if (cfg_.verbose) {
        std::cout << "============================================================\n"
                  << "Device   : " << prof.props.device_name << " (" << prof.props.driver_name << ")\n"
                  << "Topology : " << (prof.props.is_integrated_uma ? "Integrated UMA" : "Discrete GPU")
                  << " | Subgroup: " << prof.props.subgroup_size << " [" << prof.props.min_subgroup_size << ".."
                  << prof.props.max_subgroup_size
                  << "] | SharedMem limit: " << (prof.props.max_shared_mem_bytes / 1024) << " KB\n"
                  << "============================================================\n";
    }

    size_t max_bytes = cfg_.max_buffer_bytes;
    if (prof.props.device_local_mem_bytes > 0) {
        max_bytes = std::min<size_t>(max_bytes, prof.props.device_local_mem_bytes / 4);
    }
    if (prof.props.max_storage_buffer_range > 0) {
        max_bytes = std::min<size_t>(max_bytes, static_cast<size_t>(prof.props.max_storage_buffer_range));
    }
    max_bytes = std::max<size_t>(64ull * 1024 * 1024, (max_bytes / (16 * 1024 * 1024)) * (16 * 1024 * 1024));

    BufferHandle shared_src_buf = backend_.allocate_buffer(max_bytes);
    {
        KernelHandle init_krn = backend_.compile_kernel(BenchKernelId::InitPrngVec4, 0);
        std::array<BufferHandle, 1> ibufs{shared_src_buf};
        std::array<Scalar32, 3> iscalars{
            Scalar32::U32(static_cast<uint32_t>(max_bytes / 16)), Scalar32::U32(32768), Scalar32::U32(0xCAFEBABEu)};
        backend_.execute({init_krn, ibufs, iscalars, {128, 1, 1}, {256, 1, 1}, 1, false});
        backend_.destroy_kernel(init_krn);
    }

    run_warmup();
    run_mb1_dispatch(prof);
    run_mb2_cache_hierarchy(prof, shared_src_buf, max_bytes);
    run_mb3_stride(prof, shared_src_buf, max_bytes);
    run_mb4_smem_vs_cache(prof, shared_src_buf);
    run_mb5_compute_and_waves(prof);
    run_mb6_compile_cost(prof, shared_src_buf);

    backend_.free_buffer(shared_src_buf);
    return prof;
}

std::string Calibrator::to_json_string(const DeviceProfile& p) {
    nlohmann::ordered_json j;
    j["runtime"] = to_string(p.props.runtime);
    j["device"]  = {{"name", p.props.device_name},
                    {"driver", p.props.driver_name},
                    {"vendor_id", p.props.vendor_id},
                    {"device_id", p.props.device_id},
                    {"device_uuid", p.props.device_uuid_hex},
                    {"driver_uuid", p.props.driver_uuid_hex},
                    {"is_integrated_uma", p.props.is_integrated_uma},
                    {"subgroup_size", p.props.subgroup_size},
                    {"max_wg_invocations", p.props.max_wg_invocations},
                    {"max_shared_mem_bytes", p.props.max_shared_mem_bytes},
                    {"device_local_mem_mb", p.props.device_local_mem_bytes / (1024 * 1024)}};
    j["mb1_dispatch"] = {{"t_submit_cpu_us", p.t_submit_cpu_us},
                         {"t_launch_gpu_us", p.t_launch_gpu_us},
                         {"t_barrier_gpu_us", p.t_barrier_gpu_us}};
    j["mb2_memory"]   = {{"l2_effective_bytes", p.l2_effective_bytes},
                         {"l2_effective_mb", static_cast<double>(p.l2_effective_bytes) / (1024.0 * 1024.0)},
                         {"bw_l2_gbps", p.bw_l2_gbps},
                         {"bw_dram_gbps", p.bw_dram_gbps}};
    j["mb3_coalescing"] = {{"bw_l2_strided32_gbps", p.bw_l2_strided32_gbps},
                           {"coal_factor_l2_s32", p.coal_factor_l2_s32},
                           {"bw_dram_strided32_gbps", p.bw_dram_strided32_gbps},
                           {"coal_factor_dram_s32", p.coal_factor_dram_s32}};
    j["mb4_tiling"]     = {{"gflops_matmul_smem", p.gflops_matmul_smem},
                           {"gflops_matmul_cache", p.gflops_matmul_cache},
                           {"smem_vs_cache_speedup", p.smem_vs_cache_speedup},
                           {"prefer_shared_mem", p.prefer_shared_mem}};
    j["mb5_compute"]    = {{"pi_fma_gflops", p.pi_fma_gflops},
                           {"pi_gelu_tanh_gelem_s", p.pi_gelu_tanh_gelem_s},
                           {"pi_gelu_erf_gelem_s", p.pi_gelu_erf_gelem_s},
                           {"g_wave_quantum", p.g_wave_quantum},
                           {"g_half_workgroups", p.g_half_workgroups},
                           {"g_sat_workgroups", p.g_sat_workgroups}};
    j["mb6_compile"]    = {{"small_cpu_ms", p.t_compile_small_cpu_ms},
                           {"small_lazy_ms", p.t_compile_small_lazy_ms},
                           {"tiled_matmul_cpu_ms", p.t_compile_tiled_cpu_ms},
                           {"tiled_matmul_lazy_ms", p.t_compile_tiled_lazy_ms},
                           {"fused_matmul_gelu_erf_cpu_ms", p.t_compile_fused_erf_cpu_ms},
                           {"fused_matmul_gelu_erf_lazy_ms", p.t_compile_fused_erf_lazy_ms}};
    return j.dump(2);
}

void Calibrator::save_bw_csv(const DeviceProfile& p, const std::string& csv_path) {
    std::ofstream ofs(csv_path);
    ofs << "working_set_bytes,working_set_mb,bw_median_gbps,bw_iqr_gbps\n";
    for (const auto& pt : p.bw_curve) {
        ofs << pt.working_set_bytes << "," << (static_cast<double>(pt.working_set_bytes) / (1024.0 * 1024.0)) << ","
            << pt.bw_median_gbps << "," << pt.bw_iqr_gbps << "\n";
    }
}

void Calibrator::save_wave_csv(const DeviceProfile& p, const std::string& csv_path) {
    std::ofstream ofs(csv_path);
    ofs << "workgroups,gflops,gpu_time_us\n";
    for (const auto& pt : p.wave_curve) {
        ofs << pt.workgroups << "," << pt.gflops << "," << pt.gpu_time_us << "\n";
    }
}


DeviceProfile Calibrator::from_json_string(const std::string& json_text) {
    auto j = nlohmann::json::parse(json_text);
    DeviceProfile p{};
    std::string rt = j.value("runtime", "vulkan");
    p.props.runtime = (rt == "opencl") ? RuntimeKind::OpenCL : (rt == "level_zero" ? RuntimeKind::LevelZero : RuntimeKind::Vulkan);

    const auto& jd               = j.at("device");
    p.props.device_name          = jd.value("name", "");
    p.props.driver_name          = jd.value("driver", "");
    p.props.vendor_id            = jd.value("vendor_id", 0u);
    p.props.device_id            = jd.value("device_id", 0u);
    p.props.device_uuid_hex      = jd.value("device_uuid", "");
    p.props.driver_uuid_hex      = jd.value("driver_uuid", "");
    p.props.is_integrated_uma    = jd.value("is_integrated_uma", false);
    p.props.subgroup_size        = jd.value("subgroup_size", 32u);
    p.props.max_wg_invocations   = jd.value("max_wg_invocations", 256u);
    p.props.max_shared_mem_bytes = jd.value("max_shared_mem_bytes", 32768u);

    const auto& j1     = j.at("mb1_dispatch");
    p.t_submit_cpu_us  = j1.value("t_submit_cpu_us", 0.0);
    p.t_launch_gpu_us  = j1.value("t_launch_gpu_us", 0.0);
    p.t_barrier_gpu_us = j1.value("t_barrier_gpu_us", 0.0);

    const auto& j2       = j.at("mb2_memory");
    p.l2_effective_bytes = j2.value("l2_effective_bytes", size_t{0});
    p.bw_l2_gbps         = j2.value("bw_l2_gbps", 100.0);
    p.bw_dram_gbps       = j2.value("bw_dram_gbps", 50.0);

    const auto& j3           = j.at("mb3_coalescing");
    p.bw_l2_strided32_gbps   = j3.value("bw_l2_strided32_gbps", 10.0);
    p.coal_factor_l2_s32     = j3.value("coal_factor_l2_s32", 0.1);
    p.bw_dram_strided32_gbps = j3.value("bw_dram_strided32_gbps", 5.0);
    p.coal_factor_dram_s32   = j3.value("coal_factor_dram_s32", 0.1);

    const auto& j4          = j.at("mb4_tiling");
    p.gflops_matmul_smem    = j4.value("gflops_matmul_smem", 1000.0);
    p.gflops_matmul_cache   = j4.value("gflops_matmul_cache", 1000.0);
    p.smem_vs_cache_speedup = j4.value("smem_vs_cache_speedup", 1.0);
    p.prefer_shared_mem     = j4.value("prefer_shared_mem", true);

    const auto& j5          = j.at("mb5_compute");
    p.pi_fma_gflops         = j5.value("pi_fma_gflops", 1000.0);
    p.pi_gelu_tanh_gelem_s  = j5.value("pi_gelu_tanh_gelem_s", 100.0);
    p.pi_gelu_erf_gelem_s   = j5.value("pi_gelu_erf_gelem_s", 100.0);
    p.g_wave_quantum        = j5.value("g_wave_quantum", j5.value("g_half_workgroups", 1u));
    p.g_half_workgroups     = j5.value("g_half_workgroups", 1u);
    p.g_sat_workgroups      = j5.value("g_sat_workgroups", 1u);

    const auto& j6               = j.at("mb6_compile");
    p.t_compile_small_cpu_ms     = j6.value("small_cpu_ms", 0.0);
    p.t_compile_small_lazy_ms    = j6.value("small_lazy_ms", 0.0);
    p.t_compile_tiled_cpu_ms     = j6.value("tiled_matmul_cpu_ms", 0.0);
    p.t_compile_tiled_lazy_ms    = j6.value("tiled_matmul_lazy_ms", 0.0);
    p.t_compile_fused_erf_cpu_ms = j6.value("fused_matmul_gelu_erf_cpu_ms", 0.0);
    p.t_compile_fused_erf_lazy_ms = j6.value("fused_matmul_gelu_erf_lazy_ms", 0.0);
    return p;
}

} // namespace mgcm
