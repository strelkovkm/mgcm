#include "mgcm/calibrator.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

struct OlsFit {
    double intercept = 0.0;
    double slope     = 0.0;
    double r2        = 0.0;
};

OlsFit fit_ols_1d(const std::vector<double>& xs, const std::vector<double>& ys, bool non_negative_intercept = true) {
    const size_t n = xs.size();
    if (n < 2 || ys.size() != n) {
        return {};
    }
    double x_mean = std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(n);
    double y_mean = std::accumulate(ys.begin(), ys.end(), 0.0) / static_cast<double>(n);

    double sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = xs[i] - x_mean;
        double dy = ys[i] - y_mean;
        sxx += dx * dx;
        sxy += dx * dy;
    }

    OlsFit fit{};
    fit.slope     = (sxx > 1e-12) ? (sxy / sxx) : 0.0;
    fit.intercept = y_mean - fit.slope * x_mean;

    if (non_negative_intercept && fit.intercept < 0.0) {
        // Physical guard: refit ray through origin when unconstrained intercept dips below zero
        double sum_xy = 0.0, sum_xx = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum_xy += xs[i] * ys[i];
            sum_xx += xs[i] * xs[i];
        }
        fit.intercept = 0.0;
        fit.slope     = (sum_xx > 1e-12) ? std::max(0.0, sum_xy / sum_xx) : 0.0;
    }
    fit.slope = std::max(0.0, fit.slope);

    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double y_hat = fit.intercept + fit.slope * xs[i];
        double err   = ys[i] - y_hat;
        double dev   = ys[i] - y_mean;
        ss_res += err * err;
        ss_tot += dev * dev;
    }
    fit.r2 = (ss_tot > 1e-12) ? std::clamp(1.0 - ss_res / ss_tot, 0.0, 1.0) : 1.0;
    return fit;
}

} // namespace

Calibrator::Calibrator(ICalibBackend& backend, IFrontendCompilerProbe* fe_probe, CalibratorConfig cfg)
    : backend_(backend), fe_probe_(fe_probe), cfg_(cfg) {}

void Calibrator::run_warmup() {
    BufferHandle buf = backend_.allocate_buffer(64 * 1024);
    KernelHandle krn = backend_.compile_kernel(BenchKernelId::MB0_Warmup, 0);

    std::array<BufferHandle, 1> bufs{buf};
    std::array<Scalar32, 4> scalars{
        Scalar32::U32(8192), Scalar32::F32(1.00001f), Scalar32::F32(0.00001f), Scalar32::F32(-999999.0f)};
    DispatchDesc d{krn, bufs, scalars, {128, 1, 1}, {256, 1, 1}, 1, false};

    double elapsed_ms = 0.0;
    while (elapsed_ms < cfg_.warmup_ms) {
        elapsed_ms += backend_.execute(d).cpu_wall_ns * 1e-6;
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
            if (it >= 6) {
                auto st = calc_median_iqr(gpu_us);
                if (st.iqr <= 0.015 * std::max(0.01, st.median)) break;
            }
        }
        return std::pair{calc_median_iqr(cpu_us).median, calc_median_iqr(gpu_us).median};
    };

    const std::vector<uint32_t> chain_lens = {1, 8, 32, 128};
    std::vector<double> xs, y_nobar, y_bar;
    double cpu_1 = 0.0, gpu_1 = 0.0;
    out.mb1_points.clear();

    for (uint32_t n : chain_lens) {
        auto [c_nb, g_nb] = measure_chain(n, false);
        auto [c_b, g_b]   = measure_chain(n, true);
        (void)c_b;
        if (n == 1) {
            cpu_1 = c_nb;
            gpu_1 = g_nb;
        }
        xs.push_back(static_cast<double>(n));
        y_nobar.push_back(g_nb);
        y_bar.push_back(g_b);
        out.mb1_points.push_back({n, g_nb, g_b});
    }

    auto fit_nb = fit_ols_1d(xs, y_nobar, false);
    auto fit_b  = fit_ols_1d(xs, y_bar, false);

    out.t_launch_gpu_us  = std::max(0.0, fit_nb.slope);
    out.t_barrier_gpu_us = std::max(0.0, fit_b.slope - fit_nb.slope);
    out.t_submit_cpu_us  = std::max(0.0, cpu_1 - gpu_1);
    out.r2_dispatch      = 0.5 * (fit_nb.r2 + fit_b.r2);

    if (cfg_.verbose) {
        std::cout << std::fixed << std::setprecision(3) << "[MB1 Dispatch] t_submit = " << out.t_submit_cpu_us
                  << " us | t_launch = " << out.t_launch_gpu_us << " us | t_barrier = " << out.t_barrier_gpu_us
                  << " us (R2=" << out.r2_dispatch << ")\n";
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
        auto num_blocks = static_cast<uint32_t>(w_bytes / 4096ull);
        uint32_t passes = std::max<uint32_t>(1u, num_blocks / kNumGroups);
        uint32_t sweeps = std::max<uint32_t>(1u, (64u + passes - 1u) / passes);
        std::array<BufferHandle, 2> bufs{shared_src_buf, sink_buf};

        {
            std::array<Scalar32, 4> psc{
                Scalar32::U32(num_blocks), Scalar32::U32(passes), Scalar32::U32(sweeps), Scalar32::F32(-999999.0f)};
            backend_.execute({stream_krn, bufs, psc, {kNumGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false});
            auto pr = backend_.execute({stream_krn, bufs, psc, {kNumGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false});
            if (pr.gpu_execution_ns < 220000.0 && pr.gpu_execution_ns > 10.0) {
                auto mul = static_cast<uint32_t>(std::ceil(220000.0 / pr.gpu_execution_ns));
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
            gbps_samples.push_back(bytes_per_dispatch / backend_.execute(d).gpu_execution_ns);
            if (it >= 5) {
                auto s = calc_median_iqr(gbps_samples);
                if (s.iqr <= 0.015 * s.median) break;
            }
        }

        auto st = calc_median_iqr(gbps_samples);
        out.bw_curve.push_back({w_bytes, st.median, st.iqr});
        if (w_bytes >= 16ull * 1024 * 1024 && out.bw_curve.size() >= 10) {
            double peak_bw = 0.0;
            for (const auto& pt : out.bw_curve) peak_bw = std::max(peak_bw, pt.bw_median_gbps);
            const auto& c = out.bw_curve;
            double b0 = c[c.size()-1].bw_median_gbps, b1 = c[c.size()-2].bw_median_gbps;
            double b2 = c[c.size()-3].bw_median_gbps, b3 = c[c.size()-4].bw_median_gbps;
            double mean4 = 0.25 * (b0 + b1 + b2 + b3);
            double span4 = std::max({b0, b1, b2, b3}) - std::min({b0, b1, b2, b3});
            double pass_ms = (static_cast<double>(w_bytes) / std::max(0.1, st.median)) * 1e-6;
            bool cliff_passed = (mean4 < 0.45 * peak_bw) || (pass_ms >= 2.0 && mean4 < 0.82 * peak_bw);
            if (cliff_passed && (span4 / std::max(0.1, mean4)) < 0.06) break;
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
        for (int idx = std::max(1, l2_top_idx - 3); idx <= std::min<int>(static_cast<int>(c.size()) - 2, l2_top_idx + 2);
             ++idx) {
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
        std::cout << std::setprecision(2) << "[MB2 Memory]   L2 ~= "
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
            samples.push_back(useful_bytes / backend_.execute(d).gpu_execution_ns);
            if (it >= 5 && calc_median_iqr(samples).iqr <= 0.015 * calc_median_iqr(samples).median) break;
        }
        return calc_median_iqr(samples).median;
    };

    size_t w_l2 = std::clamp<size_t>(out.l2_effective_bytes / 2, 1024ull * 1024ull, 16ull * 1024ull * 1024ull);
    if (out.l2_effective_bytes > 0 && out.l2_effective_bytes <= 1024ull * 1024ull) {
        w_l2 = 512ull * 1024ull;
    }
    size_t max_w = out.bw_curve.empty() ? max_bytes : out.bw_curve.back().working_set_bytes;
    size_t w_dram = std::min<size_t>(max_w, std::max<size_t>(16ull * 1024ull * 1024ull, out.l2_effective_bytes * 8));

    double l2_s1    = measure_stride_bw(w_l2, 1);
    double l2_s32   = measure_stride_bw(w_l2, 32);
    double dram_s1  = measure_stride_bw(w_dram, 1);
    double dram_s32 = measure_stride_bw(w_dram, 32);

    out.bw_l2_strided32_gbps   = l2_s32;
    out.coal_factor_l2_s32     = std::clamp(l2_s32 / std::max(1e-6, l2_s1), 0.005, 1.0);
    out.bw_dram_strided32_gbps = dram_s32;
    out.coal_factor_dram_s32   = std::clamp(dram_s32 / std::max(1e-6, dram_s1), 0.005, 1.0);

    if (cfg_.verbose) {
        std::cout << "[MB3 Stride]   L2 coal = " << std::setprecision(3) << out.coal_factor_l2_s32
                  << " | DRAM coal = " << out.coal_factor_dram_s32 << "\n";
    }

    backend_.destroy_kernel(krn);
    backend_.free_buffer(sink_buf);
}

void Calibrator::run_mb4_tiled_gemm(DeviceProfile& out, BufferHandle shared_src_buf) {
    const bool is_emb = (out.pi_fma_gflops > 0.0 && out.pi_fma_gflops < 100.0);
    const uint32_t M = is_emb ? 256u : 1024u, N = M, K = is_emb ? 128u : 512u;
    const uint32_t Mt = M - 24u, Nt = N - 24u, Kt = K - 12u;
    BufferHandle buf_b    = backend_.allocate_buffer(K * N * sizeof(float));
    BufferHandle buf_bias = backend_.allocate_buffer(N * sizeof(float));
    BufferHandle buf_c    = backend_.allocate_buffer(M * N * sizeof(float));

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

    auto bench_gemm = [&](BenchKernelId kid, uint32_t m_sz, uint32_t n_sz, uint32_t k_sz) {
        KernelHandle krn = backend_.compile_kernel(kid, 0);
        std::array<BufferHandle, 4> bufs{shared_src_buf, buf_b, buf_bias, buf_c};
        std::array<Scalar32, 4> sc{
            Scalar32::U32(m_sz), Scalar32::U32(n_sz), Scalar32::U32(k_sz), Scalar32::F32(-999999.0f)};
        uint32_t gm = (m_sz + 63u) / 64u;
        uint32_t gn = (n_sz + 63u) / 64u;
        DispatchDesc d{krn, bufs, sc, {gn, gm, 1}, {16, 16, 1}, 1, false};

        for (int w = 0; w < 4; ++w) {
            backend_.execute(d);
        }
        // Normalize by padded tile FLOPs so tail_efficiency isolates pure bounds-check predication overhead
        double m_pad = static_cast<double>(gm * 64u);
        double n_pad = static_cast<double>(gn * 64u);
        double k_pad = static_cast<double>(((k_sz + 15u) / 16u) * 16u);
        double flops = 2.0 * m_pad * n_pad * k_pad;
        std::vector<double> gflops;
        for (uint32_t it = 0; it < cfg_.timed_iters; ++it) {
            gflops.push_back(flops / backend_.execute(d).gpu_execution_ns);
            if (it >= 5 && calc_median_iqr(gflops).iqr <= 0.015 * calc_median_iqr(gflops).median) break;
        }
        backend_.destroy_kernel(krn);
        return calc_median_iqr(gflops).median;
    };

    out.gflops_matmul_smem    = bench_gemm(BenchKernelId::MB4_MatmulSmem, M, N, K);
    out.gflops_matmul_cache   = bench_gemm(BenchKernelId::MB4_MatmulDirectCache, M, N, K);
    double gflops_tail        = bench_gemm(BenchKernelId::MB4_MatmulSmemTail, Mt, Nt, Kt);
    double gflops_fused_erf   = bench_gemm(BenchKernelId::MB6_MatmulFusedGeluErf, M, N, K);

    out.smem_vs_cache_speedup = out.gflops_matmul_smem / std::max(1e-6, out.gflops_matmul_cache);
    out.prefer_shared_mem     = (out.smem_vs_cache_speedup > 1.05);
    out.tail_efficiency       = std::clamp(gflops_tail / std::max(1e-6, out.gflops_matmul_smem), 0.25, 1.0);
    double flops_ref      = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double t_smem_ns      = flops_ref / std::max(1e-6, out.gflops_matmul_smem);
    double t_fused_ns     = flops_ref / std::max(1e-6, gflops_fused_erf);
    double t_ep_expect_ns = (static_cast<double>(M * N) / std::max(1.0, out.pi_gelu_erf_gelem_s)) +
                            (static_cast<double>(M * N * 2u) / std::max(10.0, out.pi_fma_gflops));
    double t_mm_net_ns    = std::max(t_smem_ns, t_fused_ns - t_ep_expect_ns);
    out.fusion_reg_factor = std::clamp(t_smem_ns / t_mm_net_ns, 0.20, 1.0);

    if (cfg_.verbose) {
        std::cout << std::setprecision(2) << "[MB4 Tiled]    SMEM = " << out.gflops_matmul_smem
                  << " | Cache = " << out.gflops_matmul_cache << " GFLOPS | tail_eff = " << std::setprecision(3)
                  << out.tail_efficiency << " | fusion_reg = " << out.fusion_reg_factor << "\n";
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

    uint32_t fma_iters = 4096u, sfu_iters = 512u, wave_iters = 2048u;
    {
        std::array<Scalar32, 4> psc{Scalar32::U32(256u), Scalar32::F32(1.000001f), Scalar32::F32(0.000001f), Scalar32::F32(-999999.0f)};
        auto pr = backend_.execute({krn_fma, bufs, psc, {32, 1, 1}, {kLocalSize, 1, 1}, 1, false});
        double est_gf = (static_cast<double>(32 * kLocalSize) * 256.0 * 32.0) / std::max(100.0, pr.gpu_execution_ns);
        if (est_gf < 50.0) { fma_iters = 256u; sfu_iters = 64u; wave_iters = 256u; }
        else if (est_gf < 500.0) { fma_iters = 1024u; sfu_iters = 128u; wave_iters = 512u; }
    }
    {
        uint32_t kGroups = (fma_iters <= 256u) ? 64u : 256u;
        std::array<Scalar32, 4> sc{Scalar32::U32(fma_iters), Scalar32::F32(1.000001f), Scalar32::F32(0.000001f), Scalar32::F32(-999999.0f)};
        DispatchDesc d{krn_fma, bufs, sc, {kGroups, 1, 1}, {kLocalSize, 1, 1}, 1, false};
        for (int w = 0; w < 2; ++w) backend_.execute(d);
        double flops = static_cast<double>(kGroups * kLocalSize) * static_cast<double>(fma_iters) * 32.0;
        std::vector<double> samples;
        for (uint32_t it = 0; it < cfg_.timed_iters; ++it) {
            samples.push_back(flops / backend_.execute(d).gpu_execution_ns);
            if (it >= 5 && calc_median_iqr(samples).iqr <= 0.012 * calc_median_iqr(samples).median) break;
        }
        out.pi_fma_gflops = calc_median_iqr(samples).median;
    }

    auto measure_sfu = [&](KernelHandle krn) {
        uint32_t kGroups = (sfu_iters <= 64u) ? 64u : 256u, kIters = sfu_iters;
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
            if (it >= 5 && calc_median_iqr(samples).iqr <= 0.012 * calc_median_iqr(samples).median) break;
        }
        return calc_median_iqr(samples).median;
    };
    out.pi_gelu_tanh_gelem_s = measure_sfu(krn_tanh);
    out.pi_gelu_erf_gelem_s  = measure_sfu(krn_erf);

    const std::vector<uint32_t> grid_sizes = {1,  2,  4,  6,   8,   12,  16,  24,  32,  36,
                                              48, 64, 72, 96, 108, 144, 192, 256, 384, 512};
    out.wave_curve.clear();
    double peak_wave_gflops = 0.0;
    for (uint32_t g : grid_sizes) {
        const uint32_t kIters = wave_iters;
        std::array<Scalar32, 4> sc{
            Scalar32::U32(kIters), Scalar32::F32(1.000001f), Scalar32::F32(0.000001f), Scalar32::F32(-999999.0f)};
        DispatchDesc d{krn_fma, bufs, sc, {g, 1, 1}, {kLocalSize, 1, 1}, 1, false};
        backend_.execute(d);
        double flops = static_cast<double>(g * kLocalSize) * static_cast<double>(kIters) * 32.0;
        std::vector<double> ns_vec;
        for (int it = 0; it < 9; ++it) {
            ns_vec.push_back(backend_.execute(d).gpu_execution_ns);
            if (it >= 4 && calc_median_iqr(ns_vec).iqr <= 0.015 * calc_median_iqr(ns_vec).median) break;
        }
        double med_ns    = calc_median_iqr(ns_vec).median;
        double gf        = flops / med_ns;
        peak_wave_gflops = std::max(peak_wave_gflops, gf);
        out.wave_curve.push_back({g, gf, med_ns * 1e-3});
        bool jump_seen = false;
        for (size_t wi = 0; wi + 1 < out.wave_curve.size(); ++wi)
            if (out.wave_curve[wi + 1].gpu_time_us > 1.35 * out.wave_curve[wi].gpu_time_us) jump_seen = true;
        if (jump_seen && out.wave_curve.size() >= 8 && gf >= 0.95 * out.pi_fma_gflops) break;
        if (out.wave_curve.size() >= 6 && med_ns > 10000000.0 && gf >= 0.90 * out.pi_fma_gflops) break;
    }
    out.pi_fma_gflops = std::max(out.pi_fma_gflops, peak_wave_gflops);

    out.g_wave_quantum  = grid_sizes.front();
    out.pi_wave1_gflops = out.wave_curve.front().gflops;
    if (out.wave_curve.size() >= 2) {
        double min_us = out.wave_curve.front().gpu_time_us;
        for (size_t i = 0; i + 1 < out.wave_curve.size(); ++i) {
            double curr_us = out.wave_curve[i].gpu_time_us;
            double next_us = out.wave_curve[i + 1].gpu_time_us;
            min_us = std::min(min_us, curr_us);
            if (next_us > 1.35 * curr_us && next_us > 1.35 * min_us) {
                out.g_wave_quantum  = out.wave_curve[i].workgroups;
                out.pi_wave1_gflops = out.wave_curve[i].gflops;
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
        std::cout << std::setprecision(2) << "[MB5 Compute]  Peak FMA = " << out.pi_fma_gflops
                  << " | GELU_ERF = " << out.pi_gelu_erf_gelem_s << " GElem/s | G_wave = " << out.g_wave_quantum
                  << " (pi_wave1 = " << out.pi_wave1_gflops << " GFLOPS)\n";
    }

    backend_.destroy_kernel(krn_erf);
    backend_.destroy_kernel(krn_tanh);
    backend_.destroy_kernel(krn_fma);
    backend_.free_buffer(sink_buf);
}

void Calibrator::run_mb6_compile_ols(DeviceProfile& out, BufferHandle shared_src_buf) {
    BufferHandle buf_b    = backend_.allocate_buffer(64 * 64 * sizeof(float));
    BufferHandle buf_bias = backend_.allocate_buffer(64 * sizeof(float));
    BufferHandle buf_c    = backend_.allocate_buffer(64 * 64 * sizeof(float));

    std::random_device rd;
    std::mt19937 rng(rd());

    enum class ProbeMode { Noop, Compute1D, Matmul4Buf };
    struct ProbeSpec {
        BenchKernelId id;
        ProbeMode     mode;
    };
    const std::vector<ProbeSpec> probes = {
        {BenchKernelId::MB1_DispatchNoop, ProbeMode::Noop},
        {BenchKernelId::MB5_GeluTanh, ProbeMode::Compute1D},
        {BenchKernelId::MB5_GeluErf, ProbeMode::Compute1D},
        {BenchKernelId::MB4_MatmulDirectCache, ProbeMode::Matmul4Buf},
        {BenchKernelId::MB4_MatmulSmem, ProbeMode::Matmul4Buf},
        {BenchKernelId::MB6_MatmulFusedGeluErf, ProbeMode::Matmul4Buf},
    };

    std::vector<double> drv_xs, drv_ys, fe_xs, fe_ys;
    out.mb6_drv_points.clear();
    out.mb6_fe_points.clear();
    double drv_wall_ms = 0.0, fe_wall_ms = 0.0;

    for (const auto& ps : probes) {
        auto t_drv0      = std::chrono::steady_clock::now();
        double spv_words = static_cast<double>(backend_.get_kernel_spirv_words(ps.id));
        std::vector<double> total_drv_samples;

        for (uint32_t s = 0; s < cfg_.compile_samples; ++s) {
            if (s >= 2 && total_drv_samples.back() > 150.0) break;
            if (s >= 3 && (total_drv_samples.back() > 25.0 ||
                calc_median_iqr(total_drv_samples).iqr <= 0.02 * calc_median_iqr(total_drv_samples).median)) break;
            uint32_t salt    = ((rng() & 0x007FFFFEu) | 1u);
            double comp_ns   = 0.0;
            KernelHandle krn = backend_.compile_kernel(ps.id, salt, &comp_ns);

            TimingResult tr1{}, tr2{};
            if (ps.mode == ProbeMode::Noop) {
                std::array<BufferHandle, 1> b1{buf_c};
                std::array<Scalar32, 2> sc{Scalar32::U32(1), Scalar32::F32(-999999.0f)};
                DispatchDesc d{krn, b1, sc, {1, 1, 1}, {64, 1, 1}, 1, false};
                tr1 = backend_.execute(d);
                tr2 = backend_.execute(d);
            } else if (ps.mode == ProbeMode::Compute1D) {
                std::array<BufferHandle, 1> b1{buf_c};
                std::array<Scalar32, 4> sc{
                    Scalar32::U32(1), Scalar32::F32(0.5f), Scalar32::F32(0.3f), Scalar32::F32(-999999.0f)};
                DispatchDesc d{krn, b1, sc, {1, 1, 1}, {256, 1, 1}, 1, false};
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

            double lazy_ms = std::max(0.0, (tr1.cpu_wall_ns - tr2.cpu_wall_ns) * 1e-6);
            total_drv_samples.push_back(comp_ns * 1e-6 + lazy_ms);
            backend_.destroy_kernel(krn);
        }

        double med_drv = calc_median_iqr(total_drv_samples).median;
        drv_xs.push_back(spv_words);
        drv_ys.push_back(med_drv);
        out.mb6_drv_points.push_back({static_cast<uint32_t>(spv_words), med_drv});
        drv_wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_drv0).count();

        if (fe_probe_ && fe_probe_->is_available()) {
            auto t_fe0 = std::chrono::steady_clock::now();
            std::vector<double> fe_samples;
            size_t measured_words = 0;
            uint32_t fe_runs      = std::min<uint32_t>(cfg_.compile_samples, 3u);
            for (uint32_t s = 0; s < fe_runs; ++s) {
                if (s >= 1 && !fe_samples.empty() && fe_samples.back() > 80.0) break;
                uint32_t salt = ((rng() & 0x007FFFFEu) | 1u);
                double fe_ms  = 0.0;
                if (fe_probe_->measure_compile_ms(ps.id, salt, &fe_ms, &measured_words)) {
                    fe_samples.push_back(fe_ms);
                }
            }
            if (!fe_samples.empty()) {
                double w_fe   = measured_words > 0 ? static_cast<double>(measured_words) : spv_words;
                double med_fe = calc_median_iqr(fe_samples).median;
                fe_xs.push_back(w_fe);
                fe_ys.push_back(med_fe);
                out.mb6_fe_points.push_back({static_cast<uint32_t>(w_fe), med_fe});
            }
            fe_wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fe0).count();
        }
    }

    auto drv_fit                 = fit_ols_1d(drv_xs, drv_ys, true);
    out.beta_0_drv_ms            = drv_fit.intercept;
    out.beta_1_drv_ms_per_word   = drv_fit.slope;
    out.beta_0_drv_smem_ms       = drv_fit.intercept;
    out.beta_1_drv_smem_ms_per_word = drv_fit.slope;
    out.r2_drv                   = drv_fit.r2;
    if (drv_fit.r2 < 0.95 && drv_xs.size() >= 6) {
        std::vector<double> lx(drv_xs.begin(), drv_xs.begin() + 4), ly(drv_ys.begin(), drv_ys.begin() + 4);
        std::vector<double> sx(drv_xs.begin() + 4, drv_xs.end()),   sy(drv_ys.begin() + 4, drv_ys.end());
        auto lin_fit  = fit_ols_1d(lx, ly, true);
        auto smem_fit = fit_ols_1d(sx, sy, true);
        out.beta_0_drv_ms               = lin_fit.intercept;
        out.beta_1_drv_ms_per_word      = lin_fit.slope;
        out.beta_0_drv_smem_ms          = smem_fit.intercept;
        out.beta_1_drv_smem_ms_per_word = smem_fit.slope;
    }
    out.t_cache_hit_ms           = backend_.measure_pipeline_cache_hit_ms(BenchKernelId::MB4_MatmulSmem);
    out.stage_timings.mb6_drv_ms = drv_wall_ms;
    out.stage_timings.mb6_fe_ms  = fe_wall_ms;

    if (fe_xs.size() >= 2 && fe_probe_) {
        auto fe_fit = fit_ols_1d(fe_xs, fe_ys, true);
        if (fe_probe_->kind() == CompilerKind::Slang) {
            out.alpha_0_slang_ms          = fe_fit.intercept;
            out.alpha_1_slang_ms_per_word = fe_fit.slope;
            out.r2_fe_slang               = fe_fit.r2;
        } else if (fe_probe_->kind() == CompilerKind::GraphCompiler) {
            out.alpha_0_gc_ms          = fe_fit.intercept;
            out.alpha_1_gc_ms_per_word = fe_fit.slope;
            out.r2_fe_gc               = fe_fit.r2;
        }
    }

    if (cfg_.verbose) {
        std::cout << std::setprecision(4) << "[MB6 Compile]  Drv OLS: b0 = " << out.beta_0_drv_ms
                  << " ms, b1 = " << (out.beta_1_drv_ms_per_word * 1e3) << " us/word (R2=" << out.r2_drv
                  << ") | cache_hit = " << out.t_cache_hit_ms << " ms\n";
        if (out.r2_fe_slang > 0.0) {
            std::cout << "               Slang FE OLS: a0 = " << out.alpha_0_slang_ms
                      << " ms, a1 = " << (out.alpha_1_slang_ms_per_word * 1e3) << " us/word (R2=" << out.r2_fe_slang
                      << ")\n";
        }
    }

    backend_.free_buffer(buf_c);
    backend_.free_buffer(buf_bias);
    backend_.free_buffer(buf_b);
}

DeviceProfile Calibrator::run_full() {
    auto cal_t0 = std::chrono::steady_clock::now();
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    DeviceProfile prof{};
    prof.schema_version = kDeviceProfileSchemaVersion;
    prof.props          = backend_.query_static_props();

    if (cfg_.verbose) {
        std::cout << "============================================================\n"
                  << "Device   : " << prof.props.device_name << " (" << prof.props.driver_name << ")\n"
                  << "Topology : " << (prof.props.is_integrated_uma ? "Unified UMA" : "Desktop Discrete")
                  << " | Subgroup: " << prof.props.subgroup_size
                  << " | SharedMem: " << (prof.props.max_shared_mem_bytes / 1024) << " KB\n"
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

    auto ts = std::chrono::steady_clock::now();
    BufferHandle shared_src_buf = backend_.allocate_buffer(max_bytes);
    {
        KernelHandle init_krn = backend_.compile_kernel(BenchKernelId::InitPrngVec4, 0);
        std::array<BufferHandle, 1> ibufs{shared_src_buf};
        std::array<Scalar32, 3> iscalars{
            Scalar32::U32(static_cast<uint32_t>(max_bytes / 16)), Scalar32::U32(32768), Scalar32::U32(0xCAFEBABEu)};
        backend_.execute({init_krn, ibufs, iscalars, {128, 1, 1}, {256, 1, 1}, 1, false});
        backend_.destroy_kernel(init_krn);
    }
    prof.stage_timings.init_prng_ms = ms_since(ts);

    ts = std::chrono::steady_clock::now(); run_warmup();                                             prof.stage_timings.mb0_warmup_ms   = ms_since(ts);
    ts = std::chrono::steady_clock::now(); run_mb1_dispatch(prof);                                   prof.stage_timings.mb1_dispatch_ms = ms_since(ts);
    ts = std::chrono::steady_clock::now(); run_mb2_cache_hierarchy(prof, shared_src_buf, max_bytes); prof.stage_timings.mb2_cache_ms    = ms_since(ts);
    ts = std::chrono::steady_clock::now(); run_mb3_stride(prof, shared_src_buf, max_bytes);          prof.stage_timings.mb3_stride_ms   = ms_since(ts);
    ts = std::chrono::steady_clock::now(); run_mb5_compute_and_waves(prof);                          prof.stage_timings.mb5_compute_ms  = ms_since(ts);
    ts = std::chrono::steady_clock::now(); run_mb4_tiled_gemm(prof, shared_src_buf);                 prof.stage_timings.mb4_tiled_ms    = ms_since(ts);
    run_mb6_compile_ols(prof, shared_src_buf);

    backend_.free_buffer(shared_src_buf);
    prof.calibration_wall_time_ms = ms_since(cal_t0);
    if (cfg_.verbose) {
        const auto& st = prof.stage_timings;
        std::cout << std::fixed << std::setprecision(1)
                  << "[Bench Times]  Init=" << st.init_prng_ms << " | MB0=" << st.mb0_warmup_ms
                  << " | MB1=" << st.mb1_dispatch_ms << " | MB2=" << st.mb2_cache_ms
                  << " | MB3=" << st.mb3_stride_ms << " | MB4=" << st.mb4_tiled_ms
                  << " | MB5=" << st.mb5_compute_ms << " | MB6_drv=" << st.mb6_drv_ms
                  << " | MB6_fe=" << st.mb6_fe_ms << " ms (Total=" << prof.calibration_wall_time_ms << " ms)\n";
    }
    return prof;
}

std::string Calibrator::to_json_string(const DeviceProfile& p) {
    nlohmann::ordered_json j;
    j["schema_version"]           = p.schema_version;
    j["calibration_wall_time_ms"] = p.calibration_wall_time_ms;
    j["stage_timings_ms"]         = {{"init_prng_ms", p.stage_timings.init_prng_ms},
                                     {"mb0_warmup_ms", p.stage_timings.mb0_warmup_ms},
                                     {"mb1_dispatch_ms", p.stage_timings.mb1_dispatch_ms},
                                     {"mb2_cache_ms", p.stage_timings.mb2_cache_ms},
                                     {"mb3_stride_ms", p.stage_timings.mb3_stride_ms},
                                     {"mb4_tiled_ms", p.stage_timings.mb4_tiled_ms},
                                     {"mb5_compute_ms", p.stage_timings.mb5_compute_ms},
                                     {"mb6_drv_ms", p.stage_timings.mb6_drv_ms},
                                     {"mb6_fe_ms", p.stage_timings.mb6_fe_ms}};
    j["runtime"]                  = to_string(p.props.runtime);
    j["device"]                   = {{"name", p.props.device_name},
                                     {"driver", p.props.driver_name},
                                     {"api_version", p.props.api_version_str},
                                     {"vendor_id", p.props.vendor_id},
                                     {"device_id", p.props.device_id},
                                     {"device_uuid", p.props.device_uuid_hex},
                                     {"driver_uuid", p.props.driver_uuid_hex},
                                     {"is_integrated_uma", p.props.is_integrated_uma},
                                     {"subgroup_size", p.props.subgroup_size},
                                     {"max_wg_invocations", p.props.max_wg_invocations},
                                     {"max_shared_mem_bytes", p.props.max_shared_mem_bytes},
                                     {"device_local_mem_mb", p.props.device_local_mem_bytes / (1024 * 1024)}};
    j["mb1_dispatch"]             = {{"t_submit_cpu_us", p.t_submit_cpu_us},
                                     {"t_launch_gpu_us", p.t_launch_gpu_us},
                                     {"t_barrier_gpu_us", p.t_barrier_gpu_us},
                                     {"r2_dispatch", p.r2_dispatch}, {"points", [&]{ nlohmann::json a = nlohmann::json::array(); for (auto& x : p.mb1_points) a.push_back({x.chain_len, x.nobar_gpu_us, x.bar_gpu_us}); return a; }()}};
    nlohmann::json bw_arr         = nlohmann::json::array();
    for (const auto& pt : p.bw_curve) bw_arr.push_back({pt.working_set_bytes, pt.bw_median_gbps, pt.bw_iqr_gbps});
    j["mb2_memory"]     = {{"l2_effective_bytes", p.l2_effective_bytes},
                           {"l2_effective_mb", static_cast<double>(p.l2_effective_bytes) / (1024.0 * 1024.0)},
                           {"bw_l2_gbps", p.bw_l2_gbps},
                           {"bw_dram_gbps", p.bw_dram_gbps},
                           {"bw_curve", bw_arr}};
    j["mb3_coalescing"] = {{"bw_l2_strided32_gbps", p.bw_l2_strided32_gbps},
                           {"coal_factor_l2_s32", p.coal_factor_l2_s32},
                           {"bw_dram_strided32_gbps", p.bw_dram_strided32_gbps},
                           {"coal_factor_dram_s32", p.coal_factor_dram_s32}};
    j["mb4_tiling"]     = {{"gflops_matmul_smem", p.gflops_matmul_smem},
                           {"gflops_matmul_cache", p.gflops_matmul_cache},
                           {"smem_vs_cache_speedup", p.smem_vs_cache_speedup},
                           {"prefer_shared_mem", p.prefer_shared_mem},
                           {"tail_efficiency", p.tail_efficiency},
                           {"fusion_reg_factor", p.fusion_reg_factor}};
    j["mb5_compute"]    = {{"pi_fma_gflops", p.pi_fma_gflops},
                           {"pi_gelu_erf_gelem_s", p.pi_gelu_erf_gelem_s},
                           {"pi_gelu_tanh_gelem_s", p.pi_gelu_tanh_gelem_s},
                           {"g_wave_quantum", p.g_wave_quantum},
                           {"pi_wave1_gflops", p.pi_wave1_gflops},
                           {"g_half_workgroups", p.g_half_workgroups},
                           {"g_sat_workgroups", p.g_sat_workgroups}};
    nlohmann::json drv_pts = nlohmann::json::array();
    for (const auto& pt : p.mb6_drv_points) drv_pts.push_back({pt.spirv_words, pt.time_ms});
    nlohmann::json fe_pts = nlohmann::json::array();
    for (const auto& pt : p.mb6_fe_points) fe_pts.push_back({pt.spirv_words, pt.time_ms});
    j["mb6_compile"]    = {{"alpha_0_slang_ms", p.alpha_0_slang_ms},
                           {"alpha_1_slang_ms_per_word", p.alpha_1_slang_ms_per_word},
                           {"alpha_0_gc_ms", p.alpha_0_gc_ms},
                           {"alpha_1_gc_ms_per_word", p.alpha_1_gc_ms_per_word},
                           {"beta_0_drv_ms", p.beta_0_drv_ms},
                           {"beta_1_drv_ms_per_word", p.beta_1_drv_ms_per_word},
                           {"beta_0_drv_smem_ms", p.beta_0_drv_smem_ms},
                           {"beta_1_drv_smem_ms_per_word", p.beta_1_drv_smem_ms_per_word},
                           {"t_cache_hit_ms", p.t_cache_hit_ms},
                           {"r2_fe_slang", p.r2_fe_slang},
                           {"r2_fe_gc", p.r2_fe_gc},
                           {"r2_drv", p.r2_drv},
                           {"drv_points", drv_pts},
                           {"fe_points", fe_pts}};
    return j.dump(2);
}

DeviceProfile Calibrator::from_json_string(const std::string& json_text) {
    auto j                     = nlohmann::json::parse(json_text);
    DeviceProfile p{};
    p.schema_version           = j.value("schema_version", 1u);
    p.calibration_wall_time_ms = j.value("calibration_wall_time_ms", 0.0);
    std::string rt             = j.value("runtime", "vulkan");
    p.props.runtime =
        (rt == "opencl") ? RuntimeKind::OpenCL : (rt == "level_zero" ? RuntimeKind::LevelZero : RuntimeKind::Vulkan);

    const auto& jd               = j.at("device");
    p.props.device_name          = jd.value("name", "");
    p.props.driver_name          = jd.value("driver", "");
    p.props.api_version_str      = jd.value("api_version", "");
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
    p.r2_dispatch      = j1.value("r2_dispatch", 1.0);
    if (j1.contains("points")) for (auto& r : j1["points"]) p.mb1_points.push_back({r[0].get<uint32_t>(), r[1].get<double>(), r[2].get<double>()});

    const auto& j2       = j.at("mb2_memory");
    p.l2_effective_bytes = j2.value("l2_effective_bytes", size_t{0});
    p.bw_l2_gbps         = j2.value("bw_l2_gbps", 100.0);
    p.bw_dram_gbps       = j2.value("bw_dram_gbps", 50.0);
    if (j2.contains("bw_curve") && j2["bw_curve"].is_array()) {
        for (const auto& row : j2["bw_curve"]) {
            if (row.size() >= 2) {
                p.bw_curve.push_back({row[0].get<size_t>(), row[1].get<double>(),
                                      row.size() >= 3 ? row[2].get<double>() : 0.0});
            }
        }
    }

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
    p.tail_efficiency       = j4.value("tail_efficiency", 1.0);
    p.fusion_reg_factor     = j4.value("fusion_reg_factor", 1.0);

    const auto& j5          = j.at("mb5_compute");
    p.pi_fma_gflops         = j5.value("pi_fma_gflops", 1000.0);
    p.pi_gelu_erf_gelem_s   = j5.value("pi_gelu_erf_gelem_s", 100.0);
    p.pi_gelu_tanh_gelem_s  = j5.value("pi_gelu_tanh_gelem_s", 100.0);
    p.g_wave_quantum        = j5.value("g_wave_quantum", j5.value("g_half_workgroups", 1u));
    p.pi_wave1_gflops       = j5.value("pi_wave1_gflops", p.pi_fma_gflops);
    p.g_half_workgroups     = j5.value("g_half_workgroups", 1u);
    p.g_sat_workgroups      = j5.value("g_sat_workgroups", 1u);

    const auto& j6              = j.at("mb6_compile");
    p.alpha_0_slang_ms          = j6.value("alpha_0_slang_ms", 0.0);
    p.alpha_1_slang_ms_per_word = j6.value("alpha_1_slang_ms_per_word", 0.0);
    p.alpha_0_gc_ms             = j6.value("alpha_0_gc_ms", 0.0);
    p.alpha_1_gc_ms_per_word    = j6.value("alpha_1_gc_ms_per_word", 0.0);
    p.beta_0_drv_ms             = j6.value("beta_0_drv_ms", 0.0);
    p.beta_1_drv_ms_per_word    = j6.value("beta_1_drv_ms_per_word", 0.0);
    p.beta_0_drv_smem_ms        = j6.value("beta_0_drv_smem_ms", p.beta_0_drv_ms);
    p.beta_1_drv_smem_ms_per_word = j6.value("beta_1_drv_smem_ms_per_word", p.beta_1_drv_ms_per_word);
    p.t_cache_hit_ms            = j6.value("t_cache_hit_ms", 0.0);
    p.r2_fe_slang               = j6.value("r2_fe_slang", 0.0);
    p.r2_fe_gc                  = j6.value("r2_fe_gc", 0.0);
    p.r2_drv                    = j6.value("r2_drv", 0.0);
    return p;
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

} // namespace mgcm
