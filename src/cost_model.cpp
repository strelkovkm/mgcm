#include "mgcm/cost_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace mgcm {

UniversalCostEngine::UniversalCostEngine(DeviceProfile profile)
    : prof_(std::move(profile)), topo_(TopologyPolicy::from_profile(prof_)) {}

UniversalCostEngine::UniversalCostEngine(DeviceProfile profile, TopologyPolicy topology)
    : prof_(std::move(profile)), topo_(topology) {}

double UniversalCostEngine::empirical_bw_gbps(double working_set_bytes, uint32_t stride_elems) const {
    double bw = 0.0;
    const double ws = std::max(1024.0, working_set_bytes);

    if (prof_.bw_curve.size() >= 2) {
        const auto& c = prof_.bw_curve;
        if (ws <= static_cast<double>(c.front().working_set_bytes)) {
            bw = c.front().bw_median_gbps;
        } else if (ws >= static_cast<double>(c.back().working_set_bytes)) {
            bw = c.back().bw_median_gbps;
        } else {
            auto it = std::lower_bound(
                c.begin(), c.end(), ws, [](const BandwidthPoint& pt, double val) {
                    return static_cast<double>(pt.working_set_bytes) < val;
                });
            size_t hi = static_cast<size_t>(std::distance(c.begin(), it));
            size_t lo = (hi > 0) ? hi - 1 : 0;
            double w0 = static_cast<double>(c[lo].working_set_bytes);
            double w1 = static_cast<double>(c[hi].working_set_bytes);
            double a  = (w1 > w0) ? (std::log2(ws / w0) / std::log2(w1 / w0)) : 0.0;
            bw        = (1.0 - a) * c[lo].bw_median_gbps + a * c[hi].bw_median_gbps;
        }
    } else {
        // Fallback for synthetic profiles without a full 22-point MB2 curve
        double c_l2 = static_cast<double>(std::max<size_t>(prof_.l2_effective_bytes, 256ull * 1024ull));
        bw          = (ws <= c_l2) ? prof_.bw_l2_gbps : prof_.bw_dram_gbps;
    }

    // Apply UMA memory contention factor when working set spills out of L2
    if (topo_.kind == TopologyKind::UnifiedUma &&
        ws > static_cast<double>(std::max<size_t>(prof_.l2_effective_bytes, 256ull * 1024ull))) {
        bw *= std::clamp(topo_.uma_contention_factor, 0.25, 1.0);
    }

    if (stride_elems > 1u) {
        double c_l2   = static_cast<double>(std::max<size_t>(prof_.l2_effective_bytes, 256ull * 1024ull));
        double coal32 = (ws <= c_l2) ? prof_.coal_factor_l2_s32 : prof_.coal_factor_dram_s32;
        double eff_s  = static_cast<double>(std::min<uint32_t>(stride_elems, 32u));
        double factor = std::clamp(coal32 * (32.0 / eff_s), 0.005, 1.0);
        bw *= factor;
    }
    return std::max(0.1, bw);
}

double UniversalCostEngine::wave_occupancy(uint32_t workgroups) const {
    double g      = static_cast<double>(std::max<uint32_t>(1u, workgroups));
    double g_w    = static_cast<double>(std::max<uint32_t>(1u, prof_.g_wave_quantum));
    double waves  = std::max(1.0, std::ceil(g / g_w));
    double eta_sp = g / (g_w * waves);

    double pi_max = std::max(1.0, prof_.pi_fma_gflops);
    double pi_w1  = (prof_.pi_wave1_gflops > 1.0) ? prof_.pi_wave1_gflops : pi_max;
    double kappa  = std::max(0.0, (pi_max / pi_w1) - 1.0);
    double eta_tm = waves / (waves + kappa);

    return std::clamp(eta_sp * eta_tm, 1e-4, 1.0);
}

double UniversalCostEngine::kernel_compile_ms(const KernelWorkload& kw) const {
    const double words = static_cast<double>(kw.est_spirv_words);
    double total_ms    = 0.0;

    if (kw.fe_cache_miss) {
        if (kw.frontend == CompilerKind::Slang) {
            total_ms += prof_.alpha_0_slang_ms + prof_.alpha_1_slang_ms_per_word * words;
        } else if (kw.frontend == CompilerKind::GraphCompiler) {
            total_ms += prof_.alpha_0_gc_ms + prof_.alpha_1_gc_ms_per_word * words;
        }
    }

    if (kw.drv_cache_miss) {
        total_ms += prof_.beta_0_drv_ms + prof_.beta_1_drv_ms_per_word * words;
    } else {
        total_ms += prof_.t_cache_hit_ms;
    }
    return std::max(0.0, total_ms);
}

PlanEstimate UniversalCostEngine::estimate_partition(const PartitionWorkload& pw, AblationMode ablation) const {
    const uint32_t num_kernels = static_cast<uint32_t>(pw.kernels.size());
    double t_dispatch_us =
        static_cast<double>(num_kernels) * prof_.t_launch_gpu_us +
        static_cast<double>(num_kernels > 1u ? num_kernels - 1u : 0u) * prof_.t_barrier_gpu_us +
        static_cast<double>(pw.host_sync_count) * prof_.t_submit_cpu_us;
    if (ablation == AblationMode::NoDispatchOverhead) {
        t_dispatch_us = 0.0;
    }

    double sum_comp_us = 0.0;
    double sum_mem_us  = 0.0;
    double sum_krn_us  = 0.0;
    double sum_cmp_ms  = 0.0;

    for (const auto& kw : pw.kernels) {
        double occ = wave_occupancy(kw.workgroups);

        // Compute stage across GEMM, vector FMA, and activation GElem/s pipes
        double t_comp_us = 0.0;
        if (kw.flops_contraction_padded > 0.0) {
            double occ_ref = wave_occupancy(256u);
            double pi_gemm = (kw.uses_shared_mem ? prof_.gflops_matmul_smem : prof_.gflops_matmul_cache) / occ_ref;
            pi_gemm        = std::max(1.0, pi_gemm);
            double eta_t   = kw.has_unaligned_tail ? std::clamp(prof_.tail_efficiency, 0.1, 1.0) : 1.0;
            double eta_f   = kw.has_fused_heavy_act ? std::clamp(prof_.fusion_reg_factor, 0.1, 1.0) : 1.0;
            t_comp_us += (kw.flops_contraction_padded / (occ * eta_t * eta_f * pi_gemm)) * 1e-3;
        }
        if (kw.flops_vector_fma_padded > 0.0) {
            t_comp_us += (kw.flops_vector_fma_padded / (occ * std::max(1.0, prof_.pi_fma_gflops))) * 1e-3;
        }
        if (kw.elems_act_padded > 0.0 && kw.act_pipe != ActivationPipe::None) {
            double pi_act = (kw.act_pipe == ActivationPipe::GeluErf) ? prof_.pi_gelu_erf_gelem_s
                                                                     : prof_.pi_gelu_tanh_gelem_s;
            t_comp_us += (kw.elems_act_padded / (occ * std::max(0.1, pi_act))) * 1e-3;
        }

        // Memory stage via empirical MB2 log-linear curve + MB3 stride coalescing
        double t_mem_us = 0.0;
        if (ablation != AblationMode::NoMemoryTraffic) {
            const double bw_l2_ref = std::max(1.0, prof_.bw_l2_gbps);
            auto mem_term_us = [&](double bytes, double ws, uint32_t stride) {
                double bw = empirical_bw_gbps(ws, stride);
                if (kw.flops_contraction_padded > 0.0 && stride == 1u) {
                    return bytes * std::max(0.0, (1.0 / bw) - (1.0 / bw_l2_ref)) * 1e-3;
                }
                return (bytes / bw) * 1e-3;
            };
            if (kw.bytes_cold > 0.0) {
                t_mem_us += mem_term_us(kw.bytes_cold, kw.cold_working_set_bytes, 1u);
            }
            if (kw.bytes_reuse > 0.0) {
                t_mem_us += mem_term_us(kw.bytes_reuse, kw.reuse_working_set_bytes, kw.read_stride_elems);
            }
        }

        sum_comp_us += t_comp_us;
        sum_mem_us += t_mem_us;
        sum_krn_us += std::hypot(t_comp_us, t_mem_us);

        if (ablation != AblationMode::NoCompileCost) {
            sum_cmp_ms += kernel_compile_ms(kw);
        }
    }

    double ext_cut_us =
        (ablation == AblationMode::NoMemoryTraffic) ? 0.0 : topo_.external_cut_us(pw.ext_cut_bytes, false);

    PlanEstimate est{};
    est.plan          = pw.plan;
    est.t_compile_us  = sum_cmp_ms * 1e3;
    est.t_compute_us  = sum_comp_us;
    est.t_memory_us   = sum_mem_us + ext_cut_us;
    est.t_dispatch_us = t_dispatch_us;
    est.t_execute_us  = t_dispatch_us + sum_krn_us + ext_cut_us;
    return est;
}

std::vector<PlanEstimate> UniversalCostEngine::evaluate_block(const IBlockModel& block, AblationMode ablation) const {
    auto candidates = block.build_candidates(prof_.g_wave_quantum);
    std::vector<PlanEstimate> out;
    out.reserve(candidates.size());
    for (const auto& pw : candidates) {
        out.push_back(estimate_partition(pw, ablation));
    }
    return out;
}

PartitionPlan UniversalCostEngine::select_optimal(const IBlockModel& block, uint64_t model_r_runs,
                                                  AblationMode ablation) const {
    auto ests     = evaluate_block(block, ablation);
    uint64_t reff = std::max<uint64_t>(1ull, model_r_runs) * std::max<uint32_t>(1u, block.instance_count());
    auto best     = std::min_element(ests.begin(), ests.end(), [&](const PlanEstimate& a, const PlanEstimate& b) {
        return a.total_cost_us(reff) < b.total_cost_us(reff);
    });
    return best->plan;
}

std::vector<HorizonSegment> UniversalCostEngine::compute_horizon(const IBlockModel& block,
                                                                 AblationMode ablation) const {
    auto ests         = evaluate_block(block, ablation);
    const double k_sm = static_cast<double>(std::max<uint32_t>(1u, block.instance_count()));

    struct Line {
        PartitionPlan plan;
        double        b; // T_compile_us
        double        m; // K_same * T_execute_us
    };
    std::vector<Line> lines;
    lines.reserve(ests.size());
    for (const auto& e : ests) {
        lines.push_back({e.plan, e.t_compile_us, k_sm * e.t_execute_us});
    }

    // Sort by descending slope (m) and ascending intercept (b) for standard Lower Envelope hull
    std::sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) {
        if (std::abs(a.m - b.m) > 1e-9) {
            return a.m > b.m;
        }
        return a.b < b.b;
    });

    struct HullEntry {
        Line   line;
        double x_left; // Leftmost R >= 1 where this line starts winning
    };
    std::vector<HullEntry> hull;

    for (const auto& ln : lines) {
        if (!hull.empty() && std::abs(hull.back().line.m - ln.m) <= 1e-9) {
            continue; // Same slope with higher or equal intercept is strictly dominated
        }
        while (!hull.empty()) {
            const auto& prev = hull.back();
            double x_int     = (ln.b - prev.line.b) / (prev.line.m - ln.m);
            if (x_int <= prev.x_left) {
                hull.pop_back(); // Previous line is never optimal on [x_left, +inf)
            } else {
                hull.push_back({ln, x_int});
                break;
            }
        }
        if (hull.empty()) {
            hull.push_back({ln, 1.0});
        }
    }

    // Clip segments to integer model runs R >= 1
    std::vector<HorizonSegment> segs;
    for (size_t i = 0; i < hull.size(); ++i) {
        double l_x = std::max(1.0, hull[i].x_left);
        double r_x = (i + 1 < hull.size()) ? hull[i + 1].x_left : std::numeric_limits<double>::infinity();
        if (r_x <= 1.0) {
            continue;
        }
        auto r_start = static_cast<uint64_t>(std::ceil(l_x));
        uint64_t r_end =
            std::isfinite(r_x) ? static_cast<uint64_t>(std::ceil(r_x)) : 0ull; // 0 == +infinity
        if (r_end != 0ull && r_end <= r_start) {
            continue;
        }
        if (!segs.empty() && segs.back().plan == hull[i].line.plan) {
            segs.back().r_end_exclusive = r_end;
        } else {
            segs.push_back({r_start, r_end, hull[i].line.plan});
        }
    }
    return segs;
}

double UniversalCostEngine::compute_break_even_r(const IBlockModel& block, PartitionPlan candidate_plan,
                                                 PartitionPlan reference_plan, AblationMode ablation) const {
    auto ests         = evaluate_block(block, ablation);
    const double k_sm = static_cast<double>(std::max<uint32_t>(1u, block.instance_count()));

    const PlanEstimate* cand = nullptr;
    const PlanEstimate* ref  = nullptr;
    for (const auto& e : ests) {
        if (e.plan == candidate_plan) cand = &e;
        if (e.plan == reference_plan) ref = &e;
    }
    if (!cand || !ref) {
        return -1.0;
    }
    double d_comp = cand->t_compile_us - ref->t_compile_us;
    double d_exec = k_sm * (ref->t_execute_us - cand->t_execute_us);
    if (d_comp <= 0.0 && d_exec >= 0.0) {
        return 0.0;
    }
    if (d_exec <= 1e-9) {
        return -1.0;
    }
    return std::max(0.0, d_comp / d_exec);
}

DynamicMDecision UniversalCostEngine::analyze_dynamic_m(const MlpConfig& base_cfg, uint32_t m_min, uint32_t m_max,
                                                        uint64_t model_r_runs, AblationMode ablation) const {
    auto eval_at = [&](uint32_t m) {
        MlpConfig c    = base_cfg;
        c.M            = m;
        c.is_dynamic_m = true;
        return select_optimal(MlpBlock(c), model_r_runs, ablation);
    };

    DynamicMDecision dec{};
    dec.low_m_plan  = eval_at(m_min);
    dec.high_m_plan = eval_at(m_max);
    if (dec.low_m_plan == dec.high_m_plan || m_max <= m_min) {
        dec.has_crossover = false;
        return dec;
    }

    uint32_t lo = m_min, hi = m_max;
    while (lo + 1u < hi) {
        uint32_t mid = lo + (hi - lo) / 2u;
        if (eval_at(mid) == dec.low_m_plan) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    dec.has_crossover = true;
    dec.m_threshold   = hi;
    return dec;
}

} // namespace mgcm
