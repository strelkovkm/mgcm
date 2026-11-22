#include "mgcm/cost_model.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mgcm {

CostModel::CostModel(DeviceProfile profile) : prof_(std::move(profile)) {}

double CostModel::l2_hit_fraction(double working_set_bytes) const {
    double c_l2 = static_cast<double>(std::max<size_t>(prof_.l2_effective_bytes, 256ull * 1024ull));
    double ratio = working_set_bytes / c_l2;
    // Smooth logistic step centered at C_L2 (steepness gamma = 4 matches empirical MB2 cliff)
    return 1.0 / (1.0 + std::pow(ratio, 4.0));
}

double CostModel::effective_bw_gbps(double working_set_bytes, bool strided32) const {
    double hit = l2_hit_fraction(working_set_bytes);
    double bw_l2   = strided32 ? (prof_.bw_l2_gbps * prof_.coal_factor_l2_s32) : prof_.bw_l2_gbps;
    double bw_dram = strided32 ? (prof_.bw_dram_gbps * prof_.coal_factor_dram_s32) : prof_.bw_dram_gbps;
    // Harmonic blend of L2-resident and DRAM-spilling service times
    double inv_bw = hit / std::max(0.1, bw_l2) + (1.0 - hit) / std::max(0.1, bw_dram);
    return 1.0 / inv_bw;
}

PlanEstimate CostModel::estimate_plan(const SubgraphQuery& q, PartitionPlan plan, AblationMode ablation) const {
    const double M = static_cast<double>(q.M);
    const double N = static_cast<double>(q.N);
    const double K = static_cast<double>(q.K);

    const double bytes_a    = M * K * 4.0;
    const double bytes_b    = K * N * 4.0;
    const double bytes_bias = N * 4.0;
    const double bytes_c    = M * N * 4.0;

    // 1. MatMul kernel compute & memory cost
    uint32_t wg_m = (q.M + 63u) / 64u;
    uint32_t wg_n = (q.N + 63u) / 64u;
    double num_wgs = q.tiled_matmul ? static_cast<double>(wg_m * wg_n) : std::ceil((M * N) / 256.0);

    double g_half = static_cast<double>(std::max<uint32_t>(1u, prof_.g_half_workgroups));
    double eta_occ = num_wgs / (num_wgs + g_half);

    double sustained_gemm_gflops = q.tiled_matmul
                                       ? (prof_.prefer_shared_mem ? prof_.gflops_matmul_smem : prof_.gflops_matmul_cache)
                                       : (0.35 * prof_.pi_fma_gflops);
    sustained_gemm_gflops = std::max(10.0, sustained_gemm_gflops);

    double flops_matmul   = 2.0 * M * N * K;
    double t_comp_mm_us   = (flops_matmul / (sustained_gemm_gflops * eta_occ)) * 1e-3;

    double t_mem_mm_us = 0.0;
    if (q.tiled_matmul) {
        // 2D-tiled (BM=64, BN=64): each 64xK panel of A is reused acrosswg_n, B reused across wg_m via L2/SMEM
        double ws_ab = bytes_a + bytes_b;
        double bw_ab = effective_bw_gbps(ws_ab, false);
        double dram_traffic_ab = bytes_a + bytes_b + (1.0 - l2_hit_fraction(ws_ab)) * ((wg_n - 1) * bytes_a * 0.25 + (wg_m - 1) * bytes_b * 0.25);
        t_mem_mm_us = ((dram_traffic_ab + bytes_c) / bw_ab) * 1e-3;
    } else {
        // Untiled 1-thread-per-output: every row m re-reads entire matrix B; trB=1 causes strided reads on B!
        double bw_a = effective_bw_gbps(bytes_a, false);
        double bw_b = effective_bw_gbps(bytes_b, q.transpose_b);
        double effective_b_reads = M * bytes_b * (q.transpose_b ? 1.0 : 0.25);
        t_mem_mm_us = (bytes_a / bw_a + effective_b_reads / bw_b + bytes_c / prof_.bw_dram_gbps) * 1e-3;
    }

    // 2. Epilogue (Add + GELU) compute cost
    double sfu_rate = (q.gelu == GeluKind::Erf) ? prof_.pi_gelu_erf_gelem_s : prof_.pi_gelu_tanh_gelem_s;
    sfu_rate = std::max(1.0, sfu_rate);
    double elems_mn        = M * N;
    double t_comp_add_us   = (elems_mn / std::max(10.0, prof_.pi_fma_gflops * eta_occ)) * 1e-3;
    double t_comp_gelu_us  = (elems_mn / (sfu_rate * eta_occ)) * 1e-3;

    // 3. Intermediate [M, N] memory round-trips depending on PartitionPlan
    // Each cut between kernels writes bytes_c and reads bytes_c back (hitting L2 if bytes_c < C_L2, else DRAM!)
    double bw_interm = effective_bw_gbps(bytes_c, false);
    double one_roundtrip_mem_us = ((2.0 * bytes_c) / bw_interm) * 1e-3;
    double bias_read_us         = (bytes_bias / effective_bw_gbps(bytes_bias, false)) * 1e-3;

    uint32_t num_kernels = 1;
    double t_interm_mem_us = 0.0;
    double t_compile_ms    = 0.0;

    const double comp_small_ms = prof_.t_compile_small_cpu_ms + prof_.t_compile_small_lazy_ms;
    const double comp_mm_ms    = prof_.t_compile_tiled_cpu_ms + prof_.t_compile_tiled_lazy_ms;
    const double comp_fused_ms = prof_.t_compile_fused_erf_cpu_ms + prof_.t_compile_fused_erf_lazy_ms;
    const double epilogue_compile_delta_ms = std::max(0.1, comp_fused_ms - comp_mm_ms);

    switch (plan) {
        case PartitionPlan::P0_SplitAll:
            num_kernels     = 3;
            t_interm_mem_us = 2.0 * one_roundtrip_mem_us + bias_read_us;
            t_compile_ms    = q.base_primitives_cached ? 0.0 : (comp_mm_ms + 2.0 * comp_small_ms);
            break;
        case PartitionPlan::P1_FusedBias:
            num_kernels     = 2;
            t_interm_mem_us = 1.0 * one_roundtrip_mem_us + bias_read_us;
            t_compile_ms    = q.base_primitives_cached ? comp_small_ms : (comp_mm_ms + 0.15 * epilogue_compile_delta_ms + comp_small_ms);
            break;
        case PartitionPlan::P2_FusedEpilogue:
            num_kernels     = 2;
            t_interm_mem_us = 1.0 * one_roundtrip_mem_us + bias_read_us;
            t_compile_ms    = (q.base_primitives_cached ? 0.0 : comp_mm_ms) + 1.5 * comp_small_ms;
            break;
        case PartitionPlan::P3_FullFusion:
            num_kernels     = 1;
            t_interm_mem_us = bias_read_us;
            t_compile_ms    = comp_fused_ms;
            break;
    }

    // Apply ablations cleanly
    if (ablation == AblationMode::NoMemoryTraffic) {
        t_mem_mm_us     = 0.0;
        t_interm_mem_us = 0.0;
    }
    double t_dispatch_us = static_cast<double>(num_kernels) * prof_.t_launch_gpu_us +
                           static_cast<double>(num_kernels - 1u) * prof_.t_barrier_gpu_us;
    if (ablation == AblationMode::NoDispatchOverhead) {
        t_dispatch_us = 0.0;
    }
    if (ablation == AblationMode::NoCompileCost) {
        t_compile_ms = 0.0;
    }

    // Smooth L2-norm Roofline overlap for MatMul + epilogue stages
    double t_mm_stage_us = std::hypot(t_comp_mm_us, t_mem_mm_us);
    double t_epilogue_us = 0.0;
    if (plan == PartitionPlan::P3_FullFusion) {
        // In-register epilogue overlaps with final tile writeout
        t_epilogue_us = t_comp_add_us + t_comp_gelu_us + t_interm_mem_us;
    } else {
        t_epilogue_us = std::hypot(t_comp_add_us + t_comp_gelu_us, t_interm_mem_us);
    }

    PlanEstimate est{};
    est.plan          = plan;
    est.t_compile_us  = t_compile_ms * 1e3;
    est.t_compute_us  = t_comp_mm_us + t_comp_add_us + t_comp_gelu_us;
    est.t_memory_us   = t_mem_mm_us + t_interm_mem_us;
    est.t_dispatch_us = t_dispatch_us;
    est.t_execute_us  = t_dispatch_us + t_mm_stage_us + t_epilogue_us;
    return est;
}

std::array<PlanEstimate, 4> CostModel::evaluate_all(const SubgraphQuery& q, AblationMode ablation) const {
    return {
        estimate_plan(q, PartitionPlan::P0_SplitAll, ablation),
        estimate_plan(q, PartitionPlan::P1_FusedBias, ablation),
        estimate_plan(q, PartitionPlan::P2_FusedEpilogue, ablation),
        estimate_plan(q, PartitionPlan::P3_FullFusion, ablation),
    };
}

PartitionPlan CostModel::select_optimal(const SubgraphQuery& q, uint64_t r_runs, AblationMode ablation) const {
    auto all = evaluate_all(q, ablation);
    auto best = std::min_element(all.begin(), all.end(), [&](const PlanEstimate& a, const PlanEstimate& b) {
        return a.total_cost_us(r_runs) < b.total_cost_us(r_runs);
    });
    return best->plan;
}

double CostModel::compute_break_even_r(const SubgraphQuery& q, PartitionPlan candidate_plan,
                                       PartitionPlan reference_plan) const {
    auto cand = estimate_plan(q, candidate_plan);
    auto ref  = estimate_plan(q, reference_plan);
    double d_comp = cand.t_compile_us - ref.t_compile_us;
    double d_exec = ref.t_execute_us - cand.t_execute_us;
    if (d_comp <= 0.0 && d_exec >= 0.0) {
        return 0.0;
    }
    if (d_exec <= 1e-9) {
        return -1.0;
    }
    return std::max(0.0, d_comp / d_exec);
}

} // namespace mgcm
