#include "mgcm/calibrator.hpp"
#include "mgcm/cost_model.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

mgcm::DeviceProfile make_synthetic_780m() {
    mgcm::DeviceProfile p{};
    p.props.device_name          = "Synthetic Radeon 780M";
    p.t_launch_gpu_us            = 0.16;
    p.t_barrier_gpu_us           = 0.46;
    p.l2_effective_bytes         = 2ull * 1024 * 1024;
    p.bw_l2_gbps                 = 1267.0;
    p.bw_dram_gbps               = 101.5;
    p.coal_factor_l2_s32         = 0.087;
    p.coal_factor_dram_s32       = 0.035;
    p.gflops_matmul_smem         = 1400.0;
    p.gflops_matmul_cache        = 1266.0;
    p.prefer_shared_mem          = true;
    p.pi_fma_gflops              = 4085.0;
    p.pi_gelu_tanh_gelem_s       = 134.5;
    p.pi_gelu_erf_gelem_s        = 162.6;
    p.g_half_workgroups          = 6;
    p.g_sat_workgroups           = 24;
    p.t_compile_small_cpu_ms     = 0.25;
    p.t_compile_tiled_cpu_ms     = 4.90;
    p.t_compile_fused_erf_cpu_ms = 5.71;
    return p;
}

} // namespace

int main() {
    // 1. JSON round-trip test
    auto p0       = make_synthetic_780m();
    std::string j = mgcm::Calibrator::to_json_string(p0);
    auto p1       = mgcm::Calibrator::from_json_string(j);
    assert(p1.l2_effective_bytes == p0.l2_effective_bytes);
    assert(std::abs(p1.bw_dram_gbps - p0.bw_dram_gbps) < 1e-6);

    mgcm::CostModel cm(p1);

    // 2. Execution monotonicity: P3_FullFusion must have lowest T_execute
    mgcm::SubgraphQuery q_bert{128, 3072, 768, true, mgcm::GeluKind::Erf, true, false};
    auto ests = cm.evaluate_all(q_bert);
    assert(ests[3].t_execute_us < ests[2].t_execute_us);
    assert(ests[2].t_execute_us < ests[0].t_execute_us);

    // 3. Cached primitives trade-off: at R=1 with cached base MatMul, P2 beats P3 due to compile cost; at R=10000, P3 wins!
    mgcm::SubgraphQuery q_cached{128, 512, 256, false, mgcm::GeluKind::Erf, true, true};
    auto opt_r1    = cm.select_optimal(q_cached, 1, mgcm::AblationMode::None);
    auto opt_r10k  = cm.select_optimal(q_cached, 10000, mgcm::AblationMode::None);
    double r_star  = cm.compute_break_even_r(q_cached, mgcm::PartitionPlan::P3_FullFusion, mgcm::PartitionPlan::P2_FusedEpilogue);
    assert(opt_r1 != mgcm::PartitionPlan::P3_FullFusion);
    assert(opt_r10k == mgcm::PartitionPlan::P3_FullFusion);
    assert(r_star > 1.0);

    // 4. Ablation NoCompileCost immediately picks P3 even at R=1
    assert(cm.select_optimal(q_cached, 1, mgcm::AblationMode::NoCompileCost) == mgcm::PartitionPlan::P3_FullFusion);

    // 5. Untiled Whisper trB=1 blows up into multi-second TDR territory on 2 MB L2 (> 1,000,000 us)!
    mgcm::SubgraphQuery q_whisper_untiled{1500, 3072, 768, true, mgcm::GeluKind::Erf, false, false};
    mgcm::SubgraphQuery q_whisper_tiled{1500, 3072, 768, true, mgcm::GeluKind::Erf, true, false};
    auto e_untiled = cm.estimate_plan(q_whisper_untiled, mgcm::PartitionPlan::P3_FullFusion);
    auto e_tiled   = cm.estimate_plan(q_whisper_tiled, mgcm::PartitionPlan::P3_FullFusion);
    assert(e_untiled.t_execute_us > 1500000.0); // > 1.5s TDR danger zone!
    assert(e_tiled.t_execute_us < 25000.0);     // < 25 ms with 2D tiling!

    std::cout << "[PASS] All CostModel unit tests passed! (R* sample = " << r_star
              << ", Whisper untiled = " << (e_untiled.t_execute_us * 1e-3)
              << " ms vs tiled = " << (e_tiled.t_execute_us * 1e-3) << " ms)\n";
    return 0;
}
