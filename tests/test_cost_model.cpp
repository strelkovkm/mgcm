#include "mgcm/blocks/mlp_block.hpp"
#include "mgcm/calibrator.hpp"
#include "mgcm/cost_model.hpp"
#include "mgcm/integration/openvino_bridge_stub.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

mgcm::DeviceProfile make_synthetic_profile() {
    mgcm::DeviceProfile p{};
    p.schema_version             = mgcm::kDeviceProfileSchemaVersion;
    p.calibration_wall_time_ms   = 1350.0;
    p.props.device_name          = "Synthetic Radeon 780M";
    p.props.is_integrated_uma    = true;
    p.t_submit_cpu_us            = 39.0;
    p.t_launch_gpu_us            = 0.16;
    p.t_barrier_gpu_us           = 0.46;
    p.l2_effective_bytes         = 2ull * 1024 * 1024;
    p.bw_l2_gbps                 = 1200.0;
    p.bw_dram_gbps               = 100.0;
    p.bw_curve                   = {{32768, 1200.0, 10.0},
                                    {2097152, 1100.0, 20.0},
                                    {3145728, 120.0, 15.0},
                                    {268435456, 100.0, 1.0}};
    p.coal_factor_l2_s32         = 0.087;
    p.coal_factor_dram_s32       = 0.035;
    p.gflops_matmul_smem         = 1400.0;
    p.gflops_matmul_cache        = 1266.0;
    p.prefer_shared_mem          = true;
    p.tail_efficiency            = 0.88;
    p.fusion_reg_factor          = 1.0;
    p.pi_fma_gflops              = 4085.0;
    p.pi_gelu_erf_gelem_s        = 162.6;
    p.pi_gelu_tanh_gelem_s       = 134.5;
    p.g_wave_quantum             = 6;
    p.pi_wave1_gflops            = 2338.7;
    p.beta_0_drv_ms              = 0.12;
    p.beta_1_drv_ms_per_word     = 0.000483;
    p.t_cache_hit_ms             = 0.01;
    p.r2_drv                     = 0.999;
    return p;
}

} // namespace

int main() {
    // 1. JSON v2 round-trip including bw_curve
    auto p0 = make_synthetic_profile();
    auto p1 = mgcm::Calibrator::from_json_string(mgcm::Calibrator::to_json_string(p0));
    assert(p1.schema_version == mgcm::kDeviceProfileSchemaVersion);
    assert(p1.bw_curve.size() == 4);
    assert(std::abs(p1.beta_1_drv_ms_per_word - p0.beta_1_drv_ms_per_word) < 1e-9);

    // 2. Analytical SPIR-V word estimator matches compiled Slang shaders within 2%
    auto s64 = mgcm::MatmulSchedule::tiled_64x64_smem();
    uint32_t w_noop  = mgcm::MlpBlock::estimate_spirv_words(mgcm::CompilerKind::Slang, false, 1, s64, mgcm::EpilogueProfile::none(), 1);
    uint32_t w_smem  = mgcm::MlpBlock::estimate_spirv_words(mgcm::CompilerKind::Slang, true, 4, s64, mgcm::EpilogueProfile::none(), 16);
    uint32_t w_fused = mgcm::MlpBlock::estimate_spirv_words(mgcm::CompilerKind::Slang, true, 4, s64, mgcm::EpilogueProfile::gelu_erf(), 16);
    assert(w_noop >= 240 && w_noop <= 320);
    assert(w_smem >= 9600 && w_smem <= 10200);
    assert(w_fused >= 11300 && w_fused <= 12000);

    mgcm::UniversalCostEngine eng(p1);

    // 3. Piecewise log2 empirical BW interpolation (no magic exponents)
    assert(eng.empirical_bw_gbps(1024 * 1024) > 1000.0);
    assert(eng.empirical_bw_gbps(16 * 1024 * 1024) < 130.0);

    // 4. Lower Envelope Horizon matches brute-force select_optimal(R) at every segment boundary
    mgcm::MlpConfig cfg{};
    cfg.M = 128; cfg.N = 512; cfg.K = 256;
    cfg.fe_cached = true; cfg.base_mm_drv_cached = true;
    mgcm::MlpBlock blk(cfg);
    auto horizon = eng.compute_horizon(blk);
    assert(!horizon.empty());
    assert(horizon.front().r_start_inclusive == 1);
    assert(horizon.back().r_end_exclusive == 0);
    for (const auto& seg : horizon) {
        assert(eng.select_optimal(blk, seg.r_start_inclusive) == seg.plan);
        if (seg.r_end_exclusive > seg.r_start_inclusive + 1) {
            assert(eng.select_optimal(blk, seg.r_end_exclusive - 1) == seg.plan);
        }
    }

    // 5. Negative fusion under register pressure (fusion_reg_factor = 0.70, huge K, tiny M/N)
    auto p_reg = p1;
    p_reg.fusion_reg_factor = 0.70;
    mgcm::UniversalCostEngine eng_reg(p_reg);
    mgcm::MlpConfig cfg_huge_k{};
    cfg_huge_k.M = 64; cfg_huge_k.N = 64; cfg_huge_k.K = 8192;
    cfg_huge_k.fe_cached = true; cfg_huge_k.base_mm_drv_cached = false;
    auto est_reg = eng_reg.evaluate_block(mgcm::MlpBlock(cfg_huge_k));
    assert(est_reg[2].t_execute_us < est_reg[3].t_execute_us); // Split beats FullFusion even at R -> inf!
    assert(eng_reg.select_optimal(mgcm::MlpBlock(cfg_huge_k), 1000000) != mgcm::PartitionPlan::P3_FullFusion);

    // 6. Multiplicity K_same = 12 scales break-even R* down by 12x
    mgcm::MlpConfig c1 = cfg; c1.M = 1500; c1.N = 3072; c1.K = 768; c1.instance_count = 1;
    mgcm::MlpConfig c12 = c1; c12.instance_count = 12;
    double r1  = eng.compute_break_even_r(mgcm::MlpBlock(c1), mgcm::PartitionPlan::P3_FullFusion, mgcm::PartitionPlan::P2_FusedEpilogue);
    double r12 = eng.compute_break_even_r(mgcm::MlpBlock(c12), mgcm::PartitionPlan::P3_FullFusion, mgcm::PartitionPlan::P2_FusedEpilogue);
    assert(std::abs(r1 - 12.0 * r12) < 1e-6);

    // 7. OpenVINO bridge stub: K_same counting + fan-out legality check (add_single_consumer == false forbids P2 & P3!)
    std::vector<mgcm::ov_bridge::MlpMatchDescriptor> matches;
    for (int i = 0; i < 12; ++i) {
        matches.push_back({reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1)), c1, true, true});
    }
    matches.push_back({reinterpret_cast<void*>(99), c1, true, false}); // Residual skip branch on Add output!
    auto decs = mgcm::ov_bridge::SubgraphDecisionBatch::solve(matches, eng, 100);
    assert(decs.size() == 13);
    assert(decs[0].k_same_in_model == 12);
    assert(decs[0].chosen_plan == mgcm::PartitionPlan::P3_FullFusion);
    assert(decs[12].k_same_in_model == 1);
    assert(decs[12].chosen_plan != mgcm::PartitionPlan::P3_FullFusion &&
           decs[12].chosen_plan != mgcm::PartitionPlan::P2_FusedEpilogue);

    // 8. Dynamic-M crossover finder
    auto dyn = eng.analyze_dynamic_m(c1, 1, 4096, 5);
    assert(dyn.has_crossover && dyn.m_threshold > 1 && dyn.m_threshold < 4096);

    // 9. Ablation NoCompileCost chooses P3 even at R=1 when fusion_reg_factor == 1.0
    assert(eng.select_optimal(blk, 1, mgcm::AblationMode::NoCompileCost) == mgcm::PartitionPlan::P3_FullFusion);

    std::cout << "[PASS] All 9 UniversalCostEngine & OpenVINO Bridge tests passed!\n";
    return 0;
}
