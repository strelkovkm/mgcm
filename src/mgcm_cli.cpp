#include "mgcm/backends/vulkan_backend.hpp"
#include "mgcm/blocks/mlp_block.hpp"
#include "mgcm/calibrator.hpp"
#include "mgcm/cost_model.hpp"
#include "mgcm/frontend_probe.hpp"
#include "mgcm/profile_cache.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string format_horizon(const std::vector<mgcm::HorizonSegment>& segs) {
    std::ostringstream oss;
    for (size_t i = 0; i < segs.size(); ++i) {
        if (i > 0) oss << " -> ";
        oss << "[" << segs[i].r_start_inclusive << "..";
        if (segs[i].r_end_exclusive == 0) oss << "inf";
        else oss << segs[i].r_end_exclusive;
        oss << "):" << mgcm::to_string(segs[i].plan);
    }
    return oss.str();
}

void run_evaluation_report(const std::string& json_path, const std::string& phase_csv) {
    std::ifstream ifs(json_path);
    if (!ifs) {
        throw std::runtime_error("Cannot open profile JSON: " + json_path);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    auto prof = mgcm::Calibrator::from_json_string(ss.str());
    mgcm::UniversalCostEngine eng(prof);

    std::cout << "==================================================================================================\n"
              << "UniversalCostEngine Report : " << prof.props.device_name << " (" << prof.props.driver_name << ")\n"
              << "Topology: " << mgcm::to_string(eng.topology().kind)
              << " | L2=" << (prof.l2_effective_bytes / (1024 * 1024)) << "MB"
              << " | G_wave=" << prof.g_wave_quantum
              << " | tail_eff=" << std::fixed << std::setprecision(3) << prof.tail_efficiency
              << " | fusion_reg=" << prof.fusion_reg_factor
              << " | R2_drv=" << prof.r2_drv << " | R2_slang=" << prof.r2_fe_slang << "\n"
              << "==================================================================================================\n";

    auto make_cfg = [](uint32_t m, uint32_t n, uint32_t k, bool trb, mgcm::EpilogueProfile act,
                       mgcm::MatmulSchedule sched, bool cached, uint32_t k_same) {
        mgcm::MlpConfig c{};
        c.M = m; c.N = n; c.K = k; c.transpose_b = trb; c.activation = act; c.schedule = sched;
        c.frontend = mgcm::CompilerKind::Slang; c.fe_cached = true; c.base_mm_drv_cached = cached; c.instance_count = k_same;
        return c;
    };

    struct CaseDesc { const char* label; mgcm::MlpConfig cfg; };
    const auto smem64 = mgcm::MatmulSchedule::tiled_64x64_smem();
    const auto naive  = mgcm::MatmulSchedule::naive_untiled();
    const auto erf    = mgcm::EpilogueProfile::gelu_erf();
    const auto relu   = mgcm::EpilogueProfile::relu();

    const std::vector<CaseDesc> cases = {
        {"Tiny [32,64,128] cold ERF",      make_cfg(32, 64, 128, true, erf, smem64, false, 1)},
        {"Synth [128,512,256] cold ERF",   make_cfg(128, 512, 256, false, erf, smem64, false, 1)},
        {"Synth [128,512,256] cold ReLU",  make_cfg(128, 512, 256, false, relu, smem64, false, 1)},
        {"Synth [128,512,256] cached ERF", make_cfg(128, 512, 256, false, erf, smem64, true, 1)},
        {"BERT [128,3072,768] K=12 cch",   make_cfg(128, 3072, 768, true, erf, smem64, true, 12)},
        {"Whisper [1500,3072,768] K=12",   make_cfg(1500, 3072, 768, true, erf, smem64, true, 12)},
        {"Whisper [1500,3072,768] NoTile", make_cfg(1500, 3072, 768, true, erf, naive, true, 12)},
        {"Large [4096,4096,1024] cached",  make_cfg(4096, 4096, 1024, true, erf, smem64, true, 1)},
    };

    std::cout << std::left << std::setw(32) << "Workload" << std::right << std::setw(13) << "T_exec(P2)"
              << std::setw(13) << "T_exec(P3)" << "   " << std::left << "Optimal Horizon (Model Runs R)\n"
              << std::string(106, '-') << "\n";

    for (const auto& c : cases) {
        mgcm::MlpBlock blk(c.cfg);
        auto ests = eng.evaluate_block(blk);
        auto hor  = eng.compute_horizon(blk);
        std::cout << std::left << std::setw(32) << c.label << std::right << std::fixed << std::setprecision(2)
                  << std::setw(11) << ests[2].t_execute_us << "us" << std::setw(11) << ests[3].t_execute_us << "us   "
                  << std::left << format_horizon(hor) << "\n";
    }

    auto dyn_cfg = make_cfg(1, 3072, 768, true, relu, smem64, true, 12);
    auto dyn     = eng.analyze_dynamic_m(dyn_cfg, 1, 4096, 5);
    std::cout << "\n[Dynamic-M (N=3072, K=768, ReLU, K=12, R=5)]: ";
    if (dyn.has_crossover) {
        std::cout << "Crossover M* = " << dyn.m_threshold << " (" << mgcm::to_string(dyn.low_m_plan) << " -> "
                  << mgcm::to_string(dyn.high_m_plan) << ")\n";
    } else {
        std::cout << "Uniform plan across [1..4096]: " << mgcm::to_string(dyn.low_m_plan) << "\n";
    }

    // Measure pure CostModel decision latency on CPU
    constexpr int kIters = 100000;
    mgcm::MlpBlock bench_blk(cases[5].cfg);
    auto cands = bench_blk.build_candidates(prof.g_wave_quantum);
    volatile double sink_acc = 0.0;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) sink_acc += eng.estimate_partition(cands[3]).t_execute_us;
    double ns_1p = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / kIters;

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) sink_acc += static_cast<double>(eng.select_optimal(bench_blk, 10));
    double ns_4p = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / kIters;

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters / 10; ++i) sink_acc += static_cast<double>(eng.compute_horizon(bench_blk).size());
    double ns_hor = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (kIters / 10);

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 2000; ++i) sink_acc += static_cast<double>(eng.analyze_dynamic_m(dyn_cfg, 1, 4096, 5).m_threshold);
    double us_dyn = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 2000.0;
    (void)sink_acc;

    std::cout << std::fixed << std::setprecision(1)
              << "[CostModel CPU Time] 1 Plan P = " << ns_1p << " ns | 4 Plans (select_optimal) = " << ns_4p
              << " ns | Lower Envelope = " << ns_hor << " ns | Dynamic M* = " << us_dyn << " us\n\n";

    if (!phase_csv.empty()) {
        std::ofstream pfs(phase_csv);
        pfs << "M,R,plan_id,plan_name,t_exec_p0_us,t_exec_p2_us,t_exec_p3_us\n";
        const std::vector<uint32_t> ms = {16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1500, 2048, 3072, 4096};
        const std::vector<uint64_t> rs = {1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000, 20000};
        for (uint32_t m : ms) {
            auto pcfg = make_cfg(m, 3072, 768, true, relu, smem64, true, 1);
            mgcm::MlpBlock pb(pcfg);
            auto pe = eng.evaluate_block(pb);
            for (uint64_t r : rs) {
                auto best = eng.select_optimal(pb, r);
                pfs << m << "," << r << "," << static_cast<int>(best) << "," << mgcm::to_string(best) << ","
                    << pe[0].t_execute_us << "," << pe[2].t_execute_us << "," << pe[3].t_execute_us << "\n";
            }
        }
        std::cout << "Saved 2D (M, R) Phase Map CSV -> " << phase_csv << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool list_only = false, no_fe_probe = false;
        uint32_t dev_idx = 0;
        std::string eval_json, out_json, out_bw, out_wave, cache_dir, phase_csv;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--list") list_only = true;
            else if (arg == "--no-fe-probe") no_fe_probe = true;
            else if (arg == "--eval" && i + 1 < argc) eval_json = argv[++i];
            else if (arg == "--device" && i + 1 < argc) dev_idx = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (arg == "--out" && i + 1 < argc) out_json = argv[++i];
            else if (arg == "--csv" && i + 1 < argc) out_bw = argv[++i];
            else if (arg == "--wave-csv" && i + 1 < argc) out_wave = argv[++i];
            else if (arg == "--cache-dir" && i + 1 < argc) cache_dir = argv[++i];
            else if (arg == "--phase-csv" && i + 1 < argc) phase_csv = argv[++i];
        }

        if (!eval_json.empty()) {
            run_evaluation_report(eval_json, phase_csv);
            return 0;
        }
        auto devs = mgcm::VulkanBackend::enumerate_devices(false);
        if (list_only) {
            std::cout << "Available Vulkan Hardware GPUs:\n";
            for (const auto& d : devs) std::cout << "  [" << d.index << "] " << d.name << " (" << d.driver << ")\n";
            return 0;
        }

        mgcm::VulkanBackend vk(dev_idx, false);
        mgcm::DeviceProfile prof{};
        bool hit = false;
        if (!cache_dir.empty()) {
            mgcm::ProfileCacheStore store(cache_dir);
            if (auto cached = store.try_load(vk.query_static_props())) {
                prof = *cached; hit = true;
                std::cout << "[Cache HIT] " << store.make_cache_path(prof.props) << "\n";
            }
        }
        if (!hit) {
            auto fe = no_fe_probe ? std::make_unique<mgcm::NullFrontendProbe>(mgcm::CompilerKind::Slang)
                                  : mgcm::create_slang_frontend_probe();
            mgcm::Calibrator calib(vk, fe.get());
            prof = calib.run_full();
            if (!cache_dir.empty()) mgcm::ProfileCacheStore(cache_dir).save(prof);
        }

        if (!out_json.empty()) std::ofstream(out_json) << mgcm::Calibrator::to_json_string(prof) << "\n";
        if (!out_bw.empty()) mgcm::Calibrator::save_bw_csv(prof, out_bw);
        if (!out_wave.empty()) mgcm::Calibrator::save_wave_csv(prof, out_wave);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[mgcm ERROR] " << ex.what() << "\n";
        return 1;
    }
}
