#include "mgcm/backends/vulkan_backend.hpp"
#include "mgcm/calibrator.hpp"
#include "mgcm/cost_model.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void run_evaluation_report(const std::string& json_path) {
    std::ifstream ifs(json_path);
    if (!ifs) {
        throw std::runtime_error("Cannot open profile JSON: " + json_path);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    auto prof = mgcm::Calibrator::from_json_string(ss.str());
    mgcm::CostModel cm(prof);

    std::cout << "====================================================================================\n"
              << "CostModel Evaluation on : " << prof.props.device_name << " (" << prof.props.driver_name << ")\n"
              << "L2 = " << (prof.l2_effective_bytes / (1024 * 1024)) << " MB | BW_L2 = " << std::fixed
              << std::setprecision(1) << prof.bw_l2_gbps << " GB/s | BW_DRAM = " << prof.bw_dram_gbps << " GB/s\n"
              << "====================================================================================\n";

    struct CaseDesc {
        const char*         label;
        mgcm::SubgraphQuery q;
    };
    const std::vector<CaseDesc> cases = {
        {"Tiny [32,64,128] (cold)",      {32, 64, 128, true, mgcm::GeluKind::Erf, true, false}},
        {"Synth [128,512,256] (cold)",   {128, 512, 256, false, mgcm::GeluKind::Erf, true, false}},
        {"Synth [128,512,256] (cached)", {128, 512, 256, false, mgcm::GeluKind::Erf, true, true}},
        {"BERT [128,3072,768] (cached)", {128, 3072, 768, true, mgcm::GeluKind::Erf, true, true}},
        {"Whisper [1500,3072,768] Tile", {1500, 3072, 768, true, mgcm::GeluKind::Erf, true, true}},
        {"Whisper [1500,3072,768] NoTl", {1500, 3072, 768, true, mgcm::GeluKind::Erf, false, true}},
        {"Large [4096,4096,1024] Tile",  {4096, 4096, 1024, true, mgcm::GeluKind::Erf, true, true}},
    };

    std::cout << std::left << std::setw(30) << "Workload" << std::right << std::setw(13) << "T_exec(P0)"
              << std::setw(13) << "T_exec(P2)" << std::setw(13) << "T_exec(P3)" << std::setw(10) << "R*(P3/P2)"
              << "   " << std::left << std::setw(16) << "Best(R=1)" << "Best(R=1000)\n"
              << std::string(106, '-') << "\n";

    for (const auto& c : cases) {
        auto all   = cm.evaluate_all(c.q);
        double r_s = cm.compute_break_even_r(c.q, mgcm::PartitionPlan::P3_FullFusion, mgcm::PartitionPlan::P2_FusedEpilogue);
        auto b1    = cm.select_optimal(c.q, 1);
        auto b1000 = cm.select_optimal(c.q, 1000);

        std::cout << std::left << std::setw(30) << c.label << std::right << std::fixed << std::setprecision(2)
                  << std::setw(11) << all[0].t_execute_us << "us"
                  << std::setw(11) << all[2].t_execute_us << "us"
                  << std::setw(11) << all[3].t_execute_us << "us"
                  << std::setw(10) << std::setprecision(1) << r_s << "   "
                  << std::left << std::setw(16) << mgcm::to_string(b1) << mgcm::to_string(b1000) << "\n";
    }
    std::cout << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool list_only        = false;
        uint32_t dev_idx      = 0;
        std::string eval_json = "";
        std::string out_json  = "";
        std::string out_bw    = "";
        std::string out_wave  = "";

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--list") {
                list_only = true;
            } else if (arg == "--eval" && i + 1 < argc) {
                eval_json = argv[++i];
            } else if (arg == "--device" && i + 1 < argc) {
                dev_idx = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--out" && i + 1 < argc) {
                out_json = argv[++i];
            } else if (arg == "--csv" && i + 1 < argc) {
                out_bw = argv[++i];
            } else if (arg == "--wave-csv" && i + 1 < argc) {
                out_wave = argv[++i];
            }
        }

        if (!eval_json.empty()) {
            run_evaluation_report(eval_json);
            return 0;
        }

        auto devs = mgcm::VulkanBackend::enumerate_devices(false);
        if (list_only) {
            std::cout << "Available Vulkan 1.3 Hardware GPUs:\n";
            for (const auto& d : devs) {
                std::cout << "  [" << d.index << "] " << d.name << " (" << d.driver << ")\n";
            }
            return 0;
        }

        mgcm::VulkanBackend vk(dev_idx, false);
        mgcm::Calibrator calib(vk);
        auto prof = calib.run_full();

        std::string js = mgcm::Calibrator::to_json_string(prof);
        if (!out_json.empty()) {
            std::ofstream(out_json) << js << "\n";
            std::cout << "Saved profile JSON -> " << out_json << "\n";
        } else {
            std::cout << "\n" << js << "\n";
        }
        if (!out_bw.empty()) {
            mgcm::Calibrator::save_bw_csv(prof, out_bw);
            std::cout << "Saved BW curve CSV -> " << out_bw << "\n";
        }
        if (!out_wave.empty()) {
            mgcm::Calibrator::save_wave_csv(prof, out_wave);
            std::cout << "Saved Wave curve CSV -> " << out_wave << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[mgcm ERROR] " << ex.what() << "\n";
        return 1;
    }
}
