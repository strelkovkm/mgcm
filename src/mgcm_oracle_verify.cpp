#include "mgcm/backends/vulkan_backend.hpp"
#include "mgcm/blocks/mlp_block.hpp"
#include "mgcm/calibrator.hpp"
#include "mgcm/cost_model.hpp"
#include "mgcm/frontend_probe.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    try {
        uint32_t dev_idx = 0;
        std::string mode = "cached";  // "vacuum" (cold FE + Drv) or "cached" (warm base MatMul/Add/Act)
        std::string act_str = "relu"; // "relu" or "gelu_erf"
        std::string out_csv = "results/verify_gpu0.csv";
        uint32_t N = 3072;
        uint32_t K = 768;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--device" && i + 1 < argc)
                dev_idx = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (a == "--mode" && i + 1 < argc)
                mode = argv[++i];
            else if (a == "--act" && i + 1 < argc)
                act_str = argv[++i];
            else if (a == "--out" && i + 1 < argc)
                out_csv = argv[++i];
            else if (a == "--N" && i + 1 < argc)
                N = static_cast<uint32_t>(std::stoul(argv[++i]));
            else if (a == "--K" && i + 1 < argc)
                K = static_cast<uint32_t>(std::stoul(argv[++i]));
        }

        const bool is_cached = (mode == "cached");
        const bool is_erf = (act_str == "gelu_erf");

        mgcm::VulkanBackend vk(dev_idx, false);
        auto fe_probe = mgcm::create_slang_frontend_probe();
        mgcm::Calibrator calib(vk, fe_probe.get());

        // Step 1: Run blind hardware calibration FIRST (before touching any verification shapes!)
        auto prof = calib.run_full();
        mgcm::UniversalCostEngine eng(prof);

        // Step 2: Measure REAL compilation times of the 6 Oracle building-block kernels
        const mgcm::BenchKernelId k_mm_only = mgcm::BenchKernelId::Ora_MatmulOnly;
        const mgcm::BenchKernelId k_mm_bias = mgcm::BenchKernelId::Ora_MatmulBias;
        const mgcm::BenchKernelId k_mm_full =
            is_erf ? mgcm::BenchKernelId::Ora_MatmulBiasGelu : mgcm::BenchKernelId::Ora_MatmulBiasRelu;
        const mgcm::BenchKernelId k_el_add = mgcm::BenchKernelId::Ora_ElemAdd;
        const mgcm::BenchKernelId k_el_act =
            is_erf ? mgcm::BenchKernelId::Ora_ElemGelu : mgcm::BenchKernelId::Ora_ElemRelu;
        const mgcm::BenchKernelId k_el_both =
            is_erf ? mgcm::BenchKernelId::Ora_ElemAddGelu : mgcm::BenchKernelId::Ora_ElemAddRelu;

        const std::vector<mgcm::BenchKernelId> all_kids = {k_mm_only, k_mm_bias, k_mm_full,
                                                           k_el_add,  k_el_act,  k_el_both};

        mgcm::BufferHandle dummy_buf = vk.allocate_buffer(64 * 64 * sizeof(float));
        std::random_device rd;
        std::mt19937 rng(rd());

        std::map<mgcm::BenchKernelId, double> real_cold_drv_us, real_cold_fe_us, real_hit_drv_us;
        for (auto kid : all_kids) {
            std::vector<double> drv_s, fe_s;
            for (int s = 0; s < 5; ++s) {
                uint32_t salt = ((rng() & 0x007FFFFEu) | 1u);
                double c_ns = 0.0;
                auto kh = vk.compile_kernel(kid, salt, &c_ns);
                std::array<mgcm::BufferHandle, 4> b4{dummy_buf, dummy_buf, dummy_buf, dummy_buf};
                std::array<mgcm::Scalar32, 4> sc{mgcm::Scalar32::U32(64), mgcm::Scalar32::U32(64),
                                                 mgcm::Scalar32::U32(16), mgcm::Scalar32::F32(0.0f)};
                mgcm::DispatchDesc d{kh, b4, sc, {1, 1, 1}, {16, 16, 1}, 1, false};
                auto tr1 = vk.execute(d);
                auto tr2 = vk.execute(d);
                double lazy_ns = std::max(0.0, tr1.cpu_wall_ns - tr2.cpu_wall_ns);
                drv_s.push_back((c_ns + lazy_ns) * 1e-3);
                vk.destroy_kernel(kh);

                if (!is_cached && fe_probe && fe_probe->is_available() && s < 3) {
                    double f_ms = 0.0;
                    size_t w = 0;
                    if (fe_probe->measure_compile_ms(kid, salt, &f_ms, &w)) {
                        fe_s.push_back(f_ms * 1e3);
                    }
                }
            }
            real_cold_drv_us[kid] = median_of(drv_s);
            real_cold_fe_us[kid] = fe_s.empty() ? 0.0 : median_of(fe_s);
            real_hit_drv_us[kid] = vk.measure_pipeline_cache_hit_ms(kid) * 1e3;
        }
        vk.free_buffer(dummy_buf);

        auto real_krn_compile_us = [&](mgcm::BenchKernelId kid, bool base_cached) {
            if (is_cached && base_cached) {
                return real_hit_drv_us[kid];
            }
            return (is_cached ? 0.0 : real_cold_fe_us[kid]) + real_cold_drv_us[kid];
        };

        // Compile persistent execution handles for steady-state timing across M
        std::map<mgcm::BenchKernelId, mgcm::KernelHandle> khs;
        for (auto kid : all_kids) {
            khs[kid] = vk.compile_kernel(kid, 0);
        }

        const std::vector<uint32_t> ms = {16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1500, 2048, 3072, 4096};
        const uint32_t max_m = ms.back();

        mgcm::BufferHandle buf_a = vk.allocate_buffer(static_cast<size_t>(max_m) * K * sizeof(float));
        mgcm::BufferHandle buf_b = vk.allocate_buffer(static_cast<size_t>(K) * N * sizeof(float));
        mgcm::BufferHandle buf_bias = vk.allocate_buffer(static_cast<size_t>(N) * sizeof(float));
        mgcm::BufferHandle buf_c1 = vk.allocate_buffer(static_cast<size_t>(max_m) * N * sizeof(float));
        mgcm::BufferHandle buf_c2 = vk.allocate_buffer(static_cast<size_t>(max_m) * N * sizeof(float));
        mgcm::BufferHandle buf_y = vk.allocate_buffer(static_cast<size_t>(max_m) * N * sizeof(float));

        // Initialize A, B, bias with PRNG values on GPU
        {
            auto ik = vk.compile_kernel(mgcm::BenchKernelId::InitPrngVec4, 0);
            auto init_one = [&](mgcm::BufferHandle bh, size_t elems_f32, uint32_t seed) {
                std::array<mgcm::BufferHandle, 1> ib{bh};
                std::array<mgcm::Scalar32, 3> is{mgcm::Scalar32::U32(static_cast<uint32_t>(elems_f32 / 4)),
                                                 mgcm::Scalar32::U32(32768), mgcm::Scalar32::U32(seed)};
                vk.execute({ik, ib, is, {128, 1, 1}, {256, 1, 1}, 1, false});
            };
            init_one(buf_a, static_cast<size_t>(max_m) * K, 0x11112222u);
            init_one(buf_b, static_cast<size_t>(K) * N, 0x33334444u);
            init_one(buf_bias, N, 0x55556666u);
            vk.destroy_kernel(ik);
        }

        std::ofstream ofs(out_csv);
        ofs << "M,N,K,plan_id,plan_name,pred_compile_us,real_compile_us,pred_exec_us,real_exec_us,"
               "pred_nomem_exec_us,pred_nodisp_exec_us\n";

        const double real_comp_p[4] = {
            real_krn_compile_us(k_mm_only, true) + real_krn_compile_us(k_el_add, true) +
                real_krn_compile_us(k_el_act, true),
            real_krn_compile_us(k_mm_bias, false) + real_krn_compile_us(k_el_act, true),
            real_krn_compile_us(k_mm_only, true) + real_krn_compile_us(k_el_both, false),
            real_krn_compile_us(k_mm_full, false),
        };

        {
            auto wk = vk.compile_kernel(mgcm::BenchKernelId::MB0_Warmup, 0);
            std::array<mgcm::BufferHandle, 1> wb{buf_y};
            std::array<mgcm::Scalar32, 4> ws{mgcm::Scalar32::U32(8192), mgcm::Scalar32::F32(1.00001f), mgcm::Scalar32::F32(0.00001f), mgcm::Scalar32::F32(-999999.0f)};
            double w_ms = 0.0; while (w_ms < 150.0) { w_ms += vk.execute({wk, wb, ws, {128, 1, 1}, {256, 1, 1}, 1, false}).cpu_wall_ns * 1e-6; }
            vk.destroy_kernel(wk);
        }
        for (uint32_t M : ms) {
            mgcm::MlpConfig cfg{};
            cfg.M = M;
            cfg.N = N;
            cfg.K = K;
            cfg.transpose_b = false;
            cfg.activation = is_erf ? mgcm::EpilogueProfile::gelu_erf() : mgcm::EpilogueProfile::relu();
            cfg.schedule = mgcm::MatmulSchedule::tiled_64x64_smem();
            cfg.frontend = mgcm::CompilerKind::Slang;
            cfg.fe_cached = is_cached;
            cfg.base_mm_drv_cached = is_cached;

            mgcm::MlpBlock blk(cfg);
            auto pred_full = eng.evaluate_block(blk, mgcm::AblationMode::None);
            auto pred_nomem = eng.evaluate_block(blk, mgcm::AblationMode::NoMemoryTraffic);
            auto pred_nodisp = eng.evaluate_block(blk, mgcm::AblationMode::NoDispatchOverhead);

            std::array<mgcm::Scalar32, 4> sc{mgcm::Scalar32::U32(M), mgcm::Scalar32::U32(N), mgcm::Scalar32::U32(K),
                                             mgcm::Scalar32::F32(0.0f)};
            std::array<uint32_t, 3> g2d{(N + 63u) / 64u, (M + 63u) / 64u, 1u};
            std::array<uint32_t, 3> l2d{16u, 16u, 1u};
            std::array<uint32_t, 3> g1d{(M * N + 255u) / 256u, 1u, 1u};
            std::array<uint32_t, 3> l1d{256u, 1u, 1u};

            std::array<mgcm::BufferHandle, 4> b_mm_c1{buf_a, buf_b, buf_bias, buf_c1};
            std::array<mgcm::BufferHandle, 4> b_mm_y{buf_a, buf_b, buf_bias, buf_y};
            std::array<mgcm::BufferHandle, 4> b_add_c2{buf_c1, buf_bias, buf_bias, buf_c2};
            std::array<mgcm::BufferHandle, 4> b_act_from_c2{buf_c2, buf_bias, buf_bias, buf_y};
            std::array<mgcm::BufferHandle, 4> b_act_from_c1{buf_c1, buf_bias, buf_bias, buf_y};

            std::vector<mgcm::DispatchDesc> chains[4] = {
                // P0: MatMul -> C1 | Add -> C2 | Act -> Y
                {{khs[k_mm_only], b_mm_c1, sc, g2d, l2d, 1, false},
                 {khs[k_el_add], b_add_c2, sc, g1d, l1d, 1, false},
                 {khs[k_el_act], b_act_from_c2, sc, g1d, l1d, 1, false}},
                // P1: MatMul+Bias -> C1 | Act -> Y
                {{khs[k_mm_bias], b_mm_c1, sc, g2d, l2d, 1, false},
                 {khs[k_el_act], b_act_from_c1, sc, g1d, l1d, 1, false}},
                // P2: MatMul -> C1 | Add+Act -> Y
                {{khs[k_mm_only], b_mm_c1, sc, g2d, l2d, 1, false},
                 {khs[k_el_both], b_act_from_c1, sc, g1d, l1d, 1, false}},
                // P3: MatMul+Bias+Act -> Y
                {{khs[k_mm_full], b_mm_y, sc, g2d, l2d, 1, false}},
            };

            double real_exec_p[4] = {};
            for (int p = 0; p < 4; ++p) {
                for (int w = 0; w < 4; ++w)
                    vk.execute_chain(chains[p]);
                std::vector<double> us;
                for (int it = 0; it < 25; ++it) {
                    us.push_back(vk.execute_chain(chains[p]).gpu_execution_ns * 1e-3);
                }
                real_exec_p[p] = median_of(us);

                ofs << M << "," << N << "," << K << "," << p << "," << mgcm::to_string(pred_full[p].plan) << ","
                    << pred_full[p].t_compile_us << "," << real_comp_p[p] << "," << pred_full[p].t_execute_us << ","
                    << real_exec_p[p] << "," << pred_nomem[p].t_execute_us << "," << pred_nodisp[p].t_execute_us
                    << "\n";
            }

            std::cout << "[M=" << std::setw(4) << M << "] " << std::fixed << std::setprecision(1)
                      << "P0(pred=" << pred_full[0].t_execute_us << " / real=" << real_exec_p[0] << " us) | "
                      << "P2(pred=" << pred_full[2].t_execute_us << " / real=" << real_exec_p[2] << " us) | "
                      << "P3(pred=" << pred_full[3].t_execute_us << " / real=" << real_exec_p[3] << " us)\n";
        }

        for (auto [_, kh] : khs)
            vk.destroy_kernel(kh);
        for (auto bh : {buf_y, buf_c2, buf_c1, buf_bias, buf_b, buf_a})
            vk.free_buffer(bh);
        std::cout << "Saved verification CSV -> " << out_csv << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[verify ERROR] " << ex.what() << "\n";
        return 1;
    }
}