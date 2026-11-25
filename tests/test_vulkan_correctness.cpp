#include "mgcm/backends/vulkan_backend.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::vector<double> cpu_ref_mlp_gelu_erf_fp64(uint32_t M, uint32_t N, uint32_t K, const std::vector<float>& A,
                                              const std::vector<float>& B, const std::vector<float>& bias) {
    std::vector<double> out(static_cast<size_t>(M) * N, 0.0);
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; ++k) {
                acc += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
            }
            double x = acc + static_cast<double>(bias[n]);
            out[m * N + n] = 0.5 * x * (1.0 + std::erf(x * inv_sqrt2));
        }
    }
    return out;
}

void verify_shape_all_plans(mgcm::VulkanBackend& vk, const char* label, uint32_t M, uint32_t N, uint32_t K,
                            const std::vector<float>& hA, const std::vector<float>& hB, const std::vector<float>& hBias,
                            bool print_matrix = false) {
    auto ref = cpu_ref_mlp_gelu_erf_fp64(M, N, K, hA, hB, hBias);

    auto bA = vk.allocate_buffer(hA.size() * sizeof(float));
    auto bB = vk.allocate_buffer(hB.size() * sizeof(float));
    auto bBias = vk.allocate_buffer(hBias.size() * sizeof(float));
    auto bC1 = vk.allocate_buffer(static_cast<size_t>(M) * N * sizeof(float));
    auto bC2 = vk.allocate_buffer(static_cast<size_t>(M) * N * sizeof(float));
    auto bY = vk.allocate_buffer(static_cast<size_t>(M) * N * sizeof(float));

    vk.upload_buffer(bA, hA.data(), hA.size() * sizeof(float));
    vk.upload_buffer(bB, hB.data(), hB.size() * sizeof(float));
    vk.upload_buffer(bBias, hBias.data(), hBias.size() * sizeof(float));

    auto k_mm_only = vk.compile_kernel(mgcm::BenchKernelId::Ora_MatmulOnly, 0);
    auto k_mm_bias = vk.compile_kernel(mgcm::BenchKernelId::Ora_MatmulBias, 0);
    auto k_mm_full = vk.compile_kernel(mgcm::BenchKernelId::Ora_MatmulBiasGelu, 0);
    auto k_el_add = vk.compile_kernel(mgcm::BenchKernelId::Ora_ElemAdd, 0);
    auto k_el_gelu = vk.compile_kernel(mgcm::BenchKernelId::Ora_ElemGelu, 0);
    auto k_el_both = vk.compile_kernel(mgcm::BenchKernelId::Ora_ElemAddGelu, 0);

    std::array<mgcm::Scalar32, 4> sc{mgcm::Scalar32::U32(M), mgcm::Scalar32::U32(N), mgcm::Scalar32::U32(K),
                                     mgcm::Scalar32::F32(0.0f)};
    std::array<uint32_t, 3> g2d{(N + 63u) / 64u, (M + 63u) / 64u, 1u}, l2d{16u, 16u, 1u};
    std::array<uint32_t, 3> g1d{(M * N + 255u) / 256u, 1u, 1u}, l1d{256u, 1u, 1u};

    std::array<mgcm::BufferHandle, 4> m_c1{bA, bB, bBias, bC1}, m_y{bA, bB, bBias, bY};
    std::array<mgcm::BufferHandle, 4> a_c2{bC1, bBias, bBias, bC2};
    std::array<mgcm::BufferHandle, 4> g_c2{bC2, bBias, bBias, bY}, g_c1{bC1, bBias, bBias, bY};

    std::vector<mgcm::DispatchDesc> chains[4] = {
        {{k_mm_only, m_c1, sc, g2d, l2d, 1, false},
         {k_el_add, a_c2, sc, g1d, l1d, 1, false},
         {k_el_gelu, g_c2, sc, g1d, l1d, 1, false}},
        {{k_mm_bias, m_c1, sc, g2d, l2d, 1, false}, {k_el_gelu, g_c1, sc, g1d, l1d, 1, false}},
        {{k_mm_only, m_c1, sc, g2d, l2d, 1, false}, {k_el_both, g_c1, sc, g1d, l1d, 1, false}},
        {{k_mm_full, m_y, sc, g2d, l2d, 1, false}},
    };

    std::vector<float> out_p0;
    double max_err_vs_fp64[4] = {};
    double max_diff_p0_p3 = 0.0;

    for (int p = 0; p < 4; ++p) {
        vk.execute_chain(chains[p]);
        std::vector<float> got(static_cast<size_t>(M) * N, 0.0f);
        vk.download_buffer(bY, got.data(), got.size() * sizeof(float));
        if (p == 0) out_p0 = got;

        double max_e = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            max_e = std::max(max_e, std::abs(static_cast<double>(got[i]) - ref[i]));
            if (p == 3) {
                max_diff_p0_p3 = std::max(max_diff_p0_p3, std::abs(static_cast<double>(got[i] - out_p0[i])));
            }
        }
        max_err_vs_fp64[p] = max_e;
        assert(max_e < 1e-5 && "Numerical tolerance violated vs FP64 reference!");
    }

    std::cout << std::scientific << std::setprecision(3) << "  [" << std::left << std::setw(26) << label
              << "] max|GPU - FP64|: P0=" << max_err_vs_fp64[0] << ", P1=" << max_err_vs_fp64[1]
              << ", P2=" << max_err_vs_fp64[2] << ", P3=" << max_err_vs_fp64[3] << " | max|P3 - P0|=" << max_diff_p0_p3
              << " (PASS <= 1e-5)\n";

    if (print_matrix) {
        std::cout << std::fixed << std::setprecision(6) << "    2x2 FP64 ref : [[" << ref[0] << ", " << ref[1] << "], ["
                  << ref[2] << ", " << ref[3] << "]]\n"
                  << "    2x2 GPU  P3  : [[" << out_p0[0] << ", " << out_p0[1] << "], [" << out_p0[2] << ", "
                  << out_p0[3] << "]]\n";
    }

    for (auto kh : {k_el_both, k_el_gelu, k_el_add, k_mm_full, k_mm_bias, k_mm_only})
        vk.destroy_kernel(kh);
    for (auto bh : {bY, bC2, bC1, bBias, bB, bA})
        vk.free_buffer(bh);
}

void verify_negative_abi_cases(mgcm::VulkanBackend& vk) {
    auto b = vk.allocate_buffer(256);
    auto kh = vk.compile_kernel(mgcm::BenchKernelId::MB1_DispatchNoop, 0);

    // Error Case 1: Wrong buffer binding count (passing 2 buffers to 1-binding shader)
    bool caught_buf_mismatch = false;
    try {
        std::array<mgcm::BufferHandle, 2> bad_bufs{b, b};
        std::array<mgcm::Scalar32, 2> sc{mgcm::Scalar32::U32(1), mgcm::Scalar32::F32(0.0f)};
        vk.execute({kh, bad_bufs, sc, {1, 1, 1}, {64, 1, 1}, 1, false});
    } catch (const std::runtime_error&) {
        caught_buf_mismatch = true;
    }
    assert(caught_buf_mismatch);

    // Error Case 2: PushConstants > 128B Canonical ABI limit (33 scalars = 132B)
    bool caught_pc_overflow = false;
    try {
        std::array<mgcm::BufferHandle, 1> ok_buf{b};
        std::vector<mgcm::Scalar32> huge_sc(33, mgcm::Scalar32::U32(0));
        vk.execute({kh, ok_buf, huge_sc, {1, 1, 1}, {64, 1, 1}, 1, false});
    } catch (const std::runtime_error&) {
        caught_pc_overflow = true;
    }
    assert(caught_pc_overflow);

    // Error Case 3: Invalid device index rejected cleanly
    bool caught_bad_dev = false;
    try {
        mgcm::VulkanBackend bad_vk(99, false);
    } catch (const std::runtime_error&) {
        caught_bad_dev = true;
    }
    assert(caught_bad_dev);

    vk.destroy_kernel(kh);
    vk.free_buffer(b);
    std::cout << "  [Negative ABI & Error Cases] Buffer mismatch, >128B PushConstants, invalid device_idx: ALL CAUGHT "
                 "(PASS)\n";
}

} // namespace

int main() {
    auto devs = mgcm::VulkanBackend::enumerate_devices(false);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.8f, 0.8f);

    for (const auto& d : devs) {
        std::cout << "=== Numerical & ABI Verification on [" << d.index << "] " << d.name << " ===\n";
        mgcm::VulkanBackend vk(d.index, false);

        // 1. Small worked example 2x2x2: A=[[1,2],[3,4]], B=[[0.5,-1],[1.5,2]], bias=[0.1,-0.2]
        verify_shape_all_plans(vk, "Toy 2x2x2 (Manual)", 2, 2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, {0.5f, -1.0f, 1.5f, 2.0f},
                               {0.1f, -0.2f}, true);

        // 2. Boundary non-multiple tail shape 19x37x23 (unaligned on M, N, K)
        auto make_vec = [&](size_t n) {
            std::vector<float> v(n);
            for (auto& x : v)
                x = dist(rng);
            return v;
        };
        verify_shape_all_plans(vk, "Boundary Tail 19x37x23", 19, 37, 23, make_vec(19 * 23), make_vec(23 * 37),
                               make_vec(37));

        // 3. Standard aligned workload 128x512x256
        verify_shape_all_plans(vk, "Standard 128x512x256", 128, 512, 256, make_vec(128 * 256), make_vec(256 * 512),
                               make_vec(512));

        // 4. Negative / error handling checks
        verify_negative_abi_cases(vk);
    }
    return 0;
}