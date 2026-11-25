#include "mgcm/blocks/mlp_block.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mgcm {

MlpBlock::MlpBlock(MlpConfig cfg) : cfg_(std::move(cfg)) {}

uint32_t MlpBlock::estimate_spirv_words(CompilerKind compiler, bool has_matmul, uint32_t num_buffers,
                                        const MatmulSchedule& sched, const EpilogueProfile& ep, uint32_t out_unroll) {
    // Syntax-directed SPIR-V word counts per compiler template (independent of GPU hardware)
    const bool is_gc       = (compiler == CompilerKind::GraphCompiler);
    const uint32_t l_0     = is_gc ? 260u : 240u;
    const uint32_t l_buf   = is_gc ? 22u : 18u;
    const uint32_t l_skel  = is_gc ? 480u : 440u;
    const uint32_t l_smem  = is_gc ? 40u : 36u;
    const uint32_t l_rld   = is_gc ? 16u : 14u;
    const uint32_t l_fma   = is_gc ? 27u : 25u;
    const uint32_t l_store = has_matmul ? 40u : 20u;

    uint32_t total = l_0 + l_buf * num_buffers;

    if (has_matmul) {
        uint32_t wg_thr = std::max<uint32_t>(1u, sched.wg_threads());
        // Separate integer ceilings for cooperative A and B tile loads into Workgroup SMEM
        uint32_t s_ld_a = (sched.tile_m * sched.tile_k + wg_thr - 1u) / wg_thr;
        uint32_t s_ld_b = (sched.tile_k * sched.tile_n + wg_thr - 1u) / wg_thr;
        uint32_t s_ld   = s_ld_a + s_ld_b;

        uint32_t u_k      = std::max<uint32_t>(1u, sched.unroll_k);
        uint32_t mm_body  = l_skel;
        if (sched.use_shared_mem) {
            mm_body += s_ld * l_smem;
        }
        mm_body += u_k * ((sched.reg_m + sched.reg_n) * l_rld + (sched.reg_m * sched.reg_n) * l_fma);
        total += mm_body;
    }

    uint32_t u_out = std::max<uint32_t>(1u, out_unroll);
    total += u_out * (l_store + ep.spirv_words_per_scalar);
    return total;
}

std::vector<PartitionWorkload> MlpBlock::build_candidates(uint32_t device_wave_quantum) const {
    const uint32_t b_indep = cfg_.broadcast_b ? 1u : std::max<uint32_t>(1u, cfg_.batch);
    const uint32_t m_eff   = cfg_.broadcast_b ? (std::max<uint32_t>(1u, cfg_.batch) * cfg_.M) : cfg_.M;
    const uint32_t n_eff   = cfg_.N;
    const uint32_t k_eff   = cfg_.K;

    const double bytes_a    = 4.0 * static_cast<double>(b_indep) * static_cast<double>(m_eff) * static_cast<double>(k_eff);
    const double bytes_b    = 4.0 * static_cast<double>(b_indep) * static_cast<double>(k_eff) * static_cast<double>(n_eff);
    const double bytes_bias = 4.0 * static_cast<double>(n_eff);
    const double bytes_c    = 4.0 * static_cast<double>(b_indep) * static_cast<double>(m_eff) * static_cast<double>(n_eff);

    const auto& sched      = cfg_.schedule;
    const bool  is_naive   = (!sched.use_shared_mem && sched.reg_m == 1u && sched.reg_n == 1u);
    const uint32_t gm      = (m_eff + sched.tile_m - 1u) / sched.tile_m;
    const uint32_t gn      = (n_eff + sched.tile_n - 1u) / sched.tile_n;
    const uint32_t gk      = (k_eff + sched.tile_k - 1u) / sched.tile_k;
    const double   m_pad   = static_cast<double>(gm * sched.tile_m);
    const double   n_pad   = static_cast<double>(gn * sched.tile_n);
    const double   k_pad   = static_cast<double>(gk * sched.tile_k);
    const bool     tails   = cfg_.is_dynamic_m || (m_eff % sched.tile_m != 0u) || (n_eff % sched.tile_n != 0u) ||
                             (k_eff % sched.tile_k != 0u);

    auto make_matmul_kernel = [&](const EpilogueProfile& fused_ep, bool is_base_mm_only) {
        KernelWorkload kw{};
        kw.frontend       = cfg_.frontend;
        kw.fe_cache_miss  = !(cfg_.fe_cached || (is_base_mm_only && cfg_.base_mm_drv_cached));
        kw.drv_cache_miss = !(is_base_mm_only && cfg_.base_mm_drv_cached);

        uint32_t num_bufs  = fused_ep.has_bias_buffer ? 4u : 3u;
        uint32_t out_unr   = sched.reg_m * sched.reg_n;
        kw.est_spirv_words = estimate_spirv_words(cfg_.frontend, true, num_bufs, sched, fused_ep, out_unr);

        kw.workgroups               = b_indep * gm * gn * std::max<uint32_t>(1u, sched.split_k);
        kw.flops_contraction_padded = 2.0 * static_cast<double>(b_indep) * m_pad * n_pad * k_pad;
        double e_pad                = static_cast<double>(b_indep) * m_pad * n_pad;
        kw.flops_vector_fma_padded  = e_pad * static_cast<double>(fused_ep.fma_ops_per_elem);
        kw.elems_act_padded         = (fused_ep.act_pipe != ActivationPipe::None) ? e_pad : 0.0;
        kw.act_pipe                 = fused_ep.act_pipe;
        kw.uses_shared_mem          = sched.use_shared_mem;
        kw.has_unaligned_tail       = tails;
        kw.has_fused_heavy_act      = (fused_ep.act_pipe != ActivationPipe::None);

        double ep_in = fused_ep.has_bias_buffer ? bytes_bias : 0.0;
        if (is_naive) {
            // Untiled 1-thread-per-output: every row m re-reads B (strided if transpose_b == true)
            kw.bytes_cold              = bytes_a + ep_in + bytes_c;
            kw.cold_working_set_bytes  = bytes_a + bytes_c;
            kw.bytes_reuse             = static_cast<double>(m_eff) * bytes_b;
            kw.reuse_working_set_bytes = bytes_b;
            kw.read_stride_elems       = cfg_.transpose_b ? k_eff : 1u;
        } else {
            // 2D-tiled conservation of bytes: V_cold = A+B+C, V_reuse = (G_N - 1)*A + (G_M - 1)*B
            kw.bytes_cold             = bytes_a + bytes_b + ep_in + bytes_c;
            kw.cold_working_set_bytes = bytes_a + bytes_b + bytes_c;
            kw.bytes_reuse = static_cast<double>(gn > 1u ? gn - 1u : 0u) * bytes_a +
                             static_cast<double>(gm > 1u ? gm - 1u : 0u) * bytes_b;
            if (sched.split_k > 1u) {
                kw.bytes_reuse += static_cast<double>(sched.split_k) * bytes_c;
            }
            double wave_panel = static_cast<double>(std::max<uint32_t>(1u, device_wave_quantum)) * 4.0 *
                                static_cast<double>(k_eff) * static_cast<double>(sched.tile_m + sched.tile_n);
            kw.reuse_working_set_bytes = std::min(bytes_a + bytes_b, wave_panel);
            kw.read_stride_elems       = 1u;
        }
        return kw;
    };

    auto make_elem_kernel = [&](const EpilogueProfile& ep, bool drv_cached, double producer_ws_bytes) {
        KernelWorkload kw{};
        kw.frontend       = cfg_.frontend;
        kw.fe_cache_miss  = !(cfg_.fe_cached || drv_cached);
        kw.drv_cache_miss = !drv_cached;

        uint32_t num_bufs  = ep.has_bias_buffer ? 3u : 2u;
        kw.est_spirv_words = estimate_spirv_words(cfg_.frontend, false, num_bufs, sched, ep, 1u);

        double total_elems          = static_cast<double>(b_indep) * static_cast<double>(m_eff) * static_cast<double>(n_eff);
        kw.workgroups               = static_cast<uint32_t>(std::max(1.0, std::ceil(total_elems / 256.0)));
        double e_pad                = static_cast<double>(kw.workgroups) * 256.0;
        kw.flops_contraction_padded = 0.0;
        kw.flops_vector_fma_padded  = e_pad * static_cast<double>(ep.fma_ops_per_elem);
        kw.elems_act_padded         = (ep.act_pipe != ActivationPipe::None) ? e_pad : 0.0;
        kw.act_pipe                 = ep.act_pipe;
        kw.bytes_cold               = bytes_c + (ep.has_bias_buffer ? bytes_bias : 0.0) + bytes_c;
        kw.cold_working_set_bytes   = bytes_c + producer_ws_bytes;
        kw.bytes_reuse              = 0.0;
        kw.reuse_working_set_bytes  = 0.0;
        kw.read_stride_elems        = 1u;
        return kw;
    };

    const auto ep_bias     = EpilogueProfile::bias_only();
    const auto ep_act      = cfg_.activation;
    const auto ep_bias_act = ep_bias + ep_act;

    std::vector<PartitionWorkload> out;
    if (cfg_.allowed_plans_mask & plan_bit(PartitionPlan::P0_SplitAll)) {
        out.push_back({PartitionPlan::P0_SplitAll,
                       {make_matmul_kernel(EpilogueProfile::none(), true),
                        make_elem_kernel(ep_bias, cfg_.base_mm_drv_cached, bytes_b),
                        make_elem_kernel(ep_act, cfg_.base_mm_drv_cached, bytes_c)},
                       0,
                       0.0});
    }
    if (cfg_.allowed_plans_mask & plan_bit(PartitionPlan::P1_FusedBias)) {
        out.push_back({PartitionPlan::P1_FusedBias,
                       {make_matmul_kernel(ep_bias, false), make_elem_kernel(ep_act, cfg_.base_mm_drv_cached, bytes_c)},
                       0,
                       0.0});
    }
    if (cfg_.allowed_plans_mask & plan_bit(PartitionPlan::P2_FusedEpilogue)) {
        out.push_back({PartitionPlan::P2_FusedEpilogue,
                       {make_matmul_kernel(EpilogueProfile::none(), true), make_elem_kernel(ep_bias_act, false, bytes_b)},
                       0,
                       0.0});
    }
    if (cfg_.allowed_plans_mask & plan_bit(PartitionPlan::P3_FullFusion)) {
        out.push_back({PartitionPlan::P3_FullFusion, {make_matmul_kernel(ep_bias_act, false)}, 0, 0.0});
    }
    return out;
}

} // namespace mgcm
