#pragma once
#include "mgcm/blocks/mlp_block.hpp"
#include "mgcm/cost_model.hpp"
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mgcm::ov_bridge {

// Pure C++ bridge contract for OpenVINO intel_gpu transformations/mlir/convert.cpp.
// Pass 1 (OpenVINO): walk ov::Model, emit one MlpMatchDescriptor per MatMul -> Add -> Act chain.
// Pass 2 (mgcm):     call SubgraphDecisionBatch::solve(...) to count K_same and select optimal P*.
// Pass 3 (OpenVINO): apply chosen PartitionPlan to SubgraphTracker / Partitioner.
struct MlpMatchDescriptor {
    void*           opaque_tag          = nullptr; // Pointer to matched OV node tuple
    MlpConfig       cfg{};
    bool            mm_single_consumer  = true;    // Fan-out == 1 on MatMul output
    bool            add_single_consumer = true;    // Fan-out == 1 on Add output
};

struct SubgraphDecision {
    void*         opaque_tag      = nullptr;
    PartitionPlan chosen_plan     = PartitionPlan::P0_SplitAll;
    uint32_t      k_same_in_model = 1;
};

struct MlpSignatureKey {
    uint32_t       batch;
    uint32_t       M, N, K;
    bool           broadcast_b;
    bool           transpose_b;
    bool           is_dynamic_m;
    ActivationPipe act_pipe;
    uint32_t       ep_fma;
    uint8_t        allowed_mask;

    bool operator==(const MlpSignatureKey& o) const = default;
};

struct MlpSignatureHash {
    size_t operator()(const MlpSignatureKey& k) const noexcept {
        size_t h = static_cast<size_t>(k.M) * 2654435761ull ^ static_cast<size_t>(k.N) * 2246822519ull;
        h ^= static_cast<size_t>(k.K) * 3266489917ull ^ (static_cast<size_t>(k.batch) << 7);
        h ^= (static_cast<size_t>(k.act_pipe) << 3) | (k.transpose_b ? 1u : 0u) | (k.broadcast_b ? 2u : 0u);
        h ^= (static_cast<size_t>(k.allowed_mask) << 12);
        return h;
    }
};

class SubgraphDecisionBatch {
public:
    static uint8_t legal_plans_from_fanout(bool mm_single_consumer, bool add_single_consumer) {
        if (!mm_single_consumer) {
            return plan_bit(PartitionPlan::P0_SplitAll) | plan_bit(PartitionPlan::P2_FusedEpilogue);
        }
        if (!add_single_consumer) {
            return plan_bit(PartitionPlan::P0_SplitAll) | plan_bit(PartitionPlan::P1_FusedBias);
        }
        return kAllPlansMask;
    }

    static std::vector<SubgraphDecision> solve(const std::vector<MlpMatchDescriptor>& matches,
                                               const UniversalCostEngine& engine, uint64_t model_r_runs = 1,
                                               AblationMode ablation = AblationMode::None) {
        std::unordered_map<MlpSignatureKey, uint32_t, MlpSignatureHash> counts;
        std::vector<MlpSignatureKey> keys;
        keys.reserve(matches.size());

        for (const auto& m : matches) {
            uint8_t mask = m.cfg.allowed_plans_mask & legal_plans_from_fanout(m.mm_single_consumer, m.add_single_consumer);
            MlpSignatureKey key{m.cfg.batch,       m.cfg.M, m.cfg.N, m.cfg.K, m.cfg.broadcast_b, m.cfg.transpose_b,
                                m.cfg.is_dynamic_m, m.cfg.activation.act_pipe, m.cfg.activation.fma_ops_per_elem, mask};
            counts[key]++;
            keys.push_back(key);
        }

        std::unordered_map<MlpSignatureKey, PartitionPlan, MlpSignatureHash> plan_cache;
        std::vector<SubgraphDecision> out;
        out.reserve(matches.size());

        for (size_t i = 0; i < matches.size(); ++i) {
            const auto& key = keys[i];
            auto it         = plan_cache.find(key);
            if (it == plan_cache.end()) {
                MlpConfig cfg          = matches[i].cfg;
                cfg.instance_count     = counts[key];
                cfg.allowed_plans_mask = key.allowed_mask;
                MlpBlock block(cfg);
                auto best       = engine.select_optimal(block, model_r_runs, ablation);
                it              = plan_cache.emplace(key, best).first;
            }
            out.push_back({matches[i].opaque_tag, it->second, counts[key]});
        }
        return out;
    }
};

} // namespace mgcm::ov_bridge
