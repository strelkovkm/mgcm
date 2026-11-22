#pragma once
#include "mgcm/device_profile.hpp"
#include <array>
#include <cstdint>

namespace mgcm {

enum class PartitionPlan : uint8_t {
    P0_SplitAll       = 0, // {MatMul} | {Add} | {Gelu}
    P1_FusedBias      = 1, // {MatMul + Add} | {Gelu}
    P2_FusedEpilogue  = 2, // {MatMul} | {Add + Gelu}
    P3_FullFusion     = 3  // {MatMul + Add + Gelu}
};

inline const char* to_string(PartitionPlan p) {
    switch (p) {
        case PartitionPlan::P0_SplitAll:      return "P0_SplitAll";
        case PartitionPlan::P1_FusedBias:     return "P1_FusedBias";
        case PartitionPlan::P2_FusedEpilogue: return "P2_FusedEpilogue";
        case PartitionPlan::P3_FullFusion:    return "P3_FullFusion";
    }
    return "Unknown";
}

enum class GeluKind : uint8_t { Erf = 0, Tanh = 1 };

enum class AblationMode : uint8_t {
    None               = 0, // Full analytical Cost Model C(P, R)
    NoMemoryTraffic    = 1, // Ignore memory hierarchy & intermediate tensor traffic
    NoDispatchOverhead = 2, // Ignore kernel launch & barrier overheads
    NoCompileCost      = 3  // Ignore shader compilation latency (R -> inf greedy)
};

struct SubgraphQuery {
    uint32_t M                     = 128;
    uint32_t N                     = 512;
    uint32_t K                     = 256;
    bool     transpose_b           = false;
    GeluKind gelu                  = GeluKind::Erf;
    bool     tiled_matmul          = true;  // 64x64 2D-tiled vs naive 1-thread-per-output
    bool     base_primitives_cached = false; // e.g. split MatMul already in pipeline cache
};

struct PlanEstimate {
    PartitionPlan plan          = PartitionPlan::P0_SplitAll;
    double        t_compile_us  = 0.0;
    double        t_execute_us  = 0.0;
    double        t_compute_us  = 0.0;
    double        t_memory_us   = 0.0;
    double        t_dispatch_us = 0.0;

    double total_cost_us(uint64_t r_runs) const {
        return t_compile_us + static_cast<double>(r_runs) * t_execute_us;
    }
};

class CostModel {
public:
    explicit CostModel(DeviceProfile profile);

    const DeviceProfile& profile() const { return prof_; }

    PlanEstimate              estimate_plan(const SubgraphQuery& q, PartitionPlan plan,
                                            AblationMode ablation = AblationMode::None) const;
    std::array<PlanEstimate, 4> evaluate_all(const SubgraphQuery& q,
                                             AblationMode ablation = AblationMode::None) const;
    PartitionPlan             select_optimal(const SubgraphQuery& q, uint64_t r_runs = 1,
                                             AblationMode ablation = AblationMode::None) const;

    // Break-even reuse count R* where candidate_plan becomes cheaper than reference_plan (0 if always cheaper, -1 if never)
    double compute_break_even_r(const SubgraphQuery& q, PartitionPlan candidate_plan,
                                PartitionPlan reference_plan) const;

private:
    double l2_hit_fraction(double working_set_bytes) const;
    double effective_bw_gbps(double working_set_bytes, bool strided32) const;

    DeviceProfile prof_;
};

} // namespace mgcm
