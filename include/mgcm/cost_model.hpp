#pragma once
#include "mgcm/blocks/mlp_block.hpp"
#include "mgcm/device_profile.hpp"
#include "mgcm/topology_policy.hpp"
#include "mgcm/workload.hpp"
#include <cstdint>
#include <vector>

namespace mgcm {

enum class AblationMode : uint8_t {
    None               = 0,
    NoMemoryTraffic    = 1,
    NoDispatchOverhead = 2,
    NoCompileCost      = 3
};

struct PlanEstimate {
    PartitionPlan plan          = PartitionPlan::P0_SplitAll;
    double        t_compile_us  = 0.0;
    double        t_execute_us  = 0.0;
    double        t_compute_us  = 0.0;
    double        t_memory_us   = 0.0;
    double        t_dispatch_us = 0.0;

    double total_cost_us(uint64_t r_effective) const {
        return t_compile_us + static_cast<double>(r_effective) * t_execute_us;
    }
};

struct HorizonSegment {
    uint64_t      r_start_inclusive = 1;
    uint64_t      r_end_exclusive   = 0; // 0 denotes +infinity
    PartitionPlan plan              = PartitionPlan::P0_SplitAll;
};

struct DynamicMDecision {
    bool          has_crossover = false;
    uint32_t      m_threshold   = 0;
    PartitionPlan low_m_plan    = PartitionPlan::P0_SplitAll;
    PartitionPlan high_m_plan   = PartitionPlan::P3_FullFusion;
};

class UniversalCostEngine {
public:
    explicit UniversalCostEngine(DeviceProfile profile);
    UniversalCostEngine(DeviceProfile profile, TopologyPolicy topology);

    const DeviceProfile&  profile() const { return prof_; }
    const TopologyPolicy& topology() const { return topo_; }

    double       empirical_bw_gbps(double working_set_bytes, uint32_t stride_elems = 1) const;
    PlanEstimate estimate_partition(const PartitionWorkload& pw, AblationMode ablation = AblationMode::None) const;

    std::vector<PlanEstimate> evaluate_block(const IBlockModel& block,
                                             AblationMode ablation = AblationMode::None) const;
    PartitionPlan             select_optimal(const IBlockModel& block, uint64_t model_r_runs = 1,
                                             AblationMode ablation = AblationMode::None) const;

    // Exact Lower Envelope over R_model in [1, +inf) accounting for block.instance_count()
    std::vector<HorizonSegment> compute_horizon(const IBlockModel& block,
                                                AblationMode ablation = AblationMode::None) const;

    double compute_break_even_r(const IBlockModel& block, PartitionPlan candidate_plan,
                                PartitionPlan reference_plan,
                                AblationMode ablation = AblationMode::None) const;

    // Dynamic-M analysis: finds crossover threshold M* in [m_min, m_max] for fixed model_r_runs
    DynamicMDecision analyze_dynamic_m(const MlpConfig& base_cfg, uint32_t m_min, uint32_t m_max,
                                       uint64_t model_r_runs = 1,
                                       AblationMode ablation = AblationMode::None) const;

private:
    double wave_occupancy(uint32_t workgroups) const;
    double kernel_compile_ms(const KernelWorkload& kw) const;

    DeviceProfile  prof_;
    TopologyPolicy topo_;
};

} // namespace mgcm
