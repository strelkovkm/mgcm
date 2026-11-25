#pragma once
#include "mgcm/device_profile.hpp"

namespace mgcm {

enum class TopologyKind : uint8_t { DesktopDiscrete = 0, UnifiedUma = 1 };

inline const char* to_string(TopologyKind t) {
    return (t == TopologyKind::UnifiedUma) ? "unified_uma" : "desktop_discrete";
}

struct TopologyPolicy {
    TopologyKind kind = TopologyKind::DesktopDiscrete;

    // Optional heterogeneous GPU <-> CPU extension parameters (unused for intra-GPU subgraphs)
    double pcie_h2d_bw_gbps       = 16.0;
    double pcie_d2h_bw_gbps       = 12.0;
    double host_ram_bw_gbps       = 60.0;
    double uma_contention_factor  = 1.0; // 1.0 for isolated GPU; <1.0 under concurrent CPU load
    double uma_coherency_sync_us  = 0.0;

    static TopologyPolicy from_profile(const DeviceProfile& prof) {
        TopologyPolicy tp{};
        tp.kind = prof.props.is_integrated_uma ? TopologyKind::UnifiedUma : TopologyKind::DesktopDiscrete;
        return tp;
    }

    double external_cut_us(double bytes, bool device_to_host = false) const {
        if (bytes <= 0.0) {
            return 0.0;
        }
        if (kind == TopologyKind::UnifiedUma) {
            return uma_coherency_sync_us;
        }
        double pcie_bw = device_to_host ? pcie_d2h_bw_gbps : pcie_h2d_bw_gbps;
        return ((bytes / host_ram_bw_gbps) + (bytes / pcie_bw)) * 1e-3;
    }
};

} // namespace mgcm
