#pragma once
#include "mgcm/device_profile.hpp"
#include <optional>
#include <string>

namespace mgcm {

// Integration Hook #3: Disk cache keyed by (runtime, device_uuid, driver_uuid, schema_version).
class ProfileCacheStore {
public:
    explicit ProfileCacheStore(std::string cache_dir);

    std::string                  make_cache_path(const StaticDeviceProps& props) const;
    std::optional<DeviceProfile> try_load(const StaticDeviceProps& props) const;
    void                         save(const DeviceProfile& profile) const;

private:
    std::string cache_dir_;
};

} // namespace mgcm
