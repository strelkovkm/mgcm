#include "mgcm/profile_cache.hpp"
#include "mgcm/calibrator.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace mgcm {

ProfileCacheStore::ProfileCacheStore(std::string cache_dir) : cache_dir_(std::move(cache_dir)) {}

std::string ProfileCacheStore::make_cache_path(const StaticDeviceProps& props) const {
    std::string dev = props.device_uuid_hex.empty() ? std::to_string(props.device_id) : props.device_uuid_hex;
    std::string drv = props.driver_uuid_hex.empty() ? "nodrv" : props.driver_uuid_hex;
    std::string fname = std::string("mgcm_") + to_string(props.runtime) + "_" + dev + "_" + drv + "_v" +
                        std::to_string(kDeviceProfileSchemaVersion) + ".json";
    return (std::filesystem::path(cache_dir_) / fname).string();
}

std::optional<DeviceProfile> ProfileCacheStore::try_load(const StaticDeviceProps& props) const {
    auto path = make_cache_path(props);
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    try {
        std::ifstream ifs(path);
        std::ostringstream ss;
        ss << ifs.rdbuf();
        auto prof = Calibrator::from_json_string(ss.str());
        if (prof.schema_version != kDeviceProfileSchemaVersion) {
            return std::nullopt;
        }
        if (prof.props.device_uuid_hex != props.device_uuid_hex ||
            prof.props.driver_uuid_hex != props.driver_uuid_hex) {
            return std::nullopt;
        }
        return prof;
    } catch (...) {
        return std::nullopt;
    }
}

void ProfileCacheStore::save(const DeviceProfile& profile) const {
    std::filesystem::create_directories(cache_dir_);
    auto path = make_cache_path(profile.props);
    std::ofstream ofs(path);
    ofs << Calibrator::to_json_string(profile) << "\n";
}

} // namespace mgcm
