#pragma once
#include "mgcm/device_profile.hpp"
#include <string>

namespace mgcm {

struct CalibratorConfig {
    double   warmup_ms        = 90.0;
    uint32_t timed_iters      = 25;
    uint32_t compile_samples  = 7;
    size_t   max_buffer_bytes = 256ull * 1024ull * 1024ull;
    bool     verbose          = true;
};

class Calibrator {
public:
    explicit Calibrator(ICalibBackend& backend, CalibratorConfig cfg = {});
    DeviceProfile run_full();

    static std::string   to_json_string(const DeviceProfile& p);
    static DeviceProfile from_json_string(const std::string& json_text);
    static void        save_bw_csv(const DeviceProfile& p, const std::string& csv_path);
    static void        save_wave_csv(const DeviceProfile& p, const std::string& csv_path);

private:
    void run_warmup();
    void run_mb1_dispatch(DeviceProfile& out);
    void run_mb2_cache_hierarchy(DeviceProfile& out, BufferHandle shared_src_buf, size_t max_bytes);
    void run_mb3_stride(DeviceProfile& out, BufferHandle shared_src_buf, size_t max_bytes);
    void run_mb4_smem_vs_cache(DeviceProfile& out, BufferHandle shared_src_buf);
    void run_mb5_compute_and_waves(DeviceProfile& out);
    void run_mb6_compile_cost(DeviceProfile& out, BufferHandle shared_src_buf);

    ICalibBackend&   backend_;
    CalibratorConfig cfg_;
};

} // namespace mgcm
