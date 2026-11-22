#pragma once
#include "mgcm/backend.hpp"
#include <memory>
#include <vector>

namespace mgcm {

struct VulkanDeviceEntry {
    uint32_t    index = 0;
    std::string name;
    std::string driver;
    bool        is_cpu = false;
};

class VulkanBackend final : public ICalibBackend {
public:
    static std::vector<VulkanDeviceEntry> enumerate_devices(bool include_cpu = false);

    explicit VulkanBackend(uint32_t device_index, bool include_cpu = false);
    ~VulkanBackend() override;

    RuntimeKind       runtime_kind() const override { return RuntimeKind::Vulkan; }
    StaticDeviceProps query_static_props() const override;

    BufferHandle allocate_buffer(size_t size_bytes) override;
    void         free_buffer(BufferHandle buf) override;

    KernelHandle compile_kernel(BenchKernelId id, uint32_t salt, double* out_cpu_compile_ns) override;
    void         destroy_kernel(KernelHandle kernel) override;

    TimingResult execute(const DispatchDesc& desc) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mgcm
