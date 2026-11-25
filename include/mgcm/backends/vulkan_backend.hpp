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

// Opaque Vulkan handles borrowed from OpenVINO's cldnn::vulkan_engine (non-owning).
struct ExternalVulkanHandles {
    void*    instance         = nullptr; // VkInstance
    void*    physical_device  = nullptr; // VkPhysicalDevice
    void*    device           = nullptr; // VkDevice
    void*    queue            = nullptr; // VkQueue
    uint32_t queue_family_idx = 0;
};

class VulkanBackend final : public ICalibBackend {
public:
    static std::vector<VulkanDeviceEntry> enumerate_devices(bool include_cpu = false);

    explicit VulkanBackend(uint32_t device_index, bool include_cpu = false);
    static std::unique_ptr<VulkanBackend> from_external(const ExternalVulkanHandles& handles);
    ~VulkanBackend() override;

    RuntimeKind       runtime_kind() const override { return RuntimeKind::Vulkan; }
    StaticDeviceProps query_static_props() const override;

    BufferHandle allocate_buffer(size_t size_bytes) override;
    void         free_buffer(BufferHandle buf) override;
    void         upload_buffer(BufferHandle buf, const void* src, size_t bytes) override;
    void         download_buffer(BufferHandle buf, void* dst, size_t bytes) override;

    size_t       get_kernel_spirv_words(BenchKernelId id) const override;
    KernelHandle compile_kernel(BenchKernelId id, uint32_t salt, double* out_cpu_compile_ns = nullptr) override;
    void         destroy_kernel(KernelHandle kernel) override;
    double       measure_pipeline_cache_hit_ms(BenchKernelId id) override;

    TimingResult execute(const DispatchDesc& desc) override;
    TimingResult execute_chain(std::span<const DispatchDesc> chain) override;

private:
    explicit VulkanBackend(const ExternalVulkanHandles& external);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mgcm
