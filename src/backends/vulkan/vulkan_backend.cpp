#include "mgcm/backends/vulkan_backend.hpp"

#include <vulkan/vulkan.h>

#include "init_prng_vec4.hpp"
#include "mb0_warmup.hpp"
#include "mb1_noop.hpp"
#include "mb2_stream_vec4.hpp"
#include "mb3_stride_f32.hpp"
#include "mb4_matmul_cache.hpp"
#include "mb4_matmul_smem.hpp"
#include "mb4_matmul_smem_tail.hpp"
#include "mb5_fma16.hpp"
#include "mb5_gelu_erf.hpp"
#include "mb5_gelu_tanh.hpp"
#include "mb6_matmul_fused_gelu_erf.hpp"
#include "ora_elem_add.hpp"
#include "ora_elem_add_gelu.hpp"
#include "ora_elem_add_relu.hpp"
#include "ora_elem_gelu.hpp"
#include "ora_elem_relu.hpp"
#include "ora_matmul_bias.hpp"
#include "ora_matmul_bias_gelu.hpp"
#include "ora_matmul_bias_relu.hpp"
#include "ora_matmul_only.hpp" 

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#define VK_CHECK(call)                                                                                \
    do {                                                                                              \
        VkResult res_ = (call);                                                                       \
        if (res_ != VK_SUCCESS) {                                                                     \
            throw std::runtime_error(std::string("Vulkan error in ") + #call + ": " +                 \
                                     std::to_string(static_cast<int>(res_)));                         \
        }                                                                                             \
    } while (0)

namespace mgcm {
namespace {

std::string uuid_to_hex(const uint8_t uuid[VK_UUID_SIZE]) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < VK_UUID_SIZE; ++i) {
        oss << std::setw(2) << static_cast<int>(uuid[i]);
    }
    return oss.str();
}

std::string format_vk_version(uint32_t v) {
    return std::to_string(VK_API_VERSION_MAJOR(v)) + "." + std::to_string(VK_API_VERSION_MINOR(v)) + "." +
           std::to_string(VK_API_VERSION_PATCH(v));
}

struct SpirvReflection {
    uint32_t                num_bindings = 0;
    std::array<uint32_t, 3> local_size{1, 1, 1};
};

SpirvReflection parse_and_validate_canonical_abi(std::vector<uint32_t>& words, uint32_t salt) {
    if (words.size() < 5 || words[0] != 0x07230203u) {
        throw std::runtime_error("Invalid SPIR-V magic");
    }
    bool has_shader_cap      = false;
    uint32_t entry_points    = 0;
    SpirvReflection refl{};
    std::unordered_map<uint32_t, uint32_t> id_set, id_binding, float_types;
    bool salted = (salt == 0);

    size_t i = 5;
    while (i < words.size()) {
        auto len = static_cast<uint16_t>(words[i] >> 16);
        auto op  = static_cast<uint16_t>(words[i] & 0xFFFFu);
        if (len == 0 || i + len > words.size()) {
            throw std::runtime_error("Corrupt SPIR-V stream");
        }
        if (op == 17 && len >= 2 && words[i + 1] == 11) {
            throw std::runtime_error("ABI violation: OpCapability Int64 is forbidden for mobile GPU portability");
        }
        if (op == 17 && len >= 2 && words[i + 1] == 1) {
            has_shader_cap = true;
        } else if (op == 15) {
            if (words[i + 1] == 5) {
                ++entry_points;
            }
        } else if (op == 16 && len >= 6 && words[i + 2] == 17) {
            refl.local_size = {words[i + 3], words[i + 4], words[i + 5]};
        } else if (op == 71 && len >= 4) {
            uint32_t target = words[i + 1], dec = words[i + 2];
            if (dec == 34) {
                id_set[target] = words[i + 3];
            }
            if (dec == 33) {
                id_binding[target] = words[i + 3];
            }
        } else if (op == 22 && len == 3 && words[i + 2] == 32) {
            float_types[words[i + 1]] = 32;
        } else if (op == 43 && len == 4 && !salted && float_types.count(words[i + 1])) {
            uint32_t bits = words[i + 3];
            if ((bits & 0x7F800000u) != 0 && (bits & 0x7F800000u) != 0x7F800000u) {
                words[i + 3] = (bits & 0xFF800000u) | ((bits ^ salt) & 0x007FFFFFu);
                salted       = true;
            }
        }
        i += len;
    }

    if (salt != 0 && !salted) {
        throw std::runtime_error("SPIR-V salting failed: no non-zero float32 OpConstant");
    }
    if (!has_shader_cap) {
        throw std::runtime_error("ABI violation: missing OpCapability Shader");
    }
    if (entry_points != 1) {
        throw std::runtime_error("ABI violation: expected exactly 1 GLCompute OpEntryPoint");
    }

    std::vector<uint32_t> bindings;
    for (const auto& [var_id, b] : id_binding) {
        uint32_t dset = id_set.count(var_id) ? id_set[var_id] : 999u;
        if (dset != 0) {
            throw std::runtime_error("ABI violation: DescriptorSet must be 0");
        }
        bindings.push_back(b);
    }
    std::sort(bindings.begin(), bindings.end());
    for (uint32_t idx = 0; idx < bindings.size(); ++idx) {
        if (bindings[idx] != idx) {
            throw std::runtime_error("ABI violation: non-contiguous bindings");
        }
    }
    refl.num_bindings = static_cast<uint32_t>(bindings.size());
    return refl;
}

std::span<const uint32_t> get_builtin_spirv(BenchKernelId id, const char** out_entry) {
    switch (id) {
        case BenchKernelId::InitPrngVec4:
            *out_entry = "init_prng_vec4";
            return spirv::k_init_prng_vec4;
        case BenchKernelId::MB0_Warmup:
            *out_entry = "mb0_warmup";
            return spirv::k_mb0_warmup;
        case BenchKernelId::MB1_DispatchNoop:
            *out_entry = "mb1_noop";
            return spirv::k_mb1_noop;
        case BenchKernelId::MB2_CacheStreamVec4:
            *out_entry = "mb2_stream_vec4";
            return spirv::k_mb2_stream_vec4;
        case BenchKernelId::MB3_StrideF32:
            *out_entry = "mb3_stride_f32";
            return spirv::k_mb3_stride_f32;
        case BenchKernelId::MB4_MatmulSmem:
            *out_entry = "mb4_matmul_smem";
            return spirv::k_mb4_matmul_smem;
        case BenchKernelId::MB4_MatmulSmemTail:
            *out_entry = "mb4_matmul_smem_tail";
            return spirv::k_mb4_matmul_smem_tail;
        case BenchKernelId::MB4_MatmulDirectCache:
            *out_entry = "mb4_matmul_cache";
            return spirv::k_mb4_matmul_cache;
        case BenchKernelId::MB5_Fma16:
            *out_entry = "mb5_fma16";
            return spirv::k_mb5_fma16;
        case BenchKernelId::MB5_GeluTanh:
            *out_entry = "mb5_gelu_tanh";
            return spirv::k_mb5_gelu_tanh;
        case BenchKernelId::MB5_GeluErf:
            *out_entry = "mb5_gelu_erf";
            return spirv::k_mb5_gelu_erf;
        case BenchKernelId::MB6_MatmulFusedGeluErf: *out_entry = "mb6_matmul_fused_gelu_erf"; return spirv::k_mb6_matmul_fused_gelu_erf;
        case BenchKernelId::Ora_MatmulOnly: *out_entry = "ora_matmul_only"; return spirv::k_ora_matmul_only;
        case BenchKernelId::Ora_MatmulBias: *out_entry = "ora_matmul_bias"; return spirv::k_ora_matmul_bias;
        case BenchKernelId::Ora_MatmulBiasGelu: *out_entry = "ora_matmul_bias_gelu"; return spirv::k_ora_matmul_bias_gelu;
        case BenchKernelId::Ora_MatmulBiasRelu: *out_entry = "ora_matmul_bias_relu"; return spirv::k_ora_matmul_bias_relu;
        case BenchKernelId::Ora_ElemAdd: *out_entry = "ora_elem_add"; return spirv::k_ora_elem_add;
        case BenchKernelId::Ora_ElemGelu: *out_entry = "ora_elem_gelu"; return spirv::k_ora_elem_gelu;
        case BenchKernelId::Ora_ElemRelu: *out_entry = "ora_elem_relu"; return spirv::k_ora_elem_relu;
        case BenchKernelId::Ora_ElemAddGelu: *out_entry = "ora_elem_add_gelu"; return spirv::k_ora_elem_add_gelu;
        case BenchKernelId::Ora_ElemAddRelu: *out_entry = "ora_elem_add_relu"; return spirv::k_ora_elem_add_relu;
    }
    throw std::runtime_error("Unknown BenchKernelId");
}

} // namespace

struct VulkanBackend::Impl {
    VkInstance                instance                  = VK_NULL_HANDLE;
    VkPhysicalDevice          phys_dev                  = VK_NULL_HANDLE;
    VkDevice                  device                    = VK_NULL_HANDLE;
    VkQueue                   queue                     = VK_NULL_HANDLE;
    uint32_t                  queue_family              = 0;
    VkCommandPool             cmd_pool                  = VK_NULL_HANDLE;
    VkCommandBuffer           cmd_buf                   = VK_NULL_HANDLE;
    VkFence                   fence                     = VK_NULL_HANDLE;
    VkQueryPool               query_pool                = VK_NULL_HANDLE;
    VkDescriptorPool          desc_pool                 = VK_NULL_HANDLE;
    bool                      owns_device               = true;
    bool                      use_sync2                 = false;
    PFN_vkCmdPipelineBarrier2 pfn_cmd_pipeline_barrier2 = nullptr;
    PFN_vkCmdWriteTimestamp2  pfn_cmd_write_timestamp2  = nullptr;

    StaticDeviceProps props{};

    struct BufferObj {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   size   = 0;
    };
    struct KernelObj {
        VkShaderModule        shader      = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout  = VK_NULL_HANDLE;
        VkPipelineLayout      pipe_layout = VK_NULL_HANDLE;
        VkPipeline            pipeline    = VK_NULL_HANDLE;
        VkDescriptorSet       desc_set    = VK_NULL_HANDLE;
        SpirvReflection       refl{};
    };

    uint64_t                          next_handle = 1;
    std::map<BufferHandle, BufferObj> buffers;
    std::map<KernelHandle, KernelObj> kernels;
};

std::vector<VulkanDeviceEntry> VulkanBackend::enumerate_devices(bool include_cpu) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "mgcm";
    app.apiVersion       = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;

    VkInstance inst = VK_NULL_HANDLE;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(inst, &count, devs.data());

    auto pfn_gpdp2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceProperties2"));

    std::vector<VulkanDeviceEntry> out;
    uint32_t logical_idx = 0;
    for (auto pd : devs) {
        VkPhysicalDeviceProperties base_pr{};
        vkGetPhysicalDeviceProperties(pd, &base_pr);
        bool is_cpu = (base_pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU);
        if (is_cpu && !include_cpu) {
            continue;
        }
        std::string drv = "Vulkan " + format_vk_version(base_pr.apiVersion);
        if (pfn_gpdp2 && base_pr.apiVersion >= VK_API_VERSION_1_2) {
            VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p12};
            pfn_gpdp2(pd, &p2);
            if (p12.driverName[0] != '\0') {
                drv = std::string(p12.driverName) + " / VK " + format_vk_version(base_pr.apiVersion);
            }
        }
        out.push_back({logical_idx++, base_pr.deviceName, drv, is_cpu});
    }
    vkDestroyInstance(inst, nullptr);
    return out;
}

VulkanBackend::VulkanBackend(uint32_t device_index, bool include_cpu) : impl_(std::make_unique<Impl>()) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "mgcm";
    app.apiVersion       = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &impl_->instance));

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(impl_->instance, &count, nullptr);
    std::vector<VkPhysicalDevice> all_devs(count);
    vkEnumeratePhysicalDevices(impl_->instance, &count, all_devs.data());

    std::vector<VkPhysicalDevice> filtered;
    for (auto pd : all_devs) {
        VkPhysicalDeviceProperties pr{};
        vkGetPhysicalDeviceProperties(pd, &pr);
        if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !include_cpu) {
            continue;
        }
        filtered.push_back(pd);
    }
    if (device_index >= filtered.size()) {
        throw std::runtime_error("Requested Vulkan device_index out of range");
    }
    impl_->phys_dev = filtered[device_index];

    VkPhysicalDeviceProperties base_pr{};
    vkGetPhysicalDeviceProperties(impl_->phys_dev, &base_pr);
    uint32_t dev_api = base_pr.apiVersion;
    if (dev_api < VK_API_VERSION_1_3) {
        throw std::runtime_error("Vulkan 1.3 required (device reports " + format_vk_version(dev_api) + ")");
    }

    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};

    if (dev_api >= VK_API_VERSION_1_3) {
        p11.pNext = &p12;
        p12.pNext = &p13;
    } else if (dev_api >= VK_API_VERSION_1_2) {
        p11.pNext = &p12;
    }

    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p11};
    auto pfn_gpdp2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        vkGetInstanceProcAddr(impl_->instance, "vkGetPhysicalDeviceProperties2"));
    if (pfn_gpdp2 && dev_api >= VK_API_VERSION_1_1) {
        pfn_gpdp2(impl_->phys_dev, &p2);
    } else {
        p2.properties = base_pr;
    }

    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(impl_->phys_dev, &mem_props);

    uint64_t dev_local_bytes = 0;
    for (uint32_t h = 0; h < mem_props.memoryHeapCount; ++h) {
        if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            dev_local_bytes = std::max<uint64_t>(dev_local_bytes, mem_props.memoryHeaps[h].size);
        }
    }

    auto& sp                    = impl_->props;
    sp.runtime                  = RuntimeKind::Vulkan;
    sp.device_name              = p2.properties.deviceName;
    sp.api_version_str          = format_vk_version(dev_api);
    sp.driver_name              = (dev_api >= VK_API_VERSION_1_2 && p12.driverName[0] != '\0')
                                      ? (std::string(p12.driverName) + " / VK " + sp.api_version_str)
                                      : ("VK " + sp.api_version_str);
    sp.vendor_id                = p2.properties.vendorID;
    sp.device_id                = p2.properties.deviceID;
    sp.device_uuid_hex          = uuid_to_hex(p11.deviceUUID);
    sp.driver_uuid_hex          = uuid_to_hex(p11.driverUUID);
    sp.is_integrated_uma        = (p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ||
                                  (mem_props.memoryHeapCount == 1);
    sp.subgroup_size            = (p11.subgroupSize > 0) ? p11.subgroupSize : 16u;
    sp.min_subgroup_size        = (dev_api >= VK_API_VERSION_1_3 && p13.minSubgroupSize > 0) ? p13.minSubgroupSize : sp.subgroup_size;
    sp.max_subgroup_size        = (dev_api >= VK_API_VERSION_1_3 && p13.maxSubgroupSize > 0) ? p13.maxSubgroupSize : sp.subgroup_size;
    sp.max_wg_invocations       = p2.properties.limits.maxComputeWorkGroupInvocations;
    sp.max_shared_mem_bytes     = p2.properties.limits.maxComputeSharedMemorySize;
    sp.max_push_constants_size  = p2.properties.limits.maxPushConstantsSize;
    sp.max_storage_buffer_range = p2.properties.limits.maxStorageBufferRange;
    sp.device_local_mem_bytes   = dev_local_bytes;
    sp.timestamp_period_ns      = static_cast<double>(p2.properties.limits.timestampPeriod);

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(impl_->phys_dev, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(impl_->phys_dev, &qcount, qprops.data());

    bool found_q = false;
    for (uint32_t q = 0; q < qcount; ++q) {
        if ((qprops[q].queueFlags & VK_QUEUE_COMPUTE_BIT) && qprops[q].timestampValidBits > 0) {
            impl_->queue_family = q;
            found_q             = true;
            if (!(qprops[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                break;
            }
        }
    }
    if (!found_q) {
        throw std::runtime_error("No compute queue with timestamp support found");
    }

    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(impl_->phys_dev, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(impl_->phys_dev, nullptr, &ext_count, exts.data());
    auto has_ext = [&](const char* name) {
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, name) == 0) {
                return true;
            }
        }
        return false;
    };

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = impl_->queue_family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.synchronization2 = VK_TRUE;
    f13.maintenance4     = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore       = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;

    VkPhysicalDeviceSynchronization2FeaturesKHR sync2_khr{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
    sync2_khr.synchronization2 = VK_TRUE;

    std::vector<const char*> enabled_exts;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;

    if (dev_api >= VK_API_VERSION_1_3) {
        dci.pNext = &f12;
    } else if (has_ext(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) {
        enabled_exts.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        dci.pNext = &sync2_khr;
    }
    dci.enabledExtensionCount   = static_cast<uint32_t>(enabled_exts.size());
    dci.ppEnabledExtensionNames = enabled_exts.empty() ? nullptr : enabled_exts.data();

    VK_CHECK(vkCreateDevice(impl_->phys_dev, &dci, nullptr, &impl_->device));
    vkGetDeviceQueue(impl_->device, impl_->queue_family, 0, &impl_->queue);

    if (dev_api >= VK_API_VERSION_1_3) {
        impl_->pfn_cmd_pipeline_barrier2 =
            reinterpret_cast<PFN_vkCmdPipelineBarrier2>(vkGetDeviceProcAddr(impl_->device, "vkCmdPipelineBarrier2"));
        impl_->pfn_cmd_write_timestamp2 =
            reinterpret_cast<PFN_vkCmdWriteTimestamp2>(vkGetDeviceProcAddr(impl_->device, "vkCmdWriteTimestamp2"));
    } else if (!enabled_exts.empty()) {
        impl_->pfn_cmd_pipeline_barrier2 =
            reinterpret_cast<PFN_vkCmdPipelineBarrier2>(vkGetDeviceProcAddr(impl_->device, "vkCmdPipelineBarrier2KHR"));
        impl_->pfn_cmd_write_timestamp2 =
            reinterpret_cast<PFN_vkCmdWriteTimestamp2>(vkGetDeviceProcAddr(impl_->device, "vkCmdWriteTimestamp2KHR"));
    }
    impl_->use_sync2 = (impl_->pfn_cmd_pipeline_barrier2 != nullptr && impl_->pfn_cmd_write_timestamp2 != nullptr);
    if (!impl_->use_sync2) {
        throw std::runtime_error("Vulkan 1.3 Synchronization2 entry points are required");
    }

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = impl_->queue_family;
    VK_CHECK(vkCreateCommandPool(impl_->device, &cpci, nullptr, &impl_->cmd_pool));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool        = impl_->cmd_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(impl_->device, &cbai, &impl_->cmd_buf));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(impl_->device, &fci, nullptr, &impl_->fence));

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;
    VK_CHECK(vkCreateQueryPool(impl_->device, &qpci, nullptr, &impl_->query_pool));

    VkDescriptorPoolSize pool_sz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets       = 64;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_sz;
    VK_CHECK(vkCreateDescriptorPool(impl_->device, &dpci, nullptr, &impl_->desc_pool));
}

VulkanBackend::~VulkanBackend() {
    if (!impl_ || !impl_->device) {
        return;
    }
    vkDeviceWaitIdle(impl_->device);
    for (auto& [_, k] : impl_->kernels) {
        vkDestroyPipeline(impl_->device, k.pipeline, nullptr);
        vkDestroyPipelineLayout(impl_->device, k.pipe_layout, nullptr);
        vkDestroyDescriptorSetLayout(impl_->device, k.set_layout, nullptr);
        vkDestroyShaderModule(impl_->device, k.shader, nullptr);
    }
    for (auto& [_, b] : impl_->buffers) {
        vkDestroyBuffer(impl_->device, b.buffer, nullptr);
        vkFreeMemory(impl_->device, b.memory, nullptr);
    }
    vkDestroyDescriptorPool(impl_->device, impl_->desc_pool, nullptr);
    vkDestroyQueryPool(impl_->device, impl_->query_pool, nullptr);
    vkDestroyFence(impl_->device, impl_->fence, nullptr);
    vkDestroyCommandPool(impl_->device, impl_->cmd_pool, nullptr);
    if (impl_->owns_device) {
        vkDestroyDevice(impl_->device, nullptr);
        vkDestroyInstance(impl_->instance, nullptr);
    }
}

StaticDeviceProps VulkanBackend::query_static_props() const { return impl_->props; }

BufferHandle VulkanBackend::allocate_buffer(size_t size_bytes) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = size_bytes;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    Impl::BufferObj bo{};
    bo.size = size_bytes;
    VK_CHECK(vkCreateBuffer(impl_->device, &bci, nullptr, &bo.buffer));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(impl_->device, bo.buffer, &req);

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(impl_->phys_dev, &mp);

    uint32_t chosen_type = UINT32_MAX;
    for (int pass = 0; pass < 2 && chosen_type == UINT32_MAX; ++pass) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if (!(req.memoryTypeBits & (1u << i))) {
                continue;
            }
            auto flags = mp.memoryTypes[i].propertyFlags;
            if (!(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                continue;
            }
            if (pass == 0 && !impl_->props.is_integrated_uma && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                continue;
            }
            chosen_type = i;
            break;
        }
    }
    if (chosen_type == UINT32_MAX) {
        throw std::runtime_error("Failed to find DEVICE_LOCAL memory type");
    }

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = chosen_type;
    VK_CHECK(vkAllocateMemory(impl_->device, &mai, nullptr, &bo.memory));
    VK_CHECK(vkBindBufferMemory(impl_->device, bo.buffer, bo.memory, 0));

    BufferHandle h    = impl_->next_handle++;
    impl_->buffers[h] = bo;
    return h;
}

void VulkanBackend::free_buffer(BufferHandle buf) {
    auto it = impl_->buffers.find(buf);
    if (it == impl_->buffers.end()) {
        return;
    }
    vkDeviceWaitIdle(impl_->device);
    vkDestroyBuffer(impl_->device, it->second.buffer, nullptr);
    vkFreeMemory(impl_->device, it->second.memory, nullptr);
    impl_->buffers.erase(it);
}

KernelHandle VulkanBackend::compile_kernel(BenchKernelId id, uint32_t salt, double* out_cpu_compile_ns) {
    const char* entry_name = nullptr;
    auto base_spv          = get_builtin_spirv(id, &entry_name);
    std::vector<uint32_t> words(base_spv.begin(), base_spv.end());

    Impl::KernelObj ko{};
    ko.refl = parse_and_validate_canonical_abi(words, salt);

    std::vector<VkDescriptorSetLayoutBinding> bindings(ko.refl.num_bindings);
    for (uint32_t b = 0; b < ko.refl.num_bindings; ++b) {
        bindings[b] = {b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = ko.refl.num_bindings;
    slci.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(impl_->device, &slci, nullptr, &ko.set_layout));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &ko.set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(impl_->device, &plci, nullptr, &ko.pipe_layout));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = words.size() * sizeof(uint32_t);
    smci.pCode    = words.data();

    auto t0 = std::chrono::steady_clock::now();
    VK_CHECK(vkCreateShaderModule(impl_->device, &smci, nullptr, &ko.shader));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = ko.shader;
    cpci.stage.pName  = entry_name;
    cpci.layout       = ko.pipe_layout;
    VK_CHECK(vkCreateComputePipelines(impl_->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &ko.pipeline));
    auto t1 = std::chrono::steady_clock::now();

    if (out_cpu_compile_ns) {
        *out_cpu_compile_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = impl_->desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ko.set_layout;
    VK_CHECK(vkAllocateDescriptorSets(impl_->device, &dsai, &ko.desc_set));

    KernelHandle h    = impl_->next_handle++;
    impl_->kernels[h] = ko;
    return h;
}

void VulkanBackend::destroy_kernel(KernelHandle kernel) {
    auto it = impl_->kernels.find(kernel);
    if (it == impl_->kernels.end()) {
        return;
    }
    vkDeviceWaitIdle(impl_->device);
    vkFreeDescriptorSets(impl_->device, impl_->desc_pool, 1, &it->second.desc_set);
    vkDestroyPipeline(impl_->device, it->second.pipeline, nullptr);
    vkDestroyPipelineLayout(impl_->device, it->second.pipe_layout, nullptr);
    vkDestroyDescriptorSetLayout(impl_->device, it->second.set_layout, nullptr);
    vkDestroyShaderModule(impl_->device, it->second.shader, nullptr);
    impl_->kernels.erase(it);
}

TimingResult VulkanBackend::execute(const DispatchDesc& desc) {
    auto& ko = impl_->kernels.at(desc.kernel);
    if (desc.buffers.size() != ko.refl.num_bindings) {
        throw std::runtime_error("Buffer count mismatch with shader ABI");
    }
    if (desc.scalars.size() * sizeof(Scalar32) > 128) {
        throw std::runtime_error("PushConstants exceed 128B Canonical ABI limit");
    }

    std::vector<VkDescriptorBufferInfo> dbi(desc.buffers.size());
    std::vector<VkWriteDescriptorSet> writes(desc.buffers.size());
    for (size_t i = 0; i < desc.buffers.size(); ++i) {
        const auto& bo = impl_->buffers.at(desc.buffers[i]);
        dbi[i]         = {bo.buffer, 0, bo.size};
        writes[i]      = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                          nullptr,
                          ko.desc_set,
                          static_cast<uint32_t>(i),
                          0,
                          1,
                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          nullptr,
                          &dbi[i],
                          nullptr};
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(impl_->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    std::array<Scalar32, 32> push_buf{};
    std::copy(desc.scalars.begin(), desc.scalars.end(), push_buf.begin());

    VK_CHECK(vkResetCommandBuffer(impl_->cmd_buf, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(impl_->cmd_buf, &bi));

    vkCmdResetQueryPool(impl_->cmd_buf, impl_->query_pool, 0, 2);
    vkCmdBindPipeline(impl_->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ko.pipeline);
    vkCmdBindDescriptorSets(
        impl_->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ko.pipe_layout, 0, 1, &ko.desc_set, 0, nullptr);
    vkCmdPushConstants(impl_->cmd_buf, ko.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, push_buf.data());

    if (impl_->use_sync2) {
        impl_->pfn_cmd_write_timestamp2(impl_->cmd_buf, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, impl_->query_pool, 0);
    } else {
        vkCmdWriteTimestamp(impl_->cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, impl_->query_pool, 0);
    }

    for (uint32_t r = 0; r < desc.repeat_in_cmdbuf; ++r) {
        if (r > 0 && desc.barrier_between_repeats) {
            if (impl_->use_sync2) {
                VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers    = &mb;
                impl_->pfn_cmd_pipeline_barrier2(impl_->cmd_buf, &dep);
            } else {
                VkMemoryBarrier mb1{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                mb1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(impl_->cmd_buf,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0,
                                     1,
                                     &mb1,
                                     0,
                                     nullptr,
                                     0,
                                     nullptr);
            }
        }
        vkCmdDispatch(impl_->cmd_buf, desc.group_count[0], desc.group_count[1], desc.group_count[2]);
    }

    if (impl_->use_sync2) {
        impl_->pfn_cmd_write_timestamp2(impl_->cmd_buf, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, impl_->query_pool, 1);
    } else {
        vkCmdWriteTimestamp(impl_->cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, impl_->query_pool, 1);
    }
    VK_CHECK(vkEndCommandBuffer(impl_->cmd_buf));

    VK_CHECK(vkResetFences(impl_->device, 1, &impl_->fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &impl_->cmd_buf;

    auto cpu_t0 = std::chrono::steady_clock::now();
    VK_CHECK(vkQueueSubmit(impl_->queue, 1, &si, impl_->fence));
    VK_CHECK(vkWaitForFences(impl_->device, 1, &impl_->fence, VK_TRUE, UINT64_MAX));
    auto cpu_t1 = std::chrono::steady_clock::now();

    uint64_t ts[2] = {0, 0};
    VK_CHECK(vkGetQueryPoolResults(impl_->device,
                                   impl_->query_pool,
                                   0,
                                   2,
                                   sizeof(ts),
                                   ts,
                                   sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

    TimingResult tr{};
    tr.cpu_wall_ns      = std::chrono::duration<double, std::nano>(cpu_t1 - cpu_t0).count();
    tr.gpu_execution_ns = static_cast<double>(ts[1] - ts[0]) * impl_->props.timestamp_period_ns;
    return tr;
}


std::unique_ptr<VulkanBackend> VulkanBackend::from_external(const ExternalVulkanHandles& handles) {
    return std::unique_ptr<VulkanBackend>(new VulkanBackend(handles));
}

VulkanBackend::VulkanBackend(const ExternalVulkanHandles& ext) : impl_(std::make_unique<Impl>()) {
    if (!ext.instance || !ext.physical_device || !ext.device || !ext.queue) {
        throw std::runtime_error("ExternalVulkanHandles contains null handles");
    }
    impl_->owns_device  = false;
    impl_->instance     = reinterpret_cast<VkInstance>(ext.instance);
    impl_->phys_dev     = reinterpret_cast<VkPhysicalDevice>(ext.physical_device);
    impl_->device       = reinterpret_cast<VkDevice>(ext.device);
    impl_->queue        = reinterpret_cast<VkQueue>(ext.queue);
    impl_->queue_family = ext.queue_family_idx;

    VkPhysicalDeviceProperties base_pr{};
    vkGetPhysicalDeviceProperties(impl_->phys_dev, &base_pr);
    uint32_t dev_api = base_pr.apiVersion;
    if (dev_api < VK_API_VERSION_1_3) {
        throw std::runtime_error("Vulkan 1.3 required (device reports " + format_vk_version(dev_api) + ")");
    }

    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    if (dev_api >= VK_API_VERSION_1_3) { p11.pNext = &p12; p12.pNext = &p13; }
    else if (dev_api >= VK_API_VERSION_1_2) { p11.pNext = &p12; }

    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p11};
    auto pfn_gpdp2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        vkGetInstanceProcAddr(impl_->instance, "vkGetPhysicalDeviceProperties2"));
    if (pfn_gpdp2 && dev_api >= VK_API_VERSION_1_1) { pfn_gpdp2(impl_->phys_dev, &p2); }
    else { p2.properties = base_pr; }

    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(impl_->phys_dev, &mem_props);
    uint64_t dev_local_bytes = 0;
    for (uint32_t h = 0; h < mem_props.memoryHeapCount; ++h) {
        if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            dev_local_bytes = std::max<uint64_t>(dev_local_bytes, mem_props.memoryHeaps[h].size);
        }
    }

    auto& sp                    = impl_->props;
    sp.runtime                  = RuntimeKind::Vulkan;
    sp.device_name              = p2.properties.deviceName;
    sp.api_version_str          = format_vk_version(dev_api);
    sp.driver_name              = (dev_api >= VK_API_VERSION_1_2 && p12.driverName[0] != '\0')
                                      ? (std::string(p12.driverName) + " / VK " + sp.api_version_str)
                                      : ("VK " + sp.api_version_str);
    sp.vendor_id                = p2.properties.vendorID;
    sp.device_id                = p2.properties.deviceID;
    sp.device_uuid_hex          = uuid_to_hex(p11.deviceUUID);
    sp.driver_uuid_hex          = uuid_to_hex(p11.driverUUID);
    sp.is_integrated_uma        = (p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ||
                                  (mem_props.memoryHeapCount == 1);
    sp.subgroup_size            = (p11.subgroupSize > 0) ? p11.subgroupSize : 16u;
    sp.min_subgroup_size        = (dev_api >= VK_API_VERSION_1_3 && p13.minSubgroupSize > 0) ? p13.minSubgroupSize : sp.subgroup_size;
    sp.max_subgroup_size        = (dev_api >= VK_API_VERSION_1_3 && p13.maxSubgroupSize > 0) ? p13.maxSubgroupSize : sp.subgroup_size;
    sp.max_wg_invocations       = p2.properties.limits.maxComputeWorkGroupInvocations;
    sp.max_shared_mem_bytes     = p2.properties.limits.maxComputeSharedMemorySize;
    sp.max_push_constants_size  = p2.properties.limits.maxPushConstantsSize;
    sp.max_storage_buffer_range = p2.properties.limits.maxStorageBufferRange;
    sp.device_local_mem_bytes   = dev_local_bytes;
    sp.timestamp_period_ns      = static_cast<double>(p2.properties.limits.timestampPeriod);

    impl_->pfn_cmd_pipeline_barrier2 =
        reinterpret_cast<PFN_vkCmdPipelineBarrier2>(vkGetDeviceProcAddr(impl_->device, "vkCmdPipelineBarrier2"));
    impl_->pfn_cmd_write_timestamp2 =
        reinterpret_cast<PFN_vkCmdWriteTimestamp2>(vkGetDeviceProcAddr(impl_->device, "vkCmdWriteTimestamp2"));
    impl_->use_sync2 = (dev_api >= VK_API_VERSION_1_3 && impl_->pfn_cmd_pipeline_barrier2 && impl_->pfn_cmd_write_timestamp2);
    if (!impl_->use_sync2) {
        throw std::runtime_error("Vulkan 1.3 Synchronization2 entry points are required");
    }

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = impl_->queue_family;
    VK_CHECK(vkCreateCommandPool(impl_->device, &cpci, nullptr, &impl_->cmd_pool));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool        = impl_->cmd_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(impl_->device, &cbai, &impl_->cmd_buf));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(impl_->device, &fci, nullptr, &impl_->fence));

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;
    VK_CHECK(vkCreateQueryPool(impl_->device, &qpci, nullptr, &impl_->query_pool));

    VkDescriptorPoolSize pool_sz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets       = 64;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_sz;
    VK_CHECK(vkCreateDescriptorPool(impl_->device, &dpci, nullptr, &impl_->desc_pool));
}

size_t VulkanBackend::get_kernel_spirv_words(BenchKernelId id) const {
    const char* entry = nullptr;
    return get_builtin_spirv(id, &entry).size();
}

double VulkanBackend::measure_pipeline_cache_hit_ms(BenchKernelId id) {
    const char* entry_name = nullptr;
    auto base_spv          = get_builtin_spirv(id, &entry_name);
    std::vector<uint32_t> words(base_spv.begin(), base_spv.end());
    auto refl = parse_and_validate_canonical_abi(words, 0);

    std::vector<VkDescriptorSetLayoutBinding> bindings(refl.num_bindings);
    for (uint32_t b = 0; b < refl.num_bindings; ++b) {
        bindings[b] = {b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = refl.num_bindings;
    slci.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(impl_->device, &slci, nullptr, &set_layout));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(impl_->device, &plci, nullptr, &pipe_layout));

    VkShaderModule shader = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = words.size() * sizeof(uint32_t);
    smci.pCode    = words.data();
    VK_CHECK(vkCreateShaderModule(impl_->device, &smci, nullptr, &shader));

    VkPipelineCache cache = VK_NULL_HANDLE;
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VK_CHECK(vkCreatePipelineCache(impl_->device, &pcci, nullptr, &cache));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName  = entry_name;
    cpci.layout       = pipe_layout;

    // Warm up VkPipelineCache once
    VkPipeline warm = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(impl_->device, cache, 1, &cpci, nullptr, &warm));
    vkDestroyPipeline(impl_->device, warm, nullptr);

    std::vector<double> hit_ms;
    for (int it = 0; it < 15; ++it) {
        VkPipeline p = VK_NULL_HANDLE;
        auto t0      = std::chrono::steady_clock::now();
        VK_CHECK(vkCreateComputePipelines(impl_->device, cache, 1, &cpci, nullptr, &p));
        auto t1 = std::chrono::steady_clock::now();
        hit_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        vkDestroyPipeline(impl_->device, p, nullptr);
    }

    vkDestroyPipelineCache(impl_->device, cache, nullptr);
    vkDestroyShaderModule(impl_->device, shader, nullptr);
    vkDestroyPipelineLayout(impl_->device, pipe_layout, nullptr);
    vkDestroyDescriptorSetLayout(impl_->device, set_layout, nullptr);

    std::sort(hit_ms.begin(), hit_ms.end());
    return hit_ms[hit_ms.size() / 2];
}


TimingResult VulkanBackend::execute_chain(std::span<const DispatchDesc> chain) {
    if (chain.empty()) return {};
    std::vector<VkDescriptorSet> step_sets(chain.size(), VK_NULL_HANDLE);
    for (size_t s = 0; s < chain.size(); ++s) {
        auto& ko = impl_->kernels.at(chain[s].kernel);
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, impl_->desc_pool, 1, &ko.set_layout};
        VK_CHECK(vkAllocateDescriptorSets(impl_->device, &dsai, &step_sets[s]));
        std::vector<VkDescriptorBufferInfo> dbi(chain[s].buffers.size());
        std::vector<VkWriteDescriptorSet> wr(chain[s].buffers.size());
        for (size_t i = 0; i < chain[s].buffers.size(); ++i) {
            const auto& bo = impl_->buffers.at(chain[s].buffers[i]);
            dbi[i] = {bo.buffer, 0, bo.size};
            wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, step_sets[s], static_cast<uint32_t>(i), 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbi[i], nullptr};
        }
        vkUpdateDescriptorSets(impl_->device, static_cast<uint32_t>(wr.size()), wr.data(), 0, nullptr);
    }

    VK_CHECK(vkResetCommandBuffer(impl_->cmd_buf, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    VK_CHECK(vkBeginCommandBuffer(impl_->cmd_buf, &bi));
    vkCmdResetQueryPool(impl_->cmd_buf, impl_->query_pool, 0, 2);
    impl_->pfn_cmd_write_timestamp2(impl_->cmd_buf, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, impl_->query_pool, 0);

    for (size_t s = 0; s < chain.size(); ++s) {
        if (s > 0) {
            VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &mb, 0, nullptr, 0, nullptr};
            impl_->pfn_cmd_pipeline_barrier2(impl_->cmd_buf, &dep);
        }
        auto& ko = impl_->kernels.at(chain[s].kernel);
        std::array<Scalar32, 32> push_buf{};
        std::copy(chain[s].scalars.begin(), chain[s].scalars.end(), push_buf.begin());
        vkCmdBindPipeline(impl_->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ko.pipeline);
        vkCmdBindDescriptorSets(impl_->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ko.pipe_layout, 0, 1, &step_sets[s], 0, nullptr);
        vkCmdPushConstants(impl_->cmd_buf, ko.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, push_buf.data());
        vkCmdDispatch(impl_->cmd_buf, chain[s].group_count[0], chain[s].group_count[1], chain[s].group_count[2]);
    }

    impl_->pfn_cmd_write_timestamp2(impl_->cmd_buf, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, impl_->query_pool, 1);
    VK_CHECK(vkEndCommandBuffer(impl_->cmd_buf));
    VK_CHECK(vkResetFences(impl_->device, 1, &impl_->fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &impl_->cmd_buf, 0, nullptr};
    auto t0 = std::chrono::steady_clock::now();
    VK_CHECK(vkQueueSubmit(impl_->queue, 1, &si, impl_->fence));
    VK_CHECK(vkWaitForFences(impl_->device, 1, &impl_->fence, VK_TRUE, UINT64_MAX));
    auto t1 = std::chrono::steady_clock::now();

    uint64_t ts[2] = {0, 0};
    VK_CHECK(vkGetQueryPoolResults(impl_->device, impl_->query_pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    vkFreeDescriptorSets(impl_->device, impl_->desc_pool, static_cast<uint32_t>(step_sets.size()), step_sets.data());
    return {std::chrono::duration<double, std::nano>(t1 - t0).count(),
            static_cast<double>(ts[1] - ts[0]) * impl_->props.timestamp_period_ns};
}

} // namespace mgcm
