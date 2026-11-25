#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace mgcm {

enum class RuntimeKind : uint8_t { Vulkan = 0, OpenCL = 1, LevelZero = 2 };

inline const char* to_string(RuntimeKind k) {
    switch (k) {
        case RuntimeKind::Vulkan:    return "vulkan";
        case RuntimeKind::OpenCL:    return "opencl";
        case RuntimeKind::LevelZero: return "level_zero";
    }
    return "unknown";
}

enum class CompilerKind : uint8_t { GraphCompiler = 0, Slang = 1, Clspv = 2 };

inline const char* to_string(CompilerKind c) {
    switch (c) {
        case CompilerKind::GraphCompiler: return "gc_mlir";
        case CompilerKind::Slang:         return "slang";
        case CompilerKind::Clspv:         return "clspv";
    }
    return "unknown";
}

enum class BenchKernelId : uint8_t {
    InitPrngVec4 = 0,
    MB0_Warmup,
    MB1_DispatchNoop,
    MB2_CacheStreamVec4,
    MB3_StrideF32,
    MB4_MatmulSmem,
    MB4_MatmulSmemTail,
    MB4_MatmulDirectCache,
    MB5_Fma16,
    MB5_GeluTanh,
    MB5_GeluErf,
    MB6_MatmulFusedGeluErf,
    Ora_MatmulOnly,
    Ora_MatmulBias,
    Ora_MatmulBiasGelu,
    Ora_MatmulBiasRelu,
    Ora_ElemAdd,
    Ora_ElemGelu,
    Ora_ElemRelu,
    Ora_ElemAddGelu,
    Ora_ElemAddRelu
};

using BufferHandle = uint64_t;
using KernelHandle = uint64_t;

union Scalar32 {
    uint32_t u32;
    int32_t  i32;
    float    f32;
    static Scalar32 U32(uint32_t v) { Scalar32 s{}; s.u32 = v; return s; }
    static Scalar32 F32(float v)    { Scalar32 s{}; s.f32 = v; return s; }
};

struct StaticDeviceProps {
    RuntimeKind runtime                  = RuntimeKind::Vulkan;
    std::string device_name;
    std::string driver_name;
    std::string api_version_str;
    uint32_t    vendor_id                = 0;
    uint32_t    device_id                = 0;
    std::string device_uuid_hex;
    std::string driver_uuid_hex;
    bool        is_integrated_uma        = false;
    uint32_t    subgroup_size            = 0;
    uint32_t    min_subgroup_size        = 0;
    uint32_t    max_subgroup_size        = 0;
    uint32_t    max_wg_invocations       = 0;
    uint32_t    max_shared_mem_bytes     = 0;
    uint32_t    max_push_constants_size  = 0;
    uint64_t    max_storage_buffer_range = 0;
    uint64_t    device_local_mem_bytes   = 0;
    double      timestamp_period_ns      = 1.0;
};

struct DispatchDesc {
    KernelHandle                  kernel = 0;
    std::span<const BufferHandle> buffers;
    std::span<const Scalar32>     scalars;
    std::array<uint32_t, 3>       group_count{1, 1, 1};
    std::array<uint32_t, 3>       local_size{1, 1, 1};
    uint32_t                      repeat_in_cmdbuf        = 1;
    bool                          barrier_between_repeats = true;
};

struct TimingResult {
    double cpu_wall_ns      = 0.0;
    double gpu_execution_ns = 0.0;
};

class ICalibBackend {
public:
    virtual ~ICalibBackend() = default;
    virtual RuntimeKind       runtime_kind() const = 0;
    virtual StaticDeviceProps query_static_props() const = 0;

    virtual BufferHandle allocate_buffer(size_t size_bytes) = 0;
    virtual void         free_buffer(BufferHandle buf) = 0;
    virtual void         upload_buffer(BufferHandle buf, const void* src, size_t bytes) = 0;
    virtual void         download_buffer(BufferHandle buf, void* dst, size_t bytes) = 0;

    virtual size_t       get_kernel_spirv_words(BenchKernelId id) const = 0;
    virtual KernelHandle compile_kernel(BenchKernelId id, uint32_t salt, double* out_cpu_compile_ns = nullptr) = 0;
    virtual void         destroy_kernel(KernelHandle kernel) = 0;
    virtual double       measure_pipeline_cache_hit_ms(BenchKernelId id) = 0;

    virtual TimingResult execute(const DispatchDesc& desc) = 0;
    virtual TimingResult execute_chain(std::span<const DispatchDesc> chain) = 0;
};

} // namespace mgcm
