#pragma once
#include "mgcm/backend.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mgcm {

// Integration Hook #2: Measures host IR/source -> SPIR-V frontend compilation time.
// Implement this interface around gc::gpu::GpuCompiler inside OpenVINO to calibrate alpha_0/1_gc.
class IFrontendCompilerProbe {
public:
    virtual ~IFrontendCompilerProbe() = default;
    virtual CompilerKind kind() const = 0;
    virtual bool         is_available() const = 0;
    virtual bool         measure_compile_ms(BenchKernelId id, uint32_t salt, double* out_ms,
                                            size_t* out_spv_words) = 0;
};

class NullFrontendProbe final : public IFrontendCompilerProbe {
public:
    explicit NullFrontendProbe(CompilerKind k = CompilerKind::GraphCompiler) : kind_(k) {}
    CompilerKind kind() const override { return kind_; }
    bool         is_available() const override { return false; }
    bool         measure_compile_ms(BenchKernelId, uint32_t, double*, size_t*) override { return false; }

private:
    CompilerKind kind_;
};

// Dynamically loads libslang.so on Linux hosts to benchmark real in-process Slang -> SPIR-V JIT.
// Gracefully returns an unavailable probe on Android or systems without libslang.so.
std::unique_ptr<IFrontendCompilerProbe> create_slang_frontend_probe(const std::string& slang_shader_dir = "");

} // namespace mgcm
