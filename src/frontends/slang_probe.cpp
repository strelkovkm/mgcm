#include "mgcm/frontend_probe.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#ifdef MGCM_HAS_SLANG_API
#include <slang.h>
#endif

namespace mgcm {
namespace {

#ifdef MGCM_HAS_SLANG_API
bool map_kernel_to_slang(BenchKernelId id, const char** out_file, const char** out_entry) {
    switch (id) {
        case BenchKernelId::MB1_DispatchNoop:
            *out_file  = "mb1_dispatch.slang";
            *out_entry = "mb1_noop";
            return true;
        case BenchKernelId::MB4_MatmulDirectCache:
            *out_file  = "mb4_tiled_matmul.slang";
            *out_entry = "mb4_matmul_cache";
            return true;
        case BenchKernelId::MB4_MatmulSmem:
            *out_file  = "mb4_tiled_matmul.slang";
            *out_entry = "mb4_matmul_smem";
            return true;
        case BenchKernelId::MB5_GeluTanh:
            *out_file  = "mb5_compute.slang";
            *out_entry = "mb5_gelu_tanh";
            return true;
        case BenchKernelId::MB5_GeluErf:
            *out_file  = "mb5_compute.slang";
            *out_entry = "mb5_gelu_erf";
            return true;
        case BenchKernelId::MB6_MatmulFusedGeluErf:
            *out_file = "mb4_tiled_matmul.slang"; *out_entry = "mb6_matmul_fused_gelu_erf"; return true;
        case BenchKernelId::Ora_MatmulOnly:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_matmul_only"; return true;
        case BenchKernelId::Ora_MatmulBias:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_matmul_bias"; return true;
        case BenchKernelId::Ora_MatmulBiasGelu:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_matmul_bias_gelu"; return true;
        case BenchKernelId::Ora_MatmulBiasRelu:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_matmul_bias_relu"; return true;
        case BenchKernelId::Ora_ElemAdd:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_elem_add"; return true;
        case BenchKernelId::Ora_ElemGelu:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_elem_gelu"; return true;
        case BenchKernelId::Ora_ElemRelu:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_elem_relu"; return true;
        case BenchKernelId::Ora_ElemAddGelu:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_elem_add_gelu"; return true;
        case BenchKernelId::Ora_ElemAddRelu:
            *out_file = "oracle_mlp.slang"; *out_entry = "ora_elem_add_relu"; return true;
        default:
            return false;
    }
}

class SlangFrontendProbe final : public IFrontendCompilerProbe {
public:
    explicit SlangFrontendProbe(std::string shader_dir) : shader_dir_(std::move(shader_dir)) {
        if (shader_dir_.empty()) {
#ifdef MGCM_SLANG_SHADER_DIR
            shader_dir_ = MGCM_SLANG_SHADER_DIR;
#endif
        }
        if (!shader_dir_.empty() && std::filesystem::exists(shader_dir_)) {
            if (SLANG_SUCCEEDED(slang::createGlobalSession(&global_session_)) && global_session_ != nullptr) {
                available_ = true;
            }
        }
    }

    ~SlangFrontendProbe() override {
        if (global_session_) {
            global_session_->release();
        }
    }

    CompilerKind kind() const override { return CompilerKind::Slang; }
    bool         is_available() const override { return available_; }

    bool measure_compile_ms(BenchKernelId id, uint32_t salt, double* out_ms, size_t* out_spv_words) override {
        if (!available_) {
            return false;
        }
        const char* rel_file   = nullptr;
        const char* entry_name = nullptr;
        if (!map_kernel_to_slang(id, &rel_file, &entry_name)) {
            return false;
        }

        auto full_path = std::filesystem::path(shader_dir_) / rel_file;
        std::ifstream ifs(full_path);
        if (!ifs) {
            return false;
        }
        std::ostringstream ss;
        ss << "// mgcm_salt_" << salt << "\n" << ifs.rdbuf();
        std::string source_code = ss.str();

        slang::TargetDesc target{};
        target.format  = SLANG_SPIRV;
        target.profile = global_session_->findProfile("spirv_1_6");
        target.flags   = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;

        slang::CompilerOptionEntry opt{};
        opt.name            = slang::CompilerOptionName::VulkanUseEntryPointName;
        opt.value.kind      = slang::CompilerOptionValueKind::Int;
        opt.value.intValue0 = 1;

        slang::SessionDesc sdesc{};
        sdesc.targets                  = &target;
        sdesc.targetCount              = 1;
        sdesc.compilerOptionEntries    = &opt;
        sdesc.compilerOptionEntryCount = 1;

        std::string mod_name = std::string("mgcm_mod_") + std::to_string(salt);
        auto t0              = std::chrono::steady_clock::now();

        slang::ISession* session = nullptr;
        if (SLANG_FAILED(global_session_->createSession(sdesc, &session)) || !session) {
            return false;
        }

        slang::IBlob* diag = nullptr;
        slang::IModule* mod =
            session->loadModuleFromSourceString(mod_name.c_str(), rel_file, source_code.c_str(), &diag);
        if (diag) {
            diag->release();
            diag = nullptr;
        }
        if (!mod) {
            session->release();
            return false;
        }

        slang::IEntryPoint* ep = nullptr;
        if (SLANG_FAILED(mod->findEntryPointByName(entry_name, &ep)) || !ep) {
            session->release();
            return false;
        }

        slang::IComponentType* comps[2] = {mod, ep};
        slang::IComponentType* program  = nullptr;
        session->createCompositeComponentType(comps, 2, &program, &diag);
        if (diag) {
            diag->release();
            diag = nullptr;
        }

        slang::IComponentType* linked = nullptr;
        if (program) {
            program->link(&linked, &diag);
            if (diag) {
                diag->release();
                diag = nullptr;
            }
        }

        slang::IBlob* spv_blob = nullptr;
        bool ok                = false;
        if (linked && SLANG_SUCCEEDED(linked->getEntryPointCode(0, 0, &spv_blob, &diag)) && spv_blob) {
            if (out_spv_words) {
                *out_spv_words = spv_blob->getBufferSize() / sizeof(uint32_t);
            }
            ok = true;
        }
        auto t1 = std::chrono::steady_clock::now();

        if (diag) diag->release();
        if (spv_blob) spv_blob->release();
        if (linked) linked->release();
        if (program) program->release();
        if (ep) ep->release();
        session->release();

        if (ok && out_ms) {
            *out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        return ok;
    }

private:
    std::string            shader_dir_;
    slang::IGlobalSession* global_session_ = nullptr;
    bool                   available_      = false;
};
#endif

} // namespace

std::unique_ptr<IFrontendCompilerProbe> create_slang_frontend_probe(const std::string& slang_shader_dir) {
#ifdef MGCM_HAS_SLANG_API
    return std::make_unique<SlangFrontendProbe>(slang_shader_dir);
#else
    (void)slang_shader_dir;
    return std::make_unique<NullFrontendProbe>(CompilerKind::Slang);
#endif
}

} // namespace mgcm
