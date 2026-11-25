cmake_minimum_required(VERSION 3.22)
project(mgcm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MGCM_ENABLE_VULKAN "Build Vulkan 1.3 calibration backend" ON)
option(MGCM_COMPILE_SLANG "Compile .slang shaders to embedded SPIR-V headers" ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(MGCM_COMPILE_SLANG)
    find_program(SLANGC_EXECUTABLE NAMES slangc
        HINTS "$ENV{HOME}/dev/ov_research/tools/slang/bin" "$ENV{HOME}/.local/opt/slang/bin" NO_CMAKE_FIND_ROOT_PATH)
    if(NOT SLANGC_EXECUTABLE)
        message(FATAL_ERROR "slangc not found! Set -DSLANGC_EXECUTABLE=/path/to/slangc or -DMGCM_COMPILE_SLANG=OFF")
    endif()
    find_program(SPIRV_VAL_EXECUTABLE NAMES spirv-val NO_CMAKE_FIND_ROOT_PATH)
    message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")
    message(STATUS "Found spirv-val: ${SPIRV_VAL_EXECUTABLE}")
endif()

if(NOT CMAKE_CROSSCOMPILING AND EXISTS "/usr/include/nlohmann/json.hpp")
    add_library(mgcm_json INTERFACE)
else()
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(nlohmann_json)
    add_library(mgcm_json INTERFACE)
    target_link_libraries(mgcm_json INTERFACE nlohmann_json::nlohmann_json)
endif()

include(cmake/CompileSlang.cmake)
set(MGCM_GENERATED_SPV_HEADERS "")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/common_init.slang"  "init_prng_vec4"  "init_prng_vec4")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb0_warmup.slang"   "mb0_warmup"      "mb0_warmup")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb1_dispatch.slang" "mb1_noop"        "mb1_noop")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb2_cache_bw.slang"     "mb2_stream_vec4"          "mb2_stream_vec4")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb3_stride.slang"       "mb3_stride_f32"           "mb3_stride_f32")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb4_tiled_matmul.slang" "mb4_matmul_smem"          "mb4_matmul_smem")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb4_tiled_matmul.slang" "mb4_matmul_smem_tail"     "mb4_matmul_smem_tail")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb4_tiled_matmul.slang" "mb4_matmul_cache"         "mb4_matmul_cache")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb4_tiled_matmul.slang" "mb6_matmul_fused_gelu_erf" "mb6_matmul_fused_gelu_erf")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb5_compute.slang"      "mb5_fma16"                "mb5_fma16")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb5_compute.slang"      "mb5_gelu_tanh"            "mb5_gelu_tanh")
mgcm_add_slang_shader(MGCM_GENERATED_SPV_HEADERS "shaders/slang/mb5_compute.slang"      "mb5_gelu_erf"             "mb5_gelu_erf")

add_custom_target(mgcm_shaders DEPENDS ${MGCM_GENERATED_SPV_HEADERS})

set(MGCM_SOURCES
    src/calibrator.cpp
    src/profile_cache.cpp
    src/blocks/mlp_block.cpp
    src/cost_model.cpp
    src/frontends/slang_probe.cpp
)
if(MGCM_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
    list(APPEND MGCM_SOURCES src/backends/vulkan/vulkan_backend.cpp)
endif()

add_library(mgcm_core STATIC ${MGCM_SOURCES} ${MGCM_GENERATED_SPV_HEADERS})
add_dependencies(mgcm_core mgcm_shaders)
target_include_directories(mgcm_core
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated_spirv"
)
target_link_libraries(mgcm_core PUBLIC mgcm_json)
if(MGCM_ENABLE_VULKAN)
    target_compile_definitions(mgcm_core PUBLIC MGCM_HAS_VULKAN=1)
    target_link_libraries(mgcm_core PUBLIC Vulkan::Vulkan)
endif()

add_executable(mgcm_cli src/mgcm_cli.cpp)
target_link_libraries(mgcm_cli PRIVATE mgcm_core)

if(NOT CMAKE_CROSSCOMPILING)
    find_path(SLANG_INCLUDE_DIR NAMES slang.h
        HINTS "$ENV{HOME}/dev/ov_research/tools/slang/include" "$ENV{HOME}/.local/opt/slang/include")
    find_library(SLANG_LIBRARY NAMES slang
        HINTS "$ENV{HOME}/dev/ov_research/tools/slang/lib" "$ENV{HOME}/.local/opt/slang/lib")
    if(SLANG_INCLUDE_DIR AND SLANG_LIBRARY)
        message(STATUS "Found libslang in-process API: ${SLANG_LIBRARY}")
        get_filename_component(SLANG_LIB_DIR "${SLANG_LIBRARY}" DIRECTORY)
        target_compile_definitions(mgcm_core PRIVATE
            MGCM_HAS_SLANG_API=1
            MGCM_SLANG_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/slang")
        target_include_directories(mgcm_core PRIVATE "${SLANG_INCLUDE_DIR}")
        target_link_libraries(mgcm_core PRIVATE "${SLANG_LIBRARY}")
        set_target_properties(mgcm_cli PROPERTIES BUILD_RPATH "${SLANG_LIB_DIR}")
    endif()
endif()


enable_testing()
add_executable(test_cost_model tests/test_cost_model.cpp)
target_link_libraries(test_cost_model PRIVATE mgcm_core)
add_test(NAME CostModelUnitTests COMMAND test_cost_model)
