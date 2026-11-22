# mgcm — Microbenchmark-Guided GPU Cost Model

`mgcm` is a lightweight, runtime-extensible GPU microbenchmarking and analytical cost-modeling library designed for subgraph boundary selection (`MatMul -> Add(bias) -> GELU`) in OpenVINO.

## Architecture

1. **`mgcm::CostModel` & `mgcm::DeviceProfile` (`include/mgcm/cost_model.hpp`):**
   Pure C++20 gray-box Roofline cost model $C(P, R) = T_{\text{compile}}(P) + R \cdot T_{\text{execute}}(P)$ with zero GPU API headers. Supports partitions `P0..P3`, break-even $R^*$ computation, and ablation modes (`None`, `NoMemoryTraffic`, `NoDispatchOverhead`, `NoCompileCost`).
2. **`mgcm::Calibrator` & `mgcm::ICalibBackend` (`include/mgcm/backend.hpp`):**
   Runtime-agnostic calibration engine (`MB1`–`MB6`) extensible across `Vulkan`, `OpenCL`, and `Level Zero`.
3. **`mgcm::VulkanBackend` + Slang Shaders (`shaders/slang/*.slang`):**
   Vulkan 1.3 implementation strictly adhering to OpenVINO's Canonical Compute ABI (`DescriptorSet = 0`, contiguous `StorageBuffer` `binding = 0..N-1`, 32-bit `PushConstants <= 128B`, single `OpEntryPoint`, `spirv-val` verified at build time).

## Quick Start (Linux x86_64 / ARM64)

```bash
./install_deps.sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j $(nproc)
ctest --test-dir build --output-on-failure

./build/mgcm_cli --list
./build/mgcm_cli --device 0 --out results/gpu0.json --csv results/gpu0_bw.csv --wave-csv results/gpu0_wave.csv
./build/mgcm_cli --eval results/gpu0.json
```

## Cross-Compiling for Android ARM64 (ARM Mali-G68 via NDK r28c)

Pre-generated SPIR-V headers in `shaders/generated_spirv/` allow building for Android with `-DMGCM_COMPILE_SLANG=OFF` (no `slangc` needed in the NDK sysroot):

```bash
cmake -G Ninja -S . -B build_android \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-33 \
    -DCMAKE_BUILD_TYPE=Release \
    -DMGCM_COMPILE_SLANG=OFF
cmake --build build_android -j $(nproc)

adb push build_android/mgcm_cli /data/local/tmp/
adb shell "/data/local/tmp/mgcm_cli --device 0 --out /data/local/tmp/mali_g68.json --csv /data/local/tmp/mali_g68_bw.csv"
adb pull /data/local/tmp/mali_g68.json results/
```

