#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build_android"

NDK_DIR=""
for CAND in \
    "${ANDROID_NDK_HOME:-}" \
    "${ANDROID_NDK_ROOT:-}" \
    "$HOME/dev/ov_research/tools/android-ndk-r28c" \
    "$HOME/Android/Sdk/ndk"/* \
    /opt/android-ndk*; do
    if [ -n "$CAND" ] && [ -f "$CAND/build/cmake/android.toolchain.cmake" ]; then
        NDK_DIR="$CAND"
        break
    fi
done

if [ -z "$NDK_DIR" ]; then
    echo "[build-android] Android NDK not found. Running scripts/setup_android_ndk.sh ..."
    "${REPO_ROOT}/scripts/setup_android_ndk.sh"
    NDK_DIR="$HOME/dev/ov_research/tools/android-ndk-r28c"
fi

echo "[build-android] Using NDK: ${NDK_DIR}"
cmake -G Ninja \
    -S "${REPO_ROOT}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${NDK_DIR}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DMGCM_COMPILE_SLANG=ON

cmake --build "${BUILD_DIR}" --target mgcm_cli -j "$(nproc)"
file "${BUILD_DIR}/mgcm_cli"

if [ "${1:-}" = "--run" ]; then
    echo "[build-android] Pushing mgcm_cli to connected Android device via ADB..."
    adb wait-for-device
    adb push "${BUILD_DIR}/mgcm_cli" /data/local/tmp/mgcm_cli
    adb shell "chmod +x /data/local/tmp/mgcm_cli && /data/local/tmp/mgcm_cli --list"
    adb shell "/data/local/tmp/mgcm_cli --device 0 --out /data/local/tmp/android_gpu_full.json --csv /data/local/tmp/android_gpu_bw.csv --wave-csv /data/local/tmp/android_gpu_wave.csv"
    mkdir -p "${REPO_ROOT}/results"
    adb pull /data/local/tmp/android_gpu_full.json "${REPO_ROOT}/results/"
    adb pull /data/local/tmp/android_gpu_bw.csv    "${REPO_ROOT}/results/"
    adb pull /data/local/tmp/android_gpu_wave.csv  "${REPO_ROOT}/results/"
    echo "[build-android] Pulled Android GPU profile into ${REPO_ROOT}/results/"
fi
