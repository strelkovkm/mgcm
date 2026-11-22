#!/usr/bin/env bash
set -euo pipefail

NDK_VERSION="r28c"
TOOLS_DIR="${TOOLS_DIR:-$HOME/dev/ov_research/tools}"
TARGET_NDK_DIR="${TOOLS_DIR}/android-ndk-${NDK_VERSION}"

if command -v apt-get >/dev/null 2>&1; then
    MISSING_PKGS=()
    command -v adb   >/dev/null 2>&1 || MISSING_PKGS+=(adb)
    command -v unzip >/dev/null 2>&1 || MISSING_PKGS+=(unzip)
    if [ "${#MISSING_PKGS[@]}" -gt 0 ]; then
        echo "[android-setup] Installing ${MISSING_PKGS[*]} ..."
        SUDO=""
        [ "$(id -u)" -ne 0 ] && SUDO="sudo"
        $SUDO apt-get update -qq
        $SUDO apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}"
    fi
fi

# Search existing NDK installations first
for CAND in \
    "${ANDROID_NDK_HOME:-}" \
    "${ANDROID_NDK_ROOT:-}" \
    "$TARGET_NDK_DIR" \
    "$HOME/Android/Sdk/ndk"/* \
    /opt/android-ndk*; do
    if [ -n "$CAND" ] && [ -f "$CAND/build/cmake/android.toolchain.cmake" ]; then
        echo "[android-setup] Found Android NDK: $CAND"
        exit 0
    fi
done

mkdir -p "$TOOLS_DIR"
ZIP_PATH="${TOOLS_DIR}/android-ndk-${NDK_VERSION}-linux.zip"
URL="https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip"

echo "[android-setup] Downloading Android NDK ${NDK_VERSION} (~711 MB) -> ${ZIP_PATH} ..."
curl -fL --progress-bar "$URL" -o "$ZIP_PATH"

echo "[android-setup] Extracting NDK into ${TOOLS_DIR} ..."
unzip -q "$ZIP_PATH" -d "$TOOLS_DIR"
rm -f "$ZIP_PATH"

echo "[android-setup] Ready: ${TARGET_NDK_DIR}"
