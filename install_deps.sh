#!/usr/bin/env bash
set -euo pipefail

SLANG_VERSION="${SLANG_VERSION:-2026.20}"
SLANG_INSTALL_DIR="${SLANG_INSTALL_DIR:-$HOME/.local/opt/slang}"

echo "[mgcm] Checking system packages..."
if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        SUDO="sudo"
    fi
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends \
        ca-certificates curl tar xz-utils git python3 \
        cmake ninja-build clang clang-format clang-tidy lld mold ccache \
        libvulkan-dev vulkan-tools spirv-tools nlohmann-json3-dev
else
    echo "[mgcm] Non-apt distribution detected; ensure cmake, ninja, clang++, libvulkan, spirv-tools are installed."
fi

if command -v slangc >/dev/null 2>&1; then
    echo "[mgcm] Found slangc in PATH: $(command -v slangc) ($(slangc -v 2>&1 | head -n 1))"
elif [ -x "$HOME/dev/ov_research/tools/slang/bin/slangc" ]; then
    echo "[mgcm] Found slangc in workspace: $HOME/dev/ov_research/tools/slang/bin/slangc"
elif [ -x "$SLANG_INSTALL_DIR/bin/slangc" ]; then
    echo "[mgcm] Found slangc in $SLANG_INSTALL_DIR/bin/slangc"
else
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)        SLANG_ARCH="x86_64" ;;
        aarch64|arm64) SLANG_ARCH="aarch64" ;;
        *) echo "[mgcm] Unsupported CPU architecture for prebuilt slangc: $ARCH" >&2; exit 1 ;;
    esac

    URL="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-${SLANG_ARCH}.tar.gz"
    echo "[mgcm] Downloading slangc v${SLANG_VERSION} (${SLANG_ARCH}) -> ${SLANG_INSTALL_DIR} ..."
    mkdir -p "$SLANG_INSTALL_DIR"
    curl -fL "$URL" | tar -xz -C "$SLANG_INSTALL_DIR"
    echo "[mgcm] Installed slangc: $("$SLANG_INSTALL_DIR/bin/slangc" -v 2>&1 | head -n 1)"
fi

echo "[mgcm] All dependencies are ready."
