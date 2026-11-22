#!/usr/bin/env python3
import sys
from pathlib import Path

spv_path, hdr_path, var_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = spv_path.read_bytes()
assert len(data) % 4 == 0, f"Invalid SPIR-V size: {len(data)}"
words = [int.from_bytes(data[i:i+4], "little") for i in range(0, len(data), 4)]

lines = [
    "#pragma once",
    "#include <cstdint>",
    "#include <span>",
    "namespace mgcm::spirv {",
    f"inline constexpr uint32_t k_{var_name}_words[] = {{"
]
for i in range(0, len(words), 8):
    lines.append("    " + ", ".join(f"0x{w:08x}u" for w in words[i:i+8]) + ",")
lines += [
    "};",
    f"inline constexpr std::span<const uint32_t> k_{var_name}{{k_{var_name}_words, {len(words)}}};",
    "} // namespace mgcm::spirv", ""
]
hdr_path.parent.mkdir(parents=True, exist_ok=True)
hdr_path.write_text("\n".join(lines), encoding="utf-8")
