#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

M_GRID_DEFAULT = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
R_GRID = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000, 20000]


def write_ir_const_weights(xml_path: Path, m: int, n: int, k: int, act: str = "gelu_erf"):
    rng = np.random.default_rng(42)
    w_b = rng.uniform(-0.1, 0.1, size=(k, n)).astype(np.float32)
    w_bias = rng.uniform(-0.05, 0.05, size=(n,)).astype(np.float32)

    b_bytes = w_b.tobytes()
    bias_bytes = w_bias.tobytes()
    bin_path = xml_path.with_suffix(".bin")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(b_bytes + bias_bytes)

    if act == "relu":
        act_layer = f"""\t\t<layer id="5" name="relu" type="ReLU" version="opset1">
\t\t\t<input><port id="0" precision="FP32"><dim>{m}</dim><dim>{n}</dim></port></input>
\t\t\t<output><port id="1" precision="FP32" names="act_out"><dim>{m}</dim><dim>{n}</dim></port></output>
\t\t</layer>"""
    else:
        act_layer = f"""\t\t<layer id="5" name="gelu_erf" type="Gelu" version="opset7">
\t\t\t<data approximation_mode="erf" />
\t\t\t<input><port id="0" precision="FP32"><dim>{m}</dim><dim>{n}</dim></port></input>
\t\t\t<output><port id="1" precision="FP32" names="act_out"><dim>{m}</dim><dim>{n}</dim></port></output>
\t\t</layer>"""

    xml = f"""<?xml version="1.0"?>
<net name="mlp_const_{m}_{n}_{k}_{act}" version="11">
\t<layers>
\t\t<layer id="0" name="input_A" type="Parameter" version="opset1">
\t\t\t<data shape="{m},{k}" element_type="f32" />
\t\t\t<output><port id="0" precision="FP32" names="input_A"><dim>{m}</dim><dim>{k}</dim></port></output>
\t\t</layer>
\t\t<layer id="1" name="weights_B" type="Const" version="opset1">
\t\t\t<data element_type="f32" offset="0" size="{len(b_bytes)}" shape="{k},{n}" />
\t\t\t<output><port id="0" precision="FP32"><dim>{k}</dim><dim>{n}</dim></port></output>
\t\t</layer>
\t\t<layer id="2" name="matmul" type="MatMul" version="opset1">
\t\t\t<data transpose_a="false" transpose_b="false" />
\t\t\t<input>
\t\t\t\t<port id="0" precision="FP32"><dim>{m}</dim><dim>{k}</dim></port>
\t\t\t\t<port id="1" precision="FP32"><dim>{k}</dim><dim>{n}</dim></port>
\t\t\t</input>
\t\t\t<output><port id="2" precision="FP32" names="matmul_out"><dim>{m}</dim><dim>{n}</dim></port></output>
\t\t</layer>
\t\t<layer id="3" name="const_bias" type="Const" version="opset1">
\t\t\t<data element_type="f32" offset="{len(b_bytes)}" size="{len(bias_bytes)}" shape="{n}" />
\t\t\t<output><port id="0" precision="FP32"><dim>{n}</dim></port></output>
\t\t</layer>
\t\t<layer id="4" name="add_bias" type="Add" version="opset1">
\t\t\t<data auto_broadcast="numpy" />
\t\t\t<input>
\t\t\t\t<port id="0" precision="FP32"><dim>{m}</dim><dim>{n}</dim></port>
\t\t\t\t<port id="1" precision="FP32"><dim>{n}</dim></port>
\t\t\t</input>
\t\t\t<output><port id="2" precision="FP32" names="add_out"><dim>{m}</dim><dim>{n}</dim></port></output>
\t\t</layer>
{act_layer}
\t\t<layer id="6" name="output" type="Result" version="opset1">
\t\t\t<input><port id="0" precision="FP32"><dim>{m}</dim><dim>{n}</dim></port></input>
\t\t</layer>
\t</layers>
\t<edges>
\t\t<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
\t\t<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
\t\t<edge from-layer="2" from-port="2" to-layer="4" to-port="0" />
\t\t<edge from-layer="3" from-port="0" to-layer="4" to-port="1" />
\t\t<edge from-layer="4" from-port="2" to-layer="5" to-port="0" />
\t\t<edge from-layer="5" from-port="1" to-layer="6" to-port="0" />
\t</edges>
</net>
"""
    xml_path.write_text(xml, encoding="utf-8")


def parse_time_ms(text: str, regex: str) -> float:
    m = re.search(regex, text)
    if not m:
        return float("nan")
    val = float(m.group(1))
    unit = m.group(2).lower()
    return val * 1e-3 if unit == "us" else val


def run_benchmark(ov_bin: Path, xml_path: Path, device: str, pattern_str: str, niter: int):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{ov_bin.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    env["OV_GPU_ENABLE_MLIR"] = "1"
    env["OV_GPU_MLIR_PATTERNS"] = pattern_str
    env["MESA_SHADER_CACHE_DISABLE"] = "1"
    env["CUDA_CACHE_DISABLE"] = "1"
    env["__GL_ShaderDiskCache"] = "0"

    cmd = [
        str(ov_bin),
        "-m", str(xml_path),
        "-d", device,
        "-infer_precision", "f32",
        "-hint", "latency",
        "-niter", str(niter),
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
    out = res.stdout + "\n" + res.stderr
    if res.returncode != 0:
        raise RuntimeError(f"benchmark_app failed ({res.returncode}):\n{out[-800:]}")

    t_comp_model = parse_time_ms(out, r"Compile model took\s*:?\s*([0-9.]+)\s*(ms|us)")
    t_first = parse_time_ms(out, r"First inference took\s*:?\s*([0-9.]+)\s*(ms|us)")
    t_med = parse_time_ms(out, r"Median:\s*([0-9.]+)\s*(ms|us)")
    t_min = parse_time_ms(out, r"Min:\s*([0-9.]+)\s*(ms|us)")
    t_max = parse_time_ms(out, r"Max:\s*([0-9.]+)\s*(ms|us)")
    t_comp_total = t_comp_model + max(0.0, t_first - t_med)
    return t_comp_model, t_first, t_comp_total, t_med, t_min, t_max


def plot_oracle_results(csv_files, out_dir: Path, n: int, k: int, act: str):
    cmap = mcolors.ListedColormap(["#6baed6", "#fdae6b", "#fd8d3c", "#74c476"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    short = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]

    fig_p, axs_p = plt.subplots(1, len(csv_files), figsize=(6.2 * len(csv_files), 4.9), dpi=180, squeeze=False)
    fig_c, axs_c = plt.subplots(2, len(csv_files), figsize=(6.2 * len(csv_files), 7.2), dpi=180, squeeze=False)

    for idx, cf in enumerate(csv_files):
        tag = cf.stem.replace("ov_oracle_", "")
        with cf.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ms = sorted({int(r["M"]) for r in rows})
        by_m = {m: {} for m in ms}
        for r in rows:
            by_m[int(r["M"])][int(r["plan_id"])] = (
                float(r["t_compile_total_ms"]),
                float(r["t_exec_median_ms"]),
            )

        # 1. 2D Oracle Phase Map
        ax = axs_p[0, idx]
        grid = np.zeros((len(R_GRID), len(ms)), dtype=int)
        for mi, m in enumerate(ms):
            for ri, r_val in enumerate(R_GRID):
                grid[ri, mi] = min(by_m[m].keys(), key=lambda p: by_m[m][p][0] + r_val * by_m[m][p][1])
        ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        for ri in range(len(R_GRID)):
            for mi in range(len(ms)):
                ax.text(mi, ri, short[grid[ri, mi]], ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([str(m) for m in ms], rotation=45, fontsize=8.0)
        ax.set_yticks(range(len(R_GRID))); ax.set_yticklabels([str(r) for r in R_GRID], fontsize=8.0)
        ax.set_xlabel(f"Rows M (N={n}, K={k}, {act}, Const weights)")
        ax.set_ylabel("Model Runs R")
        ax.set_title(f"Measured Oracle: {tag}", fontsize=10.0)

        # 2. Execution & Compile curves across M for P0..P3
        ax_e, ax_k = axs_c[0, idx], axs_c[1, idx]
        for pid in (0, 1, 2, 3):
            ys_e = [by_m[m][pid][1] for m in ms]
            ys_c = [by_m[m][pid][0] for m in ms]
            ax_e.plot(ms, ys_e, marker="o", lw=2.0, color=colors[pid], label=short[pid])
            ax_k.plot(ms, ys_c, marker="s", lw=2.0, color=colors[pid], label=short[pid])
        ax_e.set_xscale("log", base=2); ax_e.set_yscale("log", base=10)
        ax_e.set_title(f"{tag}: Real T_exec Median (ms)")
        ax_e.set_ylabel("Median Latency (ms, log10)"); ax_e.grid(True, which="both", ls=":", alpha=0.6); ax_e.legend()
        ax_k.set_xscale("log", base=2)
        ax_k.set_title(f"{tag}: Real T_compile Total (ms)")
        ax_k.set_xlabel("Rows M"); ax_k.set_ylabel("Total Compile (ms)")
        ax_k.grid(True, which="both", ls=":", alpha=0.6); ax_k.legend()

    fig_p.suptitle("REAL OpenVINO benchmark_app Measured Oracle Phase Map (M, R)", fontsize=11.5)
    fig_p.tight_layout(); fig_p.savefig(out_dir / "oracle_phase_MR.png"); plt.close(fig_p)
    print(f"Saved -> {out_dir / 'oracle_phase_MR.png'}")

    fig_c.suptitle(f"REAL OpenVINO MLIR P0..P3 Curves (N={n}, K={k}, {act})", fontsize=11.5)
    fig_c.tight_layout(); fig_c.savefig(out_dir / "oracle_curves.png"); plt.close(fig_c)
    print(f"Saved -> {out_dir / 'oracle_curves.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ov-bin", default=str(Path.home() / "dev/ov_research/ov/dev/bin/intel64/Release/benchmark_app"))
    ap.add_argument("--device", default="GPU.0")
    ap.add_argument("--tag", default="radeon_780m")
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--act", choices=["gelu_erf", "relu"], default="gelu_erf")
    ap.add_argument("--niter", type=int, default=30)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"ov_oracle_{args.tag}.csv"

    if not args.plot_only:
        ov_bin = Path(args.ov_bin)
        model_dir = out_dir / "ov_models"
        act_type = "Gelu" if args.act == "gelu_erf" else "Relu"
        patterns = [
            (0, "P0_SplitAll", f"p_mm=MatMul;p_add=Add;p_act={act_type}"),
            (1, "P1_FusedBias", f"p_mma=MatMul,Add;p_act={act_type}"),
            (2, "P2_FusedEpilogue", f"p_mm=MatMul;p_ag=Add,{act_type}"),
            (3, "P3_FullFusion", f"p_all=MatMul,Add,{act_type}"),
        ]

        rows_out = []
        for m in M_GRID_DEFAULT:
            xml_p = model_dir / f"mlp_const_{m}_{args.N}_{args.K}_{args.act}.xml"
            write_ir_const_weights(xml_p, m, args.N, args.K, args.act)
            parts = [f"[M={m:4d}]"]
            for pid, pname, pstr in patterns:
                tc_m, t_fi, tc_tot, t_med, t_min, t_max = run_benchmark(ov_bin, xml_p, args.device, pstr, args.niter)
                rows_out.append({
                    "M": m, "N": args.N, "K": args.K, "plan_id": pid, "plan_name": pname,
                    "t_compile_model_ms": f"{tc_m:.3f}", "t_first_infer_ms": f"{t_fi:.3f}",
                    "t_compile_total_ms": f"{tc_tot:.3f}", "t_exec_median_ms": f"{t_med:.4f}",
                    "t_exec_min_ms": f"{t_min:.4f}", "t_exec_max_ms": f"{t_max:.4f}",
                })
                parts.append(f"P{pid}:c={tc_tot:.0f}ms/e={t_med:.3f}ms")
            print(" | ".join(parts), flush=True)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"Saved Oracle CSV -> {csv_path}")

    oracle_csvs = sorted(out_dir.glob("ov_oracle_*.csv"))
    if oracle_csvs:
        plot_oracle_results(oracle_csvs, out_dir, args.N, args.K, args.act)


if __name__ == "__main__":
    main()