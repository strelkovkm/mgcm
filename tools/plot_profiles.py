#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv_cols(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    bw_files = sorted(res_dir.glob("*_bw.csv"))
    wave_files = sorted(res_dir.glob("*_wave.csv"))

    if bw_files:
        plt.figure(figsize=(8.5, 5.0), dpi=160)
        for bf in bw_files:
            rows = read_csv_cols(bf)
            xs = [float(r["working_set_mb"]) for r in rows]
            ys = [float(r["bw_median_gbps"]) for r in rows]
            label = bf.stem.replace("_bw", "")
            plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, label=label)
        plt.xscale("log", base=2)
        plt.xlabel("Working Set Size W (MB, log2 scale)")
        plt.ylabel("Effective Read Bandwidth (GB/s)")
        plt.title("MB2: GPU Cache Hierarchy & Memory Bandwidth Curve")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out_p = res_dir / "mb2_cache_hierarchy.png"
        plt.savefig(out_p)
        print(f"Saved plot -> {out_p}")

    if wave_files:
        plt.figure(figsize=(8.5, 5.0), dpi=160)
        for wf in wave_files:
            rows = read_csv_cols(wf)
            xs = [int(r["workgroups"]) for r in rows]
            ys = [float(r["gflops"]) / 1000.0 for r in rows]
            label = wf.stem.replace("_wave", "")
            plt.plot(xs, ys, marker="s", linewidth=2.0, markersize=4.5, label=label)
        plt.xscale("log", base=2)
        plt.xlabel("Launched Workgroups G (256 threads/WG, log2 scale)")
        plt.ylabel("Sustained FP32 Throughput (TFLOPS)")
        plt.title("MB5: GPU Compute Wave Saturation Curve")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out_p = res_dir / "mb5_wave_saturation.png"
        plt.savefig(out_p)
        print(f"Saved plot -> {out_p}")


if __name__ == "__main__":
    main()
