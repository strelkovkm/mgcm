#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_matching_json(res_dir: Path, prefix: str):
    jp = res_dir / f"{prefix}_full.json"
    if jp.exists():
        return json.loads(jp.read_text(encoding="utf-8"))
    return None


def main():
    res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    bw_files = sorted(res_dir.glob("*_bw.csv"))
    wave_files = sorted(res_dir.glob("*_wave.csv"))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if bw_files:
        fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=180)
        for idx, bf in enumerate(bw_files):
            prefix = bf.stem.replace("_bw", "")
            meta = load_matching_json(res_dir, prefix)
            dev_label = meta["device"]["name"] if meta else prefix
            color = colors[idx % len(colors)]

            rows = read_csv_rows(bf)
            xs = [float(r["working_set_mb"]) for r in rows]
            ys = [float(r["bw_median_gbps"]) for r in rows]
            iqrs = [float(r["bw_iqr_gbps"]) for r in rows]
            lo = [max(0.0, y - 0.5 * q) for y, q in zip(ys, iqrs)]
            hi = [y + 0.5 * q for y, q in zip(ys, iqrs)]

            ax.plot(xs, ys, marker="o", linewidth=2.1, markersize=4.5, color=color, label=dev_label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15)

            if meta and "mb2_memory" in meta:
                l2_mb = float(meta["mb2_memory"]["l2_effective_mb"])
                if l2_mb > 0:
                    ax.axvline(
                        l2_mb,
                        color=color,
                        linestyle="--",
                        linewidth=1.3,
                        alpha=0.8,
                        label=f"{prefix}: detected L2 = {l2_mb:g} MB",
                    )

        ax.set_xscale("log", base=2)
        tick_vals = [0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0, 256.0]
        tick_lbls = ["32 KB", "128 KB", "512 KB", "2 MB", "8 MB", "32 MB", "128 MB", "256 MB"]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_lbls)
        ax.set_xlabel("Working Set Size W")
        ax.set_ylabel("Effective Read Bandwidth (GB/s)")
        ax.set_title("MB2: GPU Cache Hierarchy & Memory Bandwidth Curve")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(fontsize=8.5, loc="upper right")
        fig.tight_layout()
        out_p = res_dir / "mb2_cache_hierarchy.png"
        fig.savefig(out_p)
        plt.close(fig)
        print(f"Saved plot -> {out_p}")

    if wave_files:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.0, 6.8), dpi=180, sharex=True)
        max_time_144 = 10.0

        for idx, wf in enumerate(wave_files):
            prefix = wf.stem.replace("_wave", "")
            meta = load_matching_json(res_dir, prefix)
            dev_label = meta["device"]["name"] if meta else prefix
            color = colors[idx % len(colors)]

            rows = read_csv_rows(wf)
            xs = [int(r["workgroups"]) for r in rows]
            tflops = [float(r["gflops"]) / 1000.0 for r in rows]
            times_us = [float(r["gpu_time_us"]) for r in rows]

            for x, t in zip(xs, times_us):
                if x <= 144:
                    max_time_144 = max(max_time_144, t)

            ax_top.plot(xs, tflops, marker="s", linewidth=2.0, markersize=4.2, color=color, label=dev_label)
            ax_bot.plot(xs, times_us, marker="o", linewidth=1.9, markersize=4.0, color=color, label=dev_label)

            if meta and "mb5_compute" in meta:
                gw = int(meta["mb5_compute"].get("g_wave_quantum", 0))
                if gw > 1:
                    for mult in (1, 2, 3):
                        if gw * mult <= max(xs):
                            lbl = f"{prefix}: G_wave = {gw} WGs" if mult == 1 else None
                            ax_top.axvline(gw * mult, color=color, linestyle="--", linewidth=1.1, alpha=0.55, label=lbl)
                            ax_bot.axvline(gw * mult, color=color, linestyle="--", linewidth=1.1, alpha=0.55)

        ax_top.set_xscale("log", base=2)
        ax_top.set_ylabel("Sustained FP32 (TFLOPS)")
        ax_top.set_title("MB5: Wave Saturation & Hardware Wave Quantization Staircase")
        ax_top.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_top.legend(fontsize=8.5, loc="upper left")

        w_ticks = [1, 2, 4, 6, 12, 24, 36, 72, 144, 256, 512]
        ax_bot.set_xticks(w_ticks)
        ax_bot.set_xticklabels([str(v) for v in w_ticks])
        ax_bot.set_ylim(0.0, max_time_144 * 1.18)
        ax_bot.set_xlabel("Launched Workgroups G (256 threads/WG)")
        ax_bot.set_ylabel("Kernel Time (us, G <= 144 zoom)")
        ax_bot.grid(True, which="both", linestyle=":", alpha=0.6)

        fig.tight_layout()
        out_p = res_dir / "mb5_wave_saturation.png"
        fig.savefig(out_p)
        plt.close(fig)
        print(f"Saved plot -> {out_p}")


if __name__ == "__main__":
    main()
