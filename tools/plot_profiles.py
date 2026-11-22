#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
BW_TICKS = [0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0, 256.0]
BW_LABELS = ["32 KB", "128 KB", "512 KB", "2 MB", "8 MB", "32 MB", "128 MB", "256 MB"]
WAVE_TICKS = [1, 2, 4, 6, 8, 12, 24, 36, 72, 144, 256, 512]


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_matching_json(res_dir: Path, prefix: str):
    jp = res_dir / f"{prefix}_full.json"
    return json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else None


def plot_bw(res_dir: Path, bw_files, log_y: bool):
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=180)
    for idx, bf in enumerate(bw_files):
        prefix = bf.stem.replace("_bw", "")
        meta = load_matching_json(res_dir, prefix)
        label = meta["device"]["name"] if meta else prefix
        c = COLORS[idx % len(COLORS)]
        rows = read_csv_rows(bf)
        xs = [float(r["working_set_mb"]) for r in rows]
        ys = [float(r["bw_median_gbps"]) for r in rows]
        iqrs = [float(r["bw_iqr_gbps"]) for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=2.1, markersize=4.5, color=c, label=label)
        if not log_y:
            lo = [max(0.0, y - 0.5 * q) for y, q in zip(ys, iqrs)]
            hi = [y + 0.5 * q for y, q in zip(ys, iqrs)]
            ax.fill_between(xs, lo, hi, color=c, alpha=0.15)
        if meta and "mb2_memory" in meta:
            l2_mb = float(meta["mb2_memory"]["l2_effective_mb"])
            if l2_mb > 0:
                ax.axvline(l2_mb, color=c, linestyle="--", linewidth=1.3, alpha=0.8, label=f"{prefix}: L2 = {l2_mb:g} MB")

    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log", base=10)
    ax.set_xticks(BW_TICKS)
    ax.set_xticklabels(BW_LABELS)
    ax.set_xlabel("Working Set Size W")
    ax.set_ylabel("Effective Read Bandwidth (GB/s" + (", log10)" if log_y else ")"))
    ax.set_title("MB2: Cross-Vendor Cache Hierarchy" + (" (Log-Log Scale)" if log_y else ""))
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5, loc="lower left" if log_y else "upper right")
    fig.tight_layout()
    out_p = res_dir / ("mb2_cache_hierarchy_log.png" if log_y else "mb2_cache_hierarchy.png")
    fig.savefig(out_p)
    plt.close(fig)
    print(f"Saved plot -> {out_p}")


def plot_waves(res_dir: Path, wave_files):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.0, 6.8), dpi=180, sharex=True)
    fig_n, ax_n = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    max_t = 10.0

    for idx, wf in enumerate(wave_files):
        prefix = wf.stem.replace("_wave", "")
        meta = load_matching_json(res_dir, prefix)
        label = meta["device"]["name"] if meta else prefix
        c = COLORS[idx % len(COLORS)]
        rows = read_csv_rows(wf)
        xs = [int(r["workgroups"]) for r in rows]
        gfs = [float(r["gflops"]) for r in rows]
        ts = [float(r["gpu_time_us"]) for r in rows]
        peak = max(gfs) if gfs else 1.0
        for x, t in zip(xs, ts):
            if x <= 36:
                max_t = max(max_t, t)

        ax_top.plot(xs, [g / 1000.0 for g in gfs], marker="s", linewidth=2.0, markersize=4.2, color=c, label=label)
        ax_bot.plot(xs, ts, marker="o", linewidth=1.9, markersize=4.0, color=c, label=label)
        ax_n.plot(xs, [100.0 * g / peak for g in gfs], marker="s", linewidth=2.0, markersize=4.2, color=c, label=label)

        if meta and "mb5_compute" in meta:
            gw = int(meta["mb5_compute"].get("g_wave_quantum", 0))
            if gw > 1:
                ax_n.axvline(gw, color=c, linestyle="--", linewidth=1.3, alpha=0.75, label=f"{prefix}: G_wave = {gw}")
                for m in (1, 2, 3):
                    if gw * m <= max(xs):
                        lbl = f"{prefix}: G_wave = {gw}" if m == 1 else None
                        ax_top.axvline(gw * m, color=c, linestyle="--", linewidth=1.1, alpha=0.55, label=lbl)
                        ax_bot.axvline(gw * m, color=c, linestyle="--", linewidth=1.1, alpha=0.55)

    ax_top.set_xscale("log", base=2)
    ax_top.set_ylabel("Sustained FP32 (TFLOPS)")
    ax_top.set_title("MB5: Wave Saturation & Hardware Wave Quantization Staircase")
    ax_top.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_top.legend(fontsize=8.2, loc="upper left")

    ax_bot.set_xticks(WAVE_TICKS)
    ax_bot.set_xticklabels([str(v) for v in WAVE_TICKS])
    ax_bot.set_ylim(0.0, max_t * 1.18)
    ax_bot.set_xlabel("Launched Workgroups G (256 threads/WG)")
    ax_bot.set_ylabel("Kernel Time (us, G <= 36 zoom)")
    ax_bot.grid(True, which="both", linestyle=":", alpha=0.6)
    fig.tight_layout()
    p1 = res_dir / "mb5_wave_saturation.png"
    fig.savefig(p1)
    plt.close(fig)
    print(f"Saved plot -> {p1}")

    ax_n.set_xscale("log", base=2)
    ax_n.set_xticks(WAVE_TICKS)
    ax_n.set_xticklabels([str(v) for v in WAVE_TICKS])
    ax_n.set_ylim(0.0, 108.0)
    ax_n.set_xlabel("Launched Workgroups G (256 threads/WG)")
    ax_n.set_ylabel("Normalized Compute Utilization (% of Peak)")
    ax_n.set_title("MB5: Cross-Vendor Wave Quantization & Saturation (% of Device Peak)")
    ax_n.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_n.legend(fontsize=8.5, loc="lower right")
    fig_n.tight_layout()
    p2 = res_dir / "mb5_wave_efficiency_norm.png"
    fig_n.savefig(p2)
    plt.close(fig_n)
    print(f"Saved plot -> {p2}")


def main():
    res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    bw_files = sorted(res_dir.glob("*_bw.csv"))
    wave_files = sorted(res_dir.glob("*_wave.csv"))
    if bw_files:
        plot_bw(res_dir, bw_files, log_y=False)
        plot_bw(res_dir, bw_files, log_y=True)
    if wave_files:
        plot_waves(res_dir, wave_files)


if __name__ == "__main__":
    main()
