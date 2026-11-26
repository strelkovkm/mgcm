#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
BW_TICKS = [0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0, 256.0]
BW_LABELS = ["32 KB", "128 KB", "512 KB", "2 MB", "8 MB", "32 MB", "128 MB", "256 MB"]
WAVE_TICKS = [1, 2, 4, 6, 8, 12, 24, 36, 72, 144, 256, 512]


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def plot_mb1_ols(res_dir: Path, json_files):
    devs = [(jf.stem.replace("_full", ""), load_json(jf)) for jf in json_files]
    devs = [(p, m) for p, m in devs if m and "mb1_dispatch" in m]
    if not devs:
        return
    fig, axs = plt.subplots(1, len(devs), figsize=(5.6 * len(devs), 4.6), dpi=180, squeeze=False)
    for idx, (prefix, meta) in enumerate(devs):
        ax = axs[0, idx]
        m1 = meta["mb1_dispatch"]
        t_l = float(m1["t_launch_gpu_us"])
        t_b = float(m1["t_barrier_gpu_us"])
        r2 = float(m1.get("r2_dispatch", 1.0))

        pts = m1.get("points", [])
        if pts:
            xs = np.array([p[0] for p in pts], float)
            yn = np.array([p[1] for p in pts], float)
            yb = np.array([p[2] for p in pts], float)
            # Recover OLS intercept from centroid
            b0_n = float(yn.mean() - t_l * xs.mean())
            b0_b = float(yb.mean() - (t_l + t_b) * xs.mean())
            ax.scatter(xs, yn, color="#1f77b4", s=46, zorder=4)
            ax.scatter(xs, yb, color="#d62728", marker="^", s=50, zorder=4)
        else:
            xs = np.array([1.0, 8.0, 32.0, 128.0])
            b0_n = b0_b = 0.0

        xl = np.linspace(0.0, 132.0, 100)
        y_line_n = b0_n + t_l * xl
        y_line_b = b0_b + (t_l + t_b) * xl
        ax.plot(xl, y_line_n, color="#1f77b4", lw=2.0, label=f"No Barriers: slope={t_l:.3f} us/dispatch")
        ax.plot(xl, y_line_b, color="#d62728", lw=2.0, ls="--",
                label=f"RAW Barriers: slope={t_l+t_b:.3f} us (+{t_b:.3f} us barrier)")
        ax.fill_between(xl, y_line_n, y_line_b, color="#ff9896", alpha=0.22, label="Cumulative Barrier Tax")
        ax.set_title(f"{meta['device']['name']}\n(OLS R2 = {r2:.4f})", fontsize=9.5)
        ax.set_xlabel("Dispatches in Single CommandBuffer (N)")
        ax.set_ylabel("GPU Execution Time (us)")
        ax.set_xticks([1, 32, 64, 96, 128])
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=7.6, loc="upper left")

    fig.suptitle("MB1: OLS Linear Regression for GPU Kernel Launch & RAW Barrier Cost (N = 1..128)", fontsize=11.5)
    fig.tight_layout()
    out_p = res_dir / "mb1_dispatch_ols.png"
    fig.savefig(out_p)
    plt.close(fig)
    print(f"Saved -> {out_p}")


def plot_bw(res_dir: Path, bw_files, log_y: bool):
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=180)
    for idx, bf in enumerate(bw_files):
        prefix = bf.stem.replace("_bw", "")
        meta = load_json(res_dir / f"{prefix}_full.json")
        c = COLORS[idx % len(COLORS)]
        rows = read_csv_rows(bf)
        xs = [float(r["working_set_mb"]) for r in rows]
        ys = [float(r["bw_median_gbps"]) for r in rows]
        iqrs = [float(r["bw_iqr_gbps"]) for r in rows]
        ax.plot(xs, ys, marker="o", lw=2.0, ms=4.5, color=c, label=meta["device"]["name"] if meta else prefix)
        if not log_y:
            ax.fill_between(xs, [max(0.0, y - 0.5 * q) for y, q in zip(ys, iqrs)],
                            [y + 0.5 * q for y, q in zip(ys, iqrs)], color=c, alpha=0.15)
        if meta and "mb2_memory" in meta:
            l2 = float(meta["mb2_memory"]["l2_effective_mb"])
            if l2 > 0:
                ax.axvline(l2, color=c, ls="--", lw=1.3, alpha=0.8, label=f"{prefix}: L2={l2:g}MB")
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log", base=10)
    ax.set_xticks(BW_TICKS)
    ax.set_xticklabels(BW_LABELS)
    ax.set_xlabel("Working Set Size W")
    ax.set_ylabel("Effective Read Bandwidth (GB/s" + (", log10)" if log_y else ")"))
    ax.set_title("MB2: Cross-Vendor Cache Hierarchy" + (" (Log-Log)" if log_y else ""))
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8.0, loc="lower left" if log_y else "upper right")
    fig.tight_layout()
    out_p = res_dir / ("mb2_cache_hierarchy_log.png" if log_y else "mb2_cache_hierarchy.png")
    fig.savefig(out_p)
    plt.close(fig)
    print(f"Saved -> {out_p}")


def plot_mb3_mb4_abs(res_dir: Path, json_files):
    devs = [(jf.stem.replace("_full", ""), load_json(jf)) for jf in json_files]
    devs = [(p, m) for p, m in devs if m]
    if not devs:
        return
    names, x, w = [p for p, _ in devs], np.arange(len(devs)), 0.25
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12.0, 4.7), dpi=180)

    ax3.bar(x - w, [m["mb2_memory"]["bw_dram_gbps"] for _, m in devs], w, label="DRAM Coalesced (stride=1)")
    ax3.bar(x, [m["mb3_coalescing"]["bw_l2_strided32_gbps"] for _, m in devs], w, label="L2 Strided (stride=32)")
    ax3.bar(x + w, [m["mb3_coalescing"]["bw_dram_strided32_gbps"] for _, m in devs], w, label="DRAM Strided (stride=32)")
    ax3.set_yscale("log", base=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylabel("Useful Bandwidth (GB/s, log10)")
    ax3.set_title("MB3: Coalesced vs Strided (128B) Memory Bandwidth")
    ax3.grid(axis="y", which="both", ls=":", alpha=0.6)
    ax3.legend(fontsize=8.0)

    ax4.bar(x - w, [m["mb5_compute"]["pi_fma_gflops"] for _, m in devs], w, label="Peak FMA16 (MB5)")
    ax4.bar(x, [m["mb4_tiling"]["gflops_matmul_smem"] for _, m in devs], w, label="Tiled 64x64 SMEM (MB4)")
    ax4.bar(x + w, [m["mb4_tiling"]["gflops_matmul_cache"] for _, m in devs], w, label="Tiled 64x64 DirectCache (MB4)")
    ax4.set_yscale("log", base=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.set_ylabel("Sustained FP32 Throughput (GFLOPS, log10)")
    ax4.set_title("MB4 & MB5: Peak FMA vs 2D-Tiled GEMM (SMEM vs Cache)")
    ax4.grid(axis="y", which="both", ls=":", alpha=0.6)
    ax4.legend(fontsize=8.0)

    fig.tight_layout()
    out_p = res_dir / "mb3_mb4_abs.png"
    fig.savefig(out_p)
    plt.close(fig)
    print(f"Saved -> {out_p}")


def plot_waves(res_dir: Path, wave_files):
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(9.0, 6.6), dpi=180, sharex=True)
    fig_n, ax_n = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    max_t = 10.0
    for idx, wf in enumerate(wave_files):
        prefix = wf.stem.replace("_wave", "")
        meta = load_json(res_dir / f"{prefix}_full.json")
        c = COLORS[idx % len(COLORS)]
        rows = read_csv_rows(wf)
        xs = [int(r["workgroups"]) for r in rows]
        gfs = [float(r["gflops"]) for r in rows]
        ts = [float(r["gpu_time_us"]) for r in rows]
        peak = max(gfs) if gfs else 1.0
        for x_v, t in zip(xs, ts):
            if x_v <= 36:
                max_t = max(max_t, t)
        lbl = meta["device"]["name"] if meta else prefix
        ax_t.plot(xs, [g / 1e3 for g in gfs], marker="s", lw=2.0, ms=4.2, color=c, label=lbl)
        ax_b.plot(xs, ts, marker="o", lw=1.9, ms=4.0, color=c, label=lbl)
        ax_n.plot(xs, [100.0 * g / peak for g in gfs], marker="s", lw=2.0, ms=4.2, color=c, label=lbl)
        if meta and "mb5_compute" in meta:
            gw = int(meta["mb5_compute"].get("g_wave_quantum", 0))
            if gw > 1:
                ax_n.axvline(gw, color=c, ls="--", lw=1.3, alpha=0.75, label=f"{prefix}: G_wave={gw}")
                for m in (1, 2, 3):
                    if gw * m <= max(xs):
                        ax_t.axvline(gw * m, color=c, ls="--", lw=1.1, alpha=0.55,
                                     label=(f"{prefix}: G_wave={gw}" if m == 1 else None))
                        ax_b.axvline(gw * m, color=c, ls="--", lw=1.1, alpha=0.55)
    ax_t.set_xscale("log", base=2)
    ax_t.set_ylabel("Sustained FP32 (TFLOPS)")
    ax_t.set_title("MB5: Wave Saturation & Quantization Staircase")
    ax_t.grid(True, which="both", ls=":", alpha=0.6)
    ax_t.legend(fontsize=8.0, loc="upper left")
    ax_b.set_xticks(WAVE_TICKS)
    ax_b.set_xticklabels([str(v) for v in WAVE_TICKS])
    ax_b.set_ylim(0.0, max_t * 1.18)
    ax_b.set_xlabel("Launched Workgroups G (256 threads/WG)")
    ax_b.set_ylabel("Time (us, G<=36)")
    ax_b.grid(True, which="both", ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(res_dir / "mb5_wave_saturation.png")
    plt.close(fig)

    ax_n.set_xscale("log", base=2)
    ax_n.set_xticks(WAVE_TICKS)
    ax_n.set_xticklabels([str(v) for v in WAVE_TICKS])
    ax_n.set_ylim(0.0, 108.0)
    ax_n.set_xlabel("Launched Workgroups G (256 threads/WG)")
    ax_n.set_ylabel("Utilization (% of Device Peak)")
    ax_n.set_title("MB5: Cross-Vendor Wave Quantization (% of Peak)")
    ax_n.grid(True, which="both", ls=":", alpha=0.6)
    ax_n.legend(fontsize=8.0, loc="lower right")
    fig_n.tight_layout()
    fig_n.savefig(res_dir / "mb5_wave_efficiency_norm.png")
    plt.close(fig_n)
    print(f"Saved -> {res_dir / 'mb5_wave_saturation.png'} & mb5_wave_efficiency_norm.png")


def plot_mb6_ols(res_dir: Path, json_files):
    valid = [(jf.stem.replace("_full", ""), load_json(jf)) for jf in json_files]
    valid = [(p, m) for p, m in valid if m and m.get("mb6_compile", {}).get("drv_points")]
    if not valid:
        return

    # 2x2 grid: individual scaled OLS plot per device driver + libslang frontend OLS panel
    n_panels = len(valid) + 1
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(12.2, 4.6 * nrows), dpi=180, squeeze=False)
    flat_axs = [axs[r, c] for r in range(nrows) for c in range(ncols)]

    for idx, (prefix, meta) in enumerate(valid):
        ax = flat_axs[idx]
        c = COLORS[idx % len(COLORS)]
        m6 = meta["mb6_compile"]
        pts = m6["drv_points"]
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)
        r2 = float(m6["r2_drv"])
        b0, b1 = float(m6["beta_0_drv_ms"]), float(m6["beta_1_drv_ms_per_word"])

        if r2 < 0.95 and m6.get("beta_1_drv_smem_ms_per_word", 0.0) > 0.0:
            # Two-regime OLS triggered by R2 < 0.95 (e.g. RPi4 V3D CFG/SMEM cliff)
            b0_sm, b1_sm = float(m6["beta_0_drv_smem_ms"]), float(m6["beta_1_drv_smem_ms_per_word"])
            ax.scatter(xs[:4], ys[:4], color="#1f77b4", s=48, zorder=4, label="Regime 1: 1D / Direct (no SMEM)")
            ax.scatter(xs[4:], ys[4:], color="#d62728", marker="D", s=48, zorder=4, label="Regime 2: Unrolled 2D SMEM")
            xl1 = np.linspace(0, xs[3] * 1.15, 50)
            xl2 = np.linspace(xs[4] * 0.92, xs[-1] * 1.05, 50)
            ax.plot(xl1, np.maximum(0.1, b0 + b1 * xl1), color="#1f77b4", lw=2.0,
                    label=f"OLS1: b0={b0:.2f}ms, b1={b1*1e3:.2f}us/w")
            ax.plot(xl2, np.maximum(0.1, b0_sm + b1_sm * xl2), color="#d62728", lw=2.0, ls="--",
                    label=f"OLS2: {b0_sm + b1_sm * xs[4]:.0f}ms plateau")
            ax.set_yscale("log", base=10)
            ax.set_ylabel("Compile Time (ms, log10)")
            ax.set_title(f"MB6 Driver OLS: {meta['device']['name']}\n(Two-Regime triggered: global R2={r2:.3f} < 0.95)", fontsize=9.5)
        else:
            xl = np.linspace(0, max(xs) * 1.05, 100)
            ax.scatter(xs, ys, color=c, s=48, zorder=4, label="Measured (median N=7 salted)")
            ax.plot(xl, b0 + b1 * xl, color=c, lw=2.0,
                    label=f"OLS: T = {b0:.2f} + {b1*1e3:.3f}e-3 * L (R2={r2:.4f})")
            ax.set_ylabel("Compile Time (ms)")
            ax.set_title(f"MB6 Driver OLS: {meta['device']['name']} (R2={r2:.4f})", fontsize=9.5)

        ax.set_xlabel("SPIR-V Module Length L_spv (32-bit words)")
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(fontsize=7.8, loc="upper left")

    # Last panel: Frontend libslang OLS
    ax_fe = flat_axs[len(valid)]
    has_fe = False
    for idx, (prefix, meta) in enumerate(valid):
        m6 = meta["mb6_compile"]
        fe_pts = m6.get("fe_points", [])
        if fe_pts:
            has_fe = True
            c = COLORS[idx % len(COLORS)]
            fxs = np.array([p[0] for p in fe_pts], float)
            fys = np.array([p[1] for p in fe_pts], float)
            xl = np.linspace(0, max(fxs) * 1.05, 100)
            a0, a1, r2f = m6["alpha_0_slang_ms"], m6["alpha_1_slang_ms_per_word"], m6["r2_fe_slang"]
            ax_fe.scatter(fxs, fys, color=c, s=44, zorder=4)
            ax_fe.plot(xl, a0 + a1 * xl, color=c, lw=2.0,
                       label=f"{prefix}: a1={a1*1e3:.2f} us/w (R2={r2f:.3f})")
    if has_fe:
        ax_fe.set_title("MB6 Frontend OLS: In-Process libslang (Slang -> SPIR-V)", fontsize=9.5)
        ax_fe.set_xlabel("SPIR-V Module Length L_spv (32-bit words)")
        ax_fe.set_ylabel("Frontend Compile Time (ms)")
        ax_fe.grid(True, ls=":", alpha=0.6)
        ax_fe.legend(fontsize=8.0, loc="upper left")
    else:
        ax_fe.axis("off")

    for k in range(len(valid) + 1, len(flat_axs)):
        flat_axs[k].axis("off")

    fig.suptitle("MB6: Ordinary Least Squares (OLS) Compilation Cost Calibration", fontsize=12.0)
    fig.tight_layout()
    out_p = res_dir / "mb6_compile_ols.png"
    fig.savefig(out_p)
    plt.close(fig)
    print(f"Saved -> {out_p}")


def plot_dashboard(res_dir: Path, json_files):
    devs = [(jf.stem.replace("_full", ""), load_json(jf)) for jf in json_files]
    devs = [(p, m) for p, m in devs if m]
    if not devs:
        return
    names, x, w = [p for p, _ in devs], np.arange(len(devs)), 0.25
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.0), dpi=180)

    ax = axs[0, 0]
    ax.bar(x - w / 2, [m["mb1_dispatch"]["t_launch_gpu_us"] for _, m in devs], w, label="t_launch (us)")
    ax.bar(x + w / 2, [m["mb1_dispatch"]["t_barrier_gpu_us"] for _, m in devs], w, label="t_barrier (us)")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title("MB1: GPU Launch & RAW Barrier"); ax.legend(fontsize=8.0); ax.grid(axis="y", ls=":", alpha=0.6)

    ax = axs[0, 1]
    ax.bar(x - w / 2, [m["mb3_coalescing"]["coal_factor_l2_s32"] for _, m in devs], w, label="L2 Stride=32")
    ax.bar(x + w / 2, [m["mb3_coalescing"]["coal_factor_dram_s32"] for _, m in devs], w, label="DRAM Stride=32")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title("MB3: Coalescing Efficiency"); ax.legend(fontsize=8.0); ax.grid(axis="y", ls=":", alpha=0.6)

    ax = axs[1, 0]
    ax.bar(x - w, [m["mb4_tiling"].get("smem_vs_cache_speedup", 1.0) for _, m in devs], w, label="SMEM/Cache")
    ax.bar(x, [m["mb4_tiling"].get("tail_efficiency", 1.0) for _, m in devs], w, label="tail_eff")
    ax.bar(x + w, [m["mb4_tiling"].get("fusion_reg_factor", 1.0) for _, m in devs], w, label="fusion_reg")
    ax.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title("MB4: Tiling, Tail & Reg Pressure"); ax.legend(fontsize=8.0); ax.grid(axis="y", ls=":", alpha=0.6)

    ax = axs[1, 1]
    v2 = [(p, m) for p, m in devs if "stage_timings_ms" in m]
    if v2:
        keys = ["mb0_warmup_ms", "mb1_dispatch_ms", "mb2_cache_ms", "mb3_stride_ms", "mb4_tiled_ms", "mb5_compute_ms", "mb6_drv_ms", "mb6_fe_ms"]
        lbls = ["MB0", "MB1", "MB2", "MB3", "MB4", "MB5", "MB6_drv", "MB6_fe"]
        y, left = np.arange(len(v2)), np.zeros(len(v2))
        for k, l in zip(keys, lbls):
            vals = np.array([m["stage_timings_ms"].get(k, 0.0) for _, m in v2])
            ax.barh(y, vals, left=left, height=0.45, label=l); left += vals
        ax.set_yticks(y); ax.set_yticklabels([p for p, _ in v2]); ax.set_xlabel("Wall Time (ms)")
        ax.set_title("Stage 0 Calibration Time Breakdown"); ax.legend(fontsize=7.2, ncol=2); ax.grid(axis="x", ls=":", alpha=0.6)
    else:
        ax.axis("off")

    fig.tight_layout(); fig.savefig(res_dir / "mb_dashboard.png"); plt.close(fig)
    print(f"Saved -> {res_dir / 'mb_dashboard.png'}")


def plot_phase_maps(res_dir: Path, phase_files):
    if not phase_files:
        return
    fig, axs = plt.subplots(1, len(phase_files), figsize=(6.2 * len(phase_files), 4.8), dpi=180, squeeze=False)
    cmap = mcolors.ListedColormap(["#6baed6", "#fdae6b", "#fd8d3c", "#74c476"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    short = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}
    for idx, pf in enumerate(phase_files):
        ax, prefix = axs[0, idx], pf.stem.replace("_phase_MR", "")
        meta, rows = load_json(res_dir / f"{prefix}_full.json"), read_csv_rows(pf)
        ms, rs = sorted({int(r["M"]) for r in rows}), sorted({int(r["R"]) for r in rows})
        grid = np.zeros((len(rs), len(ms)), dtype=int)
        for r in rows:
            grid[rs.index(int(r["R"])), ms.index(int(r["M"]))] = int(r["plan_id"])
        ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        for ri in range(len(rs)):
            for mi in range(len(ms)):
                ax.text(mi, ri, short[grid[ri, mi]], ha="center", va="center", fontsize=6.8)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([str(m) for m in ms], rotation=45, fontsize=7.5)
        ax.set_yticks(range(len(rs))); ax.set_yticklabels([str(r) for r in rs], fontsize=8.0)
        ax.set_xlabel("Rows M (N=3072, K=768, ReLU, cached)"); ax.set_ylabel("Model Runs R")
        title = meta["device"]["name"] if meta else prefix
        if meta and "mb2_memory" in meta:
            title += f"\n(L2 cliff M_L2 ~= {float(meta['mb2_memory']['l2_effective_bytes']) / (4.0 * 3072.0):.0f})"
        ax.set_title(title, fontsize=9.5)
    fig.suptitle("2D Decision Phase Map in (M, R): P0=Split | P2=FusedEpilogue | P3=FullFusion", fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / "phase_map_MR.png"); plt.close(fig)
    print(f"Saved -> {res_dir / 'phase_map_MR.png'}")


def main():
    res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    bw, wv = sorted(res_dir.glob("*_bw.csv")), sorted(res_dir.glob("*_wave.csv"))
    js, ph = sorted(res_dir.glob("*_full.json")), sorted(res_dir.glob("*_phase_MR.csv"))
    if bw:
        plot_bw(res_dir, bw, False)
        plot_bw(res_dir, bw, True)
    if wv:
        plot_waves(res_dir, wv)
    if js:
        plot_mb1_ols(res_dir, js)
        plot_mb3_mb4_abs(res_dir, js)
        plot_mb6_ols(res_dir, js)
        plot_dashboard(res_dir, js)
    if ph:
        plot_phase_maps(res_dir, ph)


if __name__ == "__main__":
    main()