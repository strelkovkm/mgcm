#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

R_GRID = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000, 20000]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def pearson_r(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 1.0


def mape_pct(pred, real):
    return float(np.mean(np.abs(pred - real) / np.maximum(1e-9, real)) * 100.0)


def analyze_file(csv_path: Path, out_dir: Path):
    tag = csv_path.stem.replace("verify_", "")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pred_e = np.array([float(r["pred_exec_us"]) for r in rows])
    real_e = np.array([float(r["real_exec_us"]) for r in rows])
    pred_c = np.array([float(r["pred_compile_us"]) for r in rows])
    real_c = np.array([float(r["real_compile_us"]) for r in rows])

    r_exec, mape_e = pearson_r(pred_e, real_e), mape_pct(pred_e, real_e)
    r_comp, mape_c = pearson_r(pred_c, real_c), mape_pct(pred_c, real_c)

    ms = sorted({int(r["M"]) for r in rows})
    by_m = {m: {} for m in ms}
    for r in rows:
        m, p = int(r["M"]), int(r["plan_id"])
        by_m[m][p] = {
            "pc": float(r["pred_compile_us"]), "rc": float(r["real_compile_us"]),
            "pe": float(r["pred_exec_us"]), "re": float(r["real_exec_us"]),
            "nomem_e": float(r["pred_nomem_exec_us"]),
            "nodisp_e": float(r["pred_nodisp_exec_us"]),
        }

    grid_pred = np.zeros((len(R_GRID), len(ms)), dtype=int)
    grid_real = np.zeros((len(R_GRID), len(ms)), dtype=int)

    policies = {
        "Fixed: Always Split (P0)": [],
        "Fixed: Always Partial (P2)": [],
        "Fixed: Always Fuse (P3)": [],
        "Ablation: NoMemoryTraffic": [],
        "Ablation: NoDispatch": [],
        "Ablation: NoCompileCost": [],
        "Proposed: UniversalCostEngine": [],
    }

    matches = 0
    for mi, m in enumerate(ms):
        d = by_m[m]
        for ri, rv in enumerate(R_GRID):
            real_cost = {p: d[p]["rc"] + rv * d[p]["re"] for p in range(4)}
            oracle_p = min(range(4), key=lambda p: real_cost[p])
            oracle_c = real_cost[oracle_p]

            full_p = min(range(4), key=lambda p: d[p]["pc"] + rv * d[p]["pe"])
            nomem_p = min(range(4), key=lambda p: d[p]["pc"] + rv * d[p]["nomem_e"])
            nodisp_p = min(range(4), key=lambda p: d[p]["pc"] + rv * d[p]["nodisp_e"])
            nocomp_p = min(range(4), key=lambda p: rv * d[p]["pe"])

            grid_real[ri, mi] = oracle_p
            grid_pred[ri, mi] = full_p
            if full_p == oracle_p:
                matches += 1

            auto_reg = lambda chosen: (real_cost[chosen] - oracle_c) / oracle_c * 100.0
            policies["Fixed: Always Split (P0)"].append(auto_reg(0))
            policies["Fixed: Always Partial (P2)"].append(auto_reg(2))
            policies["Fixed: Always Fuse (P3)"].append(auto_reg(3))
            policies["Ablation: NoMemoryTraffic"].append(auto_reg(nomem_p))
            policies["Ablation: NoDispatch"].append(auto_reg(nodisp_p))
            policies["Ablation: NoCompileCost"].append(auto_reg(nocomp_p))
            policies["Proposed: UniversalCostEngine"].append(auto_reg(full_p))

    agree_pct = matches / (len(ms) * len(R_GRID)) * 100.0
    print(f"\n==================== [{tag}] IDENTICAL-CONDITION VULKAN 1.3 VERIFICATION ====================")
    print(f"T_execute Correlation : Pearson r = {r_exec:.5f} (R2 = {r_exec**2:.5f}) | MAPE = {mape_e:.2f}%")
    print(f"T_compile Correlation : Pearson r = {r_comp:.5f} | MAPE = {mape_c:.2f}%")
    print(f"2D Phase Map Agreement: {agree_pct:.1f}% ({matches}/{len(ms)*len(R_GRID)} cells exact match)")
    print("-" * 86)
    print(f"{'Policy / Ablation':<34} | {'Mean Regret (%)':>16} | {'P95 Regret (%)':>16} | {'Max Regret (%)':>14}")
    print("-" * 86)
    for name, regs in policies.items():
        arr = np.array(regs)
        print(f"{name:<34} | {np.mean(arr):>15.3f}% | {np.percentile(arr, 95):>15.3f}% | {np.max(arr):>13.3f}%")

    # Side-by-side Predicted vs Measured Oracle 2D Phase Map
    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(12.2, 4.9), dpi=180)
    cmap = mcolors.ListedColormap(["#6baed6", "#fdae6b", "#fd8d3c", "#74c476"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    short = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}
    for ax, grd, ttl in [
        (ax_p, grid_pred, f"Predicted (UniversalCostEngine)\nAgreement = {agree_pct:.1f}%"),
        (ax_r, grid_real, "Measured Hardware Oracle (Vulkan 1.3)"),
    ]:
        ax.imshow(grd, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        for ri in range(len(R_GRID)):
            for mi in range(len(ms)):
                ax.text(mi, ri, short[grd[ri, mi]], ha="center", va="center", fontsize=6.8)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([str(m) for m in ms], rotation=45, fontsize=7.5)
        ax.set_yticks(range(len(R_GRID))); ax.set_yticklabels([str(r) for r in R_GRID], fontsize=8.0)
        ax.set_xlabel("Rows M (N=3072, K=768)"); ax.set_ylabel("Model Runs R")
        ax.set_title(ttl, fontsize=10.0)
    fig.suptitle(f"[{tag}] Predicted vs Measured Vulkan 1.3 Phase Map (r_exec={r_exec:.4f}, MAPE={mape_e:.1f}%)", fontsize=11)
    fig.tight_layout()
    out_p = out_dir / f"compare_phase_MR_{tag}.png"
    fig.savefig(out_p); plt.close(fig)
    print(f"Saved -> {out_p}")
    return tag, pred_e, real_e, r_exec, mape_e


def main():
    res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    vfiles = sorted(res_dir.glob("verify_*.csv"))
    if not vfiles:
        sys.exit("No verify_*.csv found in results/")

    fig_s, ax_s = plt.subplots(figsize=(6.8, 5.6), dpi=180)
    lo_all, hi_all = 1e9, 0.0
    for idx, vf in enumerate(vfiles):
        tag, pe, re_v, r_val, mp = analyze_file(vf, res_dir)
        c = COLORS[idx % len(COLORS)]
        ax_s.scatter(re_v, pe, color=c, s=32, alpha=0.82, label=f"{tag}: r={r_val:.4f}, MAPE={mp:.1f}%")
        lo_all, hi_all = min(lo_all, re_v.min(), pe.min()), max(hi_all, re_v.max(), pe.max())

    ax_s.plot([lo_all * 0.8, hi_all * 1.2], [lo_all * 0.8, hi_all * 1.2], "k--", lw=1.4, label="Ideal y = x")
    ax_s.set_xscale("log", base=10); ax_s.set_yscale("log", base=10)
    ax_s.set_xlabel("Measured Vulkan 1.3 Execution Time (us)")
    ax_s.set_ylabel("Predicted UniversalCostEngine Time (us)")
    ax_s.set_title("Blind Prediction vs Real Hardware Execution (P0..P3, M=16..4096)")
    ax_s.grid(True, which="both", ls=":", alpha=0.6); ax_s.legend(fontsize=8.5)
    fig_s.tight_layout()
    out_c = res_dir / "corr_pred_vs_meas.png"
    fig_s.savefig(out_c); plt.close(fig_s)
    print(f"\nSaved -> {out_c}")


if __name__ == "__main__":
    main()